"""HF Transformers adapter — the no-batching baseline.

This adapter intentionally has no serving optimizations:
    - One model instance, no replicas
    - No batching (each transcribe call is one forward pass on one input)
    - No KV cache management beyond what model.generate() does internally
    - No request queuing
    - Concurrent requests are serialized through a lock around model.generate
      because PyTorch model.generate() is not thread-safe (concurrent calls
      on the same model corrupt CUDA state and crash with a device-side
      assert). Serializing is the honest naive-baseline behavior: "you can
      submit N concurrent requests but they queue through one model on one
      GPU." That serialization is exactly the gap batched/replicated
      frameworks (vLLM continuous batching, Ray Serve replicas) fill.

It exists so we can measure how much the other frameworks' optimizations
actually buy you. If vLLM's continuous batching is +5x throughput vs HF
baseline at concurrency=64, that's the framework's value proposition in
a single number. If it's +1.2x, the optimization isn't earning its
complexity.

Implementation note: we deliberately do NOT use the high-level
`transformers.pipeline()` API. The pipeline does internal chunking,
batching, and audio decoding via torchcodec/ffmpeg — all confounding
variables for our measurement, and torchcodec specifically has been a
deployment-fragility pain point (it requires precise PyTorch + FFmpeg
shared-lib version alignment that the RunPod PyTorch base image doesn't
ship). Going direct via `AutoModelForSpeechSeq2Seq.generate()` lets us
own audio loading (via soundfile, no FFmpeg needed), control exactly
what the model does per request, and produce a baseline that's both
cleaner and more reproducible.

Adapter config knobs (passed via CellConfig.adapter_config):
    device: "cuda" | "cpu" (default "cuda")
    dtype: "float16" | "bfloat16" | "float32" (default "float16")
    language: optional ISO language code passed to model.generate
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from adapters.base import FrameworkAdapter
from bench.config import CellConfig
from bench.data import AudioClip

_DTYPE_MAP_NAMES = ("float16", "bfloat16", "float32")


class HfTransformersAdapter(FrameworkAdapter):
    def __init__(self, config: CellConfig):
        super().__init__(config)
        self._model: Any = None
        self._processor: Any = None
        self._torch_dtype: Any = None
        self._device: str = config.adapter_config.get("device", "cuda")
        self._dtype_name: str = config.adapter_config.get("dtype", "float16")
        if self._dtype_name not in _DTYPE_MAP_NAMES:
            raise ValueError(
                f"adapter_config.dtype must be one of {_DTYPE_MAP_NAMES}, "
                f"got {self._dtype_name!r}"
            )
        self._language: str | None = config.adapter_config.get("language")
        self._model_lock = threading.Lock()

    async def setup(self) -> None:
        # Deferred imports so that importing this module doesn't trigger
        # torch CUDA initialization (slow + side-effecty).
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self._dtype_name]
        self._torch_dtype = dtype

        loop = asyncio.get_running_loop()

        def _build():
            processor = AutoProcessor.from_pretrained(self.config.model)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.model,
                torch_dtype=dtype,
            ).to(self._device)
            model.eval()
            # Whisper-large-v3 quirk: the model ships with a generation_config
            # that has legacy forced_decoder_ids set. When you also pass
            # language/task as generate() kwargs, the two paths conflict and
            # can produce out-of-range token IDs that crash the embedding
            # lookup with a CUDA assertion. Clear the stale config so only the
            # modern kwarg path is active.
            if hasattr(model, "generation_config"):
                model.generation_config.forced_decoder_ids = None
                model.generation_config.suppress_tokens = []
            return processor, model

        self._processor, self._model = await loop.run_in_executor(None, _build)
        print(
            f"[HfTransformersAdapter] loaded {self.config.model} on {self._device} "
            f"dtype={self._dtype_name} (direct model.generate, no pipeline)"
        )

    async def transcribe(self, clip: AudioClip) -> str:
        if self._model is None:
            raise RuntimeError("Adapter.setup() must be called before transcribe()")

        import numpy as np
        import soundfile as sf
        import torch

        # max_new_tokens cap prevents runaway generation on noisy audio;
        # 440 leaves headroom under Whisper's 448 max decoder length. In
        # long-form mode this applies per 30s chunk, not per clip.
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": 440,
            # return_timestamps=True activates transformers' long-form generation
            # path: the model emits timestamp tokens that delimit 30s chunks and
            # internally re-runs the decoder for each chunk, so clips longer than
            # Whisper's 30s encoder window get fully transcribed instead of
            # silently truncated. Required for any eval set with variable-length
            # audio (our 30s+ streaming clips average 35.5s, max 46s).
            "return_timestamps": True,
        }
        if self._language:
            generate_kwargs["language"] = self._language
            generate_kwargs["task"] = "transcribe"

        loop = asyncio.get_running_loop()

        def _run():
            # Load audio with soundfile — handles WAV/FLAC/OGG natively without
            # FFmpeg or torchcodec. Force float32 because the processor expects it.
            arr, sr = sf.read(str(clip.audio_path), dtype="float32")
            # Whisper expects mono. Average channels if stereo.
            if arr.ndim > 1:
                arr = arr.mean(axis=1).astype(np.float32)

            # The processor handles resampling to 16kHz internally if sr != 16000,
            # then computes the log-mel spectrogram the encoder needs.
            # truncation=False + return_attention_mask=True is the long-form
            # contract: features cover the full audio (not capped at 30s) and the
            # mask tells the model the real length so it knows where to chunk.
            inputs = self._processor(
                arr,
                sampling_rate=sr,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(self._device, dtype=self._torch_dtype)
            attention_mask = inputs.attention_mask.to(self._device)

            # Serialize concurrent generate() calls — PyTorch's model.generate
            # is not thread-safe across workers sharing one model instance.
            with self._model_lock, torch.inference_mode():
                predicted_ids = self._model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    **generate_kwargs,
                )
            text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return text

        return await loop.run_in_executor(None, _run)

    async def aclose(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def metadata(self) -> dict[str, Any]:
        return {
            "device": self._device,
            "dtype": self._dtype_name,
            "language": self._language,
            "implementation": "direct_generate",  # vs "pipeline" — record the choice
        }
