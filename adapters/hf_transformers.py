"""HF Transformers adapter — the no-batching baseline.

This adapter intentionally has no serving optimizations:
    - One model instance, no replicas
    - No batching (each transcribe call is one forward pass on one input)
    - No KV cache management beyond what model.generate() does internally
    - No request queuing

It exists so we can measure how much the other frameworks' optimizations
actually buy you. If vLLM's continuous batching is +5x throughput vs HF
baseline at concurrency=64, that's the framework's value proposition in a
single number. If it's +1.2x, the optimization isn't earning its complexity.

We use the HF pipeline API (`pipeline("automatic-speech-recognition", ...)`)
rather than calling model.generate() directly. The pipeline handles
preprocessing (resampling, mel-spectrogram, padding) consistently with how
most users would deploy this naively. That's the right baseline to measure
against — "what you get if you `pip install transformers` and follow the
docs."

Adapter config knobs (passed via CellConfig.adapter_config):
    device: "cuda" | "cpu" (default "cuda")
    dtype: "float16" | "bfloat16" | "float32" (default "float16")
    chunk_length_s: int (default 30 — matches Whisper encoder window)
    batch_size: int (default 1 — see above; bumping this for the baseline
        defeats the point of the comparison, but the knob exists for sanity tests)
"""

from __future__ import annotations

import asyncio
from typing import Any

from adapters.base import FrameworkAdapter
from bench.config import CellConfig
from bench.data import AudioClip


class HfTransformersAdapter(FrameworkAdapter):
    def __init__(self, config: CellConfig):
        super().__init__(config)
        self._pipeline: Any = None
        self._device: str = config.adapter_config.get("device", "cuda")
        self._dtype_name: str = config.adapter_config.get("dtype", "float16")
        self._chunk_length_s: int = config.adapter_config.get("chunk_length_s", 30)
        self._batch_size: int = config.adapter_config.get("batch_size", 1)
        self._language: str | None = config.adapter_config.get("language")

    async def setup(self) -> None:
        # Imports are deferred to setup() so that importing this module doesn't
        # trigger torch CUDA initialization (which is slow and side-effecty).
        import torch
        from transformers import pipeline

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self._dtype_name]

        # Pipeline construction loads weights — non-trivial wall time, especially
        # for whisper-large-v3 (~3GB). Run in a thread so we don't block the loop.
        loop = asyncio.get_running_loop()

        def _build():
            return pipeline(
                task="automatic-speech-recognition",
                model=self.config.model,
                torch_dtype=dtype,
                device=self._device,
                chunk_length_s=self._chunk_length_s,
                batch_size=self._batch_size,
                # return_timestamps required when chunk_length_s is set
                return_timestamps=False if self._chunk_length_s is None else True,
            )

        self._pipeline = await loop.run_in_executor(None, _build)
        print(
            f"[HfTransformersAdapter] loaded {self.config.model} on {self._device} "
            f"dtype={self._dtype_name} chunk={self._chunk_length_s}s batch={self._batch_size}"
        )

    async def transcribe(self, clip: AudioClip) -> str:
        if self._pipeline is None:
            raise RuntimeError("Adapter.setup() must be called before transcribe()")

        loop = asyncio.get_running_loop()
        # The pipeline call is sync and blocks the GIL during the forward pass.
        # Run in default executor so the event loop stays responsive (other
        # workers can submit while this one waits). For the baseline this is
        # actually a bit generous — a truly naive deployment would use a sync
        # request handler — but it's the standard pattern and we're being fair.
        generate_kwargs: dict[str, Any] = {}
        if self._language:
            generate_kwargs["language"] = self._language

        def _run():
            out = self._pipeline(str(clip.audio_path), generate_kwargs=generate_kwargs or None)
            # pipeline returns {"text": "...", "chunks": [...]}
            if isinstance(out, dict) and "text" in out:
                return out["text"]
            # batch mode could return list; defensive
            if isinstance(out, list) and out and isinstance(out[0], dict):
                return out[0].get("text", "")
            return str(out)

        return await loop.run_in_executor(None, _run)

    async def aclose(self) -> None:
        if self._pipeline is None:
            return
        # Drop the reference and clear CUDA cache. Best-effort; Python GC will
        # handle the rest.
        self._pipeline = None
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
            "chunk_length_s": self._chunk_length_s,
            "batch_size": self._batch_size,
            "language": self._language,
        }
