"""Ray Serve deployment definition for the Whisper benchmark.

This module is loaded by `serve run adapters.ray_serve_deployment:app` (or
imported programmatically by the adapter). It defines a single deployment
that wraps HF Transformers' Whisper model behind a Ray Serve `@serve.batch`
decorator — Ray Serve queues incoming requests for up to a few milliseconds,
forms them into a batch, and runs them through `model.generate()` as one
GPU forward pass.

We use HF Transformers (not faster-whisper) inside the deployment because
HF supports true tensor-batched inference, which is the optimization Ray
Serve's batching primitive is designed to expose. faster-whisper's
CTranslate2 backend processes one clip at a time internally, so batching
it would conflate the framework comparison.

Configuration is read from environment variables so the adapter (which
spawns `serve run` in a subprocess) can parameterize the deployment
without modifying source.
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any

# Ray imports are deferred so this module can be inspected without Ray installed.


def _load_env_config() -> dict[str, Any]:
    return {
        "model_name": os.environ.get("BENCH_MODEL_NAME", "openai/whisper-large-v3"),
        "dtype": os.environ.get("BENCH_DTYPE", "float16"),
        "max_batch_size": int(os.environ.get("BENCH_MAX_BATCH_SIZE", "8")),
        "batch_wait_timeout_s": float(os.environ.get("BENCH_BATCH_WAIT_TIMEOUT_S", "0.02")),
        "num_replicas": int(os.environ.get("BENCH_NUM_REPLICAS", "1")),
        "language": os.environ.get("BENCH_LANGUAGE") or None,
    }


def build_app():
    """Build the Ray Serve application. Called at module import time below."""
    from ray import serve
    from starlette.requests import Request

    cfg = _load_env_config()

    @serve.deployment(
        num_replicas=cfg["num_replicas"],
        ray_actor_options={"num_gpus": 1},
    )
    class WhisperDeployment:
        def __init__(self) -> None:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            torch_dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[cfg["dtype"]]
            self._dtype = torch_dtype
            self._language = cfg["language"]

            self.processor = AutoProcessor.from_pretrained(cfg["model_name"])
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                cfg["model_name"], torch_dtype=torch_dtype
            ).to("cuda")
            self.model.eval()
            # Same fix as the HF adapter: clear the legacy generation config
            # bits that conflict with modern language/task kwargs.
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.forced_decoder_ids = None
                self.model.generation_config.suppress_tokens = []
            print(
                f"[WhisperDeployment] loaded {cfg['model_name']} dtype={cfg['dtype']}"
            )

        @serve.batch(
            max_batch_size=cfg["max_batch_size"],
            batch_wait_timeout_s=cfg["batch_wait_timeout_s"],
        )
        async def _transcribe_batch(self, audio_payloads: list[dict]) -> list[str]:
            import numpy as np
            import torch

            # All inputs must share the same sample rate (16kHz for our eval sets).
            arrays = [np.asarray(p["array"]) for p in audio_payloads]
            sr = audio_payloads[0]["sr"]
            inputs = self.processor(
                arrays, sampling_rate=sr, return_tensors="pt", padding=True
            )
            input_features = inputs.input_features.to("cuda", dtype=self._dtype)

            generate_kwargs: dict[str, Any] = {"max_new_tokens": 440}
            if self._language:
                generate_kwargs["language"] = self._language
                generate_kwargs["task"] = "transcribe"

            with torch.inference_mode():
                predicted_ids = self.model.generate(input_features, **generate_kwargs)
            return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        async def __call__(self, request: Request) -> dict[str, str]:
            import soundfile as sf

            body = await request.body()
            arr, sr = sf.read(BytesIO(body), dtype="float32")
            if arr.ndim > 1:
                # Mono-mix multi-channel input.
                arr = arr.mean(axis=1)
            text = await self._transcribe_batch({"array": arr, "sr": int(sr)})
            return {"text": text}

    return WhisperDeployment.bind()


# `serve run adapters.ray_serve_deployment:app` looks for this module-level binding.
app = build_app()
