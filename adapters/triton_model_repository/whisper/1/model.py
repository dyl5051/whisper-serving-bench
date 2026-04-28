"""Triton Python backend implementation for Whisper.

Loaded by Triton at server startup. Triton runs this in its own Python
interpreter (separate from the harness process), so this file cannot import
anything from our `bench/` or `adapters/` packages. It only depends on
packages that Triton's Python env can resolve: torch, transformers,
soundfile, numpy, and Triton's own `triton_python_backend_utils`.

Configuration is read from config.pbtxt parameters at initialize() time.

v1 design: max_batch_size=0 in config.pbtxt, so execute() receives one
request at a time. The interesting batching story for Triton lands in v1.1
with TensorRT-LLM compilation; this Python-backend cell is the "naive
Triton wrapping" baseline against which TensorRT-LLM's value is measured.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import triton_python_backend_utils as pb_utils  # type: ignore[import-not-found]


class TritonPythonModel:
    def initialize(self, args: dict[str, Any]) -> None:
        # Imports deferred so that this module can be imported by tools that
        # don't have torch installed (e.g. linting in CI).
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        model_config = json.loads(args["model_config"])
        params = {p["key"]: p["value"]["string_value"] for p in model_config.get("parameters", [])} \
            if isinstance(model_config.get("parameters"), list) \
            else {k: v["string_value"] for k, v in (model_config.get("parameters") or {}).items()}

        model_name = params.get("MODEL_NAME", "openai/whisper-large-v3")
        dtype_name = params.get("DTYPE", "float16")
        self._language = params.get("LANGUAGE") or None

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]
        self._torch_dtype = torch_dtype

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=torch_dtype
        ).to("cuda")
        self.model.eval()
        # Same Whisper-large-v3 quirk we hit elsewhere: clear stale legacy
        # decoder ids that conflict with the modern language/task kwargs.
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.forced_decoder_ids = None
            self.model.generation_config.suppress_tokens = []

        print(
            f"[triton-python whisper] initialized {model_name} dtype={dtype_name}",
            flush=True,
        )

    def execute(self, requests: list) -> list:
        import soundfile as sf
        import torch

        responses = []
        for request in requests:
            try:
                audio_bytes_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_BYTES")
                # numpy object array of byte strings; we expect one per request
                # since max_batch_size=0 means inputs are not batched here.
                raw = audio_bytes_tensor.as_numpy().reshape(-1)[0]
                if isinstance(raw, np.bytes_):
                    raw_bytes = bytes(raw)
                elif isinstance(raw, bytes):
                    raw_bytes = raw
                else:
                    raw_bytes = bytes(raw)

                arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)

                inputs = self.processor(arr, sampling_rate=sr, return_tensors="pt")
                input_features = inputs.input_features.to("cuda", dtype=self._torch_dtype)

                generate_kwargs: dict[str, Any] = {"max_new_tokens": 440}
                if self._language:
                    generate_kwargs["language"] = self._language
                    generate_kwargs["task"] = "transcribe"

                with torch.inference_mode():
                    predicted_ids = self.model.generate(input_features, **generate_kwargs)
                text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                text_tensor = pb_utils.Tensor("TEXT", np.array([text], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[text_tensor]))
            except Exception as e:  # noqa: BLE001
                err = pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("TEXT", np.array([""], dtype=object))],
                    error=pb_utils.TritonError(f"{type(e).__name__}: {e}"),
                )
                responses.append(err)
        return responses

    def finalize(self) -> None:
        # Triton lifecycle hook for shutdown. Nothing special needed.
        self.model = None
        self.processor = None
