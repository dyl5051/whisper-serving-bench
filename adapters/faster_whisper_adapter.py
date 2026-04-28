"""faster-whisper adapter — CTranslate2-compiled Whisper inference.

faster-whisper is a CTranslate2 reimplementation of Whisper. CTranslate2
fuses transformer ops into custom CUDA kernels, giving roughly 4x speedup
over vanilla HF Transformers on the same model architecture, with smaller
memory footprint. It's a "library" rather than a "server" — no HTTP, no
external batching; the harness just calls model.transcribe() per request.

We load audio ourselves with soundfile and hand faster-whisper a numpy
array, bypassing its default pyav/FFmpeg audio decoder for the same
reason we do it in the HF adapter: the RunPod PyTorch container doesn't
ship FFmpeg, and even if it did we want one consistent decode path
across all adapters so the audio bytes the model sees are byte-for-byte
identical from framework to framework.

Adapter config knobs:
    device: "cuda" | "cpu" (default "cuda")
    compute_type: "float16" | "bfloat16" | "int8_float16" | "float32"
        (default "float16" — what most production deployments use on A100)
    language: optional ISO code; None means auto-detect
    beam_size: int (default 5 — faster-whisper's default; matches Whisper paper)
    fw_model_override: optional explicit faster-whisper model name. If unset,
        we translate config.model from the HF canonical name to the
        faster-whisper short name via a known mapping.
"""

from __future__ import annotations

import asyncio
from typing import Any

from adapters.base import FrameworkAdapter
from bench.config import CellConfig
from bench.data import AudioClip

# Translate the canonical HuggingFace model IDs to faster-whisper's short names.
# faster-whisper resolves short names to its CTranslate2 weight repo
# (Systran/faster-whisper-*) automatically.
_HF_TO_FASTER_WHISPER = {
    "openai/whisper-tiny": "tiny",
    "openai/whisper-tiny.en": "tiny.en",
    "openai/whisper-base": "base",
    "openai/whisper-base.en": "base.en",
    "openai/whisper-small": "small",
    "openai/whisper-small.en": "small.en",
    "openai/whisper-medium": "medium",
    "openai/whisper-medium.en": "medium.en",
    "openai/whisper-large-v2": "large-v2",
    "openai/whisper-large-v3": "large-v3",
    "openai/whisper-large-v3-turbo": "large-v3-turbo",
}


class FasterWhisperAdapter(FrameworkAdapter):
    def __init__(self, config: CellConfig):
        super().__init__(config)
        self._model: Any = None
        self._device: str = config.adapter_config.get("device", "cuda")
        self._compute_type: str = config.adapter_config.get("compute_type", "float16")
        self._language: str | None = config.adapter_config.get("language")
        self._beam_size: int = int(config.adapter_config.get("beam_size", 5))
        override = config.adapter_config.get("fw_model_override")
        self._fw_model_id: str = override or _HF_TO_FASTER_WHISPER.get(
            config.model, config.model
        )

    async def setup(self) -> None:
        from faster_whisper import WhisperModel

        loop = asyncio.get_running_loop()

        def _build():
            return WhisperModel(
                self._fw_model_id,
                device=self._device,
                compute_type=self._compute_type,
            )

        self._model = await loop.run_in_executor(None, _build)
        print(
            f"[FasterWhisperAdapter] loaded {self._fw_model_id} (from {self.config.model}) "
            f"on {self._device} compute_type={self._compute_type}"
        )

    async def transcribe(self, clip: AudioClip) -> str:
        if self._model is None:
            raise RuntimeError("Adapter.setup() must be called before transcribe()")

        import numpy as np
        import soundfile as sf

        loop = asyncio.get_running_loop()

        def _run():
            arr, sr = sf.read(str(clip.audio_path), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1).astype(np.float32)
            if sr != 16000:
                # faster-whisper expects 16kHz. LibriSpeech is already 16kHz so
                # this branch isn't hit in v1, but other eval sets might need it.
                # We could resample here with librosa, but for now we error rather
                # than silently producing different numbers from other adapters.
                raise ValueError(
                    f"Expected 16kHz audio, got {sr}Hz. Resampling not implemented."
                )

            segments_iter, _info = self._model.transcribe(
                arr,
                language=self._language,
                beam_size=self._beam_size,
            )
            # segments_iter is a generator; consuming it runs the actual decode.
            return " ".join(seg.text.strip() for seg in segments_iter)

        return await loop.run_in_executor(None, _run)

    async def aclose(self) -> None:
        # faster-whisper has no explicit close — drop reference and let
        # CTranslate2's destructor free the GPU memory.
        self._model = None

    def metadata(self) -> dict[str, Any]:
        return {
            "device": self._device,
            "compute_type": self._compute_type,
            "language": self._language,
            "beam_size": self._beam_size,
            "fw_model_id": self._fw_model_id,
        }
