"""vLLM adapter — continuous batching + PagedAttention.

vLLM is the canonical example of a serving framework: it runs as a separate
server process, owns request scheduling, batches concurrent requests via
continuous (in-flight) batching, and manages KV-cache memory via PagedAttention.

We exercise it the way real users do: the adapter spawns `vllm serve` as a
subprocess, then issues HTTP requests against its OpenAI-compatible
`/v1/audio/transcriptions` endpoint. This deliberately INCLUDES the HTTP
serialization overhead in our latency numbers — that overhead is part of what
vLLM is offering, and pretending it doesn't exist would misrepresent the
production-deployment story.

Adapter config knobs:
    port: int (default 8765)
    base_url: str — overrides if you're connecting to an existing vLLM server
        instead of letting the adapter spawn one
    dtype: "float16" | "bfloat16" | "auto" (default "float16")
    max_num_seqs: int — vLLM's max concurrent sequences (default 256, the
        framework default; bumping this is part of what continuous batching
        buys you and worth varying in v1.x)
    gpu_memory_utilization: float in (0,1] — fraction of GPU memory vLLM may
        use for KV cache (default 0.9)
    language: optional ISO code passed in transcription requests
    server_startup_timeout_seconds: how long to wait for /health (default 180)
    spawn_server: bool (default true) — set false if pointing at an external
        server you started manually
"""

from __future__ import annotations

import asyncio
import gzip
import io
import os
import signal
import subprocess
import sys
import time
from typing import Any

import httpx

from adapters.base import FrameworkAdapter
from adapters.faster_whisper_adapter import _HF_TO_FASTER_WHISPER  # reuse the model-name map
from bench.config import CellConfig
from bench.data import AudioClip


class VllmAdapter(FrameworkAdapter):
    def __init__(self, config: CellConfig):
        super().__init__(config)
        self._port: int = int(config.adapter_config.get("port", 8765))
        self._base_url: str = config.adapter_config.get(
            "base_url", f"http://127.0.0.1:{self._port}/v1"
        )
        self._dtype: str = config.adapter_config.get("dtype", "float16")
        self._max_num_seqs: int = int(config.adapter_config.get("max_num_seqs", 256))
        self._gpu_memory_utilization: float = float(
            config.adapter_config.get("gpu_memory_utilization", 0.9)
        )
        self._language: str | None = config.adapter_config.get("language")
        self._spawn_server: bool = bool(config.adapter_config.get("spawn_server", True))
        self._server_startup_timeout_seconds: float = float(
            config.adapter_config.get("server_startup_timeout_seconds", 180.0)
        )
        # Hallucination-mitigation knobs added in v1.0.1 experiment.
        # See writeups/v1.md "Should you self-host" and METHODOLOGY.md.
        # prompt: OpenAI-compatible prompt prefix to anchor the model.
        # compression_ratio_threshold: post-hoc reject outputs with gzip ratio
        #   above this (faster-whisper default is 2.4). Replaces high-CR text
        #   with empty string — emulates faster-whisper's compression-ratio
        #   safeguard that vLLM's Whisper wrapper omits.
        # vad_filter_input: if True, preprocess audio with silero-vad to strip
        #   non-speech regions before sending. Attacks the root cause directly.
        self._prompt: str | None = config.adapter_config.get("prompt")
        self._compression_ratio_threshold: float | None = (
            float(config.adapter_config["compression_ratio_threshold"])
            if config.adapter_config.get("compression_ratio_threshold") is not None
            else None
        )
        self._vad_filter_input: bool = bool(config.adapter_config.get("vad_filter_input", False))
        self._vad_model: Any = None  # set lazily in setup() if VAD is enabled

        # vLLM serves the model under the same name passed to it. We use the
        # canonical HF id so the served model name matches what clients query.
        self._served_model_name: str = config.model
        self._server_proc: subprocess.Popen[bytes] | None = None
        self._client: httpx.AsyncClient | None = None

    async def setup(self) -> None:
        if self._spawn_server:
            await self._spawn_vllm_server()
        # HTTP client with generous per-request timeout. The harness's own
        # request_timeout_seconds bounds individual transcribe() calls; this
        # client-level timeout is just an upper bound to surface hangs.
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0),
        )
        await self._wait_for_ready()
        if self._vad_filter_input:
            # Lazy-load silero-vad so non-VAD cells don't pay the import cost.
            from silero_vad import load_silero_vad
            loop = asyncio.get_running_loop()
            self._vad_model = await loop.run_in_executor(None, load_silero_vad)
            print("[VllmAdapter] silero-vad model loaded")
        print(f"[VllmAdapter] vLLM server ready at {self._base_url}")

    async def _spawn_vllm_server(self) -> None:
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.config.model,
            "--port",
            str(self._port),
            "--host",
            "127.0.0.1",
            "--dtype",
            self._dtype,
            "--max-num-seqs",
            str(self._max_num_seqs),
            "--gpu-memory-utilization",
            str(self._gpu_memory_utilization),
            "--served-model-name",
            self._served_model_name,
            # Note: older vLLM versions required `--task transcription` for Whisper.
            # vLLM >=0.10ish removed this — the engine auto-detects task type from
            # the model architecture. If your vLLM version errors with "unrecognized
            # arguments: --task", update vllm AND remove this line. If your version
            # requires --task explicitly, add it back here.
        ]
        print(f"[VllmAdapter] launching: {' '.join(cmd)}")
        # Pipe stdout+stderr to our stdout so harness logs include vLLM's startup chatter.
        # Use a process group so we can clean up child processes on aclose.
        self._server_proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    async def _wait_for_ready(self) -> None:
        deadline = time.monotonic() + self._server_startup_timeout_seconds
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=5.0) as probe:
            while time.monotonic() < deadline:
                # vLLM's OpenAI-compatible server exposes /health.
                try:
                    r = await probe.get(f"{self._base_url}/../health")
                    if r.status_code == 200:
                        return
                except httpx.HTTPError as e:
                    last_err = e
                # Check whether the subprocess has died — pointless to keep
                # waiting if vLLM crashed during startup.
                if self._server_proc is not None and self._server_proc.poll() is not None:
                    raise RuntimeError(
                        f"vLLM server exited with code {self._server_proc.returncode} "
                        f"during startup"
                    )
                await asyncio.sleep(2.0)
        raise TimeoutError(
            f"vLLM server did not become ready within "
            f"{self._server_startup_timeout_seconds}s. Last error: {last_err}"
        )

    async def transcribe(self, clip: AudioClip) -> str:
        if self._client is None:
            raise RuntimeError("Adapter.setup() must be called before transcribe()")

        # Load audio bytes. If VAD pre-filter is enabled, decode → run VAD →
        # concatenate speech regions → re-encode WAV before upload. This
        # attacks vLLM's hallucination root cause (model freelances on silence)
        # by removing the silence the model is invited to freelance on.
        if self._vad_filter_input:
            audio_bytes, audio_filename = await self._vad_filter_clip(clip)
        else:
            with open(clip.audio_path, "rb") as f:
                audio_bytes = f.read()
            audio_filename = clip.audio_path.name

        files = {"file": (audio_filename, audio_bytes, "audio/wav")}
        data: dict[str, str] = {
            "model": self._served_model_name,
            # temperature=0 forces greedy decoding. Whisper's default temperature
            # fallback (try higher temps if compression ratio is too high) helps
            # with repetition loops, but vLLM doesn't implement that fallback.
            # Greedy at least removes one source of randomness.
            "temperature": "0",
            "response_format": "json",
        }
        if self._language:
            data["language"] = self._language
        if self._prompt is not None:
            data["prompt"] = self._prompt

        response = await self._client.post("/audio/transcriptions", files=files, data=data)
        response.raise_for_status()
        payload = response.json()
        # OpenAI-compatible response shape: {"text": "..."}
        text: str = payload.get("text", "")

        # Post-hoc compression-ratio filter emulates faster-whisper's safeguard.
        # Repetitive hallucinations like "thanks for watching ×20" gzip-compress
        # to a tiny fraction of their size, producing a high ratio. Real speech
        # has ratio ~1.5-2.0; loops push it above 2.4 quickly.
        if self._compression_ratio_threshold is not None and text:
            cr = _gzip_compression_ratio(text)
            if cr > self._compression_ratio_threshold:
                # Discarded as a hallucination loop. Emit empty string; the
                # harness reports this as a successful request with empty
                # hypothesis, which propagates correctly through WER computation
                # (empty hyp vs non-empty ref → all deletions).
                return ""

        return text

    async def _vad_filter_clip(self, clip: AudioClip) -> tuple[bytes, str]:
        """Run silero-vad on the clip's audio, return (speech-only-wav-bytes, filename)."""
        if self._vad_model is None:
            raise RuntimeError("VAD model not loaded — set vad_filter_input=True in adapter_config")

        # Heavy import deferred to actual use.
        import numpy as np
        import soundfile as sf
        import torch
        from silero_vad import get_speech_timestamps

        loop = asyncio.get_running_loop()

        def _run() -> tuple[bytes, str]:
            arr, sr = sf.read(str(clip.audio_path), dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1).astype(np.float32)
            # silero-vad expects 16 kHz; resample if needed.
            if sr != 16000:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                sr = 16000

            wav_tensor = torch.from_numpy(arr)
            timestamps = get_speech_timestamps(
                wav_tensor, self._vad_model, sampling_rate=sr
            )
            if not timestamps:
                # Fully silent clip — return a tiny silence WAV. The model
                # gets nothing to transcribe and should emit nothing.
                speech_arr = np.zeros(int(0.1 * sr), dtype=np.float32)
            else:
                # Concatenate speech regions back-to-back.
                speech_arr = np.concatenate(
                    [arr[seg["start"]:seg["end"]] for seg in timestamps]
                ).astype(np.float32)

            buf = io.BytesIO()
            sf.write(buf, speech_arr, sr, format="WAV", subtype="PCM_16")
            return buf.getvalue(), clip.audio_path.stem + ".vad.wav"

        return await loop.run_in_executor(None, _run)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

        if self._server_proc is not None and self._server_proc.poll() is None:
            # Send SIGTERM to the whole process group to catch any vLLM workers.
            try:
                os.killpg(os.getpgid(self._server_proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self._server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                # Escalate to SIGKILL if SIGTERM didn't work
                try:
                    os.killpg(os.getpgid(self._server_proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self._server_proc.wait(timeout=5)
        self._server_proc = None

    def metadata(self) -> dict[str, Any]:
        # Surface the model-name resolution so the results JSON records what was actually served.
        fw_short = _HF_TO_FASTER_WHISPER.get(self.config.model, self.config.model)
        return {
            "port": self._port,
            "base_url": self._base_url,
            "dtype": self._dtype,
            "max_num_seqs": self._max_num_seqs,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "language": self._language,
            "spawn_server": self._spawn_server,
            "served_model_name": self._served_model_name,
            "model_canonical_short": fw_short,  # informational
            "prompt": self._prompt,
            "compression_ratio_threshold": self._compression_ratio_threshold,
            "vad_filter_input": self._vad_filter_input,
        }


def _gzip_compression_ratio(text: str) -> float:
    """Faster-whisper's compression-ratio safeguard, ported.

    Real speech transcripts compress to roughly 1.5-2.0 ratio. Repetition loops
    ("thanks for watching, thanks for watching, ...") compress aggressively and
    blow past 2.4 quickly. Above the threshold = hallucination loop = discard.
    """
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    compressed = gzip.compress(raw, compresslevel=6)
    return len(raw) / len(compressed)
