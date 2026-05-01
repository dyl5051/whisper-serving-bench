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

        # Send the WAV file via the OpenAI-compatible /audio/transcriptions endpoint.
        # multipart/form-data is what real OpenAI Whisper API clients send, so this
        # measurement is representative of production usage.
        with open(clip.audio_path, "rb") as f:
            audio_bytes = f.read()

        files = {"file": (clip.audio_path.name, audio_bytes, "audio/wav")}
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

        response = await self._client.post("/audio/transcriptions", files=files, data=data)
        response.raise_for_status()
        payload = response.json()
        # OpenAI-compatible response shape: {"text": "..."}
        return payload.get("text", "")

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
        }
