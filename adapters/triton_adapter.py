"""Triton adapter — NVIDIA Triton Inference Server, Python backend.

Spawns `tritonserver` as a subprocess pointing at our model repository
(adapters/triton_model_repository/), waits for /v2/health/ready, then
issues requests via tritonclient's HTTP async client.

For v1 we use Triton's Python backend with max_batch_size=0 — the
"naive deployment" baseline. The TensorRT-LLM compilation path (where
Triton's batching actually shines) is v1.1's headline story, kept
deliberately separate so the comparison "Triton-Python vs Triton-TRT-LLM"
makes the framework's value proposition concrete.

Adapter config knobs:
    http_port: int (default 8000) — Triton's default
    grpc_port: int (default 8001)
    metrics_port: int (default 8002)
    model_repository: path (default adapters/triton_model_repository)
    model_name: str (default "whisper" — matches config.pbtxt name)
    server_startup_timeout_seconds: float (default 240) — Triton+Whisper-large
        is slow to load; budget generously
    spawn_server: bool (default true) — set false to talk to an existing
        Triton instance you started yourself
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from adapters.base import FrameworkAdapter
from bench.config import CellConfig
from bench.data import AudioClip


class TritonAdapter(FrameworkAdapter):
    def __init__(self, config: CellConfig):
        super().__init__(config)
        self._http_port: int = int(config.adapter_config.get("http_port", 8000))
        self._grpc_port: int = int(config.adapter_config.get("grpc_port", 8001))
        self._metrics_port: int = int(config.adapter_config.get("metrics_port", 8002))
        self._model_repository: Path = Path(
            config.adapter_config.get(
                "model_repository", "adapters/triton_model_repository"
            )
        )
        self._model_name: str = config.adapter_config.get("model_name", "whisper")
        self._server_startup_timeout_seconds: float = float(
            config.adapter_config.get("server_startup_timeout_seconds", 240.0)
        )
        self._spawn_server: bool = bool(config.adapter_config.get("spawn_server", True))
        self._server_proc: subprocess.Popen[bytes] | None = None
        self._client: Any = None  # tritonclient async InferenceServerClient

    async def setup(self) -> None:
        if self._spawn_server:
            await self._spawn_triton()
        # Imports deferred — tritonclient is heavy and only needed for this adapter.
        import tritonclient.http.aio as httpclient

        self._client = httpclient.InferenceServerClient(
            url=f"127.0.0.1:{self._http_port}",
            concurrency=128,  # client-side parallelism; harness drives the actual concurrency
        )
        await self._wait_for_ready()
        # Verify the model is ready (vs server merely accepting connections).
        ready = await self._client.is_model_ready(self._model_name)
        if not ready:
            raise RuntimeError(
                f"Triton server is up but model {self._model_name!r} reports not ready"
            )
        print(f"[TritonAdapter] Triton ready at 127.0.0.1:{self._http_port}, model={self._model_name}")

    async def _spawn_triton(self) -> None:
        if not self._model_repository.exists():
            raise FileNotFoundError(
                f"Triton model repository not found: {self._model_repository}"
            )
        cmd = [
            "tritonserver",
            f"--model-repository={self._model_repository}",
            f"--http-port={self._http_port}",
            f"--grpc-port={self._grpc_port}",
            f"--metrics-port={self._metrics_port}",
            "--log-verbose=0",
        ]
        print(f"[TritonAdapter] launching: {' '.join(cmd)}")
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
                try:
                    r = await probe.get(f"http://127.0.0.1:{self._http_port}/v2/health/ready")
                    if r.status_code == 200:
                        return
                except httpx.HTTPError as e:
                    last_err = e
                if self._server_proc is not None and self._server_proc.poll() is not None:
                    raise RuntimeError(
                        f"Triton subprocess exited with code "
                        f"{self._server_proc.returncode} during startup"
                    )
                await asyncio.sleep(2.0)
        raise TimeoutError(
            f"Triton server did not become ready within "
            f"{self._server_startup_timeout_seconds}s. Last error: {last_err}"
        )

    async def transcribe(self, clip: AudioClip) -> str:
        if self._client is None:
            raise RuntimeError("Adapter.setup() must be called before transcribe()")
        import tritonclient.http.aio as httpclient

        with open(clip.audio_path, "rb") as f:
            audio_bytes = f.read()

        # AUDIO_BYTES tensor: shape [1] with one byte string element.
        audio_arr = np.array([audio_bytes], dtype=object)
        in_tensor = httpclient.InferInput("AUDIO_BYTES", [1], "BYTES")
        in_tensor.set_data_from_numpy(audio_arr)

        out_request = httpclient.InferRequestedOutput("TEXT")

        result = await self._client.infer(
            model_name=self._model_name,
            inputs=[in_tensor],
            outputs=[out_request],
        )
        text_arr = result.as_numpy("TEXT")
        if text_arr is None:
            return ""
        raw = text_arr.reshape(-1)[0]
        return raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw)

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

        if self._server_proc is not None and self._server_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._server_proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self._server_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._server_proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self._server_proc.wait(timeout=5)
        self._server_proc = None

    def metadata(self) -> dict[str, Any]:
        return {
            "http_port": self._http_port,
            "model_repository": str(self._model_repository),
            "model_name": self._model_name,
            "spawn_server": self._spawn_server,
            "backend": "python",
            "dynamic_batching": False,
        }
