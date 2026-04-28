"""Ray Serve adapter — request queuing + Python-native batching primitives.

Ray Serve's positioning vs the others: you bring the inference engine and
Ray gives you the batching/queuing primitives. We pair it with HF
Transformers' direct `model.generate()` (because batching at the tensor
level is what `@serve.batch` is designed to leverage).

The adapter spawns `serve run` as a subprocess pointing at the deployment
defined in `adapters.ray_serve_deployment`. Once the deployment's HTTP
endpoint is healthy, transcribe() POSTs raw WAV bytes to it.

Adapter config knobs (passed to the deployment via env vars):
    port: int (default 8765)
    max_batch_size: int (default 8) — Ray Serve's per-batch ceiling
    batch_wait_timeout_s: float (default 0.02) — how long Ray Serve waits to
        form a batch before flushing
    num_replicas: int (default 1)
    dtype: "float16" | "bfloat16" | "float32" (default "float16")
    language: optional ISO code
    server_startup_timeout_seconds: float (default 180)
    spawn_server: bool (default true)
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
from bench.config import CellConfig
from bench.data import AudioClip


class RayServeAdapter(FrameworkAdapter):
    def __init__(self, config: CellConfig):
        super().__init__(config)
        self._port: int = int(config.adapter_config.get("port", 8765))
        self._endpoint_url: str = config.adapter_config.get(
            "endpoint_url", f"http://127.0.0.1:{self._port}/"
        )
        self._max_batch_size: int = int(config.adapter_config.get("max_batch_size", 8))
        self._batch_wait_timeout_s: float = float(
            config.adapter_config.get("batch_wait_timeout_s", 0.02)
        )
        self._num_replicas: int = int(config.adapter_config.get("num_replicas", 1))
        self._dtype: str = config.adapter_config.get("dtype", "float16")
        self._language: str | None = config.adapter_config.get("language")
        self._server_startup_timeout_seconds: float = float(
            config.adapter_config.get("server_startup_timeout_seconds", 180.0)
        )
        self._spawn_server: bool = bool(config.adapter_config.get("spawn_server", True))
        self._server_proc: subprocess.Popen[bytes] | None = None
        self._client: httpx.AsyncClient | None = None

    async def setup(self) -> None:
        if self._spawn_server:
            await self._spawn_serve_run()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0),
        )
        await self._wait_for_ready()
        print(f"[RayServeAdapter] deployment ready at {self._endpoint_url}")

    async def _spawn_serve_run(self) -> None:
        env = os.environ.copy()
        env["BENCH_MODEL_NAME"] = self.config.model
        env["BENCH_DTYPE"] = self._dtype
        env["BENCH_MAX_BATCH_SIZE"] = str(self._max_batch_size)
        env["BENCH_BATCH_WAIT_TIMEOUT_S"] = str(self._batch_wait_timeout_s)
        env["BENCH_NUM_REPLICAS"] = str(self._num_replicas)
        if self._language:
            env["BENCH_LANGUAGE"] = self._language

        cmd = [
            sys.executable,
            "-m",
            "ray.serve.scripts",
            "run",
            "adapters.ray_serve_deployment:app",
            "--port",
            str(self._port),
            "--host",
            "127.0.0.1",
            "--non-blocking",
        ]
        # Newer ray: `serve run` is exposed via `python -m ray.serve.scripts`.
        # If this entry point changes in a future ray version, adjust here.
        print(f"[RayServeAdapter] launching: {' '.join(cmd)}")
        self._server_proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )

    async def _wait_for_ready(self) -> None:
        deadline = time.monotonic() + self._server_startup_timeout_seconds
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=5.0) as probe:
            while time.monotonic() < deadline:
                try:
                    r = await probe.get(self._endpoint_url)
                    # Even a 405/422 from the deployment means it's accepting
                    # connections; the GET-with-no-body just isn't a real
                    # transcription request.
                    if r.status_code in (200, 400, 405, 415, 422):
                        return
                except httpx.HTTPError as e:
                    last_err = e
                if self._server_proc is not None and self._server_proc.poll() is not None:
                    raise RuntimeError(
                        f"Ray Serve subprocess exited with code "
                        f"{self._server_proc.returncode} during startup"
                    )
                await asyncio.sleep(2.0)
        raise TimeoutError(
            f"Ray Serve deployment did not respond on {self._endpoint_url} "
            f"within {self._server_startup_timeout_seconds}s. Last error: {last_err}"
        )

    async def transcribe(self, clip: AudioClip) -> str:
        if self._client is None:
            raise RuntimeError("Adapter.setup() must be called before transcribe()")
        with open(clip.audio_path, "rb") as f:
            audio_bytes = f.read()
        response = await self._client.post(
            self._endpoint_url,
            content=audio_bytes,
            headers={"Content-Type": "audio/wav"},
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("text", "")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

        if self._server_proc is not None and self._server_proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._server_proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self._server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._server_proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self._server_proc.wait(timeout=5)
        self._server_proc = None

    def metadata(self) -> dict[str, Any]:
        return {
            "port": self._port,
            "endpoint_url": self._endpoint_url,
            "max_batch_size": self._max_batch_size,
            "batch_wait_timeout_s": self._batch_wait_timeout_s,
            "num_replicas": self._num_replicas,
            "dtype": self._dtype,
            "language": self._language,
            "spawn_server": self._spawn_server,
        }
