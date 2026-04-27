"""FrameworkAdapter — the interface every framework integration implements.

Two adapter shapes are supported by the same interface:

1. In-process adapters (HF Transformers, faster-whisper) — load the model in
   the same Python process. setup() loads weights to GPU. transcribe() runs
   a forward pass.

2. Out-of-process adapters (Triton, vLLM in server mode, Ray Serve) — connect
   to a separately-running server. setup() establishes a client and pings the
   server. transcribe() sends an HTTP/gRPC request.

The harness doesn't care which shape the adapter is — it just calls transcribe().
This means in-process adapters lose the framework's HTTP/serialization overhead
relative to out-of-process ones, which is real and worth flagging in the writeup
(it's part of "what the framework is actually offering"). We don't try to
artificially equalize this; we measure what each framework actually does in its
intended deployment mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bench.config import CellConfig, Framework
from bench.data import AudioClip


class FrameworkAdapter(ABC):
    """Abstract base for framework adapters.

    Subclasses implement setup(), aclose(), and transcribe(). The harness owns
    the lifecycle: it calls setup() once before warmup, transcribe() many times,
    aclose() once at the end.
    """

    def __init__(self, config: CellConfig):
        self.config = config

    @abstractmethod
    async def setup(self) -> None:
        """Prepare the adapter for serving requests.

        For in-process adapters: load the model to GPU.
        For out-of-process adapters: connect to the server, verify readiness.

        Should raise on setup failure — the harness will mark the cell as 'error'.
        """

    @abstractmethod
    async def transcribe(self, clip: AudioClip) -> str:
        """Transcribe one audio clip. Return the hypothesis as a string.

        May raise on transcription failure. The load generator catches and
        records the failure as an error on the request record.
        """

    @abstractmethod
    async def aclose(self) -> None:
        """Release resources (free GPU memory, close HTTP clients, etc.).

        Called once at the end of the cell run. Idempotent.
        """

    def metadata(self) -> dict[str, Any]:
        """Optional adapter-reported metadata to attach to the results JSON.

        Useful for capturing things like the actual batch size used, the model
        revision SHA, etc. — anything the adapter knows that's worth preserving.
        Default returns empty dict.
        """
        return {}


def build_adapter(config: CellConfig) -> FrameworkAdapter:
    """Factory: construct the adapter for config.framework.

    Imports are deferred to avoid pulling heavy framework deps when only a
    different framework is being used. This means importing this module is
    cheap; importing the chosen adapter loads its deps.
    """
    if config.framework == Framework.HF:
        from adapters.hf_transformers import HfTransformersAdapter

        return HfTransformersAdapter(config)
    if config.framework == Framework.FASTER_WHISPER:
        from adapters.faster_whisper_adapter import FasterWhisperAdapter

        return FasterWhisperAdapter(config)
    if config.framework == Framework.VLLM:
        from adapters.vllm_adapter import VllmAdapter

        return VllmAdapter(config)
    if config.framework == Framework.RAY_SERVE:
        from adapters.ray_serve_adapter import RayServeAdapter

        return RayServeAdapter(config)
    if config.framework == Framework.TRITON:
        from adapters.triton_adapter import TritonAdapter

        return TritonAdapter(config)
    raise ValueError(f"Unknown framework: {config.framework}")
