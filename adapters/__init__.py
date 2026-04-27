"""Framework adapters.

Each module in this package exposes an adapter class implementing FrameworkAdapter.
The harness constructs adapters via build_adapter(config) — a factory that
dispatches on config.framework.

Adapters bundle: model loading (or server connection), the per-request
transcribe() coroutine, and any setup/teardown lifecycle.
"""

from adapters.base import FrameworkAdapter, build_adapter

__all__ = ["FrameworkAdapter", "build_adapter"]
