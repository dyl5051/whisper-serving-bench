"""Whisper serving frameworks benchmark — measurement harness.

The harness is framework-agnostic: it takes a CellConfig + a FrameworkAdapter and
produces a ResultsJSON with latency, throughput, accuracy, and GPU telemetry.

Public surface:
    from bench import run_cell, CellConfig, FrameworkAdapter
"""

from bench.config import CellConfig, WorkloadConfig
from bench.harness import run_cell

__all__ = ["CellConfig", "WorkloadConfig", "run_cell"]
