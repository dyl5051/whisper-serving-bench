"""Cell configuration — the unit of measurement in the benchmark.

A "cell" is one (framework, model, GPU, workload, concurrency) combination.
Cells are defined as YAML files under configs/cells/ and loaded into CellConfig
objects via Pydantic for validation.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class Framework(str, Enum):
    HF = "hf"
    FASTER_WHISPER = "faster_whisper"
    VLLM = "vllm"
    RAY_SERVE = "ray_serve"
    TRITON = "triton"


class WorkloadPattern(str, Enum):
    # Closed-loop concurrent workload. N workers each pull from the eval set, send
    # one request, wait for response, send the next. Models "N active users."
    CONCURRENT_STREAMING = "concurrent_streaming"

    # Batch throughput: process the entire eval set as fast as possible, framework
    # decides batching strategy. Concurrency is ignored (or interpreted as a hint).
    BATCH = "batch"

    # Single-stream low-latency: one request at a time, measure per-request latency
    # without queuing pressure.
    SINGLE_STREAM = "single_stream"


class WorkloadConfig(BaseModel):
    pattern: WorkloadPattern
    concurrency: int = Field(ge=1, description="Number of concurrent workers (ignored for batch).")
    eval_set: str = Field(description="Name of an eval set defined in configs/eval_sets.yaml.")
    iterations: int = Field(default=3, ge=1, description="How many times to run the full eval set.")
    warmup_requests: int = Field(
        default=10,
        ge=0,
        description="Sequential warm-up requests before timing starts. Excluded from metrics.",
    )
    request_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Per-request timeout. Requests exceeding this are recorded as failures.",
    )


class CellConfig(BaseModel):
    cell_id: str = Field(
        description="Unique identifier, e.g. 'hf_a100_concurrent_8'. Becomes the results filename."
    )
    framework: Framework
    model: str = Field(description="Model identifier, e.g. 'openai/whisper-large-v3'.")
    gpu: str = Field(
        description="GPU identifier for results provenance, e.g. 'A100-SXM4-40GB'. Not authoritative — actual GPU is captured via nvidia-smi at runtime."
    )
    workload: WorkloadConfig
    adapter_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Framework-specific knobs passed through to the adapter constructor.",
    )

    @field_validator("cell_id")
    @classmethod
    def _validate_cell_id(cls, v: str) -> str:
        if not v or " " in v or "/" in v:
            raise ValueError(f"cell_id must be a non-empty filename-safe string, got: {v!r}")
        return v

    @classmethod
    def from_yaml(cls, path: Path | str) -> CellConfig:
        path = Path(path)
        with path.open() as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
