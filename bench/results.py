"""Results JSON schema and serialization.

Every cell run produces a JSON file with this structure. The schema is the
public contract of the benchmark — once published, breaking changes here mean
historical results can't be re-analyzed by the new analysis code. Version it
explicitly so we can migrate cleanly.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.config import CellConfig
from bench.metrics import GpuTelemetrySummary, LatencySummary, RequestRecord
from bench.wer import WerBreakdown

RESULTS_SCHEMA_VERSION = "1.0"


@dataclass
class ProvenanceInfo:
    """Where + when this result was generated. Critical for reproducibility claims."""

    schema_version: str
    timestamp_utc: str
    git_sha: str
    docker_image_tag: str
    hostname: str
    python_version: str
    platform: str

    @classmethod
    def capture(cls) -> ProvenanceInfo:
        return cls(
            schema_version=RESULTS_SCHEMA_VERSION,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            git_sha=os.environ.get("BENCH_GIT_SHA") or _try_git_sha(),
            docker_image_tag=os.environ.get("BENCH_DOCKER_IMAGE", "unknown"),
            hostname=socket.gethostname(),
            python_version=platform.python_version(),
            platform=platform.platform(),
        )


def _try_git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return out.stdout.strip()[:12]
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


@dataclass
class CellOutcome:
    """Top-level status for the cell. Failures are findings, not omissions."""

    # 'ok' — finished cleanly, all metrics valid
    # 'degraded' — finished but >10% of requests failed; metrics may be misleading
    # 'unsupported' — framework refused this configuration (e.g. concurrency too high)
    # 'error' — exception during cell setup or execution; no metrics
    status: str
    failure_rate: float
    notes: str = ""


@dataclass
class ResultsJson:
    """The full per-cell results document. Serialize with to_dict()."""

    provenance: ProvenanceInfo
    cell_config: dict[str, Any]  # the YAML-loaded CellConfig as a dict
    outcome: CellOutcome

    # Aggregated metrics (None if outcome.status == "error" or "unsupported")
    latency: dict[str, Any] | None = None
    rtf: dict[str, Any] | None = None
    wer: dict[str, Any] | None = None
    gpu: dict[str, Any] | None = None
    throughput: dict[str, Any] | None = None

    # Per-request log. Large; can be omitted in summary views.
    requests: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": asdict(self.provenance),
            "cell_config": self.cell_config,
            "outcome": asdict(self.outcome),
            "latency": self.latency,
            "rtf": self.rtf,
            "wer": self.wer,
            "gpu": self.gpu,
            "throughput": self.throughput,
            "requests": self.requests,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def build_results(
    config: CellConfig,
    records: list[RequestRecord],
    gpu_summary: GpuTelemetrySummary,
    cell_wall_seconds: float,
) -> ResultsJson:
    """Aggregate raw per-request records + telemetry into a ResultsJson document.

    This is pure (no I/O), so it's easy to unit-test against a synthetic record list.
    """
    successful = [r for r in records if r.succeeded]
    failures = [r for r in records if not r.succeeded]
    failure_rate = len(failures) / len(records) if records else 1.0

    if not successful:
        return ResultsJson(
            provenance=ProvenanceInfo.capture(),
            cell_config=config.model_dump(mode="json"),
            outcome=CellOutcome(
                status="error",
                failure_rate=failure_rate,
                notes=f"All {len(records)} requests failed. First error: "
                + (failures[0].error[:200] if failures else "no requests issued"),
            ),
            requests=[_record_to_dict(r) for r in records],
        )

    status = "degraded" if failure_rate > 0.10 else "ok"
    notes = ""
    if failure_rate > 0.10:
        notes = f"{len(failures)}/{len(records)} requests failed ({failure_rate:.1%}). Metrics may not be representative."

    durations = [r.duration_seconds for r in successful]
    audio_seconds_total = sum(r.audio_seconds for r in successful)

    latency = LatencySummary.from_durations(durations)
    rtf_per_request = [r.duration_seconds / r.audio_seconds for r in successful if r.audio_seconds > 0]

    # WER is computed by the harness (via attach_wer) because it requires the
    # eval-set reference texts that are not carried on RequestRecord. Keeping
    # build_results pure (records + telemetry → metrics) makes it easy to
    # unit-test against synthetic record lists.

    return ResultsJson(
        provenance=ProvenanceInfo.capture(),
        cell_config=config.model_dump(mode="json"),
        outcome=CellOutcome(status=status, failure_rate=failure_rate, notes=notes),
        latency=asdict(latency),
        rtf={
            # Aggregate RTF: total audio / total wall time of timed run.
            # This is the "throughput" RTF — matches what you'd see on a dashboard.
            "aggregate": cell_wall_seconds / audio_seconds_total if audio_seconds_total > 0 else None,
            "per_request_p50": _percentile(rtf_per_request, 0.50) if rtf_per_request else None,
            "per_request_p95": _percentile(rtf_per_request, 0.95) if rtf_per_request else None,
        },
        gpu=asdict(gpu_summary),
        throughput={
            "audio_seconds_processed": audio_seconds_total,
            "wall_seconds": cell_wall_seconds,
            "requests_per_second": len(successful) / cell_wall_seconds if cell_wall_seconds > 0 else 0.0,
        },
        wer=None,  # populated by harness after build_results
        requests=[_record_to_dict(r) for r in records],
    )


def _record_to_dict(r: RequestRecord) -> dict[str, Any]:
    return {
        "clip_id": r.clip_id,
        "iteration": r.iteration,
        "worker_id": r.worker_id,
        "submitted_at": r.submitted_at,
        "completed_at": r.completed_at,
        "duration_seconds": r.duration_seconds,
        "audio_seconds": r.audio_seconds,
        "hypothesis": r.hypothesis,
        "error": r.error,
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("empty values")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def attach_wer(results: ResultsJson, wer: WerBreakdown) -> ResultsJson:
    """Attach a WerBreakdown to a results document. Mutates and returns."""
    results.wer = {
        "wer": wer.wer,
        "substitutions": wer.substitutions,
        "insertions": wer.insertions,
        "deletions": wer.deletions,
        "hits": wer.hits,
        "reference_word_count": wer.reference_word_count,
    }
    return results
