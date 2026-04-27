"""Metrics: latency aggregation + GPU telemetry sampling.

Two responsibilities:

1. LatencyAggregator — given a list of per-request durations, compute p50/p95/p99,
   mean, stddev, count. Pure function over a list.

2. GpuTelemetrySampler — background thread that polls nvidia-smi at 1Hz, accumulating
   utilization + memory samples. Started before the timed run, stopped after, then
   queried for summary statistics.

We deliberately use nvidia-smi (subprocess) over nvml/pynvml because (a) it works
inside containers without extra setup, (b) it captures the same numbers ops teams
look at, (c) sub-1Hz precision isn't useful for our purposes anyway.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from statistics import mean, median, stdev


@dataclass(frozen=True)
class LatencySummary:
    count: int
    mean_seconds: float
    median_seconds: float
    stddev_seconds: float
    p50_seconds: float
    p95_seconds: float
    p99_seconds: float
    min_seconds: float
    max_seconds: float

    @classmethod
    def from_durations(cls, durations: list[float]) -> LatencySummary:
        if not durations:
            raise ValueError("Cannot summarize empty latency list.")
        sorted_d = sorted(durations)

        def percentile(p: float) -> float:
            # Linear interpolation between closest ranks. Matches numpy default.
            if len(sorted_d) == 1:
                return sorted_d[0]
            k = (len(sorted_d) - 1) * p
            f = int(k)
            c = min(f + 1, len(sorted_d) - 1)
            if f == c:
                return sorted_d[f]
            return sorted_d[f] + (sorted_d[c] - sorted_d[f]) * (k - f)

        return cls(
            count=len(durations),
            mean_seconds=mean(durations),
            median_seconds=median(durations),
            stddev_seconds=stdev(durations) if len(durations) > 1 else 0.0,
            p50_seconds=percentile(0.50),
            p95_seconds=percentile(0.95),
            p99_seconds=percentile(0.99),
            min_seconds=min(durations),
            max_seconds=max(durations),
        )


@dataclass
class GpuSample:
    """A single nvidia-smi sample. Timestamp is monotonic seconds for easy diffing."""

    timestamp: float
    utilization_percent: float
    memory_used_mib: float


@dataclass
class GpuTelemetrySummary:
    gpu_name: str
    driver_version: str
    sample_count: int
    util_mean_percent: float
    util_max_percent: float
    util_p50_percent: float
    util_p95_percent: float
    memory_peak_mib: float
    memory_mean_mib: float

    @classmethod
    def from_samples(
        cls, samples: list[GpuSample], gpu_name: str, driver_version: str
    ) -> GpuTelemetrySummary:
        if not samples:
            return cls(
                gpu_name=gpu_name,
                driver_version=driver_version,
                sample_count=0,
                util_mean_percent=0.0,
                util_max_percent=0.0,
                util_p50_percent=0.0,
                util_p95_percent=0.0,
                memory_peak_mib=0.0,
                memory_mean_mib=0.0,
            )
        utils = sorted(s.utilization_percent for s in samples)
        mems = [s.memory_used_mib for s in samples]
        return cls(
            gpu_name=gpu_name,
            driver_version=driver_version,
            sample_count=len(samples),
            util_mean_percent=mean(utils),
            util_max_percent=max(utils),
            util_p50_percent=utils[len(utils) // 2],
            util_p95_percent=utils[min(int(len(utils) * 0.95), len(utils) - 1)],
            memory_peak_mib=max(mems),
            memory_mean_mib=mean(mems),
        )


class GpuTelemetrySampler:
    """Background thread that polls nvidia-smi at a fixed interval.

    Usage:
        sampler = GpuTelemetrySampler(interval_seconds=1.0)
        sampler.start()
        try:
            # ... run benchmark ...
        finally:
            sampler.stop()
        summary = sampler.summary()

    Safe to instantiate even if nvidia-smi is missing — sampler.start() will
    log a warning and produce an empty summary. This makes it easy to dev-test
    the harness on machines without GPUs.
    """

    def __init__(self, interval_seconds: float = 1.0, gpu_index: int = 0):
        self.interval_seconds = interval_seconds
        self.gpu_index = gpu_index
        self.samples: list[GpuSample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu_name = "unknown"
        self._driver_version = "unknown"
        self._nvidia_smi_available = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self._nvidia_smi_available:
            print("[GpuTelemetrySampler] nvidia-smi not found — running in no-op mode.")
            return
        self._capture_gpu_identity()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def summary(self) -> GpuTelemetrySummary:
        return GpuTelemetrySummary.from_samples(
            self.samples,
            gpu_name=self._gpu_name,
            driver_version=self._driver_version,
        )

    def field_samples(self) -> list[GpuSample]:
        """Raw samples, for callers that want to plot the time series."""
        return list(self.samples)

    def _capture_gpu_identity(self) -> None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            if len(parts) >= 2:
                self._gpu_name, self._driver_version = parts[0], parts[1]
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"[GpuTelemetrySampler] failed to capture GPU identity: {e}")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample_start = time.monotonic()
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_index}",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                )
                util_str, mem_str = out.stdout.strip().split(",")
                self.samples.append(
                    GpuSample(
                        timestamp=sample_start,
                        utilization_percent=float(util_str.strip()),
                        memory_used_mib=float(mem_str.strip()),
                    )
                )
            except (subprocess.SubprocessError, ValueError) as e:
                # Drop sample on error; don't kill the thread.
                print(f"[GpuTelemetrySampler] sample failed: {e}")
            # Sleep the remainder of the interval. Use the event as a sleep so
            # stop() can wake us immediately.
            elapsed = time.monotonic() - sample_start
            self._stop_event.wait(timeout=max(0.0, self.interval_seconds - elapsed))


@dataclass
class RequestRecord:
    """Per-request log entry. Aggregated by the harness into final metrics."""

    clip_id: str
    iteration: int
    worker_id: int
    submitted_at: float  # monotonic seconds since cell start
    completed_at: float
    duration_seconds: float
    audio_seconds: float  # length of the input audio (for RTF computation)
    hypothesis: str = ""  # transcript output, "" on failure
    error: str | None = None  # populated on failure; otherwise None

    @property
    def succeeded(self) -> bool:
        return self.error is None
