"""The top-level cell runner.

run_cell(config, adapter) is the one function that ties everything together:

    1. Load the eval set named in config.workload.eval_set
    2. Issue warmup_requests sequential warm-up calls (untimed, untracked)
    3. Start GPU telemetry sampler
    4. Run the load generator at the configured concurrency for `iterations` passes
    5. Stop GPU telemetry sampler
    6. Build a ResultsJson from the per-request log + telemetry
    7. Compute WER on successful requests, attach it
    8. Return the ResultsJson for the caller to write to disk

The adapter is responsible for being "ready" by the time it's passed in (model
loaded, server warm, etc.). The harness only touches it via .transcribe().
"""

from __future__ import annotations

import time
from pathlib import Path

from bench.config import CellConfig
from bench.data import load_eval_set
from bench.load_generator import TranscribeProtocol, run_concurrent, warmup
from bench.metrics import GpuTelemetrySampler
from bench.results import ResultsJson, attach_wer, build_results
from bench.wer import compute_wer


async def run_cell(
    config: CellConfig,
    adapter: TranscribeProtocol,
    data_root: Path = Path("data"),
) -> ResultsJson:
    """Run one benchmark cell against an already-initialized adapter.

    The adapter must be fully ready (model loaded, server reachable). The harness
    handles warm-up requests but not framework startup — that's the adapter's job.
    """
    eval_set = load_eval_set(config.workload.eval_set, data_root=data_root)
    print(
        f"[harness] cell={config.cell_id} framework={config.framework.value} "
        f"eval_set={eval_set.name} clips={len(eval_set)} "
        f"audio={eval_set.total_audio_seconds:.1f}s "
        f"concurrency={config.workload.concurrency} iterations={config.workload.iterations}"
    )

    # 1. Warmup
    if config.workload.warmup_requests > 0:
        print(f"[harness] warming up: {config.workload.warmup_requests} sequential requests")
        successes = await warmup(
            adapter,
            eval_set.clips,
            n_requests=config.workload.warmup_requests,
            request_timeout_seconds=config.workload.request_timeout_seconds,
        )
        print(f"[harness] warmup: {successes}/{config.workload.warmup_requests} succeeded")

    # 2. Start GPU sampler + timed run
    sampler = GpuTelemetrySampler(interval_seconds=1.0)
    sampler.start()
    cell_start = time.monotonic()
    try:
        records = await run_concurrent(
            adapter,
            eval_set.clips,
            iterations=config.workload.iterations,
            concurrency=config.workload.concurrency,
            request_timeout_seconds=config.workload.request_timeout_seconds,
            cell_start_monotonic=cell_start,
        )
    finally:
        sampler.stop()
    cell_wall = time.monotonic() - cell_start

    print(
        f"[harness] cell complete: {len(records)} requests in {cell_wall:.1f}s wall, "
        f"{sum(1 for r in records if r.succeeded)} successes, "
        f"{sum(1 for r in records if not r.succeeded)} failures"
    )

    # 3. Aggregate
    results = build_results(
        config=config,
        records=records,
        gpu_summary=sampler.summary(),
        cell_wall_seconds=cell_wall,
    )

    # 4. WER (computed over successful requests, joining hypotheses with the
    # eval-set references by clip_id).
    if results.outcome.status in ("ok", "degraded"):
        ref_by_clip = {c.clip_id: c.reference_text for c in eval_set.clips}
        refs: list[str] = []
        hyps: list[str] = []
        for r in records:
            if not r.succeeded:
                continue
            ref = ref_by_clip.get(r.clip_id)
            if ref is None:
                continue
            refs.append(ref)
            hyps.append(r.hypothesis)
        if refs:
            try:
                wer = compute_wer(refs, hyps)
                attach_wer(results, wer)
                print(f"[harness] WER: {wer.wer:.4f} over {wer.reference_word_count} reference words")
            except ValueError as e:
                print(f"[harness] WER computation failed: {e}")

    return results
