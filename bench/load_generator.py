"""Concurrent load generator.

Drives N async workers against an adapter, where each worker is a closed-loop
client (pull task → send → wait for response → repeat). This models "N active
users" rather than the open-loop "Poisson arrivals" pattern, which is the right
shape for the concurrent-streaming workload — production ASR usage looks like
some number of in-flight conversations, not Poisson-distributed arrivals.

If we want to add an open-loop generator later (for a "burst" workload pattern),
it can live alongside ConcurrentLoadGenerator with the same RequestRecord output
contract, and the rest of the harness won't care.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass

from bench.data import AudioClip
from bench.metrics import RequestRecord


@dataclass(frozen=True)
class _WorkItem:
    clip: AudioClip
    iteration: int


class TranscribeProtocol:
    """Structural protocol for what the load generator needs from an adapter.

    Adapters don't have to inherit from this — duck typing is fine — but having
    it as a documented protocol clarifies the contract.
    """

    async def transcribe(self, clip: AudioClip) -> str:  # pragma: no cover
        raise NotImplementedError


async def run_concurrent(
    adapter: TranscribeProtocol,
    clips: list[AudioClip],
    iterations: int,
    concurrency: int,
    request_timeout_seconds: float,
    cell_start_monotonic: float,
) -> list[RequestRecord]:
    """Run the eval set `iterations` times under `concurrency` concurrent workers.

    Returns a list of RequestRecords, one per request attempted. Failed requests
    are included with error populated.

    The work queue contains every (clip, iteration) pair shuffled — each worker
    just pulls the next item. This means request ordering across workers is
    deterministic only at the queue level, which is the right thing: it preserves
    apples-to-apples comparison across cells while letting workers race naturally.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if not clips:
        raise ValueError("clips list is empty")

    queue: asyncio.Queue[_WorkItem | None] = asyncio.Queue()
    for iteration in range(iterations):
        for clip in clips:
            await queue.put(_WorkItem(clip=clip, iteration=iteration))
    # Sentinel per worker so they exit cleanly.
    for _ in range(concurrency):
        await queue.put(None)

    records: list[RequestRecord] = []
    records_lock = asyncio.Lock()

    async def worker(worker_id: int) -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            record = await _run_one(
                adapter,
                item,
                worker_id=worker_id,
                request_timeout_seconds=request_timeout_seconds,
                cell_start_monotonic=cell_start_monotonic,
            )
            async with records_lock:
                records.append(record)
            queue.task_done()

    workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]
    await asyncio.gather(*workers)
    return records


async def _run_one(
    adapter: TranscribeProtocol,
    item: _WorkItem,
    worker_id: int,
    request_timeout_seconds: float,
    cell_start_monotonic: float,
) -> RequestRecord:
    submitted_abs = time.monotonic()
    submitted_rel = submitted_abs - cell_start_monotonic

    hypothesis: str = ""
    error: str | None = None

    try:
        hypothesis = await asyncio.wait_for(
            adapter.transcribe(item.clip),
            timeout=request_timeout_seconds,
        )
    except TimeoutError:
        error = f"timeout after {request_timeout_seconds}s"
    except Exception as e:  # noqa: BLE001 — we want to capture every failure
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    completed_abs = time.monotonic()
    return RequestRecord(
        clip_id=item.clip.clip_id,
        iteration=item.iteration,
        worker_id=worker_id,
        submitted_at=submitted_rel,
        completed_at=completed_abs - cell_start_monotonic,
        duration_seconds=completed_abs - submitted_abs,
        audio_seconds=item.clip.duration_seconds,
        hypothesis=hypothesis,
        error=error,
    )


async def warmup(
    adapter: TranscribeProtocol,
    clips: list[AudioClip],
    n_requests: int,
    request_timeout_seconds: float,
) -> int:
    """Issue `n_requests` warm-up requests sequentially. Returns count of successes.

    Cycles through the clips list if n_requests > len(clips).
    Failures during warmup are logged but don't abort — the cell's real run will
    record them properly.
    """
    if n_requests <= 0 or not clips:
        return 0

    successes = 0
    for i in range(n_requests):
        clip = clips[i % len(clips)]
        try:
            await asyncio.wait_for(
                adapter.transcribe(clip),
                timeout=request_timeout_seconds,
            )
            successes += 1
        except Exception as e:  # noqa: BLE001
            print(f"[warmup] request {i + 1}/{n_requests} failed: {type(e).__name__}: {e}")
    return successes
