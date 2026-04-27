# Methodology

This document explains every measurement choice in the benchmark and why it was made. If you disagree with a choice, this is the document to point at — both so we can discuss the specific decision and so you can fork the harness with your own choice.

## What we measure

Each "cell" is one (framework, GPU, concurrency) combination. For each cell we record:

| Metric | Definition | Why it matters |
|---|---|---|
| RTF (Real-Time Factor) | wall_clock_seconds / audio_seconds. Lower is faster. | The standard ASR throughput metric. RTF=0.1 means 10× faster than real-time. |
| Latency p50/p95/p99 | Per-request wall-clock from request submission to final transcript returned, measured client-side. | p95 is what bites you in production. Median is fine; tail is what you wake up to. |
| WER | Word Error Rate against reference transcript, after Whisper normalization. | Sanity check that the framework didn't silently break the model. Differences of >0.5% indicate a real problem (tokenization, decoding params), not a framework "win." |
| GPU utilization (mean, max) | Sampled at 1Hz from `nvidia-smi --query-gpu=utilization.gpu` during the cell. | Diagnostic: if mean GPU util < 70% under load, you didn't saturate the GPU and the framework's throughput number understates what it could do. |
| GPU memory (peak) | Sampled at 1Hz from `nvidia-smi --query-gpu=memory.used`. | Tells you how much headroom exists for higher concurrency or larger batches. |
| Cost per audio-hour | Computed from RTF × GPU hourly rate (cloud spot pricing snapshot). | What CTOs actually decide on. Recomputed from rate cards in `configs/pricing.yaml`. |
| Failure mode | Free-text if the cell errored (OOM at high concurrency, framework didn't support config, etc.) | Failures are findings. Recorded, not omitted. |

## How we measure

**Warm-up.** Before the timed run, we issue 10 requests sequentially to warm the model, allocate GPU memory, JIT-compile any ahead-of-time paths, and (for vLLM) populate the KV cache pages. These requests are excluded from all reported numbers.

**Iterations.** Each cell runs the full eval set 3 times. We report median across iterations for the headline number; standard deviation is included so readers can see noise.

**Concurrency.** The load generator launches N async workers, each pulling from a shared work queue of audio files. This produces a closed-loop workload (each worker waits for its previous response before sending the next), which models real production "active user" semantics better than open-loop Poisson arrivals would for this use case.

**Audio.** The streaming workload uses 50 deterministically-sliced 30-second clips from TED-LIUM v3. 30 seconds matches Whisper's encoder window exactly, which removes "did you have to pad?" as a confounding variable between frameworks. The same 50 clips are reused across cells so framework results are directly comparable.

**Text normalization.** All hypothesis and reference transcripts are passed through OpenAI's `EnglishTextNormalizer` (from the `whisper` package) before WER computation. This is critical because frameworks differ in their default decoding (some emit punctuation, some don't; some normalize numbers, some don't), and unnormalized WER comparisons are not meaningful. WER is computed via `jiwer`.

**GPU telemetry.** A background thread samples `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits` at 1Hz throughout the timed run, including during warm-up (so we can see model load behavior) but the warm-up samples are tagged separately in the JSON.

**Failure handling.** If a request errors, the failure is recorded with full traceback and the cell continues. If >10% of requests fail, the cell is marked `degraded` and its aggregate metrics are flagged. If the framework refuses to start at all (e.g. vLLM doesn't support a particular concurrency level), the cell is marked `unsupported` and an entry is still emitted to the results JSON — these are findings.

## What we deliberately don't measure

**True streaming latency.** "Time to first partial transcript" for incrementally-arriving audio is a different problem from concurrent serving of finished audio chunks. Most frameworks don't support true incremental decoding for Whisper. We measure chunked-batch concurrent serving and label it as such. Out of scope; possibly a separate v3 release.

**Cold-start latency including model load.** First-request latency in production matters, but it's almost entirely a function of model weight size and disk I/O, which are not differentiated by the serving framework. We exclude cold-start from headline latency metrics but record it separately for completeness.

**Multi-GPU / multi-replica scaling.** v1 is single-GPU per framework. Multi-replica configurations introduce a routing layer that varies dramatically by framework and would muddy the per-framework comparison. Possibly a future release.

**Framework startup time.** How long it takes to bring up the server from `docker run` to "ready for first request." This is operationally relevant but not what we set out to measure.

## Why these specific frameworks

See [README.md § Why these frameworks](../README.md#why-these-frameworks-and-not-others). Briefly: each makes a structurally different bet on serving optimization, and the comparison is most useful when the bets are most different.

## Reproducibility commitments

- Every dependency in `pyproject.toml` is version-pinned.
- The Docker image tag is captured in every results JSON.
- The git commit SHA is captured in every results JSON.
- The exact GPU model and driver version are captured (from `nvidia-smi --query-gpu=name,driver_version`).
- Eval audio is downloaded from canonical sources (HuggingFace `datasets`) with fixed split + revision.
- Pricing snapshots are dated; RTF and latency numbers do not depend on pricing.

If your reproduction differs from our published numbers by more than the standard deviation we report, please open an issue. If it differs by a lot, we want to know.

## Known limitations of this methodology

- Closed-loop workloads model "N concurrent active users" well, but a true bursty/Poisson production workload would show different tail latency behavior. Our concurrency sweep approximates this by varying N.
- The 30-second uniform clip choice removes one confounding variable (encoder padding) but hides another (variable-length input handling). v2 will address this directly.
- WER on LibriSpeech-clean-derived audio is unrealistically low for production speech (which is messier). WER here is a sanity check, not a model accuracy claim.
- Cloud GPU performance has non-trivial run-to-run variance from neighbor noise on shared infrastructure. Multiple iterations help but don't eliminate this. We report stddev for transparency.
