# whisper-serving-bench

A reproducible benchmark of serving frameworks for OpenAI Whisper inference. Compares HF Transformers, faster-whisper, vLLM, Ray Serve, and NVIDIA Triton across two GPU classes under concurrent streaming load.

The output is a decision matrix: given a workload (low-latency single-stream, batch throughput, or concurrent streaming) and a budget, which framework + GPU combination wins on cost-per-audio-hour at acceptable p95 latency?

## Status

**v1 (in progress, target ship date: TBD).** One model (whisper-large-v3), one workload pattern (concurrent streaming at 1/8/32/64/128 concurrency), two GPUs (A100, L4), five frameworks. 50 cells.

Planned follow-ups, each its own release + writeup:
- **v1.1** — Triton with TensorRT-LLM compilation (the easy path vs the fast path)
- **v1.2** — T4 and CPU baselines (is a GPU actually worth it for ASR)
- **v2** — long-form audio + variable input lengths (Whisper's chunked decoding is where benchmarks lie)
- **v2.1** — distil-whisper and whisper-large-v3-turbo (the accuracy-vs-cost frontier)

## What this benchmark is and isn't

**Is:** an apples-to-apples measurement of how production-grade serving frameworks handle Whisper inference under realistic concurrent load, with statistical rigor (multi-iteration p50/p95/p99, warmup exclusion, identical text normalization across frameworks, GPU saturation telemetry).

**Isn't:** a model accuracy benchmark (WER is reported, but the model weights are held constant — this isn't about which Whisper variant is most accurate). Not a training benchmark. Not a true-streaming benchmark — we measure chunked-batch concurrent serving, not incremental decoding as audio arrives. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the full set of measurement choices and their justifications.

## Reproduce one cell in under an hour

```bash
# On a rented GPU box with NVIDIA drivers + Docker:
git clone https://github.com/<your-username>/whisper-serving-bench
cd whisper-serving-bench

# Build the image (~10 min first time)
docker build -t whisper-bench .

# Download eval audio (~3 GB, one-time)
docker run --rm -v $(pwd)/data:/app/data whisper-bench \
    python scripts/prepare_data.py

# Run one cell (HF Transformers baseline, concurrency=8)
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    whisper-bench \
    python scripts/run_cell.py configs/cells/example_hf_a100_c8.yaml
```

Results land as JSON in `results/`. Aggregate latency, WER, RTF, and GPU utilization are all in the file.

## Reproduce the full v1 sweep

```bash
# Run all 50 cells. Takes ~6 GPU-hours on an A100 + ~8 GPU-hours on an L4.
# Budget ~$15 of cloud spend at on-demand rates.
python scripts/run_sweep.py configs/sweeps/v1.yaml
```

## Repo structure

```
bench/          # Measurement harness (data, normalize, WER, metrics, load gen, harness loop)
adapters/       # One module per framework, all implementing FrameworkAdapter
configs/        # YAML cell + sweep configs
scripts/        # Entry points: prepare_data, run_cell, run_sweep, analyze
docs/           # METHODOLOGY.md, results writeups
results/        # Output JSONs (gitignored except published artifacts)
```

## Why these frameworks and not others

The five v1 frameworks were chosen because each makes a structurally different bet on serving optimization. HF Transformers is the no-batching baseline. faster-whisper bets on compiled CTranslate2 kernels. vLLM bets on continuous batching + PagedAttention. Ray Serve bets on Python-native ergonomics with explicit batching logic. Triton bets on dynamic batching + a mature production stack.

TGI, TorchServe, and Modal-as-a-framework were considered and excluded for v1: TGI's audio support is less mature than vLLM's, TorchServe is essentially Triton's less-featured alternative, and Modal is a deployment platform rather than a serving framework. Each may appear in a follow-up if there's reader demand.

## License

MIT. See [LICENSE](LICENSE).
