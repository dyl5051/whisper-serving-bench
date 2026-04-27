# CLAUDE.md

Context for Claude (or any agent) picking up this project. Read this first.

## What this project is

A reproducible benchmark of serving frameworks (HF Transformers, faster-whisper, vLLM, Ray Serve, NVIDIA Triton) for OpenAI Whisper inference. The deliverable is a public GitHub repo + a long-form writeup with a decision matrix for "which framework + GPU + workload combination wins on cost-per-audio-hour at acceptable p95 latency."

**The goal is portfolio credibility for infra/MLOps hiring**, not building a product. Every design choice optimizes for "would this artifact survive a critical Twitter thread from someone in the field." Statistical rigor and reproducibility are non-negotiable; scope cuts are.

## Strategy: ship narrow v1, then iterate publicly

We deliberately do NOT try to build the full benchmark matrix in private over 6 weeks and drop it as a finished artifact. Instead:

- **v1 (target ~2 weeks):** one workload (concurrent streaming at 1/8/32/64/128), one model (whisper-large-v3), two GPUs (A100, L4), all five frameworks. 50 cells. Ship it.
- **v1.1:** Triton with TensorRT-LLM compilation
- **v1.2:** T4 + CPU baselines
- **v2:** long-form audio + variable input lengths
- **v2.1:** distil-whisper + whisper-large-v3-turbo

Each release is its own GitHub release + writeup. Public iteration compounds attention better than a single big drop, and the deadlines force shipping discipline.

## Architecture map

```
bench/              — framework-agnostic measurement harness
  config.py         — CellConfig (Pydantic) — the unit of work
  data.py           — EvalSet loading from manifest.jsonl
  normalize.py      — Whisper EnglishTextNormalizer wrapper (singleton)
  wer.py            — WER computation via jiwer, normalization always applied
  metrics.py        — LatencySummary, GpuTelemetrySampler (background nvidia-smi @ 1Hz), RequestRecord
  load_generator.py — async closed-loop concurrent runner + sequential warmup
  results.py        — versioned ResultsJson schema + build_results() aggregator
  harness.py        — run_cell(config, adapter): the top-level orchestrator

adapters/           — one module per framework, all implement FrameworkAdapter
  base.py           — ABC + build_adapter() factory dispatching on Framework enum
  hf_transformers.py — baseline (no batching, no serving optimizations)
  faster_whisper_adapter.py — TODO
  vllm_adapter.py   — TODO
  ray_serve_adapter.py — TODO
  triton_adapter.py — TODO

scripts/
  prepare_data.py   — download + materialize eval sets from HuggingFace
  run_cell.py       — entry point: YAML cell config → results JSON
  run_sweep.py      — TODO (orchestrates many cells)
  analyze.py        — TODO (results JSONs → tables + decision matrix)

configs/
  eval_sets.yaml    — catalog of available eval sets (consumed by prepare_data.py)
  cells/            — one YAML per (framework, gpu, concurrency) combination
  sweeps/           — TODO: groups of cells to run together

docs/
  METHODOLOGY.md    — every measurement choice + its justification. Update this BEFORE adding new metrics.

results/
  raw/              — per-cell JSON output (gitignored)
  published/        — committed results that back the writeups
```

## The contract that everything depends on

`FrameworkAdapter.transcribe(clip: AudioClip) -> str` is the only thing the harness needs from any framework. Everything else (batching, server lifecycle, transport, etc.) is the adapter's internal business. Adapters should NOT try to influence the harness's measurement choices — if the harness is measuring something the adapter thinks is unfair, the answer is to discuss it in METHODOLOGY.md, not to modify the harness silently.

## Conventions that are non-obvious from the code

- **WER is always post-normalization.** Never expose a "raw WER" function. Pre-normalization text is preserved per-request for inspection. Different frameworks emit different default punctuation/casing — unnormalized WER comparisons are not meaningful.
- **Failures are findings, not omissions.** OOM at high concurrency, framework refusing a config, etc. — record them as `unsupported`/`error`/`degraded` outcomes in the results JSON. Never silently drop a cell.
- **Per-framework Docker images, not one big image.** vLLM and Triton fight over PyTorch/CUDA versions. Don't try to unify. The base `Dockerfile` ships only the harness + HF baseline; framework images live in `docker/` and extend it (TODO for the non-HF frameworks).
- **Results JSON is versioned.** Schema version is in `bench.results.RESULTS_SCHEMA_VERSION`. Any breaking change requires bumping it AND writing migration code in `analyze.py` to read old versions.
- **Provenance is captured automatically:** git SHA, Docker tag, GPU name + driver version, hostname, Python version. Don't add cells that fudge any of this.
- **Reproducibility commitments are public.** README claims a stranger can reproduce one cell in under an hour. Don't break that — if you add a step, add the docs to keep it true.

## Common commands

```bash
# Local dev (no GPU): syntax check
python -m compileall bench adapters scripts

# Build the base image (on a GPU box)
docker build -t whisper-bench .

# Materialize eval data (one-time, ~3GB)
docker run --rm -v $(pwd)/data:/app/data whisper-bench \
    python scripts/prepare_data.py --eval-set librispeech_test_clean_subset

# Run the smoke-test cell
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    whisper-bench \
    python scripts/run_cell.py configs/cells/example_hf_a100_c8.yaml
```

## Things to NOT do

- Don't vibe-code shortcuts in the harness. The whole project's credibility hinges on the harness being defensible. If you find yourself thinking "this is good enough," stop and think again.
- Don't add metrics without updating METHODOLOGY.md first. The doc is the contract.
- Don't unify the per-framework Docker images.
- Don't skip text normalization to "make WER look better" for any framework.
- Don't cut iterations or warmup to make a cell run faster — single-run noise is real and the rigor matters.
- Don't add a framework adapter without also adding a corresponding entry in the v1 sweep config (or explicitly deferring it to a later release with a comment).

## Open decisions as of 2026-04-27

1. **Cloud provider** — RunPod recommended; not yet locked. Decision needed before first GPU rental.
2. **Where the writeup lives** — personal blog, Substack, or `writeups/` in the repo published via GitHub Pages.
3. **Public commitment to follow-up cadence** — "v1.1 next Tuesday" creates pressure (good for momentum, bad if it slips).
4. **Repo name** — currently `whisper-serving-bench` as a placeholder.

## Workflow assumptions

- Dev happens locally on a Mac (no CUDA). Code is edited in your editor of choice; syntax + YAML can be validated locally without GPU.
- Real benchmark runs happen inside the Docker image on rented cloud GPUs (RunPod planned).
- Results JSONs come back to the local repo via git or `scp` and get committed under `results/published/` for the runs that back the writeups.
- Writeup drafting happens locally in markdown.

## When in doubt

- Read [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for measurement decisions.
- Read [README.md](README.md) for the project narrative + reproduction story.
- Re-read this file (CLAUDE.md) for conventions and the strategic frame.
