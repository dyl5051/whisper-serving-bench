# Next session checklist (2026-05-05)

Pick up mid-flight. Today's plan: recover A100 per-cell JSONs from the network volume, re-run analyze.py over the now-complete L4+A100 dataset, then start the HF Transformers sweep. ~2 sessions of work; you're about to do step 1.

## Where we are right now

- `inquisitive_coral_bobolink` (volume `b8c835d0j0`, US-MO-1) is **alive and intact**.
- All 10 A100 per-cell JSONs are sitting at `s3://b8c835d0j0/data/results/` — verified by `aws s3 ls`. They were never extracted in session 5; commit `add2024` only added the session notes describing the recovery, not the data itself.
- New RunPod S3 API key generated and saved to `~/.aws/credentials` as profile `runpod`.
- Profile region must be `us-mo-1` (RunPod validates region matches endpoint, unlike AWS S3 proper).
- The two empty placeholder dirs `results/raw/a100-pull/` and `results/raw/a100-pod-pull/` are leftover from a half-started extraction on May 2 — clean these up.

## Step 1: pull A100 JSONs (~5 min, $0)

```bash
cd "/Users/dyl5051/Documents/Claude/Projects/Serving Frameworks"

aws s3 --profile runpod \
  --endpoint-url https://s3api-us-mo-1.runpod.io \
  cp s3://b8c835d0j0/data/results/ results/raw/ \
  --recursive --exclude "*" --include "v1_*_a100_concurrent_*.json"
```

Should drop 10 files into `results/raw/`:
- `v1_fw_a100_concurrent_{1,8,32,64,128}.json` (~90-130 KB each)
- `v1_vllm_a100_concurrent_{1,8,32,64,128}.json` (~62-120 KB each)

Sanity-check: `ls results/raw/v1_*a100*.json | wc -l` should print 10. Spot-check one with `head` and confirm it has the same shape (`cell_config`, `requests`, etc.) as the L4 cells in `results/raw/v1_fw_l4_concurrent_1.json`.

## Step 2: cleanup + commit

```bash
# Obsolete per session-5 notes — replaced by the new per-cell JSONs
git rm results/raw/sweep_v1_faster_whisper_a100.json

# Empty placeholder dirs from the abandoned May 2 pull attempt
rmdir results/raw/a100-pull results/raw/a100-pod-pull 2>/dev/null

git add results/raw/v1_*a100*.json
git status  # confirm only the 10 new JSONs + the deletion
git commit -m "Recover A100 per-cell JSONs (faster-whisper + vLLM) from inquisitive_coral_bobolink

Sweeps were run in session 5 onto the US-MO-1 network volume but never
extracted. Pulled today via the S3 API. Replaces the stale aggregate
sweep_v1_faster_whisper_a100.json which was reconstructed from terminal
output after the original session-3 detail JSONs were lost."
```

## Step 3: re-run analyze.py over the combined L4+A100 dataset

```bash
python scripts/analyze.py \
  --raw-dir results/raw \
  --output-dir results/published/v1_partial
```

(Decide whether to overwrite `results/published/l4_only/` or keep it as a snapshot. My lean: leave `l4_only/` alone as a historical artifact, write fresh output to `v1_partial/` since v1 isn't done yet — once HF lands and we ship v1, that becomes `results/published/v1/`.)

You'll get `v1_summary.csv` and `v1_decision_matrix.md` covering 20 cells: faster-whisper × {L4, A100} × 5 (all healthy) + vLLM × {L4, A100} × 5 (5 cells with broken WER, but the brokenness is itself the headline finding).

## Step 4: HF Transformers sweep prep

This is the meaningful next adapter to bring up. HF Transformers is the **baseline** — without it, you can't credibly claim "faster-whisper is Nx faster than serving Whisper raw" in the writeup.

Existing assets:
- `adapters/hf_transformers.py` — adapter is implemented (~156 lines)
- `configs/cells/example_hf_a100_c8.yaml` — only 1 HF cell config exists

Need to create 9 more cell YAMLs (mirror the structure of `configs/cells/v1_fw_*.yaml`):
- `v1_hf_l4_concurrent_{1,8,32,64,128}.yaml` (5 cells)
- `v1_hf_a100_concurrent_{1,32,64,128}.yaml` (4 cells — A100/c=8 already exists as `example_hf_a100_c8.yaml`; either rename it to fit the v1_* convention or add a sibling)

Then a sweep YAML for each GPU (mirror `configs/sweeps/v1_faster_whisper.yaml`):
- `configs/sweeps/v1_hf_l4.yaml`
- `configs/sweeps/v1_hf_a100.yaml`

**Critical constraint from prior debugging:** the HF adapter must NOT use `transformers.pipeline()` — it's fragile on Whisper specifically (torchcodec/ffmpeg saga from session 1). Use direct `model.generate` + `soundfile` audio loading. The current adapter should already do this; verify before launching the sweep.

## Step 5: smoke-test HF on Pod, then run the sweeps

Spin up a small L4 Pod (~$0.60/hr) with the `inquisitive_coral_bobolink` volume attached. Build the base Dockerfile (no separate framework image needed for HF — it shares the base):

```bash
docker build -t whisper-bench:base .
```

Run a single c=1 cell as a smoke test:

```bash
python scripts/run_cell.py configs/cells/v1_hf_l4_concurrent_1.yaml \
    --data-root /workspace/data \
    --results-dir /workspace/data/results
```

Confirm it produces a healthy results JSON before launching the full sweep. Once it does:

```bash
tmux new -s hf_sweep
python scripts/run_sweep.py configs/sweeps/v1_hf_l4.yaml \
    --data-root /workspace/data \
    --results-dir /workspace/data/results
```

(Caffeinate from the Mac side: `caffeinate -dimsu -t 7200 &`)

After L4 completes, redeploy onto an A100 (any region with availability) and run `v1_hf_a100.yaml`. **S3 the JSONs out before tearing down each Pod** — don't repeat the session-3 mistake.

## Step 6: pull HF results, commit, re-run analyze

```bash
aws s3 --profile runpod \
  --endpoint-url https://s3api-us-mo-1.runpod.io \
  cp s3://b8c835d0j0/data/results/ results/raw/ \
  --recursive --exclude "*" --include "v1_hf_*.json"
```

Commit, re-run analyze.py. You're now at 30/50 cells with three frameworks (HF baseline + faster-whisper + vLLM) on both GPUs.

## Open scope decision after HF lands

v1 was specced as 5 frameworks. We have data for 3 (HF, FW, vLLM). Remaining: Ray Serve and Triton. My current lean (subject to revisit): **Ray Serve in v1, Triton deferred to v1.1 with TensorRT-LLM** because vanilla Triton without TRT-LLM mostly duplicates the HF baseline's story. Make this call after seeing the 3-framework analyze.py output.

## Headline findings (carry into v1 writeup)

- faster-whisper does NOT scale with concurrency on either GPU. CTranslate2 doesn't tensor-batch internally.
- L4 is GPU-saturated at concurrency=1 with faster-whisper.
- L4 wins cost-per-audio-hour, A100 wins per-request latency. Batch → L4, real-time → A100.
- vLLM is 16-26× faster than faster-whisper on identical hardware/model/eval.
- **vLLM hallucinates badly: WER 50-116% on 30-second clips vs faster-whisper's 3-4%.** This reshapes the writeup's spine: throughput is not the only thing that matters, the fastest framework here is unusable in production.

## Operational reminders

- `tmux` for any sweep that takes more than a few minutes.
- `caffeinate -dimsu -t 7200 &` on Mac during long Pod sessions.
- HF cache env vars set BEFORE pip install or model download.
- Container disk minimum 30 GB if vLLM is involved (not relevant for HF, but in case).
- Always check terminal prompt before pasting. `root@<id>#` = Pod, `dyl5051@Dongkeuns-MacBook-Pro` = Mac.
- **STOP, don't TERMINATE** Pods. Terminate destroys the pod-volume.
- Pull results via S3 BEFORE stopping the Pod — there's no second chance if the volume gets reaped.
