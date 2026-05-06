# Next session checklist (2026-05-06)

Pick up mid-flight. Today (May 5) we brought the HF baseline up: shipped the L4 sweep cleanly, hit a noisy-neighbor A100 host on the first attempt, quarantined the poisoned smoke-test data, and stopped before publishing nonsense numbers. Tomorrow: redeploy A100 on a different host, run the sweep, get to 30/50 cells.

## Where we are right now

**Done in session 6:**
- HF Transformers adapter is functional. Two non-trivial fixes landed (committed in `892bcc1`):
  - **Long-form generation** via `return_timestamps=True` + `truncation=False` + `return_attention_mask=True`. Without this, clips >30s were silently truncated, inflating WER to ~32%. Eval set averages 35.5s, max 46s.
  - **Concurrency lock** (`threading.Lock`) around `model.generate()`. PyTorch's generate isn't thread-safe; concurrent calls crashed the GPU with device-side asserts at c≥8. Lock-serializing produces the realistic naive-baseline behavior (N submissions queue through one GPU). That serialization is exactly the value gap batched/replicated frameworks fill.
- HF L4 sweep complete (5/5 cells, all healthy):
  - WER: **1.44% across all concurrencies** (better than faster-whisper's 3-4%).
  - RTF: **identical at ~0.044 across all concurrencies** — concurrency buys nothing because the lock serializes everything.
  - p95 latency scales linearly: **1.96s @ c=1 → 209s @ c=128**.
  - GPU saturated at ~92% from c=1.
- Repo + adapter + 12 YAMLs + methodology update committed and pushed to `origin/main` (commit `892bcc1`).
- Result JSONs in `results/raw/` (Mac local, gitignored) and at `s3://b8c835d0j0/data/results/` for durability.
- New volume conveniences (saves time on every future pod redeploy):
  - `/workspace/init_pod.sh` — one-shot bootstrap: SSH keys + apt + pip + transformers pin + HF_HOME persistence.
  - `/workspace/.ssh/authorized_keys` — persistent copy of pod's authorized_keys so future pods skip the curl-keys dance.

**Stopped here because:**
- First A100 pod (deploy 2026-05-05, IP 64.247.196.124) landed on a noisy host. Symptoms: load avg 10+, `nvidia-smi` queries timing out >2s, our python at 448% CPU but GPU only 31% util, c=1 took 532s vs L4's 275s. We were CPU-starved by neighbors.
- Result is quarantined as `s3://b8c835d0j0/data/results/v1_hf_a100_concurrent_1.POISONED-noisy-host.json` (and the corresponding log at `/workspace/smoke_a100.POISONED-noisy-host.log`). Do not analyze it; it's kept only for post-mortem.

## Step 1: deploy a fresh A100 (~5 min, ~$0.05)

1. RunPod console → deploy A100 SXM4-80GB in **US-MO-1** with `b8c835d0j0` attached. Container disk ≥30GB. Hopefully a different physical host than the May-5 deploy.
2. Open the pod's web terminal. Run **one** command:
   ```bash
   bash /workspace/init_pod.sh
   ```
   This restores SSH authorized_keys, installs apt + pip deps, pins `transformers==4.46.3`, persists `HF_HOME=/workspace/data/hf_cache`. ~3-5 min.
3. Paste the new SSH command into Claude. SSH from Mac will work because `init_pod.sh` repopulates `/root/.ssh/authorized_keys` from `/workspace/.ssh/authorized_keys`.

## Step 2: smoke-test SANITY CHECK (~3 min)

This is new and important — the noisy-host episode is a real risk. Before launching the full sweep, run the c=1 smoke test and **sanity-check the wall time**:

```bash
export HF_HOME=/workspace/data/hf_cache
cd /workspace/whisper-serving-bench
git pull   # pick up any session-7 commits
python scripts/run_cell.py configs/cells/v1_hf_a100_c1.yaml \
    --data-root /workspace/data \
    --results-dir /workspace/data/results
```

**Expected wall time on a healthy A100 host: ~120-180s** (L4 was 275s; A100 should be ~1.5-2× faster per request).

- ≤200s: healthy host, proceed to sweep.
- 200-300s: marginal, retry once before giving up.
- ≥300s: noisy host. **Stop, redeploy, try again.** Do not run the full sweep on a poisoned host — the 22 minutes of nonsense numbers will need to be discarded.

Also watch `dmesg`-level signals: `[GpuTelemetrySampler] sample failed` lines in the harness output mean nvidia-smi is timing out, which means PCIe/driver contention.

## Step 3: full A100 sweep (~12-18 min on a healthy host)

```bash
tmux new -s hf_a100
python scripts/run_sweep.py configs/sweeps/v1_hf_a100.yaml \
    --data-root /workspace/data \
    --results-dir /workspace/data/results \
    --force   # overwrites the c=1 smoke-test JSON for consistency with the lock-aware adapter
```

(Caffeinate from the Mac side: `caffeinate -dimsu -t 7200 &`)

5 cells, expected ~140-180s each on A100 (vs L4's 273s) since lock-serialization means the headline doesn't change but each request is faster.

## Step 4: pull, S3, commit, push

The pod writes to `/workspace/data/results/`, which is auto-visible at `s3://b8c835d0j0/data/results/` (the volume IS the S3 bucket — no separate push needed).

From Mac:
```bash
cd "/Users/dyl5051/Documents/Claude/Projects/Serving Frameworks"
aws s3 --profile runpod \
  --endpoint-url https://s3api-us-mo-1.runpod.io \
  cp s3://b8c835d0j0/data/results/ results/raw/ \
  --recursive --exclude "*" --include "v1_hf_a100_*.json" --include "sweep_v1_hf_a100*.json"
```

Commit anything new (probably just `NEXT_SESSION.md` and any session notes), push to GitHub, stop the A100 pod.

## Step 5: analyze.py over the now-30-cell dataset

```bash
python scripts/analyze.py \
    --raw-dir results/raw \
    --output-dir results/published/v1_partial
```

That gets you the three-framework decision matrix (HF + FW + vLLM × {L4, A100} × 5 concurrencies = 30 cells). v1.1+ remaining: Ray Serve and/or Triton. Ray Serve is the meaningful next adapter; Triton vanilla mostly duplicates the HF baseline so defer that to v1.2 with TensorRT-LLM (already speccd).

## Headline findings to weave into the v1 writeup once A100 lands

- **HF baseline doesn't scale with concurrency** because PyTorch's generate isn't thread-safe; we expose the realistic single-model bottleneck via a lock. The serialization itself IS the value-prop story: throughput is identical at c=1 and c=128, p95 latency degrades linearly.
- **faster-whisper does NOT scale with concurrency either**, but for a different reason — CT2 doesn't tensor-batch internally. Same headline RTF across concurrencies.
- **vLLM is 16-26× faster than faster-whisper** on identical hardware/model/eval, but **WER 50-116% on 30s clips** (vs faster-whisper's 3-4%, HF baseline's 1.4%). The fastest framework here is unusable in production. This reshapes the writeup spine: throughput is not the only thing that matters.
- **L4 wins cost-per-audio-hour for batch workloads. A100 wins per-request latency.** Decision matrix: batch → L4, real-time → A100.

## Operational reminders

- `tmux` for any sweep that takes more than a few minutes.
- `caffeinate -dimsu -t 7200 &` on Mac during long Pod sessions.
- `bash /workspace/init_pod.sh` is the single bootstrap on every new pod that mounts `b8c835d0j0`. Don't repeat the per-step manual setup from session 6.
- Always check terminal prompt before pasting. `root@<id>#` = Pod, `dyl5051@Dongkeuns-MacBook-Pro` = Mac.
- **STOP, don't TERMINATE** Pods. Both preserve `b8c835d0j0`, but stop is reversible.
- The "no volume configured" warning on stop is misleading — RunPod is referring to the **container disk**, not the network volume. `/workspace` (mfs#us-mo-1.runpod.net) survives both stop AND terminate.
- Smoke-test sanity check before any full sweep. Noisy hosts are real and will silently poison your data.
- Pull results via S3 BEFORE stopping the Pod (paranoia, not strictly necessary — the volume's S3 endpoint stays accessible after pod stop).
