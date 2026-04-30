# Next session checklist

Pick up here. Two tasks, ~90 min total, ~$1.50 of GPU spend.

## Task 1: Build and push custom Docker image (~45 min, ~$0.50)

Goal: bake our Python deps INTO a Docker image so future Pods boot ready-to-run instead of needing `pip install` at the start of every session.

### Prerequisite: pick a registry

Either Docker Hub (https://hub.docker.com — sign up if you don't have one) or GitHub Container Registry (GHCR — uses your existing GitHub creds). GHCR is slightly easier since you already have a GitHub account.

For GHCR, you'll need a personal access token with `write:packages` scope: GitHub → Settings → Developer settings → Personal access tokens → Generate new token (classic) → check `write:packages`. Save the token somewhere; you'll use it as the password when `docker login`-ing.

### Build steps (run on a Pod, NOT your Mac)

Resume the existing Pod (or deploy any Pod — even a CPU-only one for $0.10/hr works for the build). SSH in, then:

```bash
cd /workspace/whisper-serving-bench
git pull

# Build the base image (~5-10 min for first build)
docker build -t whisper-bench:base .

# Build the faster-whisper variant on top of the base (~2 min)
docker build -t whisper-bench:faster-whisper -f docker/Dockerfile.faster-whisper .

# Tag for the registry
docker tag whisper-bench:faster-whisper ghcr.io/dyl5051/whisper-bench:faster-whisper

# Login (use GitHub username + the personal access token as password)
echo "$GHCR_PAT" | docker login ghcr.io -u dyl5051 --password-stdin

# Push (~5-10 min for ~6-8 GB image)
docker push ghcr.io/dyl5051/whisper-bench:faster-whisper
```

After push completes, make the image public on GHCR (Profile → Packages → whisper-bench → Package settings → Change visibility → Public). This lets RunPod pull without auth.

### Configure RunPod to use the custom image

When you deploy the next Pod (Task 2), instead of selecting "RunPod PyTorch 2.4" as the template, click "Custom Container Image" and specify:

```
ghcr.io/dyl5051/whisper-bench:faster-whisper
```

RunPod will pull this image on Pod startup and the Pod boots with all our deps already installed.

## Task 2: L4 faster-whisper sweep (~30 min, ~$0.50)

Deploy a fresh Pod with:
- GPU: L4 (~$0.60/hr — cheaper than A100)
- Image: `ghcr.io/dyl5051/whisper-bench:faster-whisper` (from Task 1)
- Region: wherever L4 has availability (probably any US region)
- Network volume: attach the existing one if possible (model cache survives), or accept fresh download (~3 GB, 1 minute on a fast connection)
- Pricing: On-Demand (NOT Spot)

SSH in, then:

```bash
cd /workspace/whisper-serving-bench
# No pip install needed — the custom image has it all baked in!
git pull

export HF_HOME=/workspace/hf_cache
mkdir -p $HF_HOME

# Reuse the manifest from before if the volume was attached, or re-prep:
python scripts/prepare_data.py --eval-set librispeech_streaming_30s --data-root /workspace/data

# Run the sweep — it'll skip the 5 A100 cells (whose JSONs already exist)
# and only run the 5 L4 cells.
python scripts/run_sweep.py configs/sweeps/v1_faster_whisper.yaml \
    --data-root /workspace/data \
    --results-dir /workspace/data/results
```

Expected results: at any given concurrency, L4 will have higher RTF (slower per-request) but lower cost-per-audio-hour (because $0.60/hr vs $1.49/hr). The interesting question is whether the cost savings beat the speed loss — that's the L4 cell-effectiveness story.

## Stop the Pod when done

The moment the sweep summary table prints:

```bash
# Phone alarm reminder!
```

Pods → ⋮ on your Pod → Stop. Set a phone alarm for 30 min from sweep start as a backstop.

## After this session

You'll have 10 cells of valid data: faster-whisper × {A100, L4} × all 5 concurrencies. Complete framework slice for the easiest framework. Plus a working custom-image workflow that makes subsequent sessions 5 min faster each.

Remaining for v1: 4 frameworks × 2 GPUs × 5 concurrencies = 40 cells. ~5-8 future sessions, ~$15-20 of cloud spend.

## Things you still owe yourself

- VSCode Remote-SSH extension setup (10 min one-time)
- Optional: pull JSONs to your Mac via scp (or just run analyze.py on the Pod)
- Optional: refill RunPod credits if balance is below ~$10
