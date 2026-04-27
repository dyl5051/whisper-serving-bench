# Base image for the benchmark harness + HF Transformers adapter.
# Framework-specific images (vLLM, Triton, Ray Serve) live in docker/ and extend this.
#
# Why pytorch/pytorch as the base: it pre-installs a CUDA + cuDNN + PyTorch combo that
# is mutually compatible. Building this from nvidia/cuda + pip-installing torch yourself
# is a recipe for "torch.cuda.is_available() returns False" debugging at 1am.

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# System deps for audio I/O. ffmpeg is required by openai-whisper for any non-WAV
# audio. libsndfile is required by soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first, before copying source. This lets Docker cache the
# slow pip install layer across rebuilds when only source changes.
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY LICENSE /app/LICENSE

# Install with the [hf] extra. Other framework images override this.
# Using --no-cache-dir to keep the image small.
RUN pip install --no-cache-dir -e ".[hf,dev]"

# Now copy source. Changes here only invalidate the source layer, not the deps layer.
COPY bench /app/bench
COPY adapters /app/adapters
COPY scripts /app/scripts
COPY configs /app/configs

# Capture the build commit SHA for results provenance.
ARG GIT_SHA=unknown
ENV BENCH_GIT_SHA=${GIT_SHA}

# Default entrypoint is bash so users can `docker run -it` and explore. Override
# with `python scripts/run_cell.py ...` for one-shot cell runs.
CMD ["/bin/bash"]
