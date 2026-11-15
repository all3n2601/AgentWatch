# syntax=docker/dockerfile:1.7
# The AgentWatch model pipeline in one image: pull dataset v1, rebuild split-v3, train, calibrate,
# check bias, draw the figures and push to the model registry (MLflow's local registry until the
# team's GCP registry exists). Build from the repository root:
#   make model-pipeline-image    # docker build -f pipelines/model/Dockerfile -t agentwatch-model-pipeline .
#   make model-pipeline-run      # outputs and the MLflow database land in .data/model-pipeline/
# uv with Python 3.13 on Debian bookworm, pinned by digest (ghcr.io/astral-sh/uv:python3.13-bookworm).
FROM ghcr.io/astral-sh/uv@sha256:47965cdc9d53a515f68f78241161c901e70051ce428f12e791bd7fe19f6a631a

# LightGBM needs OpenMP (libgomp1); the Makefile drives every step.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 make \
 && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=180 \
    MPLBACKEND=Agg \
    UV_NO_DEV=1

RUN useradd --create-home --uid 10001 pipeline \
 && install -d -o pipeline -g pipeline /agentwatch
WORKDIR /agentwatch
COPY --chown=pipeline:pipeline . .
USER pipeline

# The locked environments the steps use: the model package (with the data package) and DVC.
RUN uv sync --frozen --no-dev --package agentwatch-model \
 && uv sync --frozen --no-dev --project pipelines/versioning \
 && uv run --frozen --project pipelines/versioning dvc config --local core.no_scm true

ENTRYPOINT ["pipelines/model/run_pipeline.sh"]
