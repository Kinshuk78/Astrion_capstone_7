# ── Astrion DQ — Production Dockerfile ────────────────────────────────────────
#
# TEAM NOTE: We use a multi-stage build to keep the final image small.
# Stage 1 (builder): installs all deps including build tools
# Stage 2 (runtime): copies only the installed packages, no build tools
#
# Why Python 3.11? It's the minimum required by pyproject.toml and has
# significantly better performance than 3.10 for data workloads.
#
# Cloud deployment: This image runs on:
#   AWS:   ECS Fargate (task definition → container image)
#   GCP:   Cloud Run (serverless, scales to zero)
#   Azure: Container Apps
#
# For scheduled runs: use the entrypoint CMD with the CLI command.
# For dashboard:      override CMD with "streamlit run dashboard/app.py"

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (layer cache optimisation)
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

# Install into a virtual env for clean copying to stage 2
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    /opt/venv/bin/pip install --no-cache-dir -e .

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY src/ ./src/
COPY data/raw/retail/ ./data/raw/retail/
COPY config/ ./config/
COPY dashboard/ ./dashboard/
# Create output directories (these will be mounted as volumes in production)
RUN mkdir -p outputs data/injected data/processed

# Add venv to PATH
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Health check ──────────────────────────────────────────────────────────────
# Verifies the CLI entry point is reachable. A full triage run is too slow
# for a health check interval; --help exits 0 if the package is importable.
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD astrion-dq --help > /dev/null 2>&1 || exit 1

# ── Default: run full triage on clean data ────────────────────────────────────
# Override at runtime:
#   docker run astrion-dq astrion-dq inject
#   docker run astrion-dq astrion-dq snapshot
#   docker run astrion-dq astrion-dq triage --source injected
#   docker run astrion-dq astrion-dq evaluate
#   docker run astrion-dq astrion-dq report
#   docker run -p 8501:8501 astrion-dq astrion-dq dashboard
CMD ["astrion-dq", "triage", "--source", "clean", "--auto-approve"]

# 8501 — Streamlit dashboard (local / Docker Compose)
# 8000 — FastAPI REST API (local / Docker Compose)
# 10000 — Render.com default web port (used by render.yaml dockerCommand override)
EXPOSE 8501 8000 10000
