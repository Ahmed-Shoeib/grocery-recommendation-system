# syntax=docker/dockerfile:1
#
# Multi-stage production image (Phase 10). ScaNN is the intended
# production retrieval backend here (configs/docker.yaml, loaded via
# RECS_CONFIG_PATH below) - it has no Windows wheel, so this is the only
# place it runs; FAISS remains the native-Windows dev fallback (see
# docs/data-mapping.md section 10).
#
# Stages:
#   base      - shared dependency layer (ml + retrieval + retrieval-scann + serving)
#   test      - adds dev deps + tests/, runs the FULL suite (`docker build --target test`)
#   api       - runs the FastAPI service (default target)
#   dashboard - runs the Streamlit dashboard
#
# `models/` and `data/` are never COPYed in (.dockerignore) and are never
# baked into git - the api/dashboard stages expect them bind-mounted at
# runtime (see docker-compose.yml); train against the `base` stage to
# produce them first. Trained-model reproducibility comes from re-running
# the training scripts against this same image, not from baking weights
# into a layer.
FROM python:3.13-slim-bookworm AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RECS_CONFIG_PATH=/app/configs/docker.yaml

COPY pyproject.toml README.md ./
COPY src ./src
# --extra-index-url restricts torch to its CPU-only wheel (the default
# PyPI wheel pulls in multi-GB CUDA libraries this CPU-only service never
# uses) - the single largest "keep the image reasonably small" lever
# available without changing any dependency version.
#
# tensorflow~=2.20.0 is pinned explicitly (narrower than pyproject's
# ml extra, which allows >=2.16,<2.22 for native-Windows dev where
# ScaNN is never installed): scann==1.4.2's own PyPI metadata declares
# tensorflow~=2.20.0 as its compatible version, and pip otherwise
# resolves the pyproject range's newest allowed release (2.21.0) here,
# which is ABI-incompatible with scann's precompiled ops
# (`undefined symbol: ...absl...internal_log_function...` at import
# time) - discovered during Phase 10 Docker integration verification.
# Windows dev is unaffected (no scann there, so no conflict to pin
# around) and keeps the wider pyproject.toml range.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -e ".[ml,retrieval,retrieval-scann,serving]" "tensorflow~=2.20.0"

COPY configs ./configs
COPY scripts ./scripts

FROM base AS test
RUN pip install --no-cache-dir -e ".[dev]"
COPY tests ./tests
CMD ["pytest", "-q"]

FROM base AS api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request as u,sys; sys.exit(0 if u.urlopen('http://localhost:8000/v1/ready',timeout=3).status==200 else 1)" || exit 1
CMD ["python", "scripts/run_api.py"]

FROM base AS dashboard
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request as u,sys; sys.exit(0 if u.urlopen('http://localhost:8501/_stcore/health',timeout=3).status==200 else 1)" || exit 1
CMD ["python", "scripts/run_dashboard.py"]
