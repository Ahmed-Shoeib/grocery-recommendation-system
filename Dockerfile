# Phase 5 minimal image: installs the primary/production ScaNN VectorIndex
# backend on Linux and runs the test suite / index build script against it.
# This is NOT the full Phase 10 production image (no API/dashboard, no
# multi-stage/hardened build) - it exists solely because ScaNN has no
# Windows wheel, so it can only be built, tested, and run here. See
# docs/data-mapping.md section 10.
FROM python:3.13-slim

WORKDIR /app

# Only what Phase 5 (VectorIndex + config/schema plumbing) needs: no
# torch/tensorflow/sentence-transformers/fastapi/streamlit - those belong
# to later phases' images.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[retrieval,retrieval-scann,dev]"

COPY configs ./configs
COPY tests ./tests
COPY scripts ./scripts

# Selects configs/docker.yaml (retrieval.backend: scann) by default.
ENV RECS_CONFIG_PATH=/app/configs/docker.yaml

# Scoped to the Phase 5 test files, not the whole `tests/` tree - most
# other test files import the ML stack (torch/tensorflow/
# sentence-transformers), deliberately not installed in this minimal
# image (see the pip install line above). Full-suite-in-Docker is a
# Phase 10 concern.
CMD ["pytest", "-q", \
     "tests/test_vector_index.py", \
     "tests/test_vector_index_factory.py", \
     "tests/test_scann_index.py", \
     "tests/test_latency.py", \
     "tests/test_retrieval_metrics.py"]
