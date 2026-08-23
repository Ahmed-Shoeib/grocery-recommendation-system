"""Backend selection and candidate-pool sizing for `VectorIndex`.

Driven entirely by `configs/base.yaml: retrieval.*` (see
`recommendation.utils.config.RetrievalConfig`) so switching backends or
retuning pool size never touches calling code.
"""

from __future__ import annotations

from recommendation.retrieval.index.base import VectorIndex
from recommendation.retrieval.index.faiss_index import FaissVectorIndex
from recommendation.utils.config import RetrievalConfig


def build_vector_index(config: RetrievalConfig) -> VectorIndex:
    """Instantiate the `VectorIndex` backend selected by `config.backend`.

    `"scann"` is the primary, production-intended backend - it has no
    Windows wheel, so it only runs inside the Docker/Linux image
    (`configs/docker.yaml` selects it) and is imported lazily here so that
    selecting `"faiss"` never requires it to be installed. `"faiss"` is
    the native-Windows development fallback - fast local iteration without
    Docker, same cosine-equivalent normalized-inner-product semantics as
    ScaNN. Both backends now do genuine approximate nearest-neighbor
    search (FAISS: HNSW; ScaNN: tree + asymmetric hashing + reorder) -
    `config` is passed straight through to the backend constructor so
    each backend reads its own ANN tuning fields (`RetrievalConfig.
    faiss_hnsw_*` / `scann_*`, see `utils/config.py`).
    """
    if config.backend == "scann":
        from recommendation.retrieval.index.scann_index import ScannVectorIndex

        return ScannVectorIndex(config)
    if config.backend == "faiss":
        return FaissVectorIndex(config)
    raise ValueError(f"unknown retrieval backend: {config.backend!r}")


def candidate_pool_size(config: RetrievalConfig, limit: int, catalog_size: int) -> int:
    """How many candidates the `VectorIndex` should retrieve for a request
    that ultimately wants `limit` final recommendations.

    Oversized on purpose - the ranker and re-ranking stages need a larger
    pool than the final Top-N to have anything meaningful to rank and
    filter. `candidate_pool_multiplier` sets the oversampling factor,
    `min_candidate_pool` is a floor for very small `limit` values, and the
    result is always capped at `catalog_size` since retrieving more than
    the whole catalog is meaningless.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if catalog_size <= 0:
        raise ValueError(f"catalog_size must be positive, got {catalog_size}")

    pool = max(limit * config.candidate_pool_multiplier, config.min_candidate_pool)
    return min(pool, catalog_size)
