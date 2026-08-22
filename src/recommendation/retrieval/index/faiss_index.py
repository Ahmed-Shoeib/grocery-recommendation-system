"""FAISS `VectorIndex` backend - default, native on Windows.

Uses `IndexHNSWFlat` (approximate nearest-neighbor search over a
navigable small-world graph) wrapped in `IndexIDMap2` so the index stores
real product ids directly rather than requiring a separate row-index <->
product-id mapping to save/reload alongside it. HNSW was chosen over
IVF/IVF-PQ because it needs no training step - `add_with_ids` builds the
graph incrementally exactly like a flat index would, so it is correct
both at today's ~1,200-item catalog and at a much larger future one,
whereas IVF's k-means clustering step is unstable when `nlist` is large
relative to a small catalog. `M`/`efConstruction`/`efSearch` are
config-driven (`RetrievalConfig.faiss_hnsw_*`, see `utils/config.py`),
not hard-coded, so tuning never requires a code change. `efSearch` is
additionally floored at `k * 2` per query in `search()` below so recall
stays adequate as callers request larger pools (see
`retrieval.index.eligibility_filter.EligibilityRestrictedIndex`'s
oversampling), not just whatever the static config value happens to be.

`METRIC_INNER_PRODUCT` must be passed explicitly when constructing the
HNSW index - FAISS defaults to L2, and silently getting that wrong would
turn "nearest by cosine similarity" into "nearest by Euclidean distance,"
a real similarity-semantics bug, not just an approximation-quality one.

Embeddings must already be L2-normalized (Phase 4's Two-Tower output is);
this class does not renormalize them, so normalized inner product here is
exactly cosine similarity.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from recommendation.retrieval.index.base import SearchResult, VectorIndex
from recommendation.utils.config import RetrievalConfig


class FaissVectorIndex(VectorIndex):
    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self._config = config or RetrievalConfig()
        self._index: faiss.IndexIDMap2 | None = None
        # A separate reference to the inner HNSW index, kept because
        # `IndexIDMap2.index` only exposes the generic base `faiss.Index`
        # type (no `.hnsw` attribute) - `faiss.downcast_index` recovers
        # the concrete `IndexHNSWFlat` type after `load()`, mirroring how
        # `build()` already has direct access to it before wrapping.
        self._hnsw: faiss.IndexHNSWFlat | None = None
        self._dim: int | None = None

    def build(self, item_ids: list[int], embeddings: np.ndarray) -> None:
        if len(item_ids) != embeddings.shape[0]:
            raise ValueError(f"item_ids length ({len(item_ids)}) must match embeddings rows ({embeddings.shape[0]})")
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (num_items, dim), got shape {embeddings.shape}")
        if not item_ids:
            raise ValueError("cannot build an index from an empty catalog")

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._dim = embeddings.shape[1]
        ids = np.asarray(item_ids, dtype=np.int64)

        self._hnsw = faiss.IndexHNSWFlat(self._dim, self._config.faiss_hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self._hnsw.hnsw.efConstruction = self._config.faiss_hnsw_ef_construction
        self._index = faiss.IndexIDMap2(self._hnsw)
        self._index.add_with_ids(embeddings, ids)

    def search(self, query_embeddings: np.ndarray, k: int) -> list[SearchResult]:
        if self._index is None:
            raise RuntimeError("index has not been built or loaded")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        if query_embeddings.shape[1] != self._dim:
            raise ValueError(f"query dim ({query_embeddings.shape[1]}) does not match index dim ({self._dim})")

        effective_k = min(k, self.size)
        # Adaptive floor, not just the static config value: a caller
        # asking for a larger k (e.g. EligibilityRestrictedIndex's
        # oversampling) needs a correspondingly wider HNSW candidate list
        # to have a chance of finding that many good matches at all.
        self._hnsw.hnsw.efSearch = max(self._config.faiss_hnsw_ef_search, effective_k * 2)
        scores, ids = self._index.search(query_embeddings, effective_k)

        results = []
        for row_scores, row_ids in zip(scores, ids):
            valid = row_ids != -1  # FAISS pads with -1 if effective_k exceeds matches
            results.append(
                SearchResult(item_ids=row_ids[valid].tolist(), scores=row_scores[valid].tolist())
            )
        return results

    def save(self, path: str | Path) -> None:
        if self._index is None:
            raise RuntimeError("index has not been built or loaded")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no FAISS index found at {path}")
        self._index = faiss.read_index(str(path))
        self._hnsw = faiss.downcast_index(self._index.index)
        self._dim = self._index.d

    @property
    def size(self) -> int:
        return 0 if self._index is None else self._index.ntotal
