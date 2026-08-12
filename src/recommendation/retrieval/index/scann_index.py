"""ScaNN `VectorIndex` backend - the primary, production-intended ANN
retrieval engine for this project (Phase 5).

ScaNN has no Windows wheel (confirmed against PyPI - Linux-only manylinux
wheels), so it only runs inside the Docker/Linux image. `scann` is
therefore imported lazily, inside this module only, so that importing
`recommendation.retrieval.index` (and running the native-Windows dev
loop / FAISS tests) never requires it to be installed. `factory.py`
mirrors this: it imports this module only when `retrieval.backend ==
"scann"` is actually selected.

Uses `score_brute_force(quantize=False)` - exact, unquantized dot-product
search, no partitioning/quantization - rather than ScaNN's approximate
tree + asymmetric-hashing (AH) configuration. At the current V1 catalog
scale (~50 products) exact search is both correct and fast; this mirrors
the FAISS backend's use of `IndexFlatIP` for the same reason (see
`faiss_index.py`). Switching to `.tree(...).score_ah(...)` for a larger
catalog is a change inside this class, not to the `VectorIndex`
interface.

Embeddings must already be L2-normalized (Phase 4's Two-Tower output is);
`"dot_product"` as the distance measure over unit-norm vectors is then
exactly cosine similarity, and ScaNN returns neighbors sorted by
*decreasing* dot product (most similar first) for this measure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from recommendation.retrieval.index.base import SearchResult, VectorIndex


class ScannVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self._searcher = None
        self._item_ids: list[int] | None = None
        self._dim: int | None = None

    def build(self, item_ids: list[int], embeddings: np.ndarray) -> None:
        if len(item_ids) != embeddings.shape[0]:
            raise ValueError(f"item_ids length ({len(item_ids)}) must match embeddings rows ({embeddings.shape[0]})")
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D (num_items, dim), got shape {embeddings.shape}")
        if not item_ids:
            raise ValueError("cannot build an index from an empty catalog")

        import scann

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._dim = embeddings.shape[1]
        self._item_ids = list(item_ids)

        self._searcher = (
            scann.scann_ops_pybind.builder(embeddings, len(item_ids), "dot_product")
            .score_brute_force(quantize=False)
            .build()
        )

    def search(self, query_embeddings: np.ndarray, k: int) -> list[SearchResult]:
        if self._searcher is None:
            raise RuntimeError("index has not been built or loaded")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        if query_embeddings.shape[1] != self._dim:
            raise ValueError(f"query dim ({query_embeddings.shape[1]}) does not match index dim ({self._dim})")

        effective_k = min(k, self.size)
        neighbor_rows, scores = self._searcher.search_batched(query_embeddings, final_num_neighbors=effective_k)

        results = []
        for row_neighbors, row_scores in zip(neighbor_rows, scores):
            item_ids = [self._item_ids[i] for i in row_neighbors]
            results.append(SearchResult(item_ids=item_ids, scores=[float(s) for s in row_scores]))
        return results

    def save(self, path: str | Path) -> None:
        if self._searcher is None:
            raise RuntimeError("index has not been built or loaded")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._searcher.serialize(str(path))
        (path / "vector_index_metadata.json").write_text(
            json.dumps({"item_ids": self._item_ids, "dim": self._dim}), encoding="utf-8"
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no ScaNN index found at {path}")

        import scann

        self._searcher = scann.scann_ops_pybind.load_searcher(str(path))
        metadata = json.loads((path / "vector_index_metadata.json").read_text(encoding="utf-8"))
        self._item_ids = metadata["item_ids"]
        self._dim = metadata["dim"]

    @property
    def size(self) -> int:
        return 0 if self._item_ids is None else len(self._item_ids)
