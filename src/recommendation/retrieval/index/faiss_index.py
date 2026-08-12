"""FAISS `VectorIndex` backend - default, native on Windows.

Uses `IndexFlatIP` (exact, brute-force inner-product search) wrapped in
`IndexIDMap2` so the index stores real product ids directly rather than
requiring a separate row-index <-> product-id mapping to save/reload
alongside it. At the current V1 catalog scale (~50 products) brute-force
is both exact and fast enough that an approximate structure (IVF/HNSW)
would only add complexity for no measurable latency benefit; swapping to
one later is a change inside this class, not to the `VectorIndex`
interface.

Embeddings must already be L2-normalized (Phase 4's Two-Tower output is);
this class does not renormalize them, so normalized inner product here is
exactly cosine similarity.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from recommendation.retrieval.index.base import SearchResult, VectorIndex


class FaissVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self._index: faiss.IndexIDMap2 | None = None
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

        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dim))
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
        self._dim = self._index.d

    @property
    def size(self) -> int:
        return 0 if self._index is None else self._index.ntotal
