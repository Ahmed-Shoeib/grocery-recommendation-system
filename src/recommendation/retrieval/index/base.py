"""Backend-agnostic `VectorIndex` interface.

Every backend (FAISS on native Windows dev, ScaNN in Docker/Linux) does
normalized-inner-product search over the L2-normalized 128-D Two-Tower
item embeddings - since both query and item vectors are unit-norm, inner
product and cosine similarity are the same quantity, so `search` returns
cosine similarity scores regardless of backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchResult:
    """One query's ranked candidates: parallel arrays, best match first."""

    item_ids: list[int]
    scores: list[float]


class VectorIndex(ABC):
    """Nearest-neighbor search over L2-normalized item embeddings.

    `build` must be called (or `load`) before `search`. Implementations own
    the id <-> row mapping internally so callers only ever deal in the
    catalog's actual product ids, never raw matrix row indices.
    """

    @abstractmethod
    def build(self, item_ids: list[int], embeddings: np.ndarray) -> None:
        """Construct the index from parallel item ids and their
        L2-normalized embeddings, shape (len(item_ids), dim).
        """

    @abstractmethod
    def search(self, query_embeddings: np.ndarray, k: int) -> list[SearchResult]:
        """Return the top-k nearest items (by cosine similarity, descending)
        for each row of `query_embeddings`, shape (num_queries, dim). `k` is
        capped internally to the index size.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the index to disk so it can be reloaded without
        retraining/re-embedding. Written under the gitignored `models/`
        tree - regenerable from `item_embeddings.npz`, not source.

        `path` is a single file for FAISS, a directory for ScaNN (its
        native serialization format) - backend-specific, callers should
        treat it as an opaque location, not assume a file.
        """

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore a previously `save`d index in place."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of items currently indexed."""
