"""ScaNN `VectorIndex` backend - the primary, production-intended ANN
retrieval engine for this project.

ScaNN has no Windows wheel (confirmed against PyPI - Linux-only manylinux
wheels), so it only runs inside the Docker/Linux image. `scann` is
therefore imported lazily, inside this module only, so that importing
`recommendation.retrieval.index` (and running the native-Windows dev
loop / FAISS tests) never requires it to be installed. `factory.py`
mirrors this: it imports this module only when `retrieval.backend ==
"scann"` is actually selected.

Uses ScaNN's approximate tree-partitioning + asymmetric-hashing (AH)
quantization pipeline, with exact-score reordering of the top candidates
- `.tree(...)` prunes the catalog to a small number of partitions per
query, `.score_ah(...)` scores within those partitions using a quantized
approximation (fast, lossy), and `.reorder(...)` re-scores the surviving
top candidates against the exact float embeddings before returning, so
returned scores are exact dot products for whatever candidates make it
through reordering - not quantized approximations. `num_leaves`/
`num_leaves_to_search` are derived from the catalog size at build time
(`RetrievalConfig.scann_leaves_multiplier` etc., see `utils/config.py`)
rather than hard-coded, so this scales from today's ~1,200-item catalog
to a much larger future one without a code change - mirrors the FAISS
backend's `IndexHNSWFlat` migration for the same reason (see
`faiss_index.py`).

`scann_reorder_k` is fixed at BUILD time (a ScaNN API constraint - unlike
FAISS's `efSearch`, it cannot vary per query) and must stay >= the
largest `k` this deployment will ever pass to `search()`, or callers
requesting a very large pool won't get exact-reorder scores for all of
it - see the field's docstring in `RetrievalConfig`.

Embeddings must already be L2-normalized (the Two-Tower's output is);
`"dot_product"` as the distance measure over unit-norm vectors is then
exactly cosine similarity, and ScaNN returns neighbors sorted by
*decreasing* dot product (most similar first) for this measure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from recommendation.retrieval.index.base import SearchResult, VectorIndex
from recommendation.utils.config import RetrievalConfig

# ScaNN's asymmetric hashing has a hard floor, not a tunable one:
# `.score_ah(..., hash_type="lut16")` (the default, used here) fixes
# `num_clusters_per_block` at 16 - AH training raises a RuntimeError if
# the catalog has fewer than 16 points to train those clusters on
# ("Number of clusters per block (16) is greater than asymmetric hashing
# training data size (N)"), independent of every other config knob in
# this module. This is far below any real production catalog (this
# project's is ~1,200 items) but is a real crash risk for a brand-new/
# tiny catalog, so
# `build()` falls back to `.score_brute_force(quantize=False)` - exact
# search, ScaNN's own pre-migration behavior - below this floor rather
# than let AH training raise. The genuine ANN path (tree + AH + reorder)
# is unconditional at and above this size.
_MIN_CATALOG_SIZE_FOR_ASYMMETRIC_HASHING = 16


class ScannVectorIndex(VectorIndex):
    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self._config = config or RetrievalConfig()
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

        cfg = self._config
        catalog_size = len(item_ids)
        builder = scann.scann_ops_pybind.builder(embeddings, len(item_ids), "dot_product")

        if catalog_size < _MIN_CATALOG_SIZE_FOR_ASYMMETRIC_HASHING:
            # Small-catalog fallback (see module-level constant docstring)
            # - exact search, not a change to the production ANN path.
            self._searcher = builder.score_brute_force(quantize=False).build()
        else:
            # num_leaves is the min of THREE independent caps: the
            # sqrt(catalog_size)-scaling target (`raw_leaves` - the "how
            # many partitions does a catalog this size deserve" signal),
            # a hard ceiling (`scann_max_leaves`), and a density floor
            # (`density_capped_leaves`) that keeps at least
            # `scann_min_points_per_leaf` points per partition on
            # average. That last cap matters more than it looks: without
            # it (an earlier version of this code used an unconditional
            # `scann_min_leaves` FLOOR instead), a mid-size catalog (e.g.
            # 50 items) got padded up to 20 leaves regardless - only ~2.5
            # points/leaf - and ScaNN's kmeans tree partitioner degrades
            # to empty/zero-norm centroids at that point ("Could not
            # normalize centroid due to zero norm or empty or zero-weight
            # partition"), which silently produces NaN scores for items
            # assigned to the broken partition. Confirmed via a real
            # Docker/CI run: this exact failure mode NaN-poisoned ranker
            # training end-to-end (retrieval_score feature -> loss -> all
            # weights). `scann_min_points_per_leaf` fixes that by scaling
            # num_leaves DOWN for small catalogs instead of up.
            raw_leaves = max(1, round(cfg.scann_leaves_multiplier * math.sqrt(catalog_size)))
            density_capped_leaves = max(1, catalog_size // cfg.scann_min_points_per_leaf)
            num_leaves = min(catalog_size, cfg.scann_max_leaves, raw_leaves, density_capped_leaves)
            num_leaves_to_search = max(1, round(num_leaves * cfg.scann_leaves_to_search_fraction))
            training_sample_size = min(catalog_size, cfg.scann_training_sample_size)
            reorder_k = min(catalog_size, cfg.scann_reorder_k)

            self._searcher = (
                builder
                .tree(num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search, training_sample_size=training_sample_size)
                .score_ah(dimensions_per_block=cfg.scann_ah_dims_per_block, anisotropic_quantization_threshold=cfg.scann_aq_threshold)
                .reorder(reorder_k)
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
            # ScaNN's tree partitioning only visits `num_leaves_to_search`
            # of `num_leaves` partitions (fixed at build time) - if a
            # query asks for more neighbors than that visited subset
            # actually contains (most likely at large k, e.g. k close to
            # or equal to the full catalog size), it pads the tail of
            # `scores` with NaN rather than shrinking the result, with
            # `row_neighbors` for those slots not a genuine match either
            # (confirmed empirically: NaN scores + otherwise-valid-looking
            # ids, both only in the low-relevance tail). FAISS signals the
            # equivalent "fewer matches than k" case with an int id of -1
            # (filtered above via `faiss_index.py`'s `row_ids != -1`) -
            # ScaNN has no id-level sentinel, so the score's NaN is the
            # only signal; drop those slots the same way, or a phantom
            # entry with a fabricated score/rank can leak into a
            # downstream feature (this NaN-poisoned ranker training
            # end-to-end before this filter was added - confirmed via a
            # real Docker/CI run, not a hypothetical).
            valid = ~np.isnan(row_scores)
            item_ids = [self._item_ids[i] for i, ok in zip(row_neighbors, valid) if ok]
            valid_scores = [float(s) for s, ok in zip(row_scores, valid) if ok]
            results.append(SearchResult(item_ids=item_ids, scores=valid_scores))
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
