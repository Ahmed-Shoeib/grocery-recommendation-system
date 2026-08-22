"""Query-time eligibility restriction for `VectorIndex` search results.

Hard pre-retrieval eligibility (isActive/stockQuantity, see
`serving.eligibility`) must stop inactive/out-of-stock products from ever
appearing as retrieval candidates. Neither backend needs to be retrained,
re-embedded, or have its index structure rebuilt for this - `isActive`/
`stockQuantity` are serving-time catalog state, not something the
Two-Tower model or its frozen 128-D item embeddings encode. FAISS and
ScaNN both index the SAME full-catalog embeddings (built once, at service
startup, from the trained Two-Tower - see `api.dependencies
.build_recommendation_service`); this wrapper restricts what a `search()`
call is allowed to return, purely at query time, so eligibility changes
never touch the index at all.

Both backends now do genuine approximate nearest-neighbor search over a
catalog sized to grow well beyond V1's ~1,200 items (see `faiss_index.py`
/`scann_index.py`), so asking for the entire index on every query (the
old behavior) would defeat the whole point of an ANN structure - it
would be O(catalog size) per request regardless of how sublinear the
underlying search actually is. Instead, this wrapper oversamples: it asks
the index for a multiple of `k` (`RetrievalConfig.eligibility_oversample_
factor`, floored at `eligibility_oversample_floor`), filters to
`allowed_ids`, and - only if that wasn't enough - widens the request
(`eligibility_widen_multiplier`, doubling by default) and retries, up to
`eligibility_max_widen_attempts` times. Each attempt is capped at the
index's actual size, so this always terminates in a bounded number of
search calls - it never loops indefinitely. With the default config
values (`configs/base.yaml`/`docker.yaml`) applied to this project's
current pool sizes, the widening sequence happens to reach the whole
index by the final attempt when eligible items are thin and scattered,
matching the old always-ask-for-everything behavior only in that
pathological case, not on every request. That is a property of the
DEFAULT numbers, not a guarantee of this algorithm in general - a more
aggressive (smaller factor/multiplier/attempt-count) configuration can
plateau below full catalog coverage before `eligibility_max_widen_
attempts` is exhausted, deliberately bounding worst-case query cost at
the risk of an honest under-fill in extreme eligible-sparsity cases,
rather than ever falling back to an unbounded full scan. FAISS could
filter natively via
an `IDSelector`, but ScaNN's pybind searcher used here has no per-query
id-filtering hook, so a single backend-agnostic widening loop - not two
different code paths - is what actually treats both backends the same
way; it works entirely through the existing `VectorIndex.search(query,
k)` interface, no backend-specific hook needed.

An inactive/out-of-stock item is NEVER returned, at any oversample width
- every attempt re-filters against `allowed_ids` from scratch. If eligible
items are genuinely scarce (fewer than `k` exist in the whole catalog),
this honestly returns fewer than `k` rather than padding with ineligible
ids or looping forever - see `serving.pipeline.RecommendationResult
.fill_rate`, whose under-fill semantics this preserves unchanged.
"""

from __future__ import annotations

import numpy as np

from recommendation.retrieval.index.base import SearchResult, VectorIndex
from recommendation.utils.config import RetrievalConfig


class EligibilityRestrictedIndex:
    """Wraps a `VectorIndex` so `search` only ever returns ids present in
    a caller-supplied `allowed_ids` set - the pre-retrieval hard-
    eligibility gate (`serving.pipeline._personalized_candidates`).

    Deliberately does NOT implement the `VectorIndex` ABC itself (its
    `search` needs the extra `allowed_ids` argument, and it owns no
    index state of its own - `build`/`save`/`load` would be meaningless
    here) - this is a narrow, request-scoped decorator around an
    already-built index, not a second `VectorIndex` implementation.
    """

    def __init__(self, index: VectorIndex, config: RetrievalConfig | None = None) -> None:
        self._index = index
        self._config = config or RetrievalConfig()

    def search(self, query_embeddings: np.ndarray, k: int, allowed_ids: set[int]) -> list[SearchResult]:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        num_queries = 1 if query_embeddings.ndim == 1 else query_embeddings.shape[0]
        if not allowed_ids:
            return [SearchResult(item_ids=[], scores=[]) for _ in range(num_queries)]

        cfg = self._config
        index_size = self._index.size
        requested = min(max(k * cfg.eligibility_oversample_factor, cfg.eligibility_oversample_floor), index_size)

        filtered: list[SearchResult] = [SearchResult(item_ids=[], scores=[]) for _ in range(num_queries)]
        satisfied = [False] * num_queries

        attempt = 0
        while attempt < cfg.eligibility_max_widen_attempts and not all(satisfied):
            raw_results = self._index.search(query_embeddings, requested)

            for i, result in enumerate(raw_results):
                if satisfied[i]:
                    continue
                ids: list[int] = []
                scores: list[float] = []
                for item_id, score in zip(result.item_ids, result.scores):
                    if item_id in allowed_ids:
                        ids.append(item_id)
                        scores.append(score)
                        if len(ids) == k:
                            break
                filtered[i] = SearchResult(item_ids=ids, scores=scores)
                # Either we found enough, or we've already asked for the
                # whole index and can't do any better by widening further.
                if len(ids) >= k or requested >= index_size:
                    satisfied[i] = True

            if all(satisfied):
                break
            attempt += 1
            requested = min(requested * cfg.eligibility_widen_multiplier, index_size)

        return filtered
