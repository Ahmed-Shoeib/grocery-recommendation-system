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

Both backends here do EXACT (brute-force) search over a small (~50 item)
V1 catalog (see `faiss_index.py`/`scann_index.py`), so there's no
meaningful cost difference between asking for `k` neighbors and asking
for all of them. This wrapper always asks the underlying index for
everything it has (`index.size`), then filters to `allowed_ids` and
truncates to `k`. That keeps FAISS and ScaNN behaving identically (both
exact, both filtered the same way in Python) rather than diverging -
FAISS could do this natively via an `IDSelector`, but ScaNN's brute-force
pybind searcher used here has no per-query id-filtering hook, so a single
backend-agnostic post-filter - not two different code paths - is the
simplest abstraction that actually treats both backends the same way and
is "the simplest technically sound abstraction around the VectorIndex"
called for when a backend doesn't support native dynamic metadata
filtering.

This is a deliberate exact-search-only tradeoff: it costs O(catalog size)
per query rather than the sublinear cost a true approximate/partitioned
index with native filtering could offer. If either backend switches to
an approximate structure later (see both backends' own docstrings), this
wrapper's "ask for everything" strategy would need to become an adaptive
oversampling strategy instead - noted here, not solved now, since V1's
catalog scale makes it a non-issue.
"""

from __future__ import annotations

import numpy as np

from recommendation.retrieval.index.base import SearchResult, VectorIndex


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

    def __init__(self, index: VectorIndex) -> None:
        self._index = index

    def search(self, query_embeddings: np.ndarray, k: int, allowed_ids: set[int]) -> list[SearchResult]:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        num_queries = 1 if query_embeddings.ndim == 1 else query_embeddings.shape[0]
        if not allowed_ids:
            return [SearchResult(item_ids=[], scores=[]) for _ in range(num_queries)]

        raw_results = self._index.search(query_embeddings, self._index.size)

        filtered: list[SearchResult] = []
        for result in raw_results:
            ids: list[int] = []
            scores: list[float] = []
            for item_id, score in zip(result.item_ids, result.scores):
                if item_id in allowed_ids:
                    ids.append(item_id)
                    scores.append(score)
                    if len(ids) == k:
                        break
            filtered.append(SearchResult(item_ids=ids, scores=scores))
        return filtered
