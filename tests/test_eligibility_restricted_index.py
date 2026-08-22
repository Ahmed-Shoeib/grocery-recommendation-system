"""Tests for `EligibilityRestrictedIndex` (Phase 11, oversampling redesign
2026-08-22) - the backend-agnostic query-time wrapper that restricts
`VectorIndex.search` results to a caller-supplied eligible-id set via
bounded oversampling + progressive widening, never a full-index scan on
the normal path. FAISS-backed tests run everywhere; the ScaNN-backed
tests are skipped (not failed) on native Windows via `importorskip`,
same convention as `test_scann_index.py`, and run for real in the
Linux/Docker image.
"""

from __future__ import annotations

import numpy as np
import pytest

from recommendation.retrieval.index.base import SearchResult, VectorIndex
from recommendation.retrieval.index.eligibility_filter import EligibilityRestrictedIndex
from recommendation.retrieval.index.faiss_index import FaissVectorIndex
from recommendation.utils.config import RetrievalConfig


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def catalog() -> tuple[list[int], np.ndarray]:
    rng = np.random.default_rng(0)
    item_ids = [10, 20, 30, 40, 50]
    embeddings = _l2_normalize(rng.normal(size=(5, 8)).astype(np.float32))
    return item_ids, embeddings


@pytest.fixture
def faiss_index(catalog) -> FaissVectorIndex:
    item_ids, embeddings = catalog
    idx = FaissVectorIndex()
    idx.build(item_ids, embeddings)
    return idx


class _SpyVectorIndex(VectorIndex):
    """A fake `VectorIndex` that records every `search()` call's `k` and
    replays canned per-attempt results - used to prove
    `EligibilityRestrictedIndex` widens progressively rather than
    scanning the full index by default, without depending on a real
    backend's actual ANN approximation behavior.
    """

    def __init__(self, item_ids: list[int], scores_by_k: dict[int, list[tuple[int, float]]]) -> None:
        self._item_ids = item_ids
        self._scores_by_k = scores_by_k
        self.search_calls: list[int] = []

    def build(self, item_ids, embeddings) -> None:  # pragma: no cover - unused by these tests
        raise NotImplementedError

    def search(self, query_embeddings: np.ndarray, k: int) -> list[SearchResult]:
        self.search_calls.append(k)
        ranked = self._scores_by_k[k]
        return [SearchResult(item_ids=[i for i, _ in ranked], scores=[s for _, s in ranked])]

    def save(self, path) -> None:  # pragma: no cover - unused by these tests
        raise NotImplementedError

    def load(self, path) -> None:  # pragma: no cover - unused by these tests
        raise NotImplementedError

    @property
    def size(self) -> int:
        return len(self._item_ids)


def test_restricts_results_to_allowed_ids(faiss_index, catalog):
    _, embeddings = catalog
    restricted = EligibilityRestrictedIndex(faiss_index)
    allowed = {20, 40}
    [result] = restricted.search(embeddings[0].reshape(1, -1), k=5, allowed_ids=allowed)
    assert set(result.item_ids) <= allowed
    assert len(result.item_ids) == 2


def test_never_returns_disallowed_ids_even_when_they_rank_highest(faiss_index, catalog):
    """The closest match to `embeddings[0]` is `item_ids[0]` itself
    (score ~1.0) - excluding it from `allowed_ids` must still exclude it
    from results, even though it would otherwise be the top result.
    """
    item_ids, embeddings = catalog
    restricted = EligibilityRestrictedIndex(faiss_index)
    allowed = set(item_ids) - {item_ids[0]}
    [result] = restricted.search(embeddings[0].reshape(1, -1), k=1, allowed_ids=allowed)
    assert item_ids[0] not in result.item_ids
    assert result.item_ids[0] in allowed


def test_truncates_to_k(faiss_index, catalog):
    item_ids, embeddings = catalog
    restricted = EligibilityRestrictedIndex(faiss_index)
    [result] = restricted.search(embeddings[0].reshape(1, -1), k=2, allowed_ids=set(item_ids))
    assert len(result.item_ids) == 2


def test_empty_allowed_ids_returns_empty_results_for_every_query(faiss_index, catalog):
    _, embeddings = catalog
    restricted = EligibilityRestrictedIndex(faiss_index)
    results = restricted.search(embeddings[:3], k=3, allowed_ids=set())
    assert len(results) == 3
    for result in results:
        assert result.item_ids == []
        assert result.scores == []


def test_allowed_ids_not_present_in_index_are_ignored(faiss_index, catalog):
    _, embeddings = catalog
    restricted = EligibilityRestrictedIndex(faiss_index)
    [result] = restricted.search(embeddings[0].reshape(1, -1), k=5, allowed_ids={999999})
    assert result.item_ids == []


def test_handles_batch_of_queries(faiss_index, catalog):
    item_ids, embeddings = catalog
    restricted = EligibilityRestrictedIndex(faiss_index)
    results = restricted.search(embeddings[:3], k=2, allowed_ids=set(item_ids))
    assert len(results) == 3
    for result in results:
        assert len(result.item_ids) == 2


def test_rejects_non_positive_k(faiss_index, catalog):
    item_ids, embeddings = catalog
    restricted = EligibilityRestrictedIndex(faiss_index)
    with pytest.raises(ValueError):
        restricted.search(embeddings[0].reshape(1, -1), k=0, allowed_ids=set(item_ids))


def test_results_still_sorted_descending_by_score(faiss_index, catalog):
    item_ids, embeddings = catalog
    restricted = EligibilityRestrictedIndex(faiss_index)
    [result] = restricted.search(embeddings[0].reshape(1, -1), k=len(item_ids), allowed_ids=set(item_ids))
    assert result.scores == sorted(result.scores, reverse=True)


def test_all_allowed_matches_unrestricted_search(faiss_index, catalog):
    """Restricting to the full catalog id set must reproduce plain,
    unrestricted `VectorIndex.search` exactly (same ids, same scores) -
    the wrapper must not change ranking/scoring, only membership. At this
    fixture's 5-item catalog, the default `eligibility_oversample_floor`
    (20) exceeds the catalog size, so the very first oversample request
    already asks for everything - this is the same "small catalog, floor
    dominates" case where exact equality is still the right assertion (as
    opposed to `test_search_recall_against_brute_force_cosine_similarity`
    in `test_vector_index.py`, which is about the FAISS backend's own
    approximation, not this wrapper's oversampling).
    """
    item_ids, embeddings = catalog
    restricted = EligibilityRestrictedIndex(faiss_index)
    [restricted_result] = restricted.search(embeddings[2].reshape(1, -1), k=len(item_ids), allowed_ids=set(item_ids))
    [plain_result] = faiss_index.search(embeddings[2].reshape(1, -1), k=len(item_ids))
    assert restricted_result.item_ids == plain_result.item_ids
    assert restricted_result.scores == pytest.approx(plain_result.scores, abs=1e-6)


# --- Oversampling + progressive widening (2026-08-22 redesign) ---


def test_never_calls_search_with_full_index_size_as_default_k():
    """The normal path must ask for an oversampled `k`, not `index.size`
    - proves the wrapper no longer defeats an ANN backend's sublinear
    search by scanning everything on every query.
    """
    item_ids = list(range(1000))
    spy = _SpyVectorIndex(item_ids, scores_by_k={60: [(i, 1.0 - i * 0.001) for i in range(60)]})
    cfg = RetrievalConfig(eligibility_oversample_factor=3, eligibility_oversample_floor=20)
    restricted = EligibilityRestrictedIndex(spy, cfg)

    [result] = restricted.search(np.zeros((1, 4), dtype=np.float32), k=20, allowed_ids=set(item_ids))

    assert spy.search_calls == [60]  # k * oversample_factor, NOT index.size (1000)
    assert len(result.item_ids) == 20


def test_oversampling_finds_enough_eligible_when_top_hits_are_mostly_ineligible():
    """If the first oversampled batch is mostly ineligible, the wrapper
    must widen and retry rather than giving up with too few results.
    """
    item_ids = list(range(1000))
    ineligible_prefix = list(range(50))  # first attempt's top hits, none eligible
    eligible_tail = [900, 901, 902]  # only found once the request widens far enough
    cfg = RetrievalConfig(eligibility_oversample_factor=1, eligibility_oversample_floor=50, eligibility_widen_multiplier=4, eligibility_max_widen_attempts=3)
    # attempt sequence for k=3: requested = max(3*1,50)=50 -> 200 -> 800 -> (capped 1000, but max_attempts=3 stops it at 800)
    scores_by_k = {
        50: [(i, 1.0 - i * 0.001) for i in ineligible_prefix],
        200: [(i, 1.0 - i * 0.001) for i in ineligible_prefix],
        800: [(i, 1.0 - i * 0.001) for i in ineligible_prefix] + [(pid, 0.01) for pid in eligible_tail],
    }
    spy = _SpyVectorIndex(item_ids, scores_by_k)
    restricted = EligibilityRestrictedIndex(spy, cfg)

    [result] = restricted.search(np.zeros((1, 4), dtype=np.float32), k=3, allowed_ids=set(eligible_tail))

    assert spy.search_calls == [50, 200, 800]
    assert set(result.item_ids) == set(eligible_tail)


def test_progressive_widening_stops_at_max_attempts_without_infinite_loop():
    """Even when eligible items are never found, the widening loop must
    terminate within `eligibility_max_widen_attempts` calls, not loop
    forever - and must return an honest (possibly empty) under-fill, not
    raise.
    """
    item_ids = list(range(1000))
    scores_by_k = {k: [(i, 1.0 - i * 0.001) for i in range(min(k, len(item_ids)))] for k in (30, 60, 120, 240)}
    cfg = RetrievalConfig(eligibility_oversample_factor=3, eligibility_oversample_floor=30, eligibility_widen_multiplier=2, eligibility_max_widen_attempts=4)
    spy = _SpyVectorIndex(item_ids, scores_by_k)
    restricted = EligibilityRestrictedIndex(spy, cfg)

    # allowed_ids contains items never present in any of the canned
    # per-attempt result lists (all attempts only ever return low ids).
    [result] = restricted.search(np.zeros((1, 4), dtype=np.float32), k=10, allowed_ids={999})

    assert len(spy.search_calls) == cfg.eligibility_max_widen_attempts
    assert result.item_ids == []


def test_returns_fewer_than_k_when_insufficient_eligible_items_exist_globally(faiss_index, catalog):
    """Honest under-fill: if fewer than `k` eligible items exist in the
    whole catalog, the wrapper must return exactly that many, never pad
    with ineligible ids and never raise.
    """
    item_ids, embeddings = catalog
    restricted = EligibilityRestrictedIndex(faiss_index)
    allowed = {item_ids[0], item_ids[1]}  # only 2 eligible, k asks for 4
    [result] = restricted.search(embeddings[0].reshape(1, -1), k=4, allowed_ids=allowed)
    assert len(result.item_ids) == 2
    assert set(result.item_ids) == allowed


# --- ScaNN: identical behavior through the same backend-agnostic wrapper,
# real backend, Linux/Docker only (see module docstring). ---

def test_scann_backend_restricts_results_to_allowed_ids(catalog):
    pytest.importorskip("scann")
    from recommendation.retrieval.index.scann_index import ScannVectorIndex

    item_ids, embeddings = catalog
    idx = ScannVectorIndex()
    idx.build(item_ids, embeddings)
    restricted = EligibilityRestrictedIndex(idx)
    allowed = {20, 40}
    [result] = restricted.search(embeddings[0].reshape(1, -1), k=5, allowed_ids=allowed)
    assert set(result.item_ids) <= allowed
    assert len(result.item_ids) == 2


def test_faiss_and_scann_agree_through_eligibility_restriction(catalog):
    """Cross-backend consistency, extended through the Phase 11 wrapper:
    both backends are now independently approximate (HNSW vs. tree+AH),
    so exact list-order/score equality is no longer the right bar even
    at this trivial 5-item catalog - both must still restrict to the
    same allowed set and agree on which item ranks best.
    """
    pytest.importorskip("scann")
    from recommendation.retrieval.index.scann_index import ScannVectorIndex

    item_ids, embeddings = catalog
    faiss_idx = FaissVectorIndex()
    faiss_idx.build(item_ids, embeddings)
    scann_idx = ScannVectorIndex()
    scann_idx.build(item_ids, embeddings)

    allowed = {10, 30, 50}
    query = embeddings[3].reshape(1, -1)
    [faiss_result] = EligibilityRestrictedIndex(faiss_idx).search(query, k=len(item_ids), allowed_ids=allowed)
    [scann_result] = EligibilityRestrictedIndex(scann_idx).search(query, k=len(item_ids), allowed_ids=allowed)

    assert set(faiss_result.item_ids) == allowed
    assert set(scann_result.item_ids) == allowed
    assert faiss_result.item_ids[0] == scann_result.item_ids[0]
    assert faiss_result.scores == pytest.approx(scann_result.scores, abs=1e-4)
