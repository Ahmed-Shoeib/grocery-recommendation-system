"""Tests for `EligibilityRestrictedIndex` (Phase 11) - the backend-
agnostic query-time wrapper that restricts `VectorIndex.search` results
to a caller-supplied eligible-id set. FAISS-backed tests run everywhere;
the ScaNN-backed tests are skipped (not failed) on native Windows via
`importorskip`, same convention as `test_scann_index.py`, and run for
real in the Linux/Docker image.
"""

from __future__ import annotations

import numpy as np
import pytest

from recommendation.retrieval.index.eligibility_filter import EligibilityRestrictedIndex
from recommendation.retrieval.index.faiss_index import FaissVectorIndex


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
    the wrapper must not change ranking/scoring, only membership.
    """
    item_ids, embeddings = catalog
    restricted = EligibilityRestrictedIndex(faiss_index)
    [restricted_result] = restricted.search(embeddings[2].reshape(1, -1), k=len(item_ids), allowed_ids=set(item_ids))
    [plain_result] = faiss_index.search(embeddings[2].reshape(1, -1), k=len(item_ids))
    assert restricted_result.item_ids == plain_result.item_ids
    assert restricted_result.scores == pytest.approx(plain_result.scores, abs=1e-6)


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
    """Cross-backend agreement, extended through the Phase 11 wrapper:
    both backends must produce identical restricted results, not just
    identical unrestricted ones (see `test_vector_index.py`/
    `test_scann_index.py`).
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

    assert faiss_result.item_ids == scann_result.item_ids
    assert faiss_result.scores == pytest.approx(scann_result.scores, abs=1e-4)
