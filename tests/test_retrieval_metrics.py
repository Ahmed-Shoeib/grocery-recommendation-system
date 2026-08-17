import math

import pytest

from recommendation.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    mean_hit_rate_at_k,
    mean_recall_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)


def test_recall_at_k_hit_within_top_k():
    assert recall_at_k([5, 3, 1, 9], {1}, k=3) == 1.0


def test_recall_at_k_miss_outside_top_k():
    assert recall_at_k([5, 3, 1, 9], {1}, k=2) == 0.0


def test_recall_at_k_with_multiple_relevant_items_is_fraction_found():
    assert recall_at_k([5, 3, 1, 9], {1, 9, 100}, k=4) == 2 / 3


def test_recall_at_k_empty_relevant_set_is_zero():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_hit_rate_at_k_is_binary():
    assert hit_rate_at_k([5, 3, 1, 9], {1}, k=3) == 1.0
    assert hit_rate_at_k([5, 3, 1, 9], {1}, k=2) == 0.0


def test_hit_rate_at_k_any_relevant_item_present_counts_as_hit():
    assert hit_rate_at_k([5, 3, 1, 9], {1, 100}, k=3) == 1.0  # found 1 of 2 relevant -> still a hit


def test_recall_and_hit_rate_coincide_for_single_relevant_item():
    """The V1 leave-one-out protocol always has exactly one relevant item
    per query - Recall@K and HitRate@K are then mathematically identical,
    which this test documents rather than treats as a coincidence.
    """
    ranked = [7, 2, 9, 4]
    relevant = {9}
    for k in (1, 2, 3, 4):
        assert recall_at_k(ranked, relevant, k) == hit_rate_at_k(ranked, relevant, k)


def test_mean_recall_at_k_averages_across_queries():
    rankings = [[1, 2, 3], [4, 5, 6]]
    relevant_sets = [{1}, {9}]  # first query hits, second misses
    assert mean_recall_at_k(rankings, relevant_sets, k=1) == 0.5


def test_mean_hit_rate_at_k_averages_across_queries():
    rankings = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    relevant_sets = [{1}, {5}, {100}]  # 2 hits, 1 miss
    assert mean_hit_rate_at_k(rankings, relevant_sets, k=3) == pytest.approx(2 / 3)


def test_mean_recall_at_k_empty_input_is_zero():
    assert mean_recall_at_k([], [], k=5) == 0.0
    assert mean_hit_rate_at_k([], [], k=5) == 0.0


# --- multi-relevant-item worked example (temporal future-purchase spec) ----
# Relevant = {20, 50, 80}; Recommendations = [10, 50, 12, 80, 20]
# Hits at 0-indexed positions 1 (50), 3 (80), 4 (20).

_MULTI_RANKED = [10, 50, 12, 80, 20]
_MULTI_RELEVANT = {20, 50, 80}


def test_precision_at_k_multiple_relevant_items():
    assert precision_at_k(_MULTI_RANKED, _MULTI_RELEVANT, k=5) == pytest.approx(3 / 5)


def test_recall_at_k_multiple_relevant_items_full_recovery():
    assert recall_at_k(_MULTI_RANKED, _MULTI_RELEVANT, k=5) == pytest.approx(1.0)


def test_hit_rate_at_k_multiple_relevant_items():
    assert hit_rate_at_k(_MULTI_RANKED, _MULTI_RELEVANT, k=5) == 1.0


def test_ndcg_at_k_multiple_relevant_items_matches_hand_computed_value():
    dcg = 1.0 / math.log2(1 + 2) + 1.0 / math.log2(3 + 2) + 1.0 / math.log2(4 + 2)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    expected = dcg / idcg
    assert ndcg_at_k(_MULTI_RANKED, _MULTI_RELEVANT, k=5) == pytest.approx(expected)
    assert ndcg_at_k(_MULTI_RANKED, _MULTI_RELEVANT, k=5) == pytest.approx(0.6797, abs=1e-3)


def test_reciprocal_rank_multiple_relevant_items_uses_first_hit():
    # first relevant item (50) is at 0-indexed position 1 -> 1-indexed rank 2
    assert reciprocal_rank(_MULTI_RANKED, _MULTI_RELEVANT) == pytest.approx(0.5)


# --- edge cases required by the metrics spec --------------------------------

def test_no_relevant_item_retrieved():
    ranked = [1, 2, 3, 4, 5]
    relevant = {999}
    assert precision_at_k(ranked, relevant, k=5) == 0.0
    assert recall_at_k(ranked, relevant, k=5) == 0.0
    assert hit_rate_at_k(ranked, relevant, k=5) == 0.0
    assert ndcg_at_k(ranked, relevant, k=5) == 0.0
    assert reciprocal_rank(ranked, relevant) == 0.0


def test_relevant_item_at_rank_one():
    ranked = [42, 1, 2, 3]
    relevant = {42}
    assert precision_at_k(ranked, relevant, k=1) == 1.0
    assert ndcg_at_k(ranked, relevant, k=1) == pytest.approx(1.0)
    assert reciprocal_rank(ranked, relevant) == pytest.approx(1.0)


def test_relevant_item_near_k_boundary():
    ranked = [1, 2, 3, 4, 42]
    relevant = {42}
    assert hit_rate_at_k(ranked, relevant, k=5) == 1.0
    assert hit_rate_at_k(ranked, relevant, k=4) == 0.0
    assert recall_at_k(ranked, relevant, k=5) == 1.0
    assert recall_at_k(ranked, relevant, k=4) == 0.0


def test_k_larger_than_number_of_recommendations():
    ranked = [1, 2, 3]
    relevant = {2}
    # k=100 must not error and must behave as if only 3 items existed
    assert precision_at_k(ranked, relevant, k=100) == pytest.approx(1 / 100)
    assert recall_at_k(ranked, relevant, k=100) == 1.0
    assert hit_rate_at_k(ranked, relevant, k=100) == 1.0
    # NDCG still discounts by the relevant item's actual rank (position 2,
    # 1-indexed) even though k=100 vastly exceeds the list length - it is
    # NOT 1.0 just because the item was eventually found.
    assert ndcg_at_k(ranked, relevant, k=100) == pytest.approx(1.0 / math.log2(3))


def test_empty_relevant_set_behavior_is_documented_as_zero_everywhere():
    ranked = [1, 2, 3]
    empty: set[int] = set()
    assert precision_at_k(ranked, empty, k=3) == 0.0
    assert recall_at_k(ranked, empty, k=3) == 0.0
    assert hit_rate_at_k(ranked, empty, k=3) == 0.0
    assert ndcg_at_k(ranked, empty, k=3) == 0.0
    assert reciprocal_rank(ranked, empty) == 0.0


def test_duplicate_recommendations_do_not_inflate_precision_or_recall():
    """A ranked list containing the same id twice must not be counted
    twice - both metrics dedupe via `set(ranked_ids[:k])` internally.
    """
    ranked = [7, 7, 7, 8]
    relevant = {7}
    assert precision_at_k(ranked, relevant, k=3) == pytest.approx(1 / 3)  # 1 distinct hit / k, not 3/3
    assert recall_at_k(ranked, relevant, k=3) == 1.0  # still fully recovered, not >1.0


def test_empty_ranking_list():
    assert precision_at_k([], {1}, k=5) == 0.0
    assert recall_at_k([], {1}, k=5) == 0.0
    assert hit_rate_at_k([], {1}, k=5) == 0.0
    assert ndcg_at_k([], {1}, k=5) == 0.0
    assert reciprocal_rank([], {1}) == 0.0
