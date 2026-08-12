import pytest

from recommendation.evaluation.retrieval_metrics import hit_rate_at_k, mean_hit_rate_at_k, mean_recall_at_k, recall_at_k


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
