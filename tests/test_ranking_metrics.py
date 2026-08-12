import pytest

from recommendation.evaluation.retrieval_metrics import (
    mean_ndcg_at_k,
    mean_precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)


def test_precision_at_k_counts_relevant_fraction_of_k():
    assert precision_at_k([5, 3, 1, 9], {1, 9}, k=4) == pytest.approx(0.5)


def test_precision_at_k_single_hit_within_k():
    assert precision_at_k([5, 1, 3, 9], {1}, k=2) == pytest.approx(0.5)


def test_precision_at_k_miss_outside_k():
    assert precision_at_k([5, 3, 1, 9], {1}, k=1) == 0.0


def test_precision_at_k_empty_relevant_set_is_zero():
    assert precision_at_k([1, 2, 3], set(), k=3) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    assert ndcg_at_k([1, 2, 3], {1}, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_lower_when_hit_is_further_down():
    top = ndcg_at_k([1, 2, 3], {1}, k=3)
    bottom = ndcg_at_k([2, 3, 1], {1}, k=3)
    assert bottom < top


def test_ndcg_at_k_zero_when_miss_outside_k():
    assert ndcg_at_k([5, 3, 1, 9], {1}, k=1) == 0.0


def test_ndcg_at_k_zero_for_empty_relevant_set():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


def test_ndcg_at_k_multi_relevant_ideal_case():
    # Two relevant items both at the top -> ideal DCG achieved -> NDCG = 1.
    assert ndcg_at_k([1, 2, 3], {1, 2}, k=3) == pytest.approx(1.0)


def test_reciprocal_rank_first_position_is_one():
    assert reciprocal_rank([1, 2, 3], {1}) == pytest.approx(1.0)


def test_reciprocal_rank_third_position():
    assert reciprocal_rank([2, 3, 1], {1}) == pytest.approx(1 / 3)


def test_reciprocal_rank_not_found_is_zero():
    assert reciprocal_rank([2, 3, 4], {1}) == 0.0


def test_reciprocal_rank_uses_first_relevant_hit():
    assert reciprocal_rank([2, 1, 3, 1], {1}) == pytest.approx(0.5)


def test_mean_precision_at_k_averages():
    rankings = [[1, 2, 3], [4, 5, 6]]
    relevant_sets = [{1}, {9}]
    assert mean_precision_at_k(rankings, relevant_sets, k=1) == pytest.approx(0.5)


def test_mean_ndcg_at_k_averages():
    rankings = [[1, 2, 3], [4, 5, 6]]
    relevant_sets = [{1}, {9}]
    assert mean_ndcg_at_k(rankings, relevant_sets, k=1) == pytest.approx(0.5)


def test_mean_reciprocal_rank_averages():
    rankings = [[1, 2, 3], [4, 5, 6]]
    relevant_sets = [{1}, {9}]
    assert mean_reciprocal_rank(rankings, relevant_sets) == pytest.approx(0.5)


def test_mean_metrics_empty_input_is_zero():
    assert mean_precision_at_k([], [], k=5) == 0.0
    assert mean_ndcg_at_k([], [], k=5) == 0.0
    assert mean_reciprocal_rank([], []) == 0.0
