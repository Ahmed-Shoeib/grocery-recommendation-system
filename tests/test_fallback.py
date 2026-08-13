import pytest

from recommendation.features.product_features import ProductFeatures
from recommendation.serving.fallback import (
    blend_candidate_lists,
    category_popularity_ranking,
    global_popularity_ranking,
    top_affinity_category,
    waterfall_candidates,
)


def _pf(pid: int, category: str, purchase_count: int = 0, cart_add_count: int = 0) -> ProductFeatures:
    return ProductFeatures(
        product_id=pid, category_id=pid, category_name=category, parent_category_name=None, brand=None, tags=[],
        price=1.0, effective_price=1.0, discount_percentage=0.0, is_active=True, stock_quantity=10,
        purchase_count=purchase_count, distinct_purchasers=0, cart_add_count=cart_add_count, review_count=0, average_rating=None,
    )


def test_global_popularity_ranking_sorts_by_purchase_count_desc():
    features = {
        1: _pf(1, "A", purchase_count=5),
        2: _pf(2, "B", purchase_count=10),
        3: _pf(3, "A", purchase_count=1),
    }
    assert global_popularity_ranking(features) == [2, 1, 3]


def test_global_popularity_ranking_ties_broken_by_cart_then_id():
    features = {
        2: _pf(2, "A", purchase_count=1, cart_add_count=0),
        1: _pf(1, "A", purchase_count=1, cart_add_count=5),
    }
    assert global_popularity_ranking(features) == [1, 2]


def test_category_popularity_ranking_filters_to_category():
    features = {1: _pf(1, "Dairy", purchase_count=3), 2: _pf(2, "Snacks", purchase_count=10)}
    assert category_popularity_ranking(features, "Dairy") == [1]


def test_category_popularity_ranking_none_category_is_empty():
    features = {1: _pf(1, "Dairy", purchase_count=3)}
    assert category_popularity_ranking(features, None) == []


def test_top_affinity_category_picks_max_weight():
    assert top_affinity_category({"Dairy": 0.3, "Snacks": 0.7}) == "Snacks"


def test_top_affinity_category_empty_is_none():
    assert top_affinity_category({}) is None


def test_blend_candidate_lists_respects_weights_at_top():
    # personalized weighted much higher -> its top pick should lead.
    sources = [("personalized", [10, 11], 0.9), ("popularity", [20, 21], 0.1)]
    blended = blend_candidate_lists(sources, pool_size=10)
    assert blended[0].product_id == 10
    assert blended[0].source == "personalized"


def test_blend_candidate_lists_dedupes_keeping_best_score():
    sources = [("personalized", [10], 0.5), ("popularity", [10], 0.9)]
    blended = blend_candidate_lists(sources, pool_size=10)
    ids = [c.product_id for c in blended]
    assert ids.count(10) == 1
    assert blended[0].source == "popularity"  # 0.9*1.0 > 0.5*1.0


def test_blend_candidate_lists_capped_by_pool_size():
    sources = [("popularity", [1, 2, 3, 4, 5], 1.0)]
    blended = blend_candidate_lists(sources, pool_size=3)
    assert len(blended) == 3


def test_waterfall_candidates_preserves_source_priority_order():
    sources = [("preferred_category", [10, 11]), ("global_popularity", [20, 21])]
    result = waterfall_candidates(sources, pool_size=10)
    assert [c.product_id for c in result] == [10, 11, 20, 21]


def test_waterfall_candidates_dedupes_across_sources():
    sources = [("preferred_category", [10, 11]), ("global_popularity", [11, 20])]
    result = waterfall_candidates(sources, pool_size=10)
    assert [c.product_id for c in result] == [10, 11, 20]


def test_waterfall_candidates_stops_at_pool_size():
    sources = [("preferred_category", [10, 11, 12]), ("global_popularity", [20, 21])]
    result = waterfall_candidates(sources, pool_size=2)
    assert [c.product_id for c in result] == [10, 11]
