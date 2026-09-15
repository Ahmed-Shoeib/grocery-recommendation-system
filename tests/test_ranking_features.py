import numpy as np
import pytest

from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import UserFeatures
from recommendation.ranking.features import RANKING_FEATURE_NAMES, build_ranking_feature_vector


def _user_features(**overrides) -> UserFeatures:
    defaults = dict(
        user_id=1,
        preferred_categories=[],
        age_group=None,
        has_preferred_category=False,
        has_age_group=False,
        click_count=0,
        purchase_count=0,
        distinct_products_purchased=0,
        cart_item_count=0,
        search_count=0,
        has_chatbot_context=False,
        total_engagement_events=0,
        category_affinity={},
        brand_affinity={},
        semantic_embedding=None,
    )
    defaults.update(overrides)
    return UserFeatures(**defaults)


def _product_features(**overrides) -> ProductFeatures:
    defaults = dict(
        product_id=1,
        category_id=1,
        category_name="Dairy",
        parent_category_name=None,
        brand="GreenValley",
        tags=[],
        price=10.0,
        effective_price=10.0,
        discount_percentage=0.0,
        is_active=True,
        stock_quantity=5,
        purchase_count=0,
        distinct_purchasers=0,
        cart_add_count=0,
        review_count=0,
        average_rating=None,
    )
    defaults.update(overrides)
    return ProductFeatures(**defaults)


def test_feature_vector_length_matches_names():
    vec = build_ranking_feature_vector(_user_features(), _product_features(), None, 0.5, 0, 50, 100.0)
    assert vec.shape == (len(RANKING_FEATURE_NAMES),)


def test_category_affinity_match_uses_user_affinity_for_items_category():
    user = _user_features(category_affinity={"Dairy": 0.7, "Snacks": 0.3})
    vec = build_ranking_feature_vector(user, _product_features(category_name="Dairy"), None, 0.0, 0, 50, 100.0)
    idx = RANKING_FEATURE_NAMES.index("category_affinity_match")
    assert vec[idx] == pytest.approx(0.7)


def test_category_affinity_match_zero_when_category_not_in_affinity():
    user = _user_features(category_affinity={"Snacks": 1.0})
    vec = build_ranking_feature_vector(user, _product_features(category_name="Dairy"), None, 0.0, 0, 50, 100.0)
    idx = RANKING_FEATURE_NAMES.index("category_affinity_match")
    assert vec[idx] == 0.0


def test_brand_affinity_match_is_not_a_ranking_feature():
    """Production-safe contract (docs/production-feature-parity-audit.md):
    the real SQL Server `Products` table has no `Brand` column, so
    `brand_affinity_match` was removed entirely from `RANKING_FEATURE_NAMES`.
    """
    assert "brand_affinity_match" not in RANKING_FEATURE_NAMES


def test_preferred_category_match_flag_single_favorite():
    user = _user_features(preferred_categories=["Dairy"], has_preferred_category=True)
    matching = build_ranking_feature_vector(user, _product_features(category_name="Dairy"), None, 0.0, 0, 50, 100.0)
    non_matching = build_ranking_feature_vector(user, _product_features(category_name="Snacks"), None, 0.0, 0, 50, 100.0)
    idx = RANKING_FEATURE_NAMES.index("preferred_category_match")
    assert matching[idx] == 1.0
    assert non_matching[idx] == 0.0


def test_preferred_category_match_flag_multiple_favorites():
    """The real backend models preferred/favorite categories as a LIST
    (docs/production-feature-parity-audit.md) - a match against ANY
    favorite should flag true, not just a single arbitrarily-chosen one.
    """
    user = _user_features(preferred_categories=["Snacks", "Dairy", "Produce"], has_preferred_category=True)
    for category in ("Snacks", "Dairy", "Produce"):
        vec = build_ranking_feature_vector(user, _product_features(category_name=category), None, 0.0, 0, 50, 100.0)
        idx = RANKING_FEATURE_NAMES.index("preferred_category_match")
        assert vec[idx] == 1.0
    non_matching = build_ranking_feature_vector(user, _product_features(category_name="Frozen"), None, 0.0, 0, 50, 100.0)
    assert non_matching[RANKING_FEATURE_NAMES.index("preferred_category_match")] == 0.0


def test_semantic_similarity_identical_vectors_is_one():
    vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    user = _user_features(semantic_embedding=vec_a)
    vec = build_ranking_feature_vector(user, _product_features(), vec_a, 0.0, 0, 50, 100.0)
    idx = RANKING_FEATURE_NAMES.index("semantic_cosine_similarity")
    has_idx = RANKING_FEATURE_NAMES.index("has_semantic_similarity")
    assert vec[idx] == pytest.approx(1.0)
    assert vec[has_idx] == 1.0


def test_semantic_similarity_missing_embedding_is_zero_with_flag_off():
    user = _user_features(semantic_embedding=None)
    vec = build_ranking_feature_vector(user, _product_features(), np.array([1.0, 0.0]), 0.0, 0, 50, 100.0)
    idx = RANKING_FEATURE_NAMES.index("semantic_cosine_similarity")
    has_idx = RANKING_FEATURE_NAMES.index("has_semantic_similarity")
    assert vec[idx] == 0.0
    assert vec[has_idx] == 0.0


def test_retrieval_score_and_rank_pass_through():
    vec = build_ranking_feature_vector(_user_features(), _product_features(), None, 0.42, 5, 11, 100.0)
    score_idx = RANKING_FEATURE_NAMES.index("retrieval_score")
    rank_idx = RANKING_FEATURE_NAMES.index("retrieval_rank_normalized")
    assert vec[score_idx] == pytest.approx(0.42)
    assert vec[rank_idx] == pytest.approx(0.5)  # rank 5 of pool_size 11 -> 5/10


def test_is_active_is_not_a_ranking_feature():
    """Production-safe contract (docs/production-feature-parity-audit.md):
    the real SQL Server `Products` table has no `isActive` column, so
    `item_is_active` was removed entirely - `stock_quantity` (a real
    field) is what actually gates eligibility (`serving.eligibility`).
    """
    assert "item_is_active" not in RANKING_FEATURE_NAMES
    out_of_stock = _product_features(is_active=False, stock_quantity=0)
    vec = build_ranking_feature_vector(_user_features(), out_of_stock, None, 0.0, 0, 50, 100.0)
    # No exception, no filtering - the vector is still produced normally
    # (ranking never filters; that's serving.eligibility's job).
    assert vec.shape == (len(RANKING_FEATURE_NAMES),)


def test_normalized_price_capped_at_one():
    expensive = _product_features(effective_price=500.0)
    vec = build_ranking_feature_vector(_user_features(), expensive, None, 0.0, 0, 50, 100.0)
    idx = RANKING_FEATURE_NAMES.index("item_normalized_price")
    assert vec[idx] == pytest.approx(1.0)


# --- STEP 6: price-aware ranker features (docs/data-mapping.md section 15) --

def test_semantic_and_retrieval_features_are_retained_not_removed():
    """Clarification A: `semantic_cosine_similarity`, `has_semantic_similarity`,
    `retrieval_score`, and `retrieval_rank_normalized` are NOT inherently
    unsafe production features - their current values are only unsafe
    because today's embedding space was trained on brand/tag/ingredient/
    parent-category text (removed - see `embeddings.text_builder`). They
    stay in the ranker contract so they become valid again once the
    Two-Tower is retrained on the new, safe inputs.
    """
    for retained in (
        "semantic_cosine_similarity", "has_semantic_similarity",
        "retrieval_score", "retrieval_rank_normalized",
    ):
        assert retained in RANKING_FEATURE_NAMES


def test_ranking_feature_dimension_is_24_after_production_safe_redesign():
    """29 -> 24: `user_has_age_group`, `item_discount_fraction`,
    `item_is_active`, `item_is_discounted`, `brand_affinity_match` were
    removed (docs/production-feature-parity-audit.md - none of the real
    fields they depended on exist in the real SQL Server schema).
    """
    assert len(RANKING_FEATURE_NAMES) == 24
    for removed in (
        "user_has_age_group", "item_discount_fraction", "item_is_active",
        "item_is_discounted", "brand_affinity_match",
    ):
        assert removed not in RANKING_FEATURE_NAMES


def test_item_relative_price_passes_through():
    product = _product_features(category_relative_price=0.8)
    vec = build_ranking_feature_vector(_user_features(), product, None, 0.0, 0, 50, 100.0)
    assert vec[RANKING_FEATURE_NAMES.index("item_category_relative_price")] == pytest.approx(0.8)


def test_no_price_profile_gives_neutral_price_features():
    user = _user_features(price_profile=None)
    vec = build_ranking_feature_vector(user, _product_features(effective_price=10.0), None, 0.0, 0, 50, 100.0)
    assert vec[RANKING_FEATURE_NAMES.index("user_has_price_profile")] == 0.0
    assert vec[RANKING_FEATURE_NAMES.index("user_normalized_typical_price")] == 0.0
    assert vec[RANKING_FEATURE_NAMES.index("price_relative_distance")] == 0.0
    assert vec[RANKING_FEATURE_NAMES.index("price_tier_match")] == 0.0


def test_price_relative_distance_reflects_actual_gap_not_model_ranking():
    """The handcrafted feature itself must correctly reflect price
    distance for candidates at different distances from the user's
    typical price - independent of any trained model's eventual ranking
    decision (docs/data-mapping.md section 15's "do not require an
    untrained model to rank" note - this test never touches a model).
    """
    from recommendation.features.price import UserPriceProfile

    profile = UserPriceProfile(
        typical_price=6.0, price_spread=1.0, price_tier="mid", supporting_purchase_count=4, fallback_source="purchase_history"
    )
    user = _user_features(price_profile=profile)

    close = _product_features(effective_price=5.5, price_tier="mid")
    reasonable = _product_features(effective_price=7.0, price_tier="mid")
    far = _product_features(effective_price=30.0, price_tier="premium")

    vec_close = build_ranking_feature_vector(user, close, None, 0.0, 0, 50, 100.0)
    vec_reasonable = build_ranking_feature_vector(user, reasonable, None, 0.0, 0, 50, 100.0)
    vec_far = build_ranking_feature_vector(user, far, None, 0.0, 0, 50, 100.0)

    idx = RANKING_FEATURE_NAMES.index("price_relative_distance")
    assert vec_close[idx] < vec_reasonable[idx] < vec_far[idx]

    tier_idx = RANKING_FEATURE_NAMES.index("price_tier_match")
    assert vec_close[tier_idx] == 1.0  # mid == mid
    assert vec_reasonable[tier_idx] == 1.0
    assert vec_far[tier_idx] == 0.0  # premium != mid

    has_idx = RANKING_FEATURE_NAMES.index("user_has_price_profile")
    assert vec_close[has_idx] == vec_reasonable[has_idx] == vec_far[has_idx] == 1.0

    typical_idx = RANKING_FEATURE_NAMES.index("user_normalized_typical_price")
    assert vec_close[typical_idx] == pytest.approx(0.06)  # 6.0 / 100.0 max_price, same for every candidate
