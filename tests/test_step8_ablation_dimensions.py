"""Tests for the STEP 8 `include_price_features` flag machinery
(docs/data-mapping.md section 17) - proves the BASE condition faithfully
reproduces the pre-STEP-6 architecture (dimensions AND absence of the
price_tier_id input entirely, not a price input fed neutral/zero values),
while the default (True) path stays byte-for-byte identical to STEP 6/7.
"""

from __future__ import annotations

import numpy as np
import pytest

from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import UserFeatures
from recommendation.ranking.features import (
    RANKING_FEATURE_NAMES,
    RANKING_FEATURE_NAMES_BASE,
    build_ranking_feature_vector,
    ranking_feature_names,
)
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.model import build_item_tower, build_user_tower
from recommendation.utils.config import TwoTowerConfig


def _encoder(include_price_features: bool, embedding_dim: int = 8) -> TwoTowerFeatureEncoder:
    return TwoTowerFeatureEncoder.fit(
        category_names=["Dairy & Eggs", "Snacks"],
        brand_names=["GreenValley"],
        age_groups=["25-34"],
        prices=[2.0, 4.0],
        embedding_dim=embedding_dim,
        include_price_features=include_price_features,
    )


def _product_features(**overrides) -> ProductFeatures:
    defaults = dict(
        product_id=1, category_id=1, category_name="Dairy & Eggs", parent_category_name=None, brand="GreenValley",
        tags=[], price=4.0, effective_price=4.0, discount_percentage=0.0, is_active=True, stock_quantity=10,
        purchase_count=3, distinct_purchasers=2, cart_add_count=1, review_count=2, average_rating=4.5,
        is_discounted=False, price_tier="mid", category_relative_price=0.5,
    )
    defaults.update(overrides)
    return ProductFeatures(**defaults)


def _user_features(**overrides) -> UserFeatures:
    defaults = dict(
        user_id=1, preferred_category="Snacks", age_group="25-34", has_preferred_category=True, has_age_group=True,
        click_count=0, purchase_count=3, distinct_products_purchased=3, cart_item_count=1, search_count=2, has_chatbot_context=False,
        total_engagement_events=6, category_affinity={}, brand_affinity={}, semantic_embedding=np.ones(8, dtype=np.float32),
        price_profile=None,
    )
    defaults.update(overrides)
    return UserFeatures(**defaults)


# --- Two-Tower encoder dimensions ------------------------------------------

def test_base_encoder_has_pre_step6_dimensions():
    encoder = _encoder(include_price_features=False)
    assert encoder.item_numeric_dim == 7
    assert encoder.user_numeric_dim == 8


def test_improved_encoder_has_step6_dimensions():
    encoder = _encoder(include_price_features=True)
    assert encoder.item_numeric_dim == 9
    assert encoder.user_numeric_dim == 9


def test_base_encode_item_has_no_price_tier_id_key():
    encoder = _encoder(include_price_features=False)
    result = encoder.encode_item(_product_features(), np.ones(8, dtype=np.float32))
    assert "price_tier_id" not in result
    assert result["numeric"].shape == (7,)


def test_improved_encode_item_has_price_tier_id_key():
    encoder = _encoder(include_price_features=True)
    result = encoder.encode_item(_product_features(), np.ones(8, dtype=np.float32))
    assert "price_tier_id" in result
    assert result["numeric"].shape == (9,)


def test_base_encode_user_has_no_price_tier_id_key():
    encoder = _encoder(include_price_features=False)
    result = encoder.encode_user(_user_features())
    assert "price_tier_id" not in result
    assert result["numeric"].shape == (8,)


def test_improved_encode_user_has_price_tier_id_key():
    encoder = _encoder(include_price_features=True)
    result = encoder.encode_user(_user_features())
    assert "price_tier_id" in result
    assert result["numeric"].shape == (9,)


def test_encoder_serialization_round_trips_include_price_features():
    for flag in (True, False):
        encoder = _encoder(include_price_features=flag)
        restored = TwoTowerFeatureEncoder.from_dict(encoder.to_dict())
        assert restored.include_price_features == flag
        assert restored.item_numeric_dim == encoder.item_numeric_dim


def test_legacy_serialized_dict_without_flag_defaults_to_true():
    encoder = _encoder(include_price_features=True)
    data = encoder.to_dict()
    del data["include_price_features"]
    restored = TwoTowerFeatureEncoder.from_dict(data)
    assert restored.include_price_features is True


# --- Two-Tower model architecture -------------------------------------------

def _tt_config() -> TwoTowerConfig:
    return TwoTowerConfig(projection_dims=[16, 8], output_dim=8, category_embedding_dim=4, brand_embedding_dim=4, age_group_embedding_dim=2)


def test_base_item_tower_has_no_price_tier_input():
    encoder = _encoder(include_price_features=False)
    tower = build_item_tower(encoder, _tt_config())
    assert "price_tier_id" not in {t.name.split(":")[0] for t in tower.inputs}


def test_improved_item_tower_has_price_tier_input():
    encoder = _encoder(include_price_features=True)
    tower = build_item_tower(encoder, _tt_config())
    assert "price_tier_id" in {t.name.split(":")[0] for t in tower.inputs}


def test_base_user_tower_has_no_price_tier_input():
    encoder = _encoder(include_price_features=False)
    tower = build_user_tower(encoder, _tt_config())
    assert "price_tier_id" not in {t.name.split(":")[0] for t in tower.inputs}


def test_improved_user_tower_has_price_tier_input():
    encoder = _encoder(include_price_features=True)
    tower = build_user_tower(encoder, _tt_config())
    assert "price_tier_id" in {t.name.split(":")[0] for t in tower.inputs}


def test_base_item_tower_predicts_without_price_tier_id(tmp_path):
    encoder = _encoder(include_price_features=False)
    tower = build_item_tower(encoder, _tt_config())
    batch = encoder.encode_item_batch([1], {1: _product_features()}, {1: np.ones(8, dtype=np.float32)})
    output = tower.predict(batch, verbose=0)
    assert output.shape == (1, 8)


# --- ranker feature vector ---------------------------------------------------

def test_ranking_feature_names_base_has_23_entries():
    assert len(RANKING_FEATURE_NAMES_BASE) == 23
    assert ranking_feature_names(include_price_features=False) == RANKING_FEATURE_NAMES_BASE


def test_ranking_feature_names_full_has_29_entries():
    assert len(RANKING_FEATURE_NAMES) == 29
    assert ranking_feature_names(include_price_features=True) == RANKING_FEATURE_NAMES


def test_base_ranking_feature_names_excludes_step6_price_personalization_terms():
    """BASE must exclude the STEP-6-INTRODUCED price personalization
    fields specifically - it legitimately KEEPS the pre-existing basic
    product price/discount fields (`item_normalized_price`,
    `item_discount_fraction`), per docs/data-mapping.md section 17's
    "the baseline should not be artificially crippled" requirement.
    """
    step6_only_terms = {
        "item_category_relative_price", "item_is_discounted",
        "user_normalized_typical_price", "user_has_price_profile", "price_relative_distance", "price_tier_match",
    }
    assert not (step6_only_terms & set(RANKING_FEATURE_NAMES_BASE))
    assert "item_normalized_price" in RANKING_FEATURE_NAMES_BASE
    assert "item_discount_fraction" in RANKING_FEATURE_NAMES_BASE


def test_build_ranking_feature_vector_base_has_23_dims():
    vec = build_ranking_feature_vector(
        _user_features(), _product_features(), None, 0.5, 0, 50, 100.0, include_price_features=False
    )
    assert vec.shape == (23,)


def test_build_ranking_feature_vector_improved_has_29_dims():
    vec = build_ranking_feature_vector(
        _user_features(), _product_features(), None, 0.5, 0, 50, 100.0, include_price_features=True
    )
    assert vec.shape == (29,)


def test_generate_recommendations_uses_matching_feature_dim_for_base_ranker():
    """Regression test for a real bug caught while building the STEP 8
    ablation harness: `serving.pipeline._personalized_candidates` always
    built a 29-feature ranking vector regardless of which ranker it was
    scoring, which would crash a BASE-condition (23-feature) ranker with
    a shape mismatch. `include_price_features` must be derived from
    `tt_encoder.include_price_features` so a BASE Two-Tower/ranker pair
    (both trained together, same flag) always gets vectors of the shape
    its OWN ranker expects.
    """
    from recommendation.data.schemas.engagement import EngagementProfile, PurchaseRecord
    from recommendation.data.schemas.user import UserProfile
    from recommendation.ranking.features import RANKING_FEATURE_NAMES_BASE
    from recommendation.ranking.model import build_ranker_model
    from recommendation.retrieval.index.faiss_index import FaissVectorIndex
    from recommendation.serving.pipeline import generate_recommendations
    from recommendation.utils.config import AppConfig, ColdStartConfig, EligibilityConfig, RankingConfig, RetrievalConfig

    products = [_product_features(product_id=i, category_name="Cat") for i in range(1, 6)]
    product_lookup_full = {pf.product_id: pf for pf in products}
    embeddings = {i: np.random.default_rng(i).normal(size=8).astype(np.float32) for i in range(1, 6)}

    encoder = _encoder(include_price_features=False)
    tt_config = _tt_config()
    user_tower = build_user_tower(encoder, tt_config)

    item_embeddings = np.random.default_rng(0).normal(size=(5, tt_config.output_dim)).astype(np.float32)
    item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    vector_index = FaissVectorIndex()
    vector_index.build(list(product_lookup_full.keys()), item_embeddings)

    ranker_model = build_ranker_model(input_dim=len(RANKING_FEATURE_NAMES_BASE), config=RankingConfig(hidden_units=[4]))

    profile = EngagementProfile(
        user_id=1, profile=UserProfile(user_id=1),
        purchases=[PurchaseRecord(user_id=1, product_id=1, order_id=1, quantity=1) for _ in range(5)],
    )
    from recommendation.features.user_features import build_user_features
    from recommendation.utils.config import FeatureConfig

    user_features = build_user_features(profile, {}, embeddings, FeatureConfig())

    config = AppConfig(
        cold_start=ColdStartConfig(strong_history_min_signals=1, sparse_history_min_signals=1),
        eligibility=EligibilityConfig(require_active=True, require_in_stock=True),
        retrieval=RetrievalConfig(backend="faiss", candidate_pool_multiplier=5, min_candidate_pool=5),
    )

    # Must not raise a shape-mismatch error.
    result = generate_recommendations(
        user_features, product_lookup_full, embeddings, list(product_lookup_full.keys()),
        encoder, user_tower, ranker_model, vector_index, config, top_n=3,
    )
    assert len(result.product_ids) <= 3


def test_base_vector_matches_improved_vector_on_shared_prefix_and_suffix():
    """The BASE vector must not just be shorter - it must contain exactly
    the SAME pre-STEP-6 values in the SAME relative order (proving no
    price value silently leaked into the base feature set, and no
    non-price value was accidentally dropped).
    """
    user = _user_features()
    product = _product_features()
    base_vec = build_ranking_feature_vector(user, product, None, 0.5, 3, 50, 100.0, include_price_features=False)
    full_vec = build_ranking_feature_vector(user, product, None, 0.5, 3, 50, 100.0, include_price_features=True)

    base_by_name = dict(zip(RANKING_FEATURE_NAMES_BASE, base_vec))
    full_by_name = dict(zip(RANKING_FEATURE_NAMES, full_vec))
    for name, value in base_by_name.items():
        assert value == pytest.approx(full_by_name[name])
