"""Tests for the current, always-on price-aware feature set (docs/data-
mapping.md section 15): the Two-Tower encoder's price-related numeric
fields + shared `price_tier_id` categorical input, and the ranker's
price-related entries in `RANKING_FEATURE_NAMES`.

Formerly `test_step8_ablation_dimensions.py`, which also covered the
STEP 8 `include_price_features=False`/`RANKING_FEATURE_NAMES_BASE`
reduced-feature path built for the (since-removed) controlled ablation
experiment. That reduced path no longer exists in the codebase - the
current architecture always includes price features - so this file keeps
only the tests describing that current, always-on behavior.

Dimensions/feature counts below reflect the production-safe contract
redesign (docs/production-feature-parity-audit.md): no brand/age-group
inputs, no discount-related features (the real SQL Server schema has
none of those fields).
"""

from __future__ import annotations

import numpy as np

from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import UserFeatures
from recommendation.ranking.features import RANKING_FEATURE_NAMES, build_ranking_feature_vector
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.model import build_item_tower, build_user_tower
from recommendation.config import TwoTowerConfig

PRICE_RANKING_FEATURE_NAMES = [
    "item_normalized_price",
    "item_category_relative_price",
    "user_normalized_typical_price",
    "user_has_price_profile",
    "price_relative_distance",
    "price_tier_match",
]


def _encoder(embedding_dim: int = 8) -> TwoTowerFeatureEncoder:
    # Production-safe contract: no brand_names/age_groups.
    return TwoTowerFeatureEncoder.fit(
        category_names=["Dairy & Eggs", "Snacks"],
        prices=[2.0, 4.0],
        embedding_dim=embedding_dim,
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
        user_id=1, preferred_categories=["Snacks"], age_group="25-34", has_preferred_category=True, has_age_group=True,
        click_count=0, purchase_count=3, distinct_products_purchased=3, cart_item_count=1, search_count=2, has_chatbot_context=False,
        total_engagement_events=6, category_affinity={}, brand_affinity={}, semantic_embedding=np.ones(8, dtype=np.float32),
        price_profile=None,
    )
    defaults.update(overrides)
    return UserFeatures(**defaults)


# --- Two-Tower encoder dimensions (always price-aware) ----------------------

def test_encoder_has_current_price_aware_dimensions():
    encoder = _encoder()
    assert encoder.item_numeric_dim == 5
    assert encoder.user_numeric_dim == 8


def test_encode_item_has_price_tier_id_key():
    encoder = _encoder()
    result = encoder.encode_item(_product_features(), np.ones(8, dtype=np.float32))
    assert "price_tier_id" in result
    assert result["numeric"].shape == (5,)


def test_encode_user_has_price_tier_id_key():
    encoder = _encoder()
    result = encoder.encode_user(_user_features())
    assert "price_tier_id" in result
    assert result["numeric"].shape == (8,)


def test_encoder_serialization_round_trips_include_price_features_field():
    """`include_price_features` is reported metadata (`evaluation
    .offline_report`, `GET /v1/metrics/offline`, the dashboard) - it must
    still round-trip through to_dict/from_dict even though it no longer
    controls encoding shape.
    """
    encoder = _encoder()
    restored = TwoTowerFeatureEncoder.from_dict(encoder.to_dict())
    assert restored.include_price_features is True
    assert restored.item_numeric_dim == encoder.item_numeric_dim == 5


def test_legacy_serialized_dict_without_flag_defaults_to_true():
    """A serialized dict predating this field being added to
    serialization has no `include_price_features` key at all - protects
    that historical compatibility path.
    """
    encoder = _encoder()
    data = encoder.to_dict()
    del data["include_price_features"]
    restored = TwoTowerFeatureEncoder.from_dict(data)
    assert restored.include_price_features is True


# --- Two-Tower model architecture (always includes price_tier_id) -----------

def _tt_config() -> TwoTowerConfig:
    return TwoTowerConfig(projection_dims=[16, 8], output_dim=8, category_embedding_dim=4)


def test_item_tower_has_price_tier_input():
    encoder = _encoder()
    tower = build_item_tower(encoder, _tt_config())
    assert "price_tier_id" in {t.name.split(":")[0] for t in tower.inputs}


def test_user_tower_has_price_tier_input():
    encoder = _encoder()
    tower = build_user_tower(encoder, _tt_config())
    assert "price_tier_id" in {t.name.split(":")[0] for t in tower.inputs}


# --- ranker feature vector (always 22 features, always price-aware) ---------

def test_ranking_feature_names_has_22_entries():
    assert len(RANKING_FEATURE_NAMES) == 22


def test_ranking_feature_names_includes_all_price_related_features():
    assert set(PRICE_RANKING_FEATURE_NAMES) <= set(RANKING_FEATURE_NAMES)


def test_ranking_feature_names_excludes_discount_features():
    """No real SalePrice/DiscountPercentage in production - see
    docs/production-feature-parity-audit.md.
    """
    assert "item_discount_fraction" not in RANKING_FEATURE_NAMES
    assert "item_is_discounted" not in RANKING_FEATURE_NAMES


def test_build_ranking_feature_vector_has_22_dims():
    vec = build_ranking_feature_vector(_user_features(), _product_features(), None, 0.5, 0, 50, 100.0)
    assert vec.shape == (22,)
