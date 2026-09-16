"""`production_safe_v2` (docs/data-mapping.md 19.15): `log_purchase_count`/
`log_cart_add_count` removed from BOTH the Two-Tower item tower and the
ranker - the train-serve parity audit found these were genuine learned-
model inputs computed from training's COMPLETE SQLite history, while
live `backend_api` serving can only supply a bounded recent-window
approximation (no aggregate endpoint, no delta filter exists on the real
backend). Product purchase/cart popularity remains available as a
SERVING-ONLY fallback heuristic (`serving.fallback`), never again as a
model input.
"""

from __future__ import annotations

import numpy as np
import pytest

from recommendation.features.product_features import ProductFeatures
from recommendation.ranking.features import RANKING_FEATURE_NAMES, build_ranking_feature_vector
from recommendation.retrieval.two_tower.feature_encoding import (
    CURRENT_CONTRACT_VERSION,
    ITEM_NUMERIC_FEATURE_NAMES,
    TwoTowerFeatureEncoder,
)
from recommendation.serving.fallback import global_popularity_ranking
from recommendation.ranking.serialization import RankerArtifacts
from recommendation.retrieval.two_tower.serialization import TwoTowerArtifacts
from recommendation.serving.startup_validation import ArtifactValidationError, validate_ranker_artifacts, validate_two_tower_artifacts
from recommendation.config import AppConfig


def _product_features(**overrides) -> ProductFeatures:
    defaults = dict(
        product_id=1, category_id=1, category_name="Fruits", parent_category_name=None, brand=None,
        tags=[], price=4.0, effective_price=4.0, discount_percentage=0.0, is_active=True, stock_quantity=10,
        purchase_count=999, distinct_purchasers=1, cart_add_count=999, review_count=2, average_rating=4.5,
        is_discounted=False, price_tier="mid", category_relative_price=0.5,
    )
    defaults.update(overrides)
    return ProductFeatures(**defaults)


def _encoder() -> TwoTowerFeatureEncoder:
    return TwoTowerFeatureEncoder.fit(category_names=["Fruits", "Packages"], prices=[4.0, 10.0], embedding_dim=8)


# --- contract: neither feature is a learned input any more ------------------


def test_contract_version_bumped_to_v2():
    assert CURRENT_CONTRACT_VERSION == "production_safe_v2"


def test_item_purchase_count_is_not_a_two_tower_input():
    assert "log_purchase_count" not in ITEM_NUMERIC_FEATURE_NAMES


def test_item_cart_add_count_is_not_a_two_tower_input():
    assert "log_cart_add_count" not in ITEM_NUMERIC_FEATURE_NAMES


def test_item_numeric_dim_is_five():
    assert len(ITEM_NUMERIC_FEATURE_NAMES) == 5


def test_neither_feature_appears_in_ranker_feature_names():
    assert "item_log_purchase_count" not in RANKING_FEATURE_NAMES
    assert "item_log_cart_add_count" not in RANKING_FEATURE_NAMES


def test_ranker_feature_count_is_22():
    assert len(RANKING_FEATURE_NAMES) == 22


def test_encode_item_output_does_not_change_with_purchase_or_cart_volume():
    """A HUGE purchase/cart count must produce the EXACT SAME encoded
    numeric vector as a zero count - proof the model literally cannot see
    this signal any more, not just that its slot was renamed/hidden.
    """
    encoder = _encoder()
    embedding = np.ones(8, dtype=np.float32)
    low = encoder.encode_item(_product_features(purchase_count=0, cart_add_count=0), embedding)
    high = encoder.encode_item(_product_features(purchase_count=1_000_000, cart_add_count=1_000_000), embedding)
    assert np.array_equal(low["numeric"], high["numeric"])


def test_ranking_feature_vector_does_not_change_with_purchase_or_cart_volume():
    from recommendation.features.user_features import UserFeatures

    user = UserFeatures(
        user_id=1, preferred_categories=[], age_group=None, has_preferred_category=False, has_age_group=False,
        click_count=0, purchase_count=0, distinct_products_purchased=0, cart_item_count=0, search_count=0,
        has_chatbot_context=False, total_engagement_events=0,
    )
    low = build_ranking_feature_vector(user, _product_features(purchase_count=0, cart_add_count=0), None, 0.5, 0, 50, 100.0)
    high = build_ranking_feature_vector(user, _product_features(purchase_count=1_000_000, cart_add_count=1_000_000), None, 0.5, 0, 50, 100.0)
    assert np.array_equal(low, high)


# --- serving-heuristic use is preserved (bounded window is fine there) -----


def test_product_features_still_carries_purchase_and_cart_counts_for_fallback_use():
    """The fields themselves are NOT removed from `ProductFeatures` - only
    from the two learned-feature builders. `serving.fallback` still needs
    them.
    """
    pf = _product_features(purchase_count=7, cart_add_count=3)
    assert pf.purchase_count == 7
    assert pf.cart_add_count == 3


def test_global_popularity_fallback_still_uses_the_bounded_counts():
    features = {
        1: _product_features(product_id=1, purchase_count=5, cart_add_count=1),
        2: _product_features(product_id=2, purchase_count=10, cart_add_count=0),
    }
    ranked = global_popularity_ranking(features)
    assert ranked == [2, 1], "fallback heuristic ranking by popularity is unaffected by the model-input removal"


# --- old artifacts rejected, new ones validate -------------------------------


def test_old_v1_contract_two_tower_artifacts_are_rejected():
    encoder = _encoder()
    encoder.contract_version = "production_safe_v1"
    artifacts = TwoTowerArtifacts(
        encoder=encoder, user_tower=None, item_tower=None,
        item_ids=[1, 2], item_embeddings=np.zeros((2, 8), dtype=np.float32), metadata={},
    )
    with pytest.raises(ArtifactValidationError, match="production_safe_v1"):
        validate_two_tower_artifacts(artifacts, AppConfig())


def test_old_24_feature_ranker_artifacts_are_rejected():
    old_names = RANKING_FEATURE_NAMES[:6] + ["item_log_purchase_count", "item_log_cart_add_count"] + RANKING_FEATURE_NAMES[6:]

    class _FakeModel:
        input_shape = (None, len(old_names))

    artifacts = RankerArtifacts(model=_FakeModel(), feature_names=old_names, metadata={})
    with pytest.raises(ArtifactValidationError):
        validate_ranker_artifacts(artifacts)


def test_current_contract_ranker_artifacts_validate():
    class _FakeModel:
        input_shape = (None, len(RANKING_FEATURE_NAMES))

    artifacts = RankerArtifacts(model=_FakeModel(), feature_names=list(RANKING_FEATURE_NAMES), metadata={})
    validate_ranker_artifacts(artifacts)  # must not raise
