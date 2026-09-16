import numpy as np
import pytest

from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import UserFeatures
from recommendation.retrieval.two_tower.feature_encoding import (
    CURRENT_CONTRACT_VERSION,
    TwoTowerFeatureEncoder,
    Vocabulary,
)


def test_vocabulary_unknown_is_always_index_zero():
    vocab = Vocabulary.fit(["Snacks", "Dairy & Eggs"])
    assert vocab.encode(None) == 0
    assert vocab.encode("Never Seen Category") == 0


def test_vocabulary_known_values_get_distinct_nonzero_indices():
    vocab = Vocabulary.fit(["Snacks", "Dairy & Eggs", "Produce"])
    indices = {vocab.encode(v) for v in ["Snacks", "Dairy & Eggs", "Produce"]}
    assert len(indices) == 3
    assert 0 not in indices


def test_vocabulary_size_includes_unknown_bucket():
    vocab = Vocabulary.fit(["a", "b", "c"])
    assert vocab.size == 4  # 3 known + 1 unknown


def test_vocabulary_fit_deduplicates_and_ignores_empty():
    vocab = Vocabulary.fit(["a", "a", "b", "", None])
    assert vocab.size == 3  # "a", "b" + unknown


def test_vocabulary_round_trips_through_dict():
    vocab = Vocabulary.fit(["z", "a", "m"])
    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.values == vocab.values
    assert restored.encode("z") == vocab.encode("z")


def _encoder(embedding_dim: int = 8) -> TwoTowerFeatureEncoder:
    # Production-safe contract (docs/production-feature-parity-audit.md):
    # no brand_names/age_groups - the real backend has neither field.
    return TwoTowerFeatureEncoder.fit(
        category_names=["Dairy & Eggs", "Snacks", "Produce"],
        prices=[2.0, 4.0, 10.0],
        embedding_dim=embedding_dim,
    )


def _product_features(**overrides) -> ProductFeatures:
    defaults = dict(
        product_id=1, category_id=1, category_name="Dairy & Eggs", parent_category_name=None, brand="GreenValley",
        tags=["healthy"], price=4.0, effective_price=4.0, discount_percentage=0.0, is_active=True, stock_quantity=10,
        purchase_count=3, distinct_purchasers=2, cart_add_count=1, review_count=2, average_rating=4.5,
        is_discounted=False, price_tier="mid", category_relative_price=0.5,
    )
    defaults.update(overrides)
    return ProductFeatures(**defaults)


def _user_features(**overrides) -> UserFeatures:
    defaults = dict(
        user_id=1, preferred_categories=["Snacks"], age_group="25-34", has_preferred_category=True, has_age_group=True,
        click_count=0, purchase_count=3, distinct_products_purchased=3, cart_item_count=1, search_count=2, has_chatbot_context=False,
        total_engagement_events=6, category_affinity={"Snacks": 0.7, "Dairy & Eggs": 0.3}, brand_affinity={"SnackWorks": 1.0},
        semantic_embedding=np.ones(8, dtype=np.float32), price_profile=None,
    )
    defaults.update(overrides)
    return UserFeatures(**defaults)


def test_encode_item_shapes():
    encoder = _encoder(embedding_dim=8)
    result = encoder.encode_item(_product_features(), np.ones(8, dtype=np.float32))
    assert result["semantic_embedding"].shape == (8,)
    assert result["numeric"].shape == (encoder.item_numeric_dim,)
    assert result["category_id"].dtype == np.int32
    assert "brand_id" not in result  # production-safe contract: no real Product.Brand


def test_encode_item_price_normalized_by_catalog_max():
    encoder = _encoder(embedding_dim=8)  # max price fit at 10.0
    result = encoder.encode_item(_product_features(effective_price=5.0), np.ones(8, dtype=np.float32))
    assert result["numeric"][0] == pytest.approx(0.5)


def test_encode_item_missing_rating_produces_zero_with_flag():
    encoder = _encoder(embedding_dim=8)
    result = encoder.encode_item(_product_features(average_rating=None, review_count=0), np.ones(8, dtype=np.float32))
    assert result["numeric"][2] == 0.0  # average_rating slot
    assert result["numeric"][3] == 0.0  # has_rating flag


def test_encode_user_shapes():
    encoder = _encoder(embedding_dim=8)
    result = encoder.encode_user(_user_features())
    assert result["semantic_embedding"].shape == (8,)
    assert result["category_affinity"].shape == (encoder.category_affinity_dim,)
    assert "brand_affinity" not in result  # production-safe contract: no real Product.Brand
    assert "preferred_category_id" not in result  # folded into category_affinity instead
    assert "age_group_id" not in result  # production-safe contract: no real User.AgeGroup
    assert result["numeric"].shape == (encoder.user_numeric_dim,)


def test_encode_user_missing_semantic_embedding_becomes_zero_vector_with_flag():
    encoder = _encoder(embedding_dim=8)
    result = encoder.encode_user(_user_features(semantic_embedding=None))
    assert np.allclose(result["semantic_embedding"], np.zeros(8))
    assert result["numeric"][-2] == 0.0  # has_semantic_embedding flag (normalized_typical_price is last)


def test_encode_user_category_affinity_maps_to_correct_vocab_slot():
    encoder = _encoder(embedding_dim=8)
    result = encoder.encode_user(_user_features(category_affinity={"Snacks": 1.0}))
    snacks_idx = encoder.category_vocab.encode("Snacks") - 1
    assert result["category_affinity"][snacks_idx] == pytest.approx(1.0)
    assert result["category_affinity"].sum() == pytest.approx(1.0)


def test_encode_user_batch_stacks_correctly():
    encoder = _encoder(embedding_dim=8)
    batch = encoder.encode_user_batch([_user_features(user_id=1), _user_features(user_id=2)])
    assert batch["semantic_embedding"].shape == (2, 8)
    assert batch["category_affinity"].shape == (2, encoder.category_affinity_dim)


def test_encode_item_batch_stacks_correctly():
    encoder = _encoder(embedding_dim=8)
    features = {1: _product_features(product_id=1), 2: _product_features(product_id=2)}
    embeddings = {1: np.ones(8, dtype=np.float32), 2: np.zeros(8, dtype=np.float32)}
    batch = encoder.encode_item_batch([1, 2], features, embeddings)
    assert batch["semantic_embedding"].shape == (2, 8)


def test_encoder_save_and_load_round_trips(tmp_path):
    encoder = _encoder(embedding_dim=8)
    path = tmp_path / "encoder.json"
    encoder.save(path)
    loaded = TwoTowerFeatureEncoder.load(path)

    assert loaded.embedding_dim == encoder.embedding_dim
    assert loaded.max_price == encoder.max_price
    assert loaded.category_vocab.values == encoder.category_vocab.values
    assert loaded.price_tier_vocab.values == encoder.price_tier_vocab.values
    assert loaded.contract_version == encoder.contract_version == CURRENT_CONTRACT_VERSION
    # Encoding behavior is identical after round-trip.
    assert loaded.category_vocab.encode("Snacks") == encoder.category_vocab.encode("Snacks")


def test_loading_pre_step6_encoder_dict_without_price_tier_vocab_falls_back_to_fixed_vocab():
    """A JSON dict saved by a PRE-STEP-6 encoder (no `price_tier_vocab`
    key at all) must still load, degrading to the same fixed PRICE_TIERS
    vocabulary `fit()` always produces - see `TwoTowerFeatureEncoder
    .from_dict`'s docstring.
    """
    encoder = _encoder(embedding_dim=8)
    legacy_dict = encoder.to_dict()
    del legacy_dict["price_tier_vocab"]
    loaded = TwoTowerFeatureEncoder.from_dict(legacy_dict)
    assert loaded.price_tier_vocab.values == ["budget", "mid", "premium"]


def test_loading_pre_redesign_encoder_dict_is_stamped_legacy_contract_version():
    """A JSON dict saved before the production-safe contract redesign has
    no `contract_version` key (and still carries now-ignored `brand_vocab`/
    `age_group_vocab` keys) - it must load without error, but be stamped
    with the legacy marker so `serving.startup_validation` can reject it
    explicitly rather than this loader silently dropping brand/age-group
    data on the floor.
    """
    legacy_dict = {
        "embedding_dim": 8,
        "max_price": 10.0,
        "category_vocab": {"values": ["Dairy & Eggs", "Snacks"]},
        "brand_vocab": {"values": ["GreenValley"]},
        "age_group_vocab": {"values": ["25-34"]},
        "price_tier_vocab": {"values": ["budget", "mid", "premium"]},
        "include_price_features": True,
    }
    loaded = TwoTowerFeatureEncoder.from_dict(legacy_dict)
    assert loaded.contract_version != CURRENT_CONTRACT_VERSION


# --- production-safe item/user numeric dimensions -------------------------

def test_item_numeric_dim_is_five_after_train_serve_parity_fix():
    encoder = _encoder(embedding_dim=8)
    # normalized_price, log_review_count, average_rating, has_rating,
    # category_relative_price (discount_fraction/is_discounted removed -
    # no real SalePrice/DiscountPercentage in production;
    # log_purchase_count/log_cart_add_count removed in production_safe_v2
    # - docs/data-mapping.md 19.15 - the real backend cannot reproduce a
    # true lifetime aggregate for these efficiently as a learned input).
    assert encoder.item_numeric_dim == 5


def test_user_numeric_dim_is_eight_after_production_safe_redesign():
    encoder = _encoder(embedding_dim=8)
    # log_purchase_count, log_cart_item_count, log_search_count,
    # log_total_engagement_events, has_chatbot_context,
    # has_preferred_category, has_semantic_embedding,
    # normalized_typical_price (has_age_group removed - no real AgeGroup
    # in production).
    assert encoder.user_numeric_dim == 8


def test_encode_item_includes_price_tier_id_and_relative_price():
    encoder = _encoder(embedding_dim=8)
    result = encoder.encode_item(
        _product_features(price_tier="premium", category_relative_price=0.9),
        np.ones(8, dtype=np.float32),
    )
    assert result["price_tier_id"].dtype == np.int32
    assert result["price_tier_id"] == encoder.price_tier_vocab.encode("premium")
    assert result["numeric"][4] == pytest.approx(0.9)  # category_relative_price (last slot)


def test_encode_item_unknown_price_tier_falls_back_to_zero_bucket():
    encoder = _encoder(embedding_dim=8)
    result = encoder.encode_item(_product_features(price_tier="not_a_real_tier"), np.ones(8, dtype=np.float32))
    assert result["price_tier_id"] == 0


def test_encode_user_no_price_profile_gets_unknown_tier_and_zero_normalized_price():
    encoder = _encoder(embedding_dim=8)  # max_price fit at 10.0
    result = encoder.encode_user(_user_features(price_profile=None))
    assert result["price_tier_id"] == 0  # unknown bucket
    assert result["numeric"][-1] == 0.0  # normalized_typical_price


def test_encode_user_with_price_profile_sets_tier_and_normalized_typical_price():
    from recommendation.features.price import UserPriceProfile

    encoder = _encoder(embedding_dim=8)  # max_price fit at 10.0
    profile = UserPriceProfile(
        typical_price=5.0, price_spread=1.0, price_tier="mid", supporting_purchase_count=3, fallback_source="purchase_history"
    )
    result = encoder.encode_user(_user_features(price_profile=profile))
    assert result["price_tier_id"] == encoder.price_tier_vocab.encode("mid")
    assert result["numeric"][-1] == pytest.approx(0.5)  # 5.0 / 10.0


# --- multi-favorite preferred_categories folded into category_affinity ----

def test_encode_user_preferred_categories_are_not_a_separate_input():
    """`UserFeatures.preferred_categories` (a list) has no dedicated
    Two-Tower input any more - its signal is already folded into
    `category_affinity` upstream (`features.user_features
    .build_user_features`), so the encoder output for two users with
    identical `category_affinity` but different `preferred_categories`
    lists must be identical.
    """
    encoder = _encoder(embedding_dim=8)
    one_favorite = encoder.encode_user(_user_features(preferred_categories=["Snacks"]))
    many_favorites = encoder.encode_user(_user_features(preferred_categories=["Snacks", "Produce", "Dairy & Eggs"]))
    assert np.array_equal(one_favorite["category_affinity"], many_favorites["category_affinity"])
    assert np.array_equal(one_favorite["numeric"], many_favorites["numeric"])
