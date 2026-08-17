import numpy as np
import pytest

from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import UserFeatures
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder, Vocabulary


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
    return TwoTowerFeatureEncoder.fit(
        category_names=["Dairy & Eggs", "Snacks", "Produce"],
        brand_names=["GreenValley", "SnackWorks"],
        age_groups=["25-34", "35-44"],
        prices=[2.0, 4.0, 10.0],
        embedding_dim=embedding_dim,
    )


def _product_features(**overrides) -> ProductFeatures:
    defaults = dict(
        product_id=1, category_id=1, category_name="Dairy & Eggs", parent_category_name=None, brand="GreenValley",
        tags=["healthy"], price=4.0, effective_price=4.0, discount_percentage=0.0, is_active=True, stock_quantity=10,
        purchase_count=3, distinct_purchasers=2, cart_add_count=1, review_count=2, average_rating=4.5,
    )
    defaults.update(overrides)
    return ProductFeatures(**defaults)


def _user_features(**overrides) -> UserFeatures:
    defaults = dict(
        user_id=1, preferred_category="Snacks", age_group="25-34", has_preferred_category=True, has_age_group=True,
        click_count=0, purchase_count=3, distinct_products_purchased=3, cart_item_count=1, search_count=2, has_chatbot_context=False,
        total_engagement_events=6, category_affinity={"Snacks": 0.7, "Dairy & Eggs": 0.3}, brand_affinity={"SnackWorks": 1.0},
        semantic_embedding=np.ones(8, dtype=np.float32),
    )
    defaults.update(overrides)
    return UserFeatures(**defaults)


def test_encode_item_shapes():
    encoder = _encoder(embedding_dim=8)
    result = encoder.encode_item(_product_features(), np.ones(8, dtype=np.float32))
    assert result["semantic_embedding"].shape == (8,)
    assert result["numeric"].shape == (encoder.item_numeric_dim,)
    assert result["category_id"].dtype == np.int32
    assert result["brand_id"].dtype == np.int32


def test_encode_item_price_normalized_by_catalog_max():
    encoder = _encoder(embedding_dim=8)  # max price fit at 10.0
    result = encoder.encode_item(_product_features(effective_price=5.0), np.ones(8, dtype=np.float32))
    assert result["numeric"][0] == pytest.approx(0.5)


def test_encode_item_missing_rating_produces_zero_with_flag():
    encoder = _encoder(embedding_dim=8)
    result = encoder.encode_item(_product_features(average_rating=None, review_count=0), np.ones(8, dtype=np.float32))
    assert result["numeric"][5] == 0.0  # average_rating slot
    assert result["numeric"][6] == 0.0  # has_rating flag


def test_encode_user_shapes():
    encoder = _encoder(embedding_dim=8)
    result = encoder.encode_user(_user_features())
    assert result["semantic_embedding"].shape == (8,)
    assert result["category_affinity"].shape == (encoder.category_affinity_dim,)
    assert result["brand_affinity"].shape == (encoder.brand_affinity_dim,)
    assert result["numeric"].shape == (encoder.user_numeric_dim,)


def test_encode_user_missing_semantic_embedding_becomes_zero_vector_with_flag():
    encoder = _encoder(embedding_dim=8)
    result = encoder.encode_user(_user_features(semantic_embedding=None))
    assert np.allclose(result["semantic_embedding"], np.zeros(8))
    assert result["numeric"][-1] == 0.0  # has_semantic_embedding flag


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
    assert loaded.brand_vocab.values == encoder.brand_vocab.values
    assert loaded.age_group_vocab.values == encoder.age_group_vocab.values
    # Encoding behavior is identical after round-trip.
    assert loaded.category_vocab.encode("Snacks") == encoder.category_vocab.encode("Snacks")
