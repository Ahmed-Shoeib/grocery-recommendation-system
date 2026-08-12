import numpy as np
import pytest
import tensorflow as tf

from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.model import L2Normalize, TwoTowerModel, build_item_tower, build_user_tower
from recommendation.utils.config import TwoTowerConfig


@pytest.fixture
def encoder() -> TwoTowerFeatureEncoder:
    return TwoTowerFeatureEncoder.fit(
        category_names=["Dairy & Eggs", "Snacks", "Produce"],
        brand_names=["GreenValley", "SnackWorks"],
        age_groups=["25-34", "35-44"],
        prices=[2.0, 4.0, 10.0],
        embedding_dim=8,
    )


@pytest.fixture
def config() -> TwoTowerConfig:
    return TwoTowerConfig(projection_dims=[16, 8], output_dim=8, category_embedding_dim=4, brand_embedding_dim=4, age_group_embedding_dim=2)


def _item_batch(encoder: TwoTowerFeatureEncoder, n: int) -> dict[str, np.ndarray]:
    return {
        "semantic_embedding": np.random.default_rng(0).normal(size=(n, encoder.embedding_dim)).astype(np.float32),
        "category_id": np.zeros(n, dtype=np.int32),
        "brand_id": np.zeros(n, dtype=np.int32),
        "numeric": np.random.default_rng(1).normal(size=(n, encoder.item_numeric_dim)).astype(np.float32),
    }


def _user_batch(encoder: TwoTowerFeatureEncoder, n: int) -> dict[str, np.ndarray]:
    return {
        "semantic_embedding": np.random.default_rng(2).normal(size=(n, encoder.embedding_dim)).astype(np.float32),
        "preferred_category_id": np.zeros(n, dtype=np.int32),
        "age_group_id": np.zeros(n, dtype=np.int32),
        "category_affinity": np.random.default_rng(3).random((n, encoder.category_affinity_dim)).astype(np.float32),
        "brand_affinity": np.random.default_rng(4).random((n, encoder.brand_affinity_dim)).astype(np.float32),
        "numeric": np.random.default_rng(5).normal(size=(n, encoder.user_numeric_dim)).astype(np.float32),
    }


def test_item_tower_output_shape_and_l2_norm(encoder, config):
    tower = build_item_tower(encoder, config)
    output = tower(_item_batch(encoder, 5), training=False)
    assert output.shape == (5, config.output_dim)
    norms = np.linalg.norm(output.numpy(), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_user_tower_output_shape_and_l2_norm(encoder, config):
    tower = build_user_tower(encoder, config)
    output = tower(_user_batch(encoder, 5), training=False)
    assert output.shape == (5, config.output_dim)
    norms = np.linalg.norm(output.numpy(), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_l2_normalize_layer_is_registered_serializable():
    layer = L2Normalize()
    x = tf.constant([[3.0, 4.0]])  # norm 5
    out = layer(x).numpy()
    assert np.allclose(out, [[0.6, 0.8]])


def test_two_tower_model_train_step_reduces_loss(encoder, config):
    tf.random.set_seed(0)
    user_tower = build_user_tower(encoder, config)
    item_tower = build_item_tower(encoder, config)
    model = TwoTowerModel(user_tower=user_tower, item_tower=item_tower, temperature=0.1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

    n = 16
    user_batch = _user_batch(encoder, n)
    item_batch = _item_batch(encoder, n)
    ds = tf.data.Dataset.from_tensor_slices((user_batch, item_batch)).batch(n)

    history = model.fit(ds, epochs=10, verbose=0)
    losses = history.history["loss"]
    assert losses[-1] < losses[0]


def test_two_tower_model_cosine_similarity_matches_manual_dot_product(encoder, config):
    user_tower = build_user_tower(encoder, config)
    item_tower = build_item_tower(encoder, config)
    model = TwoTowerModel(user_tower=user_tower, item_tower=item_tower, temperature=0.05)

    user_batch = _user_batch(encoder, 3)
    item_batch = _item_batch(encoder, 3)
    user_emb, item_emb = model((user_batch, item_batch), training=False)

    manual_logits = np.dot(user_emb.numpy(), item_emb.numpy().T)
    # Cosine similarity of L2-normalized vectors == plain dot product, in [-1, 1].
    assert manual_logits.max() <= 1.0 + 1e-5
    assert manual_logits.min() >= -1.0 - 1e-5
