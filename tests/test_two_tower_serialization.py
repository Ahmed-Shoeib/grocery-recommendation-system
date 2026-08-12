import numpy as np
import pytest

from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.model import build_item_tower, build_user_tower
from recommendation.retrieval.two_tower.serialization import load_two_tower_artifacts, save_two_tower_artifacts
from recommendation.utils.config import TwoTowerConfig


@pytest.fixture
def encoder() -> TwoTowerFeatureEncoder:
    return TwoTowerFeatureEncoder.fit(
        category_names=["Dairy & Eggs", "Snacks"],
        brand_names=["GreenValley"],
        age_groups=["25-34"],
        prices=[2.0, 4.0],
        embedding_dim=8,
    )


@pytest.fixture
def config() -> TwoTowerConfig:
    return TwoTowerConfig(projection_dims=[16, 8], output_dim=8, category_embedding_dim=4, brand_embedding_dim=4, age_group_embedding_dim=2)


def test_save_and_load_round_trips_item_tower_predictions(tmp_path, encoder, config):
    user_tower = build_user_tower(encoder, config)
    item_tower = build_item_tower(encoder, config)

    item_batch = {
        "semantic_embedding": np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32),
        "category_id": np.array([0, 1, 1, 0], dtype=np.int32),
        "brand_id": np.array([0, 0, 1, 1], dtype=np.int32),
        "numeric": np.random.default_rng(1).normal(size=(4, encoder.item_numeric_dim)).astype(np.float32),
    }
    original_output = item_tower.predict(item_batch, verbose=0)

    out_dir = tmp_path / "two_tower"
    save_two_tower_artifacts(
        out_dir, user_tower, item_tower, encoder,
        item_ids=[1, 2, 3, 4], item_embeddings=original_output, metadata={"model_version": "test"},
    )

    artifacts = load_two_tower_artifacts(out_dir)
    reloaded_output = artifacts.item_tower.predict(item_batch, verbose=0)

    assert np.allclose(original_output, reloaded_output, atol=1e-6)


def test_save_and_load_preserves_encoder_vocab(tmp_path, encoder, config):
    user_tower = build_user_tower(encoder, config)
    item_tower = build_item_tower(encoder, config)
    out_dir = tmp_path / "two_tower"
    save_two_tower_artifacts(
        out_dir, user_tower, item_tower, encoder,
        item_ids=[1], item_embeddings=np.zeros((1, 8), dtype=np.float32), metadata={},
    )
    artifacts = load_two_tower_artifacts(out_dir)
    assert artifacts.encoder.category_vocab.values == encoder.category_vocab.values
    assert artifacts.encoder.brand_vocab.values == encoder.brand_vocab.values


def test_save_and_load_preserves_item_embeddings_and_ids(tmp_path, encoder, config):
    user_tower = build_user_tower(encoder, config)
    item_tower = build_item_tower(encoder, config)
    out_dir = tmp_path / "two_tower"
    embeddings = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    save_two_tower_artifacts(
        out_dir, user_tower, item_tower, encoder, item_ids=[10, 20, 30], item_embeddings=embeddings, metadata={"a": 1},
    )
    artifacts = load_two_tower_artifacts(out_dir)
    assert artifacts.item_ids == [10, 20, 30]
    assert np.allclose(artifacts.item_embeddings, embeddings)
    assert artifacts.metadata == {"a": 1}
