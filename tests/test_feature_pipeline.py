"""Integration test for the full AdapterBundle -> features pipeline.

Uses a fake, fast, deterministic stand-in for SentenceTransformerEncoder
(hash-based, fixed-dim) instead of the real model - this test is about
pipeline wiring (adapters -> features -> cache behavior), not encoding
quality, which is covered by test_embeddings.py against the real model.
"""

import hashlib

import numpy as np

from recommendation.data.adapters.factory import build_synthetic_adapters
from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.features.pipeline import run_feature_pipeline
from recommendation.utils.config import get_config

_FAKE_DIM = 16


class _FakeEncoder:
    model_name = "fake-hash-encoder"

    @property
    def embedding_dim(self) -> int:
        return _FAKE_DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, _FAKE_DIM), dtype=np.float32)
        rows = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).normal(size=_FAKE_DIM).astype(np.float32))
        return np.stack(rows)


def _bundle_and_config(num_users: int = 40):
    config = get_config().model_copy(deep=True)
    config.synthetic_data.num_users = num_users
    dataset = generate_synthetic_dataset(config)
    bundle = build_synthetic_adapters(dataset, config.synthetic_data)
    return bundle, config


def test_pipeline_produces_one_embedding_per_product(tmp_path):
    bundle, config = _bundle_and_config()
    config.embedding.cache_path = str(tmp_path / "embeddings.npz")
    result = run_feature_pipeline(bundle, config, encoder=_FakeEncoder())

    assert len(result.product_embeddings.product_ids) == len(bundle.products.list_products())
    assert result.product_embeddings.embeddings.shape == (len(bundle.products.list_products()), _FAKE_DIM)


def test_pipeline_produces_features_for_every_user(tmp_path):
    bundle, config = _bundle_and_config()
    config.embedding.cache_path = str(tmp_path / "embeddings.npz")
    result = run_feature_pipeline(bundle, config, encoder=_FakeEncoder())

    user_ids = set(bundle.users.list_user_ids())
    assert set(result.user_features.keys()) == user_ids
    assert set(result.engagement_profiles.keys()) == user_ids


def test_pipeline_first_run_recomputes_second_run_hits_cache(tmp_path):
    bundle, config = _bundle_and_config()
    config.embedding.cache_path = str(tmp_path / "embeddings.npz")

    result_1 = run_feature_pipeline(bundle, config, encoder=_FakeEncoder())
    assert result_1.embeddings_recomputed is True

    result_2 = run_feature_pipeline(bundle, config, encoder=_FakeEncoder())
    assert result_2.embeddings_recomputed is False
    assert np.array_equal(result_1.product_embeddings.embeddings, result_2.product_embeddings.embeddings)


def test_pipeline_users_with_purchases_have_purchase_count_features(tmp_path):
    bundle, config = _bundle_and_config()
    config.embedding.cache_path = str(tmp_path / "embeddings.npz")
    result = run_feature_pipeline(bundle, config, encoder=_FakeEncoder())

    purchasing_user_ids = {p.user_id for p in bundle.purchases.list_all_purchases()}
    assert purchasing_user_ids  # sanity: the small dataset still has some purchases
    for uid in purchasing_user_ids:
        assert result.user_features[uid].purchase_count > 0


def test_pipeline_product_features_align_with_embeddings(tmp_path):
    bundle, config = _bundle_and_config()
    config.embedding.cache_path = str(tmp_path / "embeddings.npz")
    result = run_feature_pipeline(bundle, config, encoder=_FakeEncoder())

    assert set(result.product_features.keys()) == set(result.product_embeddings.product_ids)
