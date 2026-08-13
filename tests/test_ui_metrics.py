"""Tests `ui.metrics.compute_offline_metrics` end to end - trains a real
(fake-encoder, fast) Two-Tower + ranker, wraps them in a
`RecommendationService`, and confirms the offline metrics summary has
the expected shape. Mirrors test_pipeline_evaluation.py's fixture, since
`compute_offline_metrics` is a thin wrapper around exactly that same
evaluation path.
"""

import hashlib

import numpy as np
import pytest

from recommendation.api.dependencies import RecommendationService
from recommendation.data.adapters.factory import build_synthetic_adapters
from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.features.pipeline import run_feature_pipeline
from recommendation.ranking.train import train_ranker
from recommendation.retrieval.index.factory import build_vector_index
from recommendation.retrieval.two_tower.serialization import TwoTowerArtifacts
from recommendation.retrieval.two_tower.train import train_two_tower
from recommendation.ui.metrics import compute_offline_metrics
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


@pytest.fixture(scope="module")
def service() -> RecommendationService:
    config = get_config().model_copy(deep=True)
    config.synthetic_data.num_users = 150
    config.embedding.embedding_dim = _FAKE_DIM
    config.two_tower.epochs = 3
    config.two_tower.batch_size = 16
    config.two_tower.min_distinct_products_for_holdout = 3
    config.two_tower.early_stopping_patience = 2
    config.ranking.epochs = 3
    config.ranking.batch_size = 32
    config.ranking.hidden_units = [16, 8]
    config.ranking.negatives_per_positive = 3

    dataset = generate_synthetic_dataset(config)
    bundle = build_synthetic_adapters(dataset, config.synthetic_data)
    feature_result = run_feature_pipeline(bundle, config, encoder=_FakeEncoder())
    two_tower_result = train_two_tower(bundle, feature_result, config, st_encoder=_FakeEncoder())
    two_tower_artifacts = TwoTowerArtifacts(
        user_tower=two_tower_result.user_tower, item_tower=two_tower_result.item_tower, encoder=two_tower_result.encoder,
        item_ids=two_tower_result.item_ids, item_embeddings=two_tower_result.item_embeddings, metadata={},
    )
    ranker_result = train_ranker(bundle, feature_result, two_tower_artifacts, config, st_encoder=_FakeEncoder())

    vector_index = build_vector_index(config.retrieval)
    vector_index.build(two_tower_artifacts.item_ids, two_tower_artifacts.item_embeddings)

    return RecommendationService(
        product_lookup={p.id: p for p in bundle.products.list_products()},
        product_features=feature_result.product_features,
        product_embeddings=feature_result.product_embeddings.as_dict(),
        text_embeddings={},
        all_item_ids=two_tower_artifacts.item_ids,
        tt_encoder=two_tower_artifacts.encoder,
        user_tower=two_tower_artifacts.user_tower,
        ranker_model=ranker_result.model,
        vector_index=vector_index,
        bundle=bundle,
        config=config,
        ranker_model_version="test",
        two_tower_model_version="test",
        engagement_profiles=feature_result.engagement_profiles,
    )


def test_offline_metrics_has_evaluable_users(service):
    summary = compute_offline_metrics(service, top_n=10)
    assert summary.num_eval_users > 0
    assert summary.val_report.num_cases == summary.num_eval_users
    assert summary.test_report.num_cases == summary.num_eval_users


def test_offline_metrics_reports_configured_k_values(service):
    summary = compute_offline_metrics(service, top_n=10)
    for k in service.config.ranking.eval_k_values:
        assert k in summary.val_report.ndcg_at_k
        assert k in summary.test_report.ndcg_at_k


def test_offline_metrics_values_are_valid_fractions(service):
    summary = compute_offline_metrics(service, top_n=10)
    for report in (summary.val_report, summary.test_report):
        for value in list(report.ndcg_at_k.values()) + list(report.precision_at_k.values()):
            assert 0.0 <= value <= 1.0
        assert 0.0 <= report.mrr <= 1.0
        assert 0.0 <= report.catalog_coverage <= 1.0
        assert 0.0 <= report.fill_rate <= 1.0
