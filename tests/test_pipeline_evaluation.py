"""End-to-end integration test: Phase 4 Two-Tower (trained once, reused) +
Phase 6 ranker (trained once, reused) -> Phase 7 full pipeline evaluation.

Uses the same fast, deterministic, hash-based fake encoder as the Phase 4/
6 integration tests so this stays quick.
"""

import hashlib

import numpy as np
import pytest

from recommendation.data.adapters.factory import build_synthetic_adapters
from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.features.pipeline import run_feature_pipeline
from recommendation.ranking.train import train_ranker
from recommendation.retrieval.index.factory import build_vector_index, candidate_pool_size
from recommendation.retrieval.two_tower.examples import build_eval_cases
from recommendation.retrieval.two_tower.serialization import TwoTowerArtifacts
from recommendation.retrieval.two_tower.splitting import build_user_splits
from recommendation.retrieval.two_tower.train import train_two_tower
from recommendation.serving.evaluation import evaluate_pipeline
from recommendation.serving.pipeline import generate_recommendations
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
def pipeline_eval():
    config = get_config().model_copy(deep=True)
    config.synthetic_data.num_users = 150
    config.embedding.embedding_dim = _FAKE_DIM
    config.two_tower.epochs = 3
    config.two_tower.batch_size = 16
    config.two_tower.min_distinct_products_for_holdout = 3
    config.two_tower.early_stopping_patience = 2
    config.ranking.epochs = 5
    config.ranking.batch_size = 32
    config.ranking.hidden_units = [16, 8]
    config.ranking.negatives_per_positive = 3

    dataset = generate_synthetic_dataset(config)
    bundle = build_synthetic_adapters(dataset, config.synthetic_data)
    feature_result = run_feature_pipeline(bundle, config, encoder=_FakeEncoder())
    two_tower_result = train_two_tower(bundle, feature_result, config, st_encoder=_FakeEncoder())
    two_tower_artifacts = TwoTowerArtifacts(
        user_tower=two_tower_result.user_tower,
        item_tower=two_tower_result.item_tower,
        encoder=two_tower_result.encoder,
        item_ids=two_tower_result.item_ids,
        item_embeddings=two_tower_result.item_embeddings,
        metadata={},
    )
    ranker_result = train_ranker(bundle, feature_result, two_tower_artifacts, config, st_encoder=_FakeEncoder())

    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    product_features = feature_result.product_features
    product_embeddings = feature_result.product_embeddings.as_dict()
    engagement_profiles = feature_result.engagement_profiles

    splits = build_user_splits(
        engagement_profiles,
        min_distinct_products_for_holdout=config.two_tower.min_distinct_products_for_holdout,
        random_seed=config.two_tower.random_seed,
    )
    eval_cases = build_eval_cases(engagement_profiles, splits, product_lookup, product_embeddings, config.features, {})

    vector_index = build_vector_index(config.retrieval)
    vector_index.build(two_tower_artifacts.item_ids, two_tower_artifacts.item_embeddings)
    top_n = 10

    results = [
        generate_recommendations(
            case.user_features, product_features, product_embeddings, two_tower_artifacts.item_ids,
            two_tower_artifacts.encoder, two_tower_artifacts.user_tower, ranker_result.model, vector_index, config, top_n,
        )
        for case in eval_cases
    ]

    return eval_cases, results, config, product_features, two_tower_artifacts, top_n


def test_eval_cases_and_results_are_aligned(pipeline_eval):
    eval_cases, results, *_ = pipeline_eval
    assert len(eval_cases) == len(results)
    assert len(eval_cases) > 0


def test_tier_counts_cover_at_least_strong_or_sparse(pipeline_eval):
    """Held-out eval users always have >=1 remaining train purchase after
    excluding val+test (min_distinct_products_for_holdout=3, 2 excluded),
    so they can never land in NO_HISTORY - documented, not a bug.
    """
    _, results, *_ = pipeline_eval
    tiers = {r.tier.value for r in results}
    assert tiers <= {"strong", "sparse"}
    assert tiers  # non-empty


def test_full_pipeline_report_has_all_k_values(pipeline_eval):
    eval_cases, results, config, product_features, two_tower_artifacts, _ = pipeline_eval
    catalog_size = len(two_tower_artifacts.item_ids)
    report = evaluate_pipeline(eval_cases, results, "test_product_id", "test", config.ranking.eval_k_values, catalog_size, product_features)
    for k in config.ranking.eval_k_values:
        assert k in report.ndcg_at_k
        assert k in report.precision_at_k


def test_duplicate_rate_is_zero(pipeline_eval):
    eval_cases, results, config, product_features, two_tower_artifacts, _ = pipeline_eval
    catalog_size = len(two_tower_artifacts.item_ids)
    report = evaluate_pipeline(eval_cases, results, "test_product_id", "test", config.ranking.eval_k_values, catalog_size, product_features)
    assert report.duplicate_rate == 0.0


def test_fill_rate_is_valid_fraction(pipeline_eval):
    eval_cases, results, config, product_features, two_tower_artifacts, _ = pipeline_eval
    catalog_size = len(two_tower_artifacts.item_ids)
    report = evaluate_pipeline(eval_cases, results, "test_product_id", "test", config.ranking.eval_k_values, catalog_size, product_features)
    assert 0.0 <= report.fill_rate <= 1.0


def test_catalog_coverage_is_valid_fraction(pipeline_eval):
    eval_cases, results, config, product_features, two_tower_artifacts, _ = pipeline_eval
    catalog_size = len(two_tower_artifacts.item_ids)
    report = evaluate_pipeline(eval_cases, results, "test_product_id", "test", config.ranking.eval_k_values, catalog_size, product_features)
    assert 0.0 <= report.catalog_coverage <= 1.0


def test_pre_rerank_vs_full_pipeline_report_both_computable(pipeline_eval):
    """Both reports come from the SAME pipeline run (same retrieval, same
    ranker scores) - only re-ranking/eligibility differ - isolating their effect.
    """
    eval_cases, results, config, product_features, two_tower_artifacts, _ = pipeline_eval
    catalog_size = len(two_tower_artifacts.item_ids)
    ranker_only = evaluate_pipeline(eval_cases, results, "test_product_id", "ranker_only", config.ranking.eval_k_values, catalog_size, product_features, use_pre_rerank=True)
    full_pipeline = evaluate_pipeline(eval_cases, results, "test_product_id", "full", config.ranking.eval_k_values, catalog_size, product_features)
    assert ranker_only.num_cases == full_pipeline.num_cases
    for report in (ranker_only, full_pipeline):
        for value in report.ndcg_at_k.values():
            assert 0.0 <= value <= 1.0


def test_results_respect_requested_top_n(pipeline_eval):
    _, results, _, _, _, top_n = pipeline_eval
    for r in results:
        assert len(r.product_ids) <= top_n
