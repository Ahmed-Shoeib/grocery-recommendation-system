"""End-to-end integration test: adapters -> Phase 3 features -> Phase 4
Two-Tower (trained once, reused unmodified) -> Phase 5 VectorIndex ->
Phase 6 ranker training -> evaluation -> serialization.

Uses the same fast, deterministic, hash-based fake encoder as
test_two_tower_train_pipeline.py so this stays quick.
"""

import hashlib

import numpy as np
import pytest

from recommendation.data.adapters.factory import build_synthetic_adapters
from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.features.pipeline import run_feature_pipeline
from recommendation.ranking.serialization import load_ranker_artifacts, save_ranker_artifacts
from recommendation.ranking.train import train_ranker
from recommendation.retrieval.two_tower.serialization import TwoTowerArtifacts
from recommendation.retrieval.two_tower.train import train_two_tower
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
def trained():
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
    item_tower_weights_before = [w.copy() for w in two_tower_result.item_tower.get_weights()]

    result = train_ranker(bundle, feature_result, two_tower_artifacts, config, st_encoder=_FakeEncoder())
    return result, config, two_tower_result, item_tower_weights_before


def test_training_produces_positive_and_negative_examples(trained):
    result, _, _, _ = trained
    assert result.num_train_positive > 0
    assert result.num_train_negative > 0
    assert result.num_train_examples == result.num_train_positive + result.num_train_negative


def test_eval_users_match_two_tower_eval_users(trained):
    """Phase 6 reuses Two-Tower's own split parameters, so the evaluable
    user count (and, by construction, the held-out val/test targets) must
    be IDENTICAL - otherwise ranker-vs-baseline isn't apples-to-apples.
    """
    result, _, two_tower_result, _ = trained
    assert result.num_eval_users == two_tower_result.num_eval_users
    assert result.num_eval_users > 0


def test_reports_have_all_configured_k_values(trained):
    result, config = trained[0], trained[1]
    for k in config.ranking.eval_k_values:
        for report in (result.val_report_baseline, result.val_report_ranker, result.test_report_baseline, result.test_report_ranker):
            assert k in report.ndcg_at_k
            assert k in report.precision_at_k
            assert k in report.recall_at_k
            assert k in report.hit_rate_at_k


def test_metric_values_are_valid_probabilities(trained):
    result, _, _, _ = trained
    for report in (result.val_report_baseline, result.val_report_ranker, result.test_report_baseline, result.test_report_ranker):
        for value in list(report.ndcg_at_k.values()) + list(report.precision_at_k.values()) + list(report.recall_at_k.values()):
            assert 0.0 <= value <= 1.0
        assert 0.0 <= report.mrr <= 1.0


def test_baseline_and_ranker_evaluate_identical_candidate_pools(trained):
    """Both reports are computed from the SAME RankingEvalCase.candidate_ids
    (same VectorIndex retrieval) - only the scoring function differs. This
    checks num_cases agree, i.e. they ran over the same case set.
    """
    result, _, _, _ = trained
    assert result.val_report_baseline.num_cases == result.val_report_ranker.num_cases
    assert result.test_report_baseline.num_cases == result.test_report_ranker.num_cases


def test_val_loss_present_and_early_stopping_did_not_run_forever(trained):
    result, config, _, _ = trained
    assert "val_loss" in result.history
    assert len(result.history["loss"]) <= config.ranking.epochs


def test_pool_size_capped_by_catalog_size(trained):
    result, config, _, _ = trained
    assert result.pool_size <= config.synthetic_data.num_products


def test_save_and_reload_preserves_predictions(tmp_path, trained):
    result, _, _, _ = trained
    out_dir = tmp_path / "ranker"
    save_ranker_artifacts(out_dir, result.model, ["f"] * result.model.input_shape[1], {"model_version": "test"})
    artifacts = load_ranker_artifacts(out_dir)

    x = np.random.default_rng(0).normal(size=(4, result.model.input_shape[1])).astype(np.float32)
    original = result.model.predict(x, verbose=0)
    reloaded = artifacts.model.predict(x, verbose=0)
    assert np.allclose(original, reloaded, atol=1e-6)
    assert artifacts.metadata == {"model_version": "test"}


def test_two_tower_weights_not_mutated_by_ranker_training(trained):
    """Phase 6 must not retrain or otherwise change the Two-Tower model -
    compares item_tower weights captured BEFORE `train_ranker` ran against
    the same tower's weights AFTER, not just two post-hoc predict calls.
    """
    _, _, two_tower_result, item_tower_weights_before = trained
    weights_after = two_tower_result.item_tower.get_weights()
    assert len(weights_after) == len(item_tower_weights_before)
    for before, after in zip(item_tower_weights_before, weights_after):
        assert np.array_equal(before, after)
