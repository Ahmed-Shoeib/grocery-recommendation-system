"""End-to-end integration test: adapters -> Phase 3 features -> Phase 4
Two-Tower training -> evaluation -> serialization.

Uses a fast, deterministic, hash-based fake encoder (not the real
Sentence Transformer) so this stays quick - the real model's encoding
quality is covered by test_embeddings.py; this test is about the
training/evaluation/serialization plumbing being correct end to end.
"""

import hashlib

import numpy as np
import pytest

from recommendation.data.adapters.factory import build_synthetic_adapters
from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.features.pipeline import run_feature_pipeline
from recommendation.retrieval.two_tower.serialization import load_two_tower_artifacts, save_two_tower_artifacts
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

    dataset = generate_synthetic_dataset(config)
    bundle = build_synthetic_adapters(dataset, config.synthetic_data)
    feature_result = run_feature_pipeline(bundle, config, encoder=_FakeEncoder())
    result = train_two_tower(bundle, feature_result, config, st_encoder=_FakeEncoder())
    return result, config, bundle


def test_training_produces_examples_and_evaluable_users(trained):
    result, config, _ = trained
    assert result.num_training_examples > 0
    assert result.num_eval_users > 0
    assert result.num_train_users > 0


def test_item_embeddings_shape_and_l2_normalized(trained):
    result, config, _ = trained
    assert result.item_embeddings.shape == (config.synthetic_data.num_products, config.two_tower.output_dim)
    norms = np.linalg.norm(result.item_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_val_and_test_reports_have_all_configured_k_values(trained):
    result, config, _ = trained
    for k in config.two_tower.eval_k_values:
        assert k in result.val_report.recall_at_k
        assert k in result.val_report.hit_rate_at_k
        assert k in result.test_report.recall_at_k
        assert k in result.test_report.hit_rate_at_k


def test_recall_and_hit_rate_values_are_valid_probabilities(trained):
    result, _, _ = trained
    for report in (result.val_report, result.test_report):
        for value in list(report.recall_at_k.values()) + list(report.hit_rate_at_k.values()):
            assert 0.0 <= value <= 1.0


def test_recall_at_k_is_monotonically_nondecreasing_in_k(trained):
    result, config, _ = trained
    ks = sorted(config.two_tower.eval_k_values)
    for report in (result.val_report, result.test_report):
        values = [report.recall_at_k[k] for k in ks]
        assert values == sorted(values)


def test_training_beats_random_baseline_on_test_set(trained):
    """Sanity check that the model learned SOMETHING from the
    persona-correlated synthetic data, not just noise: HitRate@10 should
    clear the random-retrieval baseline (10 / catalog_size) by a wide
    margin given the strong category/tag correlation Phase 2 built in.
    """
    result, config, _ = trained
    catalog_size = config.synthetic_data.num_products
    random_baseline = 10 / catalog_size
    assert result.test_report.hit_rate_at_k[10] > random_baseline * 1.5


def test_val_loss_present_and_early_stopping_did_not_run_forever(trained):
    result, config, _ = trained
    assert "val_loss" in result.history
    assert len(result.history["loss"]) <= config.two_tower.epochs


def test_save_and_reload_preserves_evaluation_quality(tmp_path, trained):
    result, config, bundle = trained
    out_dir = tmp_path / "two_tower"
    save_two_tower_artifacts(
        out_dir, result.user_tower, result.item_tower, result.encoder,
        result.item_ids, result.item_embeddings, {"model_version": "test"},
    )
    artifacts = load_two_tower_artifacts(out_dir)
    assert np.allclose(artifacts.item_embeddings, result.item_embeddings)
    assert artifacts.item_ids == result.item_ids


def test_no_faiss_or_scann_import_in_phase4_modules():
    """Confirms Phase 5 (ANN retrieval) has not been implemented yet. Checks
    for actual import statements, not mere mentions - evaluation.py's
    docstring legitimately explains brute-force ranking is a stand-in for
    Phase 5's FAISS/ScaNN index, which would otherwise false-positive a
    naive substring search.
    """
    import ast

    import recommendation.retrieval.two_tower.evaluation as evaluation_module
    import recommendation.retrieval.two_tower.train as train_module

    for module in (evaluation_module, train_module):
        with open(module.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        imported_names = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert "faiss" not in imported_names
        assert "scann" not in imported_names
