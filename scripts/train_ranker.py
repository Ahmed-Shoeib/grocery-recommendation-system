"""Train the Phase 6 neural ranker end to end and report results.

Reuses the ALREADY-TRAINED Phase 4 Two-Tower model from
`models/two_tower/` (does not retrain it) and Phase 5's `VectorIndex`
abstraction to source candidates - run `scripts/train_two_tower.py`
first if those artifacts don't exist yet.

Usage:
    python scripts/train_ranker.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from recommendation.data.adapters.factory import build_synthetic_adapters
from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.data.synthetic.validation import validate_dataset
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.evaluation.latency import measure_latency
from recommendation.features.pipeline import run_feature_pipeline
from recommendation.ranking.evaluation import RankingEvalReport
from recommendation.ranking.features import RANKING_FEATURE_NAMES
from recommendation.ranking.serialization import save_ranker_artifacts
from recommendation.ranking.train import train_ranker
from recommendation.retrieval.two_tower.serialization import load_two_tower_artifacts
from recommendation.utils.config import get_config, resolve_path
from recommendation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _print_report(report: RankingEvalReport) -> None:
    for k in sorted(report.ndcg_at_k):
        print(
            f"  NDCG@{k}: {report.ndcg_at_k[k]:.4f}   Precision@{k}: {report.precision_at_k[k]:.4f}   "
            f"Recall@{k}: {report.recall_at_k[k]:.4f}   HitRate@{k}: {report.hit_rate_at_k[k]:.4f}"
        )
    print(f"  MRR: {report.mrr:.4f}")


def main() -> None:
    setup_logging()
    config = get_config()

    dataset = generate_synthetic_dataset(config)
    issues = validate_dataset(dataset)
    if issues:
        raise SystemExit(f"Dataset validation failed: {issues[:5]}")
    bundle = build_synthetic_adapters(dataset, config.synthetic_data)

    st_encoder = SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    feature_result = run_feature_pipeline(bundle, config, encoder=st_encoder)

    two_tower_dir = resolve_path(config.paths.models_dir) / "two_tower"
    two_tower_artifacts = load_two_tower_artifacts(two_tower_dir)
    logger.info("Loaded Two-Tower artifacts from %s (not retrained)", two_tower_dir)

    result = train_ranker(bundle, feature_result, two_tower_artifacts, config, st_encoder=st_encoder)

    print("\n=== Phase 6 Ranker Training Report ===")
    print(f"Training examples: {result.num_train_examples} ({result.num_train_positive} positive / {result.num_train_negative} negative)")
    print(f"Validation-loss examples: {result.num_val_loss_examples}")
    print(f"Evaluable users (val+test targets): {result.num_eval_users}")
    print(f"Candidate pool size (VectorIndex retrieval depth): {result.pool_size}")
    print(f"Epochs run: {len(result.history.get('loss', []))} (early stopping patience={config.ranking.early_stopping_patience})")
    if result.history.get("loss"):
        print(f"Final train loss: {result.history['loss'][-1]:.4f}  AUC: {result.history['auc'][-1]:.4f}")
    if result.history.get("val_loss"):
        print(f"Final val loss:   {result.history['val_loss'][-1]:.4f}  AUC: {result.history['val_auc'][-1]:.4f}")

    print("\n--- Validation set: baseline (raw retrieval score order) ---")
    _print_report(result.val_report_baseline)
    print("\n--- Validation set: ranker ---")
    _print_report(result.val_report_ranker)

    print("\n--- Test set: baseline (raw retrieval score order) ---")
    _print_report(result.test_report_baseline)
    print("\n--- Test set: ranker ---")
    _print_report(result.test_report_ranker)

    print(
        f"\nTest NDCG@10: ranker={result.test_report_ranker.ndcg_at_k[10]:.4f} "
        f"vs. baseline={result.test_report_baseline.ndcg_at_k[10]:.4f}"
    )
    print("NOTE: ~50-item catalog means the retrieved candidate pool equals the full catalog -")
    print("      this demonstrates the ranking pipeline is correct and leakage-safe, not a")
    print("      production-scale test of ranking under genuine ANN-truncated candidates.")

    # --- Ranking latency: score a full candidate pool with the trained model ---
    print("\n=== Ranking latency ===")
    dummy_batch = np.random.default_rng(config.ranking.random_seed).normal(size=(result.pool_size, len(RANKING_FEATURE_NAMES))).astype(
        "float32"
    )
    latency_report = measure_latency(lambda: result.model.predict(dummy_batch, verbose=0), num_runs=50, warmup_runs=5)
    print(
        f"  Score {result.pool_size} candidates: mean={latency_report.mean_ms:.3f}ms  "
        f"p50={latency_report.p50_ms:.3f}ms  p95={latency_report.p95_ms:.3f}ms  p99={latency_report.p99_ms:.3f}ms"
    )

    out_dir = resolve_path(config.paths.models_dir) / "ranker"
    metadata = {
        "model_version": "ranker_v1",
        "feature_names": RANKING_FEATURE_NAMES,
        "num_train_examples": result.num_train_examples,
        "num_eval_users": result.num_eval_users,
        "pool_size": result.pool_size,
        "test_ndcg_at_10_ranker": result.test_report_ranker.ndcg_at_k.get(10),
        "test_ndcg_at_10_baseline": result.test_report_baseline.ndcg_at_k.get(10),
        "test_mrr_ranker": result.test_report_ranker.mrr,
        "test_mrr_baseline": result.test_report_baseline.mrr,
    }
    save_ranker_artifacts(out_dir, result.model, RANKING_FEATURE_NAMES, metadata)
    print(f"\nSaved ranker artifacts to {out_dir}")


if __name__ == "__main__":
    main()
