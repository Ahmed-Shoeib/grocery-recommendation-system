"""Train the Two-Tower retrieval model end to end and report results.

Usage:
    python scripts/train_two_tower.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation.data.adapters.factory import build_synthetic_adapters
from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.data.synthetic.validation import validate_dataset
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.features.pipeline import run_feature_pipeline
from recommendation.retrieval.two_tower.serialization import save_two_tower_artifacts
from recommendation.retrieval.two_tower.train import train_two_tower
from recommendation.utils.config import get_config, resolve_path
from recommendation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


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

    logger.info("Running Phase 3 feature pipeline (product embeddings + user/product features)...")
    feature_result = run_feature_pipeline(bundle, config, encoder=st_encoder)

    logger.info("Training Two-Tower model...")
    result = train_two_tower(bundle, feature_result, config, st_encoder=st_encoder)

    print("\n=== Two-Tower Training Report ===")
    print(f"Training examples: {result.num_training_examples}")
    print(f"Users contributing to training: {result.num_train_users}")
    print(f"Users evaluable (>= {config.two_tower.min_distinct_products_for_holdout} distinct purchases): {result.num_eval_users}")
    print(f"Epochs run: {len(result.history.get('loss', []))} (early stopping patience={config.two_tower.early_stopping_patience})")
    if result.history.get("loss"):
        print(f"Final train loss: {result.history['loss'][-1]:.4f}  in_batch_accuracy: {result.history['in_batch_accuracy'][-1]:.4f}")
    if result.history.get("val_loss"):
        print(f"Final val loss:   {result.history['val_loss'][-1]:.4f}  in_batch_accuracy: {result.history['val_in_batch_accuracy'][-1]:.4f}")

    print("\n--- Validation set (leave-one-out, unseen during training) ---")
    for k in config.two_tower.eval_k_values:
        print(f"  Recall@{k}: {result.val_report.recall_at_k[k]:.4f}   HitRate@{k}: {result.val_report.hit_rate_at_k[k]:.4f}")

    print("\n--- Test set (leave-one-out, unseen during training AND validation) ---")
    for k in config.two_tower.eval_k_values:
        print(f"  Recall@{k}: {result.test_report.recall_at_k[k]:.4f}   HitRate@{k}: {result.test_report.hit_rate_at_k[k]:.4f}")

    print(f"\nItem embeddings: {result.item_embeddings.shape} (L2-normalized, catalog size={len(result.item_ids)})")
    import numpy as np

    norms = np.linalg.norm(result.item_embeddings, axis=1)
    print(f"Item embedding norm range: [{norms.min():.4f}, {norms.max():.4f}] (expect ~1.0)")

    out_dir = resolve_path(config.paths.models_dir) / "two_tower"
    metadata = {
        "model_version": "two_tower_v1",
        "output_dim": config.two_tower.output_dim,
        "embedding_dim": config.embedding.embedding_dim,
        "sentence_transformer_model": config.embedding.sentence_transformer_model,
        "num_training_examples": result.num_training_examples,
        "num_train_users": result.num_train_users,
        "num_eval_users": result.num_eval_users,
        "val_hit_rate_at_10": result.val_report.hit_rate_at_k.get(10),
        "test_hit_rate_at_10": result.test_report.hit_rate_at_k.get(10),
    }
    save_two_tower_artifacts(
        out_dir, result.user_tower, result.item_tower, result.encoder, result.item_ids, result.item_embeddings, metadata
    )
    print(f"\nSaved model artifacts to {out_dir}")


if __name__ == "__main__":
    main()
