"""Generate the synthetic dataset, run the feature pipeline, and report
stats: embedding cache behavior, feature shapes, example vectors.

Usage:
    python scripts/build_features.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation.data.adapters.factory import build_synthetic_adapters
from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.data.synthetic.validation import validate_dataset
from recommendation.features.pipeline import run_feature_pipeline
from recommendation.utils.config import get_config
from recommendation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    config = get_config()
    setup_logging(config.log_level)

    dataset = generate_synthetic_dataset(config)
    issues = validate_dataset(dataset)
    if issues:
        raise SystemExit(f"Dataset validation failed: {issues[:5]}")
    bundle = build_synthetic_adapters(dataset, config.synthetic_data)

    start = time.perf_counter()
    result = run_feature_pipeline(bundle, config)
    elapsed = time.perf_counter() - start

    print("=== Sentence Transformer ===")
    print(f"Model:            {result.product_embeddings.model_name}")
    print(f"Embedding dim:    {result.product_embeddings.embedding_dim}")
    print(f"Products embedded:{len(result.product_embeddings.product_ids):5d}")
    print(f"Embeddings recomputed this run: {result.embeddings_recomputed}")
    print(f"Pipeline wall time: {elapsed:.2f}s")

    print("\n=== Product features (example) ===")
    example_product_id = result.product_embeddings.product_ids[0]
    pf = result.product_features[example_product_id]
    print(pf)
    print(f"embedding shape: {result.product_embeddings.get(example_product_id).shape}")

    print("\n=== User features (examples) ===")
    with_history = next(uid for uid, f in result.user_features.items() if f.purchase_count > 0)
    no_history = next(
        (uid for uid, f in result.user_features.items() if f.total_engagement_events == 0), None
    )
    uf = result.user_features[with_history]
    print(f"User {with_history} (has history):")
    print(f"  preferred_category={uf.preferred_category!r} age_group={uf.age_group!r}")
    print(f"  purchase_count={uf.purchase_count} cart_item_count={uf.cart_item_count} "
          f"search_count={uf.search_count} has_chatbot_context={uf.has_chatbot_context}")
    print(f"  total_engagement_events={uf.total_engagement_events}")
    print(f"  category_affinity={uf.category_affinity}")
    print(f"  brand_affinity={uf.brand_affinity}")
    print(f"  semantic_embedding shape={None if uf.semantic_embedding is None else uf.semantic_embedding.shape}")

    if no_history is not None:
        uf0 = result.user_features[no_history]
        print(f"\nUser {no_history} (no history):")
        print(f"  category_affinity={uf0.category_affinity} semantic_embedding={uf0.semantic_embedding}")

    n_with_embedding = sum(1 for f in result.user_features.values() if f.semantic_embedding is not None)
    print(f"\nUsers with a semantic embedding: {n_with_embedding}/{len(result.user_features)}")

    # Second run should be a pure cache hit (no re-encoding of products).
    start = time.perf_counter()
    result2 = run_feature_pipeline(bundle, config)
    elapsed2 = time.perf_counter() - start
    print(f"\nSecond run embeddings_recomputed={result2.embeddings_recomputed} wall time={elapsed2:.2f}s")


if __name__ == "__main__":
    main()
