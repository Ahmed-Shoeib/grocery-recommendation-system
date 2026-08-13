"""Run and evaluate the full Phase 7 V1 serving pipeline end to end:

Two-Tower -> VectorIndex retrieval -> Neural Ranker -> Re-ranking
-> Business Rules / Eligibility -> Final Top-N

Reuses the ALREADY-TRAINED Phase 4 Two-Tower and Phase 6 ranker artifacts
(does not retrain either) - run scripts/train_two_tower.py and
scripts/train_ranker.py first if those don't exist yet.

Usage:
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from recommendation.data.adapters.factory import build_synthetic_adapters
from recommendation.data.schemas.engagement import EngagementProfile
from recommendation.data.schemas.user import UserProfile
from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.data.synthetic.validation import validate_dataset
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.evaluation.latency import measure_latency
from recommendation.features.pipeline import run_feature_pipeline
from recommendation.features.user_features import build_user_features
from recommendation.ranking.serialization import load_ranker_artifacts
from recommendation.retrieval.index.factory import build_vector_index
from recommendation.retrieval.two_tower.examples import build_eval_cases
from recommendation.retrieval.two_tower.serialization import load_two_tower_artifacts
from recommendation.retrieval.two_tower.splitting import build_user_splits
from recommendation.serving.cold_start import HistoryTier, determine_history_tier
from recommendation.serving.evaluation import PipelineEvalReport, evaluate_pipeline
from recommendation.serving.pipeline import generate_recommendations
from recommendation.utils.config import get_config, resolve_path
from recommendation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _print_report(label: str, report: PipelineEvalReport) -> None:
    print(f"\n--- {label} ---")
    for k in sorted(report.ndcg_at_k):
        print(
            f"  NDCG@{k}: {report.ndcg_at_k[k]:.4f}   Precision@{k}: {report.precision_at_k[k]:.4f}   "
            f"Recall@{k}: {report.recall_at_k[k]:.4f}   HitRate@{k}: {report.hit_rate_at_k[k]:.4f}"
        )
    print(f"  MRR: {report.mrr:.4f}")
    print(f"  Catalog coverage: {report.catalog_coverage:.4f}   Mean distinct categories/list: {report.mean_distinct_categories:.2f}")
    print(f"  Duplicate rate: {report.duplicate_rate:.4f}   Fill rate: {report.fill_rate:.4f}")
    print(f"  Tier distribution: {report.tier_counts}")


def _print_example(label: str, product_lookup, result) -> None:
    print(f"\n--- {label} (tier={result.tier.value}) ---")
    print(f"  Requested {result.requested_n}, filled {len(result.product_ids)} (fill_rate={result.fill_rate:.2f})")
    print(f"  Pool size: {result.pool_size}   Excluded by eligibility: {result.num_excluded_by_eligibility}")
    for pid, score, source in zip(result.product_ids, result.scores, result.sources):
        product = product_lookup.get(pid)
        name = product.name if product else f"#{pid}"
        print(f"    {pid:>4}  {name:<30}  score={score:.4f}  source={source}")


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
    ranker_dir = resolve_path(config.paths.models_dir) / "ranker"
    ranker_artifacts = load_ranker_artifacts(ranker_dir)
    logger.info("Loaded Two-Tower artifacts from %s and ranker artifacts from %s (neither retrained)", two_tower_dir, ranker_dir)

    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    product_features = feature_result.product_features
    product_embeddings = feature_result.product_embeddings.as_dict()
    engagement_profiles = feature_result.engagement_profiles

    vector_index = build_vector_index(config.retrieval)
    vector_index.build(two_tower_artifacts.item_ids, two_tower_artifacts.item_embeddings)

    splits = build_user_splits(
        engagement_profiles,
        min_distinct_products_for_holdout=config.two_tower.min_distinct_products_for_holdout,
        random_seed=config.two_tower.random_seed,
    )
    eval_cases = build_eval_cases(engagement_profiles, splits, product_lookup, product_embeddings, config.features, {})
    top_n = config.api.default_recommendation_count

    print(f"\n=== Phase 7 Pipeline Evaluation ({len(eval_cases)} eval users, top_n={top_n}) ===")
    results = [
        generate_recommendations(
            case.user_features, product_features, product_embeddings, two_tower_artifacts.item_ids,
            two_tower_artifacts.encoder, two_tower_artifacts.user_tower, ranker_artifacts.model, vector_index, config, top_n,
        )
        for case in eval_cases
    ]
    catalog_size = len(two_tower_artifacts.item_ids)
    k_values = config.ranking.eval_k_values

    for split_name, target_attr in (("Validation", "val_product_id"), ("Test", "test_product_id")):
        ranker_only = evaluate_pipeline(eval_cases, results, target_attr, split_name, k_values, catalog_size, product_features, use_pre_rerank=True)
        full_pipeline = evaluate_pipeline(eval_cases, results, target_attr, split_name, k_values, catalog_size, product_features)
        _print_report(f"{split_name} set: Phase 6 ranker-only (pre re-rank/eligibility)", ranker_only)
        _print_report(f"{split_name} set: full Phase 7 pipeline", full_pipeline)

    # --- Qualitative examples ---
    print("\n\n=== Qualitative recommendation examples ===")

    strong_uid, sparse_uid = None, None
    for uid, profile in engagement_profiles.items():
        features = build_user_features(profile, product_lookup, product_embeddings, config.features)
        tier = determine_history_tier(features.total_engagement_events, config.cold_start)
        if tier is HistoryTier.STRONG and strong_uid is None:
            strong_uid = uid
        elif tier is HistoryTier.SPARSE and sparse_uid is None:
            sparse_uid = uid
        if strong_uid is not None and sparse_uid is not None:
            break

    for uid, label in ((strong_uid, "Strong-history user"), (sparse_uid, "Sparse-history user")):
        if uid is None:
            print(f"\n--- {label}: none found in this dataset instantiation ---")
            continue
        profile = engagement_profiles[uid]
        user_features = build_user_features(profile, product_lookup, product_embeddings, config.features)
        result = generate_recommendations(
            user_features, product_features, product_embeddings, two_tower_artifacts.item_ids,
            two_tower_artifacts.encoder, two_tower_artifacts.user_tower, ranker_artifacts.model, vector_index, config, top_n,
        )
        _print_example(f"{label} (user {uid})", product_lookup, result)

    # Constructed (not drawn from the dataset) - demonstrates the
    # NO_HISTORY code path deterministically regardless of whether this
    # particular synthetic dataset instantiation happens to contain a
    # genuinely zero-signal user.
    no_history_profile = EngagementProfile(user_id=-1, profile=UserProfile(user_id=-1, preferred_category=next(iter({p.category_name for p in products if p.category_name}))))
    no_history_features = build_user_features(no_history_profile, product_lookup, product_embeddings, config.features)
    no_history_result = generate_recommendations(
        no_history_features, product_features, product_embeddings, two_tower_artifacts.item_ids,
        two_tower_artifacts.encoder, two_tower_artifacts.user_tower, ranker_artifacts.model, vector_index, config, top_n,
    )
    _print_example("No-history user (constructed, preferredCategory set)", product_lookup, no_history_result)

    # --- Eligibility demonstration: find a real user whose pre-eligibility
    # pool includes an inactive/out-of-stock item, and show it excluded. ---
    print("\n\n=== Eligibility (business rules) demonstration ===")
    demo_result = None
    demo_uid = None
    for uid, profile in engagement_profiles.items():
        user_features = build_user_features(profile, product_lookup, product_embeddings, config.features)
        result = generate_recommendations(
            user_features, product_features, product_embeddings, two_tower_artifacts.item_ids,
            two_tower_artifacts.encoder, two_tower_artifacts.user_tower, ranker_artifacts.model, vector_index, config, top_n,
        )
        if result.num_excluded_by_eligibility > 0:
            demo_result, demo_uid = result, uid
            break

    if demo_result is None:
        print("  No user's candidate pool included an ineligible product in this dataset instantiation.")
    else:
        excluded_ids = [pid for pid in demo_result.pre_eligibility_product_ids if pid in demo_result.excluded_reasons]
        print(f"  User {demo_uid}: {demo_result.num_excluded_by_eligibility} candidate(s) excluded from a pool of {demo_result.num_before_eligibility}")
        for pid in excluded_ids:
            pf = product_features[pid]
            print(f"    excluded #{pid} {product_lookup[pid].name!r}: reasons={demo_result.excluded_reasons[pid]}  (is_active={pf.is_active}, stock={pf.stock_quantity})")
        print(f"  Final list still filled {len(demo_result.product_ids)}/{demo_result.requested_n} slots (fill_rate={demo_result.fill_rate:.2f})")
        _print_example(f"User {demo_uid} final recommendations", product_lookup, demo_result)

    # --- End-to-end latency ---
    print("\n\n=== End-to-end recommendation latency (single user, up through eligibility) ===")
    if strong_uid is not None:
        profile = engagement_profiles[strong_uid]
        user_features = build_user_features(profile, product_lookup, product_embeddings, config.features)
        latency_report = measure_latency(
            lambda: generate_recommendations(
                user_features, product_features, product_embeddings, two_tower_artifacts.item_ids,
                two_tower_artifacts.encoder, two_tower_artifacts.user_tower, ranker_artifacts.model, vector_index, config, top_n,
            ),
            num_runs=30,
            warmup_runs=3,
        )
        print(
            f"  mean={latency_report.mean_ms:.2f}ms  p50={latency_report.p50_ms:.2f}ms  "
            f"p95={latency_report.p95_ms:.2f}ms  p99={latency_report.p99_ms:.2f}ms"
        )

    print("\nNOTE: ~50-item catalog means retrieval/ranking pools equal the full catalog -")
    print("      metrics validate pipeline correctness and relative comparisons, not production-scale behavior.")


if __name__ == "__main__":
    main()
