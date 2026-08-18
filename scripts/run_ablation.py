"""STEP 8: controlled two-way ablation - BASE vs. RECENCY+PRICE
(docs/data-mapping.md section 17).

Compares exactly two conditions on the IDENTICAL SQLite database, temporal
split, evaluation population, seeds, and training procedure:

  A. BASE          - recency disabled, STEP 6 price-aware personalization
                      entirely absent (pre-STEP-6 architecture: item
                      numeric=7, user numeric=9->8, ranker=23 features).
                      Trained FROM SCRATCH by this script.
  B. RECENCY+PRICE  - the current STEP 7 configuration (recency enabled,
                      STEP 6 price features enabled, item numeric=9, user
                      numeric=9, ranker=29 features). REUSES the existing
                      `models/sqlite_baseline/` artifacts (verified below
                      to be identical in data fingerprint/seed/config/
                      procedure to what this script would otherwise
                      train) rather than retraining - see docs/data-
                      mapping.md section 17 "artifact reuse decision".

Both conditions are evaluated with `evaluation.temporal_training
.evaluate_primary_pipeline` (the full serving pipeline: eligibility ->
retrieval -> ranker -> re-ranking -> final validation) against the SAME
val/test `TemporalEvalCase`s (same users, cutoffs, future-PURCHASE
targets) - only each condition's OWN `user_features` differ.

Usage:
    python scripts/run_ablation.py

Does NOT commit, push, or touch models/sqlite_baseline or models/two_tower/
models/ranker (the STEP 6/7 artifacts) - BASE artifacts are written to the
isolated models/ablation/base/ directory.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from recommendation.data.adapters.sqlite_factory import build_sqlite_adapters
from recommendation.data.schemas.events import ActionType
from recommendation.data.sqlite.connection import open_readonly_connection
from recommendation.data.sqlite.loader import load_events
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.embeddings.product_embeddings import get_or_compute_product_embeddings
from recommendation.evaluation.latency import measure_latency
from recommendation.evaluation.temporal_future_purchase import (
    DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT,
    audit_no_leakage,
    build_temporal_splits,
    events_before_cutoff,
    group_events_by_user,
)
from recommendation.evaluation.temporal_training import (
    PrimaryEvalReport,
    TrainedCondition,
    build_temporal_two_tower_examples,
    evaluate_primary_pipeline,
    train_temporal_condition,
)
from recommendation.features.price import build_price_catalog_context
from recommendation.features.product_features import build_product_features
from recommendation.ranking.serialization import load_ranker_artifacts
from recommendation.retrieval.index.factory import build_vector_index
from recommendation.retrieval.two_tower.serialization import load_two_tower_artifacts
from recommendation.serving.cold_start import HistoryTier, determine_history_tier
from recommendation.serving.eligibility import apply_eligibility, build_eligibility_rules
from recommendation.serving.pipeline import generate_recommendations
from recommendation.utils.config import RecencyConfig, get_config, resolve_path
from recommendation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

K_VALUES = [5, 10, 20]
TOP_N = 20


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _print_primary_report(report: PrimaryEvalReport) -> None:
    for k in K_VALUES:
        print(
            f"  @{k:<3d} Precision={report.precision_at_k[k]:.4f}  Recall={report.recall_at_k[k]:.4f}  "
            f"HitRate={report.hit_rate_at_k[k]:.4f}  NDCG={report.ndcg_at_k[k]:.4f}"
        )
    print(f"  MRR={report.mrr:.4f}")
    print(f"  mean distinct categories in top-{TOP_N}: {report.mean_distinct_categories:.2f}")
    print(f"  catalog coverage: {report.catalog_coverage:.4f}")
    print(f"  mean fill rate: {report.mean_fill_rate:.4f}")


def _delta_report(base: PrimaryEvalReport, improved: PrimaryEvalReport) -> None:
    for k in K_VALUES:
        for metric_name, metric in [
            ("Precision", "precision_at_k"), ("Recall", "recall_at_k"), ("HitRate", "hit_rate_at_k"), ("NDCG", "ndcg_at_k")
        ]:
            b = getattr(base, metric)[k]
            i = getattr(improved, metric)[k]
            delta = i - b
            rel = (delta / b * 100.0) if b > 0 else float("nan")
            print(f"  @{k:<3d} {metric_name:<10s} BASE={b:.4f}  IMPROVED={i:.4f}  delta={delta:+.4f}  rel={rel:+.1f}%")
    b_mrr, i_mrr = base.mrr, improved.mrr
    delta_mrr = i_mrr - b_mrr
    rel_mrr = (delta_mrr / b_mrr * 100.0) if b_mrr > 0 else float("nan")
    print(f"  MRR       BASE={b_mrr:.4f}  IMPROVED={i_mrr:.4f}  delta={delta_mrr:+.4f}  rel={rel_mrr:+.1f}%")


def _paired_hits(base_cases, improved_cases, base_recs, improved_recs, k: int) -> dict[str, int]:
    both = base_only = improved_only = neither = 0
    for b_case, i_case, b_rec, i_rec in zip(base_cases, improved_cases, base_recs, improved_recs):
        assert b_case.user_id == i_case.user_id and b_case.target_ids == i_case.target_ids
        b_hit = bool(set(b_rec[:k]) & b_case.target_ids)
        i_hit = bool(set(i_rec[:k]) & i_case.target_ids)
        if b_hit and i_hit:
            both += 1
        elif b_hit and not i_hit:
            base_only += 1
        elif i_hit and not b_hit:
            improved_only += 1
        else:
            neither += 1
    return {"both": both, "base_only": base_only, "improved_only": improved_only, "neither": neither}


def main() -> None:
    config = get_config()
    setup_logging(config.log_level)

    print("=" * 78)
    print("STEP 8: CONTROLLED TWO-WAY ABLATION - BASE vs RECENCY+PRICE")
    print("=" * 78)

    db_path = resolve_path(config.paths.data_sqlite)
    assert db_path.name == "backend_shaped_synthetic.db"
    db_fingerprint = _sha256_file(db_path)
    print(f"SQLite path: {db_path}")
    print(f"dataset fingerprint (sha256[:16]): {db_fingerprint}")

    bundle = build_sqlite_adapters()
    con = open_readonly_connection(db_path)
    try:
        all_events = load_events(con)
    finally:
        con.close()

    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    user_ids = bundle.users.list_user_ids()
    action_counts = Counter(e.action_type.value for e in all_events)
    print(f"users={len(user_ids)} products={len(products)} events={len(all_events)} purchases={action_counts.get('PURCHASE', 0)}")

    product_features = build_product_features(products, [], [], [])
    eligibility_rules = build_eligibility_rules(config.eligibility)
    eligible_ids = apply_eligibility([p.id for p in products], product_features, eligibility_rules).eligible_ids
    print(f"eligible products: {len(eligible_ids)} / {len(products)}")

    events_by_user = group_events_by_user(all_events)
    splits = build_temporal_splits(events_by_user, user_ids, DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT)
    print(f"val-evaluable users: {sum(1 for s in splits.values() if s.is_val_evaluable)}")
    print(f"test-evaluable users: {sum(1 for s in splits.values() if s.is_test_evaluable)}")

    # --- leakage audit (condition-independent - depends only on splits/events, not features) ---
    print()
    print("=" * 78)
    print("LEAKAGE AUDIT (identical for both conditions - depends only on splits/events, not feature config)")
    print("=" * 78)
    total_checked = total_failed = 0
    for s in splits.values():
        for cutoff, target_ids in ((s.val_cutoff, s.val_target_ids), (s.test_cutoff, s.test_target_ids)):
            if cutoff is None:
                continue
            history = events_before_cutoff(events_by_user.get(s.user_id, []), cutoff)
            result = audit_no_leakage(s.user_id, history, cutoff, target_ids)
            total_checked += 1
            total_failed += 0 if result.ok else 1
    print(f"evaluation points audited: {total_checked}  violations: {total_failed}")
    assert total_failed == 0, "leakage audit failed - aborting"

    # --- product semantic embeddings (shared - catalog-only, condition-independent) ---
    print()
    print("=" * 78)
    print("PRODUCT SEMANTIC EMBEDDINGS (shared across both conditions)")
    print("=" * 78)
    st_encoder = SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    cache, recomputed = get_or_compute_product_embeddings(
        products, st_encoder, resolve_path("data/processed/product_embeddings_sqlite.npz")
    )
    product_embeddings = cache.as_dict()
    print(f"products represented: {len(product_embeddings)} / {len(products)}  dim={cache.embedding_dim}  recomputed={recomputed}")

    price_context = build_price_catalog_context(products)
    print(f"catalog price tier boundaries: {price_context.catalog_tier_boundaries}")

    # ======================= CONDITION A: BASE =======================
    print()
    print("=" * 78)
    print("TRAINING CONDITION A: BASE (recency OFF, price-aware personalization OFF)")
    print("=" * 78)
    base_feature_config = config.features.model_copy(update={"recency": RecencyConfig(enabled=False)})
    print(f"BASE feature_config.recency.enabled = {base_feature_config.recency.enabled}")
    print("BASE price_context passed to build_user_features = None (no price preference derived from purchase history at all)")

    base_condition: TrainedCondition = train_temporal_condition(
        label="BASE",
        events_by_user=events_by_user,
        splits=splits,
        users_adapter=bundle.users,
        reviews_adapter=bundle.reviews,
        products=products,
        product_lookup=product_lookup,
        product_features=product_features,
        product_embeddings=product_embeddings,
        feature_config=base_feature_config,
        price_context=None,
        include_price_features=False,
        two_tower_config=config.two_tower,
        ranking_config=config.ranking,
        retrieval_config=config.retrieval,
        embedding_dim=config.embedding.embedding_dim,
        default_recommendation_count=config.api.default_recommendation_count,
        k_values=K_VALUES,
    )
    print(f"BASE dims: item_numeric={base_condition.encoder.item_numeric_dim}  user_numeric={base_condition.encoder.user_numeric_dim}")
    assert base_condition.encoder.item_numeric_dim == 7
    assert base_condition.encoder.user_numeric_dim == 8
    print(f"BASE Two-Tower: train_examples={base_condition.num_tt_train_examples} val_loss_examples={base_condition.num_tt_val_loss_examples}")
    if base_condition.tt_history.get("loss"):
        print(
            f"  epochs={len(base_condition.tt_history['loss'])} final_train_loss={base_condition.tt_history['loss'][-1]:.4f} "
            f"train_acc={base_condition.tt_history['in_batch_accuracy'][-1]:.4f}"
        )
    if base_condition.tt_history.get("val_loss"):
        print(
            f"  final_val_loss={base_condition.tt_history['val_loss'][-1]:.4f} "
            f"val_acc={base_condition.tt_history['val_in_batch_accuracy'][-1]:.4f}"
        )
    print(f"  val Recall@20={base_condition.val_retrieval_report.recall_at_k.get(20, float('nan')):.4f}  "
          f"test Recall@20={base_condition.test_retrieval_report.recall_at_k.get(20, float('nan')):.4f}")
    print(
        f"BASE ranker: train_examples={base_condition.num_ranker_train_examples} "
        f"({base_condition.num_ranker_train_positive} pos / {base_condition.num_ranker_train_negative} neg)  "
        f"feature_dim={base_condition.ranker_model.input_shape[1]}"
    )
    assert base_condition.ranker_model.input_shape[1] == 23
    if base_condition.ranker_history.get("val_auc"):
        print(f"  final_val_auc={base_condition.ranker_history['val_auc'][-1]:.4f}")

    # ================= CONDITION B: RECENCY+PRICE (reuse STEP 7) =================
    print()
    print("=" * 78)
    print("CONDITION B: RECENCY+PRICE - artifact reuse decision")
    print("=" * 78)
    sqlite_baseline_dir = resolve_path(config.paths.models_dir) / "sqlite_baseline"
    reuse_ok = False
    improved_condition: TrainedCondition | None = None
    tt_metadata: dict = {}
    rk_metadata: dict = {}

    if (sqlite_baseline_dir / "two_tower" / "metadata.json").exists():
        tt_metadata = json.loads((sqlite_baseline_dir / "two_tower" / "metadata.json").read_text(encoding="utf-8"))
        rk_metadata = json.loads((sqlite_baseline_dir / "ranker" / "metadata.json").read_text(encoding="utf-8"))
        reuse_ok = (
            tt_metadata.get("dataset_fingerprint_sha256_16") == db_fingerprint
            and tt_metadata.get("random_seed") == config.two_tower.random_seed
            and rk_metadata.get("random_seed") == config.ranking.random_seed
            and tt_metadata.get("recency_enabled") is True
            and tt_metadata.get("recency_half_life_days") == config.features.recency.half_life_days
            and tt_metadata.get("item_numeric_dim") == 9
            and tt_metadata.get("user_numeric_dim") == 9
            and rk_metadata.get("feature_dim") == 29
        )

    print(f"models/sqlite_baseline metadata found: {bool(tt_metadata)}")
    print(f"fingerprint match: {tt_metadata.get('dataset_fingerprint_sha256_16') == db_fingerprint}")
    print(f"seed match (two_tower={config.two_tower.random_seed}, ranking={config.ranking.random_seed}): "
          f"{tt_metadata.get('random_seed') == config.two_tower.random_seed and rk_metadata.get('random_seed') == config.ranking.random_seed}")
    print(f"recency config match (enabled=True, half_life={config.features.recency.half_life_days}): "
          f"{tt_metadata.get('recency_enabled') is True and tt_metadata.get('recency_half_life_days') == config.features.recency.half_life_days}")
    print(f"dimension match (9/9/29): {tt_metadata.get('item_numeric_dim') == 9 and tt_metadata.get('user_numeric_dim') == 9 and rk_metadata.get('feature_dim') == 29}")
    print(f"DECISION: {'REUSE existing models/sqlite_baseline/ artifacts (no retraining)' if reuse_ok else 'artifacts not provably identical - RETRAINING IMPROVED from scratch'}")

    if reuse_ok:
        tt_artifacts = load_two_tower_artifacts(sqlite_baseline_dir / "two_tower")
        rk_artifacts = load_ranker_artifacts(sqlite_baseline_dir / "ranker")
        vector_index = build_vector_index(config.retrieval)
        vector_index.build(tt_artifacts.item_ids, tt_artifacts.item_embeddings)

        _, _, improved_val_cases, improved_test_cases = build_temporal_two_tower_examples(
            events_by_user, splits, bundle.users, bundle.reviews, product_lookup, product_embeddings,
            config.features, price_context,
        )
        improved_condition = TrainedCondition(
            label="RECENCY_PRICE(reused)",
            encoder=tt_artifacts.encoder,
            user_tower=tt_artifacts.user_tower,
            item_tower=tt_artifacts.item_tower,
            ranker_model=rk_artifacts.model,
            item_ids=tt_artifacts.item_ids,
            item_embeddings=tt_artifacts.item_embeddings,
            vector_index=vector_index,
            pool_size=rk_metadata.get("pool_size", 50),
            tt_history={},
            ranker_history={},
            num_tt_train_examples=tt_metadata.get("num_train_examples", 0),
            num_tt_val_loss_examples=tt_metadata.get("num_val_loss_examples", 0),
            num_ranker_train_examples=rk_metadata.get("num_train_examples", 0),
            num_ranker_train_positive=rk_metadata.get("num_train_positive", 0),
            num_ranker_train_negative=rk_metadata.get("num_train_negative", 0),
            val_retrieval_report=None,
            test_retrieval_report=None,
            val_cases=improved_val_cases,
            test_cases=improved_test_cases,
        )
        print(f"IMPROVED dims (from reused artifacts): item_numeric={tt_artifacts.encoder.item_numeric_dim} user_numeric={tt_artifacts.encoder.user_numeric_dim}")
        print(f"IMPROVED ranker feature_dim (reused): {rk_metadata.get('feature_dim')}")
    else:
        improved_condition = train_temporal_condition(
            label="RECENCY_PRICE",
            events_by_user=events_by_user, splits=splits, users_adapter=bundle.users, reviews_adapter=bundle.reviews,
            products=products, product_lookup=product_lookup, product_features=product_features, product_embeddings=product_embeddings,
            feature_config=config.features, price_context=price_context, include_price_features=True,
            two_tower_config=config.two_tower, ranking_config=config.ranking, retrieval_config=config.retrieval,
            embedding_dim=config.embedding.embedding_dim, default_recommendation_count=config.api.default_recommendation_count,
            k_values=K_VALUES,
        )

    # --- same evaluation population check ---
    print()
    print("=" * 78)
    print("SAME EVALUATION POPULATION CHECK")
    print("=" * 78)
    base_val_ids = [(c.user_id, c.cutoff, c.target_ids) for c in base_condition.val_cases]
    improved_val_ids = [(c.user_id, c.cutoff, c.target_ids) for c in improved_condition.val_cases]
    base_test_ids = [(c.user_id, c.cutoff, c.target_ids) for c in base_condition.test_cases]
    improved_test_ids = [(c.user_id, c.cutoff, c.target_ids) for c in improved_condition.test_cases]
    print(f"val cases: BASE={len(base_val_ids)}  IMPROVED={len(improved_val_ids)}  identical={base_val_ids == improved_val_ids}")
    print(f"test cases: BASE={len(base_test_ids)}  IMPROVED={len(improved_test_ids)}  identical={base_test_ids == improved_test_ids}")
    assert base_val_ids == improved_val_ids, "VAL evaluation populations differ between conditions - comparison invalid"
    assert base_test_ids == improved_test_ids, "TEST evaluation populations differ between conditions - comparison invalid"

    # ======================= PRIMARY EVALUATION =======================
    print()
    print("=" * 78)
    print("PRIMARY EVALUATION (full pipeline: eligibility -> retrieval -> ranker -> re-ranking -> final validation)")
    print("=" * 78)

    print("\n--- BASE / VALIDATION ---")
    base_val = evaluate_primary_pipeline(
        base_condition.val_cases, "base/val", product_features, product_embeddings, base_condition.item_ids,
        base_condition.encoder, base_condition.user_tower, base_condition.ranker_model, base_condition.vector_index,
        config, TOP_N, K_VALUES, len(eligible_ids),
    )
    _print_primary_report(base_val)

    print("\n--- IMPROVED / VALIDATION ---")
    improved_val = evaluate_primary_pipeline(
        improved_condition.val_cases, "improved/val", product_features, product_embeddings, improved_condition.item_ids,
        improved_condition.encoder, improved_condition.user_tower, improved_condition.ranker_model, improved_condition.vector_index,
        config, TOP_N, K_VALUES, len(eligible_ids),
    )
    _print_primary_report(improved_val)

    print("\n--- VALIDATION DELTA (IMPROVED - BASE) ---")
    _delta_report(base_val, improved_val)

    print("\n--- BASE / TEST ---")
    base_test = evaluate_primary_pipeline(
        base_condition.test_cases, "base/test", product_features, product_embeddings, base_condition.item_ids,
        base_condition.encoder, base_condition.user_tower, base_condition.ranker_model, base_condition.vector_index,
        config, TOP_N, K_VALUES, len(eligible_ids),
    )
    _print_primary_report(base_test)

    print("\n--- IMPROVED / TEST ---")
    improved_test = evaluate_primary_pipeline(
        improved_condition.test_cases, "improved/test", product_features, product_embeddings, improved_condition.item_ids,
        improved_condition.encoder, improved_condition.user_tower, improved_condition.ranker_model, improved_condition.vector_index,
        config, TOP_N, K_VALUES, len(eligible_ids),
    )
    _print_primary_report(improved_test)

    print("\n--- TEST DELTA (IMPROVED - BASE) - FINAL, UNBIASED ---")
    _delta_report(base_test, improved_test)

    # ======================= PAIRED DIAGNOSTICS =======================
    print()
    print("=" * 78)
    print("PAIRED HIT OUTCOMES (test split)")
    print("=" * 78)
    base_test_recs = [
        generate_recommendations(
            c.user_features, product_features, product_embeddings, base_condition.item_ids,
            base_condition.encoder, base_condition.user_tower, base_condition.ranker_model, base_condition.vector_index, config, TOP_N,
        ).product_ids
        for c in base_condition.test_cases
    ]
    improved_test_recs = [
        generate_recommendations(
            c.user_features, product_features, product_embeddings, improved_condition.item_ids,
            improved_condition.encoder, improved_condition.user_tower, improved_condition.ranker_model, improved_condition.vector_index, config, TOP_N,
        ).product_ids
        for c in improved_condition.test_cases
    ]
    for k in (10, 20):
        outcome = _paired_hits(base_condition.test_cases, improved_condition.test_cases, base_test_recs, improved_test_recs, k)
        print(f"Hit@{k}: both={outcome['both']}  base_only={outcome['base_only']}  improved_only={outcome['improved_only']}  neither={outcome['neither']}")

    # ======================= LATENCY =======================
    print()
    print("=" * 78)
    print(f"LATENCY (backend={config.retrieval.backend}, same environment for both conditions)")
    print("=" * 78)
    if base_condition.test_cases:
        sample = base_condition.test_cases[0]
        base_latency = measure_latency(
            lambda: generate_recommendations(
                sample.user_features, product_features, product_embeddings, base_condition.item_ids,
                base_condition.encoder, base_condition.user_tower, base_condition.ranker_model, base_condition.vector_index, config, TOP_N,
            ),
            num_runs=50, warmup_runs=5,
        )
        print(f"BASE:     mean={base_latency.mean_ms:.2f}ms p50={base_latency.p50_ms:.2f}ms p95={base_latency.p95_ms:.2f}ms p99={base_latency.p99_ms:.2f}ms")
    if improved_condition.test_cases:
        sample_i = improved_condition.test_cases[0]
        improved_latency = measure_latency(
            lambda: generate_recommendations(
                sample_i.user_features, product_features, product_embeddings, improved_condition.item_ids,
                improved_condition.encoder, improved_condition.user_tower, improved_condition.ranker_model, improved_condition.vector_index, config, TOP_N,
            ),
            num_runs=50, warmup_runs=5,
        )
        print(f"IMPROVED: mean={improved_latency.mean_ms:.2f}ms p50={improved_latency.p50_ms:.2f}ms p95={improved_latency.p95_ms:.2f}ms p99={improved_latency.p99_ms:.2f}ms")

    # ======================= HISTORY TIER =======================
    print()
    print("=" * 78)
    print("HISTORY-TIER DISTRIBUTION (test split)")
    print("=" * 78)
    tier_counts = Counter(
        determine_history_tier(c.user_features.total_engagement_events, config.cold_start) for c in base_condition.test_cases
    )
    for tier in HistoryTier:
        print(f"  {tier.value:12s} {tier_counts.get(tier, 0)}")

    # ======================= QUALITATIVE EXAMPLES =======================
    print()
    print("=" * 78)
    print("QUALITATIVE EXAMPLES")
    print("=" * 78)
    both_idx = base_only_idx = improved_only_idx = neither_idx = None
    for idx, (b_case, i_case) in enumerate(zip(base_condition.test_cases, improved_condition.test_cases)):
        b_hit = bool(set(base_test_recs[idx][:10]) & b_case.target_ids)
        i_hit = bool(set(improved_test_recs[idx][:10]) & i_case.target_ids)
        if b_hit and i_hit and both_idx is None:
            both_idx = idx
        elif b_hit and not i_hit and base_only_idx is None:
            base_only_idx = idx
        elif i_hit and not b_hit and improved_only_idx is None:
            improved_only_idx = idx
        elif not b_hit and not i_hit and neither_idx is None:
            neither_idx = idx

    for label, idx in [
        ("BOTH succeed", both_idx), ("IMPROVED succeeds, BASE misses", improved_only_idx),
        ("BASE succeeds, IMPROVED misses", base_only_idx), ("BOTH miss", neither_idx),
    ]:
        print(f"\n[{label}]")
        if idx is None:
            print("  no matching example in this test split")
            continue
        b_case = base_condition.test_cases[idx]
        i_case = improved_condition.test_cases[idx]
        print(f"  user_id={b_case.user_id}  cutoff={b_case.cutoff.isoformat()}  future PURCHASE target(s)={sorted(b_case.target_ids)}")
        pp = i_case.user_features.price_profile
        if pp:
            print(f"  IMPROVED price_profile: typical={pp.typical_price:.2f} tier={pp.price_tier} source={pp.fallback_source}")
        print(f"  BASE     Top-10: {base_test_recs[idx][:10]}")
        print(f"  IMPROVED Top-10: {improved_test_recs[idx][:10]}")
        b_rank = next((r + 1 for r, pid in enumerate(base_test_recs[idx]) if pid in b_case.target_ids), None)
        i_rank = next((r + 1 for r, pid in enumerate(improved_test_recs[idx]) if pid in i_case.target_ids), None)
        print(f"  target rank: BASE={b_rank or 'not in top-' + str(TOP_N)}  IMPROVED={i_rank or 'not in top-' + str(TOP_N)}")

    # ======================= SAVE BASE ARTIFACTS =======================
    print()
    print("=" * 78)
    print("SAVING ARTIFACTS")
    print("=" * 78)
    from recommendation.ranking.features import RANKING_FEATURE_NAMES_BASE
    from recommendation.ranking.serialization import save_ranker_artifacts
    from recommendation.retrieval.two_tower.serialization import save_two_tower_artifacts

    ablation_root = resolve_path(config.paths.models_dir) / "ablation"
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    common_meta = {
        "run_id": run_id,
        "data_source": str(db_path),
        "dataset_fingerprint_sha256_16": db_fingerprint,
        "condition": "BASE",
        "recency_enabled": False,
        "include_price_features": False,
    }
    save_two_tower_artifacts(
        ablation_root / "base" / "two_tower", base_condition.user_tower, base_condition.item_tower, base_condition.encoder,
        base_condition.item_ids, base_condition.item_embeddings,
        {**common_meta, "model_version": "ablation_base_two_tower_v1", "item_numeric_dim": base_condition.encoder.item_numeric_dim, "user_numeric_dim": base_condition.encoder.user_numeric_dim},
    )
    save_ranker_artifacts(
        ablation_root / "base" / "ranker", base_condition.ranker_model, RANKING_FEATURE_NAMES_BASE,
        {**common_meta, "model_version": "ablation_base_ranker_v1", "feature_dim": 23},
    )
    print(f"saved BASE artifacts to {ablation_root / 'base'}")
    print(f"RECENCY+PRICE condition: {'reused models/sqlite_baseline/ (not duplicated)' if reuse_ok else 'freshly trained, not yet saved separately - see note'}")

    print(f"\nrun_id={run_id}")
    print("STOP - review results before commit/push.")


if __name__ == "__main__":
    main()
