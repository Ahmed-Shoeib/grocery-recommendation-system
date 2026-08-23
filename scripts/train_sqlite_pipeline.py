"""Train the complete recommendation pipeline from scratch against
`data/sqlite/backend_shaped_synthetic.db`, using the temporal
future-purchase protocol for train/validation/test separation and the
recency+price-aware feature set - see docs/data-mapping.md section 16 for
the full design writeup.

Produces new artifacts under `models/sqlite_baseline/` - the pre-existing
`models/two_tower`/`models/ranker` (trained on the original synthetic
7/8/23-dimension feature set) are never read or overwritten by this
script.

Usage:
    python scripts/train_sqlite_pipeline.py

Deterministic given the same database file + config + seed (Two-Tower/
ranker random seeds come from `config.two_tower.random_seed`/
`config.ranking.random_seed`, unchanged from their existing values).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import tensorflow as tf

from recommendation.data.adapters.sqlite_factory import build_sqlite_adapters
from recommendation.data.schemas.events import ActionType
from recommendation.data.sqlite.connection import open_readonly_connection
from recommendation.data.sqlite.loader import load_events
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.embeddings.product_embeddings import get_or_compute_product_embeddings
from recommendation.evaluation.latency import measure_latency
from recommendation.evaluation.temporal_future_purchase import (
    DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT,
    TemporalEligibilityTier,
    audit_no_leakage,
    build_temporal_splits,
    events_before_cutoff,
    group_events_by_user,
    split_targets_by_eligibility,
)
from recommendation.evaluation.temporal_training import (
    PrimaryEvalReport,
    TemporalRetrievalReport,
    all_purchased_product_ids_by_user,
    build_temporal_ranking_dataset,
    build_temporal_two_tower_examples,
    evaluate_primary_pipeline,
    evaluate_temporal_retrieval,
)
from recommendation.features.price import build_price_catalog_context
from recommendation.features.product_features import build_product_features
from recommendation.ranking.features import RANKING_FEATURE_NAMES, build_ranking_feature_vector
from recommendation.ranking.model import build_ranker_model
from recommendation.ranking.serialization import save_ranker_artifacts
from recommendation.retrieval.index.factory import build_vector_index, candidate_pool_size
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.model import TwoTowerModel, build_item_tower, build_user_tower
from recommendation.retrieval.two_tower.serialization import save_two_tower_artifacts
from recommendation.serving.cold_start import HistoryTier, determine_history_tier
from recommendation.serving.eligibility import apply_eligibility, build_eligibility_rules
from recommendation.serving.pipeline import generate_recommendations
from recommendation.utils.config import get_config, resolve_path
from recommendation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

K_VALUES = [5, 10, 20]
TOP_N = 20


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _fit_encoder(products, users_adapter, embedding_dim: int) -> TwoTowerFeatureEncoder:
    observed_age_groups = sorted(
        {p.age_group for uid in users_adapter.list_user_ids() if (p := users_adapter.get_user_profile(uid)) and p.age_group}
    )
    return TwoTowerFeatureEncoder.fit(
        category_names=[p.category_name for p in products if p.category_name],
        brand_names=[p.brand for p in products if p.brand],
        age_groups=observed_age_groups,
        prices=[p.price for p in products],
        embedding_dim=embedding_dim,
    )


def _examples_to_arrays(examples, encoder, product_features, product_embeddings):
    user_batch = encoder.encode_user_batch([e.user_features for e in examples])
    item_batch = encoder.encode_item_batch([e.product_id for e in examples], product_features, product_embeddings)
    return user_batch, item_batch


def _print_retrieval_report(report: TemporalRetrievalReport) -> None:
    for k in sorted(report.recall_at_k):
        print(f"  Recall@{k}: {report.recall_at_k[k]:.4f}   HitRate@{k}: {report.hit_rate_at_k[k]:.4f}")


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


def main() -> None:
    config = get_config()
    setup_logging(config.log_level)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    # ================= SECTION 1: training data source guard =================
    print("=" * 78)
    print("SECTION 1: training data source")
    print("=" * 78)
    db_path = resolve_path(config.paths.data_sqlite)
    print(f"SQLite path: {db_path}")
    assert db_path.name == "backend_shaped_synthetic.db", f"expected backend_shaped_synthetic.db, got {db_path.name}"
    assert "ecommerce.db" not in str(db_path), "must never train against ecommerce.db"
    db_fingerprint = _sha256_file(db_path)
    print(f"dataset fingerprint (sha256[:16]): {db_fingerprint}")

    bundle = build_sqlite_adapters()
    con = open_readonly_connection(db_path)
    try:
        all_events = load_events(con)
    finally:
        con.close()

    products = bundle.products.list_products()
    user_ids = bundle.users.list_user_ids()
    action_counts = Counter(e.action_type.value for e in all_events)
    purchasing_users = {e.user_id for e in all_events if e.action_type == ActionType.PURCHASE}

    product_features = build_product_features(products, [], [], [])
    eligibility_rules = build_eligibility_rules(config.eligibility)
    eligibility_result = apply_eligibility([p.id for p in products], product_features, eligibility_rules)
    eligible_ids = eligibility_result.eligible_ids

    print(f"user count: {len(user_ids)}")
    print(f"product count: {len(products)}")
    print(f"event count: {len(all_events)}")
    for action, count in sorted(action_counts.items()):
        print(f"  {action:12s}: {count}")
    print(f"purchase count: {action_counts.get('PURCHASE', 0)}")
    print(f"users with >=1 purchase: {len(purchasing_users)}")
    print(f"eligible product count: {len(eligible_ids)} / {len(products)}")

    events_by_user = group_events_by_user(all_events)

    # History-tier distribution (diagnostic - full current visible history, not point-in-time).
    tier_counts: Counter[HistoryTier] = Counter()
    for uid in user_ids:
        n_events = len(events_by_user.get(uid, []))
        tier_counts[determine_history_tier(n_events, config.cold_start)] += 1
    print("users by history tier (full current history, diagnostic only):")
    for tier in HistoryTier:
        print(f"  {tier.value:12s} {tier_counts.get(tier, 0)}")

    # ================= SECTION 2/3: temporal splits =================
    print()
    print("=" * 78)
    print("SECTION 2/3: temporal train/validation/test split")
    print("=" * 78)
    splits = build_temporal_splits(events_by_user, user_ids, DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT)
    tier_dist = Counter(s.tier for s in splits.values())
    for tier in TemporalEligibilityTier:
        print(f"  {tier.value:20s} {tier_dist.get(tier, 0)}")
    val_evaluable = [s for s in splits.values() if s.is_val_evaluable]
    test_evaluable = [s for s in splits.values() if s.is_test_evaluable]
    print(f"val-evaluable users: {len(val_evaluable)}")
    print(f"test-evaluable users: {len(test_evaluable)}")

    # ================= SECTION 4/5: recency + price config =================
    print()
    print("=" * 78)
    print("SECTION 4/5: recency + price configuration")
    print("=" * 78)
    print(f"recency.enabled={config.features.recency.enabled} half_life_days={config.features.recency.half_life_days}")
    price_context = build_price_catalog_context(products)
    print(f"catalog price tier boundaries (lower, upper): {price_context.catalog_tier_boundaries}")
    print(f"catalog median price: {price_context.catalog_median_price:.2f}")

    # ================= SECTION 6: product semantic embeddings =================
    print()
    print("=" * 78)
    print("SECTION 6: product semantic embeddings")
    print("=" * 78)
    st_encoder = SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    embed_cache_path = resolve_path("data/processed/product_embeddings_sqlite.npz")
    cache, recomputed = get_or_compute_product_embeddings(products, st_encoder, embed_cache_path)
    product_embeddings = cache.as_dict()
    all_vecs = np.stack(list(product_embeddings.values()))
    print(f"products represented: {len(product_embeddings)} / {len(products)}")
    print(f"embedding dim: {cache.embedding_dim} (recomputed={recomputed})")
    print(f"NaN present: {bool(np.isnan(all_vecs).any())}  Inf present: {bool(np.isinf(all_vecs).any())}")
    assert set(product_embeddings.keys()) == {p.id for p in products}, "product id / embedding mismatch"

    product_lookup = {p.id: p for p in products}

    # ================= SECTION: leakage audit =================
    print()
    print("=" * 78)
    print("SECTION: leakage audit (reused, unmodified)")
    print("=" * 78)
    total_checked, total_failed = 0, 0
    for s in splits.values():
        for cutoff, target_ids in ((s.val_cutoff, s.val_target_ids), (s.test_cutoff, s.test_target_ids)):
            if cutoff is None:
                continue
            history = events_before_cutoff(events_by_user.get(s.user_id, []), cutoff)
            result = audit_no_leakage(s.user_id, history, cutoff, target_ids)
            total_checked += 1
            total_failed += 0 if result.ok else 1
    print(f"evaluation points audited: {total_checked}  violations: {total_failed}")
    assert total_failed == 0, "leakage audit failed - aborting training"

    # ================= SECTION 7: build temporal Two-Tower examples =================
    print()
    print("=" * 78)
    print("SECTION 7: Two-Tower training examples (temporal)")
    print("=" * 78)
    _set_seeds(config.two_tower.random_seed)
    train_examples, val_loss_examples, val_cases, test_cases = build_temporal_two_tower_examples(
        events_by_user, splits, bundle.users, bundle.reviews, product_lookup, product_embeddings, config.features, price_context
    )
    print(f"train examples (positives, in-batch-softmax negatives): {len(train_examples)}")
    print(f"val-loss examples: {len(val_loss_examples)}")
    print(f"val eval cases: {len(val_cases)}  test eval cases: {len(test_cases)}")
    if not train_examples:
        raise SystemExit("No temporal training examples constructed - aborting.")

    # Price fallback distribution over the eval populations (section 5).
    fallback_counts_val = Counter(c.user_features.price_profile.fallback_source for c in val_cases if c.user_features.price_profile)
    fallback_counts_test = Counter(c.user_features.price_profile.fallback_source for c in test_cases if c.user_features.price_profile)
    print(f"price fallback distribution (val cases): {dict(fallback_counts_val)}")
    print(f"price fallback distribution (test cases): {dict(fallback_counts_test)}")

    # ================= SECTION 7 (cont'd): train Two-Tower from scratch =================
    print()
    print("=" * 78)
    print("SECTION 7 (cont'd): Two-Tower training")
    print("=" * 78)
    encoder = _fit_encoder(products, bundle.users, config.embedding.embedding_dim)
    print(f"item_numeric_dim={encoder.item_numeric_dim}  user_numeric_dim={encoder.user_numeric_dim}")
    assert encoder.item_numeric_dim == 9, f"expected item_numeric_dim=9, got {encoder.item_numeric_dim}"
    assert encoder.user_numeric_dim == 9, f"expected user_numeric_dim=9, got {encoder.user_numeric_dim}"

    train_user_batch, train_item_batch = _examples_to_arrays(train_examples, encoder, product_features, product_embeddings)
    val_user_batch, val_item_batch = (
        _examples_to_arrays(val_loss_examples, encoder, product_features, product_embeddings) if val_loss_examples else (None, None)
    )

    tt_config = config.two_tower
    effective_batch_size = min(tt_config.batch_size, len(train_examples))
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_user_batch, train_item_batch))
        .shuffle(buffer_size=len(train_examples), seed=tt_config.random_seed)
        .batch(effective_batch_size, drop_remainder=True)
    )
    val_ds = None
    if val_loss_examples:
        val_batch_size = min(tt_config.batch_size, len(val_loss_examples))
        val_ds = tf.data.Dataset.from_tensor_slices((val_user_batch, val_item_batch)).batch(val_batch_size, drop_remainder=True)

    user_tower = build_user_tower(encoder, tt_config)
    item_tower = build_item_tower(encoder, tt_config)
    tt_model = TwoTowerModel(user_tower=user_tower, item_tower=item_tower, temperature=tt_config.temperature)
    tt_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tt_config.learning_rate))

    callbacks = []
    if val_ds is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=tt_config.early_stopping_patience, restore_best_weights=True))
    fit_kwargs = {"epochs": tt_config.epochs, "verbose": 0, "callbacks": callbacks}
    if val_ds is not None:
        fit_kwargs["validation_data"] = val_ds

    print(
        f"batch_size={effective_batch_size} epochs={tt_config.epochs} optimizer=Adam lr={tt_config.learning_rate} "
        f"seed={tt_config.random_seed} early_stopping_patience={tt_config.early_stopping_patience} "
        f"checkpoint_criterion=val_loss (restore_best_weights)"
    )
    tt_history = tt_model.fit(train_ds, **fit_kwargs)
    print(f"epochs run: {len(tt_history.history.get('loss', []))}")
    if tt_history.history.get("loss"):
        print(f"final train loss={tt_history.history['loss'][-1]:.4f} in_batch_accuracy={tt_history.history['in_batch_accuracy'][-1]:.4f}")
    if tt_history.history.get("val_loss"):
        print(f"final val loss={tt_history.history['val_loss'][-1]:.4f} in_batch_accuracy={tt_history.history['val_in_batch_accuracy'][-1]:.4f}")

    # Full-catalog item embeddings for the trained item tower.
    item_ids = [p.id for p in products]
    item_batch_full = encoder.encode_item_batch(item_ids, product_features, product_embeddings)
    item_embeddings = np.asarray(item_tower.predict(item_batch_full, verbose=0))
    norms = np.linalg.norm(item_embeddings, axis=1)
    print(f"item embeddings: {item_embeddings.shape}  norm range=[{norms.min():.4f}, {norms.max():.4f}]")

    print("\n--- Two-Tower brute-force retrieval report: VALIDATION ---")
    val_retrieval_report = evaluate_temporal_retrieval(val_cases, "val", user_tower, item_embeddings, item_ids, encoder, K_VALUES)
    _print_retrieval_report(val_retrieval_report)
    print("\n--- Two-Tower brute-force retrieval report: TEST ---")
    test_retrieval_report = evaluate_temporal_retrieval(test_cases, "test", user_tower, item_embeddings, item_ids, encoder, K_VALUES)
    _print_retrieval_report(test_retrieval_report)

    # ================= SECTION 9: rebuild retrieval index =================
    print()
    print("=" * 78)
    print(f"SECTION 9: retrieval index (backend={config.retrieval.backend})")
    print("=" * 78)
    vector_index = build_vector_index(config.retrieval)
    vector_index.build(item_ids, item_embeddings)
    print(f"indexed products: {vector_index.size}  dim={item_embeddings.shape[1]}")
    assert vector_index.size == len(item_ids), "index size mismatch"
    assert len(set(item_ids)) == len(item_ids), "duplicate item ids"

    # ================= SECTION 8/10: negative sampling + ranker training =================
    print()
    print("=" * 78)
    print("SECTION 8/10: ranker training examples (temporal, leakage-safe negatives)")
    print("=" * 78)
    _set_seeds(config.ranking.random_seed)
    all_purchased_by_user = all_purchased_product_ids_by_user(events_by_user)
    pool_size = candidate_pool_size(config.retrieval, limit=config.api.default_recommendation_count, catalog_size=len(item_ids))
    print(f"candidate pool size (VectorIndex retrieval depth): {pool_size}")
    print(
        "negative sampling: drawn from what VectorIndex.search retrieves for that example's OWN point-in-time "
        "query, excluding every product the user EVER purchases (past or future relative to this example's cutoff) - "
        "so a future purchase is never usable as a hard negative."
    )

    rk_train_examples, rk_val_loss_examples = build_temporal_ranking_dataset(
        train_examples, val_cases, all_purchased_by_user, product_features, product_embeddings,
        encoder, user_tower, vector_index, pool_size, config.ranking,
    )
    num_pos = sum(1 for e in rk_train_examples if e.label == 1)
    num_neg = len(rk_train_examples) - num_pos
    print(f"ranker train examples: {len(rk_train_examples)} ({num_pos} positive / {num_neg} negative)")
    print(f"ranker val-loss examples: {len(rk_val_loss_examples)}")
    if not rk_train_examples:
        raise SystemExit("No ranker training examples constructed - aborting.")

    X_train = np.stack([e.features for e in rk_train_examples])
    y_train = np.array([e.label for e in rk_train_examples], dtype=np.float32)
    print(f"ranker feature dimension: {X_train.shape[1]}")
    assert X_train.shape[1] == 29, f"expected 29 ranker features, got {X_train.shape[1]}"
    assert list(RANKING_FEATURE_NAMES) == RANKING_FEATURE_NAMES and len(RANKING_FEATURE_NAMES) == 29

    X_val = y_val = None
    if rk_val_loss_examples:
        X_val = np.stack([e.features for e in rk_val_loss_examples])
        y_val = np.array([e.label for e in rk_val_loss_examples], dtype=np.float32)

    rk_config = config.ranking
    ranker_model = build_ranker_model(input_dim=X_train.shape[1], config=rk_config)
    ranker_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=rk_config.learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    rk_callbacks = []
    rk_fit_kwargs = {"epochs": rk_config.epochs, "batch_size": min(rk_config.batch_size, len(X_train)), "verbose": 0}
    if X_val is not None and len(X_val) > 0:
        rk_callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=rk_config.early_stopping_patience, restore_best_weights=True))
        rk_fit_kwargs["validation_data"] = (X_val, y_val)
    rk_fit_kwargs["callbacks"] = rk_callbacks

    print(
        f"batch_size={rk_fit_kwargs['batch_size']} epochs={rk_config.epochs} optimizer=Adam lr={rk_config.learning_rate} "
        f"seed={rk_config.random_seed} negatives_per_positive={rk_config.negatives_per_positive} "
        f"early_stopping_patience={rk_config.early_stopping_patience} checkpoint_criterion=val_loss"
    )
    rk_history = ranker_model.fit(X_train, y_train, **rk_fit_kwargs)
    print(f"epochs run: {len(rk_history.history.get('loss', []))}")
    if rk_history.history.get("loss"):
        print(f"final train loss={rk_history.history['loss'][-1]:.4f} AUC={rk_history.history['auc'][-1]:.4f}")
    if rk_history.history.get("val_loss"):
        print(f"final val loss={rk_history.history['val_loss'][-1]:.4f} AUC={rk_history.history['val_auc'][-1]:.4f}")

    # ================= SECTION 12/13: primary offline evaluation (full pipeline) =================
    print()
    print("=" * 78)
    print("SECTION 12/13: PRIMARY offline evaluation (full pipeline: eligibility -> retrieval -> ranker -> re-ranking -> final validation)")
    print("=" * 78)

    all_val_targets = frozenset().union(*(c.target_ids for c in val_cases)) if val_cases else frozenset()
    all_test_targets = frozenset().union(*(c.target_ids for c in test_cases)) if test_cases else frozenset()
    val_target_elig = split_targets_by_eligibility(all_val_targets, product_features, eligibility_rules)
    test_target_elig = split_targets_by_eligibility(all_test_targets, product_features, eligibility_rules)
    print(f"VAL:  evaluation points={len(val_cases)}  distinct targets={len(all_val_targets)}  eligible={len(val_target_elig.eligible_ids)}  ineligible={len(val_target_elig.ineligible_ids)}")
    print(f"TEST: evaluation points={len(test_cases)}  distinct targets={len(all_test_targets)}  eligible={len(test_target_elig.eligible_ids)}  ineligible={len(test_target_elig.ineligible_ids)}")
    print("(ineligible targets were structurally impossible to recommend regardless of model quality - reported separately, not as ordinary misses)")

    print("\n--- VALIDATION (primary metrics) ---")
    val_primary = evaluate_primary_pipeline(
        val_cases, "val", product_features, product_embeddings, item_ids, encoder, user_tower, ranker_model, vector_index, config, TOP_N, K_VALUES, len(eligible_ids)
    )
    _print_primary_report(val_primary)

    print("\n--- TEST (primary metrics - final, unbiased) ---")
    test_primary = evaluate_primary_pipeline(
        test_cases, "test", product_features, product_embeddings, item_ids, encoder, user_tower, ranker_model, vector_index, config, TOP_N, K_VALUES, len(eligible_ids)
    )
    _print_primary_report(test_primary)

    # ================= SECTION 16: latency =================
    print()
    print("=" * 78)
    print(f"SECTION 16: latency (backend={config.retrieval.backend}, environment=native Windows dev)")
    print("=" * 78)
    if test_cases:
        sample_case = test_cases[0]
        latency_report = measure_latency(
            lambda: generate_recommendations(
                sample_case.user_features, product_features, product_embeddings, item_ids,
                encoder, user_tower, ranker_model, vector_index, config, TOP_N,
            ),
            num_runs=50, warmup_runs=5,
        )
        print(f"end-to-end generate_recommendations: mean={latency_report.mean_ms:.2f}ms p50={latency_report.p50_ms:.2f}ms p95={latency_report.p95_ms:.2f}ms p99={latency_report.p99_ms:.2f}ms")

    # ================= SECTION 17: history-tier diagnostics =================
    print()
    print("=" * 78)
    print("SECTION 17: performance by history tier (test split, where evaluable)")
    print("=" * 78)
    tier_by_user: dict[int, HistoryTier] = {}
    for case in test_cases:
        tier_by_user[case.user_id] = determine_history_tier(case.user_features.total_engagement_events, config.cold_start)
    for tier in HistoryTier:
        tier_cases = [c for c in test_cases if tier_by_user.get(c.user_id) == tier]
        if not tier_cases:
            print(f"  {tier.value:12s}: not evaluable in this split (no future-purchase target available for this cohort)")
            continue
        report = evaluate_primary_pipeline(
            tier_cases, f"test-{tier.value}", product_features, product_embeddings, item_ids, encoder, user_tower, ranker_model, vector_index, config, TOP_N, K_VALUES, len(eligible_ids)
        )
        print(f"  {tier.value:12s}: n={report.num_cases}  Recall@10={report.recall_at_k[10]:.4f}  NDCG@10={report.ndcg_at_k[10]:.4f}  MRR={report.mrr:.4f}")

    # ================= SECTION 18: qualitative examples =================
    print()
    print("=" * 78)
    print("SECTION 18: qualitative examples")
    print("=" * 78)
    used: set[int] = set()

    def _pick(predicate):
        for c in test_cases:
            if c.user_id in used:
                continue
            if predicate(c):
                used.add(c.user_id)
                return c
        return None

    strong = _pick(lambda c: determine_history_tier(c.user_features.total_engagement_events, config.cold_start) == HistoryTier.STRONG)
    sparse = _pick(lambda c: determine_history_tier(c.user_features.total_engagement_events, config.cold_start) == HistoryTier.SPARSE)
    budget = _pick(lambda c: c.user_features.price_profile and c.user_features.price_profile.price_tier == "budget")
    premium = _pick(lambda c: c.user_features.price_profile and c.user_features.price_profile.price_tier == "premium")
    any_other = _pick(lambda c: True)

    for label, case in [("strong history", strong), ("sparse history", sparse), ("budget-tier price profile", budget), ("premium-tier price profile", premium), ("other", any_other)]:
        print(f"\n[{label}]")
        if case is None:
            print("  no matching example in test split")
            continue
        pp = case.user_features.price_profile
        print(f"  user_id={case.user_id}  cutoff={case.cutoff.isoformat()}")
        print(f"  total_engagement_events={case.user_features.total_engagement_events}  purchase_count={case.user_features.purchase_count}")
        if pp:
            print(f"  price_profile: typical={pp.typical_price:.2f} tier={pp.price_tier} source={pp.fallback_source} n_purchases={pp.supporting_purchase_count}")
        print(f"  future PURCHASE target(s): {sorted(case.target_ids)}")
        result = generate_recommendations(case.user_features, product_features, product_embeddings, item_ids, encoder, user_tower, ranker_model, vector_index, config, TOP_N)
        print(f"  Top-{TOP_N} recommendations: {result.product_ids}")
        hit_rank = next((i + 1 for i, pid in enumerate(result.product_ids) if pid in case.target_ids), None)
        print(f"  rank of target in Top-{TOP_N}: {hit_rank if hit_rank else 'not in top-' + str(TOP_N)}")

    # ================= SECTION 19: save artifacts =================
    print()
    print("=" * 78)
    print("SECTION 19: saving artifacts")
    print("=" * 78)
    models_root = resolve_path(config.paths.models_dir) / "sqlite_baseline"
    tt_out_dir = models_root / "two_tower"
    rk_out_dir = models_root / "ranker"
    idx_out_path = models_root / "vector_index" / ("faiss_index.bin" if config.retrieval.backend == "faiss" else "scann_index")

    common_metadata = {
        "run_id": run_id,
        "data_source": str(db_path),
        "dataset_fingerprint_sha256_16": db_fingerprint,
        "num_users": len(user_ids),
        "num_products": len(products),
        "num_events": len(all_events),
        "recency_enabled": config.features.recency.enabled,
        "recency_half_life_days": config.features.recency.half_life_days,
        "price_tier_boundaries": list(price_context.catalog_tier_boundaries),
        "embedding_dim": config.embedding.embedding_dim,
        "sentence_transformer_model": config.embedding.sentence_transformer_model,
    }

    tt_metadata = {
        **common_metadata,
        "model_version": "sqlite_baseline_two_tower_v1",
        "output_dim": config.two_tower.output_dim,
        "random_seed": config.two_tower.random_seed,
        "item_numeric_dim": encoder.item_numeric_dim,
        "user_numeric_dim": encoder.user_numeric_dim,
        "num_train_examples": len(train_examples),
        "num_val_loss_examples": len(val_loss_examples),
        "val_recall_at_10": val_retrieval_report.recall_at_k.get(10),
        "test_recall_at_10": test_retrieval_report.recall_at_k.get(10),
    }
    save_two_tower_artifacts(tt_out_dir, user_tower, item_tower, encoder, item_ids, item_embeddings, tt_metadata)
    print(f"saved Two-Tower artifacts to {tt_out_dir}")

    idx_out_path.parent.mkdir(parents=True, exist_ok=True)
    vector_index.save(idx_out_path)
    print(f"saved retrieval index ({config.retrieval.backend}) to {idx_out_path}")

    rk_metadata = {
        **common_metadata,
        "model_version": "sqlite_baseline_ranker_v1",
        "random_seed": config.ranking.random_seed,
        "feature_dim": X_train.shape[1],
        "num_train_examples": len(rk_train_examples),
        "num_train_positive": num_pos,
        "num_train_negative": num_neg,
        "pool_size": pool_size,
        "negatives_per_positive": rk_config.negatives_per_positive,
        "val_recall_at_10": val_primary.recall_at_k.get(10),
        "val_ndcg_at_10": val_primary.ndcg_at_k.get(10),
        "val_mrr": val_primary.mrr,
        "test_recall_at_10": test_primary.recall_at_k.get(10),
        "test_ndcg_at_10": test_primary.ndcg_at_k.get(10),
        "test_mrr": test_primary.mrr,
    }
    save_ranker_artifacts(rk_out_dir, ranker_model, RANKING_FEATURE_NAMES, rk_metadata)
    print(f"saved ranker artifacts to {rk_out_dir}")

    print(f"\nrun_id={run_id}")
    print("Reproduction command: python scripts/train_sqlite_pipeline.py")
    print("(deterministic given the same data/sqlite/backend_shaped_synthetic.db + configs/base.yaml + the seeds above)")


if __name__ == "__main__":
    main()
