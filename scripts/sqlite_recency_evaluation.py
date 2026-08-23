"""Evaluates whether recency weighting improves future-purchase
recommendation quality, measured on the temporal future-purchase protocol
(`evaluation.temporal_future_purchase`) over
`data/sqlite/backend_shaped_synthetic.db`.

Compares, on identical evaluation points (same users, same temporal
splits, same targets, same eligibility, same candidate pool):

  A. BASELINE - `features.recency` disabled (`RecencyConfig(enabled=False)`).
  B. RECENCY  - the same config, except `recency.enabled=True` at
     `config.features.recency.half_life_days` (default from
     configs/base.yaml, 21 days).

Everything else (signal weights, embedding model, eligibility rules,
candidate pool, scoring mechanism, k values) is held fixed between arms -
the only intended difference is recency (docs/data-mapping.md section 14).

**Why cosine-similarity retrieval, not Two-Tower/ranker retraining:** the
currently-trained Two-Tower/ranker artifacts under `models/` were fit
against the original, smaller synthetic catalog (different product ids/
vocabulary) - retraining them against this larger SQLite catalog would
conflate a model change with the recency effect this script isolates.
Instead, this script scores candidates by cosine similarity between each
user's `semantic_embedding` (exactly the feature recency weighting
modifies) and the same frozen Sentence Transformer product embeddings
both arms use - a direct, retraining-free way to isolate and measure
recency's effect, restricted to the same pre-retrieval eligibility gate
(`serving.eligibility`) real serving uses. It is not a stand-in for the
production ANN/Two-Tower/ranker stack's ranking quality - only for
whether recency-weighted features pull a user's representation toward
their actual future purchases.

No randomness anywhere in this script (deterministic cosine-similarity
ranking, no sampling), so results are fully reproducible run to run.
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from recommendation.data.adapters.sqlite_factory import build_sqlite_adapters
from recommendation.data.sqlite.connection import open_readonly_connection
from recommendation.data.sqlite.loader import load_events
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.embeddings.product_embeddings import get_or_compute_product_embeddings
from recommendation.evaluation.retrieval_metrics import (
    mean_hit_rate_at_k,
    mean_ndcg_at_k,
    mean_precision_at_k,
    mean_reciprocal_rank,
    mean_recall_at_k,
)
from recommendation.evaluation.temporal_future_purchase import (
    DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT,
    build_point_in_time_engagement_profile,
    build_temporal_splits,
    group_events_by_user,
    split_targets_by_eligibility,
)
from recommendation.features.product_features import build_product_features
from recommendation.features.user_features import build_user_features
from recommendation.serving.eligibility import apply_eligibility, build_eligibility_rules
from recommendation.utils.config import RecencyConfig, get_config, resolve_path

K_VALUES = [5, 10, 20]
TOP_N = 20
# Kept isolated from the original synthetic-dataset embedding cache
# (configs/base.yaml: embedding.cache_path) - this is a different, larger
# (1,200-product) catalog, never conflated with the smaller synthetic one.
SQLITE_EMBEDDING_CACHE = "data/processed/product_embeddings_sqlite.npz"


@dataclass
class ArmAccumulator:
    rankings: list[list[int]] = field(default_factory=list)
    relevant_sets: list[set[int]] = field(default_factory=list)
    no_embedding_count: int = 0
    build_times_s: list[float] = field(default_factory=list)
    distinct_categories_top20: list[int] = field(default_factory=list)
    recommended_ids_union: set[int] = field(default_factory=set)
    underfilled_count: int = 0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _rank_by_cosine(
    user_vector: np.ndarray, eligible_ids: list[int], eligible_matrix_normalized: np.ndarray, top_n: int
) -> list[int]:
    uv = user_vector / (np.linalg.norm(user_vector) + 1e-12)
    scores = eligible_matrix_normalized @ uv
    order = np.argsort(-scores)[:top_n]
    return [eligible_ids[i] for i in order]


def _evaluate_points(
    label: str,
    points: list[tuple[int, "object", frozenset[int]]],  # (user_id, cutoff, target_ids)
    events_by_user: dict[int, list],
    users_adapter,
    reviews_adapter,
    product_lookup: dict,
    product_embeddings_by_id: dict[int, np.ndarray],
    eligible_ids: list[int],
    eligible_matrix_normalized: np.ndarray,
    feature_config,
) -> ArmAccumulator:
    acc = ArmAccumulator()
    for user_id, cutoff, target_ids in points:
        history = events_by_user.get(user_id, [])
        profile = build_point_in_time_engagement_profile(user_id, history, cutoff, users_adapter, reviews_adapter)

        t0 = time.perf_counter()
        features = build_user_features(
            profile, product_lookup, product_embeddings_by_id, feature_config, reference_time=cutoff
        )
        acc.build_times_s.append(time.perf_counter() - t0)

        if features.semantic_embedding is None:
            acc.no_embedding_count += 1
            continue

        ranked = _rank_by_cosine(features.semantic_embedding, eligible_ids, eligible_matrix_normalized, TOP_N)
        acc.rankings.append(ranked)
        acc.relevant_sets.append(set(target_ids))
        acc.recommended_ids_union.update(ranked)
        if len(ranked) < TOP_N:
            acc.underfilled_count += 1

        cats = {product_lookup[pid].category_name for pid in ranked if pid in product_lookup and product_lookup[pid].category_name}
        acc.distinct_categories_top20.append(len(cats))

    return acc


def _report(label: str, acc: ArmAccumulator, eligible_catalog_size: int) -> None:
    print(f"\n[{label}]")
    print(f"  evaluation points: {len(acc.rankings)} (skipped, no embedding: {acc.no_embedding_count})")
    for k in K_VALUES:
        p = mean_precision_at_k(acc.rankings, acc.relevant_sets, k)
        r = mean_recall_at_k(acc.rankings, acc.relevant_sets, k)
        h = mean_hit_rate_at_k(acc.rankings, acc.relevant_sets, k)
        n = mean_ndcg_at_k(acc.rankings, acc.relevant_sets, k)
        print(f"  @{k:<3d} Precision={p:.4f}  Recall={r:.4f}  HitRate={h:.4f}  NDCG={n:.4f}")
    mrr = mean_reciprocal_rank(acc.rankings, acc.relevant_sets)
    print(f"  MRR={mrr:.4f}")

    if acc.distinct_categories_top20:
        print(f"  mean distinct categories in top-{TOP_N}: {statistics.mean(acc.distinct_categories_top20):.2f}")
    coverage = len(acc.recommended_ids_union) / eligible_catalog_size if eligible_catalog_size else 0.0
    print(f"  catalog coverage (union of recommended / eligible catalog): {coverage:.4f}")
    fill_rate = 1.0 - (acc.underfilled_count / len(acc.rankings)) if acc.rankings else 0.0
    print(f"  fill rate (top-{TOP_N} fully filled): {fill_rate:.4f}")

    if acc.build_times_s:
        times_ms = [t * 1000 for t in acc.build_times_s]
        print(
            f"  feature-build latency (ms): mean={statistics.mean(times_ms):.3f} "
            f"p50={_percentile(times_ms, 50):.3f} p95={_percentile(times_ms, 95):.3f} p99={_percentile(times_ms, 99):.3f}"
        )


def main() -> None:
    config = get_config()
    bundle = build_sqlite_adapters()

    con = open_readonly_connection(config.paths.data_sqlite)
    try:
        all_events = load_events(con)
    finally:
        con.close()

    events_by_user = group_events_by_user(all_events)
    user_ids = bundle.users.list_user_ids()
    splits = build_temporal_splits(events_by_user, user_ids, DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT)

    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    product_features = build_product_features(products, [], [], [])
    eligibility_rules = build_eligibility_rules(config.eligibility)
    eligibility_result = apply_eligibility([p.id for p in products], product_features, eligibility_rules)
    eligible_ids = eligibility_result.eligible_ids
    print(f"eligible catalog: {len(eligible_ids)} / {len(products)} products")

    print("Loading/encoding product embeddings for this catalog (cached after first run)...")
    encoder = SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    cache, recomputed = get_or_compute_product_embeddings(products, encoder, resolve_path(SQLITE_EMBEDDING_CACHE))
    product_embeddings_by_id = cache.as_dict()
    print(f"product embeddings ready (recomputed={recomputed}), dim={cache.embedding_dim}")

    eligible_matrix = np.stack([product_embeddings_by_id[pid] for pid in eligible_ids])
    norms = np.linalg.norm(eligible_matrix, axis=1, keepdims=True)
    eligible_matrix_normalized = eligible_matrix / np.clip(norms, 1e-12, None)

    val_points = [(s.user_id, s.val_cutoff, s.val_target_ids) for s in splits.values() if s.is_val_evaluable]
    test_points = [(s.user_id, s.test_cutoff, s.test_target_ids) for s in splits.values() if s.is_test_evaluable]
    print(f"val evaluation points: {len(val_points)} | test evaluation points: {len(test_points)}")

    all_val_targets = frozenset().union(*(t for _, _, t in val_points)) if val_points else frozenset()
    all_test_targets = frozenset().union(*(t for _, _, t in test_points)) if test_points else frozenset()
    val_target_elig = split_targets_by_eligibility(all_val_targets, product_features, eligibility_rules)
    test_target_elig = split_targets_by_eligibility(all_test_targets, product_features, eligibility_rules)
    print(
        f"val targets: {len(all_val_targets)} distinct ({len(val_target_elig.ineligible_ids)} structurally "
        f"ineligible -> can never be hit by any arm, not a modeling failure)"
    )
    print(
        f"test targets: {len(all_test_targets)} distinct ({len(test_target_elig.ineligible_ids)} structurally "
        f"ineligible -> can never be hit by any arm, not a modeling failure)"
    )

    baseline_config = config.features.model_copy(update={"recency": RecencyConfig(enabled=False)})
    recency_config = config.features.model_copy(
        update={"recency": RecencyConfig(enabled=True, half_life_days=config.features.recency.half_life_days)}
    )
    print(f"\nrecency arm half_life_days={recency_config.recency.half_life_days}")

    print("\n" + "=" * 78)
    print("BASELINE (recency disabled) vs RECENCY - VALIDATION split")
    print("=" * 78)
    base_val = _evaluate_points("baseline/val", val_points, events_by_user, bundle.users, bundle.reviews, product_lookup, product_embeddings_by_id, eligible_ids, eligible_matrix_normalized, baseline_config)
    rec_val = _evaluate_points("recency/val", val_points, events_by_user, bundle.users, bundle.reviews, product_lookup, product_embeddings_by_id, eligible_ids, eligible_matrix_normalized, recency_config)
    _report("BASELINE - val", base_val, len(eligible_ids))
    _report("RECENCY  - val", rec_val, len(eligible_ids))

    print("\n" + "=" * 78)
    print("BASELINE (recency disabled) vs RECENCY - TEST split")
    print("=" * 78)
    base_test = _evaluate_points("baseline/test", test_points, events_by_user, bundle.users, bundle.reviews, product_lookup, product_embeddings_by_id, eligible_ids, eligible_matrix_normalized, baseline_config)
    rec_test = _evaluate_points("recency/test", test_points, events_by_user, bundle.users, bundle.reviews, product_lookup, product_embeddings_by_id, eligible_ids, eligible_matrix_normalized, recency_config)
    _report("BASELINE - test", base_test, len(eligible_ids))
    _report("RECENCY  - test", rec_test, len(eligible_ids))


if __name__ == "__main__":
    main()
