"""STEP 9 fix: produce the PERSISTED offline evaluation report artifact
that `GET /v1/metrics/offline` reads (docs/data-mapping.md section 18
follow-up - see `evaluation.offline_report` module docstring for the full
before/after).

Loads the ALREADY-TRAINED `models/sqlite_baseline/` Two-Tower + ranker
artifacts (does NOT train or retrain anything - same "reuse, don't
retrain" posture as `scripts/run_pipeline.py`) and runs
the APPROVED STEP 5/7/8 temporal future-purchase evaluation protocol
(`evaluation.temporal_future_purchase.build_temporal_splits` +
`evaluation.temporal_training.build_temporal_two_tower_examples` +
`.evaluate_primary_pipeline`) UNCHANGED - the exact same evaluation
`scripts/train_sqlite_pipeline.py` runs right after training. This script
only adds: writing the full per-K breakdown (not just the few scalar
summary fields already in ranker/metadata.json) plus provenance to
`models/sqlite_baseline/offline_report.json`.

Usage:
    python scripts/generate_offline_report.py

Re-run this any time `models/sqlite_baseline/` is retrained (i.e. after
`scripts/train_sqlite_pipeline.py`) - `GET /v1/metrics/offline` refuses to
serve a report whose `run_id`/dataset fingerprint/model_version don't
match the artifacts currently loaded (provenance check), rather than
silently showing stale numbers.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation.api.dependencies import resolve_models_root
from recommendation.data.adapters.sqlite_factory import build_sqlite_adapters
from recommendation.data.sqlite.connection import open_readonly_connection
from recommendation.data.sqlite.loader import load_events
from recommendation.embeddings.product_embeddings import get_or_compute_product_embeddings
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.evaluation.offline_report import (
    OfflineEvalSplitReport,
    OfflineEvaluationReport,
    REPORT_SCHEMA_VERSION,
    save_offline_report,
)
from recommendation.evaluation.temporal_future_purchase import DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT, build_temporal_splits, group_events_by_user
from recommendation.evaluation.temporal_training import build_temporal_two_tower_examples, evaluate_primary_pipeline
from recommendation.features.price import build_price_catalog_context
from recommendation.features.product_features import build_product_features
from recommendation.ranking.serialization import load_ranker_artifacts
from recommendation.retrieval.index.factory import build_vector_index
from recommendation.retrieval.two_tower.serialization import load_two_tower_artifacts
from recommendation.serving.eligibility import apply_eligibility, build_eligibility_rules
from recommendation.utils.config import get_config, resolve_path
from recommendation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _to_split_report(report, split_name: str) -> OfflineEvalSplitReport:
    return OfflineEvalSplitReport(
        split_name=split_name,
        num_cases=report.num_cases,
        precision_at_k=dict(report.precision_at_k),
        recall_at_k=dict(report.recall_at_k),
        hit_rate_at_k=dict(report.hit_rate_at_k),
        ndcg_at_k=dict(report.ndcg_at_k),
        mrr=report.mrr,
        mean_distinct_categories=report.mean_distinct_categories,
        catalog_coverage=report.catalog_coverage,
        mean_fill_rate=report.mean_fill_rate,
    )


def main() -> None:
    config = get_config()
    setup_logging(config.log_level)

    if config.paths.data_source != "sqlite":
        raise SystemExit(
            f"paths.data_source={config.paths.data_source!r} - this script only supports the 'sqlite' "
            "data source (the approved STEP 5/7/8 temporal future-purchase protocol requires event "
            "timestamps the synthetic V1 generator does not provide)."
        )

    models_root = resolve_models_root(config)
    two_tower_dir = models_root / "two_tower"
    ranker_dir = models_root / "ranker"
    if not two_tower_dir.exists() or not ranker_dir.exists():
        raise SystemExit(
            f"{models_root} does not contain trained two_tower/ranker artifacts - "
            "run scripts/train_sqlite_pipeline.py first. This script does not train anything."
        )

    print("=" * 78)
    print("Generating persisted offline evaluation report (reuses already-trained artifacts, no training)")
    print("=" * 78)

    db_path = resolve_path(config.paths.data_sqlite)
    db_fingerprint = _sha256_file(db_path)
    print(f"SQLite path: {db_path}")
    print(f"dataset fingerprint (sha256[:16]): {db_fingerprint}")

    two_tower_artifacts = load_two_tower_artifacts(two_tower_dir)
    ranker_artifacts = load_ranker_artifacts(ranker_dir)
    print(f"loaded Two-Tower artifacts from {two_tower_dir} (model_version={two_tower_artifacts.metadata.get('model_version')})")
    print(f"loaded ranker artifacts from {ranker_dir} (model_version={ranker_artifacts.metadata.get('model_version')})")

    run_id = ranker_artifacts.metadata.get("run_id", "")
    artifact_fingerprint = ranker_artifacts.metadata.get("dataset_fingerprint_sha256_16", "")
    if artifact_fingerprint and artifact_fingerprint != db_fingerprint:
        raise SystemExit(
            f"models/sqlite_baseline artifacts were trained against a dataset fingerprint "
            f"({artifact_fingerprint}) different from the currently-loaded database ({db_fingerprint}) - "
            "retrain (scripts/train_sqlite_pipeline.py) before generating an offline report."
        )

    bundle = build_sqlite_adapters()
    con = open_readonly_connection(db_path)
    try:
        all_events = load_events(con)
    finally:
        con.close()

    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    user_ids = bundle.users.list_user_ids()
    product_features = build_product_features(products, [], [], [])
    eligibility_rules = build_eligibility_rules(config.eligibility)
    eligible_ids = apply_eligibility([p.id for p in products], product_features, eligibility_rules).eligible_ids

    events_by_user = group_events_by_user(all_events)
    splits = build_temporal_splits(events_by_user, user_ids, DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT)

    price_context = build_price_catalog_context(products)
    print(f"recency.enabled={config.features.recency.enabled} half_life_days={config.features.recency.half_life_days}")
    print(f"include_price_features (from loaded encoder)={two_tower_artifacts.encoder.include_price_features}")

    st_encoder = SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    embed_cache_path = resolve_path("data/processed/product_embeddings_sqlite.npz")
    cache, recomputed = get_or_compute_product_embeddings(products, st_encoder, embed_cache_path)
    product_embeddings = cache.as_dict()
    print(f"product embeddings: {len(product_embeddings)} products, recomputed={recomputed}")

    _, _, val_cases, test_cases = build_temporal_two_tower_examples(
        events_by_user, splits, bundle.users, bundle.reviews, product_lookup, product_embeddings, config.features, price_context
    )
    print(f"val eval cases: {len(val_cases)}  test eval cases: {len(test_cases)}")

    vector_index = build_vector_index(config.retrieval)
    vector_index.build(two_tower_artifacts.item_ids, two_tower_artifacts.item_embeddings)

    k_values = config.ranking.eval_k_values
    top_n = config.api.default_recommendation_count

    print(f"\nEvaluating VALIDATION split ({len(val_cases)} cases, top_n={top_n}, k_values={k_values})...")
    val_primary = evaluate_primary_pipeline(
        val_cases, "val", product_features, product_embeddings, two_tower_artifacts.item_ids,
        two_tower_artifacts.encoder, two_tower_artifacts.user_tower, ranker_artifacts.model, vector_index,
        config, top_n, k_values, len(eligible_ids),
    )
    print(f"  Recall@10={val_primary.recall_at_k.get(10, float('nan')):.4f}  NDCG@10={val_primary.ndcg_at_k.get(10, float('nan')):.4f}  MRR={val_primary.mrr:.4f}")

    print(f"\nEvaluating TEST split ({len(test_cases)} cases, top_n={top_n}, k_values={k_values})...")
    test_primary = evaluate_primary_pipeline(
        test_cases, "test", product_features, product_embeddings, two_tower_artifacts.item_ids,
        two_tower_artifacts.encoder, two_tower_artifacts.user_tower, ranker_artifacts.model, vector_index,
        config, top_n, k_values, len(eligible_ids),
    )
    print(f"  Recall@10={test_primary.recall_at_k.get(10, float('nan')):.4f}  NDCG@10={test_primary.ndcg_at_k.get(10, float('nan')):.4f}  MRR={test_primary.mrr:.4f}")

    report = OfflineEvaluationReport(
        schema_version=REPORT_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc),
        run_id=run_id,
        ranker_model_version=str(ranker_artifacts.metadata.get("model_version", "unknown")),
        two_tower_model_version=str(two_tower_artifacts.metadata.get("model_version", "unknown")),
        data_source=str(db_path),
        dataset_fingerprint_sha256_16=db_fingerprint,
        recency_enabled=config.features.recency.enabled,
        recency_half_life_days=config.features.recency.half_life_days,
        include_price_features=two_tower_artifacts.encoder.include_price_features,
        price_tier_boundaries=list(price_context.catalog_tier_boundaries),
        k_values=list(k_values),
        top_n=top_n,
        val_report=_to_split_report(val_primary, "val"),
        test_report=_to_split_report(test_primary, "test"),
    )

    out_path = models_root / "offline_report.json"
    save_offline_report(out_path, report)
    print(f"\nsaved persisted offline evaluation report to {out_path}")
    print(f"run_id={run_id}  dataset_fingerprint={db_fingerprint}")


if __name__ == "__main__":
    main()
