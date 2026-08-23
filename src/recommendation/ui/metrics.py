"""Offline evaluation for the dashboard's legacy Metrics/Debug section -
reuses the Two-Tower/ranker/pipeline evaluation machinery unchanged
(`retrieval.two_tower.splitting`/`.examples`, `serving.pipeline`,
`serving.evaluation`), the exact same non-temporal leave-one-out protocol
`scripts/run_pipeline.py` reports. Nothing here invents a new metric or a
production/online metric (CTR, impressions, conversion) - there is no
event-tracking pipeline to compute those from (docs/data-mapping.md
section 7/8). Every value this module returns is an OFFLINE,
synthetic-dataset metric.

**Legacy / not on the live serving path** (docs/data-mapping.md section
18.1). `GET /v1/metrics/offline` no longer calls `compute_offline_metrics`
synchronously per request - that was ~88s against the SQLite baseline and
reported the OLD non-temporal protocol, not the temporal future-purchase
protocol the currently-served `models/sqlite_baseline/` artifacts are
actually trained/evaluated against. The route now does a cheap read of a
report PERSISTED by `scripts/generate_offline_report.py`
(`evaluation.offline_report`); `api.routes`/`ui.dashboard` no longer import
this module. Kept as-is (not deleted or refactored) - this file and its
test (`tests/test_ui_metrics.py`) still exercise the non-temporal
leave-one-out protocol machinery directly, independent of the live route.
"""

from __future__ import annotations

from dataclasses import dataclass

from recommendation.api.dependencies import RecommendationService
from recommendation.retrieval.two_tower.examples import build_eval_cases
from recommendation.retrieval.two_tower.splitting import build_user_splits
from recommendation.serving.evaluation import PipelineEvalReport, evaluate_pipeline
from recommendation.serving.pipeline import generate_recommendations


@dataclass
class OfflineMetricsSummary:
    num_eval_users: int
    val_report: PipelineEvalReport
    test_report: PipelineEvalReport


def compute_offline_metrics(service: RecommendationService, top_n: int) -> OfflineMetricsSummary:
    k_values = service.config.ranking.eval_k_values
    splits = build_user_splits(
        service.engagement_profiles,
        min_distinct_products_for_holdout=service.config.two_tower.min_distinct_products_for_holdout,
        random_seed=service.config.two_tower.random_seed,
    )
    eval_cases = build_eval_cases(
        service.engagement_profiles, splits, service.product_lookup, service.product_embeddings, service.config.features, {}
    )

    results = [
        generate_recommendations(
            case.user_features, service.product_features, service.product_embeddings, service.all_item_ids,
            service.tt_encoder, service.user_tower, service.ranker_model, service.vector_index, service.config, top_n,
        )
        for case in eval_cases
    ]

    catalog_size = len(service.all_item_ids)
    val_report = evaluate_pipeline(eval_cases, results, "val_product_id", "val", k_values, catalog_size, service.product_features)
    test_report = evaluate_pipeline(eval_cases, results, "test_product_id", "test", k_values, catalog_size, service.product_features)
    return OfflineMetricsSummary(num_eval_users=len(eval_cases), val_report=val_report, test_report=test_report)
