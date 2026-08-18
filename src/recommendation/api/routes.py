"""V1 recommendation API routes. Every handler is a thin translation
layer: validate/parse -> call `RecommendationService` (which itself only
calls `serving.pipeline.recommend`, no logic duplicated here) -> map the
result onto the versioned `api.schemas` wire contract. Eligibility
filtering (hard pre-retrieval gate + final lightweight validation)
happens entirely inside the pipeline - nothing here re-filters or
reorders results.

STEP 9 (docs/data-mapping.md section 18) additions:
`/users/{id}/recommendations`'s items are now enriched with display
fields (`ui.data_access.format_recommendation_table`, unchanged, just
called from here instead of from `dashboard.py` directly) so a client
never needs its own separate catalog access. `/users`, `/users/{id}
/profile` are small, generally-useful read-only endpoints - NOT
recommendation generation - added so the Streamlit dashboard (now a pure
HTTP client) can still browse users/engagement data without instantiating
a `RecommendationService` itself; they reuse `ui.data_access` unchanged,
no new business logic.

`/metrics/offline` (STEP 9 follow-up fix, see `evaluation.offline_report`
module docstring) is a cheap READ of a report PERSISTED by
`scripts/generate_offline_report.py` - it never runs an evaluation pass
itself, so unlike the endpoints above it does not touch `RecommendationService`'s
pipeline-facing methods at all, only `service.config`/`service
.ranker_model_version`/`service.ranker_metadata` (already-loaded values).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from recommendation.api.dependencies import RecommendationService, resolve_models_root
from recommendation.api.errors import UnknownUserError
from recommendation.api.schemas import (
    HealthResponse,
    OfflineMetricsProvenance,
    OfflineMetricsResponse,
    OfflineMetricsSplitReport,
    ReadinessResponse,
    RecommendationItem,
    RecommendationMeta,
    RecommendationResponse,
    UserListItem,
    UserListResponse,
    UserProfileResponse,
)
from recommendation.evaluation.offline_report import (
    OfflineEvaluationReport,
    OfflineReportError,
    load_offline_report,
    validate_report_provenance,
)
from recommendation.ui.data_access import (
    format_cart_items,
    format_chatbot_context,
    format_clicks,
    format_purchases,
    format_recommendation_table,
    format_searches,
    list_users,
    load_user_detail,
)
from recommendation.utils.logging import get_logger

logger = get_logger(__name__)

API_VERSION = "v1"

router = APIRouter(prefix="/v1")


def get_service(request: Request) -> RecommendationService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="recommendation service is not ready")
    return service


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: the process is up and answering requests. Does not
    check model readiness - see /v1/ready for that.
    """
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
async def ready(request: Request) -> ReadinessResponse:
    """Readiness: are the Two-Tower/ranker/VectorIndex/catalog actually
    loaded and usable. Returns 503 (via the response itself, checked by
    the caller/orchestrator) when not all checks pass.
    """
    service = getattr(request.app.state, "service", None)
    if service is None:
        payload = ReadinessResponse(status="not_ready", checks={})
        return JSONResponse(status_code=503, content=payload.model_dump())

    checks = service.readiness_checks()
    all_ready = all(checks.values())
    payload = ReadinessResponse(status="ready" if all_ready else "not_ready", checks=checks, model_version=service.ranker_model_version)
    if not all_ready:
        return JSONResponse(status_code=503, content=payload.model_dump())
    return payload


@router.get("/users/{user_id}/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int,
    limit: int | None = Query(default=None, ge=1, description="Requested Top-N; defaults to configs/base.yaml: api.default_recommendation_count"),
    service: RecommendationService = Depends(get_service),
) -> RecommendationResponse:
    effective_limit = limit if limit is not None else service.config.api.default_recommendation_count
    max_allowed = service.config.api.max_recommendation_count
    if effective_limit > max_allowed:
        raise HTTPException(status_code=422, detail=f"limit must be <= {max_allowed}, got {effective_limit}")

    start = time.perf_counter()
    try:
        result = service.recommend(user_id, effective_limit)
    except UnknownUserError as exc:
        logger.info("recommendations requested for unknown user_id=%s", user_id)
        raise HTTPException(status_code=404, detail=f"user {user_id} not found") from exc
    latency_ms = (time.perf_counter() - start) * 1000

    rows = format_recommendation_table(result, service.product_lookup)
    items = [
        RecommendationItem(
            product_id=row["product_id"],
            rank=row["rank"],
            score=row["score"],
            source=row["source"],
            product_name=row["name"],
            category=row["category"],
            brand=row["brand"],
            price=row["price"],
            is_active=row["is_active"],
            stock_quantity=row["stock_quantity"],
        )
        for row in rows
    ]
    meta = RecommendationMeta(
        user_id=user_id,
        tier=result.tier.value,
        requested_top_n=result.requested_n,
        returned_count=len(items),
        fill_rate=result.fill_rate,
        pool_size=result.pool_size,
        num_excluded_pre_retrieval=result.num_excluded_pre_retrieval,
        num_excluded_by_eligibility=result.num_excluded_by_eligibility,
        api_version=API_VERSION,
        model_version=service.ranker_model_version,
        generated_at=datetime.now(timezone.utc),
        latency_ms=latency_ms,
    )
    # tier/fill_rate/pool_size/eligibility are already logged once inside
    # serving.pipeline.generate_recommendations (shared with the
    # dashboard) - this line only adds the HTTP-request-specific fact
    # (observed latency at the API boundary) rather than repeating them.
    logger.info("request completed user_id=%s latency_ms=%.2f", user_id, latency_ms)
    return RecommendationResponse(meta=meta, items=items)


@router.get("/users", response_model=UserListResponse)
async def get_users(service: RecommendationService = Depends(get_service)) -> UserListResponse:
    """Read-only user browsing for the dashboard's user picker - reuses
    `ui.data_access.list_users` unchanged, no recommendation logic.
    """
    rows = list_users(service)
    return UserListResponse(
        users=[UserListItem(user_id=r.user_id, preferred_category=r.preferred_category, age_group=r.age_group) for r in rows]
    )


@router.get("/users/{user_id}/profile", response_model=UserProfileResponse)
async def get_user_profile(user_id: int, service: RecommendationService = Depends(get_service)) -> UserProfileResponse:
    """Read-only engagement/feature snapshot for the dashboard's user
    detail view - reuses `ui.data_access.load_user_detail` and its
    `format_*` helpers unchanged, no recommendation logic. Distinct from
    `UnknownUserError` (used by /recommendations): here a genuinely
    unknown user_id is reported as a plain 404, no pipeline is invoked.
    """
    detail = load_user_detail(service, user_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"user {user_id} not found")

    product_lookup = service.product_lookup
    features = detail.features
    return UserProfileResponse(
        user_id=detail.user_id,
        preferred_category=detail.preferred_category,
        age_group=detail.age_group,
        tier=detail.tier.value,
        total_engagement_events=features.total_engagement_events,
        distinct_products_purchased=len(set(p.product_id for p in detail.engagement.purchases)),
        click_count=len(detail.engagement.clicks),
        purchase_count=len(detail.engagement.purchases),
        cart_item_count=len(detail.engagement.cart_items),
        search_count=len(detail.engagement.searches),
        has_chatbot_context=detail.engagement.chatbot_context is not None,
        category_affinity=dict(features.category_affinity),
        brand_affinity=dict(features.brand_affinity),
        clicks=format_clicks(detail.engagement, product_lookup),
        purchases=format_purchases(detail.engagement, product_lookup),
        cart_items=format_cart_items(detail.engagement, product_lookup),
        searches=format_searches(detail.engagement, product_lookup),
        chatbot=format_chatbot_context(detail.engagement, product_lookup),
    )


@router.get("/metrics/offline", response_model=OfflineMetricsResponse)
async def get_offline_metrics(service: RecommendationService = Depends(get_service)) -> OfflineMetricsResponse:
    """Cheap READ of the PERSISTED offline evaluation report produced by
    `scripts/generate_offline_report.py` (the APPROVED STEP 5/7/8 temporal
    future-purchase protocol - see `evaluation.offline_report` module
    docstring). Never runs `generate_recommendations`/an evaluation pass
    itself - this handler does not call the pipeline at all.

    Fails clearly (409) rather than silently falling back to a live
    evaluation run if the report is missing, malformed, or was produced
    against a different model/run/dataset than the one this process
    currently has loaded (`OfflineReportError` and its subclasses).
    """
    report_path = resolve_models_root(service.config) / "offline_report.json"
    try:
        report: OfflineEvaluationReport = load_offline_report(report_path)
        validate_report_provenance(
            report,
            ranker_model_version=service.ranker_model_version,
            two_tower_model_version=service.two_tower_model_version,
            ranker_run_id=str(service.ranker_metadata.get("run_id", "")),
            ranker_dataset_fingerprint=str(service.ranker_metadata.get("dataset_fingerprint_sha256_16", "")),
        )
    except OfflineReportError as exc:
        logger.warning("offline metrics report unavailable: %s", exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    def _to_schema(split_report) -> OfflineMetricsSplitReport:
        return OfflineMetricsSplitReport(
            split_name=split_report.split_name,
            num_cases=split_report.num_cases,
            ndcg_at_k=split_report.ndcg_at_k,
            precision_at_k=split_report.precision_at_k,
            recall_at_k=split_report.recall_at_k,
            hit_rate_at_k=split_report.hit_rate_at_k,
            mrr=split_report.mrr,
            catalog_coverage=split_report.catalog_coverage,
            mean_distinct_categories=split_report.mean_distinct_categories,
            mean_fill_rate=split_report.mean_fill_rate,
        )

    provenance = OfflineMetricsProvenance(
        generated_at=report.generated_at,
        run_id=report.run_id,
        ranker_model_version=report.ranker_model_version,
        two_tower_model_version=report.two_tower_model_version,
        dataset_fingerprint_sha256_16=report.dataset_fingerprint_sha256_16,
        recency_enabled=report.recency_enabled,
        recency_half_life_days=report.recency_half_life_days,
        include_price_features=report.include_price_features,
        k_values=report.k_values,
        top_n=report.top_n,
    )
    return OfflineMetricsResponse(
        provenance=provenance,
        val_report=_to_schema(report.val_report),
        test_report=_to_schema(report.test_report),
    )
