"""V1 recommendation API routes. Every handler is a thin translation
layer: validate/parse -> call `RecommendationService` (which itself only
calls `serving.pipeline.recommend`, no logic duplicated here) -> map the
result onto the versioned `api.schemas` wire contract. Business-rule
filtering happens inside the pipeline, at the end, unchanged - nothing
here re-filters or reorders results.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from recommendation.api.dependencies import RecommendationService
from recommendation.api.errors import UnknownUserError
from recommendation.api.schemas import (
    HealthResponse,
    ReadinessResponse,
    RecommendationItem,
    RecommendationMeta,
    RecommendationResponse,
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

    items = [
        RecommendationItem(product_id=pid, rank=rank, score=score, source=source)
        for rank, (pid, score, source) in enumerate(zip(result.product_ids, result.scores, result.sources), start=1)
    ]
    meta = RecommendationMeta(
        user_id=user_id,
        tier=result.tier.value,
        requested_top_n=result.requested_n,
        returned_count=len(items),
        fill_rate=result.fill_rate,
        pool_size=result.pool_size,
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
