"""API request/response schemas (Phase 8) - deliberately separate from
the internal domain/model schemas (`data.schemas`, `features
.product_features.ProductFeatures`, `serving.pipeline
.RecommendationResult`): this is the wire contract, versioned and
stable independent of internal refactors, and it never duplicates
canonical product catalog data (name, price, description, ...) - only
`product_id` plus ranking/order information, per docs/data-mapping.md's
V1 scope. A client resolves full product details from the catalog
service itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    product_id: int
    rank: int = Field(description="1-indexed position in the final recommendation list")
    score: float = Field(description="Final pipeline score (ranker probability, or a fallback-source score for cold-start tiers)")
    source: str = Field(description="'personalized' | 'preferred_category' | 'category_popularity' | 'global_popularity'")


class RecommendationMeta(BaseModel):
    user_id: int
    tier: Literal["strong", "sparse", "no_history"]
    requested_top_n: int
    returned_count: int
    fill_rate: float = Field(description="returned_count / requested_top_n - 1.0 means the request was fully filled")
    pool_size: int = Field(description="Candidate pool size retrieval/ranking/re-ranking operated on, capped by the currently-ELIGIBLE catalog size (post pre-retrieval eligibility)")
    num_excluded_pre_retrieval: int = Field(description="Catalog products excluded by the hard PRE-retrieval eligibility gate (isActive/stockQuantity) before candidate generation ran")
    num_excluded_by_eligibility: int = Field(description="Candidates excluded by the FINAL lightweight eligibility validation (defense-in-depth safety net); normally 0 since pre-retrieval filtering already excludes ineligible products")
    api_version: str
    model_version: str = Field(description="Ranker model version that produced these scores (see models/ranker/metadata.json)")
    generated_at: datetime
    latency_ms: float = Field(description="Server-side pipeline latency only (pre-retrieval eligibility -> Two-Tower -> VectorIndex -> Ranker -> Re-ranking -> final eligibility validation) - NOT full HTTP round-trip time")


class RecommendationResponse(BaseModel):
    meta: RecommendationMeta
    items: list[RecommendationItem]


class ErrorResponse(BaseModel):
    error: str = Field(description="Short machine-readable error code, e.g. 'unknown_user'")
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]
    model_version: str | None = None
