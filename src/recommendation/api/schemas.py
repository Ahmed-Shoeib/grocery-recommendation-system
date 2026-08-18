"""API request/response schemas (Phase 8, extended STEP 9) - deliberately
separate from the internal domain/model schemas (`data.schemas`, `features
.product_features.ProductFeatures`, `serving.pipeline
.RecommendationResult`): this is the wire contract, versioned and stable
independent of internal refactors. `RecommendationItem` carries a small,
deliberately-chosen set of APPLICATION-level display fields (product
name/category/brand/price) - not the full catalog record, and never any
internal model tensor/feature (embeddings, ranking feature vectors,
recency weights, price-tier ids, ...) - see docs/data-mapping.md section
18's "preserve the API contract" note.

The STEP 9 user-listing/profile/offline-metrics schemas below exist
because the Streamlit dashboard (docs/data-mapping.md section 18) is now
a pure HTTP client with no other way to reach this data - they reuse
`ui.data_access`'s existing formatting functions server-side rather than
reimplementing anything.
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
    # STEP 9 (docs/data-mapping.md section 18): application-level display
    # fields, joined from the catalog server-side (never exposed as raw
    # internal model tensors/features) - added so a client (the Streamlit
    # dashboard, or any future frontend) can render a result WITHOUT its
    # own separate catalog access. Backward-compatible addition: existing
    # clients that only read product_id/rank/score/source are unaffected.
    product_name: str
    category: str | None = None
    brand: str | None = None
    price: float | None = None
    is_active: bool | None = None
    stock_quantity: int | None = None


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


# --- STEP 9: small, generally-useful read-only endpoints (docs/data-mapping.md section 18) ---
# Not recommendation generation (that stays exclusively the endpoint
# above) - user/catalog browsing and offline-diagnostic data any client
# might reasonably want, moved behind the API so Streamlit never needs
# its own adapter/model access. Deliberately loosely-typed (`dict`/
# `list[dict]`) for the nested engagement/metric sub-sections rather than
# a fully modeled schema for every internal record shape - these are
# debug/explainability views, not a production data contract clients
# build critical logic against.


class UserListItem(BaseModel):
    user_id: int
    preferred_category: str | None = None
    age_group: str | None = None


class UserListResponse(BaseModel):
    users: list[UserListItem]


class UserProfileResponse(BaseModel):
    user_id: int
    preferred_category: str | None = None
    age_group: str | None = None
    tier: Literal["strong", "sparse", "no_history"]
    total_engagement_events: int
    distinct_products_purchased: int
    click_count: int
    purchase_count: int
    cart_item_count: int
    search_count: int
    has_chatbot_context: bool
    category_affinity: dict[str, float]
    brand_affinity: dict[str, float]
    clicks: list[dict]
    purchases: list[dict]
    cart_items: list[dict]
    searches: list[dict]
    chatbot: dict | None = None


class OfflineMetricsSplitReport(BaseModel):
    split_name: str
    num_cases: int
    ndcg_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    mrr: float
    catalog_coverage: float
    mean_distinct_categories: float
    mean_fill_rate: float


class OfflineMetricsProvenance(BaseModel):
    """Enough to prove which model/run/dataset/config these numbers
    describe, and to let a client detect a stale report at a glance -
    see `evaluation.offline_report.OfflineEvaluationReport`.
    """

    generated_at: datetime
    run_id: str
    ranker_model_version: str
    two_tower_model_version: str
    dataset_fingerprint_sha256_16: str
    recency_enabled: bool
    recency_half_life_days: float | None = None
    include_price_features: bool
    k_values: list[int]
    top_n: int


class OfflineMetricsResponse(BaseModel):
    """Offline, SQLite-dataset evaluation metrics ONLY (the APPROVED STEP
    5/7/8 temporal future-purchase protocol - see `evaluation
    .offline_report` module docstring) - never a production/online
    metric. This is a cheap READ of a report persisted by
    `scripts/generate_offline_report.py`; the endpoint itself never runs
    an evaluation pass.
    """

    provenance: OfflineMetricsProvenance
    val_report: OfflineMetricsSplitReport
    test_report: OfflineMetricsSplitReport
