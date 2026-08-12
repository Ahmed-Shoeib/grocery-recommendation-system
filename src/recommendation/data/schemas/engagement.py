"""Canonical per-signal engagement records and the combined engagement layer.

These are the four V1 signals (purchases, cart, search, chatbot) plus
reviews and the user profile, in the shape feature engineering consumes.
Adapters (Phase 2) are responsible for producing these from either the
real ERD tables or a synthetic provider - model/feature code downstream
never sees the source distinction.

Raw timestamp fields (`order_created_at`, `review_created_at`) are kept
where the ERD actually has them (Order.CreationDate, Review.CreationDate)
so the canonical schema doesn't lie about what the backend provides, but
per the V1 scope decision (docs/data-mapping.md, section 1) feature
engineering must not derive recency/time-decay signals from them. They
exist here purely so a future V2 recency feature doesn't require a schema
change.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from recommendation.data.schemas.user import UserProfile


class PurchaseRecord(BaseModel):
    """One product line from a past order (Order + OrderItem + Product)."""

    user_id: int
    product_id: int
    order_id: int
    quantity: int
    unit_price: float
    order_status: str | None = None
    order_created_at: datetime | None = None  # not used for recency in V1


class CartAffinityRecord(BaseModel):
    """One product currently (or recently) in a user's cart (Cart + CartItem)."""

    user_id: int
    product_id: int
    quantity: int


class SearchRecord(BaseModel):
    """A search event. No backend table exists yet - always synthetic in V1.

    `source` distinguishes provider origin so a future SearchAdapter backed
    by a real API/table/event pipeline is a drop-in replacement.
    """

    user_id: int
    search_term: str
    matched_product_id: int | None = None
    source: str = "synthetic"


class ChatbotContextRecord(BaseModel):
    """Structured chatbot interaction summary for one user.

    No backend table exists yet - always synthetic in V1. Deliberately
    loose/optional fields since the real backend's eventual representation
    (API response, DB entity, event summary) is unknown; whichever shape it
    provides, a ChatbotContextAdapter maps it into this record.
    """

    user_id: int
    mentioned_product_ids: list[int] = Field(default_factory=list)
    preferred_category: str | None = None
    product_interest: str | None = None
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    source: str = "synthetic"


class ReviewRecord(BaseModel):
    user_id: int
    product_id: int
    rating: float
    comment: str | None = None
    review_created_at: datetime | None = None  # not used for recency in V1


class EngagementProfile(BaseModel):
    """Canonical combined engagement layer for one user.

    This, not any individual adapter output, is what feature engineering
    (Phase 3) and the cold-start tiering logic (Phase 7) consume.
    """

    user_id: int
    profile: UserProfile
    purchases: list[PurchaseRecord] = Field(default_factory=list)
    cart_items: list[CartAffinityRecord] = Field(default_factory=list)
    searches: list[SearchRecord] = Field(default_factory=list)
    chatbot_context: ChatbotContextRecord | None = None
    reviews: list[ReviewRecord] = Field(default_factory=list)
