"""Canonical per-signal engagement records and the combined engagement layer.

These are the five V1 signals (clicks, purchases, cart, search, chatbot)
plus reviews and the user profile, in the shape feature engineering
consumes. Adapters produce these from either the real ERD tables, the
`User_events` activity-log table (`adapters.user_events_adapter
.UserEventsAdapter`), or a synthetic provider - model/feature code
downstream never sees the source distinction.

Raw timestamp fields (`order_created_at`, `review_created_at`,
`action_time`) are kept only where the source actually has them.
`features.recency` derives a recency/time-decay weight from them where
present (docs/data-mapping.md section 14). The synthetic V1 click/cart/
search generators don't produce real timestamps, so `action_time` is
`None` on those records and falls back to a neutral (unweighted) recency
contribution rather than being dropped; a `UserEventsAdapter`-sourced
record always populates it from `UserInteraction.action_time`
(`data.schemas.events`).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from recommendation.data.schemas.user import UserProfile


class PurchaseRecord(BaseModel):
    """One product line from a past order (Order + OrderItem + Product),
    or - from a future `User_events` PURCHASE row - one resolved purchase
    interaction. `order_id`/`unit_price`/`order_status` are ERD-specific
    (Order/OrderItem) fields a `User_events` row doesn't carry, so they're
    optional: populated for the ERD-backed synthetic/real-Order path,
    `None` for the `User_events`-sourced path (`quantity` defaults to 1 -
    one event row = one interaction, not an arbitrary order-line quantity).
    """

    user_id: int
    product_id: int
    order_id: int | None = None
    quantity: int = 1
    unit_price: float | None = None
    order_status: str | None = None
    order_created_at: datetime | None = None  # not used for recency in V1


class CartAffinityRecord(BaseModel):
    """One product currently (or recently) in a user's cart (Cart +
    CartItem), or - from a future `User_events` ADD_TO_CART row - one
    resolved add-to-cart interaction (`quantity` defaults to 1 in that
    case, matching `PurchaseRecord`).
    """

    user_id: int
    product_id: int
    quantity: int = 1
    action_time: datetime | None = None  # populated only for User_events-sourced records; see module docstring


class ClickRecord(BaseModel):
    """A CLICK engagement event: the user viewed/clicked a specific
    product. First-class fifth V1 signal (docs/data-mapping.md section 4);
    no backend table exists yet distinct from the future `User_events`
    log, so V1 uses a synthetic generator the same way SEARCH/CHATBOT do.

    `source` distinguishes provider origin ("synthetic" vs a future
    `User_events`-backed value) so a real `ClickAdapter` is a drop-in
    replacement - nothing downstream branches on it.
    """

    user_id: int
    product_id: int
    source: str = "synthetic"
    action_time: datetime | None = None


class SearchRecord(BaseModel):
    """A search event.

    `search_term` is optional: the current synthetic generator always
    populates it (there's no `User_events`-style raw-query storage to
    draw from otherwise), but the confirmed `User_events` contract never
    provides raw search text at all - only resolved-to-a-product SEARCH
    rows are stored, so a future `UserEventsAdapter`-sourced `SearchRecord`
    always has `matched_product_id` set and `search_term=None`. Feature
    engineering only reads `search_term` when `matched_product_id is
    None`, so this is safe: a `User_events`-sourced search always takes
    the `matched_product_id` path.

    `source` distinguishes provider origin so a future SearchAdapter backed
    by a real API/table/event pipeline is a drop-in replacement.
    """

    user_id: int
    search_term: str | None = None
    matched_product_id: int | None = None
    source: str = "synthetic"
    action_time: datetime | None = None  # populated only for User_events-sourced records; see module docstring


class ChatbotContextRecord(BaseModel):
    """Aggregated chatbot interaction summary for one user (one record per
    user, not one per conversation - see below).

    Synthetic V1 sources this from a template-based generator with a free-
    text `summary`/`keywords`. Per the confirmed `User_events` contract,
    the real backend will NEVER provide that free text to this
    recommender - it only writes a CHATBOT row once a conversation
    resolves to a specific `product_id`, with no query/summary/keyword
    text at all (docs/data-mapping.md section 4). A future
    `UserEventsAdapter` therefore populates only `mentioned_product_ids`
    (one entry per resolved CHATBOT row for that user) and leaves
    `summary`/`keywords`/`product_interest`/`preferred_category` at their
    defaults - this record's optional/loose fields are exactly what makes
    that a non-breaking, drop-in mapping instead of a schema change.

    Known V1 simplification: this stays a single aggregated record per
    user (matching the existing synthetic shape), not a list of one
    record per resolved mention, so an individual mention's `action_time`
    isn't preserved here even for `User_events`-sourced data - only the
    MOST RECENT resolved-mention timestamp for this user is kept (`action_time`
    below), used exclusively for recency-weighting the chatbot signal's
    whole contribution as one group (docs/data-mapping.md section 14); it
    is not per-mention recency. If per-mention chatbot recency is ever
    needed, this record would need to become a list of single-mention
    records at that time.
    """

    user_id: int
    mentioned_product_ids: list[int] = Field(default_factory=list)
    preferred_category: str | None = None
    product_interest: str | None = None
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    source: str = "synthetic"
    # Most recent CHATBOT resolution `action_time` for this user; `None`
    # for the synthetic generator (no timestamp to draw from) - see class
    # docstring and `features.recency.effective_weight`'s missing-
    # timestamp fallback.
    action_time: datetime | None = None


class ReviewRecord(BaseModel):
    user_id: int
    product_id: int
    rating: float
    comment: str | None = None
    review_created_at: datetime | None = None  # not used for recency in V1


class EngagementProfile(BaseModel):
    """Canonical combined engagement layer for one user.

    This, not any individual adapter output, is what feature engineering
    (Phase 3) and the cold-start tiering logic (Phase 7) consume. Combines
    all five V1 engagement signals (clicks, purchases, cart adds,
    searches, chatbot context) plus reviews and the user profile -
    identical shape whether populated by the synthetic adapters or, later,
    by `adapters.user_events_adapter.UserEventsAdapter` reading the real
    `User_events` table.
    """

    user_id: int
    profile: UserProfile
    clicks: list[ClickRecord] = Field(default_factory=list)
    purchases: list[PurchaseRecord] = Field(default_factory=list)
    cart_items: list[CartAffinityRecord] = Field(default_factory=list)
    searches: list[SearchRecord] = Field(default_factory=list)
    chatbot_context: ChatbotContextRecord | None = None
    reviews: list[ReviewRecord] = Field(default_factory=list)
