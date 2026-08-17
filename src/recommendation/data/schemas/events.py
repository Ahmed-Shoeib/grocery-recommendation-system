"""Canonical shape of the future real-backend engagement source.

The backend team has confirmed a single append-style activity/audit-log
table, `User_events`:

    User_events
        id
        user_id
        product_id
        action_time
        action_type

Every row is one interaction at one point in time; multiple rows for the
same (user_id, product_id) pair are valid and expected (the same user
clicking, then adding to cart, then purchasing the same product each
produce their own row). `action_type` is one of the five engagement
signals this recommender consumes: CLICK, ADD_TO_CART, PURCHASE, SEARCH,
CHATBOT.

Per the confirmed contract, `product_id` is REQUIRED for every event this
recommender consumes - including SEARCH and CHATBOT. The backend only
writes a SEARCH row once a search has been resolved to a specific
product, and only writes a CHATBOT row once a chatbot conversation has
been resolved to a specific product. An unresolved search query or a
chatbot conversation that never names a product is therefore never
represented here - this recommender intentionally operates on resolved,
product-level interactions only, not raw free text. Category/brand/etc.
are derived downstream via `product_id -> Product`; age_group/
preferredCategory are derived via `user_id -> User`. No search query
text, chatbot summary/keywords/intent, or other free-text/metadata
column is part of this contract.

`UserInteraction` is the canonical, source-agnostic recommendation-side
representation of one `User_events` row (or, for today's synthetic V1,
one adapter-synthesized event). `adapters.user_events_adapter
.UserEventsAdapter` is the seam that will translate real `User_events`
rows into `UserInteraction`s and, from there, into the same per-signal
canonical records (`PurchaseRecord`, `CartAffinityRecord`, `ClickRecord`,
`SearchRecord`, `ChatbotContextRecord`) that `EngagementProfile` already
uses today - feature engineering, Two-Tower, and the ranker never see
`UserInteraction` directly and never need to know whether it came from
`User_events` or from a synthetic generator.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class ActionType(str, Enum):
    CLICK = "CLICK"
    ADD_TO_CART = "ADD_TO_CART"
    PURCHASE = "PURCHASE"
    SEARCH = "SEARCH"
    CHATBOT = "CHATBOT"


class UserInteraction(BaseModel):
    """One row of the future `User_events` table, canonicalized.

    `action_time` is preserved here so a future temporal split / recency
    feature / event-sequence model doesn't require another data-contract
    change - it is NOT currently consumed by feature engineering or any
    model (see docs/data-mapping.md section 7/12).
    """

    user_id: int
    product_id: int
    action_type: ActionType
    action_time: datetime | None = None
