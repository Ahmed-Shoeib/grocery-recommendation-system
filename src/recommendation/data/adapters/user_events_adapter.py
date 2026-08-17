"""The future real-backend adapter: `User_events` -> canonical representation.

This is the seam the whole architecture change exists to prepare. Once
the backend team's `User_events` table (id, user_id, product_id,
action_time, action_type) is real and queryable, whatever loads rows from
it maps each row into a `UserInteraction`
(`recommendation.data.schemas.events`) - the only thing this module needs
to be handed. `UserEventsAdapter` then does the `action_type` fan-out and
implements FIVE existing adapter interfaces (`ClickAdapter`,
`PurchaseAdapter`, `CartAdapter`, `SearchAdapter`, `ChatbotContextAdapter`)
from that one list, so it drops into `AdapterBundle` exactly where the
five separate synthetic adapters (`SyntheticClickAdapter`,
`InMemoryPurchaseAdapter`, `InMemoryCartAdapter`, `SyntheticSearchAdapter`,
`SyntheticChatbotAdapter`) sit today - feature engineering, the Two-Tower
model, the ranker, and the serving pipeline need no changes at all.

What this adapter does NOT cover: `ProductCatalogAdapter`, `UserAdapter`,
and `ReviewAdapter` stay backed by their own real ERD tables (Product,
User, Review) - those are unrelated to the `User_events` activity log and
untouched by this contract. `build_user_events_adapters` below accepts
those three as parameters so a real deployment can assemble a full
`AdapterBundle` from (real ERD adapters) + (this one `User_events`-backed
adapter), without needing five separate real adapter implementations.

Field-availability note: `User_events` never carries `order_id`,
`unit_price`, `quantity`, or free text (search query / chatbot summary/
keywords) - only `user_id`, `product_id`, `action_time`, `action_type`.
`PurchaseRecord`/`CartAffinityRecord`/`SearchRecord` already model those
extra ERD-specific fields as optional precisely so this adapter can
construct them with only what `User_events` actually provides (see
`data.schemas.engagement` module docstring). `ChatbotContextRecord`
aggregates every resolved CHATBOT row for a user into one record's
`mentioned_product_ids` list - see that class's docstring for why this
stays a single aggregate record rather than one row per mention.
"""

from __future__ import annotations

from collections import defaultdict

from recommendation.data.adapters.base import (
    CartAdapter,
    ChatbotContextAdapter,
    ClickAdapter,
    PurchaseAdapter,
    ReviewAdapter,
    SearchAdapter,
    UserAdapter,
)
from recommendation.data.adapters.base import AdapterBundle, ProductCatalogAdapter
from recommendation.data.schemas.engagement import (
    CartAffinityRecord,
    ChatbotContextRecord,
    ClickRecord,
    PurchaseRecord,
    SearchRecord,
)
from recommendation.data.schemas.events import ActionType, UserInteraction

_SOURCE = "user_events"


class UserEventsAdapter(ClickAdapter, PurchaseAdapter, CartAdapter, SearchAdapter, ChatbotContextAdapter):
    """Backed by ONE list of `UserInteraction` rows (the canonicalized
    `User_events` table), fanned out by `action_type` into five adapter
    roles. Construct with whatever a real query against `User_events`
    returns, mapped into `UserInteraction`s - this class does no SQL/DB
    access itself, matching every other adapter in this codebase (data
    access is the caller's job; adapters only reshape already-fetched rows
    into canonical schemas).
    """

    def __init__(self, events: list[UserInteraction]) -> None:
        self._by_user_and_type: dict[tuple[int, ActionType], list[UserInteraction]] = defaultdict(list)
        for event in events:
            self._by_user_and_type[(event.user_id, event.action_type)].append(event)

    def _for_user(self, user_id: int, action_type: ActionType) -> list[UserInteraction]:
        return self._by_user_and_type.get((user_id, action_type), [])

    def _for_all_users(self, action_type: ActionType) -> list[UserInteraction]:
        return [
            event
            for (_, at), events in self._by_user_and_type.items()
            if at == action_type
            for event in events
        ]

    # --- ClickAdapter ----------------------------------------------------

    def get_clicks(self, user_id: int) -> list[ClickRecord]:
        return [
            ClickRecord(user_id=e.user_id, product_id=e.product_id, source=_SOURCE, action_time=e.action_time)
            for e in self._for_user(user_id, ActionType.CLICK)
        ]

    def list_all_clicks(self) -> list[ClickRecord]:
        return [
            ClickRecord(user_id=e.user_id, product_id=e.product_id, source=_SOURCE, action_time=e.action_time)
            for e in self._for_all_users(ActionType.CLICK)
        ]

    # --- PurchaseAdapter ---------------------------------------------------
    # No order_id/unit_price/order_status in User_events - each PURCHASE
    # row becomes one PurchaseRecord with quantity=1 (the schema default),
    # order_created_at populated from action_time.

    def get_purchases(self, user_id: int) -> list[PurchaseRecord]:
        return [
            PurchaseRecord(user_id=e.user_id, product_id=e.product_id, order_created_at=e.action_time)
            for e in self._for_user(user_id, ActionType.PURCHASE)
        ]

    def list_all_purchases(self) -> list[PurchaseRecord]:
        return [
            PurchaseRecord(user_id=e.user_id, product_id=e.product_id, order_created_at=e.action_time)
            for e in self._for_all_users(ActionType.PURCHASE)
        ]

    # --- CartAdapter -------------------------------------------------------
    # No CartItem.quantity in User_events - each ADD_TO_CART row becomes
    # one CartAffinityRecord with quantity=1 (the schema default).

    def get_cart_items(self, user_id: int) -> list[CartAffinityRecord]:
        return [
            CartAffinityRecord(user_id=e.user_id, product_id=e.product_id, action_time=e.action_time)
            for e in self._for_user(user_id, ActionType.ADD_TO_CART)
        ]

    def list_all_cart_items(self) -> list[CartAffinityRecord]:
        return [
            CartAffinityRecord(user_id=e.user_id, product_id=e.product_id, action_time=e.action_time)
            for e in self._for_all_users(ActionType.ADD_TO_CART)
        ]

    # --- SearchAdapter -------------------------------------------------------
    # Per the confirmed contract, User_events only ever stores a SEARCH row
    # once it's resolved to a product - so matched_product_id is always
    # set and search_term is always None (no raw query text exists here).

    def get_search_history(self, user_id: int) -> list[SearchRecord]:
        return [
            SearchRecord(user_id=e.user_id, matched_product_id=e.product_id, source=_SOURCE, action_time=e.action_time)
            for e in self._for_user(user_id, ActionType.SEARCH)
        ]

    # --- ChatbotContextAdapter -------------------------------------------
    # Same resolved-product-only contract as SEARCH; every resolved
    # CHATBOT row for this user becomes one entry in mentioned_product_ids.
    # No summary/keywords/preferred_category/product_interest text exists
    # in User_events, so those stay at their schema defaults (None/[]).

    def get_chatbot_context(self, user_id: int) -> ChatbotContextRecord | None:
        events = self._for_user(user_id, ActionType.CHATBOT)
        if not events:
            return None
        return ChatbotContextRecord(
            user_id=user_id, mentioned_product_ids=[e.product_id for e in events], source=_SOURCE
        )


def build_user_events_adapters(
    events: list[UserInteraction],
    products: ProductCatalogAdapter,
    users: UserAdapter,
    reviews: ReviewAdapter,
) -> AdapterBundle:
    """Assembles a full `AdapterBundle` from the `User_events` log plus the
    real ERD-backed `products`/`users`/`reviews` adapters (unaffected by
    this contract - see module docstring). This is the real-backend
    counterpart of `adapters.factory.build_synthetic_adapters`: same
    return type, same downstream consumers, only the construction differs.
    """
    user_events_adapter = UserEventsAdapter(events)
    return AdapterBundle(
        products=products,
        users=users,
        purchases=user_events_adapter,
        cart=user_events_adapter,
        clicks=user_events_adapter,
        reviews=reviews,
        search=user_events_adapter,
        chatbot=user_events_adapter,
    )
