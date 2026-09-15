"""Explicit backend activity-type -> canonical engagement-signal mapping.

The backend's `/api/user-activities` `actionType` vocabulary is NOT the
recommender's canonical `ActionType` vocabulary. Every backend value is
handled deliberately here - mapped to exactly one canonical signal, or
explicitly ignored - so an unrecognised or intentionally-dropped backend
action can never silently become a wrong recommender signal.

Observed backend vocabulary (probed live 2026-09-01, re-probed 2026-09-14
and 2026-09-15): `ViewProduct`, `AddToCart`, `RemoveFromCart`,
`AddedToFavorites`, `RemovedFromFavorites`, `PlaceOrder`, `SearchProduct`,
`Chatbot`.

Decisions:
- `ViewProduct`  -> CLICK        (product view = weakest-intent positive
                                     signal; mapping has been live-ready
                                     since 2026-09-01, but a full exhaustive
                                     scan of the live activity table on
                                     2026-09-15 still found ZERO `ViewProduct`
                                     rows - a live-data fact, not a mapping
                                     gap; see docs/data-mapping.md 19.10)
- `AddToCart`    -> ADD_TO_CART
- `PlaceOrder`   -> PURCHASE      (rows carry the resolved product slug/id;
                                     confirmed by the backend team to be one
                                     row per order line/product, never one
                                     row per order - see docs/data-mapping.md
                                     section 19.10)
- `SearchProduct` -> SEARCH       (added 2026-09-14 - the backend's search
                                     feature resolves each search to a
                                     specific product before recording the
                                     row, i.e. it already carries a product
                                     slug like every other signal here; a row
                                     with no slug is dropped by
                                     `loader.load_backend_events` exactly
                                     like any other actionType, never
                                     invented)
- `Chatbot`      -> CHATBOT       (added 2026-09-15 after a full live scan
                                     found 8 real rows, all product-resolved,
                                     one user/session mentioning 5+ related
                                     products - exactly the multi-product
                                     chatbot-turn shape `ChatbotContextRecord`
                                     was designed to aggregate. This
                                     contradicts an earlier task assumption
                                     that CHATBOT was still pending; mapped
                                     only after presenting the live evidence
                                     and getting explicit confirmation - see
                                     docs/data-mapping.md section 19.10)
- `AddedToFavorites`     -> IGNORE  (no canonical "favorite" signal; folding it
                                     into cart/click would misrepresent it -
                                     revisit if a dedicated signal is added)
- `RemoveFromCart`       -> IGNORE  (negative action; the recommender has no
                                     retraction semantics)
- `RemovedFromFavorites` -> IGNORE  (negative action)
- anything else          -> IGNORE + one WARNING log per distinct unknown value
"""

from __future__ import annotations

from recommendation.schemas.events import ActionType

# Sentinel for "recognised backend action, deliberately not a signal".
IGNORE = "IGNORE"

_ACTION_TYPE_MAP: dict[str, ActionType | str] = {
    "viewproduct": ActionType.CLICK,
    "addtocart": ActionType.ADD_TO_CART,
    "placeorder": ActionType.PURCHASE,
    "searchproduct": ActionType.SEARCH,
    "chatbot": ActionType.CHATBOT,
    "addedtofavorites": IGNORE,
    "removefromcart": IGNORE,
    "removedfromfavorites": IGNORE,
}


def map_action_type(backend_action_type: str) -> ActionType | None:
    """Returns the canonical `ActionType`, or `None` if this backend action
    is recognised-but-ignored OR unknown. Use `is_known` to tell those two
    apart for logging.
    """
    mapped = _ACTION_TYPE_MAP.get((backend_action_type or "").strip().lower())
    return mapped if isinstance(mapped, ActionType) else None


def is_known(backend_action_type: str) -> bool:
    return (backend_action_type or "").strip().lower() in _ACTION_TYPE_MAP
