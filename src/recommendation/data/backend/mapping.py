"""Explicit backend activity-type -> canonical engagement-signal mapping.

The backend's `/api/user-activities` `actionType` vocabulary is NOT the
recommender's canonical `ActionType` vocabulary. Every backend value is
handled deliberately here - mapped to exactly one canonical signal, or
explicitly ignored - so an unrecognised or intentionally-dropped backend
action can never silently become a wrong recommender signal.

Observed backend vocabulary (probed live 2026-09-01):
`ViewProduct`, `AddToCart`, `RemoveFromCart`, `AddedToFavorites`,
`RemovedFromFavorites`, `PlaceOrder`.

Decisions:
- `ViewProduct`  -> CLICK        (product view = weakest-intent positive signal)
- `AddToCart`    -> ADD_TO_CART
- `PlaceOrder`   -> PURCHASE      (rows carry the resolved product slug)
- `AddedToFavorites`     -> IGNORE  (no canonical "favorite" signal; folding it
                                     into cart/click would misrepresent it -
                                     revisit if a dedicated signal is added)
- `RemoveFromCart`       -> IGNORE  (negative action; the recommender has no
                                     retraction semantics)
- `RemovedFromFavorites` -> IGNORE  (negative action)
- anything else          -> IGNORE + one WARNING log per distinct unknown value

The backend has no SEARCH- or CHATBOT-equivalent activity, so those
canonical signals are simply never produced by this source (exactly as
`ChatbotContextRecord` / `SearchRecord` already tolerate).
"""

from __future__ import annotations

from recommendation.data.schemas.events import ActionType

# Sentinel for "recognised backend action, deliberately not a signal".
IGNORE = "IGNORE"

_ACTION_TYPE_MAP: dict[str, ActionType | str] = {
    "viewproduct": ActionType.CLICK,
    "addtocart": ActionType.ADD_TO_CART,
    "placeorder": ActionType.PURCHASE,
    "addedtofavorites": IGNORE,
    "removefromcart": IGNORE,
    "removedfromfavorites": IGNORE,
}

# Backend values that are known and intentionally dropped - distinguished
# from genuinely unknown values so the loader only warns about the latter.
KNOWN_IGNORED = frozenset(k for k, v in _ACTION_TYPE_MAP.items() if v == IGNORE)


def map_action_type(backend_action_type: str) -> ActionType | None:
    """Returns the canonical `ActionType`, or `None` if this backend action
    is recognised-but-ignored OR unknown. Use `is_known` to tell those two
    apart for logging.
    """
    mapped = _ACTION_TYPE_MAP.get((backend_action_type or "").strip().lower())
    return mapped if isinstance(mapped, ActionType) else None


def is_known(backend_action_type: str) -> bool:
    return (backend_action_type or "").strip().lower() in _ACTION_TYPE_MAP
