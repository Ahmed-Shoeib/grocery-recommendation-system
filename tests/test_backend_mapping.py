"""Backend actionType -> canonical ActionType mapping policy."""

from recommendation.data.backend.mapping import is_known, map_action_type
from recommendation.data.schemas.events import ActionType


def test_positive_signals_map_to_canonical_types():
    assert map_action_type("ViewProduct") == ActionType.CLICK
    assert map_action_type("AddToCart") == ActionType.ADD_TO_CART
    assert map_action_type("PlaceOrder") == ActionType.PURCHASE


def test_mapping_is_case_and_whitespace_insensitive():
    assert map_action_type("  addtocart ") == ActionType.ADD_TO_CART
    assert map_action_type("VIEWPRODUCT") == ActionType.CLICK


def test_known_ignored_actions_map_to_none_but_are_known():
    for action in ("AddedToFavorites", "RemoveFromCart", "RemovedFromFavorites"):
        assert map_action_type(action) is None
        assert is_known(action) is True


def test_unknown_action_maps_to_none_and_is_not_known():
    assert map_action_type("TeleportProduct") is None
    assert is_known("TeleportProduct") is False


def test_no_backend_action_maps_to_search_or_chatbot():
    # The backend has no search/chatbot-equivalent activity; make sure a
    # future rename can't silently start feeding those slots.
    mapped = {map_action_type(a) for a in (
        "ViewProduct", "AddToCart", "PlaceOrder", "AddedToFavorites",
        "RemoveFromCart", "RemovedFromFavorites",
    )}
    assert ActionType.SEARCH not in mapped
    assert ActionType.CHATBOT not in mapped
