"""Backend actionType -> canonical ActionType mapping policy."""

from recommendation.backend.mapping import is_known, map_action_type
from recommendation.schemas.events import ActionType


def test_positive_signals_map_to_canonical_types():
    assert map_action_type("ViewProduct") == ActionType.CLICK
    assert map_action_type("AddToCart") == ActionType.ADD_TO_CART
    assert map_action_type("PlaceOrder") == ActionType.PURCHASE
    assert map_action_type("SearchProduct") == ActionType.SEARCH
    assert map_action_type("Chatbot") == ActionType.CHATBOT


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


def test_no_favorites_or_removal_action_is_repurposed_into_a_real_signal():
    # Make sure the known-ignored actions stay ignored even as the map
    # grows - none of them silently became CLICK/SEARCH/CHATBOT etc.
    mapped = {map_action_type(a) for a in (
        "AddedToFavorites", "RemoveFromCart", "RemovedFromFavorites",
    )}
    assert mapped == {None}


def test_search_mapping_is_case_and_whitespace_insensitive():
    assert map_action_type(" searchproduct ") == ActionType.SEARCH
    assert map_action_type("SEARCHPRODUCT") == ActionType.SEARCH


def test_chatbot_mapping_is_case_and_whitespace_insensitive():
    assert map_action_type(" chatbot ") == ActionType.CHATBOT
    assert map_action_type("CHATBOT") == ActionType.CHATBOT
    assert is_known("Chatbot") is True
