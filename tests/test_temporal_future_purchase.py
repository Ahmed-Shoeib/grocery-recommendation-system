"""Tests for the temporal future-purchase evaluation protocol
(`recommendation.evaluation.temporal_future_purchase`).

Covers: chronological ordering, split-tier classification, point-in-time
truncation for all five signals (no future leakage, legitimate earlier
interactions preserved), same-timestamp policy, repeat purchases,
multiple relevant targets, insufficient-history handling, eligibility
partitioning, the leakage audit, and determinism.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from recommendation.data.adapters.base import UserAdapter
from recommendation.data.adapters.review_adapter import InMemoryReviewAdapter
from recommendation.data.schemas.events import ActionType, UserInteraction
from recommendation.data.schemas.product import Product
from recommendation.data.schemas.user import UserProfile
from recommendation.evaluation.temporal_future_purchase import (
    TemporalEligibilityTier,
    audit_no_leakage,
    build_point_in_time_engagement_profile,
    build_temporal_splits,
    events_before_cutoff,
    group_events_by_user,
    purchase_targets_in_window,
    split_targets_by_eligibility,
)
from recommendation.features.product_features import ProductFeatures
from recommendation.serving.eligibility import build_eligibility_rules
from recommendation.utils.config import EligibilityConfig

T0 = datetime(2026, 1, 1, 0, 0, 0)


def t(days: int, hours: int = 0) -> datetime:
    return T0 + timedelta(days=days, hours=hours)


def ev(user_id: int, product_id: int, action_type: ActionType, when: datetime) -> UserInteraction:
    return UserInteraction(user_id=user_id, product_id=product_id, action_type=action_type, action_time=when)


class _FakeUsers(UserAdapter):
    def __init__(self, profiles: dict[int, UserProfile]) -> None:
        self._profiles = profiles

    def get_user_profile(self, user_id: int):
        return self._profiles.get(user_id)

    def list_user_ids(self) -> list[int]:
        return list(self._profiles.keys())


# --- group_events_by_user ----------------------------------------------

def test_group_events_by_user_sorts_chronologically():
    events = [
        ev(1, 10, ActionType.CLICK, t(5)),
        ev(1, 11, ActionType.CLICK, t(1)),
        ev(1, 12, ActionType.CLICK, t(3)),
        ev(2, 20, ActionType.CLICK, t(2)),
    ]
    grouped = group_events_by_user(events)
    assert [e.product_id for e in grouped[1]] == [11, 12, 10]
    assert [e.product_id for e in grouped[2]] == [20]


# --- events_before_cutoff / same-timestamp policy ----------------------

def test_events_before_cutoff_is_strict():
    events = [ev(1, 10, ActionType.CLICK, t(1)), ev(1, 11, ActionType.CLICK, t(2)), ev(1, 12, ActionType.CLICK, t(3))]
    history = events_before_cutoff(events, cutoff=t(2))
    assert [e.product_id for e in history] == [10]  # t(2) itself excluded, only t(1) < t(2)


def test_events_before_cutoff_excludes_null_timestamps():
    events = [UserInteraction(user_id=1, product_id=10, action_type=ActionType.CLICK, action_time=None)]
    assert events_before_cutoff(events, cutoff=t(5)) == []


# --- purchase_targets_in_window -----------------------------------------

def test_purchase_targets_in_window_inclusive_lower_exclusive_upper():
    events = [
        ev(1, 10, ActionType.PURCHASE, t(2)),  # == lower -> included
        ev(1, 11, ActionType.PURCHASE, t(3)),  # == upper -> excluded
        ev(1, 12, ActionType.PURCHASE, t(2, hours=12)),  # inside window -> included
    ]
    window = purchase_targets_in_window(events, lower_cutoff=t(2), upper_cutoff=t(3))
    assert window == frozenset({10, 12})


def test_purchase_targets_in_window_no_upper_bound():
    events = [ev(1, 10, ActionType.PURCHASE, t(5)), ev(1, 11, ActionType.PURCHASE, t(10))]
    window = purchase_targets_in_window(events, lower_cutoff=t(5), upper_cutoff=None)
    assert window == frozenset({10, 11})


def test_purchase_targets_in_window_ignores_non_purchase_events():
    events = [ev(1, 10, ActionType.CLICK, t(5)), ev(1, 11, ActionType.PURCHASE, t(5))]
    assert purchase_targets_in_window(events, t(5), None) == frozenset({11})


def test_purchase_targets_in_window_multiple_products_at_tied_timestamp():
    """The exact mechanism that produces a genuine multi-item relevant
    set: two different products purchased at the identical timestamp.
    """
    events = [ev(1, 10, ActionType.PURCHASE, t(5)), ev(1, 11, ActionType.PURCHASE, t(5))]
    assert purchase_targets_in_window(events, t(5), None) == frozenset({10, 11})


# --- build_temporal_splits: tier classification -------------------------

def test_no_history_user():
    splits = build_temporal_splits({}, user_ids=[1])
    s = splits[1]
    assert s.tier == TemporalEligibilityTier.NO_HISTORY
    assert not s.is_val_evaluable and not s.is_test_evaluable


def test_engagement_no_purchase_user():
    events = {1: [ev(1, 10, ActionType.CLICK, t(1)), ev(1, 11, ActionType.SEARCH, t(2))]}
    s = build_temporal_splits(events, user_ids=[1])[1]
    assert s.tier == TemporalEligibilityTier.ENGAGEMENT_NO_PURCHASE
    assert s.total_purchase_event_count == 0


def test_insufficient_depth_single_purchase():
    events = {1: [ev(1, 10, ActionType.PURCHASE, t(1))]}
    s = build_temporal_splits(events, user_ids=[1])[1]
    assert s.tier == TemporalEligibilityTier.INSUFFICIENT_DEPTH
    assert s.train_purchase_event_count == 1
    assert not s.is_val_evaluable and not s.is_test_evaluable


def test_val_only_two_purchases():
    events = {1: [ev(1, 10, ActionType.PURCHASE, t(1)), ev(1, 20, ActionType.PURCHASE, t(5))]}
    s = build_temporal_splits(events, user_ids=[1])[1]
    assert s.tier == TemporalEligibilityTier.VAL_ONLY
    assert s.val_cutoff == t(5)
    assert s.val_target_ids == frozenset({20})
    assert s.test_cutoff is None
    assert s.train_purchase_event_count == 1


def test_full_split_three_purchases():
    events = {1: [ev(1, 10, ActionType.PURCHASE, t(1)), ev(1, 20, ActionType.PURCHASE, t(5)), ev(1, 30, ActionType.PURCHASE, t(9))]}
    s = build_temporal_splits(events, user_ids=[1])[1]
    assert s.tier == TemporalEligibilityTier.FULL
    assert s.val_cutoff == t(5)
    assert s.test_cutoff == t(9)
    assert s.val_target_ids == frozenset({20})
    assert s.test_target_ids == frozenset({30})
    assert s.train_purchase_event_count == 1  # only product 10 (t(1)) is strictly before val_cutoff


def test_full_split_val_and_test_targets_are_disjoint():
    events = {1: [ev(1, i, ActionType.PURCHASE, t(i)) for i in range(1, 6)]}  # 5 purchases, distinct times/products
    s = build_temporal_splits(events, user_ids=[1])[1]
    assert s.val_target_ids.isdisjoint(s.test_target_ids)
    assert s.train_purchase_event_count == 3  # products 1,2,3 before val_cutoff=t(4)


def test_multiple_relevant_targets_from_tied_timestamps():
    """n=4 with a tie at the second-to-last timestamp produces a genuine
    multi-product val target set.
    """
    events = {
        1: [
            ev(1, 10, ActionType.PURCHASE, t(1)),
            ev(1, 20, ActionType.PURCHASE, t(5)),
            ev(1, 21, ActionType.PURCHASE, t(5)),  # tied with product 20
            ev(1, 30, ActionType.PURCHASE, t(9)),
        ]
    }
    s = build_temporal_splits(events, user_ids=[1])[1]
    assert s.tier == TemporalEligibilityTier.FULL
    assert s.val_cutoff == t(5)
    assert s.val_target_ids == frozenset({20, 21})
    assert s.test_target_ids == frozenset({30})


def test_split_generation_is_deterministic():
    events = {
        1: [
            ev(1, 100, ActionType.PURCHASE, t(3)),
            ev(1, 101, ActionType.PURCHASE, t(1)),
            ev(1, 102, ActionType.PURCHASE, t(4)),
        ]
    }
    a = build_temporal_splits(events, user_ids=[1])[1]
    b = build_temporal_splits(events, user_ids=[1])[1]
    assert a == b


# --- point-in-time engagement profile: no future leakage, all 5 signals --

@pytest.fixture
def users_adapter() -> _FakeUsers:
    return _FakeUsers({1: UserProfile(user_id=1, preferred_category="Snacks", age_group="25-34")})


@pytest.fixture
def reviews_adapter() -> InMemoryReviewAdapter:
    return InMemoryReviewAdapter([])


def test_future_click_does_not_leak(users_adapter, reviews_adapter):
    events = [ev(1, 10, ActionType.CLICK, t(1)), ev(1, 20, ActionType.CLICK, t(10))]
    profile = build_point_in_time_engagement_profile(1, events, cutoff=t(5), users_adapter=users_adapter, reviews_adapter=reviews_adapter)
    assert [c.product_id for c in profile.clicks] == [10]


def test_future_search_does_not_leak(users_adapter, reviews_adapter):
    events = [ev(1, 10, ActionType.SEARCH, t(1)), ev(1, 20, ActionType.SEARCH, t(10))]
    profile = build_point_in_time_engagement_profile(1, events, cutoff=t(5), users_adapter=users_adapter, reviews_adapter=reviews_adapter)
    assert [s.matched_product_id for s in profile.searches] == [10]


def test_future_chatbot_does_not_leak(users_adapter, reviews_adapter):
    events = [ev(1, 10, ActionType.CHATBOT, t(1)), ev(1, 20, ActionType.CHATBOT, t(10))]
    profile = build_point_in_time_engagement_profile(1, events, cutoff=t(5), users_adapter=users_adapter, reviews_adapter=reviews_adapter)
    assert profile.chatbot_context.mentioned_product_ids == [10]


def test_future_add_to_cart_does_not_leak(users_adapter, reviews_adapter):
    events = [ev(1, 10, ActionType.ADD_TO_CART, t(1)), ev(1, 20, ActionType.ADD_TO_CART, t(10))]
    profile = build_point_in_time_engagement_profile(1, events, cutoff=t(5), users_adapter=users_adapter, reviews_adapter=reviews_adapter)
    assert [c.product_id for c in profile.cart_items] == [10]


def test_held_out_purchase_does_not_leak(users_adapter, reviews_adapter):
    events = [ev(1, 10, ActionType.PURCHASE, t(1)), ev(1, 50, ActionType.PURCHASE, t(10))]
    profile = build_point_in_time_engagement_profile(1, events, cutoff=t(10), users_adapter=users_adapter, reviews_adapter=reviews_adapter)
    assert [p.product_id for p in profile.purchases] == [10]  # the t(10) purchase (the target) is excluded


def test_event_exactly_at_cutoff_does_not_leak(users_adapter, reviews_adapter):
    events = [ev(1, 10, ActionType.CLICK, t(5))]
    profile = build_point_in_time_engagement_profile(1, events, cutoff=t(5), users_adapter=users_adapter, reviews_adapter=reviews_adapter)
    assert profile.clicks == []


def test_legitimate_earlier_interaction_with_future_target_product_is_preserved(users_adapter, reviews_adapter):
    """Section 7's core scenario: SEARCH/CLICK/ADD_TO_CART on the SAME
    product that later becomes the held-out purchase target must remain
    visible as history - only the purchase AT the cutoff is excluded.
    """
    events = [
        ev(1, 50, ActionType.SEARCH, t(1)),
        ev(1, 50, ActionType.CLICK, t(2)),
        ev(1, 50, ActionType.ADD_TO_CART, t(3)),
        ev(1, 50, ActionType.PURCHASE, t(5)),  # the held-out target
    ]
    profile = build_point_in_time_engagement_profile(1, events, cutoff=t(5), users_adapter=users_adapter, reviews_adapter=reviews_adapter)
    assert [s.matched_product_id for s in profile.searches] == [50]
    assert [c.product_id for c in profile.clicks] == [50]
    assert [c.product_id for c in profile.cart_items] == [50]
    assert profile.purchases == []  # the purchase itself is excluded


def test_repeat_purchase_earlier_occurrence_preserved_as_history(users_adapter, reviews_adapter):
    """Section 9: an earlier purchase of the SAME product (e.g. milk in
    January) remains legitimate history when a LATER purchase of that
    same product (milk in February) is the held-out target.
    """
    events = [ev(1, 77, ActionType.PURCHASE, t(1)), ev(1, 77, ActionType.PURCHASE, t(30))]
    profile = build_point_in_time_engagement_profile(1, events, cutoff=t(30), users_adapter=users_adapter, reviews_adapter=reviews_adapter)
    assert [p.product_id for p in profile.purchases] == [77]  # January purchase kept
    assert len(profile.purchases) == 1  # February (the target) excluded


def test_repeat_purchase_reported_as_val_target():
    """A user whose val target IS a repeat purchase (same product bought
    before AND during the val window) is not treated specially by the
    split - it's a normal val_target_ids entry.
    """
    events = {1: [
        ev(1, 77, ActionType.PURCHASE, t(1)),
        ev(1, 77, ActionType.PURCHASE, t(10)),  # repeat of the same product -> val target
        ev(1, 88, ActionType.PURCHASE, t(20)),  # test target
    ]}
    s = build_temporal_splits(events, user_ids=[1])[1]
    assert s.val_target_ids == frozenset({77})
    assert s.test_target_ids == frozenset({88})


def test_no_history_user_produces_empty_profile(users_adapter, reviews_adapter):
    profile = build_point_in_time_engagement_profile(1, [], cutoff=t(5), users_adapter=users_adapter, reviews_adapter=reviews_adapter)
    assert profile.clicks == [] and profile.purchases == [] and profile.cart_items == []
    assert profile.searches == [] and profile.chatbot_context is None


# --- eligibility partitioning -------------------------------------------

def _pf(product_id: int, is_active: bool, stock: int) -> ProductFeatures:
    return ProductFeatures(
        product_id=product_id, category_id=1, category_name="Snacks", parent_category_name=None, brand="B",
        tags=[], price=5.0, effective_price=5.0, discount_percentage=0.0, is_active=is_active, stock_quantity=stock,
        purchase_count=0, distinct_purchasers=0, cart_add_count=0, review_count=0, average_rating=None,
    )


def test_eligible_and_ineligible_targets_partitioned_correctly():
    product_features = {
        1: _pf(1, is_active=True, stock=10),
        2: _pf(2, is_active=True, stock=0),
        3: _pf(3, is_active=False, stock=10),
    }
    rules = build_eligibility_rules(EligibilityConfig(require_active=True, require_in_stock=True))
    result = split_targets_by_eligibility(frozenset({1, 2, 3}), product_features, rules)
    assert result.eligible_ids == frozenset({1})
    assert result.ineligible_ids == frozenset({2, 3})
    assert "in_stock" in result.ineligible_reasons[2]
    assert "is_active" in result.ineligible_reasons[3]


# --- leakage audit -------------------------------------------------------

def test_audit_passes_for_correctly_truncated_history():
    events = [ev(1, 10, ActionType.CLICK, t(1)), ev(1, 11, ActionType.CLICK, t(2))]
    history = events_before_cutoff(events, cutoff=t(5))
    result = audit_no_leakage(1, history, cutoff=t(5), target_ids=frozenset({99}))
    assert result.ok
    assert result.violations == []


def test_audit_catches_an_injected_future_event():
    """A deliberately-broken history (simulating a caller bug) must be
    flagged, not silently passed - this is what makes the audit a real
    check rather than a tautology.
    """
    bad_history = [ev(1, 10, ActionType.CLICK, t(1)), ev(1, 99, ActionType.PURCHASE, t(10))]  # t(10) >= cutoff t(5)
    result = audit_no_leakage(1, bad_history, cutoff=t(5), target_ids=frozenset({99}))
    assert not result.ok
    assert len(result.violations) == 1
    assert "TARGET PRODUCT" in result.violations[0]


def test_audit_catches_null_timestamp():
    bad_history = [UserInteraction(user_id=1, product_id=10, action_type=ActionType.CLICK, action_time=None)]
    result = audit_no_leakage(1, bad_history, cutoff=t(5), target_ids=frozenset())
    assert not result.ok


def test_audit_does_not_flag_legitimate_repeat_purchase_of_target_product():
    """An earlier (< cutoff) purchase of the SAME product that later
    becomes the target must NOT be flagged - see section 7/9.
    """
    history = [ev(1, 77, ActionType.PURCHASE, t(1))]  # legitimately before cutoff
    result = audit_no_leakage(1, history, cutoff=t(30), target_ids=frozenset({77}))
    assert result.ok
