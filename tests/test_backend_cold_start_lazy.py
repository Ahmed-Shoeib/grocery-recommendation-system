"""Cold-start correctness AND train-serve behavioral-feature parity for
the bounded-activity-window architecture (docs/data-mapping.md
19.13/19.14).

Two things must both hold for a real user:

1. A user who exists on the backend must be KNOWN (never misclassified
   as not-existing) even with zero activity in the bounded global window
   - `load_backend_users_roster` (`GET /api/users`).
2. A user's BEHAVIORAL features (purchase/cart/search counts, category
   affinity, semantic embedding, price profile - everything
   `features.user_features.build_user_features` computes) must reflect
   their COMPLETE history, not just whatever fell inside the bounded
   global window - `LazyBackendUserEventsAdapter` walks a user's
   `userId`-filtered feed to genuine completion the first time they are
   actually accessed, REPLACING (never appending to) any partial rows
   already present for them, and reuses that complete history afterward
   (TTL-cached, both to skip the network entirely when fresh and to fall
   back to a cheap incremental delta rather than a full re-walk once the
   TTL elapses).

The train-serve parity audit found the ORIGINAL version of this fix only
re-checked users with literally ZERO local events - a partially-covered
active user (some activity in-window, more outside it) was silently
undercounted. `test_partial_local_activity_still_triggers_a_complete_history_fetch`
is the regression test for exactly that bug.
"""

from __future__ import annotations

from recommendation.adapters.backend_factory import build_backend_api_adapters
from recommendation.backend.dtos import ApiActivity
from recommendation.backend.identity import ExternalIdentityResolver
from tests._backend_fakes import FakeBackendClient

_CATS = [{"slug": "groceries", "name": "Groceries"}]
_PRODS = [{"slug": "orange-juice", "productId": 501, "name": "OJ", "price": 4.0, "stockQuantity": 5, "categorySlug": "groceries"}]


def _build(tmp_path, **client_kwargs):
    client = FakeBackendClient(products=_PRODS, categories=_CATS, **client_kwargs)
    resolver = ExternalIdentityResolver(tmp_path / "reg.json")
    bundle = build_backend_api_adapters(
        client=client, resolver=resolver,
        activity_cache_path=tmp_path / "activity_cache.json",
        user_activity_cache_path=tmp_path / "user_activity_cache.json",
    )
    return bundle, client


def test_roster_only_user_is_known_with_no_activity_at_all(tmp_path):
    """A user who exists on the backend but has never generated an
    activity row (or whose only activity fell outside the bounded window)
    must be a KNOWN user, not indistinguishable from "does not exist".
    """
    bundle, _ = _build(tmp_path, roster=[{"guid": "roster-only-guid", "firstName": "Roster", "email": "r@x.invalid"}])
    ids = bundle.users.list_user_ids()
    assert len(ids) == 1
    profile = bundle.users.get_user_profile(ids[0])
    assert profile is not None


def test_roster_is_additive_not_a_replacement_for_activity_derived_users(tmp_path):
    """A user discovered only via the activity stream (no roster support -
    the pre-existing, still-supported degrade path) must still be known."""
    acts = [{"userId": "activity-only-guid", "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-01T10:00:00"}]
    bundle, _ = _build(tmp_path, activities=acts)  # roster=[] by default
    assert len(bundle.users.list_user_ids()) == 1


def test_zero_activity_user_gets_a_complete_history_fetch_that_finds_real_data(tmp_path):
    """The user has real history, but none of it is in the bounded global
    activity window (`activities=[]`); the complete-history fetch must
    still find it via the `userId`-filtered feed.
    """
    bundle, client = _build(
        tmp_path,
        roster=[{"guid": "g1", "firstName": "A", "email": "a@x.invalid"}],
        activities=[],  # nothing in the bounded global window
    )
    user_id = bundle.users.list_user_ids()[0]
    # The per-user fetch is scoped separately from the (empty) global
    # window - simulate it by handing this same fake client a per-user
    # activity row it can only find via the userId filter.
    client._activities = [ApiActivity.model_validate(
        {"userId": "g1", "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-05T10:00:00"}
    )]
    cart_items = bundle.cart.get_cart_items(user_id)
    assert len(cart_items) == 1
    assert client.activity_page_calls[-1] == "g1"  # confirms the per-user fetch actually ran


def test_partial_local_activity_still_triggers_a_complete_history_fetch(tmp_path):
    """THE REGRESSION TEST for the train-serve parity bug: a user with
    SOME activity already inside the bounded global window (so the
    OLD "zero events -> check" trigger would have skipped them) but MORE
    activity outside it must still get a complete-history fetch - their
    behavioral features must reflect the complete set, not just the
    partial in-window subset.
    """
    in_window = [{"userId": "g1", "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-01T10:00:00"}]
    bundle, client = _build(
        tmp_path, roster=[{"guid": "g1", "firstName": "A"}], activities=in_window,
    )
    user_id = bundle.users.list_user_ids()[0]
    # The user's COMPLETE per-user history (userId-filtered) has TWO
    # cart adds - one the global bounded window already had, plus one
    # older event outside it. If the adapter incorrectly skipped the
    # complete-history check (because the user already had local
    # activity), this would still read as 1 cart item, not 2.
    client._activities = [
        ApiActivity.model_validate({"userId": "g1", "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-01T10:00:00"}),
        ApiActivity.model_validate({"userId": "g1", "actionType": "AddToCart", "productId": 501, "timestamp": "2026-01-01T10:00:00"}),
    ]
    cart_items = bundle.cart.get_cart_items(user_id)
    assert len(cart_items) == 2, "partial local activity must not suppress the complete-history fetch"
    assert client.activity_page_calls[-1] == "g1"


def test_complete_history_replaces_rather_than_double_counts_the_bounded_window_subset(tmp_path):
    """The one event that was ALSO in the bounded global window must not
    be counted twice once the complete per-user fetch supersedes it.
    """
    in_window = [{"userId": "g1", "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-01T10:00:00"}]
    bundle, client = _build(tmp_path, roster=[{"guid": "g1"}], activities=in_window)
    user_id = bundle.users.list_user_ids()[0]
    # Complete history == exactly the same single event the bounded
    # window already had.
    client._activities = list(ApiActivity.model_validate(r) for r in in_window)
    cart_items = bundle.cart.get_cart_items(user_id)
    assert len(cart_items) == 1, "the same event must not be counted twice (bounded window + complete fetch)"


def test_lazy_check_is_not_repeated_within_ttl(tmp_path):
    bundle, client = _build(
        tmp_path,
        roster=[{"guid": "g1", "firstName": "A", "email": "a@x.invalid"}],
        activities=[],
    )
    user_id = bundle.users.list_user_ids()[0]
    bundle.cart.get_cart_items(user_id)
    bundle.clicks.get_clicks(user_id)
    bundle.purchases.get_purchases(user_id)
    lazy_calls = [c for c in client.activity_page_calls if c == "g1"]
    assert len(lazy_calls) == 1, "one complete-history fetch, not one per signal type"


def test_disabled_lazy_mode_never_calls_the_backend(tmp_path):
    bundle, client = _build(tmp_path, roster=[{"guid": "g1"}], activities=[])
    user_id = bundle.users.list_user_ids()[0]
    bundle.purchases.lazy_enabled = False
    bundle.cart.get_cart_items(user_id)
    per_user_calls = [c for c in client.activity_page_calls if c is not None]
    assert per_user_calls == []
