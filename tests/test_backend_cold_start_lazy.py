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

from types import SimpleNamespace

import numpy as np

import recommendation.api.service as service_module
from recommendation.adapters.backend_factory import build_backend_api_adapters
from recommendation.backend.dtos import ApiActivity
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.config import AppConfig, PathsConfig
from recommendation.ui.data_access import load_user_detail
from tests._backend_fakes import FakeBackendClient

_FAKE_EMBEDDING_DIM = 384


def _fake_encoder():
    def _encode(texts, normalize=False):
        if not texts:
            return np.empty((0, _FAKE_EMBEDDING_DIM), dtype=np.float32)
        return np.zeros((len(texts), _FAKE_EMBEDDING_DIM), dtype=np.float32)

    return SimpleNamespace(model_name="fake", embedding_dim=_FAKE_EMBEDDING_DIM, encode=_encode)


def _service_from_snapshot(snapshot, config):
    """A minimal stand-in for `RecommendationService` carrying only the
    fields `ui.data_access.load_user_detail` actually reads - avoids
    needing real Two-Tower/ranker/VectorIndex artifacts for a test that
    never calls `.recommend()`.
    """
    return SimpleNamespace(
        bundle=snapshot.bundle,
        product_lookup=snapshot.product_lookup,
        product_embeddings=snapshot.product_embeddings,
        text_embeddings=snapshot.text_embeddings,
        price_context=snapshot.price_context,
        engagement_profiles=snapshot.engagement_profiles,
        config=config,
    )

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


def test_rebuilt_bundle_reuses_persisted_complete_history_without_losing_data(tmp_path):
    """Regression test for the `RecommendationService.maybe_refresh`
    cache-rehydration bug: a user whose complete history sits entirely
    OUTSIDE the bounded global window must still show their full history
    immediately after a brand-new `AdapterBundle` is built from the SAME
    persisted user-activity-cache store - exactly what `maybe_refresh` does
    on every refresh cycle (it never reuses the old adapter instance).

    Before the fix, `LazyBackendUserEventsAdapter._ensure_complete`'s
    "reuse the persisted complete history - no network call" shortcut
    returned without ever replaying `entry.rows` into the freshly
    constructed instance's `_by_user_and_type` index, so a fresh instance
    silently reverted to whatever the (possibly empty) bounded window
    contained for that user until the per-user TTL expired.
    """
    user_cache_path = tmp_path / "user_activity_cache.json"
    activity_cache_path = tmp_path / "activity_cache.json"

    client = FakeBackendClient(
        products=_PRODS, categories=_CATS,
        roster=[{"guid": "g1", "firstName": "A", "email": "a@x.invalid"}],
        activities=[],  # nothing in the bounded global window, ever, for this user
    )
    resolver = ExternalIdentityResolver(tmp_path / "reg.json")
    bundle1 = build_backend_api_adapters(
        client=client, resolver=resolver,
        activity_cache_path=activity_cache_path, user_activity_cache_path=user_cache_path,
    )
    user_id = bundle1.users.list_user_ids()[0]
    client._activities = [
        ApiActivity.model_validate(
            {"userId": "g1", "actionType": "AddToCart", "productId": 501, "timestamp": f"2026-08-0{i}T10:00:00"}
        )
        for i in range(1, 6)
    ]
    assert len(bundle1.cart.get_cart_items(user_id)) == 5

    # Simulate `RecommendationService.maybe_refresh`: a brand-new
    # AdapterBundle (brand-new adapter instance), same persisted
    # user-activity-cache path, bounded window still empty for this user.
    client._activities = []
    per_user_calls_before = len([c for c in client.activity_page_calls if c == "g1"])
    bundle2 = build_backend_api_adapters(
        client=client, resolver=resolver,
        activity_cache_path=activity_cache_path, user_activity_cache_path=user_cache_path,
    )
    cart_items_2 = bundle2.cart.get_cart_items(user_id)
    assert len(cart_items_2) == 5, "a freshly rebuilt AdapterBundle must still see the persisted complete history"
    per_user_calls_after = len([c for c in client.activity_page_calls if c == "g1"])
    assert per_user_calls_after == per_user_calls_before, (
        "reusing a fresh, already-complete persisted entry must not trigger a new per-user network fetch "
        "(the global bounded-window sync's own call is expected and irrelevant here)"
    )


def test_profile_agrees_with_recommendations_after_lazy_sync_discovers_complete_history(tmp_path, monkeypatch):
    """Regression test for the `/profile` staleness bug: `load_user_detail`
    (backing `GET /v1/users/{id}/profile`) used to read
    `RecommendationService.engagement_profiles`, a bootstrap/refresh-time
    snapshot built with the lazy per-user sync DISABLED (see
    `api.service._load_data_snapshot`'s `lazy_enabled` toggle) - so it only
    ever reflected the bounded global activity window, never a user's
    later-discovered complete history, even after `/recommendations` (which
    goes straight through `service.bundle`, not the stale snapshot dict)
    had already found and cached it.

    Sequence proven here:
    1. backend user has 5 real events, entirely OUTSIDE the bounded window;
    2. the bootstrap snapshot's `engagement_profiles` entry for this user
       reflects 0 events (proves the snapshot really is incomplete);
    3. a `/recommendations`-equivalent call triggers the lazy sync and
       correctly finds all 5 events;
    4. `/profile` afterward must report the SAME tier/count `/recommendations`
       used - not the stale bootstrap snapshot;
    5. repeated profile/recommendation calls stay stable at 5 events, never
       duplicating.
    """
    roster = [{"guid": "g1", "firstName": "A", "email": "a@x.invalid"}]
    client = FakeBackendClient(products=_PRODS, categories=_CATS, roster=roster, activities=[])
    resolver = ExternalIdentityResolver(tmp_path / "reg.json")

    def fake_build(config):
        return build_backend_api_adapters(
            client=client, resolver=resolver,
            activity_cache_path=tmp_path / "activity_cache.json",
            user_activity_cache_path=tmp_path / "user_activity_cache.json",
        )

    monkeypatch.setattr(service_module, "build_backend_api_adapters", fake_build)
    config = AppConfig(paths=PathsConfig(data_source="backend_api"))

    # --- step 1/2: bootstrap snapshot, bounded window empty for this user ---
    # (the backend already has this user's 5-event history at this point -
    # it has simply never been outside the bounded global window that the
    # bootstrap pass reads; the bootstrap pass runs with the lazy per-user
    # sync DISABLED, so its snapshot must not "accidentally" discover it
    # either - proving the snapshot really is incomplete, not just empty
    # because the backend has nothing yet.)
    snapshot = service_module._load_data_snapshot(config, _fake_encoder())
    user_id = snapshot.bundle.users.list_user_ids()[0]
    bootstrap_profile = snapshot.engagement_profiles[user_id]
    assert bootstrap_profile.cart_items == [], "the bootstrap snapshot must NOT already have the off-window history"

    service = _service_from_snapshot(snapshot, config)

    # --- step 3: the user's real, complete history becomes visible to the
    # backend's userId-filtered feed (entirely outside the bounded window) ---
    client._activities = [
        ApiActivity.model_validate(
            {"userId": "g1", "actionType": "AddToCart", "productId": 501, "timestamp": f"2026-08-0{i}T10:00:00"}
        )
        for i in range(1, 6)
    ]

    # `/recommendations` goes straight through `service.bundle`, triggering
    # the lazy per-user sync.
    from recommendation.adapters.engagement import build_engagement_profile
    live_engagement = build_engagement_profile(
        user_id, service.bundle.users, service.bundle.purchases, service.bundle.cart,
        service.bundle.clicks, service.bundle.search, service.bundle.chatbot, service.bundle.reviews,
    )
    assert len(live_engagement.cart_items) == 5, "the lazy sync must find the complete off-window history"

    # --- step 4: /profile afterward must agree, not report the stale snapshot ---
    detail_after = load_user_detail(service, user_id)
    assert detail_after.tier.value == "strong", "5 events >= strong_history_min_signals=5"
    assert len(detail_after.engagement.cart_items) == 5

    # --- step 5: repeated calls stay stable, no duplication ---
    detail_again = load_user_detail(service, user_id)
    assert detail_again.tier.value == "strong"
    assert len(detail_again.engagement.cart_items) == 5, "repeated profile calls must not duplicate events"

    # The untouched bootstrap `engagement_profiles` dict entry is left as-is
    # (by design - it's a snapshot, only refreshed on the next
    # maybe_refresh cycle); this is what /profile no longer reads from.
    assert len(snapshot.engagement_profiles[user_id].cart_items) == 0
