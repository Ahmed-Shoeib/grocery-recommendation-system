"""`backend.user_activity_cache`: atomic persistence + corruption recovery
for the per-user COMPLETE-history cache (docs/data-mapping.md 19.14).

Also covers the cross-restart integration proof (Section 8 of the
train-serve parity audit): a second `build_backend_api_adapters` call
sharing the same cache files reuses a user's already-complete history
with NO network call, and a corrupt per-user cache file degrades to
"nothing cached yet" rather than crashing the whole adapter build.
"""

from __future__ import annotations

import json

from recommendation.adapters.backend_factory import build_backend_api_adapters
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.backend.user_activity_cache import (
    CACHE_FORMAT_VERSION,
    UserActivityCacheStore,
    load_user_activity_cache_store,
    save_user_activity_cache_store,
)
from recommendation.backend.activity_cache import new_cache_state
from tests._backend_fakes import FakeBackendClient

_CATS = [{"slug": "groceries", "name": "Groceries"}]
_PRODS = [{"slug": "oj", "productId": 501, "name": "OJ", "price": 4.0, "stockQuantity": 5, "categorySlug": "groceries"}]


# --- unit: load/save -----------------------------------------------------


def test_missing_file_returns_an_empty_store(tmp_path):
    store = load_user_activity_cache_store(tmp_path / "nope.json")
    assert store.entries == {} and store.complete == {}


def test_round_trips_through_save_and_load(tmp_path):
    path = tmp_path / "store.json"
    store = UserActivityCacheStore()
    store.entries["g1"] = new_cache_state([{"user_id": "g1", "action_type": "AddToCart", "product_id": 1, "timestamp": "2026-08-01T00:00:00", "slug": None}], "2026-08-01T00:00:00", set())
    store.complete["g1"] = True
    save_user_activity_cache_store(path, store)

    loaded = load_user_activity_cache_store(path)
    assert loaded.complete["g1"] is True
    assert loaded.entries["g1"].rows == store.entries["g1"].rows


def test_save_is_atomic_no_temp_file_left_behind(tmp_path):
    path = tmp_path / "store.json"
    save_user_activity_cache_store(path, UserActivityCacheStore())
    leftovers = [p for p in tmp_path.iterdir() if p.name != "store.json"]
    assert leftovers == []


def test_corrupt_json_degrades_to_an_empty_store(tmp_path):
    path = tmp_path / "store.json"
    path.write_text("{ not valid", encoding="utf-8")
    store = load_user_activity_cache_store(path)
    assert store.entries == {} and store.complete == {}


def test_unknown_version_degrades_to_an_empty_store(tmp_path):
    path = tmp_path / "store.json"
    path.write_text(json.dumps({"version": CACHE_FORMAT_VERSION + 1, "users": {}}), encoding="utf-8")
    store = load_user_activity_cache_store(path)
    assert store.entries == {}


def test_one_malformed_user_entry_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "store.json"
    path.write_text(json.dumps({
        "version": CACHE_FORMAT_VERSION,
        "users": {
            "g1": {"fetched_at": "t", "rows": [], "complete": True},
            "g2": {"rows": []},  # missing required fetched_at
        },
    }), encoding="utf-8")
    store = load_user_activity_cache_store(path)
    assert set(store.entries) == {"g1"}


# --- integration: persistence across a simulated restart -----------------


def _build(tmp_path, roster, activities, activity_cache_name="activity_cache.json"):
    client = FakeBackendClient(products=_PRODS, categories=_CATS, roster=roster, activities=activities)
    resolver = ExternalIdentityResolver(tmp_path / "reg.json")
    bundle = build_backend_api_adapters(
        client=client, resolver=resolver,
        activity_cache_path=tmp_path / activity_cache_name,
        user_activity_cache_path=tmp_path / "user_activity_cache.json",
    )
    return bundle, client


def test_completeness_persists_across_a_simulated_restart(tmp_path):
    """Two SEPARATE `build_backend_api_adapters` calls (simulating a
    process restart / a periodic `maybe_refresh` rebuild) sharing the same
    on-disk per-user cache: the second one must recognize the user is
    already known-complete and skip the network fetch entirely (within
    the TTL), rather than re-walking their history from scratch.
    """
    acts = [{"userId": "g1", "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-01T10:00:00"}]

    bundle1, client1 = _build(tmp_path, roster=[{"guid": "g1"}], activities=acts)
    user_id = bundle1.users.list_user_ids()[0]
    bundle1.cart.get_cart_items(user_id)  # triggers + persists the first complete fetch
    assert [c for c in client1.activity_page_calls if c == "g1"] == ["g1"]

    # A fresh client/adapter instance (as a real restart or refresh would
    # produce) sharing the SAME on-disk cache files.
    bundle2, client2 = _build(tmp_path, roster=[{"guid": "g1"}], activities=acts)
    bundle2.cart.get_cart_items(user_id)
    assert [c for c in client2.activity_page_calls if c == "g1"] == [], (
        "a persisted, still-fresh complete history must not trigger a second network fetch after a restart"
    )


def test_corrupt_per_user_cache_degrades_to_a_fresh_fetch_not_a_crash(tmp_path):
    (tmp_path / "user_activity_cache.json").write_text("{ not json", encoding="utf-8")
    acts = [{"userId": "g1", "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-01T10:00:00"}]
    bundle, client = _build(tmp_path, roster=[{"guid": "g1"}], activities=acts)
    user_id = bundle.users.list_user_ids()[0]
    cart_items = bundle.cart.get_cart_items(user_id)
    assert len(cart_items) == 1
