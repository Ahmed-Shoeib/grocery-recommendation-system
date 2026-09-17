"""build_backend_api_adapters: the backend REST source produces the same
`AdapterBundle` shape as the synthetic / SQLite sources, and downstream
code (build_engagement_profile) consumes it unchanged.
"""

import json

from recommendation.adapters.backend_factory import build_backend_api_adapters
from recommendation.adapters.base import AdapterBundle
from recommendation.adapters.engagement import build_engagement_profile
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.schemas.engagement import EngagementProfile
from tests._backend_fakes import FakeBackendClient

_CATS = [{"slug": "groceries", "name": "Groceries"}]
_PRODS = [
    {"slug": "orange-juice", "name": "Orange Juice", "price": 4.0, "stockQuantity": 50, "categorySlug": "groceries"},
    {"slug": "milk", "name": "Milk", "price": 2.5, "stockQuantity": 10, "categorySlug": "groceries"},
]
_ACTS = [
    {"userId": "guid-1", "actionType": "ViewProduct", "slug": "orange-juice", "timestamp": "2026-08-01T09:00:00"},
    {"userId": "guid-1", "actionType": "AddToCart", "slug": "milk", "timestamp": "2026-08-02T09:00:00"},
    {"userId": "guid-1", "actionType": "PlaceOrder", "slug": "orange-juice", "timestamp": "2026-08-03T09:00:00"},
    {"userId": "guid-2", "actionType": "ViewProduct", "slug": "milk", "timestamp": "2026-08-04T09:00:00"},
]


def _build(tmp_path, **client_kwargs):
    client = FakeBackendClient(products=_PRODS, categories=_CATS, activities=_ACTS, **client_kwargs)
    resolver = ExternalIdentityResolver(tmp_path / "reg.json")
    bundle = build_backend_api_adapters(
        client=client, resolver=resolver, activity_cache_path=tmp_path / "activity_cache.json", user_activity_cache_path=tmp_path / "user_activity_cache.json"
    )
    return bundle, resolver


def test_produces_a_full_adapter_bundle(tmp_path):
    bundle, _ = _build(tmp_path)
    assert isinstance(bundle, AdapterBundle)
    assert len(bundle.products.list_products()) == 2
    assert sorted(bundle.users.list_user_ids()) == [1, 2]


def test_engagement_profile_builds_from_backend_bundle(tmp_path):
    bundle, _ = _build(tmp_path)
    profile = build_engagement_profile(
        1, bundle.users, bundle.purchases, bundle.cart, bundle.clicks, bundle.search, bundle.chatbot, bundle.reviews
    )
    assert isinstance(profile, EngagementProfile)
    assert len(profile.clicks) == 1
    assert len(profile.cart_items) == 1
    assert len(profile.purchases) == 1
    assert profile.searches == []          # this fixture's _ACTS has no SearchProduct rows
    assert profile.chatbot_context is None  # this fixture's _ACTS has no Chatbot rows
    # This fixture's _PRODS carries no productId, so /api/reviews rows
    # (which this bundle never fetches anyway - no reviews in FakeBackendClient
    # by default) would be product-unjoinable if present. See the dedicated
    # productId-bearing fixture test below for the live-shape, all-signals case.
    assert profile.reviews == []


def test_engagement_profile_with_product_id_bearing_fixture_gets_all_five_signals(tmp_path):
    """The live shape since the 2026-09-15 switch: every product/activity
    row carries `productId`, and reviews resolve on both sides. Exercises
    SEARCH, CHATBOT, and the review product+user join through the full
    factory, not just the loader unit tests.
    """
    prods = [{"slug": "orange-juice", "productId": 501, "name": "OJ", "price": 4.0, "categorySlug": "groceries"}]
    acts = [
        {"userId": "guid-1", "actionType": "SearchProduct", "productId": 501, "timestamp": "2026-08-01T09:00:00"},
        {"userId": "guid-1", "actionType": "Chatbot", "productId": 501, "timestamp": "2026-08-01T09:01:00"},
        {"userId": "guid-1", "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-02T09:00:00"},
    ]
    client = FakeBackendClient(
        products=prods, categories=_CATS, activities=acts,
        reviews=[{"reviewId": 1, "userId": 9, "userGuid": "guid-1", "productId": 501, "rating": 5,
                  "createdAt": "2026-09-01T10:00:00"}],
    )
    resolver = ExternalIdentityResolver(tmp_path / "reg.json")
    bundle = build_backend_api_adapters(
        client=client, resolver=resolver, activity_cache_path=tmp_path / "activity_cache.json", user_activity_cache_path=tmp_path / "user_activity_cache.json"
    )

    profile = build_engagement_profile(
        1, bundle.users, bundle.purchases, bundle.cart, bundle.clicks, bundle.search, bundle.chatbot, bundle.reviews
    )
    assert len(profile.searches) == 1
    assert profile.chatbot_context is not None
    assert profile.chatbot_context.mentioned_product_ids == [501]  # canonical id == real Product.Id, no remapping
    assert len(profile.cart_items) == 1
    assert len(profile.reviews) == 1
    assert profile.reviews[0].product_id == 501


def test_purchase_signal_comes_only_from_activities_not_orders(tmp_path):
    # No /api/orders call exists on the client at all - a PlaceOrder
    # activity row is the sole purchase source, so no double counting.
    bundle, _ = _build(tmp_path)
    assert not hasattr(FakeBackendClient(), "list_orders")
    all_purchases = bundle.purchases.list_all_purchases()
    assert len(all_purchases) == 1
    assert all_purchases[0].product_id == bundle.products.get_product(all_purchases[0].product_id).id


def test_identity_registry_is_persisted(tmp_path):
    _build(tmp_path)
    doc = json.loads((tmp_path / "reg.json").read_text(encoding="utf-8"))
    assert set(doc["namespaces"]["user"]["by_key"]) == {"guid-1", "guid-2"}
    assert set(doc["namespaces"]["product"]["by_key"]) == {"orange-juice", "milk"}


def test_rebuild_reuses_persisted_ids(tmp_path):
    bundle1, _ = _build(tmp_path)
    p1 = {p.slug: p.id for p in bundle1.products.list_products()}
    bundle2, _ = _build(tmp_path)
    p2 = {p.slug: p.id for p in bundle2.products.list_products()}
    assert p1 == p2


def test_bare_user_profiles_when_user_endpoint_unavailable(tmp_path):
    bundle, _ = _build(tmp_path, users_status=401)
    prof = bundle.users.get_user_profile(1)
    assert prof is not None
    assert prof.preferred_categories == [] and prof.age_group is None


# --- 2026-09-17 identity refactor: the canonical product id IS the
# backend's own `Product.Id`, never a value minted by
# `ExternalIdentityResolver` (docs/data-mapping.md 19.5/19.16) ---

def test_catalog_product_ids_are_the_real_backend_product_ids_not_resolver_minted(tmp_path):
    """The exact regression this refactor fixes: a live backend-integration
    trace found the recommender's product id space was a dense, unrelated
    `1..N` sequence (product_id=23 for a real `Product.Id=105`, reproduced
    for every sampled recommendation). `RawProduct.id` must now equal
    `productId` verbatim, non-contiguous gaps and all.
    """
    prods = [
        {"slug": "pineapple", "productId": 105, "name": "Pineapple", "price": 4.0, "categorySlug": "groceries"},
        {"slug": "zucchini", "productId": 162, "name": "Zucchini", "price": 2.5, "categorySlug": "groceries"},
        {"slug": "apple-red-delicious", "productId": 85, "name": "Apple Red Delicious", "price": 3.0, "categorySlug": "groceries"},
    ]
    client = FakeBackendClient(products=prods, categories=_CATS, activities=[])
    resolver = ExternalIdentityResolver(tmp_path / "reg.json")
    bundle = build_backend_api_adapters(
        client=client, resolver=resolver, activity_cache_path=tmp_path / "activity_cache.json", user_activity_cache_path=tmp_path / "user_activity_cache.json"
    )

    by_slug = {p.slug: p.id for p in bundle.products.list_products()}
    assert by_slug == {"pineapple": 105, "zucchini": 162, "apple-red-delicious": 85}

    # The resolver's `product` namespace must never have been touched -
    # numeric ids bypass it entirely now.
    doc = json.loads((tmp_path / "reg.json").read_text(encoding="utf-8"))
    assert doc["namespaces"]["product"]["by_key"] == {}
