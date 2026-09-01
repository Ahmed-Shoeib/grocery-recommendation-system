"""build_backend_api_adapters: the backend REST source produces the same
`AdapterBundle` shape as the synthetic / SQLite sources, and downstream
code (build_engagement_profile) consumes it unchanged.
"""

import json

from recommendation.data.adapters.backend_factory import build_backend_api_adapters
from recommendation.data.adapters.base import AdapterBundle
from recommendation.data.adapters.engagement import build_engagement_profile
from recommendation.data.backend.identity import ExternalIdentityResolver
from recommendation.data.schemas.engagement import EngagementProfile
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
    bundle = build_backend_api_adapters(client=client, resolver=resolver)
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
    assert profile.searches == []          # backend has no SEARCH activity
    assert profile.chatbot_context is None  # backend has no CHATBOT activity
    assert profile.reviews == []            # /api/reviews not implemented


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
    assert prof.preferred_category is None and prof.age_group is None
