"""End-to-end proof of the 2026-09-18 user-identity migration: backend
database `User.Id` (joined via the protected `GET /api/ai/users` mapping)
is the canonical recommendation `user_id` all the way through the real
FastAPI route - not a value `ExternalIdentityResolver` mints from the
GUID.

Unlike `tests/test_backend_loader.py` (unit-level: DTOs/loader functions
in isolation) and `tests/test_backend_factory.py` (bundle-level: the full
`AdapterBundle`), this file exercises the REAL `backend_api` adapter
stack (`build_backend_api_adapters`, real `InMemoryUserAdapter`, real
`LazyBackendUserEventsAdapter`) all the way through the actual
`GET /v1/users/{user_id}/recommendations` HTTP route - the exact
production path the reported incident (backend `User.Id=1547`,
GUID `81bfc1f1-36eb-4427-b680-119ec489e156`, HTTP 404 at
2026-09-17T22:31:52Z) went through, proving the fix closes that gap
without a real trained model artifact (NO_HISTORY never touches
Two-Tower/ANN/ranker, so `None` placeholders for those three fields are
safe and sufficient here - see `serving.pipeline.generate_recommendations`,
which only calls them for STRONG/SPARSE tiers).
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from recommendation.adapters.backend_factory import build_backend_api_adapters
from recommendation.api.app import create_app
from recommendation.api.service import RecommendationService
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.config import AppConfig, ColdStartConfig, PathsConfig
from recommendation.features.pipeline import run_feature_pipeline
from recommendation.features.price import build_price_catalog_context
from tests._backend_fakes import FakeBackendClient

_CATS = [{"slug": "groceries", "name": "Groceries"}]
_PRODS = [
    {"slug": "orange-juice", "productId": 501, "name": "OJ", "price": 4.0, "stockQuantity": 50, "categorySlug": "groceries"},
    {"slug": "milk", "productId": 502, "name": "Milk", "price": 2.5, "stockQuantity": 10, "categorySlug": "groceries"},
]

_CUSTOMER_USER_ID = 1547
_CUSTOMER_GUID = "81bfc1f1-36eb-4427-b680-119ec489e156"


def _fake_encoder():
    class _Encoder:
        model_name = "fake"
        embedding_dim = 8

        def encode(self, texts, normalize=False):
            if not texts:
                return np.empty((0, self.embedding_dim), dtype=np.float32)
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

    return _Encoder()


def _build_service(tmp_path, client: FakeBackendClient) -> RecommendationService:
    resolver = ExternalIdentityResolver(tmp_path / "reg.json")
    bundle = build_backend_api_adapters(
        client=client, resolver=resolver,
        activity_cache_path=tmp_path / "activity_cache.json",
        user_activity_cache_path=tmp_path / "user_activity_cache.json",
    )
    config = AppConfig(
        paths=PathsConfig(data_source="backend_api"),
        cold_start=ColdStartConfig(strong_history_min_signals=5, sparse_history_min_signals=1),
    )
    feature_result = run_feature_pipeline(bundle, config, encoder=_fake_encoder())
    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    price_context = build_price_catalog_context(products)

    # NO_HISTORY never calls tt_encoder/user_tower/ranker_model/vector_index
    # (serving.pipeline.generate_recommendations only reaches
    # `_personalized_candidates` for STRONG/SPARSE tiers) - None is a safe,
    # sufficient placeholder for a test that only exercises the NO_HISTORY/
    # unknown-user paths.
    return RecommendationService(
        product_lookup=product_lookup,
        product_features=feature_result.product_features,
        product_embeddings=feature_result.product_embeddings.as_dict(),
        text_embeddings={},
        all_item_ids=[p.id for p in products],
        tt_encoder=None,
        user_tower=None,
        ranker_model=None,
        vector_index=None,
        bundle=bundle,
        config=config,
        price_context=price_context,
        engagement_profiles=feature_result.engagement_profiles,
    )


def test_reported_customer_reaches_no_history_not_unknown_user_404(tmp_path):
    """THE regression test for the reported production incident: backend
    User.Id 1547 / GUID 81bfc1f1-36eb-4427-b680-119ec489e156, zero
    activity, must resolve through the real HTTP route into the
    NO_HISTORY/cold-start path with a 200 - never the old
    UnknownUserError-> 404 the incident actually observed.
    """
    client = FakeBackendClient(
        products=_PRODS, categories=_CATS,
        ai_identities=[{"userId": _CUSTOMER_USER_ID, "userGuid": _CUSTOMER_GUID}],
        roster=[{"guid": _CUSTOMER_GUID, "firstName": "Customer", "preferredCategories": []}],
        activities=[],  # zero history, matching the real reported customer
    )
    service = _build_service(tmp_path, client)
    app = create_app(service=service)
    with TestClient(app) as test_client:
        response = test_client.get(f"/v1/users/{_CUSTOMER_USER_ID}/recommendations?limit=10")

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["meta"]["tier"] == "no_history"
    assert body["meta"]["user_id"] == _CUSTOMER_USER_ID
    assert all(item["source"] == "global_popularity" for item in body["items"])


def test_reported_customer_with_preferred_category_uses_category_fallback(tmp_path):
    """New user + a preferred category -> the NO_HISTORY waterfall's
    preferred_category source is used, not global_popularity - proving
    the fallback hierarchy is genuinely reachable for a real backend_api
    user resolved via the new identity path.
    """
    client = FakeBackendClient(
        products=_PRODS, categories=_CATS,
        ai_identities=[{"userId": _CUSTOMER_USER_ID, "userGuid": _CUSTOMER_GUID}],
        roster=[{
            "guid": _CUSTOMER_GUID, "firstName": "Customer",
            "preferredCategories": [{"category": {"slug": "groceries", "name": "Groceries"}}],
        }],
        activities=[],
    )
    service = _build_service(tmp_path, client)
    app = create_app(service=service)
    with TestClient(app) as test_client:
        response = test_client.get(f"/v1/users/{_CUSTOMER_USER_ID}/recommendations?limit=10")

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["meta"]["tier"] == "no_history"
    assert any(item["source"] == "preferred_category" for item in body["items"])


def test_truly_unknown_integer_still_receives_the_intentional_404(tmp_path):
    """An integer never present in the authoritative /api/ai/users mapping
    must still be a genuine 404 - the migration must not turn EVERY id
    into a known user, only ones the identity mapping actually vouches
    for. This is the existing, correct behavior this migration must not
    break.
    """
    client = FakeBackendClient(
        products=_PRODS, categories=_CATS,
        ai_identities=[{"userId": _CUSTOMER_USER_ID, "userGuid": _CUSTOMER_GUID}],
        roster=[{"guid": _CUSTOMER_GUID, "firstName": "Customer"}],
        activities=[],
    )
    service = _build_service(tmp_path, client)
    app = create_app(service=service)
    with TestClient(app) as test_client:
        response = test_client.get("/v1/users/999999/recommendations?limit=10")

    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_existing_user_with_history_still_personalizes_after_migration(tmp_path):
    """A STRONG-tier user (enough real activity) must still be classified
    correctly by tier/engagement counting after the identity migration -
    the migration only changes WHICH integer represents a user, never how
    much history they need to stop being cold-start. (Full personalized
    ranking itself needs a real trained model, out of scope here - this
    asserts the identity/tier-classification half via the profile
    endpoint, which needs no model artifacts.)
    """
    other_guid = "05d74037-20a6-4399-82dd-66488575b5a8"
    other_user_id = 82
    client = FakeBackendClient(
        products=_PRODS, categories=_CATS,
        ai_identities=[{"userId": other_user_id, "userGuid": other_guid}],
        roster=[{"guid": other_guid, "firstName": "Existing", "preferredCategories": []}],
        activities=[
            {"userId": other_guid, "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-01T10:00:00"},
            {"userId": other_guid, "actionType": "AddToCart", "productId": 502, "timestamp": "2026-08-02T10:00:00"},
            {"userId": other_guid, "actionType": "PlaceOrder", "productId": 501, "timestamp": "2026-08-03T10:00:00"},
            {"userId": other_guid, "actionType": "SearchProduct", "productId": 502, "timestamp": "2026-08-04T10:00:00"},
            {"userId": other_guid, "actionType": "ViewProduct", "productId": 501, "timestamp": "2026-08-05T10:00:00"},
        ],
    )
    service = _build_service(tmp_path, client)
    app = create_app(service=service)
    with TestClient(app) as test_client:
        response = test_client.get(f"/v1/users/{other_user_id}/profile")

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["user_id"] == other_user_id
    assert body["tier"] == "strong"
    assert body["total_engagement_events"] == 5
