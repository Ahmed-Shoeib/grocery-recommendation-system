"""Phase 8 API tests. Uses minimal fake adapters (not the full synthetic
dataset generator) for precise, deterministic control over which
user_ids exist/don't exist and which products are eligible/ineligible -
exactly what the required test matrix needs. Two-Tower/ranker are tiny,
untrained models (same pattern as test_serving_pipeline.py) since these
tests exercise API plumbing/contracts, not recommendation quality.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recommendation.api.app import create_app
from recommendation.api.dependencies import RecommendationService
from recommendation.data.adapters.base import (
    AdapterBundle,
    CartAdapter,
    ChatbotContextAdapter,
    ClickAdapter,
    ProductCatalogAdapter,
    PurchaseAdapter,
    ReviewAdapter,
    SearchAdapter,
    UserAdapter,
)
from recommendation.data.schemas.engagement import PurchaseRecord
from recommendation.data.schemas.product import Product
from recommendation.data.schemas.user import UserProfile
from recommendation.features.product_features import build_product_features
from recommendation.ranking.features import RANKING_FEATURE_NAMES
from recommendation.ranking.model import build_ranker_model
from recommendation.retrieval.index.faiss_index import FaissVectorIndex
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.model import build_user_tower
from recommendation.utils.config import (
    ApiConfig,
    AppConfig,
    ColdStartConfig,
    EligibilityConfig,
    RankingConfig,
    RetrievalConfig,
    TwoTowerConfig,
)

_EMBEDDING_DIM = 8
_OUTPUT_DIM = 8


class _FakeProducts(ProductCatalogAdapter):
    def __init__(self, products: list[Product]) -> None:
        self._products = products
        self._by_id = {p.id: p for p in products}

    def list_products(self) -> list[Product]:
        return self._products

    def get_product(self, product_id: int) -> Product | None:
        return self._by_id.get(product_id)


class _FakeUsers(UserAdapter):
    def __init__(self, profiles: dict[int, UserProfile]) -> None:
        self._profiles = profiles

    def get_user_profile(self, user_id: int) -> UserProfile | None:
        return self._profiles.get(user_id)

    def list_user_ids(self) -> list[int]:
        return list(self._profiles.keys())


class _FakePurchases(PurchaseAdapter):
    def __init__(self, by_user: dict[int, list[PurchaseRecord]]) -> None:
        self._by_user = by_user

    def get_purchases(self, user_id: int) -> list[PurchaseRecord]:
        return self._by_user.get(user_id, [])

    def list_all_purchases(self) -> list[PurchaseRecord]:
        return [p for records in self._by_user.values() for p in records]


class _FakeCart(CartAdapter):
    def get_cart_items(self, user_id: int) -> list:
        return []

    def list_all_cart_items(self) -> list:
        return []


class _FakeClicks(ClickAdapter):
    def get_clicks(self, user_id: int) -> list:
        return []

    def list_all_clicks(self) -> list:
        return []


class _FakeReviews(ReviewAdapter):
    def get_reviews(self, user_id: int) -> list:
        return []

    def list_all_reviews(self) -> list:
        return []


class _FakeSearch(SearchAdapter):
    def get_search_history(self, user_id: int) -> list:
        return []


class _FakeChatbot(ChatbotContextAdapter):
    def get_chatbot_context(self, user_id: int):
        return None


def _product(pid: int, category: str, is_active: bool = True, stock: int = 10) -> Product:
    return Product(
        id=pid, category_id=pid, slug=f"p{pid}", name=f"Product {pid}", price=5.0 + pid,
        category_name=category, is_active=is_active, stock_quantity=stock,
    )


INACTIVE_PRODUCT_ID = 200
OUT_OF_STOCK_PRODUCT_ID = 201
UNKNOWN_USER_ID = 999


def _build_service(**config_overrides) -> RecommendationService:
    eligible_products = [_product(100 + i, f"Cat{i % 4}") for i in range(12)]
    ineligible_products = [
        _product(INACTIVE_PRODUCT_ID, "Cat0", is_active=False),
        _product(OUT_OF_STOCK_PRODUCT_ID, "Cat1", stock=0),
    ]
    products = eligible_products + ineligible_products
    all_ids = [p.id for p in products]
    product_lookup = {p.id: p for p in products}
    product_features = build_product_features(products, [], [], [])

    rng = np.random.default_rng(0)
    product_embeddings = {pid: rng.normal(size=_EMBEDDING_DIM).astype(np.float32) for pid in all_ids}

    tt_encoder = TwoTowerFeatureEncoder.fit(
        category_names=[p.category_name for p in products], brand_names=[], age_groups=[],
        prices=[p.price for p in products], embedding_dim=_EMBEDDING_DIM,
    )
    tt_config = TwoTowerConfig(projection_dims=[16, _OUTPUT_DIM], output_dim=_OUTPUT_DIM, category_embedding_dim=4, brand_embedding_dim=4, age_group_embedding_dim=2)
    user_tower = build_user_tower(tt_encoder, tt_config)

    item_embeddings = rng.normal(size=(len(all_ids), _OUTPUT_DIM)).astype(np.float32)
    item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    vector_index = FaissVectorIndex()
    vector_index.build(all_ids, item_embeddings)

    ranker_model = build_ranker_model(input_dim=len(RANKING_FEATURE_NAMES), config=RankingConfig(hidden_units=[8]))

    profiles = {
        1: UserProfile(user_id=1),  # strong (3 purchases below)
        2: UserProfile(user_id=2),  # sparse (1 purchase below)
        3: UserProfile(user_id=3, preferred_category="Cat2"),  # no_history (0 purchases, profile exists)
    }
    purchases = {
        1: [PurchaseRecord(user_id=1, product_id=pid, order_id=i, quantity=1, unit_price=1.0) for i, pid in enumerate([100, 101, 102])],
        2: [PurchaseRecord(user_id=2, product_id=100, order_id=0, quantity=1, unit_price=1.0)],
    }
    bundle = AdapterBundle(
        products=_FakeProducts(products),
        users=_FakeUsers(profiles),
        purchases=_FakePurchases(purchases),
        cart=_FakeCart(),
        clicks=_FakeClicks(),
        reviews=_FakeReviews(),
        search=_FakeSearch(),
        chatbot=_FakeChatbot(),
    )

    config = AppConfig(
        cold_start=ColdStartConfig(strong_history_min_signals=3, sparse_history_min_signals=1),
        eligibility=EligibilityConfig(require_active=True, require_in_stock=True),
        retrieval=RetrievalConfig(backend="faiss", candidate_pool_multiplier=5, min_candidate_pool=len(all_ids)),
        api=ApiConfig(model_version="v1", default_recommendation_count=5, max_recommendation_count=20),
        **config_overrides,
    )

    return RecommendationService(
        product_lookup=product_lookup,
        product_features=product_features,
        product_embeddings=product_embeddings,
        text_embeddings={},
        all_item_ids=all_ids,
        tt_encoder=tt_encoder,
        user_tower=user_tower,
        ranker_model=ranker_model,
        vector_index=vector_index,
        bundle=bundle,
        config=config,
        ranker_model_version="ranker_v1_test",
        two_tower_model_version="two_tower_v1_test",
    )


@pytest.fixture
def service() -> RecommendationService:
    return _build_service()


@pytest.fixture
def client(service) -> TestClient:
    app = create_app(service=service)
    with TestClient(app) as c:
        yield c


# --- health / readiness ------------------------------------------------

def test_health_always_ok(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_service_loaded(client):
    response = client.get("/v1/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert all(body["checks"].values())
    assert body["model_version"] == "ranker_v1_test"


def test_ready_returns_503_when_service_not_loaded(client):
    client.app.state.service = None
    response = client.get("/v1/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


# --- cold-start tiers ----------------------------------------------------

def test_strong_history_user(client):
    response = client.get("/v1/users/1/recommendations")
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["tier"] == "strong"
    assert len(body["items"]) > 0
    assert all(item["source"] == "personalized" for item in body["items"])


def test_sparse_history_user(client):
    response = client.get("/v1/users/2/recommendations")
    assert response.status_code == 200
    assert response.json()["meta"]["tier"] == "sparse"


def test_no_history_user(client):
    response = client.get("/v1/users/3/recommendations")
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["tier"] == "no_history"
    assert all(item["source"] != "personalized" for item in body["items"])


def test_unknown_user_returns_404(client):
    response = client.get(f"/v1/users/{UNKNOWN_USER_ID}/recommendations")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "not_found"
    assert str(UNKNOWN_USER_ID) in body["message"]


# --- top-N validation ------------------------------------------------------

def test_configurable_top_n(client):
    response = client.get("/v1/users/1/recommendations", params={"limit": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["requested_top_n"] == 3
    assert len(body["items"]) <= 3


def test_invalid_top_n_zero_rejected(client):
    response = client.get("/v1/users/1/recommendations", params={"limit": 0})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_invalid_top_n_negative_rejected(client):
    response = client.get("/v1/users/1/recommendations", params={"limit": -5})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_all_422_error_bodies_share_the_same_shape(client):
    """FastAPI's automatic query validation (limit<1) and the manually
    raised HTTPException (limit>max) must not leak two different error
    body contracts to clients.
    """
    too_small = client.get("/v1/users/1/recommendations", params={"limit": 0}).json()
    too_large = client.get("/v1/users/1/recommendations", params={"limit": 999}).json()
    assert set(too_small.keys()) == set(too_large.keys()) == {"error", "message"}


def test_top_n_exceeding_max_rejected(client):
    response = client.get("/v1/users/1/recommendations", params={"limit": 999})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_user_id_path_param_must_be_integer(client):
    response = client.get("/v1/users/not-an-int/recommendations")
    assert response.status_code == 422


# --- result quality guarantees ---------------------------------------------

def test_unavailable_products_never_returned(client):
    response = client.get("/v1/users/1/recommendations", params={"limit": 20})
    assert response.status_code == 200
    product_ids = [item["product_id"] for item in response.json()["items"]]
    assert INACTIVE_PRODUCT_ID not in product_ids
    assert OUT_OF_STOCK_PRODUCT_ID not in product_ids


def test_no_duplicate_product_ids(client):
    response = client.get("/v1/users/1/recommendations", params={"limit": 20})
    product_ids = [item["product_id"] for item in response.json()["items"]]
    assert len(product_ids) == len(set(product_ids))


def test_fewer_results_than_requested_when_pool_lacks_enough_eligible_items(client):
    # 12 eligible products total; requesting more than that must not error.
    response = client.get("/v1/users/1/recommendations", params={"limit": 20})
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 12
    assert body["meta"]["returned_count"] == 12
    assert body["meta"]["requested_top_n"] == 20
    assert body["meta"]["fill_rate"] < 1.0


# --- metadata --------------------------------------------------------------

def test_response_includes_version_and_latency_metadata(client):
    response = client.get("/v1/users/1/recommendations")
    meta = response.json()["meta"]
    assert meta["api_version"] == "v1"
    assert meta["model_version"] == "ranker_v1_test"
    assert meta["latency_ms"] >= 0.0
    assert meta["generated_at"]  # ISO timestamp string present


def test_items_have_stable_rank_ordering(client):
    response = client.get("/v1/users/1/recommendations")
    ranks = [item["rank"] for item in response.json()["items"]]
    assert ranks == list(range(1, len(ranks) + 1))


# --- failure handling --------------------------------------------------

def test_pipeline_failure_returns_structured_500():
    failing_service = _build_service()

    def _raise(user_id: int, limit: int):
        raise RuntimeError("simulated pipeline failure")

    failing_service.recommend = _raise  # instance-level override, shadows the class method
    app = create_app(service=failing_service)
    with TestClient(app, raise_server_exceptions=False) as failing_client:
        response = failing_client.get("/v1/users/1/recommendations")
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_error"
    assert "Traceback" not in body["message"]
