"""Phase 8 API tests. Uses minimal fake adapters (not the full synthetic
dataset generator) for precise, deterministic control over which
user_ids exist/don't exist and which products are eligible/ineligible -
exactly what the required test matrix needs. Two-Tower/ranker are tiny,
untrained models (same pattern as test_serving_pipeline.py) since these
tests exercise API plumbing/contracts, not recommendation quality.
"""

from __future__ import annotations

import inspect
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recommendation.api import routes as routes_module
from recommendation.api.app import create_app
from recommendation.api.dependencies import RecommendationService, resolve_models_root
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
from recommendation.data.schemas.engagement import EngagementProfile, PurchaseRecord
from recommendation.data.schemas.product import Product
from recommendation.data.schemas.user import UserProfile
from recommendation.evaluation.offline_report import (
    REPORT_SCHEMA_VERSION,
    OfflineEvalSplitReport,
    OfflineEvaluationReport,
    save_offline_report,
)
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
    PathsConfig,
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

    engagement_profiles = {
        1: EngagementProfile(user_id=1, profile=profiles[1], purchases=purchases[1]),
        2: EngagementProfile(user_id=2, profile=profiles[2], purchases=purchases[2]),
        3: EngagementProfile(user_id=3, profile=profiles[3]),
    }

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
        engagement_profiles=engagement_profiles,
        ranker_metadata={"run_id": "test_run_1", "dataset_fingerprint_sha256_16": "fingerprint_abc"},
        two_tower_metadata={"run_id": "test_run_1", "dataset_fingerprint_sha256_16": "fingerprint_abc"},
    )


@pytest.fixture
def service(tmp_path) -> RecommendationService:
    # `paths.models_dir` pointed at a per-test tmp dir so
    # `resolve_models_root(service.config)` (used by GET /v1/metrics/offline
    # to find the persisted offline_report.json) never touches the real
    # models/sqlite_baseline/ directory.
    return _build_service(paths=PathsConfig(models_dir=str(tmp_path)))


@pytest.fixture
def client(service) -> TestClient:
    app = create_app(service=service)
    with TestClient(app) as c:
        yield c


def _offline_report_path(service: RecommendationService):
    return resolve_models_root(service.config) / "offline_report.json"


def _valid_offline_report(**overrides) -> OfflineEvaluationReport:
    split = OfflineEvalSplitReport(
        split_name="val", num_cases=2, precision_at_k={5: 0.4}, recall_at_k={5: 0.3}, hit_rate_at_k={5: 0.6},
        ndcg_at_k={5: 0.5}, mrr=0.5, mean_distinct_categories=2.0, catalog_coverage=0.8, mean_fill_rate=1.0,
    )
    test_split = OfflineEvalSplitReport(
        split_name="test", num_cases=2, precision_at_k={5: 0.3}, recall_at_k={5: 0.2}, hit_rate_at_k={5: 0.5},
        ndcg_at_k={5: 0.4}, mrr=0.4, mean_distinct_categories=1.5, catalog_coverage=0.7, mean_fill_rate=0.9,
    )
    fields = dict(
        schema_version=REPORT_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc),
        run_id="test_run_1",
        ranker_model_version="ranker_v1_test",
        two_tower_model_version="two_tower_v1_test",
        data_source="C:/fake/backend_shaped_synthetic.db",
        dataset_fingerprint_sha256_16="fingerprint_abc",
        recency_enabled=True,
        recency_half_life_days=21.0,
        include_price_features=True,
        price_tier_boundaries=[4.0, 6.7],
        k_values=[5, 10, 20],
        top_n=10,
        val_report=split,
        test_report=test_split,
    )
    fields.update(overrides)
    return OfflineEvaluationReport(**fields)


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

# --- STEP 9: enriched recommendation items ----------------------------------

def test_recommendation_items_include_display_fields(client):
    response = client.get("/v1/users/1/recommendations")
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["product_name"].startswith("Product")
    assert item["category"] is not None
    assert "brand" in item and "price" in item and "is_active" in item and "stock_quantity" in item


# --- STEP 9: GET /v1/users ---------------------------------------------------

def test_get_users_lists_all_known_users(client):
    response = client.get("/v1/users")
    assert response.status_code == 200
    user_ids = [u["user_id"] for u in response.json()["users"]]
    assert user_ids == [1, 2, 3]


def test_get_users_returns_503_when_service_not_loaded(client):
    client.app.state.service = None
    response = client.get("/v1/users")
    assert response.status_code == 503


# --- STEP 9: GET /v1/users/{id}/profile --------------------------------------

def test_get_user_profile_known_user(client):
    response = client.get("/v1/users/1/profile")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 1
    assert body["tier"] == "strong"
    assert body["purchase_count"] == 3
    assert isinstance(body["purchases"], list) and len(body["purchases"]) == 3


def test_get_user_profile_no_history_user(client):
    response = client.get("/v1/users/3/profile")
    assert response.status_code == 200
    assert response.json()["tier"] == "no_history"


def test_get_user_profile_unknown_user_returns_404(client):
    response = client.get(f"/v1/users/{UNKNOWN_USER_ID}/profile")
    assert response.status_code == 404


# --- STEP 9 fix: GET /v1/metrics/offline is a cheap read of a persisted report ---
# (docs/data-mapping.md section 18 follow-up - see `evaluation.offline_report`
# module docstring). The endpoint must NEVER call `generate_recommendations`/
# run an evaluation pass itself - see the non-recomputation tests below.

def test_get_offline_metrics_valid_report_returns_200_with_provenance(service, client):
    save_offline_report(_offline_report_path(service), _valid_offline_report())
    response = client.get("/v1/metrics/offline")
    assert response.status_code == 200
    body = response.json()
    assert body["val_report"]["split_name"] == "val"
    assert body["test_report"]["split_name"] == "test"
    assert body["val_report"]["num_cases"] == 2
    provenance = body["provenance"]
    assert provenance["run_id"] == "test_run_1"
    assert provenance["ranker_model_version"] == "ranker_v1_test"
    assert provenance["two_tower_model_version"] == "two_tower_v1_test"
    assert provenance["dataset_fingerprint_sha256_16"] == "fingerprint_abc"
    assert provenance["k_values"] == [5, 10, 20]
    assert provenance["top_n"] == 10


def test_get_offline_metrics_missing_report_returns_409(client):
    # No offline_report.json written for this test's tmp models_dir.
    response = client.get("/v1/metrics/offline")
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "offline_report_unavailable"
    assert "generate_offline_report" in body["message"]


def test_get_offline_metrics_malformed_json_returns_409(service, client):
    path = _offline_report_path(service)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    response = client.get("/v1/metrics/offline")
    assert response.status_code == 409
    assert response.json()["error"] == "offline_report_unavailable"


def test_get_offline_metrics_malformed_shape_returns_409(service, client):
    path = _offline_report_path(service)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema_version": 1, "run_id": "x"}', encoding="utf-8")
    response = client.get("/v1/metrics/offline")
    assert response.status_code == 409
    assert response.json()["error"] == "offline_report_unavailable"


def test_get_offline_metrics_run_id_mismatch_returns_409(service, client):
    save_offline_report(_offline_report_path(service), _valid_offline_report(run_id="some_other_run"))
    response = client.get("/v1/metrics/offline")
    assert response.status_code == 409
    body = response.json()
    assert body["error"] == "offline_report_unavailable"
    assert "run_id" in body["message"]


def test_get_offline_metrics_dataset_fingerprint_mismatch_returns_409(service, client):
    save_offline_report(_offline_report_path(service), _valid_offline_report(dataset_fingerprint_sha256_16="different_fingerprint"))
    response = client.get("/v1/metrics/offline")
    assert response.status_code == 409
    assert "dataset_fingerprint" in response.json()["message"]


def test_get_offline_metrics_ranker_model_version_mismatch_returns_409(service, client):
    save_offline_report(_offline_report_path(service), _valid_offline_report(ranker_model_version="some_other_ranker_v2"))
    response = client.get("/v1/metrics/offline")
    assert response.status_code == 409
    assert "ranker_model_version" in response.json()["message"]


def test_get_offline_metrics_does_not_call_generate_recommendations(monkeypatch, service, client):
    """The core regression this fix closes: the endpoint must be a pure
    read of the persisted report, never a live evaluation pass over the
    (potentially hundreds-of-users) evaluation population.
    """
    import recommendation.serving.pipeline as pipeline_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("GET /v1/metrics/offline must not call generate_recommendations")

    monkeypatch.setattr(pipeline_module, "generate_recommendations", _fail_if_called)
    save_offline_report(_offline_report_path(service), _valid_offline_report())

    response = client.get("/v1/metrics/offline")
    assert response.status_code == 200


def test_get_offline_metrics_does_not_call_service_recommend(monkeypatch, service, client):
    calls = []
    monkeypatch.setattr(service, "recommend", lambda *a, **k: calls.append((a, k)))
    save_offline_report(_offline_report_path(service), _valid_offline_report())

    response = client.get("/v1/metrics/offline")
    assert response.status_code == 200
    assert calls == []


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


# --- async-blocking fix: get_recommendations is now a plain sync def --------
# (docs/production-readiness.md's "must address before real production
# deployment" item: the route used to be `async def` while calling fully
# synchronous model inference, blocking the event loop for the duration of
# every request.)

def test_get_recommendations_route_is_a_plain_sync_function():
    """A plain `def` route is run by FastAPI in Starlette's external
    threadpool automatically - confirms the fix was actually applied,
    not just that requests still succeed (which they would either way).
    """
    assert inspect.iscoroutinefunction(routes_module.get_recommendations) is False


def test_concurrent_recommendation_requests_all_succeed(client):
    """Regression/no-crash check for the sync-def change: several
    concurrent requests (including a mix of valid/known and unknown users)
    must all complete correctly with no deadlock, no shared-state
    corruption, and no change in response shape.
    """
    user_ids = [1, 2, 3, UNKNOWN_USER_ID] * 5

    def _call(user_id: int):
        return client.get(f"/v1/users/{user_id}/recommendations")

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(_call, user_ids))

    for user_id, response in zip(user_ids, responses):
        if user_id == UNKNOWN_USER_ID:
            assert response.status_code == 404
        else:
            assert response.status_code == 200
            assert response.json()["meta"]["user_id"] == user_id
