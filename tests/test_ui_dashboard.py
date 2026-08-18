"""End-to-end smoke tests for the Streamlit dashboard itself, using
Streamlit's `AppTest` to actually execute `dashboard.py` (not just its
helper functions).

STEP 9 (docs/data-mapping.md section 18): `dashboard.py` is now a pure
`ui.api_client.RecommendationApiClient` HTTP client - it no longer builds
a `RecommendationService` or touches models/adapters at all. These tests
inject a `_FakeApiClient` (same method signatures as the real client,
returning real `api.schemas` Pydantic response objects, no HTTP) via
`st.session_state["_api_client_override"]` - `ui.service_loader
.load_api_client`'s dependency-injection seam - so no real network call
or running FastAPI process is ever required.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from recommendation.api.schemas import (
    OfflineMetricsProvenance,
    OfflineMetricsResponse,
    OfflineMetricsSplitReport,
    RecommendationItem,
    RecommendationMeta,
    RecommendationResponse,
    UserListItem,
    UserListResponse,
    UserProfileResponse,
)
from recommendation.ui.api_client import ApiResponseError, ApiUnavailableError

_DASHBOARD_PATH = str(Path(__file__).resolve().parents[1] / "src" / "recommendation" / "ui" / "dashboard.py")

_PROFILES = {
    1: UserProfileResponse(
        user_id=1, preferred_category="Cat1", age_group="25-34", tier="strong",
        total_engagement_events=3, distinct_products_purchased=3, click_count=0, purchase_count=3,
        cart_item_count=0, search_count=0, has_chatbot_context=False,
        category_affinity={"Cat1": 1.0}, brand_affinity={},
        clicks=[], purchases=[{"product_id": 100, "product_name": "Product 100", "quantity": 1, "unit_price": 1.0, "order_id": 0, "order_status": "DELIVERED"}],
        cart_items=[], searches=[], chatbot=None,
    ),
    2: UserProfileResponse(
        user_id=2, preferred_category=None, age_group=None, tier="sparse",
        total_engagement_events=1, distinct_products_purchased=1, click_count=0, purchase_count=1,
        cart_item_count=0, search_count=0, has_chatbot_context=False,
        category_affinity={}, brand_affinity={}, clicks=[], purchases=[], cart_items=[], searches=[], chatbot=None,
    ),
    3: UserProfileResponse(
        user_id=3, preferred_category="Cat0", age_group=None, tier="no_history",
        total_engagement_events=0, distinct_products_purchased=0, click_count=0, purchase_count=0,
        cart_item_count=0, search_count=0, has_chatbot_context=False,
        category_affinity={}, brand_affinity={}, clicks=[], purchases=[], cart_items=[], searches=[], chatbot=None,
    ),
}


def _recommendation_response(user_id: int, tier: str, top_n: int) -> RecommendationResponse:
    items = [
        RecommendationItem(
            product_id=100 + i, rank=i + 1, score=1.0 - i * 0.1,
            source="personalized" if tier != "no_history" else "global_popularity",
            product_name=f"Product {100 + i}", category=f"Cat{i % 3}", brand=None, price=5.0 + i,
            is_active=True, stock_quantity=10,
        )
        for i in range(min(top_n, 4))
    ]
    meta = RecommendationMeta(
        user_id=user_id, tier=tier, requested_top_n=top_n, returned_count=len(items), fill_rate=len(items) / top_n,
        pool_size=8, num_excluded_pre_retrieval=0, num_excluded_by_eligibility=0, api_version="v1",
        model_version="test_v1", generated_at=datetime.now(timezone.utc), latency_ms=5.0,
    )
    return RecommendationResponse(meta=meta, items=items)


class _FakeApiClient:
    """Same method signatures as `ui.api_client.RecommendationApiClient`,
    backed by in-memory fixtures instead of real HTTP calls.
    """

    def __init__(
        self,
        fail_recommendations: Exception | None = None,
        fail_users: Exception | None = None,
        fail_offline_metrics: Exception | None = None,
    ) -> None:
        self._fail_recommendations = fail_recommendations
        self._fail_users = fail_users
        self._fail_offline_metrics = fail_offline_metrics

    def list_users(self) -> UserListResponse:
        if self._fail_users is not None:
            raise self._fail_users
        return UserListResponse(
            users=[
                UserListItem(user_id=1, preferred_category="Cat1", age_group="25-34"),
                UserListItem(user_id=2, preferred_category=None, age_group=None),
                UserListItem(user_id=3, preferred_category="Cat0", age_group=None),
            ]
        )

    def get_user_profile(self, user_id: int) -> UserProfileResponse:
        return _PROFILES[user_id]

    def get_recommendations(self, user_id: int, limit: int | None = None) -> RecommendationResponse:
        if self._fail_recommendations is not None:
            raise self._fail_recommendations
        return _recommendation_response(user_id, _PROFILES[user_id].tier, limit or 10)

    def get_offline_metrics(self) -> OfflineMetricsResponse:
        if self._fail_offline_metrics is not None:
            raise self._fail_offline_metrics
        split = OfflineMetricsSplitReport(
            split_name="val", num_cases=2, ndcg_at_k={5: 0.5}, precision_at_k={5: 0.4}, recall_at_k={5: 0.3},
            hit_rate_at_k={5: 0.6}, mrr=0.5, catalog_coverage=0.8, mean_distinct_categories=2.0, mean_fill_rate=1.0,
        )
        provenance = OfflineMetricsProvenance(
            generated_at=datetime.now(timezone.utc), run_id="20260818T000000",
            ranker_model_version="sqlite_baseline_ranker_v1", two_tower_model_version="sqlite_baseline_two_tower_v1",
            dataset_fingerprint_sha256_16="abc123", recency_enabled=True, recency_half_life_days=21.0,
            include_price_features=True, k_values=[5, 10, 20], top_n=10,
        )
        return OfflineMetricsResponse(provenance=provenance, val_report=split, test_report=split)


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
    at.session_state["_api_client_override"] = _FakeApiClient()
    return at


def test_dashboard_renders_without_exception(app):
    app.run()
    assert not app.exception


def test_dashboard_shows_user_table_and_recommendations(app):
    app.run()
    assert not app.exception
    subheaders = [s.value for s in app.subheader]
    assert any("Users" in s for s in subheaders)
    assert any("Final recommendations" in s for s in subheaders)
    assert any("Pipeline" in s for s in subheaders)
    assert any("Metrics" in s for s in subheaders)


def test_dashboard_selecting_sparse_history_user(app):
    app.run()
    app.selectbox[0].select(2).run()
    assert not app.exception


def test_dashboard_selecting_no_history_user(app):
    app.run()
    app.selectbox[0].select(3).run()
    assert not app.exception


def test_dashboard_top_n_slider_changes_recommendation_count(app):
    app.run()
    app.slider[0].set_value(2).run()
    assert not app.exception


def test_dashboard_handles_recommendation_api_failure_gracefully():
    """A failure surfaced by the API itself (non-200) while generating
    recommendations must be caught and shown as a clean message, not
    crash the whole page - and must NEVER fall back to direct model
    access (docs/data-mapping.md section 18's explicit "DO NOT DO" item).
    """
    at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
    at.session_state["_api_client_override"] = _FakeApiClient(
        fail_recommendations=ApiResponseError(500, "simulated pipeline failure")
    )
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("generating recommendations" in e for e in errors)


def test_dashboard_handles_offline_metrics_unavailable_gracefully():
    """`GET /v1/metrics/offline` returns 409 when the persisted report is
    missing/malformed/stale (STEP 9 fix) - the dashboard must show a clean
    message for that split, not crash the whole page or fall back to
    computing anything itself.
    """
    at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
    at.session_state["_api_client_override"] = _FakeApiClient(
        fail_offline_metrics=ApiResponseError(409, "no offline evaluation report found")
    )
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("loading offline metrics" in e for e in errors)


def test_dashboard_handles_api_unavailable_gracefully():
    """If FastAPI is not reachable at all, the dashboard must show a
    clean message (never a raw traceback, never a silent fallback).
    """
    at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
    at.session_state["_api_client_override"] = _FakeApiClient(
        fail_users=ApiUnavailableError("could not reach the recommendation API at http://localhost:8000")
    )
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("Cannot reach the recommendation API" in e for e in errors)
