"""Unit tests for `ui.api_client.RecommendationApiClient` - STEP 9
(docs/data-mapping.md section 18). These tests never make a real network
call: `requests.get` is monkeypatched with a fake that returns canned
responses or raises `requests.exceptions.*`, so the client's own
translation logic (connection error -> `ApiUnavailableError`, timeout ->
`ApiTimeoutError`, 404 -> `UnknownUserError`, other non-2xx ->
`ApiResponseError`, malformed body -> `MalformedResponseError`) is
exercised directly and deterministically. Confirms the client parses
success responses into the exact same `api.schemas` Pydantic models
FastAPI returns, and that it never imports or touches
`RecommendationService`/model/adapter code (see the module-level import
check at the bottom of this file).
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import requests

from recommendation.ui import api_client as api_client_module
from recommendation.ui.api_client import (
    ApiResponseError,
    ApiTimeoutError,
    ApiUnavailableError,
    MalformedResponseError,
    RecommendationApiClient,
    UnknownUserError,
)


class _FakeResponse:
    def __init__(self, status_code: int, json_body=None, text: str = "", raise_on_json: bool = False):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json:
            raise ValueError("not valid JSON")
        return self._json_body


def _install_fake_get(monkeypatch, fake_get) -> None:
    monkeypatch.setattr(api_client_module.requests, "get", fake_get)


@pytest.fixture
def client() -> RecommendationApiClient:
    return RecommendationApiClient("http://localhost:8000", timeout_seconds=5.0)


_RECOMMENDATION_PAYLOAD = {
    "meta": {
        "user_id": 1, "tier": "strong", "requested_top_n": 5, "returned_count": 1, "fill_rate": 0.2,
        "pool_size": 8, "num_excluded_pre_retrieval": 0, "num_excluded_by_eligibility": 0, "api_version": "v1",
        "model_version": "test_v1", "generated_at": "2026-08-18T00:00:00Z", "latency_ms": 4.2,
    },
    "items": [
        {"product_id": 100, "rank": 1, "score": 0.9, "source": "personalized", "product_name": "Product 100",
         "category": "Cat0", "brand": None, "price": 5.0, "is_active": True, "stock_quantity": 10},
    ],
}


# --- connection-level failures ------------------------------------------

def test_connection_error_raises_api_unavailable(monkeypatch, client):
    def _fake_get(url, params=None, timeout=None):
        raise requests.exceptions.ConnectionError("refused")

    _install_fake_get(monkeypatch, _fake_get)
    with pytest.raises(ApiUnavailableError):
        client.list_users()


def test_timeout_raises_api_timeout(monkeypatch, client):
    def _fake_get(url, params=None, timeout=None):
        raise requests.exceptions.Timeout("timed out")

    _install_fake_get(monkeypatch, _fake_get)
    with pytest.raises(ApiTimeoutError):
        client.list_users()


def test_other_request_exception_raises_generic_client_error(monkeypatch, client):
    def _fake_get(url, params=None, timeout=None):
        raise requests.exceptions.RequestException("some other transport failure")

    _install_fake_get(monkeypatch, _fake_get)
    with pytest.raises(api_client_module.ApiClientError):
        client.list_users()


# --- HTTP status handling -------------------------------------------------

def test_404_on_recommendations_raises_unknown_user(monkeypatch, client):
    _install_fake_get(monkeypatch, lambda url, params=None, timeout=None: _FakeResponse(404, {"error": "not_found", "message": "user 999 not found"}))
    with pytest.raises(UnknownUserError) as exc_info:
        client.get_recommendations(999)
    assert exc_info.value.user_id == 999


def test_404_on_profile_raises_unknown_user(monkeypatch, client):
    _install_fake_get(monkeypatch, lambda url, params=None, timeout=None: _FakeResponse(404, {"error": "not_found", "message": "user 999 not found"}))
    with pytest.raises(UnknownUserError):
        client.get_user_profile(999)


def test_500_raises_api_response_error_with_status_and_message(monkeypatch, client):
    _install_fake_get(monkeypatch, lambda url, params=None, timeout=None: _FakeResponse(500, {"error": "internal_error", "message": "boom"}))
    with pytest.raises(ApiResponseError) as exc_info:
        client.list_users()
    assert exc_info.value.status_code == 500
    assert exc_info.value.message == "boom"


def test_422_on_list_users_is_api_response_error_not_unknown_user(monkeypatch, client):
    _install_fake_get(monkeypatch, lambda url, params=None, timeout=None: _FakeResponse(422, {"error": "invalid_request", "message": "bad params"}))
    with pytest.raises(ApiResponseError):
        client.get_offline_metrics(top_n=5)


def test_error_body_without_json_falls_back_to_raw_text(monkeypatch, client):
    _install_fake_get(monkeypatch, lambda url, params=None, timeout=None: _FakeResponse(503, raise_on_json=True, text="Service Unavailable"))
    with pytest.raises(ApiResponseError) as exc_info:
        client.list_users()
    assert exc_info.value.message == "Service Unavailable"


# --- malformed 2xx bodies ---------------------------------------------------

def test_non_json_200_body_raises_malformed_response(monkeypatch, client):
    _install_fake_get(monkeypatch, lambda url, params=None, timeout=None: _FakeResponse(200, raise_on_json=True, text="<html>not json</html>"))
    with pytest.raises(MalformedResponseError):
        client.list_users()


def test_json_200_body_not_matching_schema_raises_malformed_response(monkeypatch, client):
    _install_fake_get(monkeypatch, lambda url, params=None, timeout=None: _FakeResponse(200, {"unexpected": "shape"}))
    with pytest.raises(MalformedResponseError):
        client.get_recommendations(1)


# --- success parsing into api.schemas models --------------------------------

def test_get_recommendations_parses_into_schema_model(monkeypatch, client):
    _install_fake_get(monkeypatch, lambda url, params=None, timeout=None: _FakeResponse(200, _RECOMMENDATION_PAYLOAD))
    result = client.get_recommendations(1, limit=5)
    assert result.meta.user_id == 1
    assert result.meta.tier == "strong"
    assert len(result.items) == 1
    assert result.items[0].product_name == "Product 100"


def test_list_users_parses_into_schema_model(monkeypatch, client):
    payload = {"users": [{"user_id": 1, "preferred_category": "Cat1", "age_group": "25-34"}]}
    _install_fake_get(monkeypatch, lambda url, params=None, timeout=None: _FakeResponse(200, payload))
    result = client.list_users()
    assert result.users[0].user_id == 1


def test_get_user_profile_parses_into_schema_model(monkeypatch, client):
    payload = {
        "user_id": 1, "preferred_category": "Cat1", "age_group": "25-34", "tier": "strong",
        "total_engagement_events": 3, "distinct_products_purchased": 3, "click_count": 0, "purchase_count": 3,
        "cart_item_count": 0, "search_count": 0, "has_chatbot_context": False,
        "category_affinity": {"Cat1": 1.0}, "brand_affinity": {}, "clicks": [], "purchases": [],
        "cart_items": [], "searches": [], "chatbot": None,
    }
    _install_fake_get(monkeypatch, lambda url, params=None, timeout=None: _FakeResponse(200, payload))
    result = client.get_user_profile(1)
    assert result.tier == "strong"
    assert result.category_affinity == {"Cat1": 1.0}


def test_get_offline_metrics_parses_into_schema_model(monkeypatch, client):
    split = {
        "split_name": "val", "num_cases": 2, "ndcg_at_k": {"5": 0.5}, "precision_at_k": {"5": 0.4},
        "recall_at_k": {"5": 0.3}, "hit_rate_at_k": {"5": 0.6}, "mrr": 0.5, "catalog_coverage": 0.8,
        "mean_distinct_categories": 2.0, "duplicate_rate": 0.0, "fill_rate": 1.0, "tier_counts": {"strong": 1},
    }
    payload = {"num_eval_users": 2, "val_report": split, "test_report": split}
    _install_fake_get(monkeypatch, lambda url, params=None, timeout=None: _FakeResponse(200, payload))
    result = client.get_offline_metrics(top_n=5)
    assert result.num_eval_users == 2
    assert result.val_report.ndcg_at_k == {5: 0.5}  # string key coerced to int by the schema's dict[int, float] annotation


# --- request shape -----------------------------------------------------------

def test_get_uses_configured_base_url_and_timeout(monkeypatch, client):
    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResponse(200, {"users": []})

    _install_fake_get(monkeypatch, _fake_get)
    client.list_users()
    assert captured["url"] == "http://localhost:8000/v1/users"
    assert captured["timeout"] == 5.0


def test_base_url_trailing_slash_is_normalized(monkeypatch):
    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url
        return _FakeResponse(200, {"users": []})

    _install_fake_get(monkeypatch, _fake_get)
    RecommendationApiClient("http://localhost:8000/", timeout_seconds=5.0).list_users()
    assert captured["url"] == "http://localhost:8000/v1/users"


# --- no model/adapter code reachable from this module -----------------------

def test_api_client_module_does_not_import_recommendation_service_or_model_code():
    """`ui.api_client` must be importable without ever pulling in
    `RecommendationService`, TensorFlow, or adapter/artifact loading code
    (docs/data-mapping.md section 18) - it only imports the pure-Pydantic
    `api.schemas` wire contract. Run in a fresh subprocess: within this
    test session other test modules have already imported
    `recommendation.api.dependencies` for unrelated reasons, so checking
    `sys.modules` in-process would give a false positive either way.
    """
    probe = (
        "import sys\n"
        "import recommendation.ui.api_client\n"
        "forbidden = {\n"
        "    'recommendation.api.dependencies',\n"
        "    'recommendation.serving.pipeline',\n"
        "    'recommendation.retrieval.two_tower.model',\n"
        "    'recommendation.ranking.model',\n"
        "    'tensorflow',\n"
        "}\n"
        "hit = forbidden & set(sys.modules)\n"
        "assert not hit, f'forbidden modules imported: {hit}'\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=None, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout
