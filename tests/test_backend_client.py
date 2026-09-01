"""BackendApiClient: envelope handling, pagination, error classification,
retries, TLS flag - all against a fake session (never the live backend).
"""

import pytest
import requests

from recommendation.data.backend.client import BackendApiClient
from recommendation.data.backend.errors import (
    BackendContractError,
    BackendPaginationError,
    BackendResponseError,
    BackendUnavailableError,
)
from recommendation.utils.config import BackendApiConfig


class FakeResponse:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body
        self.text = text or ""

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeSession:
    """Queues responses (or exceptions) and records the requests made."""

    def __init__(self, responses):
        self.headers = {}
        self._responses = list(responses)
        self.calls = []

    def request(self, method, url, params=None, timeout=None, verify=None):
        self.calls.append({"method": method, "url": url, "params": params or {}, "verify": verify})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(responses, **cfg):
    config = BackendApiConfig(base_url="https://backend.test", max_retries=cfg.pop("max_retries", 1), **cfg)
    session = FakeSession(responses)
    return BackendApiClient(config, session=session), session


def _envelope(rows, *, has_next=False, next_cursor=None):
    return {"success": True, "data": {"data": rows, "pagination": {"hasNext": has_next, "nextCursor": next_cursor}}}


def test_list_products_unwraps_envelope_and_pagination():
    client, session = _client([
        FakeResponse(json_body=_envelope([{"slug": "a", "name": "A", "price": 1.0, "stockQuantity": 5}],
                                         has_next=True, next_cursor="C2")),
        FakeResponse(json_body=_envelope([{"slug": "b", "name": "B", "price": 2.0, "stockQuantity": 0}])),
    ])
    products = client.list_products()
    assert [p.slug for p in products] == ["a", "b"]
    # second call carried the cursor from page 1
    assert session.calls[1]["params"]["Cursor"] == "C2"


def test_catalog_page_size_is_capped_at_100():
    client, session = _client([FakeResponse(json_body=_envelope([]))], page_size=500)
    client.list_products()
    assert session.calls[0]["params"]["Limit"] == 100


def test_activities_use_lowercase_cursor_and_pagesize_params():
    client, session = _client([FakeResponse(json_body=_envelope([]))], page_size=200)
    client.list_activities()
    assert "pageSize" in session.calls[0]["params"]
    assert session.calls[0]["params"]["pageSize"] == 200  # not a catalog endpoint, not capped


def test_has_next_without_cursor_raises_pagination_error():
    client, _ = _client([FakeResponse(json_body=_envelope([{"slug": "a", "name": "A", "price": 1.0}],
                                                          has_next=True, next_cursor=None))])
    with pytest.raises(BackendPaginationError):
        client.list_products()


def test_success_false_envelope_is_a_contract_error():
    client, _ = _client([FakeResponse(json_body={"success": False, "message": "boom", "data": None})])
    with pytest.raises(BackendContractError):
        client.list_categories()


def test_non_json_body_is_a_contract_error():
    client, _ = _client([FakeResponse(status_code=200, json_body=None, text="<html>502</html>")])
    with pytest.raises(BackendContractError):
        client.list_categories()


def test_connection_error_becomes_backend_unavailable():
    client, _ = _client([requests.exceptions.ConnectionError("refused"),
                         requests.exceptions.ConnectionError("refused")], max_retries=1)
    with pytest.raises(BackendUnavailableError):
        client.list_categories()


def test_retryable_status_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr("recommendation.data.backend.client.time.sleep", lambda *_: None)
    client, session = _client([
        FakeResponse(status_code=503, text="try later"),
        FakeResponse(json_body=_envelope([])),
    ], max_retries=2)
    assert client.list_categories() == []
    assert len(session.calls) == 2


def test_get_product_404_returns_none():
    client, _ = _client([FakeResponse(status_code=404, text="not found")])
    assert client.get_product("ghost") is None


def test_get_user_401_degrades_to_none_not_raise(monkeypatch):
    client, _ = _client([FakeResponse(status_code=401, text="")])
    assert client.get_user("some-guid") is None


def test_non_retryable_5xx_raises_response_error():
    client, _ = _client([FakeResponse(status_code=500, text="kaboom")], max_retries=0)
    with pytest.raises(BackendResponseError):
        client.list_categories()


def test_tls_verify_flag_is_passed_through():
    client, session = _client([FakeResponse(json_body=_envelope([]))], tls_verify=False)
    client.list_categories()
    assert session.calls[0]["verify"] is False
