"""BackendApiClient: envelope handling, pagination, error classification,
retries, TLS flag - all against a fake session (never the live backend).
"""

import pytest
import requests

from recommendation.backend.client import BackendApiClient
from recommendation.backend.errors import (
    BackendAuthError,
    BackendContractError,
    BackendCredentialsError,
    BackendPaginationError,
    BackendResponseError,
    BackendUnavailableError,
)
from recommendation.config import BackendApiConfig


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

    def request(self, method, url, params=None, timeout=None, verify=None, headers=None, json=None):
        self.calls.append({
            "method": method, "url": url, "params": params or {}, "verify": verify,
            "headers": headers or {}, "json": json,
        })
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _client(responses, *, token_provider=None, **cfg):
    config = BackendApiConfig(base_url="https://backend.test", max_retries=cfg.pop("max_retries", 1), **cfg)
    session = FakeSession(responses)
    return BackendApiClient(config, session=session, token_provider=token_provider), session


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
    monkeypatch.setattr("recommendation.backend.client.time.sleep", lambda *_: None)
    client, session = _client([
        FakeResponse(status_code=503, text="try later"),
        FakeResponse(json_body=_envelope([])),
    ], max_retries=2)
    assert client.list_categories() == []
    assert len(session.calls) == 2


def test_get_product_404_returns_none():
    client, _ = _client([FakeResponse(status_code=404, text="not found")])
    assert client.get_product("ghost") is None


def test_get_user_without_credentials_degrades_without_calling_backend():
    """Unconfigured service auth must not turn every profile fetch into a
    guaranteed-failing round trip - it degrades to a bare profile silently.
    """
    client, session = _client([], token_provider=_NoCredentialsProvider())
    assert client.get_user("some-guid") is None
    assert session.calls == []


def test_get_user_401_degrades_to_none_not_raise():
    # 401 twice: the first triggers the refresh-and-retry, the second is a
    # real auth failure - which get_user turns into a bare profile.
    client, _ = _client(
        [FakeResponse(status_code=401, text=""), FakeResponse(status_code=401, text="")],
        token_provider=_StubProvider("tok"),
    )
    assert client.get_user("some-guid") is None


class _StubProvider:
    """Minimal `ServiceTokenProvider` stand-in - no network, no env."""

    def __init__(self, token="tok"):
        self._token = token
        self.invalidations = 0

    def token(self):
        return self._token

    def invalidate(self):
        self.invalidations += 1

    def has_credentials(self):
        return True


class _NoCredentialsProvider(_StubProvider):
    def token(self):
        raise BackendCredentialsError("not configured")

    def has_credentials(self):
        return False


# --- /api/reviews ------------------------------------------------------
#
# `/api/reviews` is the one list endpoint that is NOT cursor-paginated:
# Swagger declares no query parameters and a flat `data` array
# (`AiProductReviewResponseListApiResponse`).


def _review_row(**overrides):
    row = {
        "reviewId": 1, "userId": 7, "productId": 3, "rating": 5,
        "comment": "tasty", "createdAt": "2026-09-01T10:00:00", "updatedAt": None,
    }
    row.update(overrides)
    return row


def test_list_reviews_parses_flat_array_and_sends_bearer():
    client, session = _client(
        [FakeResponse(json_body={"success": True, "statusCode": 200, "message": None,
                                 "data": [_review_row(), _review_row(reviewId=2, rating=3)]})],
        token_provider=_StubProvider("tok"),
    )
    reviews = client.list_reviews()
    assert [r.review_id for r in reviews] == [1, 2]
    assert (reviews[0].user_id, reviews[0].product_id, reviews[0].rating) == (7, 3, 5)
    assert session.calls[0]["headers"]["Authorization"] == "Bearer tok"
    assert session.calls[0]["params"] == {}, "endpoint takes no query parameters"
    assert len(session.calls) == 1, "not paginated - exactly one request"


def test_list_reviews_null_data_is_empty_not_an_error():
    client, _ = _client(
        [FakeResponse(json_body={"success": True, "statusCode": 200, "data": None})],
        token_provider=_StubProvider(),
    )
    assert client.list_reviews() == []


def test_list_reviews_rejects_a_non_array_data_payload():
    client, _ = _client(
        [FakeResponse(json_body={"success": True, "data": {"data": [], "pagination": {}}})],
        token_provider=_StubProvider(),
    )
    with pytest.raises(BackendContractError):
        client.list_reviews()


def test_list_reviews_401_refreshes_token_and_retries_once():
    provider = _StubProvider()
    client, session = _client(
        [FakeResponse(status_code=401, text=""),
         FakeResponse(json_body={"success": True, "data": [_review_row()]})],
        token_provider=provider,
    )
    assert len(client.list_reviews()) == 1
    assert provider.invalidations == 1
    assert len(session.calls) == 2


def test_list_reviews_raises_on_persistent_auth_failure():
    """Unlike `get_user`, reviews are fetched as a set - a persistent 401
    is surfaced to the loader, which decides to degrade.
    """
    client, _ = _client(
        [FakeResponse(status_code=401, text=""), FakeResponse(status_code=401, text="")],
        token_provider=_StubProvider(),
    )
    with pytest.raises(BackendAuthError):
        client.list_reviews()


def test_list_reviews_backend_reported_failure_raises_contract_error():
    client, _ = _client(
        [FakeResponse(json_body={"success": False, "message": "nope", "data": None})],
        token_provider=_StubProvider(),
    )
    with pytest.raises(BackendContractError):
        client.list_reviews()


def test_non_retryable_5xx_raises_response_error():
    client, _ = _client([FakeResponse(status_code=500, text="kaboom")], max_retries=0)
    with pytest.raises(BackendResponseError):
        client.list_categories()


def test_tls_verify_flag_is_passed_through():
    client, session = _client([FakeResponse(json_body=_envelope([]))], tls_verify=False)
    client.list_categories()
    assert session.calls[0]["verify"] is False
