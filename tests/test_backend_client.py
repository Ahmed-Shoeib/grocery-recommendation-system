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
from tests._backend_fakes import FakeResponse, FakeSession


def _client(responses, *, token_provider=None, **cfg):
    config = BackendApiConfig(base_url="https://backend.test", max_retries=cfg.pop("max_retries", 1), **cfg)
    session = FakeSession(responses)
    return BackendApiClient(config, session=session, token_provider=token_provider), session


def _envelope(rows, *, has_next=False, next_cursor=None):
    return {"success": True, "data": {"data": rows, "pagination": {"hasNext": has_next, "nextCursor": next_cursor}}}


# --- /api/ai/products - authoritative product source since 2026-09-15 --
#
# Bearer-gated, a flat array like /api/reviews (no pagination), carries
# productId on every row.


def test_list_products_parses_flat_ai_array_and_sends_bearer():
    client, session = _client(
        [FakeResponse(json_body={"success": True, "data": [
            {"productId": 82, "slug": "a", "name": "A", "price": 1.0, "stockQuantity": 5},
            {"productId": 83, "slug": "b", "name": "B", "price": 2.0, "stockQuantity": 0},
        ]})],
        token_provider=_StubProvider("tok"),
    )
    products = client.list_products()
    assert [(p.slug, p.product_id) for p in products] == [("a", 82), ("b", 83)]
    assert session.calls[0]["headers"]["Authorization"] == "Bearer tok"
    assert session.calls[0]["params"] == {}, "GET /api/ai/products takes no query parameters"
    assert len(session.calls) == 1, "not paginated - exactly one request"


def test_list_products_null_data_is_empty_not_an_error():
    client, _ = _client(
        [FakeResponse(json_body={"success": True, "data": None})], token_provider=_StubProvider(),
    )
    assert client.list_products() == []


def test_list_products_rejects_a_non_array_data_payload():
    client, _ = _client(
        [FakeResponse(json_body=_envelope([]))],  # cursor-paginated shape, not a flat array
        token_provider=_StubProvider(),
    )
    with pytest.raises(BackendContractError):
        client.list_products()


def test_list_products_requires_credentials_no_fallback_to_legacy_endpoint():
    """No silent degrade to the legacy slug-only /api/products - a mixed
    identity scheme across loads is worse than a loud failure (client.py
    module docstring). No request is sent at all.
    """
    client, session = _client([], token_provider=_NoCredentialsProvider())
    with pytest.raises(BackendCredentialsError):
        client.list_products()
    assert session.calls == []


# --- /api/ai/users - authoritative GUID <-> backend User.Id identity
# mapping (2026-09-18 user-identity migration, `ai-user-identity-mapping`).
# Protected, service-to-service, requires `users:read`. Modeled as a flat
# array like /api/ai/products - see list_ai_user_identities()'s docstring
# for why (the live endpoint was not independently probed for this
# change; a shape mismatch must fail loudly, which the rejection test
# below proves).


def test_list_ai_user_identities_parses_flat_array_and_sends_bearer():
    client, session = _client(
        [FakeResponse(json_body={"success": True, "data": [
            {"userId": 1547, "userGuid": "81bfc1f1-36eb-4427-b680-119ec489e156"},
            {"userId": 82, "userGuid": "05d74037-20a6-4399-82dd-66488575b5a8"},
        ]})],
        token_provider=_StubProvider("tok"),
    )
    identities = client.list_ai_user_identities()
    assert [(i.user_id, i.user_guid) for i in identities] == [
        (1547, "81bfc1f1-36eb-4427-b680-119ec489e156"),
        (82, "05d74037-20a6-4399-82dd-66488575b5a8"),
    ]
    assert session.calls[0]["headers"]["Authorization"] == "Bearer tok"
    assert len(session.calls) == 1, "not paginated - exactly one request"


def test_list_ai_user_identities_null_data_is_empty_not_an_error():
    client, _ = _client(
        [FakeResponse(json_body={"success": True, "data": None})], token_provider=_StubProvider(),
    )
    assert client.list_ai_user_identities() == []


def test_list_ai_user_identities_rejects_a_non_flat_array_payload():
    """If the backend actually ships a paginated envelope instead of a flat
    array, this must fail loudly (BackendContractError), never silently
    return an empty/wrong identity set - see the method's docstring on why
    this shape was assumed rather than live-verified.
    """
    client, _ = _client(
        [FakeResponse(json_body=_envelope([]))],  # cursor-paginated shape, not a flat array
        token_provider=_StubProvider(),
    )
    with pytest.raises(BackendContractError):
        client.list_ai_user_identities()


# --- /api/ai/user-activities - authoritative activity source since
# 2026-09-15 - Bearer-gated, cursor-paginated, carries productId per row.


def test_list_activities_uses_ai_endpoint_cursor_params_and_bearer():
    client, session = _client([FakeResponse(json_body=_envelope([]))], page_size=200, token_provider=_StubProvider("tok"))
    client.list_activities()
    assert session.calls[0]["params"]["pageSize"] == 200  # not a catalog endpoint, not capped
    assert session.calls[0]["headers"]["Authorization"] == "Bearer tok"


def test_list_activities_requires_credentials_no_fallback_to_legacy_endpoint():
    client, session = _client([], token_provider=_NoCredentialsProvider())
    with pytest.raises(BackendCredentialsError):
        client.list_activities()
    assert session.calls == []


def test_has_next_without_cursor_raises_pagination_error():
    client, _ = _client([FakeResponse(json_body=_envelope([{"slug": "a", "name": "A", "price": 1.0}],
                                                          has_next=True, next_cursor=None))])
    with pytest.raises(BackendPaginationError):
        client.list_categories()


def test_categories_page_size_is_capped_at_100():
    client, session = _client([FakeResponse(json_body=_envelope([]))], page_size=500)
    client.list_categories()
    assert session.calls[0]["params"]["Limit"] == 100


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


def test_transient_500_is_retried_then_succeeds(monkeypatch):
    """500 was added to the retryable set 2026-09-15 after directly
    observing /api/products, /api/categories and /api/ai/products all
    return a transient-looking HTTP 500 for ~30 minutes before recovering
    on their own (docs/data-mapping.md 19.5/19.8) - this is now a real,
    bounded-retry transient-failure class, not just 429/502/503/504.
    """
    monkeypatch.setattr("recommendation.backend.client.time.sleep", lambda *_: None)
    client, session = _client([
        FakeResponse(status_code=500, text="try later"),
        FakeResponse(json_body=_envelope([])),
    ], max_retries=2)
    assert client.list_categories() == []
    assert len(session.calls) == 2


def test_persistent_500_still_raises_never_hidden(monkeypatch):
    """A 500 that outlasts the retry budget must still surface loudly as
    BackendResponseError, exactly like any other non-2xx status - never
    silently swallowed into an empty/degraded result.
    """
    monkeypatch.setattr("recommendation.backend.client.time.sleep", lambda *_: None)
    client, session = _client(
        [FakeResponse(status_code=500, text="kaboom")] * 3, max_retries=2,
    )
    with pytest.raises(BackendResponseError):
        client.list_categories()
    assert len(session.calls) == 3  # 1 initial + 2 retries, then gives up


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


def test_5xx_with_zero_retry_budget_raises_immediately():
    client, _ = _client([FakeResponse(status_code=500, text="kaboom")], max_retries=0)
    with pytest.raises(BackendResponseError):
        client.list_categories()


def test_non_retryable_4xx_never_retried_even_with_budget(monkeypatch):
    """400 is not in the retryable set at all - one attempt, immediate
    raise, regardless of max_retries.
    """
    monkeypatch.setattr("recommendation.backend.client.time.sleep", lambda *_: None)
    client, session = _client([FakeResponse(status_code=400, text="bad request")], max_retries=2)
    with pytest.raises(BackendResponseError):
        client.list_categories()
    assert len(session.calls) == 1


def test_tls_verify_flag_is_passed_through():
    client, session = _client([FakeResponse(json_body=_envelope([]))], tls_verify=False)
    client.list_categories()
    assert session.calls[0]["verify"] is False
