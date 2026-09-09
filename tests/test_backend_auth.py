"""`ServiceTokenProvider`: the client-credentials exchange, in-memory
caching, refresh-before-expiry, and - the point of most of this file - the
guarantee that no credential or token is ever logged, persisted, or
attached to a public request.

Fully deterministic: a fake session and an injected clock, never the live
backend.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest
import requests

from recommendation.data.backend.auth import ENV_CLIENT_ID, ENV_CLIENT_SECRET, ServiceTokenProvider
from recommendation.data.backend.client import BackendApiClient
from recommendation.data.backend.errors import (
    BackendAuthError,
    BackendContractError,
    BackendCredentialsError,
    BackendUnavailableError,
)
from recommendation.utils.config import BackendApiConfig

CLIENT_ID = "recs-service"
CLIENT_SECRET = "s3cr3t-not-a-real-credential"
TOKEN = "header.payload.signature"

_T0 = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)


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


class Clock:
    def __init__(self, now=_T0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


def _token_body(expires_at: datetime | None = None, *, token: str = TOKEN, wrapped: bool = True):
    data = {"accessToken": token}
    if expires_at is not None:
        # The backend sends a naive UTC instant, as everywhere else.
        data["expiresAtUtc"] = expires_at.replace(tzinfo=None).isoformat()
    return {"success": True, "data": data} if wrapped else data


def _provider(responses, *, env=None, clock=None, **cfg):
    config = BackendApiConfig(base_url="https://backend.test", **cfg)
    session = FakeSession(responses)
    provider = ServiceTokenProvider(
        config,
        session=session,
        env={ENV_CLIENT_ID: CLIENT_ID, ENV_CLIENT_SECRET: CLIENT_SECRET} if env is None else env,
        now=clock or Clock(),
    )
    return provider, session


# --- acquisition ------------------------------------------------------


def test_token_is_acquired_with_client_credentials():
    clock = Clock()
    provider, session = _provider([FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15)))], clock=clock)

    assert provider.token() == TOKEN
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://backend.test/api/auth/service/token"
    assert call["json"] == {"clientId": CLIENT_ID, "clientSecret": CLIENT_SECRET}


def test_token_response_without_envelope_is_accepted():
    """Swagger publishes no response schema for this route, so an
    un-enveloped body must not break the exchange.
    """
    provider, _ = _provider(
        [FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15), wrapped=False))]
    )
    assert provider.token() == TOKEN


# --- caching / refresh ------------------------------------------------


def test_token_is_cached_and_reused_within_its_lifetime():
    clock = Clock()
    provider, session = _provider([FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15)))], clock=clock)

    assert provider.token() == TOKEN
    clock.advance(minutes=10)  # still well inside the ~15 minute lifetime
    assert provider.token() == TOKEN
    assert len(session.calls) == 1, "a cached, unexpired token must not be re-fetched"


def test_token_is_refreshed_before_expiry():
    clock = Clock()
    provider, session = _provider(
        [
            FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15))),
            FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=45), token="second.token")),
        ],
        clock=clock,
    )
    assert provider.token() == TOKEN
    # 30s before the stated expiry: inside the 60s refresh skew, so the
    # provider must refresh rather than hand out a token that could expire
    # in flight.
    clock.advance(minutes=14, seconds=30)
    assert provider.token() == "second.token"
    assert len(session.calls) == 2


def test_expired_token_is_refreshed():
    clock = Clock()
    provider, session = _provider(
        [
            FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15))),
            FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=60), token="fresh")),
        ],
        clock=clock,
    )
    provider.token()
    clock.advance(minutes=20)
    assert provider.token() == "fresh"


def test_invalidate_forces_a_new_exchange():
    provider, session = _provider([
        FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15))),
        FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15), token="reissued")),
    ])
    assert provider.token() == TOKEN
    provider.invalidate()
    assert provider.token() == "reissued"
    assert len(session.calls) == 2


def test_missing_expiry_falls_back_to_a_conservative_ttl():
    clock = Clock()
    provider, session = _provider([
        FakeResponse(json_body=_token_body(None)),
        FakeResponse(json_body=_token_body(None, token="second")),
    ], clock=clock)
    assert provider.token() == TOKEN
    clock.advance(minutes=1)
    assert provider.token() == TOKEN, "fallback TTL must still cache"
    clock.advance(minutes=5)
    assert provider.token() == "second", "fallback TTL must expire, not last forever"


def test_concurrent_callers_perform_one_exchange():
    """The API refreshes data on a background thread while request threads
    may also hit a protected endpoint; an expiry boundary must not produce
    one token exchange per thread.
    """
    provider, session = _provider([FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15)))])
    start = threading.Barrier(8)
    results: list[str] = []
    lock = threading.Lock()

    def worker():
        start.wait()
        value = provider.token()
        with lock:
            results.append(value)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [TOKEN] * 8
    assert len(session.calls) == 1


# --- failure modes ----------------------------------------------------


def test_missing_credentials_raises_without_any_network_call():
    provider, session = _provider([], env={})
    with pytest.raises(BackendCredentialsError):
        provider.token()
    assert session.calls == [], "an unconfigured provider must not contact the backend"
    assert provider.has_credentials() is False


def test_blank_credentials_are_treated_as_missing():
    provider, _ = _provider([], env={ENV_CLIENT_ID: "  ", ENV_CLIENT_SECRET: CLIENT_SECRET})
    assert provider.has_credentials() is False
    with pytest.raises(BackendCredentialsError):
        provider.token()


def test_rejected_credentials_raise_auth_error():
    provider, _ = _provider([FakeResponse(status_code=401, text="invalid client")])
    with pytest.raises(BackendAuthError):
        provider.token()


def test_token_endpoint_5xx_raises_contract_error():
    provider, _ = _provider([FakeResponse(status_code=500, text="boom")])
    with pytest.raises(BackendContractError):
        provider.token()


def test_token_endpoint_transport_failure_raises_unavailable():
    provider, _ = _provider([requests.exceptions.ConnectTimeout("timeout")])
    with pytest.raises(BackendUnavailableError):
        provider.token()


def test_response_without_access_token_raises_contract_error():
    provider, _ = _provider([FakeResponse(json_body={"success": True, "data": {"expiresAtUtc": "2026-09-09T12:15:00"}})])
    with pytest.raises(BackendContractError):
        provider.token()


def test_already_expired_token_is_rejected():
    provider, _ = _provider([FakeResponse(json_body=_token_body(_T0 - timedelta(minutes=1)))])
    with pytest.raises(BackendContractError):
        provider.token()


# --- leakage ----------------------------------------------------------


def test_no_secret_or_token_appears_in_logs_or_error_messages(caplog):
    """Every observable surface of a failed and a successful exchange is
    checked - a credential must never reach a log file or a traceback.
    """
    caplog.set_level("DEBUG")
    provider, _ = _provider([
        FakeResponse(status_code=401, text=f"rejected clientId={CLIENT_ID} secret={CLIENT_SECRET}"),
    ])
    with pytest.raises(BackendAuthError) as excinfo:
        provider.token()
    assert CLIENT_SECRET not in str(excinfo.value)

    provider2, _ = _provider([FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15)))])
    assert provider2.token() == TOKEN

    logged = caplog.text
    assert CLIENT_SECRET not in logged
    assert TOKEN not in logged
    assert CLIENT_ID not in logged


def test_transport_failure_message_omits_the_request_body():
    """`requests` exceptions can carry the request (and therefore the
    secret); only the exception type may be reported.
    """
    provider, _ = _provider([
        requests.exceptions.ConnectionError(f"failed sending {{'clientSecret': '{CLIENT_SECRET}'}}")
    ])
    with pytest.raises(BackendUnavailableError) as excinfo:
        provider.token()
    assert CLIENT_SECRET not in str(excinfo.value)


def test_token_is_never_added_to_session_headers():
    """A session-wide header would attach the token to public catalog calls
    too. It must be per-request only.
    """
    provider, session = _provider([FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15)))])
    provider.token()
    assert "Authorization" not in session.headers


def test_credentials_are_not_fields_on_the_config_model():
    """Config is loaded from committed YAML and is dumped in diagnostics;
    a credential field there would leak into both.
    """
    fields = set(BackendApiConfig.model_fields)
    assert not any("secret" in f or "client_id" in f or "token" in f for f in fields), fields


# --- client integration -----------------------------------------------


def _authed_client(responses, *, env=None, clock=None):
    config = BackendApiConfig(base_url="https://backend.test", max_retries=0)
    session = FakeSession(responses)
    provider = ServiceTokenProvider(
        config,
        session=session,
        env={ENV_CLIENT_ID: CLIENT_ID, ENV_CLIENT_SECRET: CLIENT_SECRET} if env is None else env,
        now=clock or Clock(),
    )
    return BackendApiClient(config, session=session, token_provider=provider), session


def test_public_endpoints_send_no_authorization_header():
    client, session = _authed_client([
        FakeResponse(json_body={"success": True, "data": {"data": [], "pagination": {"hasNext": False}}}),
    ])
    client.list_categories()
    assert "Authorization" not in (session.calls[0]["headers"] or {})
    assert all(c["method"] == "GET" for c in session.calls), "no token exchange for a public endpoint"


def test_protected_endpoint_sends_bearer_header():
    client, session = _authed_client([
        FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15))),
        FakeResponse(json_body={"success": True, "data": {"guid": "g-1"}}),
    ])
    user = client.get_user("g-1")
    assert user is not None and user.guid == "g-1"
    assert session.calls[1]["headers"]["Authorization"] == f"Bearer {TOKEN}"


def test_401_on_protected_request_refreshes_token_and_retries_once():
    client, session = _authed_client([
        FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15))),
        FakeResponse(status_code=401, text=""),                       # stale token
        FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15), token="reissued")),
        FakeResponse(json_body={"success": True, "data": {"guid": "g-1"}}),
    ])
    user = client.get_user("g-1")
    assert user is not None and user.guid == "g-1"
    assert session.calls[3]["headers"]["Authorization"] == "Bearer reissued"


def test_second_401_does_not_retry_again():
    """One refresh, then stop - a revoked client must not spin."""
    client, session = _authed_client([
        FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15))),
        FakeResponse(status_code=401, text=""),
        FakeResponse(json_body=_token_body(_T0 + timedelta(minutes=15), token="reissued")),
        FakeResponse(status_code=401, text=""),
    ])
    assert client.get_user("g-1") is None  # best-effort endpoint degrades
    assert len(session.calls) == 4


def test_has_service_credentials_reflects_the_environment():
    configured, _ = _authed_client([])
    assert configured.has_service_credentials() is True
    unconfigured, _ = _authed_client([], env={})
    assert unconfigured.has_service_credentials() is False
