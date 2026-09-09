"""Service-to-service authentication for the backend REST API.

The backend gates a few endpoints (`/api/users/{userId}`, `/api/reviews`)
behind a JWT `Bearer` token obtained with a client-credentials exchange:

    POST /api/auth/service/token   {clientId, clientSecret}
        -> {success, data: {accessToken, expiresAtUtc}}   (~15 minute lifetime)

`ServiceTokenProvider` is the only component that performs that exchange
and the only one that ever holds a token. `client.BackendApiClient` asks
it for a token per protected request and attaches the header; nothing
else in the codebase sees either the credentials or the token.

Secret handling - the rules this module exists to enforce:

- **Environment only.** `clientId`/`clientSecret` are read from
  `RECS_BACKEND_SERVICE_CLIENT_ID` / `RECS_BACKEND_SERVICE_CLIENT_SECRET`.
  They are deliberately NOT fields on `utils.config.BackendApiConfig`:
  that model is loaded from a committed YAML file and is dumped/logged in
  diagnostics, so a secret placed there would leak into both.
- **Memory only.** The token is cached on this instance and never written
  to disk, never added to `Session.headers` (which would attach it to
  public requests too), and never returned to a caller outside this
  package.
- **Never logged.** No log line here interpolates the secret, the token,
  or a token-endpoint response body. Only lengths/expiry instants appear.
- **No credentials, no requests.** With the env vars unset the provider
  raises `BackendCredentialsError` *without* calling the network, so an
  unconfigured deployment degrades exactly like the pre-auth behaviour
  (best-effort endpoints log once and continue) instead of hammering the
  token endpoint with empty credentials.

Concurrency: the FastAPI service refreshes data from a background thread
while request threads may also touch the client, so the cached token is
guarded by a `threading.Lock` and refreshed under double-checked locking -
concurrent callers around an expiry boundary perform exactly one token
exchange, not one each.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone

import requests

from recommendation.data.backend.dtos import ApiServiceToken
from recommendation.data.backend.errors import (
    BackendAuthError,
    BackendContractError,
    BackendCredentialsError,
    BackendUnavailableError,
)
from recommendation.utils.config import BackendApiConfig
from recommendation.utils.logging import get_logger

logger = get_logger(__name__)

TOKEN_PATH = "/api/auth/service/token"

ENV_CLIENT_ID = "RECS_BACKEND_SERVICE_CLIENT_ID"
ENV_CLIENT_SECRET = "RECS_BACKEND_SERVICE_CLIENT_SECRET"

# Refresh this long before `expiresAtUtc` so a token can never expire
# mid-flight between the header being set and the backend validating it.
# ~15 minute lifetime, so 60s is a small slice of it and still ample.
_REFRESH_SKEW = timedelta(seconds=60)

# Used only if the backend stops sending `expiresAtUtc`. Deliberately well
# under the observed ~15 minute lifetime: re-fetching early is cheap, using
# an expired token is not.
_FALLBACK_TTL = timedelta(minutes=5)


class ServiceTokenProvider:
    """Caches one service token and refreshes it just before expiry."""

    def __init__(
        self,
        config: BackendApiConfig,
        *,
        session: requests.Session | None = None,
        env: dict[str, str] | None = None,
        now: "callable[[], datetime] | None" = None,
    ) -> None:
        self._config = config
        self._base = config.base_url.rstrip("/")
        self._session = session or requests.Session()
        self._env = env if env is not None else os.environ
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at: datetime | None = None

    # --- credentials --------------------------------------------------

    def has_credentials(self) -> bool:
        """True if both env vars are set and non-blank. Callers use this to
        skip protected endpoints entirely rather than provoke a guaranteed
        failure.
        """
        return bool(self._credential("client_id", ENV_CLIENT_ID, quiet=True)) and bool(
            self._credential("client_secret", ENV_CLIENT_SECRET, quiet=True)
        )

    def _credential(self, label: str, env_var: str, *, quiet: bool = False) -> str:
        value = (self._env.get(env_var) or "").strip()
        if not value and not quiet:
            raise BackendCredentialsError(
                f"service {label} is not configured: set {env_var} in the environment "
                f"(see .env.example). No token request was made."
            )
        return value

    # --- token lifecycle ----------------------------------------------

    def token(self) -> str:
        """The current access token, fetching or refreshing if needed.

        Raises `BackendCredentialsError` when unconfigured, `BackendAuthError`
        when the backend rejects the credentials, `BackendContractError` on
        an unusable token response, `BackendUnavailableError` on transport
        failure.
        """
        cached = self._cached_token()
        if cached is not None:
            return cached
        with self._lock:
            # Re-check inside the lock: another thread may have refreshed
            # while this one waited, in which case reuse its token.
            cached = self._cached_token(locked=True)
            if cached is not None:
                return cached
            return self._fetch_locked()

    def invalidate(self) -> None:
        """Drop the cached token so the next `token()` fetches a fresh one.

        Used by the 401-retry path: the backend may reject a token this
        process still considers valid (clock skew, a server-side restart,
        or a revoked client), and the correct response is one refresh, not
        a persistent failure.
        """
        with self._lock:
            self._token = None
            self._expires_at = None

    def _cached_token(self, *, locked: bool = False) -> str | None:
        if not locked:
            with self._lock:
                return self._cached_token(locked=True)
        if self._token is None or self._expires_at is None:
            return None
        if self._now() >= self._expires_at - _REFRESH_SKEW:
            return None
        return self._token

    def _fetch_locked(self) -> str:
        # Read credentials only now - an unconfigured provider must fail
        # before any network call (and `has_credentials` stays side-effect free).
        client_id = self._credential("client_id", ENV_CLIENT_ID)
        client_secret = self._credential("client_secret", ENV_CLIENT_SECRET)
        url = f"{self._base}{TOKEN_PATH}"
        try:
            resp = self._session.request(
                "POST",
                url,
                json={"clientId": client_id, "clientSecret": client_secret},
                timeout=self._config.timeout_seconds,
                verify=self._config.tls_verify,
            )
        except requests.exceptions.RequestException as exc:
            # `exc` can carry the request body (i.e. the secret) in some
            # requests/urllib3 paths - report the type only, never the object.
            raise BackendUnavailableError(
                f"POST {TOKEN_PATH} failed: {type(exc).__name__}"
            ) from None

        if resp.status_code in (400, 401, 403):
            # Body omitted deliberately: a credential-rejection body can echo
            # the submitted clientId.
            raise BackendAuthError(
                f"POST {TOKEN_PATH} rejected the service credentials (HTTP {resp.status_code}); "
                f"check {ENV_CLIENT_ID}/{ENV_CLIENT_SECRET}",
                status_code=resp.status_code,
            )
        if not resp.ok:
            raise BackendContractError(f"POST {TOKEN_PATH} returned HTTP {resp.status_code}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise BackendContractError(f"POST {TOKEN_PATH}: response body is not valid JSON: {exc}") from None
        if not isinstance(payload, dict):
            raise BackendContractError(
                f"POST {TOKEN_PATH}: expected a JSON object, got {type(payload).__name__}"
            )
        if payload.get("success") is False:
            raise BackendAuthError(
                f"POST {TOKEN_PATH}: backend reported failure", status_code=resp.status_code
            )
        # Envelope-tolerant: the live exchange nests the token under `data`
        # like every other endpoint, but Swagger publishes no response schema
        # for this route, so a future un-enveloped body is accepted too.
        body = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        try:
            token = ApiServiceToken.model_validate(body)
        except Exception as exc:  # pydantic ValidationError - message is field-level, no values
            raise BackendContractError(
                f"POST {TOKEN_PATH}: response has no usable accessToken ({type(exc).__name__})"
            ) from None

        expires_at = _as_aware_utc(token.expires_at_utc)
        if expires_at is None:
            expires_at = self._now() + _FALLBACK_TTL
            logger.warning(
                "%s returned no expiresAtUtc; using a conservative %.0fs TTL",
                TOKEN_PATH, _FALLBACK_TTL.total_seconds(),
            )
        elif expires_at <= self._now():
            raise BackendContractError(
                f"POST {TOKEN_PATH}: token already expired at {expires_at.isoformat()}"
            )

        self._token = token.access_token
        self._expires_at = expires_at
        # Length, not value - enough to distinguish "empty token" from
        # "wrong token" in a log without ever writing the credential out.
        logger.info(
            "acquired backend service token (len=%d) valid until %s",
            len(token.access_token), expires_at.isoformat(),
        )
        return token.access_token


def _as_aware_utc(dt: datetime | None) -> datetime | None:
    """A naive `expiresAtUtc` is UTC by its own name - matching how
    `loader._as_naive_utc` treats the backend's naive timestamps.
    """
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
