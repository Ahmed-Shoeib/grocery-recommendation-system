"""HTTP client for the grocery backend REST API.

The only component that performs network I/O against the backend. Concerns
handled here and nowhere else: base URL, timeouts, connection-error vs
HTTP-error vs contract-error classification, bounded retries for transient
statuses, TLS verification, the `{success, data}` response envelope, and
both pagination styles the backend uses. Output is always a list of
`recommendation.backend.dtos` models - HTTP details never escape.

Auth: per-request, never session-wide. `/api/categories` (and the unused,
legacy `/api/products/{slug}` single-lookup helper `get_product`) are
public and called with NO Authorization header. Every endpoint in the
primary `backend_api` data path - `/api/ai/products`,
`/api/ai/user-activities`, `/api/users/{guid}`, `/api/reviews` - is
Bearer-gated, using a token from `auth.ServiceTokenProvider` (the
`POST /api/auth/service/token` client-credentials exchange). The header
is attached to the individual protected request rather than to
`Session.headers`, so a token can never leak onto a public call. See
docs/data-mapping.md section 19.5.

**`list_products`/`list_activities` require service credentials as of the
2026-09-15 atomic switch to `/api/ai/products`/`/api/ai/user-activities`**
(docs/data-mapping.md 19.5) - these are the authoritative catalog/activity
sources for `backend_api` and are no longer best-effort. With no
credentials configured, both raise `BackendCredentialsError` immediately
(no request sent) rather than silently falling back to the legacy
slug-only `/api/products`/`/api/user-activities`: a fallback here would
let a single deployment mix identity schemes across loads/refreshes,
which is worse than a loud, immediate, unambiguous failure. The legacy
plain endpoints are intentionally not called by this client at all
anymore - a single authoritative path per data type, no parallel default.

TLS: `verify` defaults to on. The dev backend presents a self-signed
`CN=localhost` certificate on a bare IP; for local work set
`BACKEND_TLS_VERIFY=false` (env) / `backend_api.tls_verify: false` (yaml).
Verification is never disabled in code.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import requests

from recommendation.backend.auth import ServiceTokenProvider
from recommendation.backend.dtos import (
    ApiActivity,
    ApiCategory,
    ApiPagination,
    ApiProduct,
    ApiReview,
    ApiUser,
    ApiUserIdentity,
)
from recommendation.backend.errors import (
    BackendAuthError,
    BackendContractError,
    BackendCredentialsError,
    BackendPaginationError,
    BackendResponseError,
    BackendUnavailableError,
)
from recommendation.config import BackendApiConfig
from recommendation.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
# 500 was added 2026-09-15 after directly observing /api/products,
# /api/categories and /api/ai/products all return persistent-looking
# HTTP 500 for ~30 minutes before recovering on their own with no
# client-side change (docs/data-mapping.md 19.5/19.8) - a real,
# demonstrated transient-failure class for this backend, not a guess.
# `max_retries` still bounds it: a *genuinely* persistent 500 exhausts
# the budget and raises `BackendResponseError` exactly like any other
# non-2xx status - this never silently hides a real, lasting failure.
_MAX_PAGES = 10_000  # hard stop so a broken `hasNext` can never loop forever
_CATALOG_MAX_LIMIT = 100  # backend rejects Limit > 100 on /api/categories with HTTP 400
_USER_LIST_MAX_LIMIT = 100  # backend caps PageSize on /api/users at 100 (verified live 2026-09-15)
_USER_LIST_MAX_PAGES = 1_000  # 547 live users / 100 per page = 6 pages; generous but still bounded


class BackendApiClient:
    def __init__(
        self,
        config: BackendApiConfig,
        *,
        session: requests.Session | None = None,
        token_provider: ServiceTokenProvider | None = None,
    ) -> None:
        self._config = config
        self._base = config.base_url.rstrip("/")
        self._session = session or requests.Session()
        # Shares the session (and therefore the connection pool) with data
        # requests; credentials come from the environment inside the provider.
        self._tokens = token_provider or ServiceTokenProvider(config, session=self._session)
        self._session.headers.setdefault("Accept", "application/json")
        self._session.headers.setdefault("User-Agent", config.user_agent)
        if not config.tls_verify:
            logger.warning(
                "backend TLS verification is DISABLED (backend_api.tls_verify=false) - "
                "development only; never run production traffic this way"
            )
            try:  # keep the log readable - one warning above is enough
                from urllib3.exceptions import InsecureRequestWarning  # type: ignore

                requests.packages.urllib3.disable_warnings(InsecureRequestWarning)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - urllib3 internals shift between versions
                pass

    # --- low-level ----------------------------------------------------

    def _request(self, path: str, params: dict[str, Any] | None = None, *, auth: bool = False) -> Any:
        """GET `path`, unwrapping the `{success, data}` envelope.

        `auth=True` attaches a service Bearer token to this request only.
        If the backend still answers 401 - a token this process considered
        valid can be rejected after a backend restart, a revoked client, or
        clock skew - the cached token is dropped and the request is retried
        exactly once with a fresh one. A second 401 is a real auth failure
        and raises, so a bad credential can never spin.
        """
        if not auth:
            return self._unwrap(self._send(path, params, None), path)

        resp = self._send(path, params, self._auth_header())
        if resp.status_code == 401:
            logger.info("GET %s returned 401; refreshing the service token and retrying once", path)
            self._tokens.invalidate()
            resp = self._send(path, params, self._auth_header())
        return self._unwrap(resp, path)

    def _auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._tokens.token()}"}

    def _send(
        self, path: str, params: dict[str, Any] | None, headers: dict[str, str] | None
    ) -> requests.Response:
        """One GET with the transport-level retry policy (connection errors
        and retryable statuses). Returns the raw response; status
        interpretation is `_unwrap`'s job.
        """
        url = f"{self._base}{path}"
        attempts = self._config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                resp = self._session.request(
                    "GET", url, params=params, headers=headers,
                    timeout=self._config.timeout_seconds, verify=self._config.tls_verify,
                )
            except requests.exceptions.RequestException as exc:
                if attempt < attempts:
                    self._backoff(attempt, f"{type(exc).__name__} for GET {path}")
                    continue
                raise BackendUnavailableError(f"GET {url} failed after {attempts} attempt(s): {exc}") from exc

            if resp.status_code in _RETRYABLE_STATUS and attempt < attempts:
                self._backoff(attempt, f"HTTP {resp.status_code} for GET {path}")
                continue
            return resp

        raise BackendUnavailableError(f"GET {url} exhausted retries")  # pragma: no cover - loop always returns/raises

    def _backoff(self, attempt: int, reason: str) -> None:
        delay = min(2.0 * attempt, 10.0)
        logger.warning("backend retry %d: %s; sleeping %.1fs", attempt, reason, delay)
        time.sleep(delay)

    def _unwrap(self, resp: requests.Response, path: str) -> Any:
        if resp.status_code in (401, 403):
            raise BackendAuthError(
                f"GET {path} returned {resp.status_code} (authentication required)",
                status_code=resp.status_code, body_excerpt=resp.text[:300],
            )
        if resp.status_code == 404:
            raise BackendResponseError(
                f"GET {path} returned 404", status_code=404, body_excerpt=resp.text[:300]
            )
        if not resp.ok:
            raise BackendResponseError(
                f"GET {path} returned HTTP {resp.status_code}",
                status_code=resp.status_code, body_excerpt=resp.text[:300],
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise BackendContractError(f"GET {path}: response body is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise BackendContractError(f"GET {path}: expected a JSON object envelope, got {type(payload).__name__}")
        if payload.get("success") is False:
            raise BackendContractError(
                f"GET {path}: backend reported failure: {payload.get('message') or payload}"
            )
        if "data" not in payload:
            raise BackendContractError(f"GET {path}: envelope has no 'data' key")
        return payload["data"]

    # --- pagination -------------------------------------------------

    def _paginated_rows(self, data: Any, path: str) -> list[dict]:
        """Validates and returns the `data: [...]` array from a paginated
        envelope (`{data: [...], pagination: {...}}`) - shared by every
        paginated endpoint here regardless of pagination STYLE (cursor:
        `_fetch_cursor_page`; page-number: `list_users`), since the
        envelope shape itself is identical either way.
        """
        if not isinstance(data, dict) or "data" not in data:
            raise BackendContractError(f"GET {path}: paginated response missing 'data' list")
        rows = data.get("data") or []
        if not isinstance(rows, list):
            raise BackendContractError(f"GET {path}: 'data' is not a list")
        return rows

    def _fetch_cursor_page(
        self,
        path: str,
        params_base: dict[str, Any] | None,
        cursor: str | None,
        *,
        cursor_param: str,
        limit_param: str,
        page_size: int,
        auth: bool,
    ) -> tuple[list[dict], ApiPagination]:
        """One page of a cursor-paginated GET, envelope-validated. Shared by
        `_iter_cursor` (run-to-completion-or-raise) and
        `iter_activity_pages` (bounded-by-design, never raises on running
        out of budget) so both pagination contracts stay backed by
        identical request/response handling.
        """
        params: dict[str, Any] = dict(params_base or {})
        params[limit_param] = page_size
        if cursor is not None:
            params[cursor_param] = cursor
        data = self._request(path, params, auth=auth)
        rows = self._paginated_rows(data, path)
        pagination = ApiPagination.model_validate(data.get("pagination") or {})
        return rows, pagination

    def _iter_cursor(
        self,
        path: str,
        *,
        cursor_param: str,
        limit_param: str,
        max_page_size: int | None = None,
        extra_params: dict[str, Any] | None = None,
        auth: bool = False,
    ) -> list[dict]:
        page_size = self._config.page_size
        if max_page_size is not None:
            page_size = min(page_size, max_page_size)
        items: list[dict] = []
        cursor: str | None = None
        for page in range(1, _MAX_PAGES + 1):
            rows, pagination = self._fetch_cursor_page(
                path, extra_params, cursor,
                cursor_param=cursor_param, limit_param=limit_param, page_size=page_size, auth=auth,
            )
            items.extend(rows)
            if not pagination.has_next:
                return items
            if not pagination.next_cursor:
                raise BackendPaginationError(
                    f"GET {path}: page {page} reports hasNext=true but no nextCursor"
                )
            cursor = pagination.next_cursor
        raise BackendPaginationError(f"GET {path}: exceeded {_MAX_PAGES}-page budget")

    # --- typed endpoints -------------------------------------------

    def list_products(self) -> list[ApiProduct]:
        """Authoritative `backend_api` product source (docs/data-mapping.md
        19.5): `GET /api/ai/products` - Bearer-gated, carries the backend's
        stable `Product.Id` on every row. A flat array like `/api/reviews`
        (Swagger declares no query parameters for this route, unlike the
        legacy cursor-paginated `/api/products`, which this client no
        longer calls at all). Requires service credentials - see the
        module docstring for why there is deliberately no slug-only
        fallback here.
        """
        data = self._request("/api/ai/products", auth=True)
        if data is None:
            return []
        if not isinstance(data, list):
            raise BackendContractError(
                f"GET /api/ai/products: expected a JSON array in 'data', got {type(data).__name__}"
            )
        return [ApiProduct.model_validate(r) for r in data]

    def list_ai_user_identities(self) -> list[ApiUserIdentity]:
        """`GET /api/ai/users` (2026-09-18 user-identity migration,
        `ai-user-identity-mapping`) - the authoritative, protected,
        service-to-service `User.Id <-> GUID` mapping. Requires `users:read`.

        Modeled as a flat array, same envelope convention as the other
        small `/api/ai/*` list resources this client already calls this
        way (`list_products` - see that method's docstring; Swagger
        declares no query parameters for those routes either). This is
        the live contract as actually observed for `/api/ai/products`;
        `/api/ai/users` was not independently live-probed for this change
        (the backend was unreachable at implementation time - see
        docs/data-mapping.md 19.17) - if the backend ships a paginated
        envelope instead, this raises `BackendContractError` below rather
        than silently returning an empty/wrong identity set, so a shape
        mismatch fails loudly at the very first load rather than quietly
        starving the roster.
        """
        data = self._request("/api/ai/users", auth=True)
        if data is None:
            return []
        if not isinstance(data, list):
            raise BackendContractError(
                f"GET /api/ai/users: expected a flat JSON array in 'data', got {type(data).__name__} - "
                "if the backend now paginates this endpoint, list_ai_user_identities() needs updating "
                "to match (see its docstring)."
            )
        return [ApiUserIdentity.model_validate(r) for r in data]

    def get_product(self, slug: str) -> ApiProduct | None:
        """Legacy single-product lookup via the public `/api/products/{slug}`
        detail route. Unused by the primary `backend_api` data path (which
        now reads the whole catalog from `list_products` /
        `GET /api/ai/products` instead) - kept only as a general-purpose,
        low-risk HTTP capability, not a parallel catalog source.
        """
        try:
            data = self._request(f"/api/products/{slug}")
        except BackendResponseError as exc:
            if exc.status_code == 404:
                return None
            raise
        return ApiProduct.model_validate(data)

    def list_categories(self) -> list[ApiCategory]:
        rows = self._iter_cursor(
            "/api/categories", cursor_param="Cursor", limit_param="Limit", max_page_size=_CATALOG_MAX_LIMIT
        )
        return [ApiCategory.model_validate(r) for r in rows]

    def list_activities(self) -> list[ApiActivity]:
        """Authoritative `backend_api` activity source (docs/data-mapping.md
        19.5): `GET /api/ai/user-activities` - Bearer-gated, cursor-paginated
        like the legacy `/api/user-activities` (same `cursor`/`pageSize`
        param names), but carries `productId` per row instead of `slug`.
        This client no longer calls the legacy plain endpoint at all.
        Requires service credentials - see the module docstring.
        """
        rows = self._iter_cursor(
            "/api/ai/user-activities", cursor_param="cursor", limit_param="pageSize", auth=True
        )
        return [ApiActivity.model_validate(r) for r in rows]

    def iter_activity_pages(
        self, *, user_guid: str | None = None, max_pages: int = _MAX_PAGES
    ) -> Iterator[list[ApiActivity]]:
        """Yields `GET /api/ai/user-activities` pages (newest-first per live
        observation, 2026-09-15) one page at a time, stopping after
        `max_pages` WITHOUT raising `BackendPaginationError`.

        Unlike `list_activities` (a run-to-completion-or-raise contract -
        the correct behavior for a caller that genuinely wants the whole
        set), this is a bounded-by-design primitive: the real
        `UserActivities` table has grown past 1.5 million rows with no
        server-side delta filter, so nothing in the normal serving path may
        ever attempt a full traversal again (docs/data-mapping.md 19.13).
        `backend.activity_sync` is the primary caller (bounded
        bootstrap/delta window); `adapters.backend_lazy_events_adapter`
        also uses this with `user_guid` set as a cold-start safety net -
        confirmed live that `userId`-filtered pagination is scoped to that
        user's own (much smaller) history, not the global table.

        Yielding page-by-page (rather than returning one list) lets a
        caller stop the instant it recognizes already-seen data, without
        pre-committing to a page count.
        """
        page_size = self._config.page_size
        cursor: str | None = None
        base_params = {"userId": user_guid} if user_guid else None
        for _ in range(max_pages):
            rows, pagination = self._fetch_cursor_page(
                "/api/ai/user-activities", base_params, cursor,
                cursor_param="cursor", limit_param="pageSize", page_size=page_size, auth=True,
            )
            yield [ApiActivity.model_validate(r) for r in rows]
            if not pagination.has_next or not pagination.next_cursor:
                return
            cursor = pagination.next_cursor

    def fetch_activity_window(
        self, *, user_guid: str | None = None, max_pages: int = _MAX_PAGES
    ) -> tuple[list[ApiActivity], bool]:
        """Like `iter_activity_pages`, but eager and completeness-aware:
        returns `(rows, exhausted)` where `exhausted=True` means the feed
        itself reported `hasNext=false` (genuine completion), not merely
        that `max_pages` was reached. `iter_activity_pages` (a lazy
        generator) cannot distinguish these two stopping reasons without
        the caller inspecting pagination internals - this exists
        specifically for `backend.activity_sync`'s bootstrap/per-user-
        completeness callers, which must never silently treat "we hit our
        page budget" as "this is the user's whole history"
        (docs/data-mapping.md 19.14).
        """
        page_size = self._config.page_size
        cursor: str | None = None
        base_params = {"userId": user_guid} if user_guid else None
        rows: list[ApiActivity] = []
        for _ in range(max_pages):
            page_rows, pagination = self._fetch_cursor_page(
                "/api/ai/user-activities", base_params, cursor,
                cursor_param="cursor", limit_param="pageSize", page_size=page_size, auth=True,
            )
            rows.extend(ApiActivity.model_validate(r) for r in page_rows)
            if not pagination.has_next or not pagination.next_cursor:
                return rows, True
            cursor = pagination.next_cursor
        return rows, False

    def list_users(self) -> list[ApiUser]:
        """`GET /api/users` (page-NUMBER pagination: `PageNumber`/`PageSize`,
        unlike every other endpoint here) - the full user roster (547 users
        live, 2026-09-15), independent of any recorded activity. Discovered
        during the activity-scaling investigation: this is what lets
        `adapters.backend_factory` know a user EXISTS (and their declared
        favorite categories) without first needing one of their activity
        rows to have been fetched - the fix for a real user being
        misclassified as cold-start purely because their history fell
        outside the bounded activity window (docs/data-mapping.md 19.13).

        Bearer-gated; reuses the `ApiUser` DTO (`extra='ignore'` tolerates
        the list projection's extra fields - phoneNumber/birthDate/role/
        isActive/createdAt - none of which this integration consumes). A
        single malformed row is skipped and logged rather than failing the
        whole roster fetch.
        """
        page_size = min(self._config.page_size, _USER_LIST_MAX_LIMIT)
        users: list[ApiUser] = []
        page = 1
        for _ in range(_USER_LIST_MAX_PAGES):
            data = self._request("/api/users", {"PageNumber": page, "PageSize": page_size}, auth=True)
            rows = self._paginated_rows(data, "/api/users")
            if not rows:
                break
            for row in rows:
                try:
                    users.append(ApiUser.model_validate(row))
                except Exception as exc:
                    logger.warning("GET /api/users: skipping one malformed roster row: %s", exc)
            pagination = ApiPagination.model_validate(data.get("pagination") or {})
            if not pagination.has_next:
                break
            page += 1
        return users

    def list_reviews(self) -> list[ApiReview]:
        """`GET /api/reviews` - Bearer-gated, and (unlike every other list
        endpoint here) NOT paginated: Swagger declares no query parameters
        and a flat `data` array (`AiProductReviewResponseListApiResponse`),
        verified live. A `null` data array means "no reviews", not an error.
        """
        data = self._request("/api/reviews", auth=True)
        if data is None:
            return []
        if not isinstance(data, list):
            raise BackendContractError(
                f"GET /api/reviews: expected a JSON array in 'data', got {type(data).__name__}"
            )
        return [ApiReview.model_validate(r) for r in data]

    def get_user(self, guid: str) -> ApiUser | None:
        """Bearer-gated; best-effort by design. Returns None (never raises)
        when credentials are absent or the backend rejects/does not have the
        user, so profile enrichment degrades to a low-signal profile instead
        of failing the whole data load. This endpoint must not become a hard
        dependency of basic user identity - the GUID from
        `/api/user-activities` is enough to serve a user.
        """
        try:
            data = self._request(f"/api/users/{guid}", auth=True)
        except BackendCredentialsError:
            return None
        except BackendAuthError as exc:
            logger.warning("GET /api/users/%s unauthorized (%s) - degrading to bare profile", guid, exc.status_code)
            return None
        except BackendResponseError as exc:
            if exc.status_code == 404:
                return None
            raise
        return ApiUser.model_validate(data)

    def has_service_credentials(self) -> bool:
        """Whether service auth is configured at all. Lets callers skip
        protected endpoints entirely (one clear log line) instead of
        producing a failure per call.
        """
        return self._tokens.has_credentials()
