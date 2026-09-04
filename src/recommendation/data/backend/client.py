"""HTTP client for the grocery backend REST API.

The only component that performs network I/O against the backend. Concerns
handled here and nowhere else: base URL, timeouts, connection-error vs
HTTP-error vs contract-error classification, bounded retries for transient
statuses, TLS verification, the `{success, data}` response envelope, and
both pagination styles the backend uses. Output is always a list of
`recommendation.data.backend.dtos` models - HTTP details never escape.

Auth: the recommender sends NO Authorization header. The endpoints it
needs (`/api/products`, `/api/categories`, `/api/user-activities`) are
public (verified live, still true as of 2026-09-04); `/api/users/{guid}`
is Bearer-gated and, per a live probe, stays that way - instead of opening
it up, the backend added a `POST /api/auth/service/token`
client-credentials exchange for service-to-service callers. This
integration deliberately does not implement that exchange yet (out of
scope for the data-mapping work here); `/api/users/{guid}` is treated as
best-effort (a 401/403 there logs once and degrades, it does not raise).
See docs/data-mapping.md section 19.

TLS: `verify` defaults to on. The dev backend presents a self-signed
`CN=localhost` certificate on a bare IP; for local work set
`BACKEND_TLS_VERIFY=false` (env) / `backend_api.tls_verify: false` (yaml).
Verification is never disabled in code.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from recommendation.data.backend.dtos import ApiActivity, ApiCategory, ApiPagination, ApiProduct, ApiUser
from recommendation.data.backend.errors import (
    BackendAuthError,
    BackendContractError,
    BackendPaginationError,
    BackendResponseError,
    BackendUnavailableError,
)
from recommendation.utils.config import BackendApiConfig
from recommendation.utils.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})
_MAX_PAGES = 10_000  # hard stop so a broken `hasNext` can never loop forever
_CATALOG_MAX_LIMIT = 100  # backend rejects Limit > 100 on /api/products and /api/categories with HTTP 400


class BackendApiClient:
    def __init__(self, config: BackendApiConfig, *, session: requests.Session | None = None) -> None:
        self._config = config
        self._base = config.base_url.rstrip("/")
        self._session = session or requests.Session()
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

    def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._base}{path}"
        attempts = self._config.max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = self._session.request(
                    "GET", url, params=params,
                    timeout=self._config.timeout_seconds, verify=self._config.tls_verify,
                )
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt < attempts:
                    self._backoff(attempt, f"{type(exc).__name__} for GET {path}")
                    continue
                raise BackendUnavailableError(f"GET {url} failed after {attempts} attempt(s): {exc}") from exc

            if resp.status_code in _RETRYABLE_STATUS and attempt < attempts:
                self._backoff(attempt, f"HTTP {resp.status_code} for GET {path}")
                continue
            return self._unwrap(resp, path)

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

    def _iter_cursor(
        self,
        path: str,
        *,
        cursor_param: str,
        limit_param: str,
        max_page_size: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> list[dict]:
        page_size = self._config.page_size
        if max_page_size is not None:
            page_size = min(page_size, max_page_size)
        items: list[dict] = []
        cursor: str | None = None
        for page in range(1, _MAX_PAGES + 1):
            params: dict[str, Any] = dict(extra_params or {})
            params[limit_param] = page_size
            if cursor is not None:
                params[cursor_param] = cursor
            data = self._request(path, params)
            if not isinstance(data, dict) or "data" not in data:
                raise BackendContractError(f"GET {path}: paginated response missing 'data' list")
            rows = data.get("data") or []
            if not isinstance(rows, list):
                raise BackendContractError(f"GET {path}: 'data' is not a list")
            items.extend(rows)
            pagination = ApiPagination.model_validate(data.get("pagination") or {})
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
        rows = self._iter_cursor(
            "/api/products", cursor_param="Cursor", limit_param="Limit", max_page_size=_CATALOG_MAX_LIMIT
        )
        return [ApiProduct.model_validate(r) for r in rows]

    def get_product(self, slug: str) -> ApiProduct | None:
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
        rows = self._iter_cursor("/api/user-activities", cursor_param="cursor", limit_param="pageSize")
        return [ApiActivity.model_validate(r) for r in rows]

    def get_user(self, guid: str) -> ApiUser | None:
        """Best-effort. Returns None (not raise) on 401/403/404 so a
        currently-protected or not-yet-migrated user endpoint degrades to
        a low-signal profile instead of failing the whole data load.
        """
        try:
            data = self._request(f"/api/users/{guid}")
        except BackendAuthError as exc:
            logger.warning("GET /api/users/%s unauthorized (%s) - degrading to bare profile", guid, exc.status_code)
            return None
        except BackendResponseError as exc:
            if exc.status_code == 404:
                return None
            raise
        return ApiUser.model_validate(data)
