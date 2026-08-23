"""HTTP client the Streamlit dashboard uses to reach FastAPI - the ONLY
way `ui.dashboard` generates recommendations or reads user/catalog data
(docs/data-mapping.md section 18). This module constructs HTTP
requests, calls the API, and parses/validates responses into the same
typed `api.schemas` wire-contract models FastAPI itself returns - nothing
here reimplements recommendation, ranking, feature-engineering,
eligibility, or model-loading logic of any kind. Importing `api.schemas`
is safe from the Streamlit process: that module is pure Pydantic models
(see its own docstring) - it does not import `RecommendationService`,
TensorFlow, or any adapter/artifact-loading code, so no model weights or
SQLite adapters are ever touched here.

Every failure mode (connection refused, timeout, non-200, malformed JSON
body) raises one of the typed exceptions below. `ui.dashboard` is required
to catch these and render a clean message - it must NEVER fall back to
building a `RecommendationService`/calling the pipeline directly (the
explicit "DO NOT DO" item in docs/data-mapping.md section 18); a fallback
would silently reintroduce the second serving path this architecture
eliminates.
"""

from __future__ import annotations

import requests
from pydantic import ValidationError

from recommendation.api.schemas import (
    OfflineMetricsResponse,
    RecommendationResponse,
    UserListResponse,
    UserProfileResponse,
)


class ApiClientError(Exception):
    """Base class for every error this client raises."""


class ApiUnavailableError(ApiClientError):
    """The API process could not be reached at all (connection refused, DNS failure, ...)."""


class ApiTimeoutError(ApiClientError):
    """The API did not respond within the configured timeout."""


class ApiResponseError(ApiClientError):
    """The API responded with a non-2xx status code."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"API returned {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class UnknownUserError(ApiClientError):
    """The API reported (via 404) that this user_id does not exist."""

    def __init__(self, user_id: int) -> None:
        super().__init__(f"user {user_id} not found")
        self.user_id = user_id


class MalformedResponseError(ApiClientError):
    """The API returned a 2xx status but the body didn't match the expected schema."""


class RecommendationApiClient:
    """Thin, stateless wrapper around `requests` - one instance per
    configured `(base_url, timeout_seconds)` pair (see
    `ui.service_loader.load_api_client`, `utils.config.DashboardConfig`).
    """

    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self._base_url}{path}"
        try:
            response = requests.get(url, params=params, timeout=self._timeout)
        except requests.exceptions.Timeout as exc:
            raise ApiTimeoutError(f"request to {url} timed out after {self._timeout}s") from exc
        except requests.exceptions.ConnectionError as exc:
            raise ApiUnavailableError(f"could not reach the recommendation API at {url}") from exc
        except requests.exceptions.RequestException as exc:
            raise ApiClientError(f"request to {url} failed: {exc}") from exc

        if response.status_code >= 400:
            try:
                message = response.json().get("message", response.text)
            except ValueError:
                message = response.text
            raise ApiResponseError(response.status_code, message)

        try:
            return response.json()
        except ValueError as exc:
            raise MalformedResponseError(f"response from {url} was not valid JSON") from exc

    def get_recommendations(self, user_id: int, limit: int | None = None) -> RecommendationResponse:
        params = {"limit": limit} if limit is not None else None
        try:
            payload = self._get(f"/v1/users/{user_id}/recommendations", params=params)
        except ApiResponseError as exc:
            if exc.status_code == 404:
                raise UnknownUserError(user_id) from exc
            raise
        try:
            return RecommendationResponse.model_validate(payload)
        except ValidationError as exc:
            raise MalformedResponseError(f"recommendation response for user {user_id} did not match the expected schema: {exc}") from exc

    def list_users(self) -> UserListResponse:
        payload = self._get("/v1/users")
        try:
            return UserListResponse.model_validate(payload)
        except ValidationError as exc:
            raise MalformedResponseError(f"user list response did not match the expected schema: {exc}") from exc

    def get_user_profile(self, user_id: int) -> UserProfileResponse:
        try:
            payload = self._get(f"/v1/users/{user_id}/profile")
        except ApiResponseError as exc:
            if exc.status_code == 404:
                raise UnknownUserError(user_id) from exc
            raise
        try:
            return UserProfileResponse.model_validate(payload)
        except ValidationError as exc:
            raise MalformedResponseError(f"user profile response for user {user_id} did not match the expected schema: {exc}") from exc

    def get_offline_metrics(self) -> OfflineMetricsResponse:
        """Reads the persisted offline evaluation report via `GET
        /v1/metrics/offline` - a cheap read, no `top_n`
        parameter since the report's own `top_n`/`k_values` are fixed at
        the time it was generated (`scripts/generate_offline_report.py`),
        not chosen per request.
        """
        payload = self._get("/v1/metrics/offline")
        try:
            return OfflineMetricsResponse.model_validate(payload)
        except ValidationError as exc:
            raise MalformedResponseError(f"offline metrics response did not match the expected schema: {exc}") from exc
