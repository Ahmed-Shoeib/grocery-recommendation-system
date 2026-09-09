"""Shared fakes for the backend-integration tests.

- `FakeResponse` / `FakeSession`: a `requests`-shaped transport double used
  by the `BackendApiClient` and `ServiceTokenProvider` tests (HTTP layer).
- `FakeBackendClient`: an in-memory stand-in for `BackendApiClient` used by
  the `loader` / `backend_factory` tests (above the HTTP layer). Duck-typed:
  only the methods those modules call.
"""

from __future__ import annotations

from recommendation.backend.dtos import ApiActivity, ApiCategory, ApiProduct, ApiReview, ApiUser


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
    """Queues `FakeResponse`s (or exceptions to raise) and records every
    request made, so a test can assert on URL / params / headers / body.
    """

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


class FakeBackendClient:
    def __init__(
        self,
        *,
        products: list[dict] | None = None,
        categories: list[dict] | None = None,
        activities: list[dict] | None = None,
        users: dict[str, dict] | None = None,
        users_status: int = 200,
        reviews: list[dict] | None = None,
        has_credentials: bool = True,
    ) -> None:
        self._products = [ApiProduct.model_validate(p) for p in (products or [])]
        self._categories = [ApiCategory.model_validate(c) for c in (categories or [])]
        self._activities = [ApiActivity.model_validate(a) for a in (activities or [])]
        self._users = {k: ApiUser.model_validate(v) for k, v in (users or {}).items()}
        self._users_status = users_status
        self._reviews = [ApiReview.model_validate(r) for r in (reviews or [])]
        self._has_credentials = has_credentials
        self.user_calls: list[str] = []
        self.review_calls = 0

    def list_products(self) -> list[ApiProduct]:
        return list(self._products)

    def list_categories(self) -> list[ApiCategory]:
        return list(self._categories)

    def list_activities(self) -> list[ApiActivity]:
        return list(self._activities)

    def get_user(self, guid: str) -> ApiUser | None:
        self.user_calls.append(guid)
        if self._users_status in (401, 403, 404):
            return None
        return self._users.get(guid)

    def has_service_credentials(self) -> bool:
        return self._has_credentials

    def list_reviews(self) -> list[ApiReview]:
        self.review_calls += 1
        return list(self._reviews)
