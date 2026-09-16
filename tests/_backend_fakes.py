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
        roster: list[dict] | None = None,
        activity_page_size: int = 100,
    ) -> None:
        self._products = [ApiProduct.model_validate(p) for p in (products or [])]
        self._categories = [ApiCategory.model_validate(c) for c in (categories or [])]
        self._activities = [ApiActivity.model_validate(a) for a in (activities or [])]
        self._users = {k: ApiUser.model_validate(v) for k, v in (users or {}).items()}
        self._users_status = users_status
        self._reviews = [ApiReview.model_validate(r) for r in (reviews or [])]
        self._has_credentials = has_credentials
        self._roster = [ApiUser.model_validate(u) for u in (roster or [])]
        self._activity_page_size = activity_page_size
        self.user_calls: list[str] = []
        self.review_calls = 0
        self.activity_page_calls: list[str | None] = []

    def list_products(self) -> list[ApiProduct]:
        return list(self._products)

    def list_categories(self) -> list[ApiCategory]:
        return list(self._categories)

    def list_activities(self) -> list[ApiActivity]:
        return list(self._activities)

    def list_users(self) -> list[ApiUser]:
        return list(self._roster)

    def iter_activity_pages(self, *, user_guid: str | None = None, max_pages: int = 10_000):
        """Mimics `BackendApiClient.iter_activity_pages`: paginates the
        fixture's `_activities` list (optionally filtered by `user_id`),
        `activity_page_size` rows at a time, stopping after `max_pages`.
        """
        self.activity_page_calls.append(user_guid)
        rows = self._activities
        if user_guid is not None:
            rows = [a for a in rows if a.user_id == user_guid]
        size = self._activity_page_size
        pages = [rows[i : i + size] for i in range(0, len(rows), size)] or [[]]
        for page in pages[:max_pages]:
            yield page

    def fetch_activity_window(self, *, user_guid: str | None = None, max_pages: int = 10_000):
        """Mimics `BackendApiClient.fetch_activity_window`: same
        pagination as `iter_activity_pages`, but eager and
        completeness-aware - returns `(rows, exhausted)`. Appends to the
        SAME `activity_page_calls` list (both are "one fetch attempt
        against `/api/ai/user-activities`" from a test's point of view).
        """
        self.activity_page_calls.append(user_guid)
        rows = self._activities
        if user_guid is not None:
            rows = [a for a in rows if a.user_id == user_guid]
        size = self._activity_page_size
        pages = [rows[i : i + size] for i in range(0, len(rows), size)] or [[]]
        truncated = pages[:max_pages]
        fetched = [r for page in truncated for r in page]
        exhausted = len(pages) <= max_pages
        return fetched, exhausted

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
