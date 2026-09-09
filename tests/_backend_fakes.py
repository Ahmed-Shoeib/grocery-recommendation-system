"""Shared in-memory fake of `BackendApiClient` for the backend-integration
tests. Duck-typed: only the methods `loader` / `backend_factory` call.
"""

from __future__ import annotations

from recommendation.backend.dtos import ApiActivity, ApiCategory, ApiProduct, ApiReview, ApiUser


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
