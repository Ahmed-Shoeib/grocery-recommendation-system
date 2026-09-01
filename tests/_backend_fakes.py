"""Shared in-memory fake of `BackendApiClient` for the backend-integration
tests. Duck-typed: only the methods `loader` / `backend_factory` call.
"""

from __future__ import annotations

from recommendation.data.backend.dtos import ApiActivity, ApiCategory, ApiProduct, ApiUser


class FakeBackendClient:
    def __init__(
        self,
        *,
        products: list[dict] | None = None,
        categories: list[dict] | None = None,
        activities: list[dict] | None = None,
        users: dict[str, dict] | None = None,
        users_status: int = 200,
    ) -> None:
        self._products = [ApiProduct.model_validate(p) for p in (products or [])]
        self._categories = [ApiCategory.model_validate(c) for c in (categories or [])]
        self._activities = [ApiActivity.model_validate(a) for a in (activities or [])]
        self._users = {k: ApiUser.model_validate(v) for k, v in (users or {}).items()}
        self._users_status = users_status
        self.user_calls: list[str] = []

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
