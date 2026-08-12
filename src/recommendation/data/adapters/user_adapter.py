"""UserAdapter backed by the ERD User entity.

Resolves `RawUser.preferred_category_id` (an FK) to the category name the
canonical `UserProfile.preferred_category` expects. Both
`preferred_category` and `age_group` are surfaced as `None` whenever the
raw value is missing or unresolvable - defensive by construction, so a
user created before the backend migration lands (or a dangling FK) never
raises, it just yields a profile with less signal.
"""

from __future__ import annotations

from recommendation.data.adapters.base import UserAdapter
from recommendation.data.schemas.user import UserProfile
from recommendation.data.synthetic.raw_schemas import RawCategory, RawUser


class InMemoryUserAdapter(UserAdapter):
    def __init__(self, users: list[RawUser], categories: list[RawCategory]) -> None:
        category_name_by_id = {c.id: c.name for c in categories}
        self._profiles: dict[int, UserProfile] = {}
        for user in users:
            preferred_category = (
                category_name_by_id.get(user.preferred_category_id)
                if user.preferred_category_id is not None
                else None
            )
            self._profiles[user.id] = UserProfile(
                user_id=user.id,
                preferred_category=preferred_category,
                age_group=user.age_group,
            )

    def get_user_profile(self, user_id: int) -> UserProfile | None:
        return self._profiles.get(user_id)

    def list_user_ids(self) -> list[int]:
        return list(self._profiles.keys())
