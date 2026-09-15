"""UserAdapter backed by the ERD User entity.

Resolves every `RawUser.preferred_category_ids` FK to a category name for
the canonical `UserProfile.preferred_categories` LIST - never reduced to a
single value here (see `schemas.user.UserProfile` docstring and
docs/production-feature-parity-audit.md). A dangling/unresolvable FK is
simply dropped from the list rather than raising, matching the existing
defensive-by-construction style (a user created before the backend
migration lands, or with a stale category reference, just yields a
profile with less signal, never an error). `age_group` is surfaced as
`None` when missing, same as before.
"""

from __future__ import annotations

from recommendation.adapters.base import UserAdapter
from recommendation.schemas.user import UserProfile
from recommendation.synthetic.raw_schemas import RawCategory, RawUser


class InMemoryUserAdapter(UserAdapter):
    def __init__(self, users: list[RawUser], categories: list[RawCategory]) -> None:
        category_name_by_id = {c.id: c.name for c in categories}
        self._profiles: dict[int, UserProfile] = {}
        for user in users:
            preferred_categories = [
                category_name_by_id[cid] for cid in user.preferred_category_ids if cid in category_name_by_id
            ]
            self._profiles[user.id] = UserProfile(
                user_id=user.id,
                preferred_categories=preferred_categories,
                age_group=user.age_group,
            )

    def get_user_profile(self, user_id: int) -> UserProfile | None:
        return self._profiles.get(user_id)

    def list_user_ids(self) -> list[int]:
        return list(self._profiles.keys())
