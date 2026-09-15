"""Canonical UserProfile schema.

PRODUCTION-SAFE FEATURE CONTRACT (docs/production-feature-parity-audit.md):

- `preferred_categories`: production-safe. The real backend models this as
  a `FavoriteCategory[]` join (`GET /api/users/{guid}`'s
  `preferredCategories` array - `backend.dtos.ApiUser`), a LIST, not a
  single scalar - so this field is a list here too, matching that shape
  exactly rather than arbitrarily collapsing to "the first favorite."
  SQLite/synthetic sources that only ever have one preferred category
  populate a length-<=1 list; nothing downstream needs to special-case
  that. Empty list = no signal (same as `None` used to mean).
- `age_group`: LEGACY / METADATA ONLY, kept Optional so a source that
  genuinely has no concept of it (the real backend's `UserResponse` schema
  has no such field at all - confirmed absent, not just unpopulated) still
  produces a valid profile. No production Two-Tower or ranker feature
  consumes this field any more - see `retrieval.two_tower.feature_encoding`
  and `ranking.features` module docstrings. Left on the schema rather than
  removed so a UI/debug view can still show it if a source happens to
  provide it, and so no downstream code needs defensive `getattr`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    user_id: int
    preferred_categories: list[str] = Field(default_factory=list)
    age_group: str | None = None
