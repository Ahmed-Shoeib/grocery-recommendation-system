"""Canonical UserProfile schema.

`preferred_category` and `age_group` are confirmed backend additions to the
`User` entity (not yet reflected in docs/erd.jpeg, which predates the
change - see docs/data-mapping.md, section 2). Both are modeled as
Optional so inference degrades gracefully for users created before the
migration lands, rather than requiring them.

`age_group` is treated as an opaque categorical label from the backend: no
assumed semantics or hard-coded age-based stereotypes are attached to it
here or in any downstream feature. Its predictive value is something to be
validated experimentally in the ranking model, not assumed up front.
"""

from __future__ import annotations

from pydantic import BaseModel


class UserProfile(BaseModel):
    user_id: int
    preferred_category: str | None = None
    age_group: str | None = None
