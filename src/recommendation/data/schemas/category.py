"""Canonical Category schema.

Mirrors the ERD's `Category` entity, including the self-referential
`ParentId` used for parent-category features.
"""

from __future__ import annotations

from pydantic import BaseModel


class Category(BaseModel):
    id: int
    name: str
    parent_id: int | None = None
