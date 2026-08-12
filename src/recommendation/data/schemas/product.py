"""Canonical Product schema.

Mirrors the ERD's `Product` entity (plus denormalized category/tag names
for convenient Sentence Transformer text-building in Phase 3). Deliberately
excludes any recency/event fields (no view/click counters, no impression
data) per the V1 scope decision - see docs/data-mapping.md, section 1.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Product(BaseModel):
    id: int
    category_id: int
    slug: str
    name: str
    description: str | None = None
    brand: str | None = None
    price: float
    sale_price: float | None = None
    discount_percentage: float | None = None
    stock_quantity: int = 0
    ingredients: str | None = None
    is_active: bool = True
    product_image: str | None = None
    alt_text: str | None = None
    tags: list[str] = Field(default_factory=list)

    # Denormalized for convenience (joined from Category at adapter time).
    category_name: str | None = None
    parent_category_name: str | None = None
