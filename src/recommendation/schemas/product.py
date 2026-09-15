"""Canonical Product schema.

Mirrors the ERD's `Product` entity (plus denormalized category/tag names).
Deliberately excludes any recency/event fields (no view/click counters, no
impression data) per the V1 scope decision - see docs/data-mapping.md,
section 1.

PRODUCTION-SAFE FEATURE CONTRACT (docs/production-feature-parity-audit.md):
the real SQL Server `Products` table has ONLY `Id, Slug, Name, Description,
Price, StockQuantity, ProductImage, AltText, CreationDate, CategoryId,
UpdatedAt` - no `Brand`, no `isActive`, no `SalePrice`/`DiscountPercentage`,
no `Ingredients`; the real `Categories` table has no parent-category
hierarchy either. The fields below are split into two groups:

  REQUIRED PRODUCTION INPUTS (consumed by embeddings/Two-Tower/ranker/
  eligibility/diversity): `id`, `category_id`/`category_name`, `name`,
  `description`, `price`, `sale_price` (only via `effective_price`'s
  graceful `None` handling - see `features.price`), `stock_quantity`.

  LEGACY / METADATA-ONLY FIELDS (`brand`, `is_active`, `discount_percentage`,
  `ingredients`, `parent_category_name`, `tags`): kept on this schema so
  the SQLite/synthetic sources that still populate them don't need a
  migration, and so display/debug code can still surface them - but as of
  the production-safe contract redesign, NO product-text template,
  Two-Tower input, ranker feature, eligibility rule, or diversity penalty
  consumes any of them any more. A source that cannot provide them (the
  real backend REST API) simply leaves them at their defaults
  (`None`/`True`/`[]`) with no loss of model behavior.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Product(BaseModel):
    id: int
    category_id: int
    slug: str
    name: str
    description: str | None = None
    brand: str | None = None  # legacy/metadata only - see module docstring
    price: float
    sale_price: float | None = None
    discount_percentage: float | None = None  # legacy/metadata only - see module docstring
    stock_quantity: int = 0
    ingredients: str | None = None  # legacy/metadata only - see module docstring
    is_active: bool = True  # legacy/metadata only - see module docstring; real eligibility uses stock_quantity
    product_image: str | None = None
    alt_text: str | None = None
    tags: list[str] = Field(default_factory=list)  # legacy/metadata only - see module docstring

    # Denormalized for convenience (joined from Category at adapter time).
    category_name: str | None = None
    parent_category_name: str | None = None  # legacy/metadata only - real Categories has no hierarchy
