"""Builds the text string fed to the Sentence Transformer for one product.

PRODUCTION-SAFE FEATURE CONTRACT (docs/production-feature-parity-audit.md):
combines ONLY `name`, `description`, and `category` - the three product-
text fields verified reproducible from BOTH the SQLite training source and
the real backend REST API. No price/stock/timestamp fields are included
either way: those are structured features (`features.product_features`),
not semantic-meaning inputs.

Previously this also included `brand`, `parent category`, `tags`, and
`ingredients`. All four were REMOVED:

  - `brand`: the real SQL Server `Products` table has no `Brand` column at
    all - a training-time text segment production could never reproduce.
  - parent category: the real `Categories` table has no parent-category
    hierarchy (no `ParentId`-equivalent) - `Category: {parent} > {name}`
    could never be built the same way at serving time.
  - `tags`: present on SQLite/synthetic products and even on the live dev
    backend's wire response, but the backend team has stated production
    will not carry them, and this remains an open, unresolved discrepancy
    (docs/data-mapping.md section 19.12) - not depended on either way.
  - `ingredients`: the real `Products` table has no such column.

Including any of these would make the training-time semantic embedding
systematically different from what the same product's text produces at
serving time against the real API (a `brand=None`/`tags=[]` product loses
those segments entirely) - training-serving text parity is exactly what
this template now guarantees: the SAME product text is built from the SAME
three fields regardless of source, verified in
`tests/test_embeddings.py`'s SQLite-vs-backend_api parity test.
"""

from __future__ import annotations

from recommendation.schemas.product import Product


def build_product_text(product: Product) -> str:
    parts: list[str] = [product.name]

    if product.category_name:
        parts.append(f"Category: {product.category_name}.")
    if product.description:
        parts.append(product.description)

    return " ".join(p.strip() for p in parts if p and p.strip())
