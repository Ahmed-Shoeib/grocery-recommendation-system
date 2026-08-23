"""Builds the text string fed to the Sentence Transformer for one product.

Combines name, brand, category, parent category, tags, description, and
ingredients into a single short passage for semantic product
representation. No price/stock/timestamp fields are included: those are
structured features (`features.product_features`), not semantic-meaning
inputs.
"""

from __future__ import annotations

from recommendation.data.schemas.product import Product


def build_product_text(product: Product) -> str:
    parts: list[str] = [product.name]

    if product.brand:
        parts.append(f"Brand: {product.brand}.")
    if product.category_name:
        category_phrase = product.category_name
        if product.parent_category_name:
            category_phrase = f"{product.parent_category_name} > {product.category_name}"
        parts.append(f"Category: {category_phrase}.")
    if product.tags:
        parts.append(f"Tags: {', '.join(product.tags)}.")
    if product.description:
        parts.append(product.description)
    if product.ingredients:
        parts.append(f"Ingredients: {product.ingredients}.")

    return " ".join(p.strip() for p in parts if p and p.strip())
