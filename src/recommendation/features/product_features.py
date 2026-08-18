"""Structured (non-semantic) product features.

Complements the Sentence Transformer embedding: price/discount/stock are
useful ranking signals a text encoder can't see, and purchase/cart/review
aggregates give a recency-free "global popularity" / rating signal (per
docs/data-mapping.md section 6 - no "trending", just all-time counts).

Depends only on `recommendation.data.adapters.base` interfaces and
`recommendation.data.schemas`, never on synthetic internals, so it works
unchanged once real adapters replace the synthetic ones.

Price fields (STEP 6, docs/data-mapping.md section 15): `effective_price`/
`is_discounted`/`price_tier`/`category_relative_price` are all computed
via `features.price` from `products` alone (never from purchase/cart/
review behavior) - deterministic, catalog-snapshot-derived, and therefore
never a temporal-leakage surface. `price_tier`/`category_relative_price`
need the FULL product list (tertile boundaries, per-category ranking),
which is why they're computed once up front in `build_product_features`
rather than per-product.
"""

from __future__ import annotations

from dataclasses import dataclass

from recommendation.data.schemas.engagement import CartAffinityRecord, PurchaseRecord, ReviewRecord
from recommendation.data.schemas.product import Product
from recommendation.features.price import (
    assign_price_tier,
    category_relative_price,
    compute_catalog_tier_boundaries,
)
from recommendation.features.price import effective_price as _effective_price
from recommendation.features.price import is_discounted as _is_discounted


@dataclass
class ProductFeatures:
    product_id: int
    category_id: int
    category_name: str | None
    parent_category_name: str | None
    brand: str | None
    tags: list[str]

    price: float
    effective_price: float  # sale_price if a VALID discount, else price - see features.price.effective_price
    discount_percentage: float
    is_active: bool
    stock_quantity: int

    # All-time aggregates, no recency (see module docstring).
    purchase_count: int
    distinct_purchasers: int
    cart_add_count: int
    review_count: int
    average_rating: float | None

    # STEP 6 price-aware features (docs/data-mapping.md section 15).
    # Defaulted (unlike every field above) so pre-existing call sites/tests
    # that construct a ProductFeatures directly without price in mind keep
    # working unchanged - "mid"/0.5 are neutral, not "cheap"/"expensive"
    # claims, matching this module's other graceful-degradation defaults.
    is_discounted: bool = False
    price_tier: str = "mid"  # "budget" | "mid" | "premium" - catalog-wide tertiles, see features.price
    category_relative_price: float = 0.5  # percentile rank (0,1] of effective_price WITHIN this product's category


def compute_product_popularity(purchases: list[PurchaseRecord]) -> dict[int, tuple[int, int]]:
    """product_id -> (purchase_count, distinct_purchaser_count)."""
    counts: dict[int, int] = {}
    purchasers: dict[int, set[int]] = {}
    for p in purchases:
        counts[p.product_id] = counts.get(p.product_id, 0) + 1
        purchasers.setdefault(p.product_id, set()).add(p.user_id)
    return {pid: (count, len(purchasers[pid])) for pid, count in counts.items()}


def compute_cart_add_counts(cart_items: list[CartAffinityRecord]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for item in cart_items:
        counts[item.product_id] = counts.get(item.product_id, 0) + 1
    return counts


def compute_review_stats(reviews: list[ReviewRecord]) -> dict[int, tuple[float, int]]:
    """product_id -> (average_rating, review_count)."""
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for r in reviews:
        sums[r.product_id] = sums.get(r.product_id, 0.0) + r.rating
        counts[r.product_id] = counts.get(r.product_id, 0) + 1
    return {pid: (sums[pid] / counts[pid], counts[pid]) for pid in counts}


def build_product_features(
    products: list[Product],
    all_purchases: list[PurchaseRecord],
    all_cart_items: list[CartAffinityRecord],
    all_reviews: list[ReviewRecord],
) -> dict[int, ProductFeatures]:
    popularity = compute_product_popularity(all_purchases)
    cart_counts = compute_cart_add_counts(all_cart_items)
    review_stats = compute_review_stats(all_reviews)

    # Catalog-wide, computed ONCE over the full `products` list (not
    # per-product) - see module/`features.price` docstrings.
    tier_boundaries = compute_catalog_tier_boundaries(products)
    relative_price_by_id = category_relative_price(products)

    features: dict[int, ProductFeatures] = {}
    for product in products:
        purchase_count, distinct_purchasers = popularity.get(product.id, (0, 0))
        avg_rating, review_count = review_stats.get(product.id, (None, 0))
        eff_price = _effective_price(product)

        features[product.id] = ProductFeatures(
            product_id=product.id,
            category_id=product.category_id,
            category_name=product.category_name,
            parent_category_name=product.parent_category_name,
            brand=product.brand,
            tags=list(product.tags),
            price=product.price,
            effective_price=eff_price,
            discount_percentage=product.discount_percentage or 0.0,
            is_active=product.is_active,
            stock_quantity=product.stock_quantity,
            is_discounted=_is_discounted(product),
            price_tier=assign_price_tier(eff_price, tier_boundaries),
            category_relative_price=relative_price_by_id.get(product.id, 0.5),
            purchase_count=purchase_count,
            distinct_purchasers=distinct_purchasers,
            cart_add_count=cart_counts.get(product.id, 0),
            review_count=review_count,
            average_rating=avg_rating,
        )
    return features
