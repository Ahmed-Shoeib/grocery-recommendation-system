"""Price-aware derived features (STEP 6 - see docs/data-mapping.md
section 15).

Backend price fields (`Product.Price`/`SalePrice`/`DiscountPercentage`)
are plain data inputs; every concept in this module is DERIVED in the
recommendation feature layer and never written back to the backend/SQLite
- see the module docstring's "non-negotiable ERD constraint" note in
docs/data-mapping.md section 15.

Layers, cheapest to most derived:

  1. `effective_price`/`is_discounted` - per-product, from `Product` alone.
  2. `PriceCatalogContext` (`build_price_catalog_context`) - CATALOG-WIDE
     statistics (median/std/tertile boundaries, per-category median/std),
     fit ONCE from the product catalog snapshot - never from user
     behavior, so this is never a temporal-leakage surface (docs/
     data-mapping.md section 15's "tier boundaries" note): the SAME
     context is valid for every user/evaluation cutoff because this
     project has exactly one static catalog snapshot throughout.
  3. `category_relative_price` - per-product percentile rank of
     `effective_price` WITHIN its own category, also catalog-only.
  4. `UserPriceProfile` (`build_user_price_profile`) - the one genuinely
     user-behavior-derived piece, built from PURCHASE history only (see
     that function's docstring for why: docs/data-mapping.md section 15,
     "purchases drive price preference"), recency-aware via the SAME
     `features.recency.effective_weight`/`reference_time` mechanism
     STEP 5 introduced (no new recency implementation), and therefore
     point-in-time-safe for the temporal future-purchase protocol under
     the exact same contract `build_user_features` already guarantees.

**Historical-price limitation**: `PurchaseRecord.unit_price` (the actual
transaction price) is populated ONLY for the ERD-based synthetic path
(`Order`/`OrderItem.UnitPrice`) - the confirmed `User_events` contract
(and therefore the SQLite-sourced path) carries no price at all (see
`data.adapters.user_events_adapter` module docstring: "`User_events`
never carries order_id, unit_price, quantity..."). For a `User_events`-
sourced purchase, `build_user_price_profile` falls back to the product's
CURRENT `effective_price` as the best available proxy for "what did this
purchase cost" - it is NOT necessarily what the product cost at the
actual moment of purchase. This is a real, honest limitation (same spirit
as docs/data-mapping.md section 12's "V1 leakage limitation - no
timestamps"), not a bug; it only matters for products whose price has
since changed, which this static synthetic catalog snapshot never does.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from recommendation.data.schemas.engagement import PurchaseRecord
from recommendation.data.schemas.product import Product
from recommendation.features.recency import effective_weight
from recommendation.utils.config import RecencyConfig

PRICE_TIERS = ["budget", "mid", "premium"]


def effective_price(product: Product) -> float:
    """The price the recommendation layer treats as "currently relevant":
    `sale_price` when it is a VALID discount (`0 < sale_price < price`),
    otherwise `price`. Slightly stricter than the pre-STEP-6 formula in
    `features.product_features` (which only checked `sale_price is not
    None`) - handles a malformed `sale_price >= price` safely by falling
    back to `price` rather than treating a non-discount as one. Not
    currently observed in `data/sqlite/backend_shaped_synthetic.db`
    (verified: 0 malformed rows), but `Product`'s schema doesn't forbid
    it, so this is defensive, not dead code.
    """
    if product.sale_price is not None and 0 < product.sale_price < product.price:
        return product.sale_price
    return product.price


def is_discounted(product: Product) -> bool:
    return product.sale_price is not None and 0 < product.sale_price < product.price


def compute_catalog_tier_boundaries(products: list[Product]) -> tuple[float, float]:
    """(lower, upper) tertile cutoffs of `effective_price` over the WHOLE
    catalog - data-relative, not arbitrary absolute constants (docs/
    data-mapping.md section 15). `assign_price_tier` uses these as
    `price <= lower -> budget`, `lower < price <= upper -> mid`,
    `price > upper -> premium`. Deterministic (numpy's default linear-
    interpolation percentile), reproducible from the same product list.
    """
    if not products:
        return (0.0, 0.0)
    prices = np.array([effective_price(p) for p in products], dtype=np.float64)
    return float(np.percentile(prices, 100.0 / 3.0)), float(np.percentile(prices, 200.0 / 3.0))


def assign_price_tier(price: float, boundaries: tuple[float, float]) -> str:
    lower, upper = boundaries
    if price <= lower:
        return "budget"
    if price <= upper:
        return "mid"
    return "premium"


def category_relative_price(products: list[Product]) -> dict[int, float]:
    """product_id -> percentile rank (0, 1] of `effective_price` WITHIN
    that product's own category (fraction of same-category products at or
    below this price) - so a $20 item can read as "premium" in one
    category and "budget" in another (docs/data-mapping.md section 15's
    "$20 cereal vs $20 meat" example). Products with no `category_name`,
    or the sole product in their category (no meaningful within-category
    comparison), are omitted here - callers should default to `0.5`
    (neutral) for a missing entry.
    """
    by_category: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for p in products:
        if p.category_name:
            by_category[p.category_name].append((p.id, effective_price(p)))

    result: dict[int, float] = {}
    for items in by_category.values():
        if len(items) == 1:
            result[items[0][0]] = 0.5
            continue
        sorted_prices = sorted(price for _, price in items)
        n = len(sorted_prices)
        for pid, price in items:
            result[pid] = bisect_right(sorted_prices, price) / n
    return result


@dataclass(frozen=True)
class PriceCatalogContext:
    """Catalog-snapshot-only price statistics - see module docstring
    point 2. Built ONCE (`build_price_catalog_context`) and reused across
    every user/evaluation point; never derived from any user's behavior,
    so it carries no temporal-leakage risk regardless of evaluation
    cutoff (docs/data-mapping.md section 15).
    """

    catalog_median_price: float
    catalog_std_price: float
    catalog_tier_boundaries: tuple[float, float]
    category_median_price: dict[str, float] = field(default_factory=dict)
    category_std_price: dict[str, float] = field(default_factory=dict)


def build_price_catalog_context(products: list[Product]) -> PriceCatalogContext:
    prices = np.array([effective_price(p) for p in products], dtype=np.float64)
    catalog_median = float(np.median(prices)) if prices.size else 0.0
    catalog_std = float(np.std(prices)) if prices.size else 0.0
    boundaries = compute_catalog_tier_boundaries(products)

    by_category: dict[str, list[float]] = defaultdict(list)
    for p in products:
        if p.category_name:
            by_category[p.category_name].append(effective_price(p))
    category_median = {cat: float(np.median(v)) for cat, v in by_category.items()}
    category_std = {cat: float(np.std(v)) for cat, v in by_category.items()}

    return PriceCatalogContext(
        catalog_median_price=catalog_median,
        catalog_std_price=catalog_std,
        catalog_tier_boundaries=boundaries,
        category_median_price=category_median,
        category_std_price=category_std,
    )


@dataclass(frozen=True)
class UserPriceProfile:
    """A user's derived price-preference summary - see module docstring
    point 4. `fallback_source` makes the 3-level hierarchy (docs/
    data-mapping.md section 15) explicit and inspectable rather than
    silent:

      "purchase_history"          - >=1 purchase with a resolvable price;
                                     `typical_price` is the RECENCY-
                                     WEIGHTED mean of those prices (reuses
                                     STEP 5's `effective_weight`, so a
                                     recent price shift is recognized -
                                     see `build_user_price_profile`).
      "preferred_category_prior"  - zero usable purchases, but the user
                                     has a `preferredCategory`; falls back
                                     to that category's catalog median.
      "catalog_prior"             - zero usable purchases AND no
                                     preferred category; falls back to the
                                     whole-catalog median. This is the
                                     ONLY fallback for a NO_HISTORY user.

    `price_spread` is the (unweighted) population standard deviation of
    the underlying price sample - the user's own purchase prices for
    "purchase_history" (0.0 for exactly one purchase - a single point has
    no variance, not NaN), the category's price spread for
    "preferred_category_prior", or the catalog's price spread for
    "catalog_prior" - i.e. always "how much does this price estimate's
    source vary", never left undefined.
    """

    typical_price: float
    price_spread: float
    price_tier: str
    supporting_purchase_count: int
    fallback_source: str


def _purchase_price(purchase: PurchaseRecord, product_lookup: dict[int, Product]) -> float | None:
    if purchase.unit_price is not None:
        return purchase.unit_price
    product = product_lookup.get(purchase.product_id)
    return effective_price(product) if product is not None else None


def build_user_price_profile(
    purchases: list[PurchaseRecord],
    product_lookup: dict[int, Product],
    preferred_category: str | None,
    price_context: PriceCatalogContext,
    reference_time: datetime | None,
    recency_config: RecencyConfig,
) -> UserPriceProfile:
    """`purchases` must already be whatever leave-one-out-excluded / point-
    in-time-truncated list the caller (`build_user_features`) is using for
    every other signal - this function does no exclusion or truncation of
    its own, exactly mirroring how category/brand affinity and the
    semantic embedding consume `purchases` (see
    `features.user_features.build_user_features`). `reference_time`/
    `recency_config` are forwarded UNCHANGED to `features.recency
    .effective_weight` - no separate recency implementation, no separate
    wall-clock lookup, so this inherits STEP 5's exact leakage-safety
    contract: recency (and therefore `typical_price`) can never be
    computed relative to a time the caller didn't explicitly provide.
    """
    priced: list[tuple[float, datetime | None]] = []
    for p in purchases:
        price = _purchase_price(p, product_lookup)
        if price is not None:
            priced.append((price, p.order_created_at))

    if priced:
        prices = np.array([pr for pr, _ in priced], dtype=np.float64)
        weights = np.array(
            [effective_weight(1.0, t, reference_time, recency_config) for _, t in priced], dtype=np.float64
        )
        weight_sum = weights.sum()
        typical = float((prices * weights).sum() / weight_sum) if weight_sum > 0 else float(prices.mean())
        spread = float(prices.std())  # unweighted: "how variable has this user's purchasing been", 0.0 for n=1
        tier = assign_price_tier(typical, price_context.catalog_tier_boundaries)
        return UserPriceProfile(typical, spread, tier, len(priced), "purchase_history")

    if preferred_category is not None and preferred_category in price_context.category_median_price:
        typical = price_context.category_median_price[preferred_category]
        spread = price_context.category_std_price.get(preferred_category, 0.0)
        tier = assign_price_tier(typical, price_context.catalog_tier_boundaries)
        return UserPriceProfile(typical, spread, tier, 0, "preferred_category_prior")

    tier = assign_price_tier(price_context.catalog_median_price, price_context.catalog_tier_boundaries)
    return UserPriceProfile(
        price_context.catalog_median_price, price_context.catalog_std_price, tier, 0, "catalog_prior"
    )


def price_relative_distance(candidate_price: float, user_typical_price: float | None, epsilon: float = 1e-6) -> float:
    """`|candidate_price - user_typical_price| / max(user_typical_price,
    epsilon)` - scale-invariant (a $5 gap means something different for a
    $6 shopper vs. a $60 shopper), a compact, interpretable cross-feature
    (docs/data-mapping.md section 15). `0.0` (neutral - not a
    perfect-match claim; callers should also check a `has_price_profile`
    flag, see `ranking.features`) when there is no user price signal at all.
    """
    if user_typical_price is None:
        return 0.0
    return abs(candidate_price - user_typical_price) / max(user_typical_price, epsilon)
