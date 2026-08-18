"""Tests for `recommendation.features.price` (STEP 6, docs/data-mapping.md
section 15): effective price, catalog/category price statistics, price
tier assignment, and the user price profile fallback hierarchy - including
explicit temporal-leakage proof reusing STEP 5's `reference_time`/
`effective_weight` contract.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from recommendation.data.schemas.engagement import PurchaseRecord
from recommendation.data.schemas.product import Product
from recommendation.features.price import (
    PriceCatalogContext,
    assign_price_tier,
    build_price_catalog_context,
    build_user_price_profile,
    category_relative_price,
    compute_catalog_tier_boundaries,
    effective_price,
    is_discounted,
    price_relative_distance,
)
from recommendation.utils.config import RecencyConfig

T0 = datetime(2026, 6, 1)


def t(days_ago: float) -> datetime:
    return T0 - timedelta(days=days_ago)


def _product(**overrides) -> Product:
    defaults = dict(id=1, category_id=1, slug="p", name="P", price=10.0, category_name="Dairy & Eggs")
    defaults.update(overrides)
    return Product(**defaults)


# --- effective_price / is_discounted (section 27) -------------------------

def test_effective_price_regular_product_no_sale_price():
    assert effective_price(_product(price=10.0, sale_price=None)) == 10.0


def test_effective_price_valid_discount_uses_sale_price():
    assert effective_price(_product(price=10.0, sale_price=8.0)) == 8.0


def test_is_discounted_true_for_valid_discount():
    assert is_discounted(_product(price=10.0, sale_price=8.0)) is True


def test_is_discounted_false_when_no_sale_price():
    assert is_discounted(_product(price=10.0, sale_price=None)) is False


def test_effective_price_malformed_sale_price_gte_price_falls_back_to_price():
    """A `sale_price >= price` is not a real discount - safety net (see
    module docstring); not observed in the actual SQLite dataset but the
    schema doesn't forbid it.
    """
    assert effective_price(_product(price=10.0, sale_price=12.0)) == 10.0
    assert is_discounted(_product(price=10.0, sale_price=12.0)) is False
    assert effective_price(_product(price=10.0, sale_price=10.0)) == 10.0  # equal - not a discount either


def test_effective_price_zero_price_is_safe():
    """Canonical `Product` (unlike `RawProduct`) has no `gt=0` constraint
    on `price` - zero/degenerate prices must not crash or produce NaN/inf.
    """
    p = _product(price=0.0, sale_price=None)
    assert effective_price(p) == 0.0
    assert is_discounted(p) is False


# --- catalog tier boundaries / assignment (section 27) --------------------

def test_compute_catalog_tier_boundaries_is_deterministic():
    products = [_product(id=i, price=float(i)) for i in range(1, 10)]
    a = compute_catalog_tier_boundaries(products)
    b = compute_catalog_tier_boundaries(products)
    assert a == b


def test_compute_catalog_tier_boundaries_empty_catalog():
    assert compute_catalog_tier_boundaries([]) == (0.0, 0.0)


def test_assign_price_tier_boundary_behavior():
    boundaries = (10.0, 20.0)
    assert assign_price_tier(5.0, boundaries) == "budget"
    assert assign_price_tier(10.0, boundaries) == "budget"  # at-or-below lower -> budget
    assert assign_price_tier(15.0, boundaries) == "mid"
    assert assign_price_tier(20.0, boundaries) == "mid"  # at-or-below upper -> mid
    assert assign_price_tier(20.01, boundaries) == "premium"


def test_price_tier_is_deterministic_across_repeated_calls():
    products = [_product(id=i, price=float(i), category_name="Dairy & Eggs") for i in range(1, 31)]
    boundaries = compute_catalog_tier_boundaries(products)
    tiers_1 = [assign_price_tier(effective_price(p), boundaries) for p in products]
    tiers_2 = [assign_price_tier(effective_price(p), boundaries) for p in products]
    assert tiers_1 == tiers_2
    assert set(tiers_1) <= {"budget", "mid", "premium"}


# --- category_relative_price (section 20/27) -------------------------------

def test_category_relative_price_ranks_cheapest_to_priciest():
    products = [
        _product(id=1, price=1.0, category_name="Cat"),
        _product(id=2, price=2.0, category_name="Cat"),
        _product(id=3, price=3.0, category_name="Cat"),
    ]
    rel = category_relative_price(products)
    assert rel[1] < rel[2] < rel[3]
    assert rel[3] == pytest.approx(1.0)  # priciest -> top of its category


def test_category_relative_price_ties_get_equal_value():
    products = [
        _product(id=1, price=5.0, category_name="Cat"),
        _product(id=2, price=5.0, category_name="Cat"),
        _product(id=3, price=9.0, category_name="Cat"),
    ]
    rel = category_relative_price(products)
    assert rel[1] == rel[2]
    assert rel[1] < rel[3]


def test_category_relative_price_single_product_category_is_neutral():
    products = [_product(id=1, price=100.0, category_name="Lonely")]
    rel = category_relative_price(products)
    assert rel[1] == 0.5


def test_category_relative_price_differs_across_categories_for_same_absolute_price():
    """docs/data-mapping.md section 15's "$20 cereal vs $20 meat" example:
    the SAME price can rank differently depending on category context.
    """
    products = [
        _product(id=1, price=20.0, category_name="Cereal"),
        _product(id=2, price=5.0, category_name="Cereal"),
        _product(id=3, price=20.0, category_name="Meat"),
        _product(id=4, price=50.0, category_name="Meat"),
    ]
    rel = category_relative_price(products)
    assert rel[1] == pytest.approx(1.0)  # most expensive cereal
    assert rel[3] == pytest.approx(0.5)  # cheaper of the two meats


def test_category_relative_price_omits_products_without_category():
    products = [_product(id=1, price=5.0, category_name=None)]
    rel = category_relative_price(products)
    assert 1 not in rel


# --- build_price_catalog_context (section 8/9) ------------------------------

def test_build_price_catalog_context_hand_computed():
    products = [
        _product(id=1, price=2.0, category_name="A"),
        _product(id=2, price=4.0, category_name="A"),
        _product(id=3, price=6.0, category_name="B"),
    ]
    ctx = build_price_catalog_context(products)
    assert ctx.catalog_median_price == pytest.approx(4.0)
    assert ctx.category_median_price["A"] == pytest.approx(3.0)
    assert ctx.category_median_price["B"] == pytest.approx(6.0)
    assert ctx.category_std_price["B"] == pytest.approx(0.0)  # single product -> no spread


def test_build_price_catalog_context_is_catalog_only_never_reads_purchases():
    """Signature-level proof: `build_price_catalog_context` takes only a
    product list - there is no purchase/user parameter it could leak
    through, satisfying section 9's "tier boundaries must not use future
    user interactions" requirement structurally, not just by convention.
    """
    import inspect

    sig = inspect.signature(build_price_catalog_context)
    assert list(sig.parameters) == ["products"]


# --- build_user_price_profile: fallback hierarchy (section 16/28) ---------

def _ctx() -> PriceCatalogContext:
    return build_price_catalog_context(
        [
            _product(id=1, price=5.0, category_name="Snacks"),
            _product(id=2, price=7.0, category_name="Snacks"),
            _product(id=3, price=50.0, category_name="Meat"),
        ]
    )


def _lookup() -> dict[int, Product]:
    return {
        1: _product(id=1, price=5.0, category_name="Snacks"),
        2: _product(id=2, price=7.0, category_name="Snacks"),
        3: _product(id=3, price=50.0, category_name="Meat"),
    }


ENABLED = RecencyConfig(enabled=True, half_life_days=21.0)
DISABLED = RecencyConfig(enabled=False)


def test_purchase_history_drives_typical_price_when_available():
    purchases = [
        PurchaseRecord(user_id=1, product_id=1, quantity=1, order_created_at=t(5)),
        PurchaseRecord(user_id=1, product_id=2, quantity=1, order_created_at=t(3)),
    ]
    profile = build_user_price_profile(purchases, _lookup(), None, _ctx(), T0, DISABLED)
    assert profile.fallback_source == "purchase_history"
    assert profile.supporting_purchase_count == 2
    assert profile.typical_price == pytest.approx(6.0)  # unweighted mean of 5.0, 7.0


def test_single_purchase_has_zero_spread_not_nan():
    purchases = [PurchaseRecord(user_id=1, product_id=1, quantity=1, order_created_at=t(1))]
    profile = build_user_price_profile(purchases, _lookup(), None, _ctx(), T0, DISABLED)
    assert profile.supporting_purchase_count == 1
    assert profile.price_spread == 0.0
    assert math.isfinite(profile.typical_price)


def test_no_purchases_falls_back_to_preferred_category_prior():
    profile = build_user_price_profile([], _lookup(), "Snacks", _ctx(), T0, DISABLED)
    assert profile.fallback_source == "preferred_category_prior"
    assert profile.supporting_purchase_count == 0
    assert profile.typical_price == pytest.approx(6.0)  # median(5.0, 7.0)


def test_no_purchases_no_preferred_category_falls_back_to_catalog_prior():
    profile = build_user_price_profile([], _lookup(), None, _ctx(), T0, DISABLED)
    assert profile.fallback_source == "catalog_prior"
    assert profile.typical_price == pytest.approx(_ctx().catalog_median_price)


def test_preferred_category_not_in_catalog_falls_back_to_catalog_prior():
    profile = build_user_price_profile([], _lookup(), "Nonexistent Category", _ctx(), T0, DISABLED)
    assert profile.fallback_source == "catalog_prior"


def test_no_history_user_never_produces_nan_or_inf():
    profile = build_user_price_profile([], _lookup(), None, _ctx(), T0, DISABLED)
    assert math.isfinite(profile.typical_price)
    assert math.isfinite(profile.price_spread)
    assert profile.price_tier in ("budget", "mid", "premium")


def test_fallback_is_deterministic():
    a = build_user_price_profile([], _lookup(), "Snacks", _ctx(), T0, DISABLED)
    b = build_user_price_profile([], _lookup(), "Snacks", _ctx(), T0, DISABLED)
    assert a == b


def test_raw_purchase_count_reflects_only_resolvable_priced_purchases():
    """A purchase of a product NOT in `product_lookup` (and with no
    `unit_price`) cannot contribute a price - it's silently skipped from
    `supporting_purchase_count`, exactly like every other signal already
    handles an unresolvable product_id (see `features.user_features`).
    """
    purchases = [
        PurchaseRecord(user_id=1, product_id=1, quantity=1, order_created_at=t(1)),
        PurchaseRecord(user_id=1, product_id=999, quantity=1, order_created_at=t(2)),  # unknown product
    ]
    profile = build_user_price_profile(purchases, _lookup(), None, _ctx(), T0, DISABLED)
    assert profile.supporting_purchase_count == 1


# --- recency interaction (section 7/28): "recent spending shift" ----------

def test_recent_purchases_outweigh_old_ones_when_recency_enabled():
    """docs/data-mapping.md section 15's worked example: old purchases at
    ~20-30, recent purchases at ~70-80 - the recency-weighted typical
    price should sit much closer to the recent cluster than a plain mean
    would (which would land near the midpoint, ~46).
    """
    lookup = {i: _product(id=i, price=float(p), category_name="X") for i, p in enumerate([20, 25, 30, 70, 75, 80], start=1)}
    purchases = [
        PurchaseRecord(user_id=1, product_id=1, quantity=1, order_created_at=t(180)),
        PurchaseRecord(user_id=1, product_id=2, quantity=1, order_created_at=t(175)),
        PurchaseRecord(user_id=1, product_id=3, quantity=1, order_created_at=t(170)),
        PurchaseRecord(user_id=1, product_id=4, quantity=1, order_created_at=t(2)),
        PurchaseRecord(user_id=1, product_id=5, quantity=1, order_created_at=t(1)),
        PurchaseRecord(user_id=1, product_id=6, quantity=1, order_created_at=t(0)),
    ]
    ctx = build_price_catalog_context(list(lookup.values()))

    unweighted = build_user_price_profile(purchases, lookup, None, ctx, T0, DISABLED)
    recency_weighted = build_user_price_profile(purchases, lookup, None, ctx, T0, ENABLED)

    assert unweighted.typical_price == pytest.approx(50.0)  # plain mean of 20..80
    assert recency_weighted.typical_price > 65.0  # pulled strongly toward the recent 70/75/80 cluster
    assert recency_weighted.typical_price > unweighted.typical_price


def test_recency_disabled_or_no_reference_time_gives_plain_unweighted_mean():
    """Mirrors STEP 5's opt-in contract exactly (`features.recency
    .effective_weight`): omitting `reference_time`, or `enabled=False`,
    must give the identical, non-recency-adjusted result.
    """
    lookup = {1: _product(id=1, price=10.0), 2: _product(id=2, price=20.0)}
    purchases = [
        PurchaseRecord(user_id=1, product_id=1, quantity=1, order_created_at=t(200)),
        PurchaseRecord(user_id=1, product_id=2, quantity=1, order_created_at=t(1)),
    ]
    ctx = build_price_catalog_context(list(lookup.values()))

    disabled = build_user_price_profile(purchases, lookup, None, ctx, T0, DISABLED)
    no_reference = build_user_price_profile(purchases, lookup, None, ctx, None, ENABLED)
    assert disabled.typical_price == pytest.approx(15.0)
    assert no_reference.typical_price == pytest.approx(15.0)


# --- historical-price source: unit_price vs. product's current price -----

def test_prefers_unit_price_when_present_erd_synthetic_path():
    lookup = {1: _product(id=1, price=999.0)}  # current catalog price, deliberately different
    purchases = [PurchaseRecord(user_id=1, product_id=1, quantity=1, unit_price=12.5, order_created_at=t(1))]
    ctx = build_price_catalog_context(list(lookup.values()))
    profile = build_user_price_profile(purchases, lookup, None, ctx, T0, DISABLED)
    assert profile.typical_price == pytest.approx(12.5)


def test_falls_back_to_product_effective_price_when_unit_price_absent_user_events_path():
    """The `User_events`/SQLite-sourced path never populates `unit_price`
    (see module docstring's historical-price limitation) - falls back to
    the product's CURRENT effective_price as the best available proxy.
    """
    lookup = {1: _product(id=1, price=8.0, sale_price=6.0)}
    purchases = [PurchaseRecord(user_id=1, product_id=1, quantity=1, unit_price=None, order_created_at=t(1))]
    ctx = build_price_catalog_context(list(lookup.values()))
    profile = build_user_price_profile(purchases, lookup, None, ctx, T0, DISABLED)
    assert profile.typical_price == pytest.approx(6.0)  # effective_price, not raw price


# --- price_relative_distance (section 11) ----------------------------------

def test_price_relative_distance_no_profile_is_neutral():
    assert price_relative_distance(10.0, None) == 0.0


def test_price_relative_distance_scales_with_gap():
    close = price_relative_distance(6.0, 5.0)
    far = price_relative_distance(30.0, 5.0)
    assert close < far


def test_price_relative_distance_zero_when_prices_match():
    assert price_relative_distance(5.0, 5.0) == pytest.approx(0.0)


# --- temporal leakage (section 22/29 - essential test) ---------------------

def test_future_purchase_price_does_not_affect_profile_before_its_cutoff():
    """The core leakage scenario: an old purchase, a cutoff, and a FUTURE
    purchase at a very different price. Built the CORRECT way (history
    truncated to `action_time < cutoff` before profile construction,
    mirroring `evaluation.temporal_future_purchase
    .build_point_in_time_engagement_profile` + `build_user_features
    (reference_time=cutoff)`), the future purchase must have zero
    influence on `typical_price`/`price_tier`/`price_spread`.
    """
    lookup = {1: _product(id=1, price=10.0, category_name="X"), 2: _product(id=2, price=200.0, category_name="X")}
    ctx = build_price_catalog_context(list(lookup.values()))
    cutoff = t(30)

    old_purchase = PurchaseRecord(user_id=1, product_id=1, quantity=1, order_created_at=t(60))
    future_purchase = PurchaseRecord(user_id=1, product_id=2, quantity=1, order_created_at=t(1))  # AFTER cutoff (more recent than cutoff)

    # Correct: only history strictly before the cutoff is visible.
    visible_before_cutoff = [p for p in [old_purchase, future_purchase] if p.order_created_at < cutoff]
    profile_at_cutoff = build_user_price_profile(visible_before_cutoff, lookup, None, ctx, cutoff, ENABLED)

    assert profile_at_cutoff.supporting_purchase_count == 1
    assert profile_at_cutoff.typical_price == pytest.approx(10.0)  # only the $10 old purchase, never the $200 future one
    assert profile_at_cutoff.price_tier == assign_price_tier(10.0, ctx.catalog_tier_boundaries)

    # Now advance the cutoff PAST the future purchase - it legitimately
    # becomes history (section 23's repeat-purchase policy) and DOES
    # influence the profile.
    later_cutoff = t(0)
    visible_at_later_cutoff = [p for p in [old_purchase, future_purchase] if p.order_created_at < later_cutoff]
    profile_at_later_cutoff = build_user_price_profile(visible_at_later_cutoff, lookup, None, ctx, later_cutoff, ENABLED)
    assert profile_at_later_cutoff.supporting_purchase_count == 2
    # With recency enabled and the $200 purchase now much more recent, the
    # typical price should be pulled well above the old-only estimate.
    assert profile_at_later_cutoff.typical_price > profile_at_cutoff.typical_price


def test_leakage_guard_composes_with_recency_future_event_raises():
    """If a caller (bug) fails to truncate and hands `build_user_price_profile`
    a purchase dated AT/AFTER its own `reference_time`, the SAME
    `RecencyLeakageError` STEP 5 introduced must fire - no separate/weaker
    guard was added for price.
    """
    from recommendation.features.recency import RecencyLeakageError

    lookup = {1: _product(id=1, price=10.0)}
    ctx = build_price_catalog_context(list(lookup.values()))
    future_purchase = PurchaseRecord(user_id=1, product_id=1, quantity=1, order_created_at=T0 + timedelta(days=1))
    with pytest.raises(RecencyLeakageError):
        build_user_price_profile([future_purchase], lookup, None, ctx, T0, ENABLED)
