"""Price-aware feature diagnostics for STEP 6 (docs/data-mapping.md
section 15).

Purely illustrative - does NOT train, retrain, or touch any model
artifact. Prints, over `data/sqlite/backend_shaped_synthetic.db`:

  1. Several real products spanning budget/mid/premium/discounted/
     non-discounted, with their Price/SalePrice/effective_price/
     DiscountPercentage/category/category-relative-price/tier.
  2. Several real users covering: strong purchase history, a recent
     spending shift, a single purchase, engagement-with-zero-purchases,
     and NO_HISTORY - showing their historical purchase prices/
     timestamps, derived typical price/spread/tier, and which fallback
     tier produced the profile.
  3. User x candidate price-compatibility cross-features
     (`price_relative_distance`/`price_tier_match`) for a few real users
     against a close and a far candidate - the RAW feature values only,
     never a model score (no model is loaded here).

Reference time convention: identical to `scripts/recency_diagnostics.py`
- this dataset is static, so "now" is taken as the dataset's own latest
`User_events.action_time`, not wall-clock time.
"""

from __future__ import annotations

import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation.data.adapters.engagement import build_engagement_profile
from recommendation.data.adapters.sqlite_factory import build_sqlite_adapters
from recommendation.data.schemas.events import ActionType
from recommendation.data.sqlite.connection import open_readonly_connection
from recommendation.data.sqlite.loader import load_events
from recommendation.features.price import build_price_catalog_context, price_relative_distance
from recommendation.features.product_features import build_product_features
from recommendation.features.user_features import UserFeatures, build_user_features
from recommendation.utils.config import FeatureConfig, RecencyConfig, get_config


def _print_products(product_features: dict) -> None:
    print("=" * 78)
    print("SECTION: PRODUCT PRICE EXAMPLES")
    print("=" * 78)

    by_tier = {"budget": [], "mid": [], "premium": []}
    discounted, non_discounted = [], []
    for pf in product_features.values():
        by_tier[pf.price_tier].append(pf)
        (discounted if pf.is_discounted else non_discounted).append(pf)

    examples = []
    for tier in ("budget", "mid", "premium"):
        if by_tier[tier]:
            examples.append((tier, by_tier[tier][0]))
    if discounted:
        examples.append(("discounted", discounted[0]))
    if non_discounted:
        examples.append(("non-discounted", non_discounted[0]))

    for label, pf in examples:
        print(f"[{label}] product_id={pf.product_id} category={pf.category_name}")
        print(f"  price={pf.price:.2f} discount_percentage={pf.discount_percentage:.1f} effective_price={pf.effective_price:.2f}")
        print(f"  is_discounted={pf.is_discounted} price_tier={pf.price_tier} category_relative_price={pf.category_relative_price:.3f}")
        print()


def _user_features_for(uid, bundle, product_lookup, reference_time, feature_config, price_context) -> UserFeatures:
    profile = build_engagement_profile(
        uid, bundle.users, bundle.purchases, bundle.cart, bundle.clicks, bundle.search, bundle.chatbot, bundle.reviews
    )
    return build_user_features(profile, product_lookup, {}, feature_config, reference_time=reference_time, price_context=price_context)


def _describe_user(uid, events_by_user, product_lookup, reference_time, bundle, feature_config, price_context) -> UserFeatures:
    events = events_by_user.get(uid, [])
    purchase_events = sorted(
        (e for e in events if e.action_type == ActionType.PURCHASE and e.action_time), key=lambda e: e.action_time
    )

    print(f"\n--- USER {uid} ({len(events)} total events, {len(purchase_events)} purchases) ---")
    for e in purchase_events:
        product = product_lookup.get(e.product_id)
        age_days = (reference_time - e.action_time).days
        eff_price = product.price if product else None
        print(f"  {e.action_time.isoformat()} (age={age_days}d) product_id={e.product_id} effective_price={eff_price}")

    features = _user_features_for(uid, bundle, product_lookup, reference_time, feature_config, price_context)
    pp = features.price_profile
    print(
        f"  DERIVED: fallback_source={pp.fallback_source} typical_price={pp.typical_price:.2f} "
        f"price_spread={pp.price_spread:.2f} price_tier={pp.price_tier} supporting_purchase_count={pp.supporting_purchase_count}"
    )
    return features


def _pick_example_users(user_ids, events_by_user, reference_time) -> dict[str, int | None]:
    def purchase_times(uid):
        return sorted(
            e.action_time for e in events_by_user.get(uid, []) if e.action_type == ActionType.PURCHASE and e.action_time
        )

    picked: dict[str, int | None] = {
        "strong purchase history": None,
        "recent spending shift": None,
        "only one purchase": None,
        "engagements but zero purchases": None,
        "NO_HISTORY": None,
    }
    for uid in user_ids:
        times = purchase_times(uid)
        events = events_by_user.get(uid, [])
        if picked["strong purchase history"] is None and len(times) >= 6:
            picked["strong purchase history"] = uid
        if picked["recent spending shift"] is None and len(times) >= 4:
            early_age = (reference_time - times[0]).days
            late_age = (reference_time - times[-1]).days
            if early_age - late_age > 60:
                picked["recent spending shift"] = uid
        if picked["only one purchase"] is None and len(times) == 1:
            picked["only one purchase"] = uid
        if picked["engagements but zero purchases"] is None and events and not times:
            picked["engagements but zero purchases"] = uid
        if picked["NO_HISTORY"] is None and not events:
            picked["NO_HISTORY"] = uid
        if all(picked.values()):
            break
    return picked


def main() -> None:
    config = get_config()
    bundle = build_sqlite_adapters()

    con = open_readonly_connection(config.paths.data_sqlite)
    try:
        all_events = load_events(con)
    finally:
        con.close()

    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    product_features = build_product_features(products, [], [], [])
    price_context = build_price_catalog_context(products)

    _print_products(product_features)

    events_by_user: dict[int, list] = {}
    for e in all_events:
        events_by_user.setdefault(e.user_id, []).append(e)

    reference_time = max(e.action_time for e in all_events if e.action_time is not None)
    feature_config: FeatureConfig = config.features.model_copy(
        update={"recency": RecencyConfig(enabled=True, half_life_days=config.features.recency.half_life_days)}
    )

    print()
    print("=" * 78)
    print("SECTION: USER PRICE PROFILE EXAMPLES")
    print("=" * 78)

    user_ids = bundle.users.list_user_ids()
    picked = _pick_example_users(user_ids, events_by_user, reference_time)

    for label, uid in picked.items():
        print(f"\n[{label}]")
        if uid is None:
            print("  no matching example found in this dataset")
            continue
        _describe_user(uid, events_by_user, product_lookup, reference_time, bundle, feature_config, price_context)

    print()
    print("=" * 78)
    print("SECTION: USER x CANDIDATE PRICE COMPATIBILITY EXAMPLES")
    print("=" * 78)
    print("(raw cross-feature values only - no model is loaded/scored here)")

    candidates_sorted = sorted(product_features.values(), key=lambda pf: pf.effective_price)
    prices_sorted = [pf.effective_price for pf in candidates_sorted]

    for label in ("strong purchase history", "recent spending shift", "only one purchase"):
        uid = picked.get(label)
        if uid is None:
            continue
        features = _user_features_for(uid, bundle, product_lookup, reference_time, feature_config, price_context)
        pp = features.price_profile
        print(f"\n[{label}] user_id={uid} typical_price={pp.typical_price:.2f} price_tier={pp.price_tier}")

        idx = bisect.bisect_left(prices_sorted, pp.typical_price)
        close = candidates_sorted[min(idx, len(candidates_sorted) - 1)]
        far = candidates_sorted[0] if idx > len(candidates_sorted) / 2 else candidates_sorted[-1]
        for tag, cand in [("close", close), ("far", far)]:
            dist = price_relative_distance(cand.effective_price, pp.typical_price)
            tier_match = 1.0 if pp.price_tier == cand.price_tier else 0.0
            print(
                f"  [{tag}] candidate product_id={cand.product_id} effective_price={cand.effective_price:.2f} "
                f"price_tier={cand.price_tier} -> price_relative_distance={dist:.3f} price_tier_match={tier_match}"
            )


if __name__ == "__main__":
    main()
