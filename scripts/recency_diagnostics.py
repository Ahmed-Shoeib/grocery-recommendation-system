"""Recency diagnostics for STEP 5 (recency weighting - see
docs/data-mapping.md section 14).

Does NOT train or evaluate anything - purely illustrative:

  1. Prints the exponential half-life decay table for a fixed set of ages
     (0/7/30/60/120/210 days) at the configured half-life.
  2. Picks 3 real users from `data/sqlite/backend_shaped_synthetic.db` with
     a genuine old-vs-recent interaction spread and prints their
     category/brand affinity with recency DISABLED vs ENABLED, side by
     side, so the "recent interactions dominate appropriately, old ones
     still contribute but less" property (section 23) is directly
     inspectable rather than just asserted by unit tests.

Reference time: this dataset is static (generated once, not live), so
there is no meaningful "now" to use as the recency reference the way live
serving would. This script uses the dataset's own LATEST `User_events.
action_time` as the reference point instead (documented, not wall-clock)
- i.e. "as of the most recent activity in this dataset." This is only a
diagnostic convention; the temporal evaluation script
(`scripts/sqlite_recency_evaluation.py`) uses the real, leakage-safe
per-user evaluation cutoff instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation.data.adapters.engagement import build_engagement_profile
from recommendation.data.adapters.sqlite_factory import build_sqlite_adapters
from recommendation.data.sqlite.connection import open_readonly_connection
from recommendation.data.sqlite.loader import load_events
from recommendation.features.recency import recency_weight
from recommendation.features.user_features import build_user_features
from recommendation.utils.config import RecencyConfig, get_config

AGES_DAYS = [0, 7, 30, 60, 120, 210]


def print_decay_table(half_life_days: float) -> None:
    print("=" * 78)
    print(f"DECAY TABLE (half_life_days={half_life_days})")
    print("=" * 78)
    print(f"{'age_days':>10}  {'weight':>10}")
    for age in AGES_DAYS:
        # A pure math illustration - not tied to any real event, so any
        # fixed reference_time works; use age directly as (reference - event).
        from datetime import datetime, timedelta

        ref = datetime(2026, 1, 1)
        w = recency_weight(ref - timedelta(days=age), ref, half_life_days)
        print(f"{age:>10}  {w:>10.4f}")
    print()


def main() -> None:
    config = get_config()
    half_life = config.features.recency.half_life_days
    print_decay_table(half_life)

    bundle = build_sqlite_adapters()
    con = open_readonly_connection(config.paths.data_sqlite)
    try:
        events = load_events(con)
    finally:
        con.close()

    reference_time = max(e.action_time for e in events if e.action_time is not None)
    print(f"reference_time (dataset's latest User_events.action_time): {reference_time.isoformat()}")
    print()

    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    product_embeddings: dict[int, "object"] = {}  # embeddings not needed for affinity-only diagnostics

    events_by_user: dict[int, list] = {}
    for e in events:
        events_by_user.setdefault(e.user_id, []).append(e)

    # Pick 3 users with a genuine old-vs-recent spread: at least one event
    # older than 3 half-lives AND at least one event within one half-life
    # of the reference time, so the "recent dominates, old still
    # contributes" contrast is actually visible.
    candidates = []
    for uid, user_events in events_by_user.items():
        times = [e.action_time for e in user_events if e.action_time is not None]
        if not times:
            continue
        oldest_age = (reference_time - min(times)).days
        newest_age = (reference_time - max(times)).days
        if oldest_age >= 3 * half_life and newest_age <= half_life and len(user_events) >= 5:
            candidates.append(uid)
        if len(candidates) >= 3:
            break

    print("=" * 78)
    print(f"EXAMPLE USERS (n={len(candidates)})")
    print("=" * 78)

    for uid in candidates:
        profile = build_engagement_profile(
            uid, bundle.users, bundle.purchases, bundle.cart, bundle.clicks, bundle.search, bundle.chatbot, bundle.reviews
        )
        baseline_config = config.features.model_copy(update={"recency": RecencyConfig(enabled=False)})
        recency_config = config.features.model_copy(update={"recency": RecencyConfig(enabled=True, half_life_days=half_life)})

        baseline = build_user_features(profile, product_lookup, product_embeddings, baseline_config, reference_time=reference_time)
        recency = build_user_features(profile, product_lookup, product_embeddings, recency_config, reference_time=reference_time)

        user_events_sorted = sorted((e for e in events_by_user[uid] if e.action_time is not None), key=lambda e: e.action_time)
        print(f"\n--- USER {uid} ({len(user_events_sorted)} timestamped events) ---")
        oldest, newest = user_events_sorted[0], user_events_sorted[-1]
        print(f"  oldest event: {oldest.action_type.value} product_id={oldest.product_id} age={(reference_time - oldest.action_time).days}d")
        print(f"  newest event: {newest.action_type.value} product_id={newest.product_id} age={(reference_time - newest.action_time).days}d")
        print(f"  category_affinity WITHOUT recency: {_fmt(baseline.category_affinity)}")
        print(f"  category_affinity WITH    recency: {_fmt(recency.category_affinity)}")
        print(f"  brand_affinity    WITHOUT recency: {_fmt(baseline.brand_affinity)}")
        print(f"  brand_affinity    WITH    recency: {_fmt(recency.brand_affinity)}")


def _fmt(d: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.3f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    main()
