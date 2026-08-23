"""Dry-run report for the temporal future-purchase evaluation protocol
against `data/sqlite/backend_shaped_synthetic.db`.

Does NOT train, retrain, or touch any model artifact - this only exercises
`recommendation.evaluation.temporal_future_purchase`'s split construction,
point-in-time truncation, eligibility partitioning, and leakage audit
against the real (entirely synthetic) SQLite dataset, and reports the
resulting statistics, example timelines, and audit results.

Computing actual Top-N recommendation metrics (Recall/NDCG/etc. from a
real ranked list) is intentionally not part of this script: doing so
meaningfully would require the Two-Tower item tower to encode this
database's 1,200-product catalog and a VectorIndex built over those
embeddings - the currently-trained artifacts under `models/` were fit
against the original, smaller synthetic catalog (different product ids,
different brand vocabulary), so rebuilding the index for this dataset is
out of scope here. The metric functions themselves (`evaluation
.retrieval_metrics`) are proven correct against hand-computed worked
examples in `tests/test_retrieval_metrics.py`, including a multi-relevant-
item case - they are ready to be plugged into real Top-N output once
retraining happens against this catalog.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation.data.adapters.sqlite_factory import build_sqlite_adapters
from recommendation.data.sqlite.connection import open_readonly_connection
from recommendation.data.sqlite.loader import load_events
from recommendation.evaluation.temporal_future_purchase import (
    DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT,
    TemporalEligibilityTier,
    audit_no_leakage,
    build_point_in_time_engagement_profile,
    build_temporal_splits,
    events_before_cutoff,
    group_events_by_user,
    split_targets_by_eligibility,
)
from recommendation.features.product_features import build_product_features
from recommendation.serving.eligibility import build_eligibility_rules
from recommendation.utils.config import get_config


def main() -> None:
    config = get_config()
    bundle = build_sqlite_adapters()

    con = open_readonly_connection(config.paths.data_sqlite)
    try:
        all_events = load_events(con)
    finally:
        con.close()

    events_by_user = group_events_by_user(all_events)
    user_ids = bundle.users.list_user_ids()
    splits = build_temporal_splits(events_by_user, user_ids, DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT)

    products = bundle.products.list_products()
    product_features = build_product_features(products, [], [], [])
    eligibility_rules = build_eligibility_rules(config.eligibility)

    # ================= SECTION 29: dataset / split statistics =================
    print("=" * 78)
    print("SECTION 29: SQLite temporal split statistics")
    print("=" * 78)

    n_purchase_events = sum(1 for e in all_events if e.action_type.value == "PURCHASE")
    purchasing_users = {e.user_id for e in all_events if e.action_type.value == "PURCHASE"}

    tier_counts = Counter(s.tier for s in splits.values())
    val_eligible = [s for s in splits.values() if s.is_val_evaluable]
    test_eligible = [s for s in splits.values() if s.is_test_evaluable]
    excluded = [s for s in splits.values() if s.tier in (TemporalEligibilityTier.ENGAGEMENT_NO_PURCHASE, TemporalEligibilityTier.NO_HISTORY, TemporalEligibilityTier.INSUFFICIENT_DEPTH)]

    train_purchase_count = sum(s.train_purchase_event_count for s in splits.values())
    val_target_count = sum(len(s.val_target_ids) for s in splits.values())
    test_target_count = sum(len(s.test_target_ids) for s in splits.values())

    def _is_repeat(user_id: int, target_ids: frozenset[int], cutoff) -> bool:
        history = events_before_cutoff(events_by_user.get(user_id, []), cutoff)
        purchased_before = {e.product_id for e in history if e.action_type.value == "PURCHASE"}
        return bool(target_ids & purchased_before)

    repeat_val = sum(1 for s in splits.values() if s.is_val_evaluable and _is_repeat(s.user_id, s.val_target_ids, s.val_cutoff))
    repeat_test = sum(1 for s in splits.values() if s.is_test_evaluable and _is_repeat(s.user_id, s.test_target_ids, s.test_cutoff))

    all_val_targets = frozenset().union(*(s.val_target_ids for s in splits.values())) if val_eligible else frozenset()
    all_test_targets = frozenset().union(*(s.test_target_ids for s in splits.values())) if test_eligible else frozenset()
    val_elig = split_targets_by_eligibility(all_val_targets, product_features, eligibility_rules)
    test_elig = split_targets_by_eligibility(all_test_targets, product_features, eligibility_rules)

    times = [e.action_time for e in all_events if e.action_time is not None]
    purchase_times = [e.action_time for e in all_events if e.action_type.value == "PURCHASE" and e.action_time is not None]

    print(f"total users: {len(user_ids)}")
    print(f"total User_events: {len(all_events)}")
    print(f"total PURCHASE events: {n_purchase_events}")
    print(f"unique purchasing users: {len(purchasing_users)}")
    print()
    print("temporal eligibility tier distribution:")
    for tier in TemporalEligibilityTier:
        print(f"  {tier.value:20s} {tier_counts.get(tier, 0)}")
    print()
    print(f"users eligible for validation (is_val_evaluable): {len(val_eligible)}")
    print(f"users eligible for test (is_test_evaluable):       {len(test_eligible)}")
    print(f"users excluded from future-purchase evaluation entirely: {len(excluded)}")
    print()
    print(f"train purchase count (events used as historical input, summed): {train_purchase_count}")
    print(f"validation future-purchase target count (summed across users): {val_target_count}")
    print(f"test future-purchase target count (summed across users):       {test_target_count}")
    print(f"repeat-purchase val targets (product also purchased earlier):  {repeat_val} / {len(val_eligible)} val-evaluable users")
    print(f"repeat-purchase test targets (product also purchased earlier): {repeat_test} / {len(test_eligible)} test-evaluable users")
    print()
    print(f"distinct val target product ids across all users:  {len(all_val_targets)}")
    print(f"  eligible (active+in-stock):   {len(val_elig.eligible_ids)}")
    print(f"  ineligible (structurally impossible to recommend): {len(val_elig.ineligible_ids)}")
    print(f"distinct test target product ids across all users: {len(all_test_targets)}")
    print(f"  eligible (active+in-stock):   {len(test_elig.eligible_ids)}")
    print(f"  ineligible (structurally impossible to recommend): {len(test_elig.ineligible_ids)}")
    print()
    if times:
        print(f"overall event date range: {min(times).isoformat()} .. {max(times).isoformat()}")
    if purchase_times:
        print(f"purchase-only date range: {min(purchase_times).isoformat()} .. {max(purchase_times).isoformat()}")
    if val_eligible:
        val_cutoffs = [s.val_cutoff for s in val_eligible]
        print(f"val cutoff range:  {min(val_cutoffs).isoformat()} .. {max(val_cutoffs).isoformat()}")
    if test_eligible:
        test_cutoffs = [s.test_cutoff for s in test_eligible]
        print(f"test cutoff range: {min(test_cutoffs).isoformat()} .. {max(test_cutoffs).isoformat()}")

    # ================= SECTION 30: five example user timelines =================
    print()
    print("=" * 78)
    print("SECTION 30: example user timelines")
    print("=" * 78)

    product_lookup = {p.id: p for p in products}

    def _describe(uid: int) -> None:
        s = splits[uid]
        events = events_by_user.get(uid, [])
        print(f"\n--- USER {uid} (tier={s.tier.value}) ---")
        cutoff = s.test_cutoff or s.val_cutoff
        if cutoff is None:
            print("  (no evaluable cutoff - showing full event list)")
            visible, target_events = events, []
        else:
            visible = events_before_cutoff(events, cutoff)
            target_events = [e for e in events if e not in visible]
        print("  PAST INPUT (visible to the recommender):")
        for e in visible:
            name = product_lookup[e.product_id].name if e.product_id in product_lookup else "?"
            print(f"    {e.action_time.isoformat() if e.action_time else '?':26s} {e.action_type.value:12s} product_id={e.product_id} ({name})")
        if cutoff is not None:
            print(f"  ---------------- CUT-OFF: {cutoff.isoformat()} ----------------")
            print("  FUTURE (held out):")
            for e in sorted(target_events, key=lambda e: (e.action_time is None, e.action_time)):
                name = product_lookup[e.product_id].name if e.product_id in product_lookup else "?"
                marker = " <= TARGET" if e.action_type.value == "PURCHASE" and e.action_time and e.action_time >= cutoff else ""
                print(f"    {e.action_time.isoformat() if e.action_time else '?':26s} {e.action_type.value:12s} product_id={e.product_id} ({name}){marker}")

    used_uids: set[int] = set()

    def _pick(predicate) -> int | None:
        for s in splits.values():
            if s.user_id in used_uids:
                continue
            if predicate(s):
                used_uids.add(s.user_id)
                return s.user_id
        return None

    repeat_example = _pick(lambda s: s.tier == TemporalEligibilityTier.FULL and _is_repeat(s.user_id, s.test_target_ids, s.test_cutoff))

    def _has_prior_interaction(s) -> bool:
        if not s.is_test_evaluable:
            return False
        history = events_before_cutoff(events_by_user.get(s.user_id, []), s.test_cutoff)
        return any(e.product_id in s.test_target_ids and e.action_type.value in ("SEARCH", "CLICK", "ADD_TO_CART") for e in history)

    prior_interaction_example = _pick(_has_prior_interaction)
    multi_purchase_example = _pick(lambda s: s.tier == TemporalEligibilityTier.FULL and s.total_purchase_event_count >= 5)
    sparse_example = _pick(lambda s: s.tier == TemporalEligibilityTier.INSUFFICIENT_DEPTH)
    # "Ordinary" = any other FULL-tier user not already used above. Note:
    # essentially every FULL-tier purchase in this dataset has SOME prior
    # funnel interaction (search/chatbot -> click -> cart -> purchase all
    # target the same product per generated session - see
    # scripts/generate_backend_shaped_sqlite.py) so "zero prior
    # interaction on the target product" is not a realistic category to
    # search for separately here; "ordinary" just means "a typical
    # FULL-tier user, not specifically chosen to also repeat-purchase or
    # have >=5 purchases."
    ordinary = _pick(lambda s: s.tier == TemporalEligibilityTier.FULL)

    for label, uid in [
        ("ordinary future purchase", ordinary),
        ("repeat purchase as target", repeat_example),
        ("prior click/search/cart on future-purchased item", prior_interaction_example),
        ("multiple purchases (>=5 purchase events)", multi_purchase_example),
        ("sparse / insufficient-history user", sparse_example),
    ]:
        print(f"\n[{label}]")
        if uid is None:
            print("  no matching example found in this dataset")
        else:
            _describe(uid)

    # ================= SECTION 31: leakage audit =================
    print()
    print("=" * 78)
    print("SECTION 31: leakage audit")
    print("=" * 78)

    total_checked = 0
    total_failed = 0
    failures = []
    for s in splits.values():
        for cutoff, target_ids in ((s.val_cutoff, s.val_target_ids), (s.test_cutoff, s.test_target_ids)):
            if cutoff is None:
                continue
            history = events_before_cutoff(events_by_user.get(s.user_id, []), cutoff)
            result = audit_no_leakage(s.user_id, history, cutoff, target_ids)
            total_checked += 1
            if not result.ok:
                total_failed += 1
                failures.append(result)

    print(f"evaluation points audited: {total_checked}")
    print(f"violations found: {total_failed}")
    for f in failures[:10]:
        print(f"  user={f.user_id} cutoff={f.cutoff} violations={f.violations}")
    print("AUDIT RESULT:", "PASS - no leakage detected" if total_failed == 0 else "FAIL - leakage detected, see above")


if __name__ == "__main__":
    main()
