"""Temporal, future-purchase-based offline evaluation protocol.

Answers "what are the recommendation metrics actually based on?" with:
**future PURCHASE events**, held out strictly by `action_time`, evaluated
against recommendations built from only the history that existed before
the held-out purchase(s) happened.

This is a SEPARATE, ADDITIONAL protocol alongside the existing V1 leave-
one-out-by-product-id protocol (`retrieval.two_tower.splitting`), not a
replacement for it - that protocol is what the currently-trained Two-Tower/
ranker artifacts were fit against and remains the one used by
`scripts/train_two_tower.py`/`train_ranker.py`/`run_pipeline.py`. This
module targets `data/sqlite/backend_shaped_synthetic.db` (via the SQLite
adapters, `data.adapters.sqlite_factory`), which has real per-event
timestamps the original protocol was explicitly designed to work without
(see docs/data-mapping.md section 1/7/12's "V1 leakage limitation - no
timestamps").

INPUT SIGNALS vs. TARGET (docs/data-mapping.md - see this module's
section in that file for the full writeup):
  - INPUT to the point-in-time user representation: CLICK, SEARCH,
    CHATBOT, ADD_TO_CART, and *historical* PURCHASE events (anything with
    `action_time` strictly before the evaluation cutoff).
  - TARGET (what "relevant" means for every metric in this module):
    PURCHASE events at/after the cutoff, up to the next split boundary.
  CLICK/SEARCH/CHATBOT/ADD_TO_CART are never evaluation targets here -
  they describe intent/history, not the offline relevance judgment.

Leakage mechanism (deliberately simpler and MORE PRECISE than the non-
temporal protocol's `exclude_product_ids`-by-product-id guard): because
every event now has a real `action_time`, history is truncated by TIME
(`action_time < cutoff`), not by product id. This means:
  - The held-out purchase itself is excluded automatically (its own
    `action_time` is never `< cutoff` for its own cutoff).
  - A genuinely earlier interaction with the SAME product (a search,
    click, or earlier purchase of that product before the cutoff) is
    correctly KEPT - exactly the "legitimate earlier interaction" /
    repeat-purchase behavior real grocery recommendation needs (docs/
    data-mapping.md section 4's SQLite integration writeup and the
    confirmed grocery repeat-purchase requirement) - see
    `build_point_in_time_engagement_profile`.
  - No product-id exclusion is applied at all for this protocol; product
    ids are never "forbidden" from being recommended merely because the
    user interacted with (or even purchased) them before the cutoff.

Same-timestamp policy (deterministic, does not depend on row order):
history uses strict `<`; the target/relevant-set window uses `>=`. An
event with `action_time == cutoff` is therefore NEVER part of history and
IS a candidate for the target window - see `events_before_cutoff` /
`purchase_targets_in_window`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from recommendation.data.adapters.base import ReviewAdapter, UserAdapter
from recommendation.data.adapters.user_events_adapter import UserEventsAdapter
from recommendation.data.adapters.engagement import build_engagement_profile
from recommendation.data.schemas.engagement import EngagementProfile
from recommendation.data.schemas.events import ActionType, UserInteraction
from recommendation.features.product_features import ProductFeatures
from recommendation.serving.eligibility import EligibilityRule, apply_eligibility

DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT = 3


class TemporalEligibilityTier(str, Enum):
    """Classification of a user's readiness for this protocol - distinct
    from (but reportable alongside) `serving.cold_start.HistoryTier`,
    which measures overall engagement volume rather than purchase-holdout
    depth specifically.
    """

    FULL = "full"  # enough purchase events for train + val + test
    VAL_ONLY = "val_only"  # exactly one holdout-worthy purchase event beyond train
    INSUFFICIENT_DEPTH = "insufficient_depth"  # has purchase(s), not enough to hold any out
    ENGAGEMENT_NO_PURCHASE = "engagement_no_purchase"  # other signals present, zero purchases
    NO_HISTORY = "no_history"  # zero events of any kind


@dataclass(frozen=True)
class TemporalUserSplit:
    user_id: int
    tier: TemporalEligibilityTier
    val_cutoff: datetime | None
    test_cutoff: datetime | None
    val_target_ids: frozenset[int]
    test_target_ids: frozenset[int]
    train_purchase_event_count: int  # purchase events strictly before the earliest active cutoff
    total_purchase_event_count: int

    @property
    def is_val_evaluable(self) -> bool:
        return self.val_cutoff is not None and bool(self.val_target_ids)

    @property
    def is_test_evaluable(self) -> bool:
        return self.test_cutoff is not None and bool(self.test_target_ids)


def group_events_by_user(events: list[UserInteraction]) -> dict[int, list[UserInteraction]]:
    by_user: dict[int, list[UserInteraction]] = defaultdict(list)
    for e in events:
        by_user[e.user_id].append(e)
    for user_events in by_user.values():
        user_events.sort(key=lambda e: (e.action_time is None, e.action_time))
    return dict(by_user)


def _purchase_times(events: list[UserInteraction]) -> list[datetime]:
    """Sorted ascending `action_time` of every PURCHASE event with a
    non-null timestamp (all synthetic/SQLite-sourced events have one; a
    real-backend event with a null `action_time` is defensively skipped
    here rather than treated as sortable).
    """
    times = [e.action_time for e in events if e.action_type == ActionType.PURCHASE and e.action_time is not None]
    return sorted(times)


def build_temporal_splits(
    events_by_user: dict[int, list[UserInteraction]],
    user_ids: list[int],
    min_purchase_events_for_full_split: int = DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT,
) -> dict[int, TemporalUserSplit]:
    """Deterministic (no RNG - chronological order is a total order given
    by the data itself, not a random assignment) per-user temporal split.

    For a user with purchase-event timestamps `t_1 <= t_2 <= ... <= t_n`
    (ties possible - see module docstring's same-timestamp policy):

      n >= min_purchase_events_for_full_split (default 3):
        test_cutoff = t_n ; val_cutoff = t_{n-1}
        val_target_ids   = distinct product_ids of PURCHASE events with
                            val_cutoff <= action_time < test_cutoff
        test_target_ids  = distinct product_ids of PURCHASE events with
                            action_time >= test_cutoff
        (val and test target windows are disjoint by construction, so a
        product purchased only once never counts as relevant for both)

      n == 2:
        val_cutoff = t_2 (the later of the two) ; no test_cutoff
        val_target_ids = product_ids of PURCHASE events with
                          action_time >= val_cutoff

      n == 1: INSUFFICIENT_DEPTH - the single purchase remains available
        as history/input but is never held out (holding out a user's only
        purchase would leave nothing to build a "before" representation
        from, mirroring the non-temporal protocol's
        `min_distinct_products_for_holdout` guard).

      n == 0: ENGAGEMENT_NO_PURCHASE if the user has any other event,
        else NO_HISTORY.
    """
    splits: dict[int, TemporalUserSplit] = {}

    for user_id in user_ids:
        events = events_by_user.get(user_id, [])
        purchase_times = _purchase_times(events)
        n = len(purchase_times)

        if n == 0:
            tier = TemporalEligibilityTier.ENGAGEMENT_NO_PURCHASE if events else TemporalEligibilityTier.NO_HISTORY
            splits[user_id] = TemporalUserSplit(
                user_id=user_id, tier=tier, val_cutoff=None, test_cutoff=None,
                val_target_ids=frozenset(), test_target_ids=frozenset(),
                train_purchase_event_count=0, total_purchase_event_count=0,
            )
            continue

        if n == 1:
            splits[user_id] = TemporalUserSplit(
                user_id=user_id, tier=TemporalEligibilityTier.INSUFFICIENT_DEPTH,
                val_cutoff=None, test_cutoff=None, val_target_ids=frozenset(), test_target_ids=frozenset(),
                train_purchase_event_count=1, total_purchase_event_count=1,
            )
            continue

        if n == 2:
            val_cutoff = purchase_times[-1]
            val_targets = purchase_targets_in_window(events, val_cutoff, None)
            train_count = sum(1 for t in purchase_times if t < val_cutoff)
            splits[user_id] = TemporalUserSplit(
                user_id=user_id, tier=TemporalEligibilityTier.VAL_ONLY,
                val_cutoff=val_cutoff, test_cutoff=None,
                val_target_ids=val_targets, test_target_ids=frozenset(),
                train_purchase_event_count=train_count, total_purchase_event_count=n,
            )
            continue

        test_cutoff = purchase_times[-1]
        val_cutoff = purchase_times[-2]
        val_targets = purchase_targets_in_window(events, val_cutoff, test_cutoff)
        test_targets = purchase_targets_in_window(events, test_cutoff, None)
        train_count = sum(1 for t in purchase_times if t < val_cutoff)
        splits[user_id] = TemporalUserSplit(
            user_id=user_id, tier=TemporalEligibilityTier.FULL,
            val_cutoff=val_cutoff, test_cutoff=test_cutoff,
            val_target_ids=val_targets, test_target_ids=test_targets,
            train_purchase_event_count=train_count, total_purchase_event_count=n,
        )

    return splits


def purchase_targets_in_window(
    events: list[UserInteraction], lower_cutoff: datetime, upper_cutoff: datetime | None
) -> frozenset[int]:
    """Distinct product_ids of PURCHASE events with
    `lower_cutoff <= action_time < upper_cutoff` (or no upper bound when
    `upper_cutoff is None`). This is the "relevant set" for one evaluation
    point - deliberately a SET, not a single id, so a user who purchased
    more than one product in the target window produces a multi-item
    relevant set (section 10/15 of the driving spec) without any special
    casing downstream: `evaluation.retrieval_metrics`'s functions already
    accept `relevant_ids: set[int]`.
    """
    ids: set[int] = set()
    for e in events:
        if e.action_type != ActionType.PURCHASE or e.action_time is None:
            continue
        if e.action_time < lower_cutoff:
            continue
        if upper_cutoff is not None and e.action_time >= upper_cutoff:
            continue
        ids.add(e.product_id)
    return frozenset(ids)


def events_before_cutoff(events: list[UserInteraction], cutoff: datetime) -> list[UserInteraction]:
    """Strict `<` - see module docstring's same-timestamp policy. Events
    with a `None` action_time are excluded (never usable as "known before
    cutoff" if we don't actually know when they happened).
    """
    return [e for e in events if e.action_time is not None and e.action_time < cutoff]


def build_point_in_time_engagement_profile(
    user_id: int,
    events: list[UserInteraction],
    cutoff: datetime,
    users_adapter: UserAdapter,
    reviews_adapter: ReviewAdapter,
) -> EngagementProfile:
    """The core point-in-time construction: truncate the RAW event list to
    `action_time < cutoff` FIRST, then build the canonical per-signal
    records from that truncated list via the EXISTING, unmodified
    `UserEventsAdapter` + `build_engagement_profile`.

    Truncating at the raw-event level (rather than trying to truncate an
    already-built `EngagementProfile`) is what makes this correct for
    CHATBOT in particular: `ChatbotContextRecord` aggregates every
    resolved mention into one record with no per-mention timestamp (a
    known, documented V1 simplification - see `data.schemas.engagement
    .ChatbotContextRecord`'s docstring). Truncating the events BEFORE that
    aggregate is built means the aggregate itself only ever reflects
    pre-cutoff mentions - no separate per-mention timestamp is needed.

    `reviews_adapter` is passed through UNFILTERED (not cutoff-truncated):
    `features.user_features.build_user_features` never reads
    `EngagementProfile.reviews` at all (only catalog-wide product-side
    rating aggregates use reviews, a pre-existing, already-global,
    out-of-scope-for-this-per-user-protocol characteristic - see this
    module's top-of-file docs section reference), so there is no leakage
    surface here to guard.
    """
    truncated = events_before_cutoff(events, cutoff)
    events_adapter = UserEventsAdapter(truncated)
    return build_engagement_profile(
        user_id, users_adapter, events_adapter, events_adapter, events_adapter, events_adapter, events_adapter, reviews_adapter
    )


@dataclass
class TargetEligibilityResult:
    eligible_ids: frozenset[int]
    ineligible_ids: frozenset[int]
    ineligible_reasons: dict[int, list[str]]


def split_targets_by_eligibility(
    target_ids: frozenset[int], product_features: dict[int, ProductFeatures], rules: list[EligibilityRule]
) -> TargetEligibilityResult:
    """A future-purchase target the hard pre-retrieval eligibility gate
    would have excluded (inactive / out of stock, using the ONLY catalog
    snapshot this synthetic SQLite dataset has - see docs/data-mapping.md,
    this module's section, on why there's no historical stock snapshot to
    use instead) was structurally impossible for the recommender to
    surface, regardless of model quality. Reporting these separately
    (rather than silently letting them count as ordinary misses) is what
    section 18 of the driving spec calls a fair policy - see this
    function's caller in `scripts/sqlite_temporal_dry_run.py`.
    """
    result = apply_eligibility(list(target_ids), product_features, rules)
    return TargetEligibilityResult(
        eligible_ids=frozenset(result.eligible_ids),
        ineligible_ids=frozenset(result.excluded_ids),
        ineligible_reasons=result.excluded_reasons,
    )


@dataclass
class LeakageAuditResult:
    user_id: int
    cutoff: datetime
    ok: bool
    violations: list[str]


def audit_no_leakage(
    user_id: int, historical_events: list[UserInteraction], cutoff: datetime, target_ids: frozenset[int]
) -> LeakageAuditResult:
    """Programmatic proof (not just a claim) that one (user, cutoff)
    evaluation point is leakage-safe.

    Deliberately audits `historical_events` - whatever was ACTUALLY
    handed to feature construction for this evaluation example (typically
    `events_before_cutoff(...)`'s own output, but callers may pass
    anything, including a deliberately-broken list in a test) - rather
    than recomputing the filter internally. Recomputing the same filter
    and comparing it to itself would only prove the function agrees with
    itself, not that the events actually used were safe; auditing the
    real input catches a bug anywhere upstream (a caller forgetting to
    truncate, a future refactor of `events_before_cutoff` that stops
    filtering correctly, a temporal split boundary computed wrong), which
    is the actual point of an audit per section 31 of the driving spec.

    Checks, per event in `historical_events`:
      1. `action_time` is not None (an event with no known time is never
         legitimately "known before the cutoff").
      2. `action_time < cutoff` strictly (the same-timestamp policy - see
         module docstring).

    Does NOT flag a historical PURCHASE of a target product_id as a
    violation by itself - an earlier, genuinely-prior purchase of the
    product that later becomes a held-out target is legitimate repeat-
    purchase history (section 7/9); only an out-of-window `action_time`
    is ever a violation, regardless of which product_id it references.
    """
    violations: list[str] = []
    for e in historical_events:
        if e.action_time is None:
            violations.append(f"event product_id={e.product_id} action_type={e.action_type.value} has no action_time")
        elif e.action_time >= cutoff:
            is_target = " [TARGET PRODUCT]" if e.product_id in target_ids else ""
            violations.append(
                f"event product_id={e.product_id} action_type={e.action_type.value} "
                f"action_time={e.action_time.isoformat()} >= cutoff={cutoff.isoformat()}{is_target}"
            )

    return LeakageAuditResult(user_id=user_id, cutoff=cutoff, ok=not violations, violations=violations)
