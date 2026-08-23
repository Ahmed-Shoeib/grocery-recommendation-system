"""Recency (time-decay) weighting for historical engagement events.

Implements the single formula this phase is scoped to:

    recency_weight = 0.5 ** (age_days / half_life_days)

age = 0                -> weight = 1.0
age = half_life_days    -> weight = 0.5
age = 2 * half_life_days -> weight = 0.25
age -> infinity          -> weight -> 0 (asymptotic, never exactly 0)

`age_days` is always measured against a caller-supplied `reference_time`,
never wall-clock "now" implicitly - see `effective_weight`'s docstring and
`docs/data-mapping.md` section 14. For offline (temporal future-purchase)
evaluation, `reference_time` MUST be the evaluation cutoff so an
interaction's "age" can never be computed relative to a point in time
after the cutoff (that would leak future information into how strongly a
pre-cutoff event counts). For live serving, `reference_time` is the
request/current time.

Naive `datetime`s throughout (no `tzinfo`), matching every other timestamp
already flowing through this codebase (`data.sqlite.loader._parse_timestamp`,
`evaluation.temporal_future_purchase`'s own cutoffs) - callers must not mix
naive and timezone-aware datetimes here.
"""

from __future__ import annotations

from datetime import datetime

from recommendation.utils.config import RecencyConfig


class RecencyLeakageError(ValueError):
    """Raised when `recency_weight` is asked to score an event whose
    `event_time` is at or after `reference_time`. An event can never be
    validly "older" than the point in time it's being weighted relative
    to - if this fires, either a caller forgot to truncate history before
    the evaluation cutoff (a leakage bug) or passed a `reference_time`
    that doesn't actually bound the event set. Deliberately loud (an
    exception, not a clamped weight): a future-relative event must never
    be silently accepted - see `tests/test_recency.py`.
    """


def recency_weight(event_time: datetime, reference_time: datetime, half_life_days: float) -> float:
    """Pure exponential half-life decay. `event_time` must be `<=
    reference_time` (see `RecencyLeakageError`); `half_life_days` must be
    positive. Deterministic: identical inputs always produce identical
    output.
    """
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days!r}")
    age_days = (reference_time - event_time).total_seconds() / 86400.0
    if age_days < 0:
        raise RecencyLeakageError(
            f"event_time={event_time.isoformat()} is after reference_time={reference_time.isoformat()} "
            "(age_days < 0) - recency can never be computed for a future-relative event"
        )
    return 0.5 ** (age_days / half_life_days)


def effective_weight(
    base_weight: float,
    event_time: datetime | None,
    reference_time: datetime | None,
    config: RecencyConfig,
) -> float:
    """`base_weight * recency_weight(...)`, i.e. the
    `effective_weight = signal_weight x recency_weight` formula this phase
    is built around - with the two policy fallbacks feature engineering
    needs on top of the pure math in `recency_weight`:

    - `config.enabled is False`: recency is switched off entirely,
      returns `base_weight` unchanged (byte-for-byte the pre-recency
      behavior).
    - `event_time is None`: neutral weight (per this phase's missing-
      timestamp policy - see module docstring section 14 in
      docs/data-mapping.md) - `base_weight` unchanged. This is what keeps
      the old, timestamp-less synthetic V1 signals (CLICK/ADD_TO_CART/
      SEARCH from the ERD-based synthetic generators) working exactly as
      before; only signals that actually carry a real `action_time`
      (the `User_events`/SQLite-sourced path) are ever decayed.
    - `reference_time is None`: ALSO neutral weight, deliberately - recency
      is opt-in per call, never an implicit wall-clock lookup buried
      inside feature engineering. A caller that wants recency (offline
      evaluation, live serving) must decide and pass an explicit
      `reference_time` (the evaluation cutoff, or `datetime.now()` at the
      live-request boundary - see `serving.pipeline`). This is what keeps
      the pre-existing, non-temporal leave-one-out training path
      (`retrieval.two_tower.examples`/`ranking.examples`, which never
      passes `reference_time`) producing byte-for-byte the same features
      it always has - those training examples have no natural per-example
      "now," and silently defaulting to actual wall-clock time here was
      found (via the existing Two-Tower training test suite) to
      arbitrarily decay historical synthetic order dates relative to
      whatever day the code happens to run on, degrading an otherwise
      unrelated model with no temporal semantics of its own.
    """
    if not config.enabled or event_time is None or reference_time is None:
        return base_weight
    return base_weight * recency_weight(event_time, reference_time, config.half_life_days)
