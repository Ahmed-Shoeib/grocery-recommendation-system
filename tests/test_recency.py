"""Tests for `recommendation.features.recency` - the exponential
half-life decay math and its `effective_weight` feature-integration
wrapper. See docs/data-mapping.md section 14.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from recommendation.features.recency import (
    RecencyLeakageError,
    effective_weight,
    recency_weight,
)
from recommendation.utils.config import RecencyConfig

T0 = datetime(2026, 1, 1, 0, 0, 0)


def t(days: float) -> datetime:
    return T0 + timedelta(days=days)


# --- recency_weight: pure decay math -------------------------------------

def test_age_zero_gives_weight_one():
    assert recency_weight(T0, T0, half_life_days=21.0) == pytest.approx(1.0)


def test_age_equal_half_life_gives_weight_half():
    assert recency_weight(t(-21), T0, half_life_days=21.0) == pytest.approx(0.5, rel=1e-9)


def test_age_two_half_lives_gives_weight_quarter():
    assert recency_weight(t(-42), T0, half_life_days=21.0) == pytest.approx(0.25, rel=1e-9)


def test_age_three_half_lives_gives_weight_eighth():
    assert recency_weight(t(-63), T0, half_life_days=21.0) == pytest.approx(0.125, rel=1e-9)


def test_very_old_event_has_positive_but_small_weight():
    w = recency_weight(t(-3650), T0, half_life_days=21.0)  # 10 years old
    assert w > 0.0
    assert w < 1e-10


def test_weight_is_monotonically_decreasing_with_age():
    ages = [0, 7, 30, 60, 120, 210]
    weights = [recency_weight(t(-a), T0, half_life_days=21.0) for a in ages]
    assert weights == sorted(weights, reverse=True)
    assert weights[0] == pytest.approx(1.0)
    for w in weights:
        assert 0.0 < w <= 1.0


def test_future_event_relative_to_reference_time_raises():
    with pytest.raises(RecencyLeakageError):
        recency_weight(t(1), T0, half_life_days=21.0)


def test_non_positive_half_life_raises():
    with pytest.raises(ValueError):
        recency_weight(T0, T0, half_life_days=0.0)
    with pytest.raises(ValueError):
        recency_weight(T0, T0, half_life_days=-5.0)


def test_recency_weight_is_deterministic():
    a = recency_weight(t(-14), T0, half_life_days=21.0)
    b = recency_weight(t(-14), T0, half_life_days=21.0)
    assert a == b


# --- effective_weight: feature-integration policy -------------------------

def test_effective_weight_disabled_returns_base_weight_unchanged():
    config = RecencyConfig(enabled=False, half_life_days=21.0)
    assert effective_weight(0.45, t(-365), T0, config) == 0.45


def test_effective_weight_missing_event_time_is_neutral():
    config = RecencyConfig(enabled=True, half_life_days=21.0)
    assert effective_weight(0.45, None, T0, config) == 0.45


def test_effective_weight_missing_reference_time_is_neutral():
    """Recency is opt-in per call - omitting `reference_time` must never
    fall back to wall-clock "now" implicitly (see `user_features
    .build_user_features` docstring: this is what keeps the non-temporal
    leave-one-out training path unaffected by recency).
    """
    config = RecencyConfig(enabled=True, half_life_days=21.0)
    assert effective_weight(0.45, t(-365), None, config) == 0.45


def test_effective_weight_applies_decay_when_fully_configured():
    config = RecencyConfig(enabled=True, half_life_days=21.0)
    w = effective_weight(0.45, t(-21), T0, config)
    assert w == pytest.approx(0.45 * 0.5, rel=1e-9)


def test_effective_weight_future_event_raises():
    config = RecencyConfig(enabled=True, half_life_days=21.0)
    with pytest.raises(RecencyLeakageError):
        effective_weight(0.45, t(1), T0, config)


def test_effective_weight_recent_beats_old_for_identical_base_weight():
    config = RecencyConfig(enabled=True, half_life_days=21.0)
    recent = effective_weight(0.5, t(-1), T0, config)
    old = effective_weight(0.5, t(-180), T0, config)
    assert recent > old
