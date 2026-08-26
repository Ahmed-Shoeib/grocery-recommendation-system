"""Focused tests for the timestamp/timezone fix.

`data.sqlite.loader._parse_timestamp` and `serving.pipeline.recommend`'s
`reference_time` construction now share one explicit convention: a naive
datetime (no `tzinfo`) always represents UTC wall-clock time; a value that
DOES carry explicit offset/`Z` info is converted to UTC first. Before this
fix, `serving.pipeline.recommend` used the server PROCESS's local clock
(`datetime.now()`) as `reference_time` - so a backend writing UTC-
timestamped `User_events` rows (a common backend convention) could have a
brand-new, genuinely fresh event misread as "in the future" purely because
the server machine's own timezone differed from UTC, incorrectly raising
`features.recency.RecencyLeakageError` (observed as an HTTP 500 in
practice - see the manual runtime reproduction in the accompanying report).

Only `_parse_timestamp`, `serving.pipeline.recommend`'s `reference_time`
construction, and their docstrings changed - `features.recency`'s pure
math, `evaluation.temporal_future_purchase`, the database schema, and
every other consumer of these naive datetimes are untouched.
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from recommendation.api.dependencies import _load_data_snapshot
from recommendation.data.sqlite.loader import _parse_timestamp
from recommendation.features.recency import RecencyLeakageError
from recommendation.features.user_features import build_user_features
from recommendation.utils.config import AppConfig, PathsConfig, get_config, resolve_path

DB_PATH = resolve_path(get_config().paths.data_sqlite)
pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="backend_shaped_synthetic.db not present")


def _fake_encoder() -> SimpleNamespace:
    # See tests/test_service_refresh.py for why a stub encoder is safe here
    # (the committed embedding cache is content-hash-valid, and the
    # User_events-sourced dataset never has free-text search/chatbot
    # content, so `.encode(...)` is never actually called).
    return SimpleNamespace(model_name="all-MiniLM-L6-v2")


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# --- _parse_timestamp: pure unit tests --------------------------------------
# Establishes the convention: naive == UTC; explicit offset/Z == converted
# to UTC then stripped; result is always naive either way.

def test_parse_timestamp_naive_string_is_used_as_is():
    assert _parse_timestamp("2026-08-26T15:20:00") == datetime(2026, 8, 26, 15, 20, 0)


def test_parse_timestamp_explicit_utc_offset_is_normalized_to_naive():
    assert _parse_timestamp("2026-08-26T15:20:00+00:00") == datetime(2026, 8, 26, 15, 20, 0)


def test_parse_timestamp_z_suffix_is_normalized_to_naive():
    assert _parse_timestamp("2026-08-26T15:20:00Z") == datetime(2026, 8, 26, 15, 20, 0)


def test_parse_timestamp_non_utc_offset_is_converted_to_true_utc():
    # +02:00 local wall-clock 15:20 -> 13:20 UTC.
    assert _parse_timestamp("2026-08-26T15:20:00+02:00") == datetime(2026, 8, 26, 13, 20, 0)


def test_parse_timestamp_none_stays_none():
    assert _parse_timestamp(None) is None


def test_parse_timestamp_result_is_always_naive():
    for value in ("2026-08-26T15:20:00", "2026-08-26T15:20:00Z", "2026-08-26T15:20:00+05:30"):
        assert _parse_timestamp(value).tzinfo is None


# --- reproduces the exact originally-discovered bug, now fixed --------------

def test_utc_timestamped_fresh_event_does_not_trigger_recency_leakage(tmp_path):
    """The scenario that crashed the running server before this fix: a
    backend writes a `User_events` row using UTC (SQLite's own
    `datetime('now')` is UTC and naive - exactly the common backend
    convention this fix targets), on a machine whose local clock is behind
    UTC (true for roughly half the world's timezones at any given moment).
    """
    test_db = tmp_path / "tz_test.db"
    shutil.copyfile(DB_PATH, test_db)

    # A separate connection - simulating the backend team's own write path,
    # never this service's read-only connection.
    writer = sqlite3.connect(test_db)
    utc_now_naive_str = writer.execute("SELECT datetime('now')").fetchone()[0]
    user_id = writer.execute("SELECT user_id FROM User_events LIMIT 1").fetchone()[0]
    product_id = writer.execute("SELECT Id FROM Product LIMIT 1").fetchone()[0]
    writer.execute(
        "INSERT INTO User_events (user_id, product_id, action_time, action_type) VALUES (?, ?, ?, 'PURCHASE')",
        (user_id, product_id, utc_now_naive_str),
    )
    writer.commit()
    writer.close()

    config = AppConfig(paths=PathsConfig(data_sqlite=str(test_db), data_source="sqlite"))
    snapshot = _load_data_snapshot(config, _fake_encoder())
    profile = snapshot.engagement_profiles[user_id]

    # The exact reference_time construction serving.pipeline.recommend()
    # now uses (naive UTC, not the local `datetime.now()` it used before).
    reference_time = _utc_now_naive()

    # Must not raise RecencyLeakageError - this is what produced the HTTP
    # 500 before the fix.
    features = build_user_features(
        profile, snapshot.product_lookup, snapshot.product_embeddings,
        config.features, text_embeddings=snapshot.text_embeddings, reference_time=reference_time,
        price_context=snapshot.price_context,
    )
    assert features is not None


def test_sqlite_utc_now_insert_round_trips_to_within_a_few_seconds(tmp_path):
    """Confirms the insert used above lands a real, very-recent timestamp
    (not silently truncated/misparsed) by checking it round-trips to
    within a few seconds of the moment it was written.
    """
    test_db = tmp_path / "tz_test2.db"
    shutil.copyfile(DB_PATH, test_db)
    writer = sqlite3.connect(test_db)
    before = _utc_now_naive()
    writer.execute(
        "INSERT INTO User_events (user_id, product_id, action_time, action_type) VALUES (1, 1, datetime('now'), 'PURCHASE')"
    )
    writer.commit()
    row = writer.execute("SELECT action_time FROM User_events ORDER BY id DESC LIMIT 1").fetchone()
    writer.close()
    after = _utc_now_naive()

    parsed = _parse_timestamp(row[0])
    assert before - timedelta(seconds=5) <= parsed <= after + timedelta(seconds=5)


# --- the leakage guard must still catch genuinely future/bad data -----------

def test_genuinely_future_event_is_still_rejected(tmp_path):
    """The fix must stop a correctly-UTC-timestamped FRESH event from being
    misread as future - it must NOT defeat the guard for an event that is
    actually, unambiguously in the future (e.g. bad data, a clock way out
    of sync, or a genuine leakage bug elsewhere).
    """
    test_db = tmp_path / "tz_test3.db"
    shutil.copyfile(DB_PATH, test_db)

    writer = sqlite3.connect(test_db)
    user_id = writer.execute("SELECT user_id FROM User_events LIMIT 1").fetchone()[0]
    product_id = writer.execute("SELECT Id FROM Product LIMIT 1").fetchone()[0]
    far_future = (datetime.now(timezone.utc) + timedelta(days=3)).replace(tzinfo=None).isoformat()
    writer.execute(
        "INSERT INTO User_events (user_id, product_id, action_time, action_type) VALUES (?, ?, ?, 'PURCHASE')",
        (user_id, product_id, far_future),
    )
    writer.commit()
    writer.close()

    config = AppConfig(paths=PathsConfig(data_sqlite=str(test_db), data_source="sqlite"))
    snapshot = _load_data_snapshot(config, _fake_encoder())
    profile = snapshot.engagement_profiles[user_id]
    reference_time = _utc_now_naive()

    with pytest.raises(RecencyLeakageError):
        build_user_features(
            profile, snapshot.product_lookup, snapshot.product_embeddings,
            config.features, text_embeddings=snapshot.text_embeddings, reference_time=reference_time,
            price_context=snapshot.price_context,
        )
