"""`backend.activity_cache`: atomic persistence + corruption recovery for
the bounded activity-window cache (docs/data-mapping.md 19.13 - the fix
for the 1.5M-row `/api/ai/user-activities` startup blocker).
"""

from __future__ import annotations

import json

from recommendation.backend.activity_cache import (
    CACHE_FORMAT_VERSION,
    load_activity_cache,
    new_cache_state,
    save_activity_cache,
)


def test_missing_file_returns_none(tmp_path):
    assert load_activity_cache(tmp_path / "nope.json") is None


def test_round_trips_through_save_and_load(tmp_path):
    path = tmp_path / "cache.json"
    state = new_cache_state(
        rows=[{"user_id": "g1", "action_type": "AddToCart", "product_id": 1, "timestamp": "2026-08-01T00:00:00"}],
        high_water_timestamp="2026-08-01T00:00:00",
        boundary_keys={("g1", "AddToCart", 1, None, "2026-08-01T00:00:00")},
    )
    save_activity_cache(path, state)
    loaded = load_activity_cache(path)
    assert loaded is not None
    assert loaded.rows == state.rows
    assert loaded.high_water_timestamp == "2026-08-01T00:00:00"
    assert loaded.boundary_key_set() == {("g1", "AddToCart", 1, None, "2026-08-01T00:00:00")}


def test_save_is_atomic_no_temp_file_left_behind(tmp_path):
    path = tmp_path / "cache.json"
    save_activity_cache(path, new_cache_state([], None, set()))
    leftovers = [p for p in tmp_path.iterdir() if p.name != "cache.json"]
    assert leftovers == []


def test_corrupt_json_degrades_to_none_not_raise(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_activity_cache(path) is None


def test_wrong_top_level_shape_degrades_to_none(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert load_activity_cache(path) is None


def test_unknown_version_degrades_to_none(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"version": CACHE_FORMAT_VERSION + 1, "fetched_at": "x", "rows": []}), encoding="utf-8")
    assert load_activity_cache(path) is None


def test_missing_required_field_degrades_to_none(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"version": CACHE_FORMAT_VERSION}), encoding="utf-8")
    assert load_activity_cache(path) is None


def test_a_second_save_fully_replaces_the_first(tmp_path):
    path = tmp_path / "cache.json"
    save_activity_cache(path, new_cache_state([{"a": 1}], "t1", set()))
    save_activity_cache(path, new_cache_state([{"a": 2}], "t2", set()))
    loaded = load_activity_cache(path)
    assert loaded.rows == [{"a": 2}]
    assert loaded.high_water_timestamp == "t2"
