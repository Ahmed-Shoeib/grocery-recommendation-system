"""Tests for the freshness fix: `RecommendationService.maybe_refresh` and
the `_load_data_snapshot` helper it shares with `build_recommendation_service`.

`test_load_data_snapshot_reflects_new_user_events_row_without_restart` is
the core proof requested for this fix - it runs against a throwaway COPY
of the real, committed `data/sqlite/backend_shaped_synthetic.db` (never
the committed file itself), inserts a new `User_events` row through a
separate connection (simulating the backend team's write path), and shows
a fresh `_load_data_snapshot` call picks it up with no service restart.

The remaining tests exercise `maybe_refresh`'s TTL gating, the
`interval_seconds<=0` disable switch, and the lock that prevents two
threads from reloading concurrently - using a monkeypatched loader so
they run fast and never touch a real database or model.
"""

from __future__ import annotations

import shutil
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from recommendation.api.dependencies import RecommendationService, _DataSnapshot, _load_data_snapshot
import recommendation.api.dependencies as dependencies_module
from recommendation.api.errors import UnknownUserError
from recommendation.features.price import PriceCatalogContext
from recommendation.utils.config import AppConfig, PathsConfig, RefreshConfig, get_config, resolve_path

DB_PATH = resolve_path(get_config().paths.data_sqlite)
pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="backend_shaped_synthetic.db not present")


def _fake_encoder() -> SimpleNamespace:
    # `_load_data_snapshot` only ever touches `.model_name` (cache-validity
    # check) and `.encode(...)` (only called on a cache miss, or when a
    # user has free-text search/chatbot content - never true for the
    # User_events-sourced dataset, see data/adapters/user_events_adapter.py).
    # The committed embedding cache is content-hash-valid for the unmodified
    # catalog, so this stub never needs a real Sentence Transformer model.
    return SimpleNamespace(model_name="all-MiniLM-L6-v2")


def _fake_snapshot(bundle=None) -> _DataSnapshot:
    if bundle is None:
        bundle = SimpleNamespace(users=SimpleNamespace(list_user_ids=lambda: [1]))
    return _DataSnapshot(
        bundle=bundle,
        product_lookup={},
        product_features={},
        product_embeddings={},
        text_embeddings={},
        price_context=PriceCatalogContext(0.0, 0.0, (0.0, 0.0)),
        engagement_profiles={},
    )


def _minimal_service(config: AppConfig, sentence_encoder=None) -> RecommendationService:
    return RecommendationService(
        product_lookup={},
        product_features={},
        product_embeddings={},
        text_embeddings={},
        all_item_ids=[],
        tt_encoder=None,
        user_tower=None,
        ranker_model=None,
        vector_index=None,
        bundle=SimpleNamespace(users=SimpleNamespace(list_user_ids=lambda: [1])),
        config=config,
        sentence_encoder=sentence_encoder if sentence_encoder is not None else _fake_encoder(),
    )


# --- the core freshness proof ------------------------------------------------

def test_load_data_snapshot_reflects_new_user_events_row_without_restart(tmp_path):
    test_db = tmp_path / "freshness_test.db"
    shutil.copyfile(DB_PATH, test_db)

    config = AppConfig(paths=PathsConfig(data_sqlite=str(test_db), data_source="sqlite"))
    encoder = _fake_encoder()

    snapshot_before = _load_data_snapshot(config, encoder)
    user_id = next(
        uid for uid in snapshot_before.bundle.users.list_user_ids() if snapshot_before.bundle.purchases.get_purchases(uid)
    )
    purchases_before = snapshot_before.bundle.purchases.get_purchases(user_id)
    already_purchased = {p.product_id for p in purchases_before}
    new_product_id = next(pid for pid, product in snapshot_before.product_lookup.items() if pid not in already_purchased)

    # A SEPARATE connection, simulating the backend team's own write path
    # into User_events - never through this service's read-only connection.
    writer = sqlite3.connect(test_db)
    try:
        writer.execute(
            "INSERT INTO User_events (user_id, product_id, action_time, action_type) VALUES (?, ?, datetime('now'), 'PURCHASE')",
            (user_id, new_product_id),
        )
        writer.commit()
    finally:
        writer.close()

    snapshot_after = _load_data_snapshot(config, encoder)
    purchases_after = snapshot_after.bundle.purchases.get_purchases(user_id)

    assert len(purchases_after) == len(purchases_before) + 1
    assert new_product_id in {p.product_id for p in purchases_after}


# --- maybe_refresh gating ----------------------------------------------------

def test_maybe_refresh_skips_when_ttl_not_elapsed(monkeypatch):
    calls = []
    monkeypatch.setattr(dependencies_module, "_load_data_snapshot", lambda *a, **k: calls.append(1) or _fake_snapshot())

    config = AppConfig(refresh=RefreshConfig(interval_seconds=9999))
    service = _minimal_service(config)

    assert service.maybe_refresh() is False
    assert calls == []


def test_maybe_refresh_force_bypasses_ttl_and_updates_fields(monkeypatch):
    calls = []
    new_bundle = SimpleNamespace(users=SimpleNamespace(list_user_ids=lambda: [1, 2, 3]))

    def fake_loader(config, encoder):
        calls.append(1)
        return _fake_snapshot(bundle=new_bundle)

    monkeypatch.setattr(dependencies_module, "_load_data_snapshot", fake_loader)

    config = AppConfig(refresh=RefreshConfig(interval_seconds=9999))
    service = _minimal_service(config)
    old_bundle = service.bundle

    assert service.maybe_refresh(force=True) is True
    assert calls == [1]
    assert service.bundle is new_bundle
    assert service.bundle is not old_bundle


def test_maybe_refresh_disabled_when_interval_non_positive(monkeypatch):
    calls = []
    monkeypatch.setattr(dependencies_module, "_load_data_snapshot", lambda *a, **k: calls.append(1) or _fake_snapshot())

    config = AppConfig(refresh=RefreshConfig(interval_seconds=0))
    service = _minimal_service(config)

    assert service.maybe_refresh() is False
    assert calls == []
    # force still works even when periodic refresh is disabled by config.
    assert service.maybe_refresh(force=True) is True
    assert calls == [1]


def test_maybe_refresh_reloads_once_ttl_elapses(monkeypatch):
    calls = []
    monkeypatch.setattr(dependencies_module, "_load_data_snapshot", lambda *a, **k: calls.append(1) or _fake_snapshot())

    config = AppConfig(refresh=RefreshConfig(interval_seconds=0.05))
    service = _minimal_service(config)

    assert calls == []
    time.sleep(0.1)
    assert service.maybe_refresh() is True
    assert calls == [1]


def test_maybe_refresh_skips_synthetic_data_source(monkeypatch):
    # The synthetic generator is deterministic (fixed seed) - there is no
    # live User_events source behind it, so a non-forced refresh is a
    # deliberate no-op rather than wasted work.
    calls = []
    monkeypatch.setattr(dependencies_module, "_load_data_snapshot", lambda *a, **k: calls.append(1) or _fake_snapshot())

    config = AppConfig(paths=PathsConfig(data_source="synthetic"), refresh=RefreshConfig(interval_seconds=0.01))
    service = _minimal_service(config)
    time.sleep(0.05)

    assert service.maybe_refresh() is False
    assert calls == []


def test_recommend_triggers_maybe_refresh(monkeypatch):
    """`recommend()` must call `maybe_refresh()` on every invocation - the
    TTL gating inside `maybe_refresh` itself decides whether a reload
    actually happens.
    """
    config = AppConfig(refresh=RefreshConfig(interval_seconds=9999))
    service = _minimal_service(config)
    calls = []
    monkeypatch.setattr(service, "maybe_refresh", lambda **k: calls.append(k) or False)
    monkeypatch.setattr(service, "is_known_user", lambda uid: False)

    with pytest.raises(UnknownUserError):
        service.recommend(user_id=1, limit=5)

    assert calls == [{}]


# --- thread-safety of the refresh lock ---------------------------------------

def test_maybe_refresh_lock_prevents_concurrent_reloads(monkeypatch):
    state = {"active": False, "overlap_detected": False, "count": 0}
    guard = threading.Lock()

    def slow_loader(config, encoder):
        with guard:
            if state["active"]:
                state["overlap_detected"] = True
            state["active"] = True
            state["count"] += 1
        time.sleep(0.05)
        with guard:
            state["active"] = False
        return _fake_snapshot()

    monkeypatch.setattr(dependencies_module, "_load_data_snapshot", slow_loader)

    config = AppConfig(refresh=RefreshConfig(interval_seconds=9999))
    service = _minimal_service(config)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: service.maybe_refresh(force=True), range(8)))

    assert state["overlap_detected"] is False
    # At least one thread actually reloaded; threads that lost the
    # non-blocking lock race correctly report False rather than blocking.
    assert any(results)
    assert state["count"] >= 1
