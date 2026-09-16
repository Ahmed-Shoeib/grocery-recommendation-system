"""`backend.activity_sync`: bounded bootstrap + incremental delta + a
self-healing fallback to a fresh bootstrap - the fix for the real
`/api/ai/user-activities` blocker (the table has grown past 1.5 million
rows, no server-side delta filter, and a full traversal is refused by
`BackendApiClient`'s 10,000-page safety cap).

Uses small hand-built fake clients (not `FakeBackendClient`, which models
the whole `BackendApiClient` surface) so each test controls exactly what
`iter_activity_pages` yields, page by page.
"""

from __future__ import annotations

import itertools

from recommendation.backend.activity_cache import load_activity_cache
from recommendation.backend.activity_sync import sync_activities
from recommendation.backend.dtos import ApiActivity


def _row(guid, action, product_id, ts):
    return {"userId": guid, "actionType": action, "productId": product_id, "timestamp": ts}


class _PagedClient:
    """Fixed page list - `iter_activity_pages` yields at most `max_pages`
    of them, exactly like the real client's own bounded contract.
    """

    def __init__(self, pages: list[list[dict]]):
        self._pages = pages
        self.calls: list[int] = []

    def iter_activity_pages(self, *, user_guid=None, max_pages=10_000):
        self.calls.append(max_pages)
        for page in self._pages[:max_pages]:
            yield [ApiActivity.model_validate(r) for r in page]

    def fetch_activity_window(self, *, user_guid=None, max_pages=10_000):
        self.calls.append(max_pages)
        truncated = self._pages[:max_pages]
        rows = [ApiActivity.model_validate(r) for page in truncated for r in page]
        exhausted = len(self._pages) <= max_pages
        return rows, exhausted


class _InfiniteFeedClient:
    """Never runs out - simulates a table with effectively unlimited pages
    (i.e. the real 1.5M+-row table) so a test can prove boundedness
    without allocating anything close to that many rows: if `sync`
    fetched more than `max_pages`, `pages_yielded` would show it.
    """

    def __init__(self):
        self.pages_yielded = 0

    def iter_activity_pages(self, *, user_guid=None, max_pages=10_000):
        for i in itertools.count():
            if i >= max_pages:
                return
            self.pages_yielded += 1
            yield [ApiActivity.model_validate(_row("g1", "AddToCart", 1, f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}"))]

    def fetch_activity_window(self, *, user_guid=None, max_pages=10_000):
        rows = []
        for i in range(max_pages):
            self.pages_yielded += 1
            rows.append(ApiActivity.model_validate(_row("g1", "AddToCart", 1, f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}")))
        return rows, False  # never exhausts - simulates the 1.5M-row table


def test_bootstrap_never_exceeds_bootstrap_max_pages_even_against_an_effectively_unlimited_feed(tmp_path):
    """The core fix: a table with 1.5M+ rows (unboundedly many pages) must
    never be traversed to completion on an ordinary load - the walk stops
    at the configured bound, not when the feed runs out.
    """
    client = _InfiniteFeedClient()
    rows = sync_activities(
        client, tmp_path / "cache.json", bootstrap_max_pages=50, delta_max_pages=10, max_rows=10_000,
    )
    assert client.pages_yielded == 50
    assert len(rows) == 50


def test_first_sync_bootstraps_and_persists(tmp_path):
    pages = [[_row("g1", "AddToCart", 1, "2026-08-01T10:00:00"), _row("g2", "ViewProduct", 2, "2026-08-01T09:00:00")]]
    client = _PagedClient(pages)
    cache_path = tmp_path / "cache.json"
    rows = sync_activities(client, cache_path, bootstrap_max_pages=10, delta_max_pages=5, max_rows=1000)
    assert len(rows) == 2
    cached = load_activity_cache(cache_path)
    assert cached is not None
    assert cached.high_water_timestamp == "2026-08-01T10:00:00"


def test_repeated_sync_with_no_new_rows_does_not_duplicate(tmp_path):
    pages = [[_row("g1", "AddToCart", 1, "2026-08-01T10:00:00")]]
    client = _PagedClient(pages)
    cache_path = tmp_path / "cache.json"
    rows1 = sync_activities(client, cache_path, bootstrap_max_pages=10, delta_max_pages=5, max_rows=1000)
    rows2 = sync_activities(client, cache_path, bootstrap_max_pages=10, delta_max_pages=5, max_rows=1000)
    assert len(rows1) == len(rows2) == 1
    # Second call was a delta (recognized the boundary on page 1), not a
    # second bootstrap re-walking the whole configured budget.
    assert client.calls[1] == 5


def test_delta_merges_new_head_rows_without_duplicating_old_ones(tmp_path):
    cache_path = tmp_path / "cache.json"
    old_pages = [[_row("g1", "AddToCart", 1, "2026-08-01T10:00:00")]]
    sync_activities(_PagedClient(old_pages), cache_path, bootstrap_max_pages=10, delta_max_pages=5, max_rows=1000)

    # Newest-first feed: new row now leads, old row (the previous
    # high-water mark) still trails it on the next page.
    new_pages = [
        [_row("g2", "PlaceOrder", 3, "2026-08-02T11:00:00")],
        [_row("g1", "AddToCart", 1, "2026-08-01T10:00:00")],
    ]
    rows = sync_activities(_PagedClient(new_pages), cache_path, bootstrap_max_pages=10, delta_max_pages=5, max_rows=1000)
    keys = sorted((r.user_id, r.product_id) for r in rows)
    assert keys == [("g1", 1), ("g2", 3)]  # exactly one of each - no duplicate of the old row


def test_stale_checkpoint_falls_back_to_a_fresh_bootstrap(tmp_path):
    """If the delta walk cannot find the previously-known boundary within
    `delta_max_pages`, it must not guess or silently drop data - it falls
    back to a full bounded bootstrap of the CURRENT feed.
    """
    cache_path = tmp_path / "cache.json"
    old_pages = [[_row("g1", "AddToCart", 1, "2026-08-01T10:00:00")]]
    sync_activities(_PagedClient(old_pages), cache_path, bootstrap_max_pages=10, delta_max_pages=2, max_rows=1000)

    # A completely different feed (e.g. simulating drift/a reset upstream)
    # that never reproduces the old boundary within the small delta budget.
    unrelated_pages = [
        [_row("gX", "AddToCart", 99, "2026-09-01T00:00:00")],
        [_row("gY", "AddToCart", 98, "2026-09-01T00:01:00")],
        [_row("gZ", "AddToCart", 97, "2026-09-01T00:02:00")],
    ]
    client = _PagedClient(unrelated_pages)
    rows = sync_activities(client, cache_path, bootstrap_max_pages=10, delta_max_pages=2, max_rows=1000)
    # Fell back to bootstrap: the whole (small) unrelated feed is now cached,
    # not merged with the stale, unrelated old row.
    assert {r.product_id for r in rows} == {99, 98, 97}
    # Two calls: the failed delta walk (bounded at 2), then the bootstrap
    # fallback (bounded at 10) - never an unbounded walk.
    assert client.calls == [2, 10]


def test_force_bootstrap_ignores_an_existing_cache(tmp_path):
    cache_path = tmp_path / "cache.json"
    sync_activities(
        _PagedClient([[_row("g1", "AddToCart", 1, "2026-08-01T10:00:00")]]),
        cache_path, bootstrap_max_pages=10, delta_max_pages=5, max_rows=1000,
    )
    client = _PagedClient([[_row("g2", "AddToCart", 2, "2026-08-02T10:00:00")]])
    rows = sync_activities(
        client, cache_path, bootstrap_max_pages=10, delta_max_pages=5, max_rows=1000, force_bootstrap=True,
    )
    assert [r.product_id for r in rows] == [2]
    assert client.calls == [10]  # a bootstrap-sized call, not a delta-sized one


def test_max_rows_caps_the_retained_window(tmp_path):
    pages = [[_row(f"g{i}", "AddToCart", i, f"2026-08-01T00:{i:02d}:00") for i in range(20)]]
    rows = sync_activities(
        _PagedClient(pages), tmp_path / "cache.json", bootstrap_max_pages=10, delta_max_pages=5, max_rows=5,
    )
    assert len(rows) == 5
    # The newest 5 (highest timestamps) are kept, not an arbitrary slice.
    assert {r.product_id for r in rows} == {15, 16, 17, 18, 19}


def test_corrupt_cache_file_recovers_via_fresh_bootstrap(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("{ not json", encoding="utf-8")
    client = _PagedClient([[_row("g1", "AddToCart", 1, "2026-08-01T10:00:00")]])
    rows = sync_activities(client, cache_path, bootstrap_max_pages=10, delta_max_pages=5, max_rows=1000)
    assert len(rows) == 1
    assert client.calls == [10]  # treated as a first run (bootstrap-sized), not a delta
