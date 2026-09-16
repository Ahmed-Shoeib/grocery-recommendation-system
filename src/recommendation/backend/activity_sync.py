"""Bounded, restart-durable sync strategy for
`GET /api/ai/user-activities` - the fix for the real blocker this phase
exists to resolve: the backend's `UserActivities` table has grown past
1.5 million rows with no server-side delta/since query parameter
(verified via Swagger, 2026-09-15), so a full traversal is both
infeasible (15,000+ requests) and actively refused by
`BackendApiClient`'s 10,000-page safety cap (`BackendPaginationError`).
Nothing in the normal startup/refresh path may call the old
run-to-completion `client.list_activities()` any more.

Strategy (ranked option A from the task - incremental delta sync - with
a self-healing fallback to option B, a bounded window): maintain a
persisted, bounded window of the most recent activity rows
(`activity_cache.ActivityCacheState`).

- **Bootstrap** (first load, or a stale-checkpoint/corrupt-cache
  fallback): walk up to `bootstrap_max_pages` pages from the feed's head.
  This is a deliberate BOUNDED RECENT WINDOW, not full lifetime history -
  see docs/production-feature-parity-audit.md Section 5 for the honest
  disclosure of what this changes for lifetime-count features.
- **Delta** (every later load): walk from the head again, but stop the
  moment a row is recognized as already-cached (via the persisted
  high-water timestamp plus a same-timestamp tie-break key set) - cheap
  in steady state because the feed is newest-first (verified live
  2026-09-15: an unfiltered page 1 consistently returns near-"now"
  timestamps). New rows are merged into the existing window (no
  duplicates - the boundary check exists precisely to prevent
  re-collecting rows already on disk).
- **Fallback**: if a delta walk does NOT find the previously-known
  boundary within `delta_max_pages` pages - an implausible burst of new
  events, or the newest-first assumption not holding for some reason -
  this does not guess or silently drop data: it falls back to a fresh
  bounded bootstrap, exactly like a first run. The sync is therefore
  self-correcting rather than dependent on the ordering assumption being
  perfect.

Always returns RAW `ApiActivity` rows; translation to canonical
`UserInteraction`s stays entirely inside the existing, unchanged
`backend.loader.load_backend_events` - this module never re-implements
action-type or identity semantics (no second identity path).

**`sync_user_activities`** (docs/data-mapping.md 19.14, added by the
train-serve parity audit) reuses the exact same bootstrap/delta/fallback
primitives, scoped to one user via the `userId` filter, but tracks
genuine COMPLETION rather than a deliberately-bounded window: a per-user
history is confirmed dramatically smaller than the global table, so
walking it to `hasNext=false` (not just to a page cap) is the correct
fix for behavioral model features that were trained from each user's
complete history in the production-aligned SQLite dataset - a bounded
recent window is right for the GLOBAL popularity/fallback signal, wrong
for a specific user's own purchase/cart/search counts, category
affinity, semantic embedding, or price profile.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from recommendation.backend.activity_cache import (
    ActivityCacheState,
    load_activity_cache,
    new_cache_state,
    save_activity_cache,
)
from recommendation.backend.client import BackendApiClient
from recommendation.backend.dtos import ApiActivity
from recommendation.backend.user_activity_cache import UserActivityCacheStore
from recommendation.logging import get_logger

logger = get_logger(__name__)

_EPOCH = datetime(1970, 1, 1)


def _row_key(row: ApiActivity) -> tuple:
    ts = row.timestamp.isoformat() if row.timestamp is not None else None
    return (row.user_id, row.action_type, row.product_id, row.slug, ts)


def _max_timestamp(rows: list[ApiActivity]) -> str | None:
    stamped = [r.timestamp for r in rows if r.timestamp is not None]
    return max(stamped).isoformat() if stamped else None


def _cap_rows(rows: list[ApiActivity], max_rows: int) -> list[ApiActivity]:
    if len(rows) <= max_rows:
        return rows
    return sorted(rows, key=lambda r: r.timestamp or _EPOCH, reverse=True)[:max_rows]


def _to_cache_rows(rows: list[ApiActivity]) -> list[dict]:
    # `populate_by_name=True` on the DTO lets these round-trip back through
    # `ApiActivity.model_validate` unchanged - no alias gymnastics needed.
    return [r.model_dump(mode="json") for r in rows]


def _bootstrap(
    client: BackendApiClient, max_pages: int, max_rows: int, *, user_guid: str | None = None
) -> tuple[ActivityCacheState, bool]:
    """Returns `(state, exhausted)` - `exhausted=True` iff the feed itself
    reported `hasNext=false` within `max_pages` (genuine completion, only
    meaningful for the per-user caller); for the GLOBAL window this is
    always expected to end up `False` in practice (the table is far
    larger than any sane page budget) and callers that don't care about
    completeness simply ignore the flag.
    """
    rows, exhausted = client.fetch_activity_window(user_guid=user_guid, max_pages=max_pages)
    if not exhausted:
        logger.warning(
            "activity bootstrap hit its %d-page cap (%d rows)%s without the feed running out - "
            "this is a BOUNDED window, not full history; increase the relevant max-pages setting "
            "if more history is needed",
            max_pages, len(rows), f" for user {user_guid}" if user_guid else "",
        )
    rows = _cap_rows(rows, max_rows)
    high_water = _max_timestamp(rows)
    boundary = {_row_key(r) for r in rows if r.timestamp is not None and r.timestamp.isoformat() == high_water}
    logger.info(
        "activity bootstrap complete%s: %d row(s) cached (high_water=%s, exhausted=%s)",
        f" for user {user_guid}" if user_guid else "", len(rows), high_water, exhausted,
    )
    return new_cache_state(_to_cache_rows(rows), high_water, boundary), exhausted


def _delta(
    client: BackendApiClient, cache: ActivityCacheState, max_pages: int, max_rows: int, *, user_guid: str | None = None
) -> ActivityCacheState | None:
    """Returns the merged cache state, or `None` if the walk did not reach
    the previously-known checkpoint within `max_pages` - the caller falls
    back to `_bootstrap` in that case.
    """
    boundary = cache.boundary_key_set()
    high_water = cache.high_water_timestamp
    new_rows: list[ApiActivity] = []
    found_boundary = high_water is None  # an empty prior cache has nothing to catch up to
    for page in client.iter_activity_pages(user_guid=user_guid, max_pages=max_pages):
        for row in page:
            ts = row.timestamp.isoformat() if row.timestamp is not None else None
            if high_water is not None and ts is not None and ts < high_water:
                found_boundary = True
                continue
            if _row_key(row) in boundary:
                found_boundary = True
                continue
            new_rows.append(row)
        if found_boundary:
            break
    if not found_boundary:
        return None

    existing_rows = [ApiActivity.model_validate(r) for r in cache.rows]
    merged = _cap_rows(new_rows + existing_rows, max_rows)
    new_high_water = _max_timestamp(merged) or high_water
    new_boundary = {
        _row_key(r) for r in merged if r.timestamp is not None and r.timestamp.isoformat() == new_high_water
    }
    if new_rows:
        logger.info("activity delta sync: %d new row(s) merged (window now %d row(s))", len(new_rows), len(merged))
    return new_cache_state(_to_cache_rows(merged), new_high_water, new_boundary)


def sync_activities(
    client: BackendApiClient,
    cache_path: Path,
    *,
    bootstrap_max_pages: int,
    delta_max_pages: int,
    max_rows: int,
    force_bootstrap: bool = False,
) -> list[ApiActivity]:
    """The one entry point `adapters.backend_factory` calls instead of the
    old, now-infeasible `client.list_activities()`. Returns the current
    bounded window of raw activity rows and persists the updated cache
    (atomic write - see `activity_cache.save_activity_cache`).

    `force_bootstrap=True` is the manual full-reset path (equivalently:
    delete the file at `cache_path` and call this normally).
    """
    cache = None if force_bootstrap else load_activity_cache(cache_path)
    if cache is None:
        state, _ = _bootstrap(client, bootstrap_max_pages, max_rows)
    else:
        state = _delta(client, cache, delta_max_pages, max_rows)
        if state is None:
            logger.warning(
                "activity delta sync did not reach the last known checkpoint within %d page(s) - "
                "falling back to a fresh bounded bootstrap", delta_max_pages,
            )
            state, _ = _bootstrap(client, bootstrap_max_pages, max_rows)
    save_activity_cache(cache_path, state)
    return [ApiActivity.model_validate(r) for r in state.rows]


def sync_user_activities(
    client: BackendApiClient,
    store: UserActivityCacheStore,
    user_guid: str,
    *,
    max_pages: int,
    max_rows: int,
    force_bootstrap: bool = False,
) -> tuple[list[ApiActivity], bool]:
    """Per-user counterpart of `sync_activities` (docs/data-mapping.md
    19.14): walks ONE user's `userId`-filtered activity feed, tracking an
    explicit completeness flag instead of inferring it from "has some
    rows" - a user with 20 cached events must never be treated as if that
    were their whole lifetime unless the feed itself confirmed there is
    nothing more (`hasNext=false`).

    Mutates `store` in place (`store.entries[user_guid]`,
    `store.complete[user_guid]`) and returns `(rows, is_complete)`. The
    caller (`adapters.backend_lazy_events_adapter
    .LazyBackendUserEventsAdapter`) owns persisting `store` to disk -
    this function is pure I/O-against-the-backend, no disk access.

    Completeness rules:
      - No prior cache for this user (or `force_bootstrap=True`): a fresh
        bootstrap: complete iff it reached `hasNext=false` within
        `max_pages`.
      - A prior cache exists: a delta walk. If it finds the previously-
        known checkpoint, completeness is UNCHANGED (a successful catch-up
        of an already-complete user stays complete; of a not-yet-complete
        one stays not-yet-complete - a delta never independently proves
        completeness on its own). If the delta does NOT find the
        checkpoint within `max_pages`, falls back to a fresh bootstrap
        (self-healing), exactly like the global sync.
    """
    cache = None if force_bootstrap else store.entries.get(user_guid)
    was_complete = False if force_bootstrap else store.complete.get(user_guid, False)

    if cache is None:
        state, exhausted = _bootstrap(client, max_pages, max_rows, user_guid=user_guid)
        is_complete = exhausted
    else:
        merged = _delta(client, cache, max_pages, max_rows, user_guid=user_guid)
        if merged is None:
            logger.warning(
                "per-user activity delta sync did not reach the last known checkpoint for a user "
                "within %d page(s) - falling back to a fresh complete-history bootstrap for them",
                max_pages,
            )
            state, exhausted = _bootstrap(client, max_pages, max_rows, user_guid=user_guid)
            is_complete = exhausted
        else:
            state = merged
            is_complete = was_complete

    store.entries[user_guid] = state
    store.complete[user_guid] = is_complete
    return [ApiActivity.model_validate(r) for r in state.rows], is_complete
