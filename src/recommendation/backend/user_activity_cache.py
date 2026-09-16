"""Persisted, per-user COMPLETE activity-history cache
(docs/data-mapping.md 19.14).

Separate from `backend.activity_cache` (a single, bounded GLOBAL recent
window used for product-popularity aggregates/fallback): this one holds,
per user GUID, the result of walking `GET /api/ai/user-activities
?userId=...` to genuine completion (`hasNext=false`) - the fix for the
train-serve parity gap found in the audit: a user with SOME activity in
the bounded global window and MORE activity outside it must not have
their behavioral features (purchase/cart/search counts, category
affinity, semantic embedding, price profile - all trained from COMPLETE
per-user history in the production-aligned SQLite dataset) silently
computed from only the partial subset.

Only ever grows for users who are ACTUALLY served a recommendation - not
the whole roster - so this stays small in practice even though a single
entry, once fetched, holds that user's whole (or whole-so-far) history.

Each entry carries an explicit completeness marker (see
`UserActivityCacheStore.complete`) rather than inferring "complete" from
"has some rows": a user with 20 cached events must never be treated as
if 20 were their entire lifetime unless the feed itself confirmed
`hasNext=false` for that fetch (`backend.client
.BackendApiClient.fetch_activity_window`'s `exhausted` return value).

Runtime data, never committed: lives under `data/processed/` (gitignored
wholesale - see `.gitignore`). Contains real user GUIDs and product ids,
never tokens/secrets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from recommendation.backend.activity_cache import ActivityCacheState, atomic_write_json
from recommendation.logging import get_logger

logger = get_logger(__name__)

CACHE_FORMAT_VERSION = 1


@dataclass
class UserActivityCacheStore:
    """`entries[guid]` is that user's cached activity window (same shape
    as the global cache's `ActivityCacheState`, including its own
    `fetched_at`/high-water/boundary bookkeeping so `activity_sync`'s
    bootstrap/delta primitives can be reused unchanged, just scoped to
    one user). `complete[guid]` is `True` only once a fetch for that user
    has genuinely reached `hasNext=false` - the explicit completeness
    marker Section 7 of the audit requires. A guid absent from `complete`
    (or present with `... .complete` didn't ever occur) means "unknown -
    never successfully completed," which is treated as "not yet
    reusable" regardless of whatever partial rows might already sit in
    `entries` from an interrupted/capped attempt.
    """

    entries: dict[str, ActivityCacheState] = field(default_factory=dict)
    complete: dict[str, bool] = field(default_factory=dict)


def load_user_activity_cache_store(path: Path) -> UserActivityCacheStore:
    """Returns an EMPTY store on any problem (missing file, corrupt JSON,
    unknown version, unexpected shape) - never raises. Exactly like
    `activity_cache.load_activity_cache`, a bad file degrades to "nothing
    cached yet," which is always safe: every user simply gets a fresh
    complete-history fetch on next access.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return UserActivityCacheStore()
    except (OSError, ValueError) as exc:
        logger.warning("user activity cache at %s is unreadable/corrupt (%s) - starting empty", path, exc)
        return UserActivityCacheStore()

    if not isinstance(raw, dict) or raw.get("version") != CACHE_FORMAT_VERSION:
        logger.warning(
            "user activity cache at %s has an unexpected version/shape - starting empty", path
        )
        return UserActivityCacheStore()

    store = UserActivityCacheStore()
    for guid, entry in (raw.get("users") or {}).items():
        try:
            store.entries[guid] = ActivityCacheState(
                fetched_at=entry["fetched_at"],
                high_water_timestamp=entry.get("high_water_timestamp"),
                boundary_keys=list(entry.get("boundary_keys") or []),
                rows=list(entry.get("rows") or []),
            )
            store.complete[guid] = bool(entry.get("complete", False))
        except (KeyError, TypeError) as exc:
            logger.warning("user activity cache at %s: skipping one malformed user entry (%s)", path, exc)
    return store


def save_user_activity_cache_store(path: Path, store: UserActivityCacheStore) -> None:
    """Atomic write - see `activity_cache.atomic_write_json`."""
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "users": {
            guid: {
                "fetched_at": state.fetched_at,
                "high_water_timestamp": state.high_water_timestamp,
                "boundary_keys": state.boundary_keys,
                "rows": state.rows,
                "complete": store.complete.get(guid, False),
            }
            for guid, state in store.entries.items()
        },
    }
    atomic_write_json(path, payload)
