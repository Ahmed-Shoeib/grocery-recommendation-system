"""Persisted, restart-durable cache for the bounded window of
`GET /api/ai/user-activities` rows `backend.activity_sync` maintains.

Exists because the real backend's UserActivities table has grown past
1.5 million rows with no server-side delta/since filter (verified via
Swagger, 2026-09-15) - a full traversal needs 15,000+ HTTP requests and
is refused outright by `BackendApiClient`'s 10,000-page safety cap
(`BackendPaginationError`). Every runtime reload therefore works off a
bounded window plus an incremental "what's new since last time" delta
(`activity_sync.sync_activities`), persisted here so a process restart
does not have to re-walk the window from scratch every time.

Stores RAW activity rows (dicts matching `backend.dtos.ApiActivity`'s
shape), never canonical `UserInteraction`s - translation always goes
through the one existing `backend.loader.load_backend_events` path (the
same canonical action-type/identity resolution every other source uses),
so this cache can never become a second, drifting semantics path.

Runtime data, never committed: lives under `data/processed/`
(gitignored wholesale, same as the identity registry and embedding
caches - see `.gitignore`). Contains real user GUIDs and product ids,
never tokens/secrets.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from recommendation.logging import get_logger

logger = get_logger(__name__)

CACHE_FORMAT_VERSION = 1


@dataclass
class ActivityCacheState:
    fetched_at: str
    high_water_timestamp: str | None
    # JSON has no tuple type, so boundary keys round-trip as lists.
    boundary_keys: list[list] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)

    def boundary_key_set(self) -> set[tuple]:
        return {tuple(k) for k in self.boundary_keys}


def load_activity_cache(path: Path) -> ActivityCacheState | None:
    """Returns `None` on ANY problem - missing file, unreadable, corrupt
    JSON, unknown format version, or an unexpected shape - never raises.
    This is the cache's entire corruption-recovery story: a `None` result
    tells `activity_sync.sync_activities` to treat this exactly like a
    first run (a fresh bounded bootstrap), which is always safe.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("activity cache at %s is unreadable/corrupt (%s) - starting a fresh bootstrap", path, exc)
        return None

    if not isinstance(raw, dict):
        logger.warning("activity cache at %s is not a JSON object - starting a fresh bootstrap", path)
        return None
    if raw.get("version") != CACHE_FORMAT_VERSION:
        logger.warning(
            "activity cache at %s has version %r, expected %d - starting a fresh bootstrap",
            path, raw.get("version"), CACHE_FORMAT_VERSION,
        )
        return None
    try:
        return ActivityCacheState(
            fetched_at=raw["fetched_at"],
            high_water_timestamp=raw.get("high_water_timestamp"),
            boundary_keys=list(raw.get("boundary_keys") or []),
            rows=list(raw.get("rows") or []),
        )
    except (KeyError, TypeError) as exc:
        logger.warning("activity cache at %s has an unexpected shape (%s) - starting a fresh bootstrap", path, exc)
        return None


def save_activity_cache(path: Path, state: ActivityCacheState) -> None:
    """Atomic write (temp file in the same directory + `os.replace`), so a
    crash mid-write can never leave a half-written, corrupt cache file
    behind - the previous file (or its absence) is the only state ever
    observable to a reader.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "fetched_at": state.fetched_at,
        "high_water_timestamp": state.high_water_timestamp,
        "boundary_keys": state.boundary_keys,
        "rows": state.rows,
    }
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def new_cache_state(rows: list[dict], high_water_timestamp: str | None, boundary_keys: set[tuple]) -> ActivityCacheState:
    return ActivityCacheState(
        fetched_at=datetime.now(timezone.utc).isoformat(),
        high_water_timestamp=high_water_timestamp,
        boundary_keys=[list(k) for k in boundary_keys],
        rows=rows,
    )
