"""Per-user COMPLETE-history adapter for the `backend_api` `AdapterBundle`
(docs/data-mapping.md 19.14, the train-serve parity fix).

**Why this exists, and why the earlier "cold-start safety net" version of
this class was not sufficient.** `backend.activity_sync.sync_activities`
only ever loads a BOUNDED recent window of `GET /api/ai/user-activities`
(the real table has grown past 1.5 million rows with no server-side
delta filter). The train-serve parity audit found that the first version
of this adapter only re-checked the backend for a user with ZERO events
in that bounded window - a real, active user with SOME activity inside
the window and MORE activity outside it was silently served behavioral
features (`purchase_count`, `cart_item_count`, `search_count`,
`total_engagement_events`, `category_affinity`, the semantic embedding,
the price profile - every one of `RANKING_FEATURE_NAMES_USER` and the
Two-Tower's user numeric/categorical inputs) computed from only the
partial subset, while the trained model was fit on COMPLETE per-user
history from the production-aligned SQLite dataset. That is a real
train-serve semantic mismatch, not just a cold-start edge case.

The fix: EVERY user's first per-request access (regardless of whether
they already have some local events) triggers exactly one bounded
attempt to walk `GET /api/ai/user-activities?userId=...` to genuine
completion (`backend.activity_sync.sync_user_activities`, `hasNext=false`
- confirmed live that a single user's own history is orders of magnitude
smaller than the global table: an unfiltered cursor decoded to a ~1.5M-
row offset vs ~2,600 for one user's own filtered history). The result
REPLACES (never appends to) whatever partial rows this user already had
from the bounded global window, so nothing is ever double-counted.
Completeness is tracked explicitly and persisted
(`backend.user_activity_cache`) - once a user is known-complete, later
accesses within `ttl_seconds` reuse the cached history with NO network
call at all, and after the TTL do a cheap incremental delta (not a
re-fetch) rather than walking the whole history again.

Only ever used for `data_source=backend_api`, and only by
`adapters.backend_factory.build_backend_api_adapters`. Never re-
implements canonical action-type or identity semantics - the fetched raw
rows go through the exact same `backend.loader.load_backend_events` used
everywhere else.
"""

from __future__ import annotations

from datetime import datetime, timezone

from recommendation.adapters.user_events_adapter import UserEventsAdapter
from recommendation.backend.activity_sync import sync_user_activities
from recommendation.backend.client import BackendApiClient
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.backend.dtos import ApiActivity
from recommendation.backend.loader import BackendCatalog, load_backend_events
from recommendation.backend.user_activity_cache import UserActivityCacheStore, save_user_activity_cache_store
from recommendation.schemas.engagement import (
    CartAffinityRecord,
    ChatbotContextRecord,
    ClickRecord,
    PurchaseRecord,
    SearchRecord,
)
from recommendation.schemas.events import ActionType, UserInteraction
from recommendation.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_TTL_SECONDS = 300.0


def _seconds_since(iso_timestamp: str | None) -> float:
    if not iso_timestamp:
        return float("inf")
    try:
        then = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return float("inf")
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds()


class LazyBackendUserEventsAdapter(UserEventsAdapter):
    def __init__(
        self,
        events: list[UserInteraction],
        *,
        client: BackendApiClient,
        resolver: ExternalIdentityResolver,
        catalog: BackendCatalog,
        guid_by_internal: dict[int, str],
        max_pages: int,
        max_rows: int,
        store: UserActivityCacheStore,
        store_path,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    ) -> None:
        super().__init__(events)
        self._client = client
        self._resolver = resolver
        self._catalog = catalog
        self._guid_by_internal = guid_by_internal
        self._max_pages = max_pages
        self._max_rows = max_rows
        self._store = store
        self._store_path = store_path
        self._ttl_seconds = ttl_seconds
        # Gate for the EAGER "build an engagement profile for every known
        # user" bulk pass (`features.pipeline.run_feature_pipeline`, used
        # only to populate the dashboard's user list -
        # `api.service.RecommendationService.engagement_profiles`) -
        # `api.service._load_data_snapshot` toggles this off around that
        # one call for `backend_api`. Without this, a 547-user roster (or
        # a much larger production one) would turn every snapshot
        # build/refresh into one per-user complete-history fetch PER USER
        # - exactly the per-user-request fan-out this fix exists to
        # avoid. The real per-request serving path (`recommend(user_id)`)
        # always runs with this left at its default `True`.
        self.lazy_enabled = True

    def _has_any_local_event(self, user_id: int) -> bool:
        return any((user_id, action_type) in self._by_user_and_type for action_type in ActionType)

    def _replace_user_events(self, user_id: int, events: list[UserInteraction]) -> None:
        """Replaces (never appends to) this user's events across every
        action type - the freshly-fetched-to-completion set supersedes
        whatever partial rows this user had from the bounded global
        window, so nothing is ever double-counted.
        """
        for action_type in ActionType:
            self._by_user_and_type.pop((user_id, action_type), None)
        for event in events:
            self._by_user_and_type[(event.user_id, event.action_type)].append(event)

    def _ensure_complete(self, user_id: int) -> None:
        if not self.lazy_enabled:
            return
        guid = self._guid_by_internal.get(user_id)
        if guid is None:
            return  # not a backend-known user at all - nothing to look up

        entry = self._store.entries.get(guid)
        already_complete = self._store.complete.get(guid, False)
        if already_complete and entry is not None and _seconds_since(entry.fetched_at) < self._ttl_seconds:
            # Reuse the persisted complete history - no network call. But a
            # freshly constructed adapter instance (e.g. every
            # RecommendationService.maybe_refresh rebuilds the whole
            # AdapterBundle from scratch) seeds `_by_user_and_type` from
            # only the CURRENT bounded global window, which excludes this
            # user's off-window rows - so the persisted `entry.rows` must
            # be replayed into the in-memory index on every hit of this
            # shortcut, not just on the original fetch. `_replace_user_events`
            # is replace-not-append, so re-running this on an instance that
            # already has the right data (same long-lived instance, repeat
            # call) is a harmless no-op change.
            cached_activities = [ApiActivity.model_validate(row) for row in entry.rows]
            interactions, _ = load_backend_events(cached_activities, self._resolver, self._catalog)
            found = [e for e in interactions if e.user_id == user_id]
            self._replace_user_events(user_id, found)
            return

        try:
            rows, is_complete = sync_user_activities(
                self._client, self._store, guid, max_pages=self._max_pages, max_rows=self._max_rows,
            )
        except Exception:
            logger.warning(
                "per-user complete-history sync failed for a user - leaving their profile as-is", exc_info=True
            )
            return
        save_user_activity_cache_store(self._store_path, self._store)

        if not is_complete:
            logger.warning(
                "per-user activity sync did not reach completion within the configured safety cap for a "
                "user - their behavioral features may still undercount until the next sync"
            )

        interactions, _ = load_backend_events(rows, self._resolver, self._catalog)
        found = [e for e in interactions if e.user_id == user_id]
        self._replace_user_events(user_id, found)
        if found:
            logger.info(
                "per-user complete-history sync: %d activity row(s) now backing this user's profile "
                "(complete=%s)",
                len(found), is_complete,
            )

    # --- overrides: ensure-complete-then-delegate to the base implementation --

    def get_clicks(self, user_id: int) -> list[ClickRecord]:
        self._ensure_complete(user_id)
        return super().get_clicks(user_id)

    def get_purchases(self, user_id: int) -> list[PurchaseRecord]:
        self._ensure_complete(user_id)
        return super().get_purchases(user_id)

    def get_cart_items(self, user_id: int) -> list[CartAffinityRecord]:
        self._ensure_complete(user_id)
        return super().get_cart_items(user_id)

    def get_search_history(self, user_id: int) -> list[SearchRecord]:
        self._ensure_complete(user_id)
        return super().get_search_history(user_id)

    def get_chatbot_context(self, user_id: int) -> ChatbotContextRecord | None:
        self._ensure_complete(user_id)
        return super().get_chatbot_context(user_id)
