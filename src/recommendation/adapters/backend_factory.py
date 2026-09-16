"""Wires the real-backend REST source into an `AdapterBundle`.

Third `AdapterBundle` factory, alongside `adapters.factory
.build_synthetic_adapters` and `adapters.sqlite_factory
.build_sqlite_adapters` - same return type, same downstream consumers
(feature engineering, Two-Tower, ranker, serving, dashboard), which never
learn that the data came from an HTTP API. Reuses the existing
`InMemoryProductCatalogAdapter` / `InMemoryUserAdapter` /
`InMemoryReviewAdapter` / `UserEventsAdapter` classes unchanged - the only
new code is `backend.*` (HTTP + DTO -> Raw* mapping + identity
resolution).

Load model: fetch the catalog + a BOUNDED activity window once, in
memory, exactly like `build_sqlite_adapters` reads the whole SQLite file
once - network calls live at this boundary only, never per feature or
per candidate. `RecommendationService.maybe_refresh` re-invokes this on
the configured TTL so new `User_events`-style rows the backend records
after startup become visible without a restart.

**Activity loading is bounded at startup, complete per-user on demand**
(docs/data-mapping.md 19.13/19.14): the real `GET /api/ai/user-activities`
table has grown past 1.5 million rows with no server-side delta filter,
so a full traversal would need 15,000+ requests and is refused outright
by `BackendApiClient`'s 10,000-page safety cap. `backend.activity_sync
.sync_activities` maintains a persisted, restart-durable BOUNDED window
(bootstrap once, incremental delta thereafter, self-healing fallback to
a fresh bootstrap) for GLOBAL product-popularity aggregates and as a
cheap fallback signal - never for a specific user's own behavioral
features. A user whose history is not (fully) captured by that window is
still known to exist (`load_backend_users_roster`, via the independent
`GET /api/users` roster), and the moment they are actually recommended,
`backend_lazy_events_adapter.LazyBackendUserEventsAdapter` walks their
`userId`-filtered feed to genuine completion
(`backend.activity_sync.sync_user_activities`) and caches that complete
history (`backend.user_activity_cache`) - this is what keeps
`user_log_purchase_count`/`category_affinity`/the semantic embedding/the
price profile computed from COMPLETE history, matching how the model was
trained, rather than silently truncated to whatever fell inside the
bounded global window.

Purchase/cart authoritative source: `GET /api/ai/user-activities`
(PlaceOrder / AddToCart rows - the authoritative activity source since
the 2026-09-15 switch to the Ai-tagged routes, docs/data-mapping.md 19.5)
is the sole engagement-truth source consumed here - `/api/orders` and
`/api/cart` are never read, so the same real-world action cannot be
double-counted through two code paths (mirrors the SQLite factory's
`User_events`-only contract).
"""

from __future__ import annotations

from pathlib import Path

from recommendation.adapters.backend_lazy_events_adapter import LazyBackendUserEventsAdapter
from recommendation.adapters.base import AdapterBundle
from recommendation.adapters.product_adapter import InMemoryProductCatalogAdapter
from recommendation.adapters.review_adapter import InMemoryReviewAdapter
from recommendation.adapters.user_adapter import InMemoryUserAdapter
from recommendation.backend.activity_sync import sync_activities
from recommendation.backend.client import BackendApiClient
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.backend.loader import (
    RawUser,
    load_backend_catalog,
    load_backend_events,
    load_backend_reviews,
    load_backend_users,
    load_backend_users_roster,
)
from recommendation.backend.user_activity_cache import load_user_activity_cache_store
from recommendation.config import AppConfig, get_config, resolve_path
from recommendation.logging import get_logger

logger = get_logger(__name__)


def _merge_users(roster: list[RawUser], activity_derived: list[RawUser]) -> list[RawUser]:
    """Additive union keyed by internal id - `activity_derived` wins on
    conflict (it went through the existing per-user `/api/users/{guid}`
    enrichment path). A source that contributes nothing (e.g. an empty
    roster from a client/test double with no `/api/users` support)
    degrades exactly to the other source alone.
    """
    by_id = {u.id: u for u in roster}
    by_id.update({u.id: u for u in activity_derived})
    return list(by_id.values())


def build_backend_api_adapters(
    config: AppConfig | None = None,
    *,
    client: BackendApiClient | None = None,
    resolver: ExternalIdentityResolver | None = None,
    activity_cache_path: Path | None = None,
    user_activity_cache_path: Path | None = None,
) -> AdapterBundle:
    """Build a full `AdapterBundle` from the backend REST API.

    `client` / `resolver` are injectable for tests; in production both are
    constructed from `config` (`config.backend_api` and
    `config.paths.backend_identity_registry`). The identity registry is
    persisted after the load so slug/GUID -> int assignments survive
    process restarts and refreshes. `activity_cache_path` (the bounded
    GLOBAL window) and `user_activity_cache_path` (the per-user COMPLETE-
    history store) are likewise injectable (tests point them at a tmp
    path so they never read/write the real, shared files under
    `data/processed/`); production always uses
    `config.paths.backend_activity_cache` /
    `config.paths.backend_user_activity_cache`.
    """
    config = config or get_config()
    client = client or BackendApiClient(config.backend_api)
    resolver = resolver or ExternalIdentityResolver(resolve_path(config.paths.backend_identity_registry))
    cache_path = activity_cache_path or resolve_path(config.paths.backend_activity_cache)
    user_cache_path = user_activity_cache_path or resolve_path(config.paths.backend_user_activity_cache)
    user_activity_store = load_user_activity_cache_store(user_cache_path)

    catalog = load_backend_catalog(client, resolver)

    activities = sync_activities(
        client,
        cache_path,
        bootstrap_max_pages=config.backend_api.activity_bootstrap_max_pages,
        delta_max_pages=config.backend_api.activity_delta_max_pages,
        max_rows=config.backend_api.activity_cache_max_rows,
    )
    interactions, activity_guid_by_internal = load_backend_events(activities, resolver, catalog)

    roster_users, roster_guid_by_internal = load_backend_users_roster(client, resolver, catalog)
    # Only enrich users the roster call did NOT already cover (normally
    # none, or very few - a roster/activity-stream inconsistency) via the
    # old per-user `GET /api/users/{guid}` loop: that loop now being live
    # (rather than auth-blocked) turned it into its OWN 500+-request
    # startup cost once every activity-stream user got enriched
    # individually - exactly the kind of per-user-request explosion this
    # phase exists to avoid, and made entirely redundant by the roster
    # call already providing the same (and more complete) data for every
    # user in ~6 requests total (docs/data-mapping.md 19.13).
    roster_ids = {u.id for u in roster_users}
    activity_only_guid_by_internal = {
        uid: guid for uid, guid in activity_guid_by_internal.items() if uid not in roster_ids
    }
    activity_users = load_backend_users(client, activity_only_guid_by_internal, catalog)
    raw_users = _merge_users(roster_users, activity_users)
    guid_by_internal = {**roster_guid_by_internal, **activity_guid_by_internal}

    raw_reviews = load_backend_reviews(client, catalog, guid_by_internal)

    resolver.save()

    products_adapter = InMemoryProductCatalogAdapter(
        catalog.categories, catalog.tags, catalog.products, catalog.product_tags
    )
    users_adapter = InMemoryUserAdapter(raw_users, catalog.categories)
    reviews_adapter = InMemoryReviewAdapter(raw_reviews)
    events_adapter = LazyBackendUserEventsAdapter(
        interactions,
        client=client,
        resolver=resolver,
        catalog=catalog,
        guid_by_internal=guid_by_internal,
        max_pages=config.backend_api.activity_user_history_max_pages,
        max_rows=config.backend_api.activity_cache_max_rows,
        store=user_activity_store,
        store_path=user_cache_path,
    )

    logger.info(
        "backend AdapterBundle ready: %d products, %d users (%d via activity stream, %d via roster), "
        "%d interactions in the current bounded activity window (identity registry: %s)",
        len(catalog.products), len(raw_users), len(activity_users), len(roster_users),
        len(interactions), resolver.counts(),
    )
    return AdapterBundle(
        products=products_adapter,
        users=users_adapter,
        purchases=events_adapter,
        cart=events_adapter,
        clicks=events_adapter,
        reviews=reviews_adapter,
        search=events_adapter,
        chatbot=events_adapter,
    )
