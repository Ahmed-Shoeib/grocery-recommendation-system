"""Wires the real-backend REST source into an `AdapterBundle`.

Third `AdapterBundle` factory, alongside `adapters.factory
.build_synthetic_adapters` and `adapters.sqlite_factory
.build_sqlite_adapters` - same return type, same downstream consumers
(feature engineering, Two-Tower, ranker, serving, dashboard), which never
learn that the data came from an HTTP API. Reuses the existing
`InMemoryProductCatalogAdapter` / `InMemoryUserAdapter` /
`InMemoryReviewAdapter` / `UserEventsAdapter` classes unchanged - the only
new code is `data.backend.*` (HTTP + DTO -> Raw* mapping + identity
resolution).

Load model: fetch the whole catalog + activity stream once, in memory,
exactly like `build_sqlite_adapters` reads the whole SQLite file once -
network calls live at this boundary only, never per feature or per
candidate. `RecommendationService.maybe_refresh` re-invokes this on the
configured TTL so `User_events`-style rows the backend records after
startup become visible without a restart.

Purchase/cart authoritative source: `/api/user-activities` (PlaceOrder /
AddToCart rows) is the sole engagement-truth source consumed here -
`/api/orders` and `/api/cart` are never read, so the same real-world
action cannot be double-counted through two code paths (mirrors the
SQLite factory's `User_events`-only contract).
"""

from __future__ import annotations

from recommendation.data.adapters.base import AdapterBundle
from recommendation.data.adapters.product_adapter import InMemoryProductCatalogAdapter
from recommendation.data.adapters.review_adapter import InMemoryReviewAdapter
from recommendation.data.adapters.user_adapter import InMemoryUserAdapter
from recommendation.data.adapters.user_events_adapter import build_user_events_adapters
from recommendation.data.backend.client import BackendApiClient
from recommendation.data.backend.identity import ExternalIdentityResolver
from recommendation.data.backend.loader import (
    load_backend_catalog,
    load_backend_events,
    load_backend_reviews,
    load_backend_users,
)
from recommendation.utils.config import AppConfig, get_config, resolve_path
from recommendation.utils.logging import get_logger

logger = get_logger(__name__)


def build_backend_api_adapters(
    config: AppConfig | None = None,
    *,
    client: BackendApiClient | None = None,
    resolver: ExternalIdentityResolver | None = None,
) -> AdapterBundle:
    """Build a full `AdapterBundle` from the backend REST API.

    `client` / `resolver` are injectable for tests; in production both are
    constructed from `config` (`config.backend_api` and
    `config.paths.backend_identity_registry`). The identity registry is
    persisted after the load so slug/GUID -> int assignments survive
    process restarts and refreshes.
    """
    config = config or get_config()
    client = client or BackendApiClient(config.backend_api)
    resolver = resolver or ExternalIdentityResolver(resolve_path(config.paths.backend_identity_registry))

    catalog = load_backend_catalog(client, resolver)
    activities = client.list_activities()
    interactions, guid_by_internal = load_backend_events(activities, resolver, catalog)
    raw_users = load_backend_users(client, guid_by_internal, catalog)
    raw_reviews = load_backend_reviews(client)

    resolver.save()

    products_adapter = InMemoryProductCatalogAdapter(
        catalog.categories, catalog.tags, catalog.products, catalog.product_tags
    )
    users_adapter = InMemoryUserAdapter(raw_users, catalog.categories)
    reviews_adapter = InMemoryReviewAdapter(raw_reviews)

    logger.info(
        "backend AdapterBundle ready: %d products, %d users, %d interactions (identity registry: %s)",
        len(catalog.products), len(raw_users), len(interactions), resolver.counts(),
    )
    return build_user_events_adapters(interactions, products_adapter, users_adapter, reviews_adapter)
