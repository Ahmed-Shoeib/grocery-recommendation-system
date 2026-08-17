"""Wires the SQLite-backed adapters together into an `AdapterBundle`.

This is the SQLite counterpart of `adapters.factory.build_synthetic_adapters`
- same return type (`AdapterBundle`), same downstream consumers (feature
engineering, Two-Tower, the ranker, serving, the dashboard). Nothing
downstream of this function needs to know or care that the data came from
`data/sqlite/backend_shaped_synthetic.db` rather than the in-memory
synthetic generators:

    Today (synthetic V1):
      synthetic generators -> build_synthetic_adapters -> AdapterBundle -> ...

    This phase (SQLite integration):
      backend_shaped_synthetic.db -> build_sqlite_adapters -> AdapterBundle -> ...

    Later (real backend):
      real backend DB/API -> a new build_<real>_adapters -> AdapterBundle -> ...

Reuses the EXISTING `InMemoryProductCatalogAdapter`/`InMemoryUserAdapter`/
`InMemoryReviewAdapter` (fed with rows loaded from SQLite instead of the
synthetic generators) and the EXISTING `UserEventsAdapter`/
`build_user_events_adapters` (already built for exactly this "unified
event log -> five per-signal adapter roles" purpose) - no new adapter
classes were needed, only the SQL-to-Raw-object mapping in `data.sqlite
.loader`.

Purchase/cart double-counting note: this factory never reads `Cart`/
`Cart_Item` or `"Order"`/`Order_Item` - `User_events` (ADD_TO_CART/
PURCHASE rows) is the sole engagement-truth source consumed here, exactly
per the integration contract. Those four backend tables exist in the
database (and remain available for a future non-recommendation use), they
are simply never queried by this adapter path.
"""

from __future__ import annotations

from pathlib import Path

from recommendation.data.adapters.base import AdapterBundle
from recommendation.data.adapters.product_adapter import InMemoryProductCatalogAdapter
from recommendation.data.adapters.review_adapter import InMemoryReviewAdapter
from recommendation.data.adapters.user_adapter import InMemoryUserAdapter
from recommendation.data.adapters.user_events_adapter import build_user_events_adapters
from recommendation.data.sqlite.connection import open_readonly_connection
from recommendation.data.sqlite.loader import (
    load_categories,
    load_events,
    load_product_tags,
    load_products,
    load_reviews,
    load_tags,
    load_users,
)
from recommendation.utils.config import get_config, resolve_path


def build_sqlite_adapters(db_path: str | Path | None = None) -> AdapterBundle:
    """Builds a full `AdapterBundle` by reading `db_path` (defaulting to
    `config.paths.data_sqlite`) read-only, once, into memory - the same
    "load everything up front" pattern `build_synthetic_adapters` already
    uses, so this is a drop-in alternative construction path, not a
    different runtime strategy.
    """
    resolved_path = resolve_path(db_path) if db_path is not None else resolve_path(get_config().paths.data_sqlite)

    con = open_readonly_connection(resolved_path)
    try:
        categories = load_categories(con)
        tags = load_tags(con)
        products = load_products(con)
        product_tags = load_product_tags(con)
        users = load_users(con)
        reviews = load_reviews(con)
        events = load_events(con)
    finally:
        con.close()

    products_adapter = InMemoryProductCatalogAdapter(categories, tags, products, product_tags)
    users_adapter = InMemoryUserAdapter(users, categories)
    reviews_adapter = InMemoryReviewAdapter(reviews)

    return build_user_events_adapters(events, products_adapter, users_adapter, reviews_adapter)
