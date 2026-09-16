"""Loads rows from the backend-shaped SQLite database into the SAME raw/
canonical schemas the synthetic generator path already uses.

Every function here does a `SELECT *`-equivalent read and maps each row
into an existing pydantic model - `RawCategory`/`RawTag`/`RawProduct`/
`RawProductTag`/`RawUser`/`RawReview` (`recommendation.synthetic
.raw_schemas` - despite the module path, these are ERD-shaped models, not
synthetic-specific ones; see that module's docstring) or `UserInteraction`
(`recommendation.schemas.events`). This is deliberate: it lets
`adapters.sqlite_factory.build_sqlite_adapters` hand the results straight
to the EXISTING `InMemoryProductCatalogAdapter`/`InMemoryUserAdapter`/
`InMemoryReviewAdapter`/`UserEventsAdapter` without a single new adapter
class - only the SQL-to-Raw-object mapping is new.

Authoritative-source note (see module docstring in `sqlite_factory.py` for
the full rationale): `Cart`/`Cart_Item` and `Order`/`Order_Item` are
INTENTIONALLY not loaded by anything here. `User_events` (ADD_TO_CART/
PURCHASE rows) is the sole engagement-truth source for this adapter path -
loading Cart_Item/Order_Item here as a second cart/purchase source would
risk exactly the double-counting the integration was told to avoid.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from recommendation.schemas.events import ActionType, UserInteraction
from recommendation.synthetic.raw_schemas import (
    RawCategory,
    RawProduct,
    RawProductTag,
    RawReview,
    RawTag,
    RawUser,
)


def load_categories(con: sqlite3.Connection) -> list[RawCategory]:
    rows = con.execute("SELECT Id, ParentId, Name FROM Category").fetchall()
    return [RawCategory(id=r["Id"], name=r["Name"], parent_id=r["ParentId"]) for r in rows]


def load_tags(con: sqlite3.Connection) -> list[RawTag]:
    rows = con.execute("SELECT Id, Name FROM Tag").fetchall()
    return [RawTag(id=r["Id"], name=r["Name"]) for r in rows]


def load_products(con: sqlite3.Connection) -> list[RawProduct]:
    rows = con.execute(
        "SELECT Id, CategoryId, Slug, Name, Description, Brand, Price, SalePrice, "
        "DiscountPercentage, StockQuantity, Ingredients, isActive, ProductImage, AltText FROM Product"
    ).fetchall()
    return [
        RawProduct(
            id=r["Id"], category_id=r["CategoryId"], slug=r["Slug"], name=r["Name"],
            description=r["Description"], brand=r["Brand"], price=r["Price"], sale_price=r["SalePrice"],
            discount_percentage=r["DiscountPercentage"], stock_quantity=r["StockQuantity"],
            ingredients=r["Ingredients"], is_active=bool(r["isActive"]), product_image=r["ProductImage"],
            alt_text=r["AltText"],
        )
        for r in rows
    ]


def load_product_tags(con: sqlite3.Connection) -> list[RawProductTag]:
    rows = con.execute("SELECT Id, ProductId, TagId FROM ProductTags").fetchall()
    return [RawProductTag(id=r["Id"], product_id=r["ProductId"], tag_id=r["TagId"]) for r in rows]


def _has_table(con: sqlite3.Connection, name: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def load_users(con: sqlite3.Connection) -> list[RawUser]:
    """Two supported schema shapes, chosen automatically per-database so a
    single loader works against both without a config flag:

    - `backend_shaped_synthetic.db` (legacy): `User.PreferredCategoryId` is
      a single FK (a modeling simplification of the real backend's
      `FavoriteCategory[]` join - see `synthetic.raw_schemas.RawUser`
      docstring) - wrapped into a length-<=1 list so this source produces
      the SAME canonical shape (`UserProfile.preferred_categories:
      list[str]`) the real backend_api source does.
    - A production-aligned dataset with a `UserFavoriteCategory(UserId,
      CategoryId)` join table (docs/production-feature-parity-audit.md -
      genuine multi-favorite semantics, matching the real backend's
      `FavoriteCategory[]` join shape exactly rather than a single-value
      approximation): every row for a user is collected into
      `preferred_category_ids`, in insertion order.

    Detected via `sqlite_master`, not a config flag, so the SAME code path
    serves both databases with no call-site branching required.
    """
    has_favorites_table = _has_table(con, "UserFavoriteCategory")

    rows = con.execute(
        "SELECT Id, FirstName, LastName, Email, PreferredCategoryId, AgeGroup FROM User"
    ).fetchall()

    favorites_by_user: dict[int, list[int]] = {}
    if has_favorites_table:
        for r in con.execute("SELECT UserId, CategoryId FROM UserFavoriteCategory ORDER BY UserId, Id"):
            favorites_by_user.setdefault(r["UserId"], []).append(r["CategoryId"])

    users: list[RawUser] = []
    for r in rows:
        if has_favorites_table:
            preferred_category_ids = favorites_by_user.get(r["Id"], [])
        else:
            preferred_category_ids = [r["PreferredCategoryId"]] if r["PreferredCategoryId"] is not None else []
        users.append(
            RawUser(
                id=r["Id"], first_name=r["FirstName"], last_name=r["LastName"], email=r["Email"],
                preferred_category_ids=preferred_category_ids, age_group=r["AgeGroup"],
            )
        )
    return users


def load_reviews(con: sqlite3.Connection) -> list[RawReview]:
    rows = con.execute("SELECT Id, UserId, ProductId, Rating, Comment, CreationDate FROM Review").fetchall()
    return [
        RawReview(
            id=r["Id"], user_id=r["UserId"], product_id=r["ProductId"], rating=r["Rating"],
            comment=r["Comment"], creation_date=_parse_timestamp(r["CreationDate"]),
        )
        for r in rows
    ]


def load_events(con: sqlite3.Connection) -> list[UserInteraction]:
    """Loads every `User_events` row - one row per action, per the
    confirmed backend contract - into a canonical `UserInteraction`.
    `action_time` is parsed into a real `datetime` here (not left as a raw
    string) so it survives the adapter boundary as an actual datetime;
    downstream, `features.recency` and `evaluation.temporal_future_purchase`
    both consume it.
    """
    rows = con.execute("SELECT id, user_id, product_id, action_time, action_type FROM User_events").fetchall()
    return [
        UserInteraction(
            user_id=r["user_id"], product_id=r["product_id"],
            action_type=ActionType(r["action_type"]), action_time=_parse_timestamp(r["action_time"]),
        )
        for r in rows
    ]


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parses `action_time` into the naive datetime every downstream
    consumer expects (`features.recency`, `evaluation
    .temporal_future_purchase` - see those modules' "naive datetimes
    throughout" docstrings). Required contract, adopted so a fresh
    `User_events` row (see `api.service.RecommendationService
    .maybe_refresh`) can never be misread as "in the future" purely
    because of a timezone-convention mismatch between the backend writer
    and this server's own clock (`serving.pipeline.recommend`'s
    `reference_time`, which uses this SAME convention):

    - A naive value (no offset/`Z`) is taken to ALREADY be UTC wall-clock
      time - the common backend convention (e.g. Python `datetime
      .utcnow()`, Postgres `now() AT TIME ZONE 'utc'`) - and used as-is.
      This is also byte-for-byte how every naive `action_time` already in
      `data/sqlite/backend_shaped_synthetic.db` has always been parsed,
      so existing data/tests are unaffected.
    - A value that DOES carry explicit offset/`Z` info is converted to
      UTC first, then stripped of tzinfo, landing in that exact same
      naive-UTC representation.
    """
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed
