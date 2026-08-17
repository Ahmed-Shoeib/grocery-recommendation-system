"""Tests for the SQLite integration layer:
`recommendation.data.sqlite.*` + `adapters.sqlite_factory.build_sqlite_adapters`.

Runs against the real, committed `data/sqlite/backend_shaped_synthetic.db`
(deterministically generated - see `scripts/generate_backend_shaped_sqlite.py`)
rather than a synthetic in-memory fixture, since the whole point of this
layer is to prove it correctly reads that specific database. Tests look up
representative users dynamically (by querying for "any user with N events")
rather than hardcoding specific user ids, so they stay valid if the
database is ever regenerated with different tuning parameters.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from recommendation.data.adapters.engagement import build_engagement_profile
from recommendation.data.adapters.sqlite_factory import build_sqlite_adapters
from recommendation.data.sqlite.connection import open_readonly_connection
from recommendation.data.sqlite.loader import load_events, load_products, load_reviews, load_users
from recommendation.utils.config import get_config, resolve_path

DB_PATH = resolve_path(get_config().paths.data_sqlite)
pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="backend_shaped_synthetic.db not present")


def _raw_connection() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


# --- connection layer -------------------------------------------------------

def test_open_readonly_connection_reads_data():
    con = open_readonly_connection(DB_PATH)
    try:
        (count,) = con.execute("SELECT COUNT(*) FROM User").fetchone()
        assert count > 0
    finally:
        con.close()


def test_open_readonly_connection_blocks_writes():
    con = open_readonly_connection(DB_PATH)
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute("DELETE FROM User WHERE Id = 1")
    finally:
        con.close()


def test_open_readonly_connection_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_readonly_connection(tmp_path / "does_not_exist.db")


def test_build_sqlite_adapters_default_path_matches_config():
    bundle = build_sqlite_adapters()
    assert len(bundle.users.list_user_ids()) == 1000


# --- User mapping -------------------------------------------------------

def test_user_ids_match_sqlite():
    con = _raw_connection()
    sqlite_ids = {r["Id"] for r in con.execute("SELECT Id FROM User").fetchall()}
    con.close()

    bundle = build_sqlite_adapters(DB_PATH)
    assert set(bundle.users.list_user_ids()) == sqlite_ids


def test_user_age_group_preserved():
    con = _raw_connection()
    row = con.execute("SELECT Id, AgeGroup FROM User WHERE AgeGroup IS NOT NULL LIMIT 1").fetchone()
    con.close()

    bundle = build_sqlite_adapters(DB_PATH)
    profile = bundle.users.get_user_profile(row["Id"])
    assert profile is not None
    assert profile.age_group == row["AgeGroup"]


def test_user_preferred_category_resolved_from_id_to_name():
    con = _raw_connection()
    row = con.execute(
        "SELECT u.Id, c.Name FROM User u JOIN Category c ON u.PreferredCategoryId = c.Id LIMIT 1"
    ).fetchone()
    con.close()

    bundle = build_sqlite_adapters(DB_PATH)
    profile = bundle.users.get_user_profile(row["Id"])
    assert profile.preferred_category == row["Name"]


def test_user_with_null_preferred_category_and_age_group():
    con = _raw_connection()
    row = con.execute(
        "SELECT Id FROM User WHERE PreferredCategoryId IS NULL AND AgeGroup IS NULL LIMIT 1"
    ).fetchone()
    con.close()
    assert row is not None, "expected at least one incomplete-profile user in the fixture db"

    bundle = build_sqlite_adapters(DB_PATH)
    profile = bundle.users.get_user_profile(row["Id"])
    assert profile.preferred_category is None
    assert profile.age_group is None


def test_unknown_user_id_returns_none():
    bundle = build_sqlite_adapters(DB_PATH)
    assert bundle.users.get_user_profile(999_999_999) is None


# --- Product mapping -------------------------------------------------------

def test_product_fields_match_sqlite():
    con = _raw_connection()
    row = con.execute(
        "SELECT p.Id, p.Name, p.Brand, p.Price, p.StockQuantity, p.isActive, c.Name as CategoryName "
        "FROM Product p JOIN Category c ON p.CategoryId = c.Id LIMIT 1"
    ).fetchone()
    con.close()

    bundle = build_sqlite_adapters(DB_PATH)
    product = bundle.products.get_product(row["Id"])
    assert product is not None
    assert product.name == row["Name"]
    assert product.brand == row["Brand"]
    assert product.price == row["Price"]
    assert product.stock_quantity == row["StockQuantity"]
    assert product.is_active == bool(row["isActive"])
    assert product.category_name == row["CategoryName"]


def test_product_count_matches_sqlite():
    con = _raw_connection()
    (n,) = con.execute("SELECT COUNT(*) FROM Product").fetchone()
    con.close()

    bundle = build_sqlite_adapters(DB_PATH)
    assert len(bundle.products.list_products()) == n


def test_product_tags_are_populated():
    con = _raw_connection()
    row = con.execute(
        "SELECT ProductId FROM ProductTags GROUP BY ProductId HAVING COUNT(*) >= 2 LIMIT 1"
    ).fetchone()
    con.close()

    bundle = build_sqlite_adapters(DB_PATH)
    product = bundle.products.get_product(row["ProductId"])
    assert len(product.tags) >= 2


def test_eligibility_quadrants_present_via_products_adapter():
    """Sanity: SQLite product data reaches the canonical Product shape the
    pre-retrieval eligibility gate reads (isActive + stock_quantity) - all
    four active/stock combinations should be observable.
    """
    bundle = build_sqlite_adapters(DB_PATH)
    products = bundle.products.list_products()
    active_instock = [p for p in products if p.is_active and p.stock_quantity > 0]
    active_oos = [p for p in products if p.is_active and p.stock_quantity == 0]
    inactive_instock = [p for p in products if not p.is_active and p.stock_quantity > 0]
    inactive_oos = [p for p in products if not p.is_active and p.stock_quantity == 0]
    assert active_instock and active_oos and inactive_instock and inactive_oos


# --- User_events mapping (five signals) -------------------------------------

@pytest.mark.parametrize("action_type", ["CLICK", "ADD_TO_CART", "PURCHASE", "SEARCH", "CHATBOT"])
def test_each_action_type_reaches_the_right_canonical_records(action_type):
    con = _raw_connection()
    row = con.execute(
        "SELECT user_id, product_id, action_time FROM User_events WHERE action_type = ? LIMIT 1", (action_type,)
    ).fetchone()
    con.close()
    assert row is not None, f"expected at least one {action_type} event in the fixture db"

    bundle = build_sqlite_adapters(DB_PATH)
    user_id, product_id = row["user_id"], row["product_id"]

    if action_type == "CLICK":
        records = bundle.clicks.get_clicks(user_id)
        assert any(r.product_id == product_id for r in records)
    elif action_type == "ADD_TO_CART":
        records = bundle.cart.get_cart_items(user_id)
        assert any(r.product_id == product_id for r in records)
    elif action_type == "PURCHASE":
        records = bundle.purchases.get_purchases(user_id)
        assert any(r.product_id == product_id for r in records)
    elif action_type == "SEARCH":
        records = bundle.search.get_search_history(user_id)
        assert any(r.matched_product_id == product_id for r in records)
        assert all(r.search_term is None for r in records if r.matched_product_id == product_id)
    elif action_type == "CHATBOT":
        chatbot = bundle.chatbot.get_chatbot_context(user_id)
        assert chatbot is not None
        assert product_id in chatbot.mentioned_product_ids
        assert chatbot.summary is None
        assert chatbot.keywords == []


def test_action_time_is_parsed_and_preserved_as_datetime():
    events = load_events(_raw_connection())
    assert events
    assert all(isinstance(e.action_time, datetime) for e in events)

    con = _raw_connection()
    row = con.execute("SELECT action_time FROM User_events LIMIT 1").fetchone()
    con.close()
    expected = datetime.fromisoformat(row["action_time"])
    matching = [e for e in events if e.action_time == expected]
    assert matching


def test_no_event_loses_user_id_or_product_id():
    events = load_events(_raw_connection())
    assert events
    assert all(e.user_id is not None for e in events)
    assert all(e.product_id is not None for e in events)


def test_event_count_matches_sqlite():
    con = _raw_connection()
    (n,) = con.execute("SELECT COUNT(*) FROM User_events").fetchone()
    con.close()
    events = load_events(_raw_connection())
    assert len(events) == n


# --- No double-counting from Order_Item / Cart_Item -------------------------

def test_purchases_are_not_inflated_by_order_item():
    """User_events.PURCHASE is the sole purchase-engagement source. Even
    though Order_Item may contain a different row shape/grouping (orders
    group multiple purchases by day), the PurchaseAdapter's per-user
    purchase count must equal the distinct PURCHASE event count for that
    user, never the (unrelated) Order_Item row count.
    """
    con = _raw_connection()
    row = con.execute(
        "SELECT user_id, COUNT(*) as c FROM User_events WHERE action_type='PURCHASE' GROUP BY user_id ORDER BY c DESC LIMIT 1"
    ).fetchone()
    user_id, expected_count = row["user_id"], row["c"]
    con.close()

    bundle = build_sqlite_adapters(DB_PATH)
    purchases = bundle.purchases.get_purchases(user_id)
    assert len(purchases) == expected_count


def test_cart_items_are_not_inflated_by_cart_item_table():
    con = _raw_connection()
    row = con.execute(
        "SELECT user_id, COUNT(*) as c FROM User_events WHERE action_type='ADD_TO_CART' GROUP BY user_id ORDER BY c DESC LIMIT 1"
    ).fetchone()
    user_id, expected_count = row["user_id"], row["c"]
    con.close()

    bundle = build_sqlite_adapters(DB_PATH)
    cart_items = bundle.cart.get_cart_items(user_id)
    assert len(cart_items) == expected_count


def test_loader_module_never_queries_cart_or_order_tables():
    """Precise check (SQL `FROM <table>` clauses only, not prose mentions
    in docstrings/comments) that the loader never reads Cart_Item/
    Order_Item/"Order" - the actual guarantee against double-counting.
    """
    import inspect

    from recommendation.data.sqlite import loader

    source = inspect.getsource(loader)
    for forbidden in ("FROM Cart_Item", "FROM Order_Item", 'FROM "Order"'):
        assert forbidden not in source, f"loader.py must never query {forbidden}"


# --- Reviews -------------------------------------------------------

def test_reviews_load_and_are_not_a_sixth_action_type():
    con = _raw_connection()
    (n,) = con.execute("SELECT COUNT(*) FROM Review").fetchone()
    con.close()

    reviews = load_reviews(_raw_connection())
    assert len(reviews) == n

    con = _raw_connection()
    distinct_types = {r["action_type"] for r in con.execute("SELECT DISTINCT action_type FROM User_events").fetchall()}
    con.close()
    assert distinct_types == {"CLICK", "ADD_TO_CART", "PURCHASE", "SEARCH", "CHATBOT"}


# --- Empty-signal / NO_HISTORY behavior -------------------------------------

def test_no_history_user_produces_empty_engagement_profile():
    con = _raw_connection()
    row = con.execute(
        "SELECT Id FROM User WHERE Id NOT IN (SELECT DISTINCT user_id FROM User_events) LIMIT 1"
    ).fetchone()
    con.close()
    assert row is not None, "expected at least one NO_HISTORY user in the fixture db"

    bundle = build_sqlite_adapters(DB_PATH)
    profile = build_engagement_profile(
        row["Id"], bundle.users, bundle.purchases, bundle.cart, bundle.clicks, bundle.search, bundle.chatbot, bundle.reviews
    )
    assert profile.clicks == []
    assert profile.purchases == []
    assert profile.cart_items == []
    assert profile.searches == []
    assert profile.chatbot_context is None
    assert profile.reviews == [] or all(r.user_id == row["Id"] for r in profile.reviews)


# --- Referential integrity end-to-end ---------------------------------------

def test_full_engagement_profile_referential_ids_map_correctly():
    """Pick a STRONG user (>=5 events) and confirm every product_id
    referenced by their engagement resolves to a real product via the
    SAME adapter bundle - i.e. the whole User_events -> per-signal ->
    EngagementProfile chain is internally consistent.
    """
    con = _raw_connection()
    row = con.execute(
        "SELECT user_id, COUNT(*) c FROM User_events GROUP BY user_id HAVING c >= 5 LIMIT 1"
    ).fetchone()
    con.close()
    user_id = row["user_id"]

    bundle = build_sqlite_adapters(DB_PATH)
    profile = build_engagement_profile(
        user_id, bundle.users, bundle.purchases, bundle.cart, bundle.clicks, bundle.search, bundle.chatbot, bundle.reviews
    )
    all_product_ids = (
        [c.product_id for c in profile.clicks]
        + [p.product_id for p in profile.purchases]
        + [c.product_id for c in profile.cart_items]
        + [s.matched_product_id for s in profile.searches if s.matched_product_id is not None]
        + (profile.chatbot_context.mentioned_product_ids if profile.chatbot_context else [])
    )
    assert all_product_ids
    for pid in all_product_ids:
        assert bundle.products.get_product(pid) is not None
