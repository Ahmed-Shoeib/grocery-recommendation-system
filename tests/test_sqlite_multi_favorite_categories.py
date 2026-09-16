"""Tests for `sqlite.loader.load_users`' two supported schema shapes
(section 11, docs/production-feature-parity-audit.md): the legacy single
`PreferredCategoryId` column (`backend_shaped_synthetic.db`) and the new
`UserFavoriteCategory` join table (production-aligned dataset,
`scripts/generate_production_aligned_sqlite.py`) - auto-detected via
`sqlite_master`, no config flag, no call-site branching.

Uses small in-memory/temp-file databases built inline (not the large
committed fixture databases) so these tests are fast and exercise the
schema-detection logic directly and deterministically.
"""

from __future__ import annotations

import sqlite3

from recommendation.sqlite.loader import load_users

_BASE_SCHEMA = """
CREATE TABLE Category (Id INTEGER PRIMARY KEY, ParentId INTEGER, Name TEXT NOT NULL);
CREATE TABLE User (
    Id INTEGER PRIMARY KEY, FirstName TEXT, LastName TEXT, Email TEXT NOT NULL,
    PreferredCategoryId INTEGER, AgeGroup TEXT
);
"""


def _connect_with_schema(extra_sql: str = "") -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(_BASE_SCHEMA + extra_sql)
    con.executemany("INSERT INTO Category (Id, ParentId, Name) VALUES (?,?,?)", [
        (1, None, "Fruits"), (2, None, "Packages"), (3, None, "vegetables"),
    ])
    return con


def test_legacy_single_column_schema_wraps_into_length_one_list():
    con = _connect_with_schema()
    con.execute(
        "INSERT INTO User (Id, FirstName, LastName, Email, PreferredCategoryId, AgeGroup) VALUES (1,'A','B','a@b.invalid',2,'25-34')"
    )
    con.execute(
        "INSERT INTO User (Id, FirstName, LastName, Email, PreferredCategoryId, AgeGroup) VALUES (2,'C','D','c@d.invalid',NULL,NULL)"
    )
    users = {u.id: u for u in load_users(con)}
    assert users[1].preferred_category_ids == [2]
    assert users[2].preferred_category_ids == []


def test_favorite_category_table_is_preferred_when_present():
    """When `UserFavoriteCategory` exists, it is used INSTEAD of the single
    `PreferredCategoryId` column, even if that column also has a value -
    the join table is the source of truth for the new schema shape.
    """
    con = _connect_with_schema(
        "CREATE TABLE UserFavoriteCategory (Id INTEGER PRIMARY KEY, UserId INTEGER, CategoryId INTEGER);"
    )
    con.execute(
        "INSERT INTO User (Id, FirstName, LastName, Email, PreferredCategoryId, AgeGroup) VALUES (1,'A','B','a@b.invalid',1,'25-34')"
    )
    con.executemany("INSERT INTO UserFavoriteCategory (Id, UserId, CategoryId) VALUES (?,?,?)", [
        (1, 1, 2), (2, 1, 3),
    ])
    users = {u.id: u for u in load_users(con)}
    # The join table (categories 2, 3) wins over PreferredCategoryId (1).
    assert users[1].preferred_category_ids == [2, 3]


def test_favorite_category_table_present_but_empty_for_user_yields_empty_list():
    con = _connect_with_schema(
        "CREATE TABLE UserFavoriteCategory (Id INTEGER PRIMARY KEY, UserId INTEGER, CategoryId INTEGER);"
    )
    con.execute(
        "INSERT INTO User (Id, FirstName, LastName, Email, PreferredCategoryId, AgeGroup) VALUES (1,'A','B','a@b.invalid',NULL,NULL)"
    )
    users = {u.id: u for u in load_users(con)}
    assert users[1].preferred_category_ids == []


def test_favorite_category_table_supports_zero_to_many_favorites_per_user():
    con = _connect_with_schema(
        "CREATE TABLE UserFavoriteCategory (Id INTEGER PRIMARY KEY, UserId INTEGER, CategoryId INTEGER);"
    )
    con.executemany(
        "INSERT INTO User (Id, FirstName, LastName, Email, PreferredCategoryId, AgeGroup) VALUES (?,?,?,?,?,?)",
        [(1, "A", "B", "a@b.invalid", None, None), (2, "C", "D", "c@d.invalid", None, None), (3, "E", "F", "e@f.invalid", None, None)],
    )
    con.executemany("INSERT INTO UserFavoriteCategory (Id, UserId, CategoryId) VALUES (?,?,?)", [
        (1, 2, 1), (2, 3, 1), (3, 3, 2), (4, 3, 3),
    ])
    users = {u.id: u for u in load_users(con)}
    assert users[1].preferred_category_ids == []  # user 1: zero favorites
    assert users[2].preferred_category_ids == [1]  # user 2: one favorite
    assert users[3].preferred_category_ids == [1, 2, 3]  # user 3: three favorites
