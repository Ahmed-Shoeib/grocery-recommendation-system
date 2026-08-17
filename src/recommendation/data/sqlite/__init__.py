"""SQLite data-access layer for the backend-shaped synthetic database.

This package is the ONLY place in the codebase that knows SQL/sqlite3 -
`recommendation.data.adapters.sqlite_factory.build_sqlite_adapters` uses it
to populate the same `AdapterBundle` interface type the synthetic path
already produces, so feature engineering, Two-Tower, the ranker, and
serving never know (or need to know) that a request originated from a
SQLite file rather than the in-memory synthetic generators.

The database this package targets, `data/sqlite/backend_shaped_synthetic.db`
(see `scripts/generate_backend_shaped_sqlite.py`), is entirely SYNTHETIC -
a backend-ERD-shaped stand-in for the eventual real backend, not real user/
purchase data. Access here is read-only by construction (see
`connection.open_readonly_connection`) - this layer must never mutate that
dataset.
"""
