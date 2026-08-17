"""Read-only SQLite connection helper.

The recommender only ever needs to READ the backend-shaped synthetic (and,
later, real-backend) database - it must never mutate it. `mode=ro` in the
SQLite URI enforces this at the OS/SQLite level (not just by convention):
any write attempt through a connection opened this way raises
`sqlite3.OperationalError`, verified in this module's test coverage.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def open_readonly_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open `db_path` read-only. Raises `FileNotFoundError` with a clear
    message if the file doesn't exist (rather than the more cryptic
    `sqlite3.OperationalError` SQLite itself raises for a missing file in
    `mode=ro`), and `sqlite3.Row` row factory so loader code can access
    columns by name instead of positional index.
    """
    resolved = Path(db_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"SQLite database not found: {resolved}")

    uri = f"file:{resolved.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    return con
