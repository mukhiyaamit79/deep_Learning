"""SQLite connection management.

* Short-lived connections: each operation opens and closes its own connection. A connection is
  never shared between threads (Streamlit sessions and the WebRTC video thread are different threads).
* ``PRAGMA foreign_keys = ON`` on EVERY connection (SQLite's default is OFF).
* Writes run inside ``BEGIN IMMEDIATE`` ... ``COMMIT`` and are rolled back on any exception.
* All SQL uses ``?`` / ``:name`` parameters.
* The video thread never writes to the database.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

Params = Sequence[Any] | Mapping[str, Any]


class DatabaseError(RuntimeError):
    """Database problem that the UI should show as a friendly message."""


class Database:
    """Lightweight handle around a SQLite file path. Holds no connection, so it is safe to share."""

    def __init__(self, path: Path, timeout_s: float = 10.0) -> None:
        self.path = Path(path)
        self.timeout_s = timeout_s

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if not self.path.parent.exists():
            raise DatabaseError(f"Database folder does not exist: {self.path.parent}. Restart the app.")
        conn = sqlite3.connect(str(self.path), timeout=self.timeout_s, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"PRAGMA busy_timeout = {int(self.timeout_s * 1000)}")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def query_all(self, sql: str, params: Params = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: Params = ()) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row is not None else None

    def scalar(self, sql: str, params: Params = (), default: Any = None) -> Any:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row is not None and row[0] is not None else default
