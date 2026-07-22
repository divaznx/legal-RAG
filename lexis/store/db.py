"""Connection handling and migration for the legal object model.

SQLite with WAL, because the product has to install with zero infrastructure
for the air-gapped deployments that are the point. WAL matters specifically:
extraction writes while dashboards read, and the default journal mode would
block the reader.

Nothing here is SQLite-specific beyond the PRAGMAs and the connect call, so
swapping in Postgres for multi-tenant hosting is a driver change.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from ..config import settings
from .schema import DDL, SCHEMA_VERSION

_local = threading.local()


def _database_path() -> Path:
    return Path(settings.data_dir) / "lexis.db"


def connect() -> sqlite3.Connection:
    """Thread-local connection.

    Thread-local rather than a shared singleton because FastAPI serves
    requests on a thread pool and SQLite connections are not safe to share
    across threads.
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        return conn

    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")       # readers never block on the extractor
    conn.execute("PRAGMA foreign_keys=ON")        # ON DELETE CASCADE must actually fire
    conn.execute("PRAGMA busy_timeout=5000")
    _local.conn = conn
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(DDL)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def reset() -> None:
    """Drop the store. Used by tests and by `cli.py store reset`."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
    path = _database_path()
    for suffix in ("", "-wal", "-shm"):
        target = Path(str(path) + suffix)
        target.unlink(missing_ok=True)
