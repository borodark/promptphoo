"""SQLite access shared with Node promptfoo. Ports src/database/index.ts, src/database/signal.ts
and drizzle's libsql migrator (src/migrate.ts).

The schema is owned by the bundled drizzle migrations, so Node and Python can open the same
`<configDir>/promptfoo.db`. JSON columns are written with JSON.stringify semantics, booleans as
0/1 and `created_at` as epoch milliseconds where Node passes one explicitly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from promptphoo import envars
from promptphoo.jsvalues import json_stringify
from promptphoo.paths import get_config_directory_path

logger = logging.getLogger(__name__)

MIGRATIONS_FOLDER = Path(__file__).resolve().parent.parent / "_data" / "drizzle"
MIGRATIONS_TABLE = "__drizzle_migrations"
STATEMENT_BREAKPOINT = "--> statement-breakpoint"
DEFAULT_BUSY_TIMEOUT_MS = 5000

_conn: sqlite3.Connection | None = None
_lock = threading.RLock()


class DatabaseVersionError(RuntimeError):
    pass


def get_db_path() -> str:
    return os.path.join(get_config_directory_path(True), "promptfoo.db")


def get_db_signal_path() -> str:
    return os.path.join(get_config_directory_path(True), "evalLastWritten")


def _configure_connection(conn: sqlite3.Connection, busy_timeout_ms: int, wal: bool) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    if not wal or envars.get_env_bool("PROMPTFOO_DISABLE_WAL_MODE", False):
        return
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
        if mode.lower() == "wal":
            conn.execute("PRAGMA wal_autocheckpoint = 1000")
            conn.execute("PRAGMA synchronous = NORMAL")
        else:
            logger.warning(
                "Failed to enable WAL mode (got '%s'). Database performance may be reduced. This "
                "can happen on network filesystems. Set PROMPTFOO_DISABLE_WAL_MODE=true to "
                "suppress this warning.",
                mode,
            )
    except sqlite3.Error as e:
        logger.warning(
            "Error configuring SQLite WAL mode: %s. Database will use default journal mode. "
            "Performance may be reduced. This can happen on network filesystems or certain "
            "containerized environments. Set PROMPTFOO_DISABLE_WAL_MODE=true to suppress this "
            "warning.",
            e,
        )


def get_db() -> sqlite3.Connection:
    """Open (once) the shared connection in autocommit mode."""
    global _conn
    with _lock:
        if _conn is None:
            testing = envars.get_env_bool("IS_TESTING")
            conn = sqlite3.connect(
                ":memory:" if testing else get_db_path(),
                isolation_level=None,
                check_same_thread=False,
            )
            _configure_connection(conn, DEFAULT_BUSY_TIMEOUT_MS, wal=not testing)
            _conn = conn
        return _conn


def close_db() -> None:
    global _conn
    with _lock:
        if _conn is None:
            return
        try:
            if not envars.get_env_bool("IS_TESTING") and not envars.get_env_bool(
                "PROMPTFOO_DISABLE_WAL_MODE", False
            ):
                try:
                    _conn.execute("PRAGMA busy_timeout = 0")
                    _conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error as e:
                    logger.warning("Could not checkpoint WAL file before close: %s", e)
        finally:
            _conn.close()
            _conn = None


class transaction:  # noqa: N801 - used as `with transaction(conn):`
    """BEGIN/COMMIT around a block, ROLLBACK on error. Re-entrant via savepoints."""

    def __init__(self, conn: sqlite3.Connection, mode: str = "DEFERRED") -> None:
        self.conn = conn
        self.mode = mode
        self.nested = False

    def __enter__(self) -> sqlite3.Connection:
        _lock.acquire()
        self.nested = self.conn.in_transaction
        self.conn.execute("SAVEPOINT pf" if self.nested else f"BEGIN {self.mode}")
        return self.conn

    def __exit__(self, exc_type: Any, *_: Any) -> None:
        try:
            if exc_type is None:
                self.conn.execute("RELEASE pf" if self.nested else "COMMIT")
            elif self.nested:
                self.conn.execute("ROLLBACK TO pf")
                self.conn.execute("RELEASE pf")
            else:
                self.conn.execute("ROLLBACK")
        finally:
            _lock.release()


def to_db_json(value: Any) -> str | None:
    """Drizzle `mode: 'json'` columns: JSON.stringify on write, NULL for null/undefined."""
    if value is None:
        return None
    return json_stringify(value)


def from_db_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes | bytearray):
        value = value.decode("utf-8")
    return json.loads(value) if isinstance(value, str) else value


# --- migrations -----------------------------------------------------------------------------


def read_migration_files(folder: Path = MIGRATIONS_FOLDER) -> list[dict[str, Any]]:
    """drizzle-orm readMigrationFiles: one entry per journal tag, hashed over the raw file."""
    journal = json.loads((folder / "meta" / "_journal.json").read_text(encoding="utf-8"))
    migrations = []
    for entry in journal["entries"]:
        query = (folder / f"{entry['tag']}.sql").read_text(encoding="utf-8")
        migrations.append(
            {
                "sql": query.split(STATEMENT_BREAKPOINT),
                "bps": entry.get("breakpoints"),
                "folderMillis": entry["when"],
                "hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "tag": entry["tag"],
            }
        )
    return migrations


def _has_sql(statement: str) -> bool:
    lines = (line.strip() for line in statement.splitlines())
    return any(line and not line.startswith("--") for line in lines)


def migrate(conn: sqlite3.Connection, folder: Path = MIGRATIONS_FOLDER) -> int:
    """Apply journal entries newer than the newest recorded one. Returns the number applied.

    Mirrors drizzle's libsql migrator: compares only `created_at` of the last applied row with
    each journal `when`, and runs everything in one transaction with foreign keys disabled.
    """
    migrations = read_migration_files(folder)
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{MIGRATIONS_TABLE}" '
        "(id SERIAL PRIMARY KEY, hash text NOT NULL, created_at numeric)"
    )
    row = conn.execute(
        f'SELECT id, hash, created_at FROM "{MIGRATIONS_TABLE}" ORDER BY created_at DESC LIMIT 1'
    ).fetchone()
    last_applied = float(row[2]) if row and row[2] is not None else None
    newest_bundled = max(m["folderMillis"] for m in migrations)
    if last_applied is not None and last_applied > newest_bundled:
        raise DatabaseVersionError(
            "The promptfoo database was migrated by a newer promptfoo version than this "
            "promptphoo release supports. Upgrade promptphoo before writing to it."
        )
    pending = [m for m in migrations if last_applied is None or last_applied < m["folderMillis"]]
    if not pending:
        return 0
    with _lock:
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            with transaction(conn):
                for migration in pending:
                    for statement in migration["sql"]:
                        if _has_sql(statement):
                            conn.execute(statement)
                    conn.execute(
                        f'INSERT INTO "{MIGRATIONS_TABLE}" ("hash", "created_at") VALUES(?, ?)',
                        (migration["hash"], migration["folderMillis"]),
                    )
        finally:
            conn.execute("PRAGMA foreign_keys=ON")
    logger.debug("Applied %d database migrations", len(pending))
    return len(pending)


def run_db_migrations() -> None:
    try:
        migrate(get_db())
    except Exception as e:
        logger.error("Database migration failed: %s", e)
        raise


# --- change signal --------------------------------------------------------------------------


def _iso_now() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def update_signal_file(eval_id: str | None = None) -> None:
    """Touch `evalLastWritten` so a running Node viewer refreshes (legacy text form)."""
    now = _iso_now()
    content = f"{eval_id}:{now}" if eval_id and len(eval_id) >= 9 else now
    if eval_id and len(eval_id) < 9:
        content = json_stringify(
            {"type": "update", "evalId": eval_id, "updatedEvalIds": [eval_id], "timestamp": now}
        )
    try:
        Path(get_db_signal_path()).write_text(content, encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to write database signal file: %s", e)
