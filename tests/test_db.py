import hashlib
import json
import shutil
import sqlite3

import pytest

from promptphoo import db


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", isolation_level=None)
    yield c
    c.close()


def test_read_migration_files_matches_journal():
    migrations = db.read_migration_files()
    journal = json.loads((db.MIGRATIONS_FOLDER / "meta" / "_journal.json").read_text())
    assert [m["tag"] for m in migrations] == [e["tag"] for e in journal["entries"]]
    first = migrations[0]
    raw = (db.MIGRATIONS_FOLDER / f"{first['tag']}.sql").read_text(encoding="utf-8")
    assert first["hash"] == hashlib.sha256(raw.encode()).hexdigest()
    assert first["folderMillis"] == journal["entries"][0]["when"]


def test_migrate_fresh_then_noop(conn):
    count = len(db.read_migration_files())
    assert db.migrate(conn) == count
    assert db.migrate(conn) == 0
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert {"evals", "eval_results", "prompts", "datasets", "tags", "evals_to_prompts"} <= tables
    rows = conn.execute("select hash, created_at from __drizzle_migrations").fetchall()
    assert len(rows) == count


def test_migrate_applies_only_newer_entries(conn, tmp_path):
    shutil.copytree(db.MIGRATIONS_FOLDER, tmp_path / "drizzle")
    journal_path = tmp_path / "drizzle" / "meta" / "_journal.json"
    journal = json.loads(journal_path.read_text())
    full = list(journal["entries"])
    journal["entries"] = full[:-1]
    journal_path.write_text(json.dumps(journal))
    assert db.migrate(conn, tmp_path / "drizzle") == len(full) - 1
    assert db.migrate(conn) == 1


def test_migrate_refuses_newer_database(conn):
    db.migrate(conn)
    conn.execute(
        'INSERT INTO "__drizzle_migrations" ("hash", "created_at") VALUES (?, ?)',
        ("future", 99999999999999),
    )
    with pytest.raises(db.DatabaseVersionError):
        db.migrate(conn)


def test_transaction_rolls_back(conn):
    conn.execute("create table t (x int)")
    with pytest.raises(RuntimeError), db.transaction(conn):
        conn.execute("insert into t values (1)")
        raise RuntimeError("boom")
    assert conn.execute("select count(*) from t").fetchone()[0] == 0
    with db.transaction(conn):
        conn.execute("insert into t values (2)")
        with db.transaction(conn):
            conn.execute("insert into t values (3)")
    assert conn.execute("select count(*) from t").fetchone()[0] == 2


def test_json_columns_use_js_semantics():
    assert db.to_db_json({"a": 1.0, "b": None}) == '{"a":1,"b":null}'
    assert db.to_db_json(None) is None
    assert db.from_db_json('{"a":1}') == {"a": 1}
