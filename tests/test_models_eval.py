import asyncio
import re

import pytest

from promptphoo import db, paths
from promptphoo.evaluator.run import evaluate
from promptphoo.models.eval import Eval, create_eval_id
from promptphoo.providers.simple import EchoProvider


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("IS_TESTING", raising=False)
    paths.set_config_directory_path(None)
    db.close_db()
    db.run_db_migrations()
    yield tmp_path
    db.close_db()


def _suite():
    return {
        "providers": [EchoProvider({"id": "echo"})],
        "prompts": [{"raw": "Hello {{name}}", "label": "greet"}],
        "tests": [
            {"vars": {"name": "World"}, "assert": [{"type": "contains", "value": "World"}]},
            {"vars": {"name": "Bob"}, "assert": [{"type": "equals", "value": "nope"}]},
        ],
    }


def test_create_eval_id_format():
    assert re.fullmatch(r"eval-[A-Za-z0-9]{3}-\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", create_eval_id())


def test_persisted_eval_round_trip(config_dir):
    suite = _suite()
    config = {"description": "smoke", "tests": suite["tests"], "tags": {"team": "x"}}
    record = Eval.create(config, suite["prompts"], author="me@example.com")
    asyncio.run(evaluate(suite, record, {"maxConcurrency": 2, "cache": False}))

    conn = db.get_db()
    assert conn.execute("select count(*) from eval_results").fetchone()[0] == 2
    assert conn.execute("select count(*) from evals_to_tags").fetchone()[0] == 1
    assert (config_dir / "evalLastWritten").read_text().startswith(record.id)

    loaded = Eval.find_by_id(record.id)
    assert loaded is not None
    assert loaded.author == "me@example.com"
    assert loaded.vars == ["name"]
    assert loaded.duration_ms is not None
    stats = loaded.get_stats()
    assert (stats["successes"], stats["failures"], stats["errors"]) == (1, 1, 0)

    out = loaded.to_results_file()
    assert out["version"] == 4
    assert out["config"]["description"] == "smoke"
    results = out["results"]["results"]
    assert [r["success"] for r in results] == [True, False]
    assert results[0]["response"]["output"] == "Hello World"
    assert results[1]["error"] == 'Expected output "Hello Bob" to equal "nope"'
    assert results[0]["provider"] == {"id": "echo"}
    assert results[0]["promptId"] == out["prompts"][0]["id"]


def test_in_memory_eval_keeps_results(config_dir):
    suite = _suite()
    record = Eval({"tests": suite["tests"]}, author=None)
    asyncio.run(evaluate(suite, record, {"cache": False}))
    assert not record.persisted
    assert len(record.results) == 2
    assert db.get_db().execute("select count(*) from evals").fetchone()[0] == 0
    summary = record.to_evaluate_summary()
    assert summary["version"] == 3
    assert summary["stats"]["successes"] == 1
