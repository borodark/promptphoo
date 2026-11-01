import asyncio
import re

import pytest

from promptphoo import db, paths
from promptphoo.evaluator.run import evaluate
from promptphoo.models.eval import Eval, EvalResult, create_eval_id
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


def _reverse_row_order(eval_id):
    # Reinsert the rows newest-first, as a concurrent run can store them.
    conn = db.get_db()
    conn.execute("CREATE TEMP TABLE r AS SELECT * FROM eval_results WHERE eval_id = ?", (eval_id,))
    conn.execute("DELETE FROM eval_results WHERE eval_id = ?", (eval_id,))
    conn.execute("INSERT INTO eval_results SELECT * FROM r ORDER BY test_idx DESC, prompt_idx DESC")
    conn.execute("DROP TABLE r")
    conn.commit()


def test_results_load_in_test_and_prompt_order(config_dir):
    # D70: Node reads eval_results without ORDER BY; promptphoo orders them.
    suite = _suite()
    suite["prompts"].append({"raw": "Bye {{name}}", "label": "bye"})
    record = Eval.create({"tests": suite["tests"]}, suite["prompts"])
    asyncio.run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    _reverse_row_order(record.id)
    stored = db.get_db().execute(
        "SELECT test_idx, prompt_idx FROM eval_results WHERE eval_id = ? ORDER BY rowid",
        (record.id,),
    )
    assert [tuple(r) for r in stored] == [(1, 1), (1, 0), (0, 1), (0, 0)]

    loaded = Eval.find_by_id(record.id)
    assert loaded is not None
    expected = [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [(r.test_idx, r.prompt_idx) for r in loaded.load_results()] == expected
    batches = list(EvalResult.find_many_by_eval_id_batched(record.id, batch_size=1))
    assert [[(r.test_idx, r.prompt_idx) for r in b] for b in batches] == [
        expected[:2],
        expected[2:],
    ]
    by_test = asyncio.run(loaded.read_results_by_test_idx(1))
    assert [r.prompt_idx for r in by_test] == [0, 1]
    rows = loaded.to_results_file()["results"]["results"]
    assert [(r["testIdx"], r["promptIdx"]) for r in rows] == expected
