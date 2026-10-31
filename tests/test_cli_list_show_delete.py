"""`list`, `show` and `delete`, plus their DB queries.

Ports test/commands/list.test.ts, show.test.ts, delete.test.ts and test/util/
databaseDelete.test.ts. The golden tests replay Node-written DB rows from
tests/fixtures/list_show/golden.json (tools/list_show_golden.py) and require promptphoo's
output to match Node's byte for byte. The interop tests at the bottom run real Node and are
skipped unless NODE_PROMPTFOO is set.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from promptphoo import db, paths
from promptphoo.cli import delete as delete_mod
from promptphoo.cli import list_ as list_mod
from promptphoo.cli import show as show_mod
from promptphoo.cli.main import build_cli, setup_logging
from promptphoo.db import queries
from promptphoo.evaluator.run import evaluate, sha256
from promptphoo.jsvalues import json_stringify
from promptphoo.models.eval import Eval
from promptphoo.providers.simple import EchoProvider

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "list_show" / "golden.json").read_text("utf-8")
)
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.delenv("IS_TESTING", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    paths.set_config_directory_path(None)
    db.close_db()
    db.run_db_migrations()
    db.close_db()
    setup_logging()
    yield tmp_path / "home"
    db.close_db()


def load_rows(config_dir: Path, stage: dict[str, Any]) -> None:
    conn = sqlite3.connect(config_dir / "promptfoo.db")
    conn.execute("PRAGMA foreign_keys = ON")
    with conn:
        for table, data in stage.items():
            cols = ", ".join(f'"{c}"' for c in data["columns"])
            marks = ", ".join("?" for _ in data["columns"])
            conn.executemany(f'INSERT INTO "{table}" ({cols}) VALUES ({marks})', data["rows"])
    conn.close()


def invoke(*args: str, input: str | None = None) -> tuple[int, str]:  # noqa: A002
    try:
        result = CliRunner().invoke(build_cli(list(args)), list(args), input=input)
    finally:
        db.close_db()
    if result.exception and not isinstance(result.exception, SystemExit):
        raise result.exception
    return result.exit_code, result.output


def assert_same(want: str, got: str) -> None:
    if want != got:
        diff = "".join(difflib.unified_diff(want.splitlines(1), got.splitlines(1), "node", "py"))
        pytest.fail("output differs from Node:\n" + diff)


def _cases(stage: str) -> list[dict[str, Any]]:
    return [c for c in GOLDEN["cases"] if c["stage"] == stage]


# --- golden output from Node ----------------------------------------------------------------


@pytest.mark.parametrize(
    "case", [c for c in GOLDEN["cases"] if c["stage"] != "deletes"], ids=lambda c: c["name"]
)
def test_matches_node_golden(case, config_dir):
    if case["stage"] != "empty":
        load_rows(config_dir, GOLDEN["stages"][case["stage"]])
    code, output = invoke(*case["args"])
    assert_same(case["output"], output)
    assert code == case["exitCode"]


def test_delete_sequence_matches_node_golden(config_dir):
    load_rows(config_dir, GOLDEN["stages"]["stage2"])
    for case in _cases("deletes"):
        code, output = invoke(*case["args"])
        assert_same(case["output"], output)
        assert code == case["exitCode"], case["name"]
    conn = sqlite3.connect(config_dir / "promptfoo.db")
    counts = {t: conn.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
              for t in GOLDEN["countsAfterDeletes"]}  # fmt: skip
    conn.close()
    assert counts == GOLDEN["countsAfterDeletes"]


# --- list ------------------------------------------------------------------------------------


def _suite(prompts: list[str], tests: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "providers": [EchoProvider({"id": "echo"})],
        "prompts": [{"raw": p, "label": p} for p in prompts],
        "tests": tests,
    }


def make_eval(prompts: list[str], tests: list[dict[str, Any]], created_at: int, **config) -> Eval:
    suite = _suite(prompts, tests)
    record = Eval.create({"tests": tests, **config}, suite["prompts"], created_at=created_at)
    asyncio.run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    db.close_db()
    return record


T0 = 1_700_000_000_000


def test_list_registers_subcommands(config_dir):
    cli = build_cli(["list"])
    group = cli.commands["list"]
    assert sorted(group.commands) == ["datasets", "evals", "prompts"]  # type: ignore[attr-defined]
    params = {p.name for p in group.commands["evals"].params}  # type: ignore[attr-defined]
    assert {"n", "ids_only", "env_path"} <= params


def test_list_evals_ids_only_newest_first(config_dir):
    a = make_eval(["A {{x}}"], [{"vars": {"x": "1"}}], T0)
    b = make_eval(["B {{x}}"], [{"vars": {"x": "2"}}], T0 + 1000)
    code, out = invoke("list", "evals", "--ids-only")
    assert code == 0
    assert out.splitlines() == [b.id, a.id]
    _, out = invoke("list", "evals", "--ids-only", "-n", "1")
    assert out.splitlines() == [b.id]


def test_list_evals_table_oldest_first_with_vars(config_dir, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "0")
    a = make_eval(["A {{x}} {{y}}"], [{"vars": {"y": "1", "x": "2"}}], T0, description="first")
    make_eval(["B"], [{}], T0 + 1000)
    code, out = invoke("list", "evals")
    assert code == 0
    lines = out.splitlines()
    assert lines[3].startswith(f"│ {a.id}")
    assert "│ first " in lines[3]
    assert f"│ {sha256('A {{x}} {{y}}')[:6]}   │" in lines[3]
    assert "│ x, y       │" in lines[3]
    assert lines[-2] == "Run promptfoo show eval <id> to see details of a specific evaluation."
    assert lines[-1] == "Run promptfoo show prompt <id> to see details of a specific prompt."


def test_list_prompts_and_datasets_sorted_by_recent_eval(config_dir):
    tests = [{"vars": {"x": "1"}}]
    e1 = make_eval(["P1"], tests, T0)
    e2 = make_eval(["P2"], [{"vars": {"x": "2"}}], T0 + 1000)
    _, out = invoke("list", "prompts", "--ids-only")
    ids = out.splitlines()
    by_recent = sorted(
        [(e1.id, sha256("P1")), (e2.id, sha256("P2"))],
        key=lambda t: queries.locale_compare_key(t[0]),
    )
    assert ids == [p for _, p in by_recent]
    _, out = invoke("list", "datasets", "--ids-only")
    expected = sorted(
        [(e.id, _stored_dataset_id(e.id)) for e in (e1, e2)],
        key=lambda t: queries.locale_compare_key(t[0]),
        reverse=True,
    )
    assert out.splitlines() == [d for _, d in expected]


def _stored_dataset_id(eval_id: str) -> str:
    # The evaluator fills in each test's defaults in place, and the stored config keeps them.
    record = Eval.find_by_id(eval_id)
    db.close_db()
    assert record is not None
    return sha256(json_stringify(record.config["tests"]))


def test_list_datasets_best_prompt_is_highest_score(config_dir, monkeypatch):
    def metrics(score: float) -> dict[str, Any]:
        return {"score": score, "testPassCount": 0, "testFailCount": 0, "testErrorCount": 0}

    dataset = {
        "id": "dataset-1",
        "testCases": [],
        "count": 3,
        "recentEvalDate": "2025-01-10T00:00:00.000Z",
        "recentEvalId": "eval-10",
        "prompts": [
            {"id": "low-001", "evalId": "eval-10", "prompt": {"metrics": metrics(0.3)}},
            {"id": "high-001", "evalId": "eval-10", "prompt": {"metrics": metrics(0.9)}},
        ],
    }
    monkeypatch.setattr(queries, "get_test_cases", lambda limit=None: [dataset])
    captured: list[Any] = []
    monkeypatch.setattr(
        "promptphoo.output.table.wrap_table", lambda rows, widths=None: captured.append(rows) or ""
    )
    list_mod.list_datasets(None, False, show_mod.ExitCode())
    assert captured[0][0] == {
        "dataset id": "datase",
        "best prompt": "high-0",
        "evals": 3,
        "prompts": 2,
        "recent eval": "eval-10",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None), ("abc", None), ("0", None), ("", None), ("5", 5), (" 7 ", 7),
        ("1e1", 10), ("0x10", 16), ("2.5", 2.5), ("Infinity", -1), ("inf", None),
    ],
)  # fmt: skip
def test_parse_limit_follows_number_semantics(raw, expected):
    assert list_mod.parse_limit(raw) == expected


# --- show ------------------------------------------------------------------------------------


def test_metric_formatting():
    m = {"testPassCount": 18, "testFailCount": 5, "testErrorCount": 4, "score": 18.125}
    assert show_mod.format_pass_rate(m) == "66.67%"
    assert show_mod.format_pass_count(m) == "18"
    assert show_mod.format_fail_count(m) == "5 (+4 errors)"
    assert show_mod._raw_score(m) == "18.13"  # toFixed rounds half up on the exact value
    assert show_mod._raw_score({"score": 0}) == "0.00"
    assert show_mod._raw_score({}) == "-"
    assert show_mod.format_pass_rate(None) == "-"
    assert show_mod.format_pass_rate({"testPassCount": 0, "testFailCount": 0,
                                      "testErrorCount": 0}) == "-"  # fmt: skip
    assert show_mod.format_pass_rate({"testPassCount": 1}) == "-"
    assert show_mod.format_fail_count(None) == "-"
    assert show_mod.format_fail_count({"testFailCount": 2}) == "2"


def test_show_python_written_eval(config_dir, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "0")
    record = make_eval(["Hi {{n}}"], [{"vars": {"n": "Ann"}}], T0)
    code, out = invoke("show", record.id)
    assert code == 0
    assert "│ n" in out and "[PASS] Hi Ann" in out
    assert out.splitlines()[-1] == "1 variables: n"
    assert f"\nEval {record.id}\n" in out
    assert invoke("show", "eval", record.id) == (code, out)


def test_show_prefix_routes_to_find_by_id(config_dir):
    record = make_eval(["Hi"], [{}], T0)
    code, out = invoke("show", record.id[:8])
    assert code == 1
    assert out.startswith(f"No evaluation found with ID {record.id[:8]}\n")


def test_show_eval_more_than_25_rows(config_dir):
    tests = [{"vars": {"i": str(i)}} for i in range(27)]
    record = make_eval(["{{i}}"], tests, T0)
    _, out = invoke("show", "eval", record.id)
    assert "... 2 more rows not shown ...\n\n" in out
    tests = [{"vars": {"i": str(i)}} for i in range(26)]
    record = make_eval(["{{i}}"], tests, T0 + 1000)
    _, out = invoke("show", "eval", record.id)
    assert "... 1 more row not shown ...\n\n" in out


def test_show_eval_prompt_id_preview(config_dir):
    prompts = [f"P{i}" for i in range(7)]
    record = make_eval(prompts, [{}], T0)
    _, out = invoke("show", "eval", record.id)
    ids = [p["id"] for p in Eval.find_by_id(record.id).prompts]  # type: ignore[union-attr]
    db.close_db()
    assert f"Prompt IDs: {', '.join(ids[:5])} (and 2 more...)" in out


# --- delete ----------------------------------------------------------------------------------


def _eval_ids() -> list[str]:
    return [r[0] for r in _query("SELECT id FROM evals ORDER BY created_at")]


def test_delete_by_id_and_errors(config_dir):
    record = make_eval(["Hi"], [{}], T0)
    code, out = invoke("delete", record.id)
    assert code == 0
    assert out == f"Evaluation with ID {record.id} has been successfully deleted.\n"
    code, out = invoke("delete", record.id)
    assert (code, out) == (1, f"No resource found with ID {record.id}\n")
    code, out = invoke("delete", "eval", "nope")
    assert code == 1
    assert out == "Could not delete evaluation with ID nope:\nError: Eval with ID nope not found\n"


def test_delete_eval_latest(config_dir):
    a = make_eval(["A"], [{}], T0)
    b = make_eval(["B"], [{}], T0 + 1000)
    code, out = invoke("delete", "eval", "latest")
    assert (code, out) == (0, f"Evaluation with ID {b.id} has been successfully deleted.\n")
    assert _eval_ids() == [a.id]
    db.close_db()


def test_delete_eval_all_prompts_for_confirmation(config_dir):
    make_eval(["A"], [{}], T0)
    code, out = invoke("delete", "eval", "all", input="n\n")
    assert code == 0
    assert out == f"? {delete_mod.CONFIRM_DELETE_ALL} (Y/n) "
    assert len(_eval_ids()) == 1
    db.close_db()
    code, out = invoke("delete", "eval", "all", input="")  # end of input: Node exits 1
    assert code == 1
    assert len(_eval_ids()) == 1
    db.close_db()
    code, out = invoke("delete", "eval", "all", input="\n")  # inquirer defaults to yes
    assert out.endswith("All evaluations have been deleted.\n")
    assert _eval_ids() == []
    db.close_db()


def test_delete_eval_all_yes_skips_prompt(config_dir):
    make_eval(["A"], [{}], T0)
    code, out = invoke("delete", "eval", "all", "--yes")
    assert (code, out) == (0, "All evaluations have been deleted.\n")
    signal = json.loads((config_dir / "evalLastWritten").read_text())
    assert signal["type"] == "delete" and "deletedEvalIds" not in signal


# --- databaseDelete.test.ts ----------------------------------------------------------------


def _add_trace(eval_id: str, trace_id: str, spans: int = 1) -> None:
    conn = db.get_db()
    conn.execute(
        "INSERT INTO traces (id, trace_id, evaluation_id, test_case_id) VALUES (?, ?, ?, ?)",
        (f"id-{trace_id}", trace_id, eval_id, "test-case-id"),
    )
    for i in range(spans):
        conn.execute(
            "INSERT INTO spans (id, trace_id, span_id, name, start_time) VALUES (?, ?, ?, ?, ?)",
            (f"{trace_id}-{i}", trace_id, f"{trace_id}-span-{i}", "test-span", 1 + i),
        )


def _query(sql: str) -> list[tuple[Any, ...]]:
    cur = db.get_db().cursor()
    cur.row_factory = None  # Eval.find_by_id leaves a dict row factory on the connection
    return cur.execute(sql).fetchall()


def _count(table: str) -> int:
    return _query(f"SELECT count(*) FROM {table}")[0][0]


def _bare_eval(created_at: int) -> str:
    return Eval.create({}, [], created_at=created_at, author=None).id


def test_delete_eval_removes_traces_spans_and_blob_refs(config_dir):
    eval_id = _bare_eval(T0)
    _add_trace(eval_id, "trace-single", spans=2)
    conn = db.get_db()
    conn.execute(
        "INSERT INTO blob_assets (hash, size_bytes, mime_type, provider) VALUES (?, ?, ?, ?)",
        ("h1", 3, "image/png", "filesystem"),
    )
    conn.execute(
        "INSERT INTO blob_references (id, blob_hash, eval_id) VALUES (?, ?, ?)",
        ("r1", "h1", eval_id),
    )
    queries.delete_eval(eval_id)
    assert Eval.find_by_id(eval_id) is None
    assert (_count("traces"), _count("spans"), _count("blob_references")) == (0, 0, 0)
    assert _count("blob_assets") == 1  # Node leaves the asset for blob GC
    signal = json.loads((config_dir / "evalLastWritten").read_text())
    assert signal["deletedEvalIds"] == [eval_id]


def test_delete_evals_only_selected(config_dir):
    ids = [_bare_eval(T0 + i) for i in range(3)]
    for i, eval_id in enumerate(ids):
        _add_trace(eval_id, f"trace-{i}")
    queries.delete_evals(ids[:2])
    assert [r[0] for r in _query("SELECT trace_id FROM traces")] == ["trace-2"]
    assert [r[0] for r in _query("SELECT trace_id FROM spans")] == ["trace-2"]
    assert _eval_ids() == [ids[2]]


def test_delete_evals_empty_list_writes_no_signal(config_dir):
    signal = config_dir / "evalLastWritten"
    signal.unlink(missing_ok=True)
    queries.delete_evals([])
    assert not signal.exists()


def test_delete_all_evals_removes_traces(config_dir):
    for i in range(2):
        _add_trace(_bare_eval(T0 + i), f"trace-all-{i}")
    queries.delete_all_evals()
    assert Eval.get_many() == []
    assert (_count("traces"), _count("spans")) == (0, 0)


def test_delete_missing_eval_raises(config_dir):
    with pytest.raises(LookupError, match="Eval with ID nope not found"):
        queries.delete_eval("nope")


# --- query helpers -------------------------------------------------------------------------


def test_get_eval_from_id_prefers_newest(config_dir):
    _bare_eval(T0)
    newest = _bare_eval(T0 + 1000)
    found = queries.get_eval_from_id("eval-")
    assert found is not None and found["id"] == newest


def test_get_prompts_groups_by_raw(config_dir):
    tests = [{"vars": {"x": "1"}}]
    make_eval(["Same", "Other"], tests, T0)
    newer = make_eval(["Same"], tests, T0 + 1000)
    prompts = {p["prompt"]["raw"]: p for p in queries.get_prompts()}
    same = prompts["Same"]
    assert same["count"] == 2
    assert same["recentEvalId"] == newer.id
    assert same["recentEvalDate"] == "2023-11-14T22:13:21.000Z"
    assert [e["id"] for e in same["evals"]][0] == newer.id
    by_hash = queries.get_prompts_for_test_cases_hash(_stored_dataset_id(newer.id))
    assert {p["prompt"]["raw"] for p in by_hash} == {"Same", "Other"}
    datasets = queries.get_test_cases()
    assert len(datasets) == 1 and datasets[0]["count"] == 2
    assert [p["prompt"]["raw"] for p in datasets[0]["prompts"]] == ["Same", "Other"]


def test_get_vars_from_evals_sorted(config_dir):
    record = make_eval(["{{b}}{{a}}"], [{"vars": {"b": "1", "a": "2"}}], T0)
    assert queries.get_vars_from_evals([record]) == {record.id: ["a", "b"]}
    assert queries.get_vars_from_evals([]) == {}


def test_locale_compare_key_orders_like_icu():
    ids = ["eval-IcN-x", "eval-epU-x", "eval-5JM-x", "eval-icN-x"]
    assert sorted(ids, key=queries.locale_compare_key) == [
        "eval-5JM-x", "eval-epU-x", "eval-icN-x", "eval-IcN-x",
    ]  # fmt: skip


# --- interop with real Node (NODE_PROMPTFOO) -------------------------------------------------

NODE_PROMPTFOO = os.environ.get("NODE_PROMPTFOO")
needs_node = pytest.mark.skipif(not NODE_PROMPTFOO, reason="NODE_PROMPTFOO is not set")
FOOTER = "\n\nThere were some errors during the operation."


def _run(cmd: list[str], config_dir: Path, cwd: Path, *args: str) -> tuple[int, str]:
    env = {**os.environ, "PROMPTFOO_CONFIG_DIR": str(config_dir),
           "PROMPTFOO_DISABLE_TELEMETRY": "1", "PROMPTFOO_DISABLE_UPDATE": "1",
           "PYTHONPATH": str(ROOT / "src")}  # fmt: skip
    env.pop("FORCE_COLOR", None)
    env.pop("IS_TESTING", None)
    proc = subprocess.run(
        [*cmd, *args], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, text=True, timeout=300,
    )  # fmt: skip
    out = proc.stdout
    return proc.returncode, out.split(FOOTER)[0] + "\n" if FOOTER in out else out


NODE = shlex.split(NODE_PROMPTFOO or "")
PY = [sys.executable, "-m", "promptphoo"]
CONFIG = """\
description: interop {{n}}
prompts: ["Hello {{name}}", "Bye {{name}}"]
providers: [echo]
tests:
  - vars: {name: World}
    assert: [{type: contains, value: World}]
  - vars: {name: Bob}
    assert: [{type: equals, value: nope}]
"""
READS = [
    ["list", "evals"], ["list", "prompts"], ["list", "datasets"], ["show"],
    ["list", "evals", "--ids-only"], ["show", "prompt", sha256("Hello {{name}}")[:6]],
]  # fmt: skip


def _populate(writer: list[str], tmp_path: Path) -> Path:
    home = tmp_path / "home"
    (tmp_path / "c.yaml").write_text(CONFIG)
    for _ in range(2):
        _run(writer, home, tmp_path, "eval", "-c", "c.yaml", "--no-cache", "--no-progress-bar")
        time.sleep(1.1)
    return home


@needs_node
@pytest.mark.parametrize("writer", ["node", "python"])
def test_interop_reads_match(writer, tmp_path):
    home = _populate(NODE if writer == "node" else PY, tmp_path)
    for args in READS:
        node_out = _run(NODE, home, tmp_path, *args)
        py_out = _run(PY, home, tmp_path, *args)
        assert py_out[0] == node_out[0], args
        assert_same(node_out[1], py_out[1])


@needs_node
def test_interop_deletes_remove_the_same_rows(tmp_path):
    home = _populate(NODE, tmp_path)
    conn = sqlite3.connect(home / "promptfoo.db")
    with conn:
        eval_id = conn.execute("SELECT id FROM evals ORDER BY created_at LIMIT 1").fetchone()[0]
        conn.execute("INSERT INTO traces (id, trace_id, evaluation_id, test_case_id) "
                     "VALUES ('t', 't1', ?, 'tc')", (eval_id,))  # fmt: skip
        conn.execute("INSERT INTO spans (id, trace_id, span_id, name, start_time) "
                     "VALUES ('s', 't1', 's1', 'n', 1)")  # fmt: skip
        conn.execute("INSERT INTO blob_assets (hash, size_bytes, mime_type, provider) "
                     "VALUES ('h', 1, 'image/png', 'filesystem')")  # fmt: skip
        conn.execute("INSERT INTO blob_references (id, blob_hash, eval_id) VALUES ('r', 'h', ?)",
                     (eval_id,))  # fmt: skip
    conn.close()
    copies = {}
    for name, cmd in (("node", NODE), ("python", PY)):
        copies[name] = tmp_path / name
        shutil.copytree(home, copies[name])
        outputs = [
            _run(cmd, copies[name], tmp_path, "delete", "eval", eval_id),
            _run(cmd, copies[name], tmp_path, "delete", "eval", "latest"),
            _run(cmd, copies[name], tmp_path, "delete", "eval", "latest"),
        ]
        copies[name + "_out"] = outputs  # type: ignore[assignment]
    assert copies["python_out"] == copies["node_out"]

    def dump(d: Path) -> dict[str, Any]:
        c = sqlite3.connect(d / "promptfoo.db")
        names = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        rows = {n: sorted(map(repr, c.execute(f'SELECT * FROM "{n}"'))) for n in names}
        c.close()
        return rows

    assert dump(copies["python"]) == dump(copies["node"])
