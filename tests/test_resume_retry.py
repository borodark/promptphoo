"""`eval --resume`, `eval --retry-errors`, `retry <evalId>` and the Ctrl+C pause, end to end with
echo and Python providers on a temp config dir. Ports the resume / retry-errors cases of
test/commands/eval.test.ts, test/node/retry.test.ts, test/node/retry.integration.test.ts,
test/models/evalResult.test.ts and test/smoke/resume.test.ts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from promptphoo import cli_state, db
from promptphoo.cli import eval as eval_mod
from promptphoo.cli.main import build_cli
from promptphoo.cli.retry import (
    delete_error_results,
    get_error_result_ids,
    get_persisted_provider_filter_options,
    recalculate_prompt_metrics,
)
from promptphoo.models.eval import Eval, EvalResult

SRC = str(Path(__file__).resolve().parents[1] / "src")

PROVIDER = """\
import os


def call_api(prompt, options, context):
    name = context["vars"]["name"]
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "fail.flag")) and name in ("b", "d"):
        return {"error": f"boom {name}"}
    return {"output": f"hello {name}", "tokenUsage": {"total": 3, "prompt": 2, "completion": 1}}
"""

CONFIG = """\
description: retry test
prompts:
  - "Say hi to {{name}}"
providers:
  - id: file://provider.py
    label: py
  - echo
tests:
  - vars: {name: a}
    assert: [{type: contains, value: "a"}]
  - vars: {name: b}
    assert: [{type: contains, value: "b"}]
  - vars: {name: c}
    assert: [{type: contains, value: "zzz"}]
  - vars: {name: d}
    assert: [{type: contains, value: "d"}]
"""

# Pauses the run (as Ctrl+C would) while calling the third test, through the CLI's handler.
PAUSING_PROVIDER = """\
import os
import sys
import time


def call_api(prompt, options, context):
    name = context["vars"]["name"]
    here = os.path.dirname(os.path.abspath(__file__))
    if name == "c" and os.path.exists(os.path.join(here, "pause.flag")):
        handler = sys.modules["promptphoo.cli.eval"]._TEST_HANDLERS[-1]
        handler.loop.call_soon_threadsafe(handler._on_sigint)
        time.sleep(0.5)
    return {"output": f"hello {name}"}
"""

PAUSE_CONFIG = """\
description: pause test
prompts:
  - "Hello {{name}}"
providers:
  - id: file://provider.py
tests:
  - vars: {name: a}
  - vars: {name: b}
  - vars: {name: c}
  - vars: {name: d}
  - vars: {name: e}
defaultTest:
  assert:
    - type: contains
      value: "{{name}}"
"""


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "provider.py").write_text(PROVIDER)
    (tmp_path / "promptfooconfig.yaml").write_text(CONFIG)
    db.close_db()
    yield tmp_path
    db.close_db()
    cli_state.resume = False
    cli_state.retry_mode = False
    cli_state.retry_error_result_ids = None


_CAPLOG: list = []


@pytest.fixture(autouse=True)
def _capture(caplog):
    _CAPLOG[:] = [caplog]
    caplog.set_level(logging.INFO)
    yield
    _CAPLOG.clear()


class _Run:
    def __init__(self, result, output: str) -> None:
        self.exit_code = result.exit_code
        self.output = output


def _cli(*args: str, catch: bool = False) -> _Run:
    caplog = _CAPLOG[0]
    caplog.clear()
    try:
        result = CliRunner().invoke(build_cli(list(args)), list(args), catch_exceptions=catch)
    finally:
        db.close_db()
    return _Run(result, caplog.text)


def _rows(eval_id: str) -> list[tuple]:
    conn = db.get_db()
    conn.row_factory = None
    try:
        return sorted(
            conn.execute(
                "SELECT test_idx, prompt_idx, success, failure_reason FROM eval_results "
                "WHERE eval_id = ?",
                (eval_id,),
            ).fetchall()
        )
    finally:
        db.close_db()


def _first_run(workdir: Path) -> Eval:
    (workdir / "fail.flag").write_text("x")
    result = _cli("eval", "--no-table", "--no-cache")
    assert result.exit_code == 100, result.output
    (workdir / "fail.flag").unlink()
    record = Eval.latest()
    assert record is not None
    db.close_db()
    return record


def _metrics(eval_id: str) -> list[dict]:
    record = Eval.find_by_id(eval_id)
    assert record is not None
    db.close_db()
    return [p["metrics"] for p in record.prompts]


# --- EvalResult / Eval store helpers ------------------------------------------------------------


def test_completed_index_pairs_and_error_ids(workdir):
    record = _first_run(workdir)
    pairs = EvalResult.get_completed_index_pairs(record.id)
    assert pairs == {f"{t}:{p}" for t in range(4) for p in range(2)}
    without_errors = EvalResult.get_completed_index_pairs(record.id, exclude_errors=True)
    assert pairs - without_errors == {"1:0", "3:0"}
    assert len(get_error_result_ids(record.id)) == 2
    db.close_db()


def test_delete_error_results_and_recalculate(workdir):
    record = _first_run(workdir)
    before = _metrics(record.id)
    assert [m["testErrorCount"] for m in before] == [2, 0]
    delete_error_results(get_error_result_ids(record.id))
    record = Eval.find_by_id(record.id)
    assert record is not None
    asyncio.run(recalculate_prompt_metrics(record))
    db.close_db()
    after = _metrics(record.id)
    assert [(m["testPassCount"], m["testFailCount"], m["testErrorCount"]) for m in after] == [
        (1, 1, 0),
        (3, 1, 0),
    ]
    assert after[0]["tokenUsage"]["total"] == before[0]["tokenUsage"]["total"]
    assert after[1]["score"] == before[1]["score"]
    delete_error_results([])  # no-op


def test_persisted_provider_filter_options():
    assert get_persisted_provider_filter_options(None) == {}
    assert get_persisted_provider_filter_options("echo") == {"filterProviders": "echo"}
    with pytest.raises(ValueError, match="Stored provider filter is invalid"):
        get_persisted_provider_filter_options("")


def test_failed_results_are_reconstructed_once(workdir):
    db.run_db_migrations()
    record = Eval.create({}, [])
    row = {
        "provider": {"id": "echo"},
        "prompt": {"raw": "p", "label": "p"},
        "vars": {},
        "promptIdx": 1,
        "testIdx": 3,
        "testCase": {"vars": {}},
        "success": True,
        "score": 1,
        "namedScores": {},
        "failureReason": 0,
    }
    record.record_result_persistence_failure(row)
    assert record.result_persistence_failed
    assert record.has_result_persistence_failure(row)
    first = asyncio.run(record.read_failed_results_by_test_idx(3))
    again = asyncio.run(record.read_failed_results_by_test_idx(3))
    assert len(first) == 1 and first[0] is again[0]
    assert asyncio.run(record.read_failed_results_by_test_idx(0)) == []
    db.close_db()


# --- eval --retry-errors ----------------------------------------------------------------------


def test_retry_errors_reruns_only_errors(workdir):
    record = _first_run(workdir)
    result = _cli("eval", "--retry-errors", "--no-table", "-o", "out.json")
    assert "🔄 Retrying ERROR results from latest evaluation..." in result.output
    assert "Found 2 ERROR results to retry" in result.output
    assert "Resuming: skipping 6 previously completed cases" in result.output
    assert result.exit_code == 100  # test c still fails its assertion
    rows = _rows(record.id)
    assert len(rows) == 8
    assert [r for r in rows if r[3] == 2] == []
    assert [(m["testPassCount"], m["testFailCount"], m["testErrorCount"])
            for m in _metrics(record.id)] == [(3, 1, 0), (3, 1, 0)]  # fmt: skip
    # -o is ignored on replay; the saved config has no outputPath.
    assert not (workdir / "out.json").exists()


def test_retry_errors_without_errors(workdir):
    (workdir / "fail.flag").unlink(missing_ok=True)
    assert _cli("eval", "--no-table").exit_code == 100
    result = _cli("eval", "--retry-errors", "--no-table")
    assert "✅ No ERROR results found in the latest evaluation" in result.output
    assert result.exit_code == 0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--resume", "--retry-errors"],
         "Cannot use --resume and --retry-errors together. Please use one or the other."),
        (["--resume", "--no-write"],
         "Cannot use --resume with --no-write. Resume functionality requires database "
         "persistence."),
        (["--retry-errors", "--no-write"],
         "Cannot use --retry-errors with --no-write. Retry functionality requires database "
         "persistence."),
        (["--resume", "--tag", "a=b"],
         "Cannot use --tag with --resume. Resumed evaluations keep their original tags."),
        (["--retry-errors", "--tag", "a=b"],
         "Cannot use --tag with --retry-errors. Retried evaluations keep their original tags."),
        (["--resume", "eval-missing"], "Could not find evaluation to resume: eval-missing"),
        (["--retry-errors"], "No previous evaluation found to retry errors from"),
    ],
)  # fmt: skip
def test_conflicting_options(workdir, args, message):
    result = _cli("eval", "--no-table", *args)
    assert result.exit_code == 1
    assert message in result.output


# --- retry <evalId> -----------------------------------------------------------------------------


def test_retry_command(workdir):
    record = _first_run(workdir)
    result = _cli("retry", record.id, "--max-concurrency", "2")
    assert result.exit_code == 0, result.output
    assert f"🔄 Retrying failed tests for evaluation: {record.id}" in result.output
    assert "Found 2 ERROR results to retry" in result.output
    assert f"✅ Retry completed for evaluation: {record.id}" in result.output
    assert [r for r in _rows(record.id) if r[3] == 2] == []
    assert len(_rows(record.id)) == 8
    assert cli_state.resume is False and cli_state.retry_mode is False
    assert cli_state.max_concurrency() is None


def test_retry_command_with_config(workdir):
    record = _first_run(workdir)
    result = _cli("retry", record.id, "-c", "promptfooconfig.yaml")
    assert result.exit_code == 0, result.output
    assert [m["testErrorCount"] for m in _metrics(record.id)] == [0, 0]


def test_retry_command_without_errors(workdir):
    assert _cli("eval", "--no-table").exit_code == 100
    record = Eval.latest()
    db.close_db()
    assert record is not None
    result = _cli("retry", record.id)
    assert result.exit_code == 0
    assert "✅ No ERROR results found in this evaluation" in result.output


def test_retry_command_unknown_eval(workdir):
    db.run_db_migrations()
    db.close_db()
    result = _cli("retry", "eval-nope")
    assert result.exit_code == 1
    assert "Failed to retry evaluation: Evaluation with ID eval-nope not found" in result.output
    assert "Recovery options:" in result.output


def test_retry_keeps_errors_when_retry_fails(workdir):
    record = _first_run(workdir)
    (workdir / "fail.flag").write_text("x")
    result = _cli("retry", record.id)
    assert result.exit_code == 0
    # The replacement rows are errors again; the old ones were deleted after the retry.
    assert len([r for r in _rows(record.id) if r[3] == 2]) == 2
    assert len(_rows(record.id)) == 8


def test_retry_rewrites_jsonl_output(workdir):
    (workdir / "promptfooconfig.yaml").write_text(CONFIG + "outputPath: out.jsonl\n")
    record = _first_run(workdir)
    lines = (workdir / "out.jsonl").read_text().splitlines()
    assert sum(json.loads(line)["failureReason"] == 2 for line in lines) == 2
    assert _cli("retry", record.id).exit_code == 0
    lines = (workdir / "out.jsonl").read_text().splitlines()
    assert len(lines) == 8
    assert sum(json.loads(line)["failureReason"] == 2 for line in lines) == 0


# --- eval --resume ------------------------------------------------------------------------------


def _drop_rows(eval_id: str, pairs: list[tuple[int, int]]) -> None:
    conn = db.get_db()
    with db.transaction(conn):
        for test_idx, prompt_idx in pairs:
            conn.execute(
                "DELETE FROM eval_results WHERE eval_id = ? AND test_idx = ? AND prompt_idx = ?",
                (eval_id, test_idx, prompt_idx),
            )
    db.close_db()


def test_resume_runs_missing_pairs_only(workdir):
    assert _cli("eval", "--no-table", "--repeat", "1", "-j", "3").exit_code == 100
    record = Eval.latest()
    assert record is not None
    db.close_db()
    _drop_rows(record.id, [(1, 0), (2, 1), (3, 1)])
    result = _cli("eval", "--resume", "--no-table", "--filter-first-n", "1")
    assert f"Resuming evaluation {record.id}..." in result.output
    assert "Resuming: skipping 5 previously completed cases" in result.output
    assert "Running 3 test cases (up to 3 at a time)..." in result.output
    assert len(_rows(record.id)) == 8


def test_resume_by_id_restores_saved_columns(workdir):
    assert _cli("eval", "--no-table").exit_code == 100
    record = Eval.latest()
    assert record is not None
    db.close_db()
    _drop_rows(record.id, [(3, 0)])
    # Changing the config after the run does not change what a resume runs.
    (workdir / "promptfooconfig.yaml").write_text(CONFIG.replace("  - echo\n", ""))
    result = _cli("eval", "--resume", record.id, "--no-table", "--filter-providers", "echo")
    assert (
        f'Ignoring --filter-providers/--filter-targets "echo": resuming evaluation {record.id} '
        "with stored provider filter (none) to preserve test indices."
    ) in result.output
    assert len(_rows(record.id)) == 8
    record = Eval.find_by_id(record.id)
    assert record is not None
    db.close_db()
    assert [p["provider"] for p in record.prompts] == ["py", "echo"]


def test_resume_rejects_changed_columns(workdir):
    assert _cli("eval", "--no-table").exit_code == 100
    record = Eval.latest()
    assert record is not None
    conn = db.get_db()
    prompts = record.prompts
    prompts[1]["provider"] = "renamed"
    with db.transaction(conn):
        conn.execute("UPDATE evals SET prompts = ? WHERE id = ?", (json.dumps(prompts), record.id))
    db.close_db()
    with pytest.raises(RuntimeError, match="saved providers are missing or reordered"):
        _cli("eval", "--resume", "--no-table")


# --- Ctrl+C pause, then resume -------------------------------------------------------------------


def test_pause_then_resume(workdir, monkeypatch):
    (workdir / "provider.py").write_text(PAUSING_PROVIDER)
    (workdir / "promptfooconfig.yaml").write_text(PAUSE_CONFIG)
    (workdir / "pause.flag").write_text("x")
    handlers: list = []

    def install(self):
        self.loop = asyncio.get_running_loop()
        handlers.append(self)

    monkeypatch.setattr(eval_mod._PauseHandler, "install", install)
    monkeypatch.setattr(eval_mod, "_TEST_HANDLERS", handlers, raising=False)
    result = _cli("eval", "--no-table", "--no-cache", "-j", "1")
    assert result.exit_code == 0, result.output
    assert "Pausing evaluation... Press Ctrl+C again to force exit." in result.output
    assert "Evaluation interrupted, saving progress..." in result.output
    record = Eval.latest()
    assert record is not None
    db.close_db()
    assert f"⏸ Evaluation paused. ID: {record.id}" in result.output
    assert f"» Resume with: promptfoo eval --resume {record.id}" in result.output
    assert [r[0] for r in _rows(record.id)] == [0, 1]
    assert _metrics(record.id)[0]["testPassCount"] == 2

    (workdir / "pause.flag").unlink()
    result = _cli("eval", "--resume", record.id, "--no-table")
    assert result.exit_code == 0, result.output
    assert "Resuming: skipping 2 previously completed cases" in result.output
    assert "Running at concurrency" not in result.output
    assert [r[0] for r in _rows(record.id)] == [0, 1, 2, 3, 4]
    metrics = _metrics(record.id)[0]
    assert (metrics["testPassCount"], metrics["testFailCount"]) == (5, 0)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigint_pauses_cli_process(tmp_path):
    (tmp_path / "promptfooconfig.yaml").write_text(
        PAUSE_CONFIG.replace("file://provider.py", "echo")
    )
    env = {**os.environ, "PROMPTFOO_CONFIG_DIR": str(tmp_path / "home"), "PYTHONPATH": SRC,
           "NO_COLOR": "1"}  # fmt: skip
    proc = subprocess.Popen(
        [sys.executable, "-m", "promptphoo", "eval", "--no-cache", "--no-table",
         "--delay", "400"],
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    time.sleep(2.5)
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0, out
    assert "Evaluation paused. ID:" in out
    assert "--resume" in out
    resumed = subprocess.run(
        [sys.executable, "-m", "promptphoo", "eval", "--resume", "--no-table"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "Resuming: skipping" in resumed.stdout
    assert "✓ 5 passed (100%)" in resumed.stdout
