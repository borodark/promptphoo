"""The comparison pass on resume, retry-errors, persistence failures and cancellation. Ports the
resume / retry cases of test/evaluator/selectBestRuntime.test.ts and the comparison cases of
test/evaluator/{execution,inMemoryStore}.test.ts (see D29, D36)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from promptphoo import cli_state, db, paths
from promptphoo.evaluator.run import ResultFailureReason, evaluate
from promptphoo.models.eval import Eval
from promptphoo.providers.base import ApiProvider
from promptphoo.scheduler.cancellation import AbortController


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("IS_TESTING", raising=False)
    paths.set_config_directory_path(None)
    db.close_db()
    db.run_db_migrations()
    yield tmp_path
    db.close_db()
    cli_state.resume = False
    cli_state.retry_mode = False
    cli_state.retry_error_result_ids = None


class FakeProvider(ApiProvider):
    def __init__(self, provider_id: str, fn: Any) -> None:
        super().__init__({"id": provider_id})
        self.fn = fn
        self.calls: list[str] = []

    async def call_api(self, prompt, context=None, options=None):
        self.calls.append(prompt)
        result = self.fn(prompt, context)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, BaseException):
            raise result
        return result


def run(coro):
    return asyncio.run(coro)


def _pick(needle: str, failures: list[BaseException] | None = None):
    failures = list(failures or [])

    def grade(prompt, _ctx):
        if failures:
            return failures.pop(0)
        return {"output": str(json.loads(prompt).index(needle))}

    return grade


def _suite(grader: FakeProvider, *extra_asserts: dict[str, Any]) -> tuple[FakeProvider, dict]:
    target = FakeProvider("select-best-target", lambda prompt, _ctx: {"output": prompt})
    suite = {
        "providers": [target],
        "prompts": [{"raw": "first", "label": "first"}, {"raw": "second", "label": "second"}],
        "tests": [
            {
                "options": {"rubricPrompt": "{{ outputs | dump }}"},
                "assert": [
                    *extra_asserts,
                    {"type": "select-best", "value": "choose the best", "provider": grader},
                ],
            }
        ],
    }
    return target, suite


def _rows(record: Eval) -> list[Any]:
    return sorted(run(record.read_results_by_test_idx(0)), key=lambda r: r.prompt_idx)


def _stats(record: Eval) -> tuple[int, int, int]:
    stats = record.get_stats()
    return stats["successes"], stats["failures"], stats["errors"]


def test_resume_skips_completed_comparisons(config_dir):
    grader = FakeProvider("grader", _pick("second"))
    target, suite = _suite(grader)
    record = Eval.create({}, suite["prompts"])
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    assert len(grader.calls) == 1
    cli_state.resume = True
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    assert len(target.calls) == 2
    assert len(grader.calls) == 1
    assert [r.success for r in _rows(record)] == [False, True]
    assert _stats(record) == (1, 1, 0)


def test_comparison_error_survives_max_score_and_recovers_on_resume(config_dir):
    grader = FakeProvider("grader", _pick("first", [RuntimeError("grader unavailable")]))
    target, suite = _suite(grader, {"type": "contains", "value": "first"})
    suite["tests"][0]["assert"].append({"type": "max-score"})
    record = Eval.create({}, suite["prompts"])
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    for row in _rows(record):
        assert row.success is False
        assert row.failure_reason == ResultFailureReason.ERROR
        assert "Check the grader configuration and credentials" in row.error
    assert _stats(record) == (0, 0, 2)

    cli_state.resume = True
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    assert _stats(record) == (1, 1, 0)
    rows = _rows(record)
    winner = next(r for r in rows if r.success)
    assert winner.error is None
    assert all("comparisonError" not in ((r.metadata or {}).get("__promptfoo") or {})
               for r in rows)  # fmt: skip
    assert len(target.calls) == 2
    assert len(grader.calls) == 2


def test_retry_compares_only_replacement_rows(config_dir):
    grader = FakeProvider("grader", _pick("second", [RuntimeError("temporary grader failure")]))
    target, suite = _suite(grader)
    record = Eval.create({}, suite["prompts"])
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    previous = [r.id for r in _rows(record)]
    assert all(r.failure_reason == ResultFailureReason.ERROR for r in _rows(record))
    cli_state.resume = True
    cli_state.retry_mode = True
    cli_state.retry_error_result_ids = previous
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    assert len(target.calls) == 4
    assert len(grader.calls) == 2
    assert len(json.loads(grader.calls[1])) == 2
    rows = _rows(record)
    replacements = [r for r in rows if r.id not in previous]
    assert len(replacements) == 2
    assert [r.success for r in replacements] == [False, True]
    assert all(r.failure_reason == ResultFailureReason.ERROR for r in rows if r.id in previous)


def test_rows_that_failed_to_persist_join_the_comparison(config_dir, monkeypatch):
    grader = FakeProvider("grader", _pick("first"))
    _target, suite = _suite(grader)
    record = Eval.create({}, suite["prompts"])
    original = record.append_result

    async def flaky(row):
        if row["promptIdx"] == 0:
            raise RuntimeError("disk full")
        await original(row)

    monkeypatch.setattr(record, "append_result", flaky)
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    assert record.result_persistence_failed
    # The grader saw both outputs; the failed row was graded in memory, not saved.
    assert json.loads(grader.calls[0]) == ["first", "second"]
    saved = _rows(record)
    assert [r.prompt_idx for r in saved] == [1]
    assert saved[0].success is False
    failed = run(record.read_failed_results_by_test_idx(0))
    assert failed[0].success is True
    assert failed[0].grading_result["componentResults"][-1]["assertion"]["type"] == "select-best"


def test_cancelled_grading_marks_rows_aborted_then_resume_regrades(config_dir):
    controller = AbortController()

    async def blocking(prompt, ctx):
        controller.abort()
        await asyncio.sleep(10)
        return {"output": "0"}

    grader = FakeProvider("grader", blocking)
    target, suite = _suite(grader, {"type": "contains", "value": "s"})
    record = Eval.create({}, suite["prompts"])
    returned = run(
        evaluate(suite, record, {"maxConcurrency": 1, "cache": False,
                                 "abortSignal": controller.signal})
    )  # fmt: skip
    assert returned is record
    rows = _rows(record)
    assert [r.error for r in rows] == ["Aborted: This operation was aborted"] * 2
    assert all(r.failure_reason == ResultFailureReason.ERROR for r in rows)
    states = [r.metadata["__promptfoo"]["comparisonError"] for r in rows]
    assert states == [
        {"success": True, "score": 1, "failureReason": 0},
        {"success": True, "score": 1, "failureReason": 0},
    ]
    assert _stats(record) == (0, 0, 2)

    grader.fn = _pick("second")
    cli_state.resume = True
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    rows = _rows(record)
    assert [r.success for r in rows] == [False, True]
    assert rows[1].error is None
    assert _stats(record) == (1, 1, 0)
    assert len(target.calls) == 2


def test_abort_during_steps_keeps_completed_rows(config_dir):
    controller = AbortController()

    def target_fn(prompt, ctx):
        if ctx["vars"]["n"] == 2 and not controller.signal.aborted:
            controller.abort()

            async def hang():
                await asyncio.sleep(10)
                return {"output": "late"}

            return hang()
        return {"output": f"out {ctx['vars']['n']}"}

    target = FakeProvider("target", target_fn)
    suite = {
        "providers": [target],
        "prompts": [{"raw": "p {{n}}", "label": "p"}],
        "tests": [{"vars": {"n": n}} for n in range(4)],
    }
    record = Eval.create({}, suite["prompts"])
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False,
                                 "abortSignal": controller.signal}))  # fmt: skip
    saved = sorted(r.test_idx for r in record.fetch_results_batched().__next__())
    assert saved == [0, 1]
    assert record.prompts[0]["metrics"]["testPassCount"] == 2

    cli_state.resume = True
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    assert sorted(r.test_idx for r in run(record.read_results_by_test_idx(2))) == [2]
    assert record.prompts[0]["metrics"]["testPassCount"] == 4
