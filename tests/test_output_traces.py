"""Trace export: createOutputData's `traces` (test/util/output.test.ts and jsonExport.test.ts
trace cases), projectTracesForOutput under PROMPTFOO_STRIP_*, and the fetch-failure warning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from promptphoo.models.eval import Eval
from promptphoo.output import write_output
from promptphoo.output.writer import create_output_data, project_traces_for_output
from promptphoo.tracing import store as store_module


class TracesStore:
    def __init__(self, traces: list[dict[str, Any]] | Exception) -> None:
        self.traces = traces
        self.calls: list[str] = []

    def get_traces_by_evaluation_sync(self, evaluation_id, options=None):
        self.calls.append(evaluation_id)
        if isinstance(self.traces, Exception):
            raise self.traces
        return self.traces


@pytest.fixture
def traces(monkeypatch):
    def install(value):
        fake = TracesStore(value)
        monkeypatch.setattr(store_module, "get_trace_store", lambda: fake)
        return fake

    return install


def _written(tmp_path: Path, record: Any) -> tuple[str, dict]:
    path = tmp_path / "output.json"
    write_output(str(path), record, None)
    text = path.read_text("utf-8")
    return text, json.loads(text)


async def test_traces_follow_runtime_options_in_the_output(traces):
    store = traces([{"traceId": "t", "evaluationId": "e", "testCaseId": "c", "spans": []}])
    record = Eval({}, created_at=0, runtime_options={"maxConcurrency": 2})
    data = create_output_data(record, None)
    assert store.calls == [record.id]
    assert data["traces"] == [{"traceId": "t", "evaluationId": "e", "testCaseId": "c",
                               "spans": []}]  # fmt: skip
    keys = list(data)
    assert keys.index("traces") == keys.index("runtimeOptions") + 1


async def test_no_traces_key_without_traces(traces):
    traces([])
    assert "traces" not in create_output_data(Eval({}, created_at=0), None)


async def test_fetch_failure_omits_traces_with_a_warning(traces, caplog):
    traces(RuntimeError("db locked"))
    record = Eval({}, created_at=0)
    assert "traces" not in create_output_data(record, None)
    assert (
        f"Failed to fetch traces for output {record.id}; traces omitted from export: db locked"
        in caplog.text
    )


async def test_strip_test_vars_removes_trace_vars(traces, tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_STRIP_TEST_VARS", "true")
    traces([{"traceId": "trace-strip-vars", "evaluationId": "e", "testCaseId": "c",
             "metadata": {"testIdx": 0, "promptIdx": 0, "source": "trace export",
                          "vars": {"customerEmail": "trace-var-secret@example.com"}},
             "spans": []}])  # fmt: skip
    text, parsed = _written(tmp_path, Eval({}, created_at=0))
    assert parsed["traces"][0]["metadata"] == {"testIdx": 0, "promptIdx": 0,
                                               "source": "trace export"}  # fmt: skip
    assert "trace-var-secret@example.com" not in text


async def test_strip_test_vars_drops_metadata_left_empty(traces, tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_STRIP_TEST_VARS", "true")
    traces(
        [
            {
                "traceId": "t",
                "evaluationId": "e",
                "testCaseId": "c",
                "metadata": {"vars": {"customerEmail": "only-var@example.com"}},
                "spans": [],
            }
        ]
    )
    text, parsed = _written(tmp_path, Eval({}, created_at=0))
    assert "metadata" not in parsed["traces"][0]
    assert "only-var@example.com" not in text


async def test_strip_metadata_drops_trace_metadata(traces, tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_STRIP_METADATA", "true")
    traces([{"traceId": "t", "evaluationId": "e", "testCaseId": "c",
             "metadata": {"note": "trace-metadata-secret",
                          "vars": {"customerEmail": "trace-var-secret@example.com"}},
             "spans": []}])  # fmt: skip
    text, parsed = _written(tmp_path, Eval({}, created_at=0))
    assert "metadata" not in parsed["traces"][0]
    assert "trace-metadata-secret" not in text and "trace-var-secret" not in text


async def test_strip_prompt_and_response_bodies_from_span_attributes(traces, tmp_path,
                                                                     monkeypatch):  # fmt: skip
    monkeypatch.setenv("PROMPTFOO_STRIP_PROMPT_TEXT", "true")
    monkeypatch.setenv("PROMPTFOO_STRIP_RESPONSE_OUTPUT", "true")
    traces([{"traceId": "t", "evaluationId": "e", "testCaseId": "c", "spans": [
        {"spanId": "s", "name": "provider", "startTime": 1,
         "attributes": {"promptfoo.request.body": "trace-prompt-secret",
                        "promptfoo.response.body": "trace-response-secret",
                        "operation": "provider-call"}}]}])  # fmt: skip
    text, parsed = _written(tmp_path, Eval({}, created_at=0))
    assert parsed["traces"][0]["spans"][0]["attributes"] == {"operation": "provider-call"}
    assert "trace-prompt-secret" not in text and "trace-response-secret" not in text


def _flags(**on: bool) -> dict[str, bool]:
    names = ("shouldStripMetadata", "shouldStripPromptText", "shouldStripResponseOutput",
             "shouldStripTestVars")  # fmt: skip
    return {name: on.get(name, False) for name in names}


def test_projection_is_identity_without_strip_flags():
    traces = [{"traceId": "t", "metadata": {"vars": {}}, "spans": []}]
    assert project_traces_for_output(traces, _flags()) is traces


def test_projection_drops_attributes_left_empty_and_keeps_other_spans():
    traces = [{"traceId": "t", "spans": [
        {"spanId": "a", "attributes": {"promptfoo.request.body": "x"}},
        {"spanId": "b"},
        {"spanId": "c", "attributes": {}},
    ]}]  # fmt: skip
    [projected] = project_traces_for_output(traces, _flags(shouldStripPromptText=True))
    # `{}` is truthy in JS, so it is projected too, and dropped for being empty.
    assert projected["spans"] == [{"spanId": "a"}, {"spanId": "b"}, {"spanId": "c"}]
    assert traces[0]["spans"][0]["attributes"] == {"promptfoo.request.body": "x"}


async def test_traces_survive_a_python_export_import_round_trip(tmp_path, monkeypatch):
    """`import` writes the exported traces through the trace store, and `export` reads them
    back unchanged (the D43 gap: a traced eval no longer loses `traces`)."""
    from click.testing import CliRunner

    from promptphoo import db
    from promptphoo.cli.main import build_cli

    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.delenv("IS_TESTING", raising=False)
    exported_traces = [{
        "traceId": "a" * 32, "evaluationId": "eval-src", "testCaseId": "0-0",
        "metadata": {"testIdx": 0, "promptIdx": 0},
        "spans": [{"spanId": "b" * 16, "name": "root", "startTime": 1, "endTime": 2,
                   "attributes": {"k": "v"}, "statusCode": 1}],
    }]  # fmt: skip
    export = {
        "evalId": "eval-src",
        "results": {"version": 3, "timestamp": "2025-01-01T00:00:00.000Z", "results": [],
                    "prompts": [], "stats": {"successes": 0, "failures": 0, "errors": 0,
                                             "tokenUsage": {}}},
        "config": {"description": "traced"},
        "shareableUrl": None,
        "metadata": {"promptfooVersion": "0.125.1"},
        "traces": exported_traces,
    }  # fmt: skip
    src = tmp_path / "in.json"
    src.write_text(json.dumps(export))
    out = tmp_path / "out.json"
    try:
        db.close_db()
        runner = CliRunner()
        for args in (["import", str(src)], ["export", "eval", "latest", "-o", str(out)]):
            result = runner.invoke(build_cli(args), args, catch_exceptions=False)
            assert result.exit_code == 0, result.output
    finally:
        db.close_db()
    reexported = json.loads(out.read_text())
    [trace] = reexported["traces"]
    assert trace == {**exported_traces[0], "evaluationId": reexported["evalId"]}
