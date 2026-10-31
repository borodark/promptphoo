"""End-to-end: `promptfoo eval` with tracing enabled. A Python provider writes spans straight
into the trace store for its traceparent's trace id, and the trace and trajectory assertions
grade them through the assertion context's `trace`."""

from __future__ import annotations

import json

from click.testing import CliRunner

from promptphoo import db
from promptphoo.cli.main import build_cli

PROVIDER = """\
import time

from promptphoo.tracing.store import get_trace_store


async def call_api(prompt, options, context):
    traceparent = context["traceparent"]
    _, trace_id, parent_span_id, _ = traceparent.split("-")
    start = int(time.time() * 1000)
    spans = [
        {"spanId": "a1" * 8, "parentSpanId": parent_span_id, "name": "agent.run",
         "startTime": start, "endTime": start + 120,
         "attributes": {"gen_ai.operation.name": "invoke_agent"}},
        {"spanId": "b2" * 8, "parentSpanId": "a1" * 8, "name": "llm.chat",
         "startTime": start + 5, "endTime": start + 45,
         "attributes": {"gen_ai.request.model": "fake-model"}},
        {"spanId": "c3" * 8, "parentSpanId": "a1" * 8, "name": "tool.search",
         "startTime": start + 50, "endTime": start + 80,
         "attributes": {"tool.name": "search", "tool.arguments": '{"q": "paris"}'}},
        {"spanId": "d4" * 8, "parentSpanId": "a1" * 8, "name": "tool.fetch",
         "startTime": start + 85, "endTime": start + 110, "statusCode": 2,
         "statusMessage": "fetch failed: timeout",
         "attributes": {"tool.name": "fetch"}},
    ]
    result = await get_trace_store().add_spans(trace_id, spans)
    return {
        "output": "Paris" if result["stored"] else "not stored",
        "metadata": {"seenTestCaseId": context.get("testCaseId"),
                     "seenEvaluationId": context.get("evaluationId")},
    }
"""

GET_ASSERT = """\
def get_assert(output, context):
    trace = context.get("trace") or {}
    names = sorted(span["name"] for span in trace.get("spans", []))
    return {"pass": len(names) == 4, "score": 1, "reason": ",".join(names)}
"""

CONFIG = """\
description: trace assertions e2e
tracing:
  enabled: true
prompts:
  - "Where is the Eiffel tower?"
providers:
  - id: file://provider.py
    label: traced
tests:
  - assert:
      - type: trace-span-count
        value: {pattern: "tool.*", min: 2, max: 2}
      - type: trace-span-count
        value: {pattern: "llm.*", attributes: {gen_ai.request.model: fake-model}, max: 1}
      - type: trace-span-duration
        value: {pattern: "*", max: 200}
      - type: trace-error-spans
        value: {max_count: 1}
      - type: trajectory:tool-used
        value: [search, fetch]
      - type: trajectory:tool-sequence
        value: {steps: [search, fetch]}
      - type: python
        value: file://check_trace.py
  - assert:
      - type: trace-error-spans
        value: 0
      - type: trace-span-duration
        value: {pattern: "agent.*", max: 50}
"""


def _eval(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.setenv("PROMPTFOO_TRACE_FETCH_RETRY_DELAY_MS", "0")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "provider.py").write_text(PROVIDER)
    (tmp_path / "check_trace.py").write_text(GET_ASSERT)
    (tmp_path / "promptfooconfig.yaml").write_text(CONFIG)
    args = ["eval", "-c", "promptfooconfig.yaml", "-o", "out.json", "--no-table", "--no-cache"]
    try:
        result = CliRunner().invoke(build_cli(args), args, catch_exceptions=False)
        data = json.loads((tmp_path / "out.json").read_text())
        traces = db.get_db().execute("SELECT trace_id, evaluation_id FROM traces").fetchall()
        names = [r[0] for r in db.get_db().execute("SELECT name FROM spans").fetchall()]
    finally:
        db.close_db()
    return result, data, traces, names


def test_trace_assertions_grade_spans_written_by_the_provider(tmp_path, monkeypatch):
    result, data, traces, names = _eval(tmp_path, monkeypatch)
    rows = sorted(data["results"]["results"], key=lambda r: r["testIdx"])
    assert len(rows) == 2
    eval_id = data["evalId"]
    assert {t[1] for t in traces} == {eval_id}
    # The provider's 4 spans per test, plus promptfoo's own: the test case, the target and one
    # grader span per assertion. Those are batch-exported after grading, as under Node, so the
    # assertions below see only the provider's spans.
    provider_spans = ["agent.run", "llm.chat", "tool.fetch", "tool.search"] * 2
    graders = [
        "grader trace-span-count", "grader trace-span-count", "grader trace-span-duration",
        "grader trace-error-spans", "grader trajectory:tool-used",
        "grader trajectory:tool-sequence", "grader python",
        "grader trace-error-spans", "grader trace-span-duration",
    ]  # fmt: skip
    own_spans = ["promptfoo.test_case", "traced"] * 2 + graders
    assert sorted(names) == sorted(provider_spans + own_spans)

    first, second = rows
    for row in rows:
        assert row["response"]["output"] == "Paris"
        assert row["traceId"] in {t[0] for t in traces}
        assert row["evaluationId"] == eval_id
        assert row["testCase"]["metadata"] == {"tracingEnabled": True, "evaluationId": eval_id}
        assert row["metadata"]["seenEvaluationId"] == eval_id
    assert first["metadata"]["seenTestCaseId"] == "0-0"
    assert second["metadata"]["seenTestCaseId"] == "1-0"
    assert first["traceId"] != second["traceId"]

    reasons = [c["reason"] for c in first["gradingResult"]["componentResults"]]
    assert first["success"] is True, reasons
    assert reasons[:4] == [
        'Found 2 spans matching pattern "tool.*" (expected 2-2). Matched spans: tool.search, '
        "tool.fetch",
        'Found 1 spans matching pattern "llm.*" (expected at most 1). Matched spans: llm.chat. '
        'Attribute filters applied to keys: ["gen_ai.request.model"].',
        'All 4 spans matching pattern "*" completed within 200ms (max: 120ms)',
        "Found 1 error(s) in 4 spans (25.0%), within threshold of 1",
    ]
    assert reasons[6] == "agent.run,llm.chat,tool.fetch,tool.search"

    assert second["success"] is False
    assert [c["reason"] for c in second["gradingResult"]["componentResults"]] == [
        "Found 1 error spans, expected at most 0. Errors: tool.fetch (fetch failed: timeout)",
        "1 span(s) exceed duration threshold 50ms. Slowest: agent.run (120ms)",
    ]
    assert result.exit_code == 100


def test_untraced_eval_has_no_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "promptfooconfig.yaml").write_text(
        "prompts: [hi]\nproviders: [echo]\ntests:\n"
        "  - assert:\n      - type: trace-span-count\n        value: {pattern: '*', min: 1}\n"
    )
    args = ["eval", "-c", "promptfooconfig.yaml", "-o", "out.json", "--no-table"]
    try:
        CliRunner().invoke(build_cli(args), args, catch_exceptions=False)
        data = json.loads((tmp_path / "out.json").read_text())
    finally:
        db.close_db()
    [row] = data["results"]["results"]
    assert "traceId" not in row
    assert "No trace data available for trace-span-count assertion" in row["error"]
