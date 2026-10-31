"""Trace assertions. Ports test/assertions/{traceSpanCount,traceSpanDuration,traceErrorSpans,
traceUtils,trace}.test.ts.

JavaScript-assertion cases in trace.test.ts are ported as inline or file:// Python assertions
with the same intent (promptphoo runs no JavaScript).
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any

import pytest

from promptphoo import cli_state
from promptphoo.assertions.base import AssertionParams, AssertionValueError
from promptphoo.assertions.index import (
    assertion_uses_trace,
    has_trace_aware_assertions,
    run_assertion,
    run_assertions,
)
from promptphoo.assertions.trace import (
    filter_trace_spans,
    handle_trace_error_spans,
    handle_trace_span_count,
    handle_trace_span_duration,
    matches_pattern,
)
from promptphoo.jsvalues import UNDEFINED
from promptphoo.scheduler.provider_call_execution_context import (
    with_provider_call_tracing_context,
)
from promptphoo.tracing import store as trace_store_module


def _params(base_type: str, assertion: dict[str, Any], trace: Any = None) -> AssertionParams:
    context: dict[str, Any] = {
        "vars": {},
        "test": {},
        "prompt": "test prompt",
        "logProbs": None,
        "provider": None,
        "providerResponse": {"output": "test output"},
    }
    if trace is not None:
        context["trace"] = trace
    return AssertionParams(
        assertion=assertion,
        base_type=base_type,
        inverse=False,
        output="test output",
        output_string="test output",
        rendered_value=assertion.get("value"),
        provider_response={"output": "test output"},
        test={},
        assertion_value_context=context,
    )


def _result(passed: bool, reason: str, assertion: dict[str, Any]) -> dict[str, Any]:
    return {"pass": passed, "score": 1 if passed else 0, "reason": reason, "assertion": assertion}


# --- traceSpanCount.test.ts -------------------------------------------------------------------

COUNT_TRACE = {
    "traceId": "test-trace-id",
    "evaluationId": "test-evaluation-id",
    "testCaseId": "test-test-case-id",
    "metadata": {"test": "value"},
    "spans": [
        {"spanId": "span-1", "name": "llm.completion", "startTime": 1000, "endTime": 1500},
        {"spanId": "span-2", "name": "llm.chat", "startTime": 1600, "endTime": 2000},
        {"spanId": "span-3", "name": "database.query", "startTime": 2100, "endTime": 2200},
        {"spanId": "span-4", "name": "retrieval.search", "startTime": 2300, "endTime": 2400},
        {"spanId": "span-5", "name": "api.external_call", "startTime": 2500, "endTime": 2600},
    ],
}


@pytest.mark.parametrize(
    ("value", "passed", "reason"),
    [
        ({"pattern": "*llm*", "max": 3}, True,
         'Found 2 spans matching pattern "*llm*" (expected at most 3). '
         "Matched spans: llm.completion, llm.chat"),
        ({"pattern": "*llm*", "max": 1}, False,
         'Found 2 spans matching pattern "*llm*", expected at most 1. '
         "Matched spans: llm.completion, llm.chat"),
        ({"pattern": "*retrieval*", "min": 1}, True,
         'Found 1 spans matching pattern "*retrieval*" (expected at least 1). '
         "Matched spans: retrieval.search"),
        ({"pattern": "*retrieval*", "min": 2}, False,
         'Found 1 spans matching pattern "*retrieval*", expected at least 2. '
         "Matched spans: retrieval.search"),
        ({"pattern": "*", "min": 3, "max": 10}, True,
         'Found 5 spans matching pattern "*" (expected 3-10). Matched spans: llm.completion, '
         "llm.chat, database.query, retrieval.search, api.external_call"),
        ({"pattern": "llm.c?at", "max": 1}, True,
         'Found 1 spans matching pattern "llm.c?at" (expected at most 1). '
         "Matched spans: llm.chat"),
        ({"pattern": "*LLM*", "max": 5}, True,
         'Found 2 spans matching pattern "*LLM*" (expected at most 5). '
         "Matched spans: llm.completion, llm.chat"),
    ],
)  # fmt: skip
def test_span_count(value: dict[str, Any], passed: bool, reason: str) -> None:
    assertion = {"type": "trace-span-count", "value": value}
    result = handle_trace_span_count(_params("trace-span-count", assertion, COUNT_TRACE))
    assert result == _result(passed, reason, assertion)


def test_span_count_without_trace() -> None:
    assertion = {"type": "trace-span-count", "value": {"pattern": "*", "min": 1}}
    with pytest.raises(
        AssertionValueError, match="No trace data available for trace-span-count assertion"
    ):
        handle_trace_span_count(_params("trace-span-count", assertion))


def test_span_count_empty_spans() -> None:
    assertion = {"type": "trace-span-count", "value": {"pattern": "*", "max": 0}}
    trace = {"traceId": "empty-trace", "spans": []}
    result = handle_trace_span_count(_params("trace-span-count", assertion, trace))
    assert result == _result(
        True, 'Found 0 spans matching pattern "*" (expected at most 0)', assertion
    )


@pytest.mark.parametrize("value", ["invalid", {"max": 5}])
def test_span_count_invalid_value(value: Any) -> None:
    assertion = {"type": "trace-span-count", "value": value}
    with pytest.raises(
        AssertionValueError,
        match="trace-span-count assertion must have a value object with pattern property",
    ):
        handle_trace_span_count(_params("trace-span-count", assertion, COUNT_TRACE))


# --- traceSpanDuration.test.ts ----------------------------------------------------------------

DURATION_TRACE = {
    "traceId": "test-trace-id",
    "spans": [
        {"spanId": "span-1", "name": "llm.completion", "startTime": 1000, "endTime": 1500},
        {"spanId": "span-2", "name": "llm.chat", "startTime": 1600, "endTime": 2800},
        {"spanId": "span-3", "name": "database.query", "startTime": 2900, "endTime": 2950},
        {"spanId": "span-4", "name": "api.external", "startTime": 3000, "endTime": 6000},
        {"spanId": "span-5", "name": "cache.lookup", "startTime": 6100, "endTime": 6105},
    ],
}
SINGLE_SPAN_TRACE = {
    "traceId": "single-span",
    "spans": [{"spanId": "1", "name": "single.op", "startTime": 0, "endTime": 750}],
}
FAST_TRACE = {
    "traceId": "fast-trace",
    "spans": [
        {"spanId": "1", "name": "fast.op1", "startTime": 0, "endTime": 100},
        {"spanId": "2", "name": "fast.op2", "startTime": 100, "endTime": 500},
        {"spanId": "3", "name": "fast.op3", "startTime": 500, "endTime": 1000},
    ],
}
INCOMPLETE_TRACE = {
    "traceId": "incomplete-trace",
    "spans": [
        {"spanId": "1", "name": "complete.op", "startTime": 0, "endTime": 500},
        {"spanId": "2", "name": "incomplete.op", "startTime": 600},
    ],
}


@pytest.mark.parametrize(
    ("value", "trace", "passed", "reason"),
    [
        ({"max": 1500}, FAST_TRACE, True,
         'All 3 spans matching pattern "*" completed within 1500ms (max: 500ms)'),
        ({"max": 1000}, DURATION_TRACE, False,
         "2 span(s) exceed duration threshold 1000ms. Slowest: api.external (3000ms), "
         "llm.chat (1200ms)"),
        ({"pattern": "*llm*", "max": 1000}, DURATION_TRACE, False,
         "1 span(s) exceed duration threshold 1000ms. Slowest: llm.chat (1200ms)"),
        ({"max": 2000, "percentile": 90}, DURATION_TRACE, False,
         "90th percentile duration (3000ms) exceeds threshold 2000ms. "
         "Slowest spans: api.external (3000ms)"),
        ({"max": 1500, "percentile": 50}, DURATION_TRACE, True,
         "50th percentile duration (500ms) is within threshold 1500ms"),
        ({"max": 1000}, INCOMPLETE_TRACE, True,
         'All 1 spans matching pattern "*" completed within 1000ms (max: 500ms)'),
        ({"max": 1000}, {"traceId": "empty-trace", "spans": []}, True,
         'No spans found matching pattern "*" with complete timing data'),
        ({"max": 100}, DURATION_TRACE, False,
         "3 span(s) exceed duration threshold 100ms. Slowest: api.external (3000ms), "
         "llm.chat (1200ms), llm.completion (500ms)"),
        ({"max": 1000, "percentile": 100}, SINGLE_SPAN_TRACE, True,
         "100th percentile duration (750ms) is within threshold 1000ms"),
        ({"max": 1000, "percentile": 0}, DURATION_TRACE, True,
         "0th percentile duration (5ms) is within threshold 1000ms"),
        ({"max": 1000, "percentile": 95}, SINGLE_SPAN_TRACE, True,
         "95th percentile duration (750ms) is within threshold 1000ms"),
    ],
)  # fmt: skip
def test_span_duration(value: dict[str, Any], trace: Any, passed: bool, reason: str) -> None:
    assertion = {"type": "trace-span-duration", "value": value}
    result = handle_trace_span_duration(_params("trace-span-duration", assertion, trace))
    assert result == _result(passed, reason, assertion)


def test_span_duration_without_trace() -> None:
    assertion = {"type": "trace-span-duration", "value": {"max": 1000}}
    with pytest.raises(
        AssertionValueError, match="No trace data available for trace-span-duration assertion"
    ):
        handle_trace_span_duration(_params("trace-span-duration", assertion))


@pytest.mark.parametrize("value", ["invalid", {"pattern": "*"}])
def test_span_duration_invalid_value(value: Any) -> None:
    assertion = {"type": "trace-span-duration", "value": value}
    with pytest.raises(
        AssertionValueError,
        match="trace-span-duration assertion must have a value object with max property",
    ):
        handle_trace_span_duration(_params("trace-span-duration", assertion, DURATION_TRACE))


@pytest.mark.parametrize("percentile", [150, -5, math.nan, "95"])
def test_span_duration_rejects_bad_percentile(percentile: Any) -> None:
    # Regression: an out-of-range percentile used to pass with an "undefinedms" reason.
    assertion = {"type": "trace-span-duration", "value": {"max": 1000, "percentile": percentile}}
    with pytest.raises(
        AssertionValueError,
        match="trace-span-duration assertion percentile must be a number between 0 and 100",
    ):
        handle_trace_span_duration(_params("trace-span-duration", assertion, DURATION_TRACE))


# --- traceErrorSpans.test.ts ------------------------------------------------------------------

ERROR_TRACE = {
    "traceId": "test-trace-id",
    "spans": [
        {"spanId": "span-1", "name": "http.request", "startTime": 1000, "endTime": 1500,
         "statusCode": 200},
        {"spanId": "span-2", "name": "api.call", "startTime": 1600, "endTime": 2000,
         "statusCode": 500, "statusMessage": "Internal Server Error"},
        {"spanId": "span-3", "name": "database.query", "startTime": 2100, "endTime": 2200,
         "attributes": {"error": True, "error.message": "Connection timeout"}},
        {"spanId": "span-4", "name": "cache.get", "startTime": 2300, "endTime": 2400,
         "attributes": {"http.status_code": 404}},
        {"spanId": "span-5", "name": "service.process", "startTime": 2500, "endTime": 2600,
         "attributes": {"otel.status_code": "ERROR"}},
        {"spanId": "span-6", "name": "llm.completion", "startTime": 2700, "endTime": 3000,
         "statusCode": 200},
    ],
}  # fmt: skip
FOUR_ERRORS = "Errors: api.call (Internal Server Error), database.query, cache.get and 1 more"


def _trace(trace_id: str, spans: list[dict[str, Any]]) -> dict[str, Any]:
    return {"traceId": trace_id, "spans": spans}


@pytest.mark.parametrize(
    ("value", "trace", "passed", "reason"),
    [
        ({"max_count": 0}, _trace("no-errors", [
            {"spanId": "1", "name": "op1", "startTime": 0, "endTime": 100, "statusCode": 200},
            {"spanId": "2", "name": "op2", "startTime": 100, "endTime": 200, "statusCode": 201},
        ]), True, 'No errors found in 2 spans matching pattern "*"'),
        ({"max_count": 2}, ERROR_TRACE, False,
         f"Found 4 error spans, expected at most 2. {FOUR_ERRORS}"),
        ({"max_count": 0}, _trace("status-errors", [
            {"spanId": "1", "name": "api.call", "startTime": 0, "endTime": 100, "statusCode": 500},
            {"spanId": "2", "name": "api.call", "startTime": 100, "endTime": 200,
             "statusCode": 403},
        ]), False, "Found 2 error spans, expected at most 0. Errors: api.call (status: 500), "
                   "api.call (status: 403)"),
        ({"max_count": 0}, _trace("otlp-status-errors", [
            {"spanId": "1", "name": "tool.lookup", "startTime": 0, "endTime": 100,
             "statusCode": 2, "statusMessage": "Tool failed"},
        ]), False, "Found 1 error spans, expected at most 0. Errors: tool.lookup (Tool failed)"),
        ({"max_count": 0}, _trace("attr-errors", [
            {"spanId": "1", "name": "op1", "startTime": 0, "endTime": 100,
             "attributes": {"error": True}},
            {"spanId": "2", "name": "op2", "startTime": 100, "endTime": 200,
             "attributes": {"exception": {"type": "RuntimeError", "message": "Failed"}}},
        ]), False, "Found 2 error spans, expected at most 0. Errors: op1, op2"),
        ({"max_percentage": 50}, ERROR_TRACE, False,
         "Error rate 66.7% exceeds threshold 50% (4 errors out of 6 spans)"),
        ({"max_percentage": 70}, ERROR_TRACE, True,
         "Found 4 error(s) in 6 spans (66.7%), within threshold of 70%"),
        ({"pattern": "*api*", "max_count": 0}, ERROR_TRACE, False,
         "Found 1 error spans, expected at most 0. Errors: api.call (Internal Server Error)"),
        (1, ERROR_TRACE, False, f"Found 4 error spans, expected at most 1. {FOUR_ERRORS}"),
        ({"max_count": 0}, _trace("msg-errors", [
            {"spanId": "1", "name": "op1", "startTime": 0, "endTime": 100,
             "statusMessage": "Operation failed due to timeout"},
            {"spanId": "2", "name": "op2", "startTime": 100, "endTime": 200,
             "statusMessage": "Exception: NullPointerException"},
        ]), False, "Found 2 error spans, expected at most 0. Errors: op1 (Operation failed due "
                   "to timeout), op2 (Exception: NullPointerException)"),
        ({"max_count": 0}, _trace("empty-trace", []), True, 'No spans found matching pattern "*"'),
        ({}, ERROR_TRACE, False, f"Found 4 error spans, expected at most 0. {FOUR_ERRORS}"),
        ({"max_count": 5, "max_percentage": 50}, ERROR_TRACE, False,
         "Error rate 66.7% exceeds threshold 50% (4 errors out of 6 spans)"),
    ],
)  # fmt: skip
def test_error_spans(value: Any, trace: Any, passed: bool, reason: str) -> None:
    assertion = {"type": "trace-error-spans", "value": value}
    result = handle_trace_error_spans(_params("trace-error-spans", assertion, trace))
    assert result == _result(passed, reason, assertion)


def test_error_spans_without_trace() -> None:
    assertion = {"type": "trace-error-spans", "value": {"max_count": 0}}
    with pytest.raises(
        AssertionValueError, match="No trace data available for trace-error-spans assertion"
    ):
        handle_trace_error_spans(_params("trace-error-spans", assertion))


# --- traceUtils.test.ts -----------------------------------------------------------------------

UTIL_SPANS: list[dict[str, Any]] = [
    {"spanId": "search", "name": "execute_tool", "startTime": 0,
     "attributes": {"gen_ai.tool.name": "search", "cached": False, "retries": 0}},
    {"spanId": "fetch", "name": "execute_tool", "startTime": 0,
     "attributes": {"gen_ai.tool.name": "fetch", "cached": False, "retries": 0}},
    {"spanId": "other-operation", "name": "invoke_agent", "startTime": 0,
     "attributes": {"gen_ai.tool.name": "search", "cached": False, "retries": 0}},
    {"spanId": "missing", "name": "execute_tool", "startTime": 0},
]  # fmt: skip


def test_filter_combines_pattern_and_attributes() -> None:
    matching = filter_trace_spans(
        UTIL_SPANS, "execute_*", {"gen_ai.tool.name": "search", "cached": False, "retries": 0}
    )
    assert [s["spanId"] for s in matching] == ["search"]
    assert matching[0] is UTIL_SPANS[0]


@pytest.mark.parametrize(
    "attributes",
    [
        {"gen_ai.tool.name": "SEARCH"},
        {"gen_ai.tool.name": "search*"},
        {"gen_ai.tool.name": "search", "retries": 1},
        {"cached": "false"},
        {"retries": "0"},
        {"missing": False},
        # Python-specific: False == 0 and True == 1, but JS === does not coerce.
        {"cached": 0},
        {"retries": False},
    ],
)
def test_filter_does_not_coerce(attributes: dict[str, Any]) -> None:
    assert filter_trace_spans(UTIL_SPANS, "execute_*", attributes) == []


def test_filter_name_only() -> None:
    expected = [UTIL_SPANS[0], UTIL_SPANS[1], UTIL_SPANS[3]]
    assert filter_trace_spans(UTIL_SPANS, "EXECUTE_*") == expected
    assert filter_trace_spans(UTIL_SPANS, "EXECUTE_*", {}) == expected


def test_filter_reads_only_own_literal_keys() -> None:
    candidates = [
        {**UTIL_SPANS[0], "spanId": "nested",
         "attributes": {"gen_ai": {"tool": {"name": "search"}}}},
        {**UTIL_SPANS[0], "spanId": "own",
         "attributes": {"__proto__": "literal", "gen_ai.tool.name": "search"}},
    ]  # fmt: skip
    found = filter_trace_spans(candidates, "*", {"gen_ai.tool.name": "search"})
    assert [s["spanId"] for s in found] == ["own"]
    found = filter_trace_spans(candidates, "*", {"__proto__": "literal"})
    assert [s["spanId"] for s in found] == ["own"]


# JS `null` cannot be told apart from an omitted filter in Python (both are None), and Map /
# Date have no JSON-config equivalent, so those upstream cases are not ported.
@pytest.mark.parametrize(
    "attributes",
    [[], "search", False, 0, {"tool": None}, {"tool": []}, {"tool": {}}, {"tool": UNDEFINED},
     {"retries": math.nan}, {"retries": math.inf}],
)  # fmt: skip
def test_filter_rejects_malformed_attributes(attributes: Any) -> None:
    with pytest.raises(
        AssertionValueError,
        match=re.escape(
            "Trace assertion attributes must be an object with string, boolean, or finite "
            "number values"
        ),
    ):
        filter_trace_spans([], "*", attributes)


@pytest.mark.parametrize(
    ("name", "pattern", "expected"),
    [
        ("llm.completion", "llm.completion", True),
        ("database.query", "database.query", True),
        ("llm.completion", "llm.chat", False),
        ("database.query", "api.call", False),
        ("llm.completion", "*", True),
        ("llm.completion", "llm.*", True),
        ("llm.completion", "*.completion", True),
        ("llm.completion", "*.*", True),
        ("llm.chat.stream", "*.*.*", True),
        ("llm.completion", "llm*completion", True),
        ("llm.chat.completion", "llm*completion", True),
        ("api.external.call", "api*call", True),
        ("llm.chat", "llm.c?at", True),
        ("llm.coat", "llm.c?at", True),
        ("llm.chat", "llm.???t", True),
        ("llm.ct", "llm.c?at", False),
        ("llm.chaat", "llm.c?at", False),
        ("LLM.COMPLETION", "llm.completion", True),
        ("llm.completion", "LLM.COMPLETION", True),
        ("LLM.Completion", "*llm*", True),
        ("test[0]", "test[0]", True),
        ("test(1)", "test(1)", True),
        ("price$100", "price$100", True),
        ("a+b", "a+b", True),
        ("a^b", "a^b", True),
        ("a|b", "a|b", True),
        ("path\\file", "path\\file", True),
        ("", "", True),
        ("something", "", False),
        ("", "*", True),
        ("", "something", False),
        ("api.v2.users.get", "api.*.users.*", True),
        ("api.v2.posts.get", "api.*.users.*", False),
        ("retrieval.search.vector", "*retrieval*", True),
        ("llm.openai.chat", "llm.*.chat", True),
    ],
)
def test_matches_pattern(name: str, pattern: str, expected: bool) -> None:
    assert matches_pattern(name, pattern) is expected


# --- trace.test.ts ----------------------------------------------------------------------------

MOCK_TEST: dict[str, Any] = {"vars": {"test": "value"}}
MOCK_RESPONSE: dict[str, Any] = {"output": "Test output"}
MOCK_TRACE: dict[str, Any] = {
    "traceId": "test-trace-id",
    "evaluationId": "test-evaluation-id",
    "testCaseId": "test-test-case-id",
    "metadata": {"test": "value"},
    "spans": [
        {"spanId": "span-1", "name": "http.request", "startTime": 1000, "endTime": 1500,
         "attributes": {"http.method": "GET"}, "statusCode": 200},
        {"spanId": "span-2", "parentSpanId": "span-1", "name": "api.call", "startTime": 1100,
         "endTime": 1400, "attributes": {"api.name": "test-api"}},
    ],
}  # fmt: skip
TRACE_UNDEFINED = "'trace' not in context"  # JS: context.trace === undefined


class FakeTraceStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.results: list[Any] = [None]
        self.error: Exception | None = None

    def returns(self, *results: Any) -> None:
        """Resolve each result once; the last one repeats (mockResolvedValue[Once])."""
        self.results = list(results)

    async def get_trace(self, trace_id: str, options: Any = None) -> Any:
        self.calls.append((trace_id, options))
        if self.error is not None:
            raise self.error
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        return copy.deepcopy(result)


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> FakeTraceStore:
    store = FakeTraceStore()
    monkeypatch.setattr(trace_store_module, "get_trace_store", lambda: store)
    for name in ("PROMPTFOO_TRACE_FETCH_MAX_ATTEMPTS",):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PROMPTFOO_TRACE_FETCH_RETRY_DELAY_MS", "0")
    monkeypatch.setenv("PROMPTFOO_TRACE_FETCH_STABLE_POLLS", "1")
    return store


async def _run(assertion: dict[str, Any], trace_id: str | None = "test-trace-id", **kw: Any):
    return await run_assertion(
        assertion=assertion,
        test=kw.pop("test", MOCK_TEST),
        provider_response=kw.pop("provider_response", MOCK_RESPONSE),
        trace_id=trace_id,
        **kw,
    )


FILTER_SPANS = [
    {"spanId": span_id, "name": "execute_tool", "startTime": 1000, "endTime": 1000 + duration,
     "attributes": {"gen_ai.tool.name": tool} if tool else {}, "statusCode": status}
    for span_id, duration, tool, status in [
        ("search-ok", 100, "search", 0),
        ("search-error", 300, "search", 2),
        ("fetch-error", 5000, "fetch", 2),
        ("unknown-error", 9000, None, 2),
    ]
]  # fmt: skip


@pytest.mark.parametrize(
    ("assertion", "passed"),
    [
        ({"type": "trace-span-count", "value": {"max": 2}}, True),
        ({"type": "trace-span-count", "value": {"min": 3}}, False),
        ({"type": "trace-span-duration", "value": {"max": 500}}, True),
        ({"type": "trace-span-duration", "value": {"max": 200, "percentile": 50}}, True),
        ({"type": "trace-span-duration", "value": {"max": 200, "percentile": 95}}, False),
        ({"type": "trace-error-spans", "value": {"max_count": 1}}, True),
        ({"type": "trace-error-spans", "value": {"max_percentage": 49}}, False),
        ({"type": "trace-error-spans", "value": {"max_percentage": 50}}, True),
    ],
)
async def test_filters_apply_before_grading(
    fake_store: FakeTraceStore, assertion: dict[str, Any], passed: bool
) -> None:
    fake_store.returns({**MOCK_TRACE, "spans": FILTER_SPANS})
    result = await _run(
        {
            **assertion,
            "value": {
                **assertion["value"],
                "pattern": "execute_*",
                "attributes": {"gen_ai.tool.name": "search"},
            },
        }
    )
    assert result["pass"] is passed
    assert result["score"] == (1 if passed else 0)
    assert 'Attribute filters applied to keys: ["gen_ai.tool.name"]' in result["reason"]
    assert "search" not in result["reason"]


@pytest.mark.parametrize(
    ("assertion", "passed"),
    [
        ({"type": "trace-span-count", "value": {"min": 1}}, False),
        ({"type": "trace-span-duration", "value": {"max": 500}}, True),
        ({"type": "trace-error-spans", "value": {"max_count": 0}}, True),
    ],
)
async def test_empty_attribute_selection_is_explained(
    fake_store: FakeTraceStore, assertion: dict[str, Any], passed: bool
) -> None:
    fake_store.returns(MOCK_TRACE)
    result = await _run(
        {
            **assertion,
            "value": {
                **assertion["value"],
                "pattern": "*",
                "attributes": {"gen_ai.tool.name": "private-tool-value"},
            },
        }
    )
    assert result["pass"] is passed
    assert 'Attribute filters applied to keys: ["gen_ai.tool.name"]' in result["reason"]
    assert "private-tool-value" not in result["reason"]


async def test_grader_span_uses_evaluation_tracing_context(fake_store: FakeTraceStore) -> None:
    calls: list[dict[str, Any]] = []
    test = {"metadata": {"evaluationId": "test-evaluation-id"},
            "vars": {"input": "ordinary user variable"}}  # fmt: skip

    async def with_grader_span(options: dict[str, Any], invoke: Any) -> Any:
        calls.append(options)
        return await invoke()

    async def with_provider_span(options: dict[str, Any], invoke: Any) -> Any:
        return await invoke(options["callContext"])

    await with_provider_call_tracing_context(
        {
            "getActiveTraceparent": lambda: None,
            "testIndex": 5,
            "withGraderSpan": with_grader_span,
            "withProviderSpan": with_provider_span,
        },
        lambda: _run({"type": "contains", "value": "Test"}, test=test),
    )
    assert calls == [{"graderId": "contains", "evalId": "test-evaluation-id", "testIndex": 5}]
    assert test["vars"] == {"input": "ordinary user variable"}


def test_ruby_assertions_are_trace_aware() -> None:
    assert assertion_uses_trace(
        {"type": "ruby", "value": "context.trace && context.trace.spans.length > 0"}
    )
    assert has_trace_aware_assertions([{"type": "ruby", "value": "true"}])
    assert has_trace_aware_assertions([{"type": "equals", "value": "file://check.py"}])
    assert not has_trace_aware_assertions([{"type": "equals", "value": "x"}])


async def test_passes_trace_data_to_script_assertion(fake_store: FakeTraceStore) -> None:
    fake_store.returns(MOCK_TRACE)
    code = (
        "if 'trace' not in context:\n    return False\n"
        "return len(context['trace']['spans']) == 2 and "
        "context['trace']['traceId'] == 'test-trace-id'"
    )
    result = await _run({"type": "python", "value": code})
    assert result["pass"] is True
    assert fake_store.calls == [("test-trace-id", {"sanitizeAttributes": False})]


async def test_retries_until_spans_are_available(
    fake_store: FakeTraceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROMPTFOO_TRACE_FETCH_MAX_ATTEMPTS", "3")
    fake_store.returns({**MOCK_TRACE, "spans": []}, MOCK_TRACE)
    result = await _run({"type": "python", "value": "len(context['trace']['spans']) == 2"})
    assert result["pass"] is True
    assert len(fake_store.calls) == 2


async def test_default_stable_polls_need_two_reads(
    fake_store: FakeTraceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Upstream also leaves the 250ms default retry delay in place under fake timers; the
    # delay stays 0 here to keep the test fast, the default stable-poll count still applies.
    monkeypatch.delenv("PROMPTFOO_TRACE_FETCH_STABLE_POLLS")
    fake_store.returns(MOCK_TRACE)
    result = await _run({"type": "python", "value": "len(context['trace']['spans']) == 2"})
    assert result["pass"] is True
    assert len(fake_store.calls) == 2


async def test_waits_for_span_count_to_stabilize(
    fake_store: FakeTraceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROMPTFOO_TRACE_FETCH_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("PROMPTFOO_TRACE_FETCH_STABLE_POLLS", "2")
    fake_store.returns({**MOCK_TRACE, "spans": [MOCK_TRACE["spans"][0]]}, MOCK_TRACE, MOCK_TRACE)
    result = await _run({"type": "python", "value": "len(context['trace']['spans']) == 2"})
    assert result["pass"] is True
    assert len(fake_store.calls) == 3


async def test_missing_trace_is_handled(fake_store: FakeTraceStore) -> None:
    fake_store.returns(None)
    result = await _run({"type": "python", "value": TRACE_UNDEFINED}, "non-existent-trace")
    assert result["pass"] is True


async def test_preloaded_missing_trace_is_reused(
    fake_store: FakeTraceStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROMPTFOO_TRACE_FETCH_MAX_ATTEMPTS", "2")
    fake_store.returns(None)
    assertion = {"type": "python", "value": TRACE_UNDEFINED}
    result = await run_assertions(
        test={**MOCK_TEST, "assert": [assertion, dict(assertion)]},
        provider_response=MOCK_RESPONSE,
        trace_id="non-existent-trace",
    )
    assert result["pass"] is True
    assert len(fake_store.calls) == 2


async def test_file_script_value_gets_trace_for_non_trace_types(
    fake_store: FakeTraceStore, tmp_path: Any
) -> None:
    fake_store.returns(MOCK_TRACE)
    (tmp_path / "rubric_generator.py").write_text(
        "def trace_span_name(output, context):\n"
        "    spans = (context.get('trace') or {}).get('spans') or []\n"
        "    return spans[0]['name'] if spans else 'missing-trace'\n"
    )
    with cli_state.with_base_path(str(tmp_path)):
        result = await _run(
            {"type": "equals", "value": "file://rubric_generator.py:trace_span_name"},
            provider_response={**MOCK_RESPONSE, "output": "http.request"},
        )
    assert result["pass"] is True


async def test_trace_duration(fake_store: FakeTraceStore) -> None:
    fake_store.returns(MOCK_TRACE)
    code = (
        "spans = context['trace']['spans']\n"
        "duration = max(s.get('endTime') or 0 for s in spans) - "
        "min(s['startTime'] for s in spans)\n"
        "return duration == 500"
    )
    result = await _run({"type": "python", "value": code})
    assert result["pass"] is True


async def test_detects_error_spans(fake_store: FakeTraceStore) -> None:
    fake_store.returns(
        {
            **MOCK_TRACE,
            "traceId": "error-trace",
            "spans": [
                *MOCK_TRACE["spans"],
                {"spanId": "error-span", "name": "failed.request", "startTime": 2000,
                 "endTime": 2100, "statusCode": 500, "statusMessage": "Internal Server Error"},
            ],
        }
    )  # fmt: skip
    code = (
        "errors = [s for s in context['trace']['spans'] if (s.get('statusCode') or 0) >= 400]\n"
        "return len(errors) == 1 and errors[0]['statusCode'] == 500"
    )
    result = await _run({"type": "python", "value": code}, "error-trace")
    assert result["pass"] is True


async def test_works_without_trace_id(fake_store: FakeTraceStore) -> None:
    code = "'trace' not in context and output == 'Test output'"
    result = await _run({"type": "python", "value": code}, None)
    assert result["pass"] is True
    assert fake_store.calls == []


async def test_python_assertion_gets_trace(fake_store: FakeTraceStore) -> None:
    fake_store.returns(MOCK_TRACE)
    code = (
        "if context.get('trace') is None:\n"
        "    return False\n"
        "return len(context['trace']['spans']) == 2"
    )
    result = await _run({"type": "python", "value": code})
    assert result["pass"] is True


async def test_python_span_hierarchy(fake_store: FakeTraceStore) -> None:
    fake_store.returns(MOCK_TRACE)
    code = (
        "root_spans = [s for s in context['trace']['spans'] if not s.get('parentSpanId')]\n"
        "leaf_spans = [s for s in context['trace']['spans'] if s.get('parentSpanId')]\n"
        "return len(root_spans) == 1 and len(leaf_spans) == 1"
    )
    result = await _run({"type": "python", "value": code})
    assert result["pass"] is True


async def test_python_grading_result(fake_store: FakeTraceStore) -> None:
    fake_store.returns(MOCK_TRACE)
    code = (
        "avg_duration = sum(s['endTime'] - s['startTime'] for s in context['trace']['spans'] "
        "if s.get('endTime')) / len(context['trace']['spans'])\n"
        "return {\n"
        "    'pass': True,\n"
        "    'score': 0.9,\n"
        "    'reason': f\"Average span duration: {avg_duration}ms\"\n"
        "}"
    )
    result = await _run({"type": "python", "value": code})
    assert result["pass"] is True
    assert result["score"] == 0.9
    assert "Average span duration" in result["reason"]


async def test_trace_store_errors_are_handled(fake_store: FakeTraceStore) -> None:
    fake_store.error = RuntimeError("Database error")
    result = await _run({"type": "python", "value": TRACE_UNDEFINED})
    assert result["pass"] is True
    assert fake_store.calls == [("test-trace-id", {"sanitizeAttributes": False})]
