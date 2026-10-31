"""Ports test/tracing/traceContext.test.ts, plus the local-store path of fetchTraceContext and
the TraceStore shim against a real SQLite database."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from promptphoo import db
from promptphoo.scheduler.cancellation import AbortController
from promptphoo.tracing import providers as providers_module
from promptphoo.tracing import store as store_module
from promptphoo.tracing import trace_context
from promptphoo.tracing.providers.types import TraceProviderError
from promptphoo.tracing.sanitize_attributes import sanitize_trace_attributes
from promptphoo.tracing.span_filter import is_relevant_span, matches_span_filter
from promptphoo.tracing.trace_context import (
    create_trace_spans,
    derive_insights,
    extract_trace_id_from_traceparent,
    fetch_trace_context,
)

PROVIDER_CONFIG = {"id": "tempo", "endpoint": "http://tempo:3200"}


class FakeStore:
    def __init__(self) -> None:
        self.stored: list[dict[str, Any]] = []
        self.add_calls: list[tuple[str, list[dict[str, Any]], Any]] = []
        self.get_calls: list[tuple[str, dict[str, Any]]] = []
        self.add_results: list[dict[str, Any]] = []
        self.get_overrides: list[list[dict[str, Any]]] = []

    async def add_spans(self, trace_id, spans, options=None):
        self.add_calls.append((trace_id, list(spans), options))
        if self.add_results:
            return self.add_results.pop(0)
        self.stored.extend(spans)
        return {"stored": True}

    async def get_spans(self, trace_id, options=None):
        options = options or {}
        self.get_calls.append((trace_id, dict(options)))
        if self.get_overrides:
            return self.get_overrides.pop(0)
        out = []
        for span in self.stored:
            earliest = options.get("earliestStartTime")
            if earliest and span["startTime"] < earliest:
                continue
            if options.get("spanFilter"):
                if not matches_span_filter(span["name"], options["spanFilter"]):
                    continue
            elif options.get("includeInternalSpans") is False and not is_relevant_span(
                {"attributes": span.get("attributes"), "statusCode": span.get("statusCode")}
            ):
                continue
            out.append(span)
        if options.get("maxSpans") is not None:
            out = out[: options["maxSpans"]]
        return [
            {
                **span,
                "attributes": span.get("attributes")
                if options.get("sanitizeAttributes") is False
                else sanitize_trace_attributes(span.get("attributes")),
            }
            for span in out
        ]


class FakeProvider:
    id = "tempo"

    def __init__(self, results: list[Any]) -> None:
        self.results = results
        self.calls: list[tuple[str, Any]] = []

    async def fetch_trace(self, trace_id, options=None):
        self.calls.append((trace_id, options))
        result = self.results[0] if len(self.results) == 1 else self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return result(options)
        return result


@pytest.fixture
def fake(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(store_module, "get_trace_store", lambda: store)
    monkeypatch.setattr(providers_module, "is_external_trace_provider", lambda config: True)
    state: dict[str, Any] = {"store": store, "created": 0}

    def mock_external(results: list[Any]) -> FakeProvider:
        provider = FakeProvider(results)

        def create(config):
            state["created"] += 1
            return provider

        monkeypatch.setattr(providers_module, "create_trace_provider", create)
        return provider

    state["mock"] = mock_external
    return state


def _trace(spans, trace_id="trace-1"):
    return {"fetchedAt": 123, "spans": spans, "traceId": trace_id}


async def test_persists_complete_trace_and_delegates_filtering(fake):
    internal = {"spanId": "internal", "name": "internal.setup", "startTime": 1,
                "attributes": {"otel.span.kind": "internal"}}  # fmt: skip
    target = {"spanId": "target", "name": "target.call", "startTime": 2,
              "attributes": {"otel.span.kind": "client"}}  # fmt: skip
    provider = fake["mock"]([_trace([internal, target], "other")])
    result = await fetch_trace_context(
        "trace-1",
        {"providerConfig": PROVIDER_CONFIG, "queryDelay": 0, "maxRetries": 0,
         "includeInternalSpans": False, "maxSpans": 1, "spanFilter": ["target"]},
    )  # fmt: skip
    assert provider.calls == [("trace-1", None)]
    assert fake["store"].add_calls == [
        ("trace-1", [internal, target], {"warnIfMissingTrace": False})
    ]
    assert fake["store"].get_calls == [
        ("trace-1", {"includeInternalSpans": False, "maxSpans": 1, "sanitizeAttributes": True,
                     "spanFilter": ["target"]})
    ]  # fmt: skip
    assert result is not None
    assert result["traceId"] == "trace-1"
    assert [s["name"] for s in result["spans"]] == ["target.call"]


async def test_keeps_meaningful_internal_spans_before_the_limit(fake):
    spans = [
        {"spanId": "http", "name": "POST /chat", "startTime": 1,
         "attributes": {"otel.span.kind": "server", "http.request.method": "POST"}},
        {"spanId": "handler", "name": "request handler", "startTime": 2,
         "attributes": {"otel.span.kind": "internal"}},
        {"spanId": "model", "parentSpanId": "tool", "name": "chat gpt-4.1-mini", "startTime": 3,
         "attributes": {"otel.span.kind": "internal", "gen_ai.operation.name": "chat"}},
        {"spanId": "tool", "parentSpanId": "handler", "name": "execute_tool search",
         "startTime": 4,
         "attributes": {"otel.span.kind": "internal", "gen_ai.tool.name": "search"}},
    ]  # fmt: skip
    fake["mock"]([_trace(spans)])
    result = await fetch_trace_context(
        "trace-1",
        {"providerConfig": PROVIDER_CONFIG, "queryDelay": 0, "maxRetries": 0,
         "includeInternalSpans": False, "maxSpans": 2},
    )  # fmt: skip
    assert fake["store"].stored == spans
    assert result is not None
    assert [s["depth"] for s in result["spans"]] == [1, 0]
    assert [s["name"] for s in result["spans"]] == ["chat gpt-4.1-mini", "execute_tool search"]


async def test_applies_wildcard_filters_to_external_spans(fake):
    fake["mock"](
        [_trace([
            {"spanId": "model", "name": "chat gpt-4.1-mini", "startTime": 1,
             "attributes": {"otel.span.kind": "internal", "gen_ai.operation.name": "chat"}},
            {"spanId": "tool", "name": "execute_tool search", "startTime": 2,
             "attributes": {"otel.span.kind": "internal", "gen_ai.tool.name": "search"}},
        ])]
    )  # fmt: skip
    result = await fetch_trace_context(
        "trace-1",
        {"providerConfig": PROVIDER_CONFIG, "queryDelay": 0, "maxRetries": 0,
         "includeInternalSpans": False, "spanFilter": ["*tool*"]},
    )  # fmt: skip
    assert result is not None
    assert [s["name"] for s in result["spans"]] == ["execute_tool search"]


async def test_uses_the_store_time_window(fake):
    previous = {"spanId": "previous", "name": "previous.call", "startTime": 100}
    current = {"spanId": "current", "name": "current.call", "startTime": 200}
    provider = fake["mock"]([_trace([previous, current])])
    result = await fetch_trace_context(
        "trace-1",
        {"providerConfig": PROVIDER_CONFIG, "earliestStartTime": 150, "maxRetries": 0,
         "queryDelay": 0},
    )  # fmt: skip
    assert provider.calls == [("trace-1", {"earliestStartTime": 150})]
    assert fake["store"].stored == [previous, current]
    assert fake["store"].get_calls[0][1]["earliestStartTime"] == 150
    assert result is not None
    assert [s["name"] for s in result["spans"]] == ["current.call"]


async def test_forwards_query_bounds_and_cancellation(fake):
    controller = AbortController()
    provider = fake["mock"]([_trace([{"spanId": "current", "name": "target.call",
                                      "startTime": 200}])])  # fmt: skip
    await fetch_trace_context(
        "trace-1",
        {"providerConfig": PROVIDER_CONFIG, "abortSignal": controller.signal,
         "earliestStartTime": 150, "includeInternalSpans": True, "maxRetries": 0,
         "maxSpans": 50, "queryDelay": 0, "sanitizeAttributes": False},
    )  # fmt: skip
    assert provider.calls == [
        ("trace-1", {"earliestStartTime": 150, "maxSpans": 50, "abortSignal": controller.signal})
    ]


@pytest.mark.parametrize("filters", [{"includeInternalSpans": False}, {"spanFilter": ["*tool*"]}])
async def test_applies_span_limit_after_filtering(fake, filters):
    spans = [
        {"spanId": "internal", "name": "internal.setup", "startTime": 1,
         "attributes": {"otel.span.kind": "internal"}},
        {"spanId": "tool", "name": "execute_tool search", "startTime": 2,
         "attributes": {"gen_ai.tool.name": "search"}},
    ]  # fmt: skip

    def result(options):
        limit = (options or {}).get("maxSpans")
        return _trace(spans if limit is None else spans[:limit])

    provider = fake["mock"]([result])
    out = await fetch_trace_context(
        "trace-1",
        {"providerConfig": PROVIDER_CONFIG, "maxRetries": 0, "maxSpans": 1, "queryDelay": 0,
         **filters},
    )  # fmt: skip
    assert provider.calls == [("trace-1", None)]
    assert out is not None
    assert [s["name"] for s in out["spans"]] == ["execute_tool search"]


async def test_preserves_time_bounds_when_limit_is_local(fake):
    controller = AbortController()
    provider = fake["mock"]([_trace([{"spanId": "tool", "name": "execute_tool search",
                                      "startTime": 200}])])  # fmt: skip
    await fetch_trace_context(
        "trace-1",
        {"providerConfig": PROVIDER_CONFIG, "abortSignal": controller.signal,
         "earliestStartTime": 150, "maxRetries": 0, "maxSpans": 50, "queryDelay": 0,
         "spanFilter": ["tool"]},
    )  # fmt: skip
    assert provider.calls == [
        ("trace-1", {"earliestStartTime": 150, "abortSignal": controller.signal})
    ]


async def test_discards_cyclic_parent_relationships(fake):
    fake["mock"](
        [_trace([
            {"spanId": "cycle-a", "parentSpanId": "cycle-b", "name": "cycle.a", "startTime": 1},
            {"spanId": "cycle-b", "parentSpanId": "cycle-a", "name": "cycle.b", "startTime": 2},
            {"spanId": "valid", "name": "target.call", "startTime": 3},
        ])]
    )  # fmt: skip
    result = await fetch_trace_context(
        "trace-1", {"providerConfig": PROVIDER_CONFIG, "queryDelay": 0, "maxRetries": 0}
    )
    assert [s["spanId"] for s in fake["store"].stored] == ["valid"]
    assert result is not None
    assert [s["name"] for s in result["spans"]] == ["target.call"]


async def test_redacts_configured_attribute_values_before_persistence(fake):
    fake["mock"](
        [_trace([{
            "spanId": "target", "name": "request used secret-token",
            "statusMessage": "account pin 123456 failed", "startTime": 1,
            "attributes": {"nested": {"authorization": "secret-token"}, "account.pin": 123456},
        }])]
    )  # fmt: skip
    await fetch_trace_context(
        "trace-1",
        {"providerConfig": PROVIDER_CONFIG, "queryDelay": 0, "maxRetries": 0,
         "redactAttributes": ["authorization", "pin"]},
    )  # fmt: skip
    [stored] = fake["store"].stored
    assert stored["name"] == "request used [REDACTED]"
    assert stored["statusMessage"] == "account pin [REDACTED] failed"
    assert stored["attributes"] == {
        "nested": {"authorization": "[REDACTED]"},
        "account.pin": "[REDACTED]",
    }


async def test_stores_large_traces_in_batches(fake):
    spans = [{"spanId": str(i), "name": "target.call", "startTime": i} for i in range(501)]
    fake["mock"]([_trace(spans)])
    await fetch_trace_context(
        "trace-1", {"providerConfig": PROVIDER_CONFIG, "queryDelay": 0, "maxRetries": 0}
    )
    calls = fake["store"].add_calls
    assert [len(c[1]) for c in calls] == [500, 1]
    assert calls[1][2] == {"warnIfMissingTrace": False, "skipTraceCheck": True}


async def test_waits_before_the_initial_request_and_retries(fake, monkeypatch):
    waits: list[float] = []

    async def fake_wait(delay, signal=None):
        waits.append(delay)

    monkeypatch.setattr(trace_context, "_wait_for_retry", fake_wait)
    provider = fake["mock"](
        [None, _trace([{"spanId": "target", "name": "target.call", "startTime": 1}])]
    )
    result = await fetch_trace_context(
        "trace-1",
        {"providerConfig": PROVIDER_CONFIG, "queryDelay": 3000, "maxRetries": 1,
         "retryDelayMs": 500},
    )  # fmt: skip
    assert result is not None
    assert waits == [3000, 500]
    assert len(provider.calls) == 2


async def test_cancels_the_initial_delay_without_invoking_the_provider(fake):
    controller = AbortController()
    provider = fake["mock"]([_trace([])])
    task = asyncio.ensure_future(
        fetch_trace_context(
            "trace-1",
            {
                "providerConfig": PROVIDER_CONFIG,
                "queryDelay": 60_000,
                "abortSignal": controller.signal,
            },
        )  # fmt: skip
    )
    await asyncio.sleep(0)
    controller.abort()
    with pytest.raises(Exception) as excinfo:
        await task
    assert getattr(excinfo.value, "name", None) == "AbortError"
    assert str(excinfo.value) == "cancelled by user"
    assert provider.calls == []


async def test_permanent_provider_errors_are_not_retried(fake):
    provider = fake["mock"]([TraceProviderError("unauthorized", status_code=401)])
    result = await fetch_trace_context(
        "trace-1", {"providerConfig": PROVIDER_CONFIG, "queryDelay": 0, "maxRetries": 3}
    )
    assert result is None
    assert len(provider.calls) == 1


async def test_retryable_provider_errors_are_retried(fake, monkeypatch):
    async def no_wait(delay, signal=None):
        return None

    monkeypatch.setattr(trace_context, "_wait_for_retry", no_wait)
    provider = fake["mock"](
        [
            TraceProviderError("busy", status_code=503, retryable=True),
            _trace([{"spanId": "t", "name": "target.call", "startTime": 1}]),
        ]
    )
    result = await fetch_trace_context(
        "trace-1", {"providerConfig": PROVIDER_CONFIG, "queryDelay": 0, "maxRetries": 3}
    )
    assert result is not None
    assert len(provider.calls) == 2


async def test_returns_none_when_spans_cannot_be_persisted(fake):
    fake["mock"]([_trace([{"spanId": "target", "name": "target.call", "startTime": 1}])])
    fake["store"].add_results.append({"stored": False})
    result = await fetch_trace_context(
        "trace-1", {"providerConfig": PROVIDER_CONFIG, "queryDelay": 0, "maxRetries": 0}
    )
    assert result is None
    assert fake["store"].get_calls == []


async def test_shares_simultaneous_requests(fake):
    provider = fake["mock"]([_trace([{"spanId": "target", "name": "target.call",
                                      "startTime": 1}])])  # fmt: skip
    options = {"providerConfig": PROVIDER_CONFIG, "queryDelay": 0, "maxRetries": 0}
    await asyncio.gather(
        fetch_trace_context("trace-1", options), fetch_trace_context("trace-1", options)
    )
    assert len(provider.calls) == 1
    assert len(fake["store"].add_calls) == 1


async def test_local_traces_use_the_store(fake, monkeypatch):
    monkeypatch.setattr(providers_module, "is_external_trace_provider", lambda config: False)
    fake["store"].get_overrides.append([{"spanId": "local", "name": "local.call", "startTime": 1}])
    result = await fetch_trace_context("trace-1", {"maxRetries": 0})
    assert result is not None
    assert [s["name"] for s in result["spans"]] == ["local.call"]
    assert fake["created"] == 0


@pytest.mark.parametrize(
    "traceparent",
    [
        "",
        "oops-../../admin-nothex-01",
        "00-00000000000000000000000000000000-0123456789abcdef-01",
        "00-0123456789abcdef0123456789abcdef-0000000000000000-01",
    ],
)
def test_rejects_invalid_traceparents(traceparent):
    assert extract_trace_id_from_traceparent(traceparent) is None


def test_normalizes_valid_traceparents():
    assert (
        extract_trace_id_from_traceparent("00-0123456789ABCDEF0123456789ABCDEF-0123456789abcdef-01")
        == "0123456789abcdef0123456789abcdef"
    )


# --- TraceContextData shape and insights -------------------------------------------------------


def test_trace_spans_and_insights():
    spans = [
        {"spanId": "root0000aaaa", "name": "agent", "startTime": 10, "endTime": 50,
         "attributes": {"otel.span.kind_code": 2}, "statusCode": 1},
        {"spanId": "tool0000bbbb", "parentSpanId": "root0000aaaa", "name": "call", "startTime": 20,
         "endTime": 25, "attributes": {"tool.name": "search", "span.kind": "CLIENT"}},
        {"spanId": "err00000cccc", "parentSpanId": "tool0000bbbb", "name": "boom", "startTime": 30,
         "statusCode": 2, "statusMessage": "kaput"},
        {"spanId": "guard000dddd", "name": "check", "startTime": 40, "endTime": 30,
         "attributes": {"guardrail.name": "pii", "guardrail.decision": "block"}},
    ]  # fmt: skip
    trace_spans = create_trace_spans(spans)
    assert trace_spans[0] == {
        "spanId": "root0000aaaa", "name": "agent", "kind": "server", "startTime": 10,
        "endTime": 50, "durationMs": 40, "attributes": {"otel.span.kind_code": 2},
        "status": {"code": "ok"}, "depth": 0, "events": [],
    }  # fmt: skip
    assert [s["kind"] for s in trace_spans] == ["server", "client", "unspecified", "unspecified"]
    assert [s["depth"] for s in trace_spans] == [0, 1, 2, 0]
    assert [s["durationMs"] for s in trace_spans] == [40, 5, 0, 0]
    assert trace_spans[2]["status"] == {"code": "error", "message": "kaput"}
    assert derive_insights(trace_spans) == [
        'Error span "boom" (err00000): kaput',
        'Tool call search via "call" (duration 5ms)',
        "Guardrail pii decision: block",
    ]


# --- the local store path over SQLite ---------------------------------------------------------


@pytest.fixture
def real_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("IS_TESTING", raising=False)
    db.close_db()
    db.run_db_migrations()
    conn = db.get_db()
    conn.execute(
        "INSERT INTO evals (id, created_at, config, results) VALUES ('eval-1', 0, '{}', '{}')"
    )
    yield
    db.close_db()


async def test_local_store_round_trip(real_db):
    store = store_module.get_trace_store()
    trace_id = "a" * 32
    await store.create_trace(
        {"traceId": trace_id, "evaluationId": "eval-1", "testCaseId": "0-0",
         "metadata": {"testIdx": 0}}
    )  # fmt: skip
    missing = await store.add_spans("b" * 32, [{"spanId": "x", "name": "x", "startTime": 1}])
    assert missing == {"stored": False, "reason": f"Trace {'b' * 32} not found"}
    spans = [
        {"spanId": "root", "name": "agent.run", "startTime": 100, "endTime": 400,
         "attributes": {"api_key": "sk-123", "gen_ai.request.model": "m"}},
        {"spanId": "grader", "parentSpanId": "root", "name": "grade", "startTime": 150,
         "endTime": 160, "attributes": {"promptfoo.span.role": "grader"}},
        {"spanId": "graded", "parentSpanId": "grader", "name": "chat judge", "startTime": 155,
         "attributes": {"gen_ai.operation.name": "chat"}},
        {"spanId": "tool", "parentSpanId": "root", "name": "tool.search", "startTime": 200,
         "endTime": 260, "attributes": {"tool.name": "search"}, "statusCode": 2,
         "statusMessage": "timeout"},
    ]  # fmt: skip
    assert await store.add_spans(trace_id, spans) == {"stored": True}
    # Duplicate span ids are ignored.
    assert await store.add_spans(trace_id, spans[:1]) == {"stored": True}

    trace = await store.get_trace(trace_id, {"sanitizeAttributes": False})
    assert trace is not None
    assert trace["metadata"] == {"testIdx": 0}
    assert len(trace["spans"]) == 4
    assert trace["spans"][0]["attributes"]["api_key"] == "sk-123"
    sanitized = await store.get_trace(trace_id)
    assert sanitized is not None
    assert sanitized["spans"][0]["attributes"]["api_key"] == "<redacted>"
    assert await store.get_trace_metadata(trace_id) == {"testIdx": 0}
    assert [t["traceId"] for t in await store.get_traces_by_evaluation("eval-1")] == [trace_id]

    relevant = await store.get_spans(trace_id, {"includeInternalSpans": False})
    assert [s["spanId"] for s in relevant] == ["root", "tool"]
    shallow = await store.get_spans(trace_id, {"maxDepth": 2})
    assert [s["spanId"] for s in shallow] == ["root", "grader", "tool"]

    context = await fetch_trace_context(trace_id, {"maxRetries": 0, "spanFilter": ["tool*"]})
    assert context is not None
    assert [s["name"] for s in context["spans"]] == ["tool.search"]
    assert context["insights"] == [
        'Error span "tool.search" (tool): timeout',
        'Tool call search via "tool.search" (duration 60ms)',
    ]

    await store.delete_old_traces(0)
    assert await store.get_trace(trace_id) is None
