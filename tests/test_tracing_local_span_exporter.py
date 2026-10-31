"""Ports test/tracing/localSpanExporter.test.ts, plus a round trip through the SQLite store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from promptphoo import db
from promptphoo.tracing import local_span_exporter
from promptphoo.tracing import store as store_module
from promptphoo.tracing.local_span_exporter import LocalSpanExporter, to_span_data
from promptphoo.tracing.otel_api import SpanContext, SpanStatus, SpanStatusCode
from promptphoo.tracing.otel_trace_sdk import ExportResultCode, resource_from_attributes

OPTIONS = {"skipTraceCheck": False, "warnIfMissingTrace": False}


@dataclass
class FakeSpan:
    trace_id: str = "trace-id-123"
    span_id: str = "span-id-456"
    parent_span_id: str | None = None
    name: str = "test-span"
    start_time: tuple[int, int] = (1000, 500_000_000)
    end_time: tuple[int, int] = (1001, 200_000_000)
    attributes: dict[str, Any] = field(default_factory=lambda: {"test.attr": "value"})
    resource_attributes: dict[str, Any] = field(default_factory=dict)
    status: SpanStatus = field(default_factory=lambda: SpanStatus(SpanStatusCode.OK))

    def span_context(self) -> SpanContext:
        return SpanContext(self.trace_id, self.span_id, 1)

    @property
    def parent_span_context(self) -> SpanContext | None:
        return SpanContext(self.trace_id, self.parent_span_id, 1) if self.parent_span_id else None

    @property
    def resource(self) -> Any:
        return resource_from_attributes(self.resource_attributes)


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, Any]], Any]] = []
        self.results: list[Any] = []

    async def add_spans(self, trace_id: str, spans: list[dict[str, Any]], options: Any = None):
        self.calls.append((trace_id, spans, options))
        result = self.results.pop(0) if self.results else {"stored": True}
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore()
    monkeypatch.setattr(store_module, "get_trace_store", lambda: fake)
    return fake


async def export(spans: list[Any]) -> Any:
    return await LocalSpanExporter().export(spans)  # type: ignore[arg-type]


async def test_exports_an_empty_batch(fake_store: FakeStore) -> None:
    assert (await export([])).code == ExportResultCode.SUCCESS
    assert fake_store.calls == []


async def test_exports_a_single_span(fake_store: FakeStore) -> None:
    result = await export([FakeSpan()])
    assert result.code == ExportResultCode.SUCCESS
    assert fake_store.calls == [
        (
            "trace-id-123",
            [
                {
                    "spanId": "span-id-456",
                    "name": "test-span",
                    "startTime": 1000500.0,
                    "endTime": 1001200.0,
                    "attributes": {"test.attr": "value"},
                    "statusCode": 1,
                }
            ],
            OPTIONS,
        )
    ]


async def test_groups_spans_by_trace_id(fake_store: FakeStore) -> None:
    await export(
        [
            FakeSpan(trace_id="trace-1", span_id="span-1"),
            FakeSpan(trace_id="trace-1", span_id="span-2"),
            FakeSpan(trace_id="trace-2", span_id="span-3"),
        ]
    )
    assert [(t, [s["spanId"] for s in spans]) for t, spans, _ in fake_store.calls] == [
        ("trace-1", ["span-1", "span-2"]),
        ("trace-2", ["span-3"]),
    ]


async def test_retries_once_when_the_trace_is_not_visible_yet(
    fake_store: FakeStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(local_span_exporter.asyncio, "sleep", fake_sleep)
    fake_store.results = [{"stored": False, "reason": "Trace trace-id-123 not found"}]
    result = await export([FakeSpan()])
    assert result.code == ExportResultCode.SUCCESS
    assert delays == [0.05]
    assert [(t, s[0]["spanId"], o) for t, s, o in fake_store.calls] == [
        ("trace-id-123", "span-id-456", OPTIONS),
        ("trace-id-123", "span-id-456", OPTIONS),
    ]


async def test_orphan_spans_are_skipped_after_the_retry(fake_store: FakeStore) -> None:
    fake_store.results = [{"stored": False}, {"stored": False, "reason": "missing"}]
    assert (await export([FakeSpan()])).code == ExportResultCode.SUCCESS
    assert len(fake_store.calls) == 2


def test_converts_hr_times_to_milliseconds() -> None:
    data = to_span_data(FakeSpan(start_time=(1234, 567_890_000), end_time=(1235, 0)))  # type: ignore[arg-type]
    assert data["startTime"] == pytest.approx(1234567.89)
    assert data["endTime"] == 1235000


def test_includes_the_parent_span_id() -> None:
    data = to_span_data(FakeSpan(parent_span_id="parent-span-789"))  # type: ignore[arg-type]
    assert data["parentSpanId"] == "parent-span-789"
    assert list(data)[:2] == ["spanId", "parentSpanId"]
    assert "parentSpanId" not in to_span_data(FakeSpan())  # type: ignore[arg-type]


def test_span_attributes_override_resource_attributes() -> None:
    data = to_span_data(
        FakeSpan(  # type: ignore[arg-type]
            attributes={"service.name": "span-override", "http.method": "GET", "count": 42},
            resource_attributes={"service.name": "promptfoo", "service.version": "0.125.1"},
        )
    )
    assert data["attributes"] == {
        "service.name": "span-override",
        "service.version": "0.125.1",
        "http.method": "GET",
        "count": 42,
    }


def test_includes_status_code_and_message() -> None:
    data = to_span_data(
        FakeSpan(status=SpanStatus(SpanStatusCode.ERROR, "Error occurred"))  # type: ignore[arg-type]
    )
    assert (data["statusCode"], data["statusMessage"]) == (2, "Error occurred")


async def test_returns_failed_on_export_error(fake_store: FakeStore) -> None:
    error = RuntimeError("Database error")
    fake_store.results = [error]
    result = await export([FakeSpan()])
    assert result.code == ExportResultCode.FAILED
    assert result.error is error


async def test_continues_with_other_traces_when_one_fails(fake_store: FakeStore) -> None:
    fake_store.results = [RuntimeError("First trace failed"), {"stored": True}]
    result = await export([FakeSpan(trace_id="trace-1"), FakeSpan(trace_id="trace-2")])
    assert [t for t, _, _ in fake_store.calls] == ["trace-1", "trace-2"]
    assert result.code == ExportResultCode.FAILED


async def test_foreign_key_errors_are_skipped(fake_store: FakeStore) -> None:
    fake_store.results = [RuntimeError("FOREIGN KEY constraint failed")]
    assert (await export([FakeSpan()])).code == ExportResultCode.SUCCESS


async def test_shutdown_and_force_flush_resolve() -> None:
    exporter = LocalSpanExporter()
    await exporter.shutdown()
    await exporter.force_flush()


@pytest.fixture
def sqlite_store(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("IS_TESTING", raising=False)
    monkeypatch.setattr(store_module, "_trace_store", None)
    db.close_db()
    db.run_db_migrations()
    db.get_db().execute(
        "INSERT INTO evals (id, created_at, config, results) VALUES ('eval-1', 0, '{}', '{}')"
    )
    yield store_module.get_trace_store()
    db.close_db()


async def test_round_trip_through_the_sqlite_store(sqlite_store: Any) -> None:
    trace_id = "ab" * 16
    await sqlite_store.create_trace(
        {"traceId": trace_id, "evaluationId": "eval-1", "testCaseId": "0-0"}
    )
    span = FakeSpan(
        trace_id=trace_id,
        span_id="cd" * 8,
        parent_span_id="ef" * 8,
        resource_attributes={"service.name": "promptfoo"},
        status=SpanStatus(SpanStatusCode.ERROR, "bad"),
    )
    orphan = FakeSpan(trace_id="99" * 16, span_id="11" * 8)
    assert (await export([span, orphan])).code == ExportResultCode.SUCCESS

    stored = await sqlite_store.get_spans(trace_id, {"sanitizeAttributes": False})
    assert stored == [
        {
            "spanId": "cd" * 8,
            "parentSpanId": "ef" * 8,
            "name": "test-span",
            "startTime": 1000500,
            "endTime": 1001200,
            "attributes": {"service.name": "promptfoo", "test.attr": "value"},
            "statusCode": 2,
            "statusMessage": "bad",
        }
    ]
    assert await sqlite_store.get_spans("99" * 16) == []
