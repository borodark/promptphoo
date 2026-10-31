"""Ports test/tracing/otelSdk.test.ts, test/tracing/otelConfig.test.ts and the
resourceAttributes.ts constants, plus behaviour tests for the in-house OpenTelemetry API and
trace SDK stand-ins (otel_api.py, otel_trace_sdk.py) and the OTLP/HTTP JSON exporter."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx
import pytest
import respx

from promptphoo import UPSTREAM_VERSION
from promptphoo.tracing import otel_sdk, otlp_http_exporter
from promptphoo.tracing.local_span_exporter import LocalSpanExporter
from promptphoo.tracing.otel_api import (
    ROOT_CONTEXT,
    NonRecordingSpan,
    NoopTextMapPropagator,
    SpanContext,
    SpanKind,
    SpanOptions,
    SpanStatusCode,
    W3CTraceContextPropagator,
    context,
    parse_trace_parent,
    propagation,
    suppress_tracing,
    trace,
)
from promptphoo.tracing.otel_config import get_default_otel_config, get_otel_config_from_env
from promptphoo.tracing.otel_trace_sdk import (
    BasicTracerProvider,
    BatchSpanProcessor,
    ExportResult,
    ExportResultCode,
    InMemorySpanExporter,
    SimpleSpanProcessor,
    SpanImpl,
    SpanLimits,
    add_hr_times,
    hr_time_duration,
    hr_time_to_nanoseconds,
    millis_to_hr_time,
    resource_from_attributes,
)
from promptphoo.tracing.otlp_http_exporter import (
    OTLPTraceExporter,
    create_export_trace_service_request,
    parse_key_pairs_into_record,
    parse_retry_after_to_millis,
    to_any_value,
)
from promptphoo.tracing.resource_attributes import (
    PROMPTFOO_RESOURCE_ATTR_PARENT_SPAN_ID,
    PROMPTFOO_RESOURCE_ATTR_TRACE_ID,
)

OTEL_ENV = (
    "PROMPTFOO_OTEL_ENABLED",
    "PROMPTFOO_OTEL_SERVICE_NAME",
    "PROMPTFOO_OTEL_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "PROMPTFOO_OTEL_LOCAL_EXPORT",
    "PROMPTFOO_OTEL_DEBUG",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_EXPORTER_OTLP_TIMEOUT",
    "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT",
    "OTEL_BSP_MAX_EXPORT_BATCH_SIZE",
    "OTEL_BSP_MAX_QUEUE_SIZE",
    "OTEL_BSP_SCHEDULE_DELAY",
    "OTEL_BSP_EXPORT_TIMEOUT",
)


@pytest.fixture(autouse=True)
def _clean_globals(monkeypatch: pytest.MonkeyPatch):
    for name in OTEL_ENV:
        monkeypatch.delenv(name, raising=False)
    level = logging.getLogger("promptphoo.tracing").level
    yield
    otel_sdk._provider = None
    otel_sdk._initialized = False
    trace.disable()
    propagation.disable()
    logging.getLogger("promptphoo.tracing").setLevel(level)


def _in_memory_provider(**kwargs: Any) -> tuple[BasicTracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = BasicTracerProvider(span_processors=[SimpleSpanProcessor(exporter)], **kwargs)
    provider.register()
    return provider, exporter


# --- resourceAttributes.ts ------------------------------------------------------------------


def test_resource_attribute_keys():
    assert PROMPTFOO_RESOURCE_ATTR_TRACE_ID == "promptfoo.trace_id"
    assert PROMPTFOO_RESOURCE_ATTR_PARENT_SPAN_ID == "promptfoo.parent_span_id"


# --- otelConfig.test.ts ---------------------------------------------------------------------


def test_default_config_when_no_env_vars_are_set():
    assert get_otel_config_from_env() == {
        "enabled": False,
        "serviceName": "promptfoo",
        "localExport": True,
        "debug": False,
    }


def test_reads_promptfoo_otel_enabled(monkeypatch):
    monkeypatch.setenv("PROMPTFOO_OTEL_ENABLED", "true")
    assert get_otel_config_from_env()["enabled"] is True


def test_reads_service_name(monkeypatch):
    monkeypatch.setenv("PROMPTFOO_OTEL_SERVICE_NAME", "my-service")
    assert get_otel_config_from_env()["serviceName"] == "my-service"


def test_prefers_promptfoo_endpoint_over_standard(monkeypatch):
    monkeypatch.setenv("PROMPTFOO_OTEL_ENDPOINT", "http://custom:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://standard:4318")
    assert get_otel_config_from_env().get("endpoint") == "http://custom:4318"


def test_falls_back_to_standard_endpoint(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://standard:4318")
    assert get_otel_config_from_env().get("endpoint") == "http://standard:4318"


def test_reads_local_export(monkeypatch):
    monkeypatch.setenv("PROMPTFOO_OTEL_LOCAL_EXPORT", "false")
    assert get_otel_config_from_env()["localExport"] is False


def test_reads_debug(monkeypatch):
    monkeypatch.setenv("PROMPTFOO_OTEL_DEBUG", "true")
    assert get_otel_config_from_env()["debug"] is True


def test_default_config_switches_tracing_on():
    assert get_default_otel_config()["enabled"] is True


# --- otelSdk.test.ts ------------------------------------------------------------------------

BASE_CONFIG: dict[str, Any] = {
    "enabled": True,
    "serviceName": "promptfoo-test",
    "localExport": False,
    "debug": False,
}


def _processors() -> list[Any]:
    assert otel_sdk._provider is not None
    return otel_sdk._provider._active_span_processor._processors


def test_does_not_initialize_when_disabled():
    otel_sdk.initialize_otel({**BASE_CONFIG, "enabled": False})  # type: ignore[typeddict-item]
    assert otel_sdk.is_otel_initialized() is False
    assert trace.get_tracer_provider() is None


def test_initializes_and_registers_provider():
    otel_sdk.initialize_otel(BASE_CONFIG)  # type: ignore[arg-type]
    assert otel_sdk.is_otel_initialized() is True
    assert trace.get_tracer_provider() is otel_sdk._provider
    assert isinstance(propagation._propagator, W3CTraceContextPropagator)


def test_adds_local_span_processor_when_local_export():
    otel_sdk.initialize_otel({**BASE_CONFIG, "localExport": True})  # type: ignore[arg-type]
    [processor] = _processors()
    assert isinstance(processor, BatchSpanProcessor)
    assert isinstance(processor._exporter, LocalSpanExporter)


def test_adds_otlp_exporter_when_endpoint_configured():
    otel_sdk.initialize_otel(
        {**BASE_CONFIG, "localExport": True, "endpoint": "http://localhost:4318/v1/traces"}  # type: ignore[arg-type]
    )
    local, otlp = _processors()
    assert isinstance(local._exporter, LocalSpanExporter)
    assert isinstance(otlp._exporter, OTLPTraceExporter)
    assert otlp._exporter.url == "http://localhost:4318/v1/traces"


def test_skips_local_export_when_disabled():
    otel_sdk.initialize_otel(
        {**BASE_CONFIG, "endpoint": "http://localhost:4318/v1/traces"}  # type: ignore[arg-type]
    )
    [processor] = _processors()
    assert isinstance(processor._exporter, OTLPTraceExporter)


def test_enables_debug_logging():
    logging.getLogger("promptphoo.tracing").setLevel(logging.WARNING)
    otel_sdk.initialize_otel({**BASE_CONFIG, "debug": True})  # type: ignore[arg-type]
    assert logging.getLogger("promptphoo.tracing").level == logging.DEBUG


def test_does_not_reinitialize():
    otel_sdk.initialize_otel(BASE_CONFIG)  # type: ignore[arg-type]
    first = otel_sdk._provider
    otel_sdk.initialize_otel(BASE_CONFIG)  # type: ignore[arg-type]
    assert otel_sdk._provider is first


def test_resource_has_service_name_and_version():
    otel_sdk.initialize_otel(BASE_CONFIG)  # type: ignore[arg-type]
    assert otel_sdk._provider is not None
    assert otel_sdk._provider.resource.attributes == {
        "service.name": "promptfoo-test",
        "service.version": UPSTREAM_VERSION,
    }


async def test_shutdown_without_initialization_is_a_no_op():
    await otel_sdk.shutdown_otel()
    await otel_sdk.flush_otel()
    assert otel_sdk.is_otel_initialized() is False


class _RecordingProcessor:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []
        self.ended: list[Any] = []

    def on_start(self, span, parent_context):
        self.calls.append("start")

    def on_end(self, span):
        self.ended.append(span)

    async def force_flush(self):
        self.calls.append("flush")
        if self.fail:
            raise RuntimeError("flush failed")

    async def shutdown(self):
        self.calls.append("shutdown")
        if self.fail:
            raise RuntimeError("shutdown failed")


def _init_with(processor: _RecordingProcessor) -> None:
    otel_sdk.initialize_otel(BASE_CONFIG)  # type: ignore[arg-type]
    assert otel_sdk._provider is not None
    otel_sdk._provider._active_span_processor._processors.append(processor)


async def test_shutdown_calls_provider_shutdown():
    processor = _RecordingProcessor()
    _init_with(processor)
    await otel_sdk.shutdown_otel()
    assert processor.calls == ["shutdown"]
    assert otel_sdk.is_otel_initialized() is False
    assert trace.get_tracer_provider() is None


async def test_shutdown_errors_are_handled(caplog):
    _init_with(_RecordingProcessor(fail=True))
    with caplog.at_level(logging.ERROR, logger="promptphoo.tracing"):
        await otel_sdk.shutdown_otel()
    assert "Error shutting down OpenTelemetry SDK" in caplog.text
    assert otel_sdk.is_otel_initialized() is False


async def test_flush_calls_force_flush():
    processor = _RecordingProcessor()
    _init_with(processor)
    await otel_sdk.flush_otel()
    assert processor.calls == ["flush"]


async def test_flush_errors_are_handled(caplog):
    _init_with(_RecordingProcessor(fail=True))
    with caplog.at_level(logging.ERROR, logger="promptphoo.tracing"):
        await otel_sdk.flush_otel()
    assert "Error flushing spans" in caplog.text


async def test_is_initialized_lifecycle_and_reinit_after_shutdown():
    assert otel_sdk.is_otel_initialized() is False
    otel_sdk.initialize_otel(BASE_CONFIG)  # type: ignore[arg-type]
    first = otel_sdk._provider
    assert otel_sdk.is_otel_initialized() is True
    await otel_sdk.shutdown_otel()
    assert otel_sdk.is_otel_initialized() is False
    # Unlike Node, a later initialize installs a fresh provider.
    otel_sdk.initialize_otel(BASE_CONFIG)  # type: ignore[arg-type]
    assert otel_sdk._provider is not first
    assert trace.get_tracer_provider() is otel_sdk._provider


# --- HrTime ---------------------------------------------------------------------------------


def test_hr_time_helpers():
    assert millis_to_hr_time(1_700_000_000_123) == (1_700_000_000, 123_000_000)
    assert millis_to_hr_time(1.5) == (0, 1_500_000)
    assert add_hr_times((1, 900_000_000), (0, 200_000_000)) == (2, 100_000_000)
    assert hr_time_duration((1, 900_000_000), (3, 100_000_000)) == (1, 200_000_000)
    assert hr_time_to_nanoseconds((2, 5)) == 2_000_000_005


def test_span_times_are_hr_time_from_wall_clock():
    _, exporter = _in_memory_provider()
    before = time.time() * 1000
    span = trace.get_tracer("t").start_span("s")
    span.end()
    after = time.time() * 1000
    [finished] = exporter.get_finished_spans()
    start_ms = finished.start_time[0] * 1e3 + finished.start_time[1] / 1e6
    end_ms = finished.end_time[0] * 1e3 + finished.end_time[1] / 1e6
    # Start is Date.now() as integer milliseconds.
    assert finished.start_time[1] % 1_000_000 == 0
    assert before - 1 <= start_ms <= after
    assert start_ms <= end_ms <= after + 1


def test_explicit_times_and_end_before_start():
    _, exporter = _in_memory_provider()
    span = trace.get_tracer("t").start_span("s", SpanOptions(start_time=2000))
    span.end(1000)
    [finished] = exporter.get_finished_spans()
    assert finished.start_time == (2, 0)
    assert finished.end_time == (2, 0)
    assert finished.duration == (0, 0)


def test_end_twice_is_ignored():
    _, exporter = _in_memory_provider()
    span = trace.get_tracer("t").start_span("s")
    span.end()
    span.end()
    span.set_attribute("late", 1)
    assert len(exporter.get_finished_spans()) == 1
    assert "late" not in exporter.get_finished_spans()[0].attributes


# --- status ---------------------------------------------------------------------------------


def _span(**kwargs: Any) -> SpanImpl:
    _in_memory_provider(**kwargs)
    span = trace.get_tracer("t").start_span("s")
    assert isinstance(span, SpanImpl)
    return span


def test_status_unset_is_ignored_and_ok_is_final():
    span = _span()
    span.set_status(SpanStatusCode.ERROR, "boom")
    assert (span.status.code, span.status.message) == (SpanStatusCode.ERROR, "boom")
    span.set_status(SpanStatusCode.UNSET)
    assert span.status.code == SpanStatusCode.ERROR
    span.set_status(SpanStatusCode.OK, "ignored message")
    assert (span.status.code, span.status.message) == (SpanStatusCode.OK, None)
    span.set_status(SpanStatusCode.ERROR, "later")
    assert span.status.code == SpanStatusCode.OK


def test_status_message_must_be_a_string():
    span = _span()
    span.set_status(SpanStatusCode.ERROR, 42)
    assert (span.status.code, span.status.message) == (SpanStatusCode.ERROR, None)


# --- attributes -----------------------------------------------------------------------------


def test_attribute_drop_rules():
    span = _span()
    span.set_attribute("none", None)
    span.set_attribute("", "empty key")
    span.set_attribute("obj", {"a": 1})
    span.set_attribute("mixed", [1, "a"])
    span.set_attribute("tuple", (1, 2))
    span.set_attribute("with_none", ["a", None, "b"])
    span.set_attributes({"s": "x", "n": 1.5, "b": True})
    assert span.attributes == {
        "tuple": [1, 2],
        "with_none": ["a", None, "b"],
        "s": "x",
        "n": 1.5,
        "b": True,
    }


def test_attribute_count_limit_is_128():
    span = _span()
    for i in range(130):
        span.set_attribute(f"k{i}", i)
    assert len(span.attributes) == 128
    assert span.dropped_attributes_count == 2
    span.set_attribute("k0", "overwrite")  # existing keys can still change
    assert span.attributes["k0"] == "overwrite"


def test_attribute_value_length_limit_truncates():
    span = _span(span_limits=SpanLimits(attribute_value_length_limit=3))
    span.set_attribute("s", "abcdef")
    span.set_attribute("a", ["abcdef", "xy"])
    assert span.attributes == {"s": "abc", "a": ["abc", "xy"]}


def test_start_span_sanitizes_option_attributes():
    span = trace.get_tracer("t")
    _in_memory_provider()
    started = span.start_span("s", SpanOptions(attributes={"ok": 1, "bad": {"x": 1}}))
    assert isinstance(started, SpanImpl)
    assert started.attributes == {"ok": 1}


def test_record_exception_adds_event():
    span = _span()
    span.record_exception(ValueError("bad value"))
    [event] = span.events
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "ValueError"
    assert event.attributes["exception.message"] == "bad value"


# --- parentage and context ------------------------------------------------------------------


async def test_children_inherit_trace_id_through_active_context():
    _, exporter = _in_memory_provider()
    tracer = trace.get_tracer("t")

    async def outer(span: Any) -> None:
        async def inner(child: Any) -> None:
            child.end()

        await tracer.start_active_span("child", inner)
        span.end()

    await tracer.start_active_span("parent", outer, SpanOptions(kind=SpanKind.CLIENT))
    child, parent = exporter.get_finished_spans()
    assert parent.parent_span_context is None
    assert parent.kind == SpanKind.CLIENT
    assert child.parent_span_context == parent.span_context()
    assert child.span_context().trace_id == parent.span_context().trace_id
    assert trace.get_active_span() is None


def test_root_option_ignores_the_active_parent():
    _in_memory_provider()
    parent_ctx = trace.set_span_context(ROOT_CONTEXT, SpanContext("a" * 32, "b" * 16, 1))
    span = trace.get_tracer("t").start_span("s", SpanOptions(root=True), parent_ctx)
    assert isinstance(span, SpanImpl)
    assert span.parent_span_context is None
    assert span.span_context().trace_id != "a" * 32


def test_unsampled_parent_gives_a_non_recording_span():
    _, exporter = _in_memory_provider()
    parent_ctx = trace.set_span_context(ROOT_CONTEXT, SpanContext("a" * 32, "b" * 16, 0))
    span = trace.get_tracer("t").start_span("s", None, parent_ctx)
    assert isinstance(span, NonRecordingSpan)
    assert span.span_context().trace_id == "a" * 32
    span.end()
    assert exporter.get_finished_spans() == []


def test_suppressed_tracing_records_nothing():
    _in_memory_provider()
    span = trace.get_tracer("t").start_span("s", None, suppress_tracing(ROOT_CONTEXT))
    assert isinstance(span, NonRecordingSpan)
    assert span.is_recording() is False


def test_noop_tracer_before_registration_propagates_a_valid_parent():
    tracer = trace.get_tracer("t")
    assert isinstance(tracer.start_span("s"), NonRecordingSpan)
    assert tracer.start_span("s").span_context().trace_id == "0" * 32
    parent = SpanContext("a" * 32, "b" * 16, 1)
    ctx = trace.set_span_context(ROOT_CONTEXT, parent)
    assert tracer.start_span("s", None, ctx).span_context() == parent
    assert tracer.start_span("s", SpanOptions(root=True), ctx).span_context().trace_id == "0" * 32
    # A tracer taken before registration delegates once a provider is registered.
    _in_memory_provider()
    assert isinstance(tracer.start_span("s"), SpanImpl)


# --- W3C propagation ------------------------------------------------------------------------

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
SPAN_ID = "b7ad6b7169203331"


def test_parse_trace_parent():
    parsed = parse_trace_parent(f"00-{TRACE_ID}-{SPAN_ID}-01")
    assert parsed is not None
    assert (parsed.trace_id, parsed.span_id, parsed.trace_flags) == (TRACE_ID, SPAN_ID, 1)
    assert parse_trace_parent(f"00-{TRACE_ID.upper()}-{SPAN_ID}-01") is None
    assert parse_trace_parent(f"00-{'0' * 32}-{SPAN_ID}-01") is None
    assert parse_trace_parent(f"00-{TRACE_ID}-{'0' * 16}-01") is None
    assert parse_trace_parent(f"ff-{TRACE_ID}-{SPAN_ID}-01") is None
    assert parse_trace_parent(f"00-{TRACE_ID}-{SPAN_ID}-01-extra") is None
    future = parse_trace_parent(f"01-{TRACE_ID}-{SPAN_ID}-01-extra")
    assert future is not None and future.trace_id == TRACE_ID


def test_w3c_inject_and_extract_round_trip():
    propagator = W3CTraceContextPropagator()
    ctx = propagator.extract(
        ROOT_CONTEXT, {"traceparent": f"00-{TRACE_ID}-{SPAN_ID}-01", "tracestate": "k=v"}
    )
    extracted = trace.get_span_context(ctx)
    assert extracted is not None
    assert extracted.is_remote is True
    assert extracted.trace_state == "k=v"
    carrier: dict[str, Any] = {}
    propagator.inject(ctx, carrier)
    assert carrier == {"traceparent": f"00-{TRACE_ID}-{SPAN_ID}-01", "tracestate": "k=v"}


def test_w3c_extract_rejects_bad_headers_and_inject_skips_invalid():
    propagator = W3CTraceContextPropagator()
    for bad in (f"00-{TRACE_ID.upper()}-{SPAN_ID}-01", "garbage", ""):
        assert propagator.extract(ROOT_CONTEXT, {"traceparent": bad}) is ROOT_CONTEXT
    carrier: dict[str, Any] = {}
    propagator.inject(ROOT_CONTEXT, carrier)
    valid = trace.set_span_context(ROOT_CONTEXT, SpanContext(TRACE_ID, SPAN_ID, 1))
    propagator.inject(suppress_tracing(valid), carrier)
    assert carrier == {}


def test_global_propagation_is_noop_until_registered():
    assert isinstance(propagation._propagator, NoopTextMapPropagator)
    header = {"traceparent": f"00-{TRACE_ID}-{SPAN_ID}-01"}
    assert propagation.extract(ROOT_CONTEXT, header) is ROOT_CONTEXT
    _in_memory_provider()
    ctx = propagation.extract(ROOT_CONTEXT, header)
    with context.use(ctx):
        span = trace.get_tracer("t").start_span("s")
    assert span.span_context().trace_id == TRACE_ID
    carrier: dict[str, Any] = {}
    propagation.inject(trace.set_span(ROOT_CONTEXT, span), carrier)
    assert carrier["traceparent"] == f"00-{TRACE_ID}-{span.span_context().span_id}-01"


# --- BatchSpanProcessor ---------------------------------------------------------------------


class _BatchRecorder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.shut_down = False

    async def export(self, spans):
        self.batches.append([s.name for s in spans])
        return ExportResult(ExportResultCode.SUCCESS)

    async def shutdown(self):
        self.shut_down = True

    async def force_flush(self):
        return None


def _batch_provider(**kwargs: Any) -> tuple[BatchSpanProcessor, _BatchRecorder]:
    recorder = _BatchRecorder()
    processor = BatchSpanProcessor(recorder, **kwargs)
    BasicTracerProvider(span_processors=[processor]).register()
    return processor, recorder


def _end_spans(count: int) -> None:
    tracer = trace.get_tracer("t")
    for i in range(count):
        tracer.start_span(f"s{i}").end()


async def test_batch_waits_for_the_delay_or_force_flush():
    processor, recorder = _batch_provider(scheduled_delay_millis=60_000)
    _end_spans(3)
    await asyncio.sleep(0)
    assert recorder.batches == []
    await processor.force_flush()
    assert recorder.batches == [["s0", "s1", "s2"]]


async def test_batch_exports_on_the_timer():
    _, recorder = _batch_provider(scheduled_delay_millis=10)
    _end_spans(2)
    for _ in range(50):
        if recorder.batches:
            break
        await asyncio.sleep(0.01)
    assert recorder.batches == [["s0", "s1"]]


async def test_full_batch_exports_immediately_and_flush_splits_batches():
    processor, recorder = _batch_provider(max_export_batch_size=2, scheduled_delay_millis=60_000)
    _end_spans(5)
    await processor.force_flush()
    assert sorted(name for batch in recorder.batches for name in batch) == [
        f"s{i}" for i in range(5)
    ]
    assert all(len(batch) <= 2 for batch in recorder.batches)


async def test_queue_limit_drops_spans():
    processor, recorder = _batch_provider(
        max_queue_size=2, max_export_batch_size=10, scheduled_delay_millis=60_000
    )
    assert processor._max_export_batch_size == 2  # clamped to the queue size
    processor._max_export_batch_size = 10
    _end_spans(4)
    await processor.force_flush()
    assert recorder.batches == [["s0", "s1"]]


async def test_shutdown_flushes_then_stops_accepting():
    processor, recorder = _batch_provider(scheduled_delay_millis=60_000)
    _end_spans(1)
    await processor.shutdown()
    assert recorder.batches == [["s0"]]
    assert recorder.shut_down is True
    _end_spans(1)
    await processor.force_flush()
    assert recorder.batches == [["s0"]]


def test_batch_settings_from_env(monkeypatch):
    monkeypatch.setenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "7")
    monkeypatch.setenv("OTEL_BSP_MAX_QUEUE_SIZE", "70")
    monkeypatch.setenv("OTEL_BSP_SCHEDULE_DELAY", "123")
    monkeypatch.setenv("OTEL_BSP_EXPORT_TIMEOUT", "not-a-number")
    processor = BatchSpanProcessor(_BatchRecorder())
    assert processor._max_export_batch_size == 7
    assert processor._max_queue_size == 70
    assert processor._scheduled_delay_millis == 123
    assert processor._export_timeout_millis == 30000
    explicit = BatchSpanProcessor(_BatchRecorder(), max_export_batch_size=3)
    assert explicit._max_export_batch_size == 3


def test_batch_defaults():
    processor = BatchSpanProcessor(_BatchRecorder())
    assert (
        processor._max_export_batch_size,
        processor._max_queue_size,
        processor._scheduled_delay_millis,
        processor._export_timeout_millis,
    ) == (512, 2048, 5000, 30000)


# --- OTLP/HTTP JSON exporter ----------------------------------------------------------------


def _finished_span() -> Any:
    exporter = InMemorySpanExporter()
    provider = BasicTracerProvider(
        resource=resource_from_attributes({"service.name": "promptfoo"}),
        span_processors=[SimpleSpanProcessor(exporter)],
    )
    tracer = provider.get_tracer("promptfoo.providers", "1.0.0")
    parent_ctx = trace.set_span_context(ROOT_CONTEXT, SpanContext(TRACE_ID, SPAN_ID, 1))
    span = tracer.start_span(
        "chat gpt",
        SpanOptions(kind=SpanKind.CLIENT, attributes={"i": 3, "f": 0.5, "whole": 2.0, "b": False}),
        parent_ctx,
    )
    span.add_event("evt", {"x": "y"}, 1500)
    span.set_status(SpanStatusCode.ERROR, "bad")
    span.end()
    [finished] = exporter.get_finished_spans()
    return finished


def test_to_any_value():
    assert to_any_value("s") == {"stringValue": "s"}
    assert to_any_value(True) == {"boolValue": True}
    assert to_any_value(3) == {"intValue": 3}
    assert to_any_value(2.0) == {"intValue": 2}
    assert to_any_value(0.5) == {"doubleValue": 0.5}
    assert to_any_value([1, "a"]) == {
        "arrayValue": {"values": [{"intValue": 1}, {"stringValue": "a"}]}
    }


def test_export_request_shape():
    span = _finished_span()
    request = create_export_trace_service_request([span])
    [resource_spans] = request["resourceSpans"]
    assert resource_spans["resource"] == {
        "attributes": [{"key": "service.name", "value": {"stringValue": "promptfoo"}}],
        "droppedAttributesCount": 0,
    }
    [scope_spans] = resource_spans["scopeSpans"]
    assert scope_spans["scope"] == {"name": "promptfoo.providers", "version": "1.0.0"}
    [otlp] = scope_spans["spans"]
    assert otlp["traceId"] == TRACE_ID
    assert otlp["parentSpanId"] == SPAN_ID
    assert otlp["spanId"] == span.span_context().span_id
    assert otlp["kind"] == 3  # CLIENT (2) + 1
    assert otlp["startTimeUnixNano"] == str(hr_time_to_nanoseconds(span.start_time))
    assert isinstance(otlp["endTimeUnixNano"], str)
    assert otlp["attributes"] == [
        {"key": "i", "value": {"intValue": 3}},
        {"key": "f", "value": {"doubleValue": 0.5}},
        {"key": "whole", "value": {"intValue": 2}},
        {"key": "b", "value": {"boolValue": False}},
    ]
    assert otlp["events"] == [
        {
            "attributes": [{"key": "x", "value": {"stringValue": "y"}}],
            "name": "evt",
            "timeUnixNano": "1500000000",
            "droppedAttributesCount": 0,
        }
    ]
    assert otlp["status"] == {"code": 2, "message": "bad"}
    assert otlp["flags"] == 0x101


def test_parse_key_pairs_and_retry_after():
    assert parse_key_pairs_into_record("a=1, b = x%20y ,bad") == {"a": "1", "b": "x y"}
    assert parse_retry_after_to_millis("2") == 2000
    assert parse_retry_after_to_millis("0") == -1
    assert parse_retry_after_to_millis(None) is None
    assert parse_retry_after_to_millis("Wed, 21 Oct 2015 07:28:00 GMT") == 0


@respx.mock
async def test_otlp_exporter_posts_json_with_env_headers(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer%20abc,x-a=1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "x-a=2")
    route = respx.post("http://collector:4318/v1/traces").mock(return_value=httpx.Response(200))
    exporter = OTLPTraceExporter("http://collector:4318/v1/traces")
    result = await exporter.export([_finished_span()])
    assert result.code == ExportResultCode.SUCCESS
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer abc"
    assert request.headers["x-a"] == "2"
    assert request.headers["content-type"] == "application/json"
    assert request.headers["user-agent"] == otlp_http_exporter.USER_AGENT
    body = json.loads(request.content)
    assert body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["traceId"] == TRACE_ID


@respx.mock
async def test_otlp_exporter_retries_503_with_retry_after(monkeypatch):
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(otlp_http_exporter.asyncio, "sleep", fake_sleep)
    route = respx.post("http://collector:4318/v1/traces").mock(
        side_effect=[httpx.Response(503, headers={"retry-after": "1"}), httpx.Response(200)]
    )
    exporter = OTLPTraceExporter("http://collector:4318/v1/traces")
    result = await exporter.export([_finished_span()])
    assert result.code == ExportResultCode.SUCCESS
    assert route.call_count == 2
    assert sleeps == [1.0]


@respx.mock
async def test_otlp_exporter_fails_without_retry_on_400():
    route = respx.post("http://collector:4318/v1/traces").mock(return_value=httpx.Response(400))
    exporter = OTLPTraceExporter("http://collector:4318/v1/traces")
    result = await exporter.export([_finished_span()])
    assert result.code == ExportResultCode.FAILED
    assert "400" in str(result.error)
    assert route.call_count == 1
    await exporter.shutdown()
    after = await exporter.export([_finished_span()])
    assert after.code == ExportResultCode.FAILED
    assert route.call_count == 1
