"""The parts of `@opentelemetry/sdk-trace` (2.x) that promptfoo's own tracing relies on.

Written in-house instead of depending on the OpenTelemetry Python SDK (see D56). It keeps the JS
SDK's observable behaviour, since the exported spans must match Node's:

- `SpanImpl` follows SpanImpl in Span.js: `set_attribute` drops null values and invalid types,
  `set_status` ignores UNSET, never leaves OK, and keeps a message only for ERROR; operations on
  an ended span are ignored; `record_exception` adds an `exception` event.
- Times are `(seconds, nanoseconds)` HrTime tuples. The start time is `Date.now()` (whole
  milliseconds) and the end time adds the monotonic duration, as the JS SDK does, so stored
  start times are integral and end times fractional in both ports.
- `Tracer.start_span` uses the default ParentBased(AlwaysOn) sampler, random W3C ids and the
  default span limits (128 attributes, events and links; no value-length limit).
- `BatchSpanProcessor` keeps the JS defaults and their OTEL_BSP_* environment overrides
  (512 spans per batch, a 2048-span queue, a 5 s delay, a 30 s export timeout).
  `force_flush` also waits for an export a timer already started, which the JS processor does
  not.
- Exporters return an `ExportResult` (or an awaitable of one) instead of taking a callback.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import secrets
import time
import traceback
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol

from promptphoo.tracing.otel_api import (
    INVALID_SPAN_CONTEXT,
    Context,
    HrTime,
    NonRecordingSpan,
    Span,
    SpanContext,
    SpanKind,
    SpanOptions,
    SpanStatus,
    SpanStatusCode,
    TraceFlags,
    context,
    is_span_context_valid,
    is_tracing_suppressed,
    suppress_tracing,
    trace,
)

logger = logging.getLogger("promptphoo.tracing.sdk")

NANOS_PER_SECOND = 1_000_000_000
EXCEPTION_EVENT_NAME = "exception"
ATTR_EXCEPTION_TYPE = "exception.type"
ATTR_EXCEPTION_MESSAGE = "exception.message"
ATTR_EXCEPTION_STACKTRACE = "exception.stacktrace"


# --- time -----------------------------------------------------------------------------------


def millis_to_hr_time(epoch_millis: float) -> HrTime:
    seconds = math.trunc(epoch_millis / 1000)
    nanos = round((math.fmod(epoch_millis, 1000)) * 1_000_000)
    return (seconds, nanos)


def add_hr_times(time1: HrTime, time2: HrTime) -> HrTime:
    seconds, nanos = time1[0] + time2[0], time1[1] + time2[1]
    if nanos >= NANOS_PER_SECOND:
        nanos -= NANOS_PER_SECOND
        seconds += 1
    return (seconds, nanos)


def hr_time_duration(start: HrTime, end: HrTime) -> HrTime:
    seconds = end[0] - start[0]
    nanos = end[1] - start[1]
    if nanos < 0:
        seconds -= 1
        nanos += NANOS_PER_SECOND
    return (seconds, nanos)


def hr_time_to_nanoseconds(hr_time: HrTime) -> int:
    return int(hr_time[0]) * NANOS_PER_SECOND + int(hr_time[1])


def _date_now() -> int:
    return time.time_ns() // 1_000_000


def _is_hr_time(value: Any) -> bool:
    return (
        isinstance(value, tuple | list)
        and len(value) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    )


# --- attributes -----------------------------------------------------------------------------


def _primitive_kind(value: Any) -> str | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    return None


def is_attribute_value(value: Any) -> bool:
    """@opentelemetry/core isAttributeValue: a primitive, or a homogeneous primitive array."""
    if value is None:
        return True
    if isinstance(value, list | tuple):
        kind: str | None = None
        for element in value:
            if element is None:
                continue
            element_kind = _primitive_kind(element)
            if element_kind is None:
                return False
            if kind is None:
                kind = element_kind
            elif kind != element_kind:
                return False
        return True
    return _primitive_kind(value) is not None


def sanitize_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(attributes, Mapping):
        return out
    for key, value in attributes.items():
        if not isinstance(key, str) or not key:
            logger.debug("Invalid attribute key: %s", key)
            continue
        if not is_attribute_value(value):
            logger.debug("Invalid attribute value set for key: %s", key)
            continue
        out[key] = list(value) if isinstance(value, list | tuple) else value
    return out


# --- resource, scope, limits ----------------------------------------------------------------


@dataclass(frozen=True)
class Resource:
    attributes: dict[str, Any] = field(default_factory=dict)
    schema_url: str | None = None


def resource_from_attributes(attributes: Mapping[str, Any]) -> Resource:
    return Resource(sanitize_attributes(attributes))


EMPTY_RESOURCE = Resource()


@dataclass(frozen=True)
class InstrumentationScope:
    name: str
    version: str | None = None
    schema_url: str | None = None


@dataclass(frozen=True)
class SpanLimits:
    attribute_count_limit: int = 128
    attribute_value_length_limit: float = math.inf
    event_count_limit: int = 128
    link_count_limit: int = 128
    attribute_per_event_count_limit: int = 128
    attribute_per_link_count_limit: int = 128


@dataclass
class TimedEvent:
    name: str
    attributes: dict[str, Any]
    time: HrTime
    dropped_attributes_count: int = 0


# --- spans ----------------------------------------------------------------------------------


class SpanProcessor(Protocol):
    def on_start(self, span: SpanImpl, parent_context: Context) -> None: ...
    def on_end(self, span: SpanImpl) -> None: ...
    async def force_flush(self) -> None: ...
    async def shutdown(self) -> None: ...


class SpanImpl:
    """A recording span; also the ReadableSpan that processors and exporters receive."""

    def __init__(
        self,
        *,
        resource: Resource,
        scope: InstrumentationScope,
        parent_context: Context,
        span_context: SpanContext,
        name: str,
        kind: SpanKind,
        parent_span_context: SpanContext | None,
        attributes: Mapping[str, Any] | None,
        start_time: Any,
        span_processor: SpanProcessor,
        span_limits: SpanLimits,
    ) -> None:
        now = _date_now()
        self._span_context = span_context
        self._performance_start = time.perf_counter()
        self._start_time_provided = start_time is not None
        self._span_limits = span_limits
        self._span_processor = span_processor
        self.name = name
        self.parent_span_context = parent_span_context
        self.kind = kind
        self.attributes: dict[str, Any] = {}
        self.links: list[Any] = []
        self.events: list[TimedEvent] = []
        self.status = SpanStatus(SpanStatusCode.UNSET)
        self.end_time: HrTime = (0, 0)
        self.duration: HrTime = (-1, -1)
        self.dropped_attributes_count = 0
        self.dropped_events_count = 0
        self.dropped_links_count = 0
        self._ended = False
        self.start_time: HrTime = (0, 0)
        self.start_time = self._get_time(start_time if start_time is not None else now)
        self.resource = resource
        self.instrumentation_scope = scope
        if attributes is not None:
            self.set_attributes(attributes)
        self._span_processor.on_start(self, parent_context)

    def span_context(self) -> SpanContext:
        return self._span_context

    @property
    def ended(self) -> bool:
        return self._ended

    def _is_span_ended(self) -> bool:
        if self._ended:
            logger.debug(
                "Cannot execute the operation on ended Span {traceId: %s, spanId: %s}",
                self._span_context.trace_id,
                self._span_context.span_id,
            )
        return self._ended

    def _truncate_to_size(self, value: Any) -> Any:
        limit = self._span_limits.attribute_value_length_limit
        if limit == math.inf or limit <= 0:
            return value
        size = int(limit)
        if isinstance(value, str):
            return value[:size]
        if isinstance(value, list):
            return [v[:size] if isinstance(v, str) else v for v in value]
        return value

    def set_attribute(self, key: str, value: Any) -> SpanImpl:
        if value is None or self._is_span_ended():
            return self
        if not key:
            logger.debug("Invalid attribute key: %s", key)
            return self
        if not is_attribute_value(value):
            logger.debug("Invalid attribute value set for key: %s", key)
            return self
        is_new_key = key not in self.attributes
        if is_new_key and len(self.attributes) >= self._span_limits.attribute_count_limit:
            self.dropped_attributes_count += 1
            return self
        if isinstance(value, tuple):
            value = list(value)
        self.attributes[key] = self._truncate_to_size(value)
        return self

    def set_attributes(self, attributes: Mapping[str, Any]) -> SpanImpl:
        for key, value in attributes.items():
            self.set_attribute(key, value)
        return self

    def add_event(
        self, name: str, attributes: Mapping[str, Any] | None = None, time: Any = None
    ) -> SpanImpl:
        if self._is_span_ended():
            return self
        limit = self._span_limits.event_count_limit
        if limit == 0:
            self.dropped_events_count += 1
            return self
        if len(self.events) >= limit:
            self.events.pop(0)
            self.dropped_events_count += 1
        sanitized = sanitize_attributes(attributes)
        kept: dict[str, Any] = {}
        dropped = 0
        for key, value in sanitized.items():
            if len(kept) >= self._span_limits.attribute_per_event_count_limit:
                dropped += 1
                continue
            kept[key] = self._truncate_to_size(value)
        self.events.append(TimedEvent(name, kept, self._get_time(time), dropped))
        return self

    def set_status(self, code: SpanStatusCode, message: Any = None) -> SpanImpl:
        if self._is_span_ended():
            return self
        if code == SpanStatusCode.UNSET:
            return self
        if self.status.code == SpanStatusCode.OK:
            return self
        status_message = None
        if code == SpanStatusCode.ERROR:
            if isinstance(message, str):
                status_message = message
            elif message is not None:
                logger.debug(
                    "Dropping invalid status.message of type '%s', expected 'string'",
                    type(message).__name__,
                )
        self.status = SpanStatus(SpanStatusCode(code), status_message)
        return self

    def update_name(self, name: str) -> SpanImpl:
        if not self._is_span_ended():
            self.name = name
        return self

    def end(self, end_time: Any = None) -> None:
        if self._is_span_ended():
            logger.debug(
                "%s %s-%s - You can only call end() on a span once.",
                self.name,
                self._span_context.trace_id,
                self._span_context.span_id,
            )
            return
        self.end_time = self._get_time(end_time)
        self.duration = hr_time_duration(self.start_time, self.end_time)
        if self.duration[0] < 0:
            self.end_time = self.start_time
            self.duration = (0, 0)
        self._ended = True
        self._span_processor.on_end(self)

    def _get_time(self, value: Any) -> HrTime:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return millis_to_hr_time(value)
        if isinstance(value, tuple | list) and _is_hr_time(value):
            return (int(value[0]), int(value[1]))
        if self._start_time_provided:
            return millis_to_hr_time(_date_now())
        elapsed_ms = (time.perf_counter() - self._performance_start) * 1000
        return add_hr_times(self.start_time, millis_to_hr_time(elapsed_ms))

    def is_recording(self) -> bool:
        return not self._ended

    def record_exception(self, exception: Any, time: Any = None) -> None:
        attributes: dict[str, Any] = {}
        if isinstance(exception, str):
            attributes[ATTR_EXCEPTION_MESSAGE] = exception
        elif exception:
            # Node system errors carry a string `code` (ECONNREFUSED); Python's `code`
            # attributes (HTTP status, exit status) are not error types.
            code = getattr(exception, "code", None)
            name = getattr(exception, "name", None)
            if isinstance(code, str) and code:
                attributes[ATTR_EXCEPTION_TYPE] = code
            elif isinstance(exception, BaseException):
                attributes[ATTR_EXCEPTION_TYPE] = type(exception).__name__
            elif isinstance(name, str) and name:
                attributes[ATTR_EXCEPTION_TYPE] = name
            message = str(exception) if isinstance(exception, BaseException) else None
            message = getattr(exception, "message", message)
            if message:
                attributes[ATTR_EXCEPTION_MESSAGE] = message
            stack = getattr(exception, "stack", None)
            if stack is None and isinstance(exception, BaseException):
                stack = "".join(traceback.format_exception(exception)).rstrip("\n")
            if stack:
                attributes[ATTR_EXCEPTION_STACKTRACE] = stack
        if attributes.get(ATTR_EXCEPTION_TYPE) or attributes.get(ATTR_EXCEPTION_MESSAGE):
            self.add_event(EXCEPTION_EVENT_NAME, attributes, time)
        else:
            logger.debug("Failed to record an exception %s", exception)


ReadableSpan = SpanImpl


# --- export ---------------------------------------------------------------------------------


class ExportResultCode(IntEnum):
    SUCCESS = 0
    FAILED = 1


@dataclass
class ExportResult:
    code: ExportResultCode
    error: BaseException | None = None


class SpanExporter(Protocol):
    def export(self, spans: Sequence[ReadableSpan]) -> ExportResult | Awaitable[ExportResult]: ...
    async def shutdown(self) -> None: ...
    async def force_flush(self) -> None: ...


async def _resolve_export(result: ExportResult | Awaitable[ExportResult]) -> ExportResult:
    if inspect.isawaitable(result):
        return await result
    return result


class InMemorySpanExporter:
    """Collects finished spans in memory (tests)."""

    def __init__(self) -> None:
        self._finished_spans: list[ReadableSpan] = []
        self._stopped = False

    def export(self, spans: Sequence[ReadableSpan]) -> ExportResult:
        if self._stopped:
            return ExportResult(ExportResultCode.FAILED, RuntimeError("Exporter has been stopped"))
        self._finished_spans.extend(spans)
        return ExportResult(ExportResultCode.SUCCESS)

    async def shutdown(self) -> None:
        self._stopped = True
        self._finished_spans = []

    async def force_flush(self) -> None:
        return None

    def reset(self) -> None:
        self._finished_spans = []

    def get_finished_spans(self) -> list[ReadableSpan]:
        return list(self._finished_spans)


def _sampled(span: ReadableSpan) -> bool:
    return (span.span_context().trace_flags & TraceFlags.SAMPLED) != 0


def _spawn(tasks: set[asyncio.Task[Any]], coro: Awaitable[Any]) -> asyncio.Task[Any] | None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if inspect.iscoroutine(coro):
            coro.close()
        return None
    task = loop.create_task(coro)  # type: ignore[arg-type]
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return task


class SimpleSpanProcessor:
    """Exports each sampled span as soon as it ends."""

    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter
        self._pending: set[asyncio.Task[Any]] = set()
        self._shutdown = False

    def on_start(self, span: SpanImpl, parent_context: Context) -> None:
        return None

    def on_end(self, span: SpanImpl) -> None:
        if self._shutdown or not _sampled(span):
            return
        with context.use(suppress_tracing(context.active())):
            result = self._exporter.export([span])
        if inspect.isawaitable(result):
            _spawn(self._pending, self._finish_export(result))
        elif result.code != ExportResultCode.SUCCESS:
            logger.error("SimpleSpanProcessor: span export failed: %s", result.error)

    async def _finish_export(self, result: Awaitable[ExportResult]) -> None:
        outcome = await result
        if outcome.code != ExportResultCode.SUCCESS:
            logger.error("SimpleSpanProcessor: span export failed: %s", outcome.error)

    async def force_flush(self) -> None:
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)
        await self._exporter.force_flush()

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        await self.force_flush()
        self._shutdown = True
        await self._exporter.shutdown()


def _number_from_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.debug("Unknown value %r for %s, expected a number, using defaults", raw, name)
        return None
    return int(value) if math.isfinite(value) else None


class BatchSpanProcessor:
    """BatchSpanProcessorBase with the sdk-trace-base env fallbacks."""

    def __init__(
        self,
        exporter: SpanExporter,
        *,
        max_export_batch_size: int | None = None,
        max_queue_size: int | None = None,
        scheduled_delay_millis: int | None = None,
        export_timeout_millis: int | None = None,
    ) -> None:
        def pick(value: int | None, env: str, default: int) -> int:
            if value is not None:
                return value
            from_env = _number_from_env(env)
            return from_env if from_env is not None else default

        self._exporter = exporter
        self._max_export_batch_size = pick(
            max_export_batch_size, "OTEL_BSP_MAX_EXPORT_BATCH_SIZE", 512
        )
        self._max_queue_size = pick(max_queue_size, "OTEL_BSP_MAX_QUEUE_SIZE", 2048)
        self._scheduled_delay_millis = pick(scheduled_delay_millis, "OTEL_BSP_SCHEDULE_DELAY", 5000)
        self._export_timeout_millis = pick(export_timeout_millis, "OTEL_BSP_EXPORT_TIMEOUT", 30000)
        if self._max_export_batch_size > self._max_queue_size:
            logger.debug(
                "BatchSpanProcessor: maxExportBatchSize must be smaller or equal to "
                "maxQueueSize, setting maxExportBatchSize to match maxQueueSize"
            )
            self._max_export_batch_size = self._max_queue_size
        self._finished_spans: list[ReadableSpan] = []
        self._is_exporting = False
        self._timer: asyncio.TimerHandle | None = None
        self._dropped_spans_count = 0
        self._background: set[asyncio.Task[Any]] = set()
        self._shutdown_task: asyncio.Future[None] | None = None

    def on_start(self, span: SpanImpl, parent_context: Context) -> None:
        return None

    def on_end(self, span: SpanImpl) -> None:
        if self._shutdown_task is not None or not _sampled(span):
            return
        self._add_to_buffer(span)

    async def force_flush(self) -> None:
        if self._shutdown_task is not None:
            await self._shutdown_task
            return
        await self._flush_all()

    async def shutdown(self) -> None:
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.ensure_future(self._do_shutdown())
        await self._shutdown_task

    async def _do_shutdown(self) -> None:
        await self._flush_all()
        await self._exporter.shutdown()

    def _add_to_buffer(self, span: ReadableSpan) -> None:
        if len(self._finished_spans) >= self._max_queue_size:
            if self._dropped_spans_count == 0:
                logger.debug("maxQueueSize reached, dropping spans")
            self._dropped_spans_count += 1
            return
        if self._dropped_spans_count > 0:
            logger.warning(
                "Dropped %d spans because maxQueueSize reached", self._dropped_spans_count
            )
            self._dropped_spans_count = 0
        self._finished_spans.append(span)
        self._maybe_start_timer()

    async def _flush_all(self) -> None:
        in_flight = [task for task in self._background if not task.done()]
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)
        count = math.ceil(len(self._finished_spans) / self._max_export_batch_size)
        await asyncio.gather(*(self._flush_one_batch() for _ in range(count)))

    async def _flush_one_batch(self) -> None:
        self._clear_timer()
        if not self._finished_spans:
            return
        if len(self._finished_spans) <= self._max_export_batch_size:
            spans, self._finished_spans = self._finished_spans, []
        else:
            spans = self._finished_spans[: self._max_export_batch_size]
            del self._finished_spans[: self._max_export_batch_size]
        with context.use(suppress_tracing(context.active())):
            pending = self._exporter.export(spans)
        try:
            result = await asyncio.wait_for(
                _resolve_export(pending), self._export_timeout_millis / 1000
            )
        except TimeoutError as e:
            raise TimeoutError("Timeout") from e
        if result.code != ExportResultCode.SUCCESS:
            raise result.error or RuntimeError("BatchSpanProcessor: span export failed")

    def _maybe_start_timer(self) -> None:
        if self._is_exporting:
            return

        async def flush() -> None:
            self._is_exporting = True
            try:
                await self._flush_one_batch()
            except Exception as e:
                logger.error("BatchSpanProcessor export failed: %s", e)
            finally:
                self._is_exporting = False
                if self._finished_spans:
                    self._clear_timer()
                    self._maybe_start_timer()

        if len(self._finished_spans) >= self._max_export_batch_size:
            _spawn(self._background, flush())
            return
        if self._timer is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # No loop: the spans wait for force_flush.

        def on_timer() -> None:
            self._timer = None
            _spawn(self._background, flush())

        self._timer = loop.call_later(self._scheduled_delay_millis / 1000, on_timer)

    def _clear_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


class MultiSpanProcessor:
    def __init__(self, processors: Sequence[SpanProcessor]) -> None:
        self._processors = list(processors)

    def on_start(self, span: SpanImpl, parent_context: Context) -> None:
        for processor in self._processors:
            processor.on_start(span, parent_context)

    def on_end(self, span: SpanImpl) -> None:
        for processor in self._processors:
            processor.on_end(span)

    async def force_flush(self) -> None:
        results = await asyncio.gather(
            *(p.force_flush() for p in self._processors), return_exceptions=True
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise errors[0]

    async def shutdown(self) -> None:
        results = await asyncio.gather(
            *(p.shutdown() for p in self._processors), return_exceptions=True
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise errors[0]


# --- tracer ---------------------------------------------------------------------------------


def generate_trace_id() -> str:
    while True:
        value = secrets.token_hex(16)
        if value != "0" * 32:
            return value


def generate_span_id() -> str:
    while True:
        value = secrets.token_hex(8)
        if value != "0" * 16:
            return value


def _should_sample(parent: SpanContext | None) -> bool:
    """ParentBasedSampler with the AlwaysOn root and the default parent-based delegates."""
    if parent is None or not is_span_context_valid(parent):
        return True
    return (parent.trace_flags & TraceFlags.SAMPLED) != 0


class SdkTracer:
    def __init__(
        self,
        scope: InstrumentationScope,
        *,
        resource: Resource,
        span_limits: SpanLimits,
        span_processor: SpanProcessor,
    ) -> None:
        self.instrumentation_scope = scope
        self._resource = resource
        self._span_limits = span_limits
        self._span_processor = span_processor

    def start_span(
        self, name: str, options: SpanOptions | None = None, ctx: Context | None = None
    ) -> Span:
        options = options or SpanOptions()
        ctx = ctx if ctx is not None else context.active()
        if options.root:
            ctx = trace.delete_span(ctx)
        parent_span = trace.get_span(ctx)
        if is_tracing_suppressed(ctx):
            return NonRecordingSpan(INVALID_SPAN_CONTEXT)
        parent_span_context = parent_span.span_context() if parent_span is not None else None
        span_id = generate_span_id()
        valid_parent: SpanContext | None = None
        trace_state = None
        if parent_span_context is None or not is_span_context_valid(parent_span_context):
            trace_id = generate_trace_id()
        else:
            trace_id = parent_span_context.trace_id
            trace_state = parent_span_context.trace_state
            valid_parent = parent_span_context
        attributes = sanitize_attributes(options.attributes)
        sampled = _should_sample(parent_span_context)
        span_context = SpanContext(
            trace_id,
            span_id,
            TraceFlags.SAMPLED if sampled else TraceFlags.NONE,
            trace_state=trace_state,
        )
        if not sampled:
            return NonRecordingSpan(span_context)
        return SpanImpl(
            resource=self._resource,
            scope=self.instrumentation_scope,
            parent_context=ctx,
            span_context=span_context,
            name=name,
            kind=options.kind,
            parent_span_context=valid_parent,
            attributes=attributes,
            start_time=options.start_time,
            span_processor=self._span_processor,
            span_limits=self._span_limits,
        )


class BasicTracerProvider:
    def __init__(
        self,
        *,
        resource: Resource | None = None,
        span_processors: Sequence[SpanProcessor] = (),
        span_limits: SpanLimits | None = None,
    ) -> None:
        self.resource = resource or EMPTY_RESOURCE
        self._span_limits = span_limits or SpanLimits()
        self._active_span_processor = MultiSpanProcessor(span_processors)
        self._tracers: dict[str, SdkTracer] = {}

    def get_tracer(self, name: str, version: str | None = None) -> SdkTracer:
        key = f"{name}@{version or ''}:"
        if key not in self._tracers:
            self._tracers[key] = SdkTracer(
                InstrumentationScope(name, version),
                resource=self.resource,
                span_limits=self._span_limits,
                span_processor=self._active_span_processor,
            )
        return self._tracers[key]

    def register(self) -> None:
        """NodeTracerProvider.register(): become the global provider, with W3C propagation."""
        from promptphoo.tracing.otel_api import W3CTraceContextPropagator, propagation

        trace.set_global_tracer_provider(self)
        propagation.set_global_propagator(W3CTraceContextPropagator())

    async def force_flush(self) -> None:
        await self._active_span_processor.force_flush()

    async def shutdown(self) -> None:
        await self._active_span_processor.shutdown()


NodeTracerProvider = BasicTracerProvider
