"""The subset of `@opentelemetry/api` that promptfoo's own spans use, written in-house.

The OpenTelemetry Python SDK is not a dependency (see D56), so this module stands in for the
JS API package: span contexts with W3C ids, an immutable `Context` held in a ContextVar (Node's
AsyncLocalStorage context manager), the global tracer provider and propagator, and the no-op
tracer the API falls back to before the SDK registers a provider.

Names follow the JS API, snake_cased: `context.active()`, `trace.get_tracer()`,
`trace.set_span()`, `propagation.extract()`. Async callbacks run inside a context through
`context.with_(ctx, fn)`, which awaits `fn()` with the ContextVar set, so the context stays
active across the callback's awaits and is inherited by tasks it spawns, as in Node.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol, TypeVar

T = TypeVar("T")

AttributeValue = str | int | float | bool | list[str] | list[int] | list[float] | list[bool]
Attributes = dict[str, Any]
# [seconds, nanoseconds], as OpenTelemetry JS's HrTime.
HrTime = tuple[int, int]


class SpanKind(IntEnum):
    INTERNAL = 0
    SERVER = 1
    CLIENT = 2
    PRODUCER = 3
    CONSUMER = 4


class SpanStatusCode(IntEnum):
    UNSET = 0
    OK = 1
    ERROR = 2


class TraceFlags:
    NONE = 0
    SAMPLED = 1


INVALID_TRACEID = "0" * 32
INVALID_SPANID = "0" * 16

_HEX_RE = re.compile(r"^[0-9a-f]+$")


def is_valid_trace_id(trace_id: str) -> bool:
    return (
        isinstance(trace_id, str)
        and len(trace_id) == 32
        and bool(_HEX_RE.match(trace_id))
        and trace_id != INVALID_TRACEID
    )


def is_valid_span_id(span_id: str) -> bool:
    return (
        isinstance(span_id, str)
        and len(span_id) == 16
        and bool(_HEX_RE.match(span_id))
        and span_id != INVALID_SPANID
    )


@dataclass(frozen=True)
class SpanContext:
    trace_id: str
    span_id: str
    trace_flags: int = TraceFlags.NONE
    is_remote: bool = False
    trace_state: str | None = None


INVALID_SPAN_CONTEXT = SpanContext(INVALID_TRACEID, INVALID_SPANID, TraceFlags.NONE)


def is_span_context_valid(span_context: SpanContext | None) -> bool:
    return (
        span_context is not None
        and is_valid_trace_id(span_context.trace_id)
        and is_valid_span_id(span_context.span_id)
    )


@dataclass(frozen=True)
class SpanStatus:
    code: SpanStatusCode = SpanStatusCode.UNSET
    message: str | None = None


class Span(Protocol):
    def span_context(self) -> SpanContext: ...
    def set_attribute(self, key: str, value: Any) -> Span: ...
    def set_attributes(self, attributes: Mapping[str, Any]) -> Span: ...
    def add_event(
        self, name: str, attributes: Mapping[str, Any] | None = None, time: Any = None
    ) -> Span: ...
    def set_status(self, code: SpanStatusCode, message: Any = None) -> Span: ...
    def update_name(self, name: str) -> Span: ...
    def end(self, end_time: Any = None) -> None: ...
    def is_recording(self) -> bool: ...
    def record_exception(self, exception: Any, time: Any = None) -> None: ...


class NonRecordingSpan:
    """The API's NonRecordingSpan: carries a span context and records nothing."""

    def __init__(self, span_context: SpanContext = INVALID_SPAN_CONTEXT) -> None:
        self._span_context = span_context

    def span_context(self) -> SpanContext:
        return self._span_context

    def set_attribute(self, key: str, value: Any) -> NonRecordingSpan:
        return self

    def set_attributes(self, attributes: Mapping[str, Any]) -> NonRecordingSpan:
        return self

    def add_event(
        self, name: str, attributes: Mapping[str, Any] | None = None, time: Any = None
    ) -> NonRecordingSpan:
        return self

    def set_status(self, code: SpanStatusCode, message: Any = None) -> NonRecordingSpan:
        return self

    def update_name(self, name: str) -> NonRecordingSpan:
        return self

    def end(self, end_time: Any = None) -> None:
        return None

    def is_recording(self) -> bool:
        return False

    def record_exception(self, exception: Any, time: Any = None) -> None:
        return None


# --- context --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextKey:
    description: str
    _id: object = field(default_factory=object, compare=True)


def create_context_key(description: str) -> ContextKey:
    return ContextKey(description)


class Context:
    """An immutable key/value map, as BaseContext in the JS API."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[ContextKey, Any] | None = None) -> None:
        self._values: dict[ContextKey, Any] = dict(values or {})

    def get_value(self, key: ContextKey) -> Any:
        return self._values.get(key)

    def set_value(self, key: ContextKey, value: Any) -> Context:
        return Context({**self._values, key: value})

    def delete_value(self, key: ContextKey) -> Context:
        values = dict(self._values)
        values.pop(key, None)
        return Context(values)


ROOT_CONTEXT = Context()

_active_context: ContextVar[Context] = ContextVar("promptphoo_otel_context", default=ROOT_CONTEXT)


class _ContextAPI:
    """`context` from the JS API, backed by a ContextVar."""

    def active(self) -> Context:
        return _active_context.get()

    @contextmanager
    def use(self, ctx: Context) -> Iterator[Context]:
        """Make `ctx` active for a synchronous block."""
        token = _active_context.set(ctx)
        try:
            yield ctx
        finally:
            _active_context.reset(token)

    async def with_(self, ctx: Context, fn: Callable[..., Any], *args: Any) -> Any:
        """`context.with(ctx, fn, undefined, ...args)`: run `fn` (sync or async) inside `ctx`."""
        token = _active_context.set(ctx)
        try:
            result = fn(*args)
            if inspect.isawaitable(result):
                result = await result
            return result
        finally:
            _active_context.reset(token)


context = _ContextAPI()


# --- trace ----------------------------------------------------------------------------------

_SPAN_KEY = create_context_key("OpenTelemetry Context Key SPAN")
_SUPPRESS_TRACING_KEY = create_context_key("OpenTelemetry SDK Context Key SUPPRESS_TRACING")


def suppress_tracing(ctx: Context) -> Context:
    return ctx.set_value(_SUPPRESS_TRACING_KEY, True)


def is_tracing_suppressed(ctx: Context) -> bool:
    return ctx.get_value(_SUPPRESS_TRACING_KEY) is True


@dataclass
class SpanOptions:
    kind: SpanKind = SpanKind.INTERNAL
    attributes: Attributes | None = None
    start_time: Any = None
    root: bool = False


class Tracer(Protocol):
    def start_span(
        self, name: str, options: SpanOptions | None = None, ctx: Context | None = None
    ) -> Span: ...


class TracerProvider(Protocol):
    def get_tracer(self, name: str, version: str | None = None) -> Tracer: ...


class NoopTracer:
    """The API's NoopTracer: propagates a valid parent context, records nothing."""

    def start_span(
        self, name: str, options: SpanOptions | None = None, ctx: Context | None = None
    ) -> Span:
        ctx = ctx if ctx is not None else context.active()
        if options is not None and options.root:
            return NonRecordingSpan()
        parent = trace.get_span_context(ctx)
        if parent is not None and is_span_context_valid(parent):
            return NonRecordingSpan(parent)
        return NonRecordingSpan()


class ProxyTracer:
    """Resolves the global provider on every call, so tracers taken before registration work."""

    def __init__(self, name: str, version: str | None) -> None:
        self.name = name
        self.version = version

    def _delegate(self) -> Tracer:
        provider = trace.get_tracer_provider()
        return provider.get_tracer(self.name, self.version) if provider else NoopTracer()

    def start_span(
        self, name: str, options: SpanOptions | None = None, ctx: Context | None = None
    ) -> Span:
        return self._delegate().start_span(name, options, ctx)

    async def start_active_span(
        self,
        name: str,
        fn: Callable[[Span], Awaitable[T]],
        options: SpanOptions | None = None,
        ctx: Context | None = None,
    ) -> T:
        """`startActiveSpan(name, options, context, fn)`: start a span, make it active while
        `fn(span)` runs. Like the JS API, it does not end the span."""
        parent = ctx if ctx is not None else context.active()
        span = self.start_span(name, options, parent)
        return await context.with_(trace.set_span(parent, span), fn, span)


class _TraceAPI:
    def __init__(self) -> None:
        self._provider: TracerProvider | None = None

    def set_global_tracer_provider(self, provider: TracerProvider) -> bool:
        self._provider = provider
        return True

    def get_tracer_provider(self) -> TracerProvider | None:
        return self._provider

    def disable(self) -> None:
        self._provider = None

    def get_tracer(self, name: str, version: str | None = None) -> ProxyTracer:
        return ProxyTracer(name, version)

    def get_span(self, ctx: Context) -> Span | None:
        return ctx.get_value(_SPAN_KEY)

    def get_active_span(self) -> Span | None:
        return self.get_span(context.active())

    def set_span(self, ctx: Context, span: Span) -> Context:
        return ctx.set_value(_SPAN_KEY, span)

    def delete_span(self, ctx: Context) -> Context:
        return ctx.delete_value(_SPAN_KEY)

    def set_span_context(self, ctx: Context, span_context: SpanContext) -> Context:
        return self.set_span(ctx, NonRecordingSpan(span_context))

    def get_span_context(self, ctx: Context) -> SpanContext | None:
        span = self.get_span(ctx)
        return span.span_context() if span is not None else None

    def wrap_span_context(self, span_context: SpanContext) -> NonRecordingSpan:
        return NonRecordingSpan(span_context)

    is_span_context_valid = staticmethod(is_span_context_valid)


trace = _TraceAPI()


# --- propagation ----------------------------------------------------------------------------

TRACE_PARENT_HEADER = "traceparent"
TRACE_STATE_HEADER = "tracestate"
_TRACE_PARENT_RE = re.compile(
    r"^\s?((?!ff)[\da-f]{2})-((?![0]{32})[\da-f]{32})-((?![0]{16})[\da-f]{16})-([\da-f]{2})(-.*)?\s?$"
)


def parse_trace_parent(trace_parent: str) -> SpanContext | None:
    """W3CTraceContextPropagator's parseTraceParent."""
    match = _TRACE_PARENT_RE.match(trace_parent)
    if not match:
        return None
    if match.group(1) == "00" and match.group(5):
        return None
    return SpanContext(match.group(2), match.group(3), int(match.group(4), 16))


class TextMapPropagator(Protocol):
    def inject(self, ctx: Context, carrier: dict[str, Any]) -> None: ...
    def extract(self, ctx: Context, carrier: Mapping[str, Any]) -> Context: ...


class NoopTextMapPropagator:
    def inject(self, ctx: Context, carrier: dict[str, Any]) -> None:
        return None

    def extract(self, ctx: Context, carrier: Mapping[str, Any]) -> Context:
        return ctx


class W3CTraceContextPropagator:
    """@opentelemetry/core's W3CTraceContextPropagator."""

    def inject(self, ctx: Context, carrier: dict[str, Any]) -> None:
        span_context = trace.get_span_context(ctx)
        if (
            span_context is None
            or is_tracing_suppressed(ctx)
            or not is_span_context_valid(span_context)
        ):
            return
        flags = format(int(span_context.trace_flags or TraceFlags.NONE), "x")
        carrier[TRACE_PARENT_HEADER] = f"00-{span_context.trace_id}-{span_context.span_id}-0{flags}"
        if span_context.trace_state:
            carrier[TRACE_STATE_HEADER] = span_context.trace_state

    def extract(self, ctx: Context, carrier: Mapping[str, Any]) -> Context:
        header = carrier.get(TRACE_PARENT_HEADER)
        if not header:
            return ctx
        trace_parent = header[0] if isinstance(header, list) else header
        if not isinstance(trace_parent, str):
            return ctx
        span_context = parse_trace_parent(trace_parent)
        if span_context is None:
            return ctx
        state_header = carrier.get(TRACE_STATE_HEADER)
        state = None
        if state_header:
            state = ",".join(state_header) if isinstance(state_header, list) else state_header
        span_context = SpanContext(
            span_context.trace_id,
            span_context.span_id,
            span_context.trace_flags,
            is_remote=True,
            trace_state=state if isinstance(state, str) else None,
        )
        return trace.set_span_context(ctx, span_context)


class _PropagationAPI:
    def __init__(self) -> None:
        self._propagator: TextMapPropagator = NoopTextMapPropagator()

    def set_global_propagator(self, propagator: TextMapPropagator) -> bool:
        self._propagator = propagator
        return True

    def disable(self) -> None:
        self._propagator = NoopTextMapPropagator()

    def inject(self, ctx: Context, carrier: dict[str, Any]) -> None:
        self._propagator.inject(ctx, carrier)

    def extract(self, ctx: Context, carrier: Mapping[str, Any]) -> Context:
        return self._propagator.extract(ctx, carrier)


propagation = _PropagationAPI()
