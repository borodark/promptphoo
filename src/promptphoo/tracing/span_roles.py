"""Ports src/tracing/spanRoles.ts.

Upstream keeps the span role in the OpenTelemetry context. Here it is a ContextVar, which
asyncio tasks copy the same way. promptphoo records no OpenTelemetry spans in this module, so
`get_active_traceparent` returns None unless a span recorder publishes the active span through
`ACTIVE_SPAN_CONTEXT` (see D54).
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from typing import Any, Literal, TypeVar

T = TypeVar("T")

SPAN_ROLE_ATTRIBUTE = "promptfoo.span.role"

PromptfooSpanRole = Literal["test_case", "target", "grader"]

_SPAN_ROLE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "promptfoo.span.role", default=None
)
# {traceId, spanId, traceFlags} of the active recorded span, when one exists.
ACTIVE_SPAN_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "promptfoo.active_span_context", default=None
)

_INVALID_TRACE_ID = "0" * 32
_INVALID_SPAN_ID = "0" * 16


def get_active_span_role() -> str | None:
    """Keep evaluator-owned grading activity distinct from the target's behavior."""
    return _SPAN_ROLE.get()


def with_span_role(role: str, fn: Callable[[], T]) -> T:
    """Run `fn` with `role` active. For a coroutine function, the coroutine is created (not
    awaited) inside the role; wrap it in a task to keep the role across awaits."""
    token = _SPAN_ROLE.set(role)
    try:
        return fn()
    finally:
        _SPAN_ROLE.reset(token)


def _is_span_context_valid(span_context: dict[str, Any]) -> bool:
    trace_id = span_context.get("traceId")
    span_id = span_context.get("spanId")
    return (
        isinstance(trace_id, str)
        and len(trace_id) == 32
        and trace_id != _INVALID_TRACE_ID
        and isinstance(span_id, str)
        and len(span_id) == 16
        and span_id != _INVALID_SPAN_ID
    )


def get_active_traceparent() -> str | None:
    """Return a usable traceparent only when the active span has a valid recorded context."""
    span_context = ACTIVE_SPAN_CONTEXT.get()
    if not span_context or not _is_span_context_valid(span_context):
        return None
    flags = format(int(span_context.get("traceFlags") or 0), "x").rjust(2, "0")
    return f"00-{span_context['traceId']}-{span_context['spanId']}-{flags}"
