"""Which part of a test case a span belongs to. Ports src/tracing/spanRoles.ts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from promptphoo.tracing.otel_api import context, create_context_key, is_span_context_valid, trace

SPAN_ROLE_ATTRIBUTE = "promptfoo.span.role"

PromptfooSpanRole = Literal["test_case", "target", "grader"]

_SPAN_ROLE_CONTEXT_KEY = create_context_key("promptfoo.span.role")


def get_active_span_role() -> PromptfooSpanRole | None:
    """Keep evaluator-owned grading activity distinct from the target's behavior."""
    return context.active().get_value(_SPAN_ROLE_CONTEXT_KEY)


async def with_span_role(role: PromptfooSpanRole, fn: Callable[[], Any]) -> Any:
    return await context.with_(context.active().set_value(_SPAN_ROLE_CONTEXT_KEY, role), fn)


def get_active_traceparent() -> str | None:
    """A usable traceparent, only when the active span has a valid recorded context."""
    active_span = trace.get_active_span()
    if active_span is None:
        return None
    span_context = active_span.span_context()
    if not is_span_context_valid(span_context):
        return None
    trace_flags = format(span_context.trace_flags, "x").rjust(2, "0")
    return f"00-{span_context.trace_id}-{span_context.span_id}-{trace_flags}"
