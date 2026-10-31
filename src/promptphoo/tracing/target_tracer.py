"""Test-case, target and grader spans the evaluator records. Ports src/tracing/targetTracer.ts.

Span contexts and options are camelCase dicts with upstream's keys (`TargetSpanContext`,
`TracedProviderCallOptions`, `GraderSpanContext`).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

from promptphoo.jsvalues import is_truthy, js_string
from promptphoo.tracing.genai_tracer import (
    GenAIAttributes,
    PromptfooAttributes,
    get_genai_tracer,
    record_span_error,
    sanitize_body,
    with_genai_span,
)
from promptphoo.tracing.otel_api import (
    ROOT_CONTEXT,
    Span,
    SpanKind,
    SpanOptions,
    SpanStatusCode,
    context,
    propagation,
    trace,
)
from promptphoo.tracing.span_roles import (
    SPAN_ROLE_ATTRIBUTE,
    get_active_traceparent,
    with_span_role,
)

T = TypeVar("T")


class TargetAttributes:
    TARGET_TYPE = "promptfoo.target.type"
    TARGET_LABEL = "promptfoo.target.label"


class GraderAttributes:
    GRADER_ID = "promptfoo.grader.id"


MAX_GRADING_EXPLANATION_LENGTH = 1024


class _JsError:
    """What `new Error(message)` records: type `Error`, the message, a one-line stack."""

    name = "Error"

    def __init__(self, message: str) -> None:
        self.message = message
        self.stack = f"Error: {message}" if message else "Error"

    def __bool__(self) -> bool:
        return True


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


async def with_test_case_span(
    root_span: Span | None,
    fn: Callable[[], Awaitable[T]],
    get_deferred_completion: Callable[[T], Awaitable[Any] | None] | None = None,
) -> T:
    """Keep one recorded test-case root active until immediate or deferred grading finishes."""
    if root_span is None:
        return await fn()

    try:
        root_context = trace.set_span(ROOT_CONTEXT, root_span)
        result: T = await context.with_(root_context, lambda: with_span_role("test_case", fn))
    except BaseException as error:
        record_span_error(root_span, error)
        root_span.end()
        raise

    def finish(*_: Any) -> None:
        row = result[0] if isinstance(result, list) and result else None
        if isinstance(row, dict):
            if isinstance(row.get("success"), bool):
                root_span.set_attribute("promptfoo.test.success", row["success"])
            if _is_number(row.get("score")):
                root_span.set_attribute("promptfoo.test.score", row["score"])
            if "error" in row and is_truthy(row["error"]):
                root_span.set_status(SpanStatusCode.ERROR, js_string(row["error"]))
            else:
                root_span.set_status(SpanStatusCode.OK)
        else:
            root_span.set_status(SpanStatusCode.OK)
        root_span.end()

    completion = get_deferred_completion(result) if get_deferred_completion else None
    if completion is not None and inspect.isawaitable(completion):
        asyncio.ensure_future(completion).add_done_callback(finish)
    else:
        finish()
    return result


def _parent_context_for(traceparent: str) -> Any:
    active_traceparent = get_active_traceparent()
    active_trace_id = active_traceparent.split("-")[1] if active_traceparent else None
    parts = traceparent.split("-")
    if active_trace_id is not None and len(parts) > 1 and active_trace_id == parts[1]:
        return context.active()
    return propagation.extract(ROOT_CONTEXT, {"traceparent": traceparent})


async def with_target_span(ctx: Mapping[str, Any], fn: Callable[[Span], Awaitable[T]]) -> T:
    """Record one target span for every evaluated provider, including custom providers."""
    parent_context = _parent_context_for(ctx["traceparent"])
    role = ctx.get("role") or "target"
    attributes: dict[str, Any] = {
        PromptfooAttributes.PROVIDER_ID: ctx["providerId"],
        SPAN_ROLE_ATTRIBUTE: role,
    }
    if role == "target":
        attributes[TargetAttributes.TARGET_TYPE] = ctx["targetType"]
        if ctx.get("label"):
            attributes[TargetAttributes.TARGET_LABEL] = ctx["label"]
    if ctx.get("promptLabel"):
        attributes[PromptfooAttributes.PROMPT_LABEL] = ctx["promptLabel"]
    if ctx.get("evalId"):
        attributes[PromptfooAttributes.EVAL_ID] = ctx["evalId"]
    if ctx.get("testIndex") is not None:
        attributes[PromptfooAttributes.TEST_INDEX] = ctx["testIndex"]

    name = ctx.get("label") or ctx["providerId"]

    async def run(span: Span) -> T:
        try:
            result: T = await with_span_role(role, lambda: fn(span))
            _record_target_result(span, result)
            return result
        except BaseException as error:
            record_span_error(span, error)
            raise
        finally:
            span.end()

    return await get_genai_tracer().start_active_span(
        f"grader provider {name}" if role == "grader" else name,
        run,
        SpanOptions(kind=SpanKind.CLIENT, attributes=attributes),
        parent_context,
    )


def _record_target_result(span: Span, result: Any) -> None:
    if isinstance(result, dict) and "cached" in result:
        span.set_attribute(PromptfooAttributes.CACHE_HIT, is_truthy(result["cached"]))
    if isinstance(result, dict) and "error" in result and is_truthy(result["error"]):
        message = js_string(result["error"])
        span.set_status(SpanStatusCode.ERROR, message)
        span.record_exception(_JsError(message))
    else:
        span.set_status(SpanStatusCode.OK)


def get_target_type(provider_id: str) -> str:
    if provider_id.startswith("mcp"):
        return "mcp"
    if provider_id.startswith("ws"):
        return "websocket"
    return "http" if provider_id.startswith("http") else "provider"


def _provider_model(provider: Any, provider_id: str) -> str:
    for attribute in ("modelName", "model_name", "deploymentName", "deployment_name"):
        value = getattr(provider, attribute, None)
        if isinstance(value, str):
            return value
    return ":".join(provider_id.split(":")[1:]) or provider_id


async def with_traced_provider_call(
    options: Mapping[str, Any],
    invoke: Callable[[dict[str, Any] | None], Awaitable[T]],
) -> T:
    """Apply consistent provider spans and propagation without modifying providers.

    `options` is TracedProviderCallOptions: provider, callContext, operationName ('embeddings'),
    role ('target' or 'grader'), promptLabel, evalId, testIndex.
    """
    provider = options["provider"]
    call_context: dict[str, Any] | None = options.get("callContext")
    traceparent = get_active_traceparent() or (call_context or {}).get("traceparent")
    if not traceparent:
        return await invoke(call_context)

    provider_id = provider.id()
    prompt_label = options.get("promptLabel")
    if prompt_label is None:
        prompt_label = ((call_context or {}).get("prompt") or {}).get("label")
    eval_id = options.get("evalId")
    if eval_id is None:
        eval_id = (call_context or {}).get("evaluationId")

    async def call(_span: Span) -> T:
        child_traceparent = get_active_traceparent() or traceparent

        async def invoke_provider() -> T:
            if call_context is not None:
                return await invoke({**call_context, "traceparent": child_traceparent})
            return await invoke(None)

        if options.get("operationName") == "embeddings":
            return await with_genai_span(
                {
                    "system": provider_id.split(":", 1)[0],
                    "operationName": "embeddings",
                    "model": _provider_model(provider, provider_id),
                    "providerId": provider_id,
                    "promptLabel": options.get("promptLabel"),
                    "traceparent": child_traceparent,
                },
                lambda _s: invoke_provider(),
                lambda response: {
                    "tokenUsage": response.get("tokenUsage")
                    if isinstance(response, dict)
                    else None,
                    "cacheHit": response.get("cached") if isinstance(response, dict) else None,
                },
            )
        return await invoke_provider()

    return await with_target_span(
        {
            "targetType": get_target_type(provider_id),
            "providerId": provider_id,
            "label": getattr(provider, "label", None),
            "traceparent": traceparent,
            "promptLabel": prompt_label,
            "evalId": eval_id,
            "testIndex": options.get("testIndex"),
            "role": options.get("role") or "target",
        },
        call,
    )


def _set_evaluation_result_attributes(span: Span, result: Any) -> None:
    grade = result.get("grade") if isinstance(result, dict) and "grade" in result else result
    if not isinstance(grade, dict):
        return
    if isinstance(grade.get("pass"), bool):
        span.set_attribute(
            GenAIAttributes.EVALUATION_SCORE_LABEL, "pass" if grade["pass"] else "fail"
        )
    if _is_number(grade.get("score")):
        span.set_attribute(GenAIAttributes.EVALUATION_SCORE_VALUE, grade["score"])
    reason = grade.get("reason")
    if isinstance(reason, str) and reason:
        explanation = sanitize_body(reason)
        span.set_attribute(
            GenAIAttributes.EVALUATION_EXPLANATION,
            _truncate_explanation(explanation),
        )


def _truncate_explanation(explanation: str) -> str:
    data = explanation.encode("utf-16-le", "surrogatepass")
    if len(data) // 2 <= MAX_GRADING_EXPLANATION_LENGTH:
        return explanation
    head = data[: (MAX_GRADING_EXPLANATION_LENGTH - 1) * 2].decode("utf-16-le", "surrogatepass")
    return f"{head}…"


async def with_grader_span(ctx: Mapping[str, Any], fn: Callable[[], Awaitable[T]]) -> T:
    """Instrument assertion dispatch and registered graders at their invocation boundary.

    `ctx` is GraderSpanContext: graderId, traceparent, evalId, testIndex.
    """
    traceparent = get_active_traceparent() or ctx.get("traceparent")
    if not traceparent:
        return await fn()

    parent_context = _parent_context_for(traceparent)
    attributes: dict[str, Any] = {
        GraderAttributes.GRADER_ID: ctx["graderId"],
        GenAIAttributes.EVALUATION_NAME: ctx["graderId"],
        SPAN_ROLE_ATTRIBUTE: "grader",
    }
    if ctx.get("evalId"):
        attributes[PromptfooAttributes.EVAL_ID] = ctx["evalId"]
    if ctx.get("testIndex") is not None:
        attributes[PromptfooAttributes.TEST_INDEX] = ctx["testIndex"]

    async def run(span: Span) -> T:
        try:
            result = await with_span_role("grader", fn)
            _set_evaluation_result_attributes(span, result)
            span.set_status(SpanStatusCode.OK)
            return result
        except BaseException as error:
            record_span_error(span, error)
            raise
        finally:
            span.end()

    return await get_genai_tracer().start_active_span(
        f"grader {ctx['graderId']}",
        run,
        SpanOptions(kind=SpanKind.INTERNAL, attributes=attributes),
        parent_context,
    )
