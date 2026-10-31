"""GenAI semantic-convention spans around provider calls. Ports src/tracing/genaiTracer.ts.

Span contexts (`GenAISpanContext`) and results (`GenAISpanResult`) are camelCase dicts with
upstream's keys; a missing key or a None value both mean `undefined`. Callbacks receive the
span: `await with_genai_span(ctx, lambda span: provider._call_api_internal(...), extractor)`.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

from promptphoo.jsvalues import json_stringify
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
from promptphoo.tracing.resource_attributes import (
    PROMPTFOO_RESOURCE_ATTR_PARENT_SPAN_ID,
    PROMPTFOO_RESOURCE_ATTR_TRACE_ID,
)
from promptphoo.tracing.span_roles import SPAN_ROLE_ATTRIBUTE, get_active_span_role

__all__ = [
    "PROMPTFOO_RESOURCE_ATTR_PARENT_SPAN_ID",
    "PROMPTFOO_RESOURCE_ATTR_TRACE_ID",
    "GenAIAttributes",
    "PromptfooAttributes",
    "build_chat_span_context",
    "extract_genai_response",
    "extract_provider_response_attributes",
    "get_genai_tracer",
    "get_traceparent",
    "record_span_error",
    "sanitize_body",
    "set_genai_response_attributes",
    "with_genai_span",
    "with_genai_tool_span",
]

logger = logging.getLogger("promptphoo.tracing")

T = TypeVar("T")

TRACER_NAME = "promptfoo.providers"
TRACER_VERSION = "1.0.0"


class GenAIAttributes:
    """GenAI semantic-convention attribute names."""

    # Read-only compatibility for spans emitted before gen_ai.provider.name.
    SYSTEM = "gen_ai.system"
    PROVIDER_NAME = "gen_ai.provider.name"
    OPERATION_NAME = "gen_ai.operation.name"
    AGENT_ID = "gen_ai.agent.id"
    AGENT_NAME = "gen_ai.agent.name"

    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    REQUEST_TOP_K = "gen_ai.request.top_k"
    REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"

    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    EVALUATION_NAME = "gen_ai.evaluation.name"
    EVALUATION_SCORE_VALUE = "gen_ai.evaluation.score.value"
    EVALUATION_SCORE_LABEL = "gen_ai.evaluation.score.label"
    EVALUATION_EXPLANATION = "gen_ai.evaluation.explanation"

    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"
    USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
    USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"


class PromptfooAttributes:
    PROVIDER_ID = "promptfoo.provider.id"
    EVAL_ID = "promptfoo.eval.id"
    TEST_INDEX = "promptfoo.test.index"
    PROMPT_LABEL = "promptfoo.prompt.label"
    CACHE_HIT = "promptfoo.cache_hit"
    REQUEST_BODY = "promptfoo.request.body"
    RESPONSE_BODY = "promptfoo.response.body"
    USAGE_TOTAL_TOKENS = "promptfoo.usage.total_tokens"
    USAGE_CACHED_RESPONSE_TOKENS = "promptfoo.usage.cached_response_tokens"
    USAGE_ACCEPTED_PREDICTION_TOKENS = "promptfoo.usage.accepted_prediction_tokens"
    USAGE_REJECTED_PREDICTION_TOKENS = "promptfoo.usage.rejected_prediction_tokens"


GEN_AI_PROVIDER_NAMES: dict[str, str] = {
    "alibaba": "alibaba_cloud",
    "aws_bedrock": "aws.bedrock",
    "azure": "azure.ai.openai",
    "azure_ai_inference": "azure.ai.inference",
    "azure_openai": "azure.ai.openai",
    "bedrock": "aws.bedrock",
    "gemini": "gcp.gemini",
    "google": "gcp.gen_ai",
    "gcp_vertex_ai": "gcp.vertex_ai",
    "ibm_watsonx": "ibm.watsonx.ai",
    "mistral": "mistral_ai",
    "vertex": "gcp.vertex_ai",
    "watsonx": "ibm.watsonx.ai",
    "xai": "x_ai",
}

# Maximum length for request/response body attributes (UTF-16 code units, as in Node).
MAX_BODY_LENGTH = 4096

# JavaScript's `\s` (re.ASCII would otherwise narrow it); `\b` stays ASCII as in JS.
_WS = r"[\t\n\v\f\r    -     　﻿]"
_I = re.IGNORECASE | re.ASCII


def _redact_base64_secret(match: re.Match[str]) -> str:
    # Only redact if it looks like a base64-encoded secret (not normal text).
    text = match.group(0)
    if re.fullmatch(r"[A-Za-z0-9+/=]{40}", text) and "/" in text:
        return "<REDACTED_SECRET>"
    return text


SENSITIVE_PATTERNS: list[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]] = [
    # API keys with common prefixes (allow hyphens/underscores for keys like sk-proj-...)
    (re.compile(r"\b(sk-[a-zA-Z0-9_-]{20,})", re.ASCII), "<REDACTED_API_KEY>"),
    (re.compile(r"\b(pk-[a-zA-Z0-9_-]{20,})", re.ASCII), "<REDACTED_API_KEY>"),
    (
        re.compile(rf"\b(api[_-]?key[\"']?{_WS}*[:=]{_WS}*[\"']?)([a-zA-Z0-9_-]{{16,}})", _I),
        r"\1<REDACTED>",
    ),
    (
        re.compile(rf"\b(secret[\"']?{_WS}*[:=]{_WS}*[\"']?)([a-zA-Z0-9_-]{{16,}})", _I),
        r"\1<REDACTED>",
    ),
    (
        re.compile(rf"\b(token[\"']?{_WS}*[:=]{_WS}*[\"']?)([a-zA-Z0-9_-]{{16,}})", _I),
        r"\1<REDACTED>",
    ),
    (
        re.compile(
            rf"\b(password[\"']?{_WS}*[:=]{_WS}*[\"']?)([^\t\n\v\f\r    - "
            r"    　﻿\"',}{]+)",
            _I,
        ),
        r"\1<REDACTED>",
    ),
    # Authorization headers
    (
        re.compile(
            rf"(Authorization[\"']?{_WS}*[:=]{_WS}*[\"']?)(Bearer{_WS}+)?([a-zA-Z0-9_.-]{{16,}})",
            _I,
        ),
        r"\1\2<REDACTED>",
    ),
    # AWS credentials
    (re.compile(r"\b(AKIA[A-Z0-9]{16})", re.ASCII), "<REDACTED_AWS_KEY>"),
    (re.compile(r"\b([a-zA-Z0-9/+=]{40})", re.ASCII), _redact_base64_secret),
    # Generic long alphanumeric strings that look like secrets (64+ chars)
    (re.compile(r"\b[a-f0-9]{64,}\b", _I), "<REDACTED_HASH>"),
]


def _js_length(text: str) -> int:
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _js_slice(text: str, end: int) -> str:
    data = text.encode("utf-16-le", "surrogatepass")[: max(end, 0) * 2]
    return data.decode("utf-16-le", "surrogatepass")


def sanitize_body(body: str) -> str:
    """Redact API keys, secrets, tokens and other sensitive patterns from a body string."""
    sanitized = body
    for pattern, replacement in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def truncate_body(body: str, sanitize: bool | None = True) -> str:
    """Sanitize (unless `sanitize` is False) and truncate to MAX_BODY_LENGTH."""
    processed = body if sanitize is False else sanitize_body(body)
    if _js_length(processed) <= MAX_BODY_LENGTH:
        return processed
    return _js_slice(processed, MAX_BODY_LENGTH - 15) + "... [truncated]"


def get_genai_tracer() -> Any:
    return trace.get_tracer(TRACER_NAME, TRACER_VERSION)


def add_active_span_role_attribute(attributes: dict[str, Any]) -> dict[str, Any]:
    """Preserve whether provider-created child spans belong to the target or the grader."""
    role = get_active_span_role()
    return {**attributes, SPAN_ROLE_ATTRIBUTE: role} if role else attributes


def _error_name(error: BaseException | Any) -> str:
    return type(error).__name__ if isinstance(error, BaseException) else "_OTHER"


def _error_message(error: Any) -> str:
    return str(error)


def _serialize_tool_attribute(value: Any) -> str | None:
    if value is None:
        return None
    try:
        serialized = value if isinstance(value, str) else json_stringify(value)
    except (TypeError, ValueError, RecursionError):
        return None
    return truncate_body(serialized)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def with_genai_tool_span(tool: Mapping[str, Any], fn: Callable[[], Any]) -> Any:
    """Record tool execution beneath its existing target or grader span."""
    if trace.get_active_span() is None:
        return await _maybe_await(fn())

    attributes = add_active_span_role_attribute(
        {
            GenAIAttributes.OPERATION_NAME: "execute_tool",
            "gen_ai.tool.name": tool["name"],
            "tool.name": tool["name"],
        }
    )
    if tool.get("callId"):
        attributes["gen_ai.tool.call.id"] = tool["callId"]
    tool_arguments = _serialize_tool_attribute(tool.get("arguments"))
    if tool_arguments is not None:
        attributes["tool.arguments"] = tool_arguments
    result_format = tool.get("resultFormat")

    async def run(span: Span) -> Any:
        try:
            result = await _maybe_await(fn())
            record = result if isinstance(result, dict) else None
            output = _serialize_tool_attribute(
                record["content"]
                if result_format == "mcp" and record is not None and "content" in record
                else result
            )
            if output is not None:
                span.set_attribute("tool.output", output)
            if record is not None and (
                record.get("isError") is True
                or (result_format == "mcp" and bool(record.get("error")))
            ):
                span.set_attribute("tool.is_error", True)
                span.set_attribute("error.type", "tool_error")
                error = record.get("error")
                span.set_status(
                    SpanStatusCode.ERROR,
                    truncate_body(error) if isinstance(error, str) else None,
                )
            else:
                span.set_attribute("tool.is_error", False)
                span.set_status(SpanStatusCode.OK)
            return result
        except BaseException as error:
            span.set_attribute("tool.is_error", True)
            span.set_attribute("error.type", _error_name(error))
            span.set_status(SpanStatusCode.ERROR, truncate_body(_error_message(error)))
            if isinstance(error, BaseException):
                span.record_exception(_SanitizedError(error))
            raise
        finally:
            span.end()

    return await get_genai_tracer().start_active_span(
        f"execute_tool {tool['name']}",
        run,
        SpanOptions(kind=SpanKind.INTERNAL, attributes=attributes),
    )


class _SanitizedError:
    """An exception record with its message and stack sanitized and bounded."""

    def __init__(self, error: BaseException) -> None:
        import traceback

        self.name = type(error).__name__
        self.message = truncate_body(str(error))
        self.stack = truncate_body("".join(traceback.format_exception(error)).rstrip("\n"))

    def __bool__(self) -> bool:
        return True


def record_span_error(span: Span, error: Any) -> None:
    span.set_status(SpanStatusCode.ERROR, _error_message(error))
    if isinstance(error, BaseException):
        span.record_exception(error)


def extract_genai_response(
    response: Mapping[str, Any], include_finish_reason: bool = False
) -> dict[str, Any]:
    """Token usage (and optionally the finish reason) for the span's response attributes."""
    result: dict[str, Any] = {}
    if response.get("tokenUsage"):
        result["tokenUsage"] = dict(response["tokenUsage"])
    if include_finish_reason and response.get("finishReason"):
        result["finishReasons"] = [response["finishReason"]]
    return result


def normalize_operation_name(operation_name: str) -> str:
    if operation_name == "completion":
        return "text_completion"
    return "embeddings" if operation_name == "embedding" else operation_name


def get_provider_name(system: str) -> str:
    base_system = system.split(":", 1)[0]
    normalized = re.sub(r"[-.\s]", "_", base_system.lower())
    return GEN_AI_PROVIDER_NAMES.get(normalized, base_system)


def build_request_attributes(ctx: Mapping[str, Any], operation_name: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        GenAIAttributes.PROVIDER_NAME: get_provider_name(ctx["system"]),
        GenAIAttributes.OPERATION_NAME: operation_name,
        PromptfooAttributes.PROVIDER_ID: ctx.get("providerId"),
    }
    agent_name = ctx.get("agentName")
    agent_id = ctx.get("agentId")
    if operation_name == "invoke_agent" and agent_name:
        attrs[GenAIAttributes.AGENT_NAME] = agent_name
    if operation_name == "invoke_agent" and agent_id:
        attrs[GenAIAttributes.AGENT_ID] = agent_id
    if operation_name != "invoke_agent" or (
        ctx.get("model") != agent_name and ctx.get("model") != agent_id
    ):
        attrs[GenAIAttributes.REQUEST_MODEL] = ctx.get("model")
    if ctx.get("openaiApiType") and attrs[GenAIAttributes.PROVIDER_NAME] == "openai":
        attrs["openai.api.type"] = ctx["openaiApiType"]

    span_role = get_active_span_role()
    if span_role:
        attrs[SPAN_ROLE_ATTRIBUTE] = span_role

    for key, attribute in (
        ("maxTokens", GenAIAttributes.REQUEST_MAX_TOKENS),
        ("temperature", GenAIAttributes.REQUEST_TEMPERATURE),
        ("topP", GenAIAttributes.REQUEST_TOP_P),
        ("topK", GenAIAttributes.REQUEST_TOP_K),
    ):
        if ctx.get(key) is not None:
            attrs[attribute] = ctx[key]
    stop_sequences = ctx.get("stopSequences")
    if stop_sequences and len(stop_sequences) > 0:
        attrs[GenAIAttributes.REQUEST_STOP_SEQUENCES] = stop_sequences
    if ctx.get("frequencyPenalty") is not None:
        attrs[GenAIAttributes.REQUEST_FREQUENCY_PENALTY] = ctx["frequencyPenalty"]
    if ctx.get("presencePenalty") is not None:
        attrs[GenAIAttributes.REQUEST_PRESENCE_PENALTY] = ctx["presencePenalty"]

    if ctx.get("evalId"):
        attrs[PromptfooAttributes.EVAL_ID] = ctx["evalId"]
    if ctx.get("testIndex") is not None:
        attrs[PromptfooAttributes.TEST_INDEX] = ctx["testIndex"]
    if ctx.get("promptLabel"):
        attrs[PromptfooAttributes.PROMPT_LABEL] = ctx["promptLabel"]

    if ctx.get("requestBody"):
        attrs[PromptfooAttributes.REQUEST_BODY] = truncate_body(
            ctx["requestBody"], ctx.get("sanitizeBodies")
        )
    return attrs


async def with_genai_span(
    ctx: Mapping[str, Any],
    fn: Callable[[Span], Awaitable[T]],
    result_extractor: Callable[[T], dict[str, Any]] | None = None,
) -> T:
    """Run `fn(span)` inside a GenAI CLIENT span: request attributes before the call, response
    attributes (token usage, finish reasons, cache status, body) after it, an ERROR status for
    thrown errors and `{error: ...}` responses."""
    tracer = get_genai_tracer()
    operation_name = normalize_operation_name(ctx["operationName"])

    # Agent spans identify the invoked agent; inference spans identify the requested model.
    if operation_name == "invoke_agent":
        agent_name = ctx.get("agentName")
        span_name = f"{operation_name} {agent_name}" if agent_name else operation_name
    else:
        span_name = f"{operation_name} {ctx.get('model')}"

    # Link to the evaluation's trace through the explicit traceparent, unless it already names
    # the active span.
    parent_context = context.active()
    active_span = trace.get_span(parent_context)
    active_span_context = active_span.span_context() if active_span is not None else None
    traceparent = ctx.get("traceparent")
    parts = traceparent.split("-") if traceparent else []
    explicit_trace_id = parts[1] if len(parts) > 1 else None
    explicit_span_id = parts[2] if len(parts) > 2 else None

    def _lower(value: str | None) -> str | None:
        return value.lower() if value is not None else None

    if traceparent and (
        _lower(active_span_context.trace_id if active_span_context else None)
        != _lower(explicit_trace_id)
        or _lower(active_span_context.span_id if active_span_context else None)
        != _lower(explicit_span_id)
    ):
        parent_context = propagation.extract(ROOT_CONTEXT, {"traceparent": traceparent})

    async def span_callback(span: Span) -> T:
        try:
            value = await fn(span)
            if result_extractor is not None:
                result = result_extractor(value)
                if (
                    result.get("cacheHit") is None
                    and isinstance(value, dict)
                    and isinstance(value.get("cached"), bool)
                ):
                    result["cacheHit"] = value["cached"]
                set_genai_response_attributes(span, result, ctx.get("sanitizeBodies"))
            error = value.get("error") if isinstance(value, dict) else None
            if isinstance(error, str) and error:
                span.set_attribute("error.type", "provider_error")
                span.set_status(SpanStatusCode.ERROR, error)
            else:
                span.set_status(SpanStatusCode.OK)
            return value
        except BaseException as error:
            span.set_attribute("error.type", _error_name(error))
            record_span_error(span, error)
            raise
        finally:
            span.end()

    return await tracer.start_active_span(
        span_name,
        span_callback,
        SpanOptions(kind=SpanKind.CLIENT, attributes=build_request_attributes(ctx, operation_name)),
        parent_context,
    )


def set_genai_response_attributes(
    span: Span, result: Mapping[str, Any], sanitize: bool | None = True
) -> None:
    """Set response attributes on a span after the API call completes."""
    usage = result.get("tokenUsage")
    if usage:
        if usage.get("prompt") is not None:
            span.set_attribute(GenAIAttributes.USAGE_INPUT_TOKENS, usage["prompt"])
        if usage.get("completion") is not None:
            span.set_attribute(GenAIAttributes.USAGE_OUTPUT_TOKENS, usage["completion"])
        if usage.get("total") is not None:
            span.set_attribute(PromptfooAttributes.USAGE_TOTAL_TOKENS, usage["total"])
        details = usage.get("completionDetails")
        if usage.get("cached") is not None:
            if result.get("cacheHit") is True:
                span.set_attribute(
                    PromptfooAttributes.USAGE_CACHED_RESPONSE_TOKENS, usage["cached"]
                )
            elif (details or {}).get("cacheReadInputTokens") is None:
                span.set_attribute(GenAIAttributes.USAGE_CACHE_READ_INPUT_TOKENS, usage["cached"])
        if details:
            for key, attribute in (
                ("reasoning", GenAIAttributes.USAGE_REASONING_OUTPUT_TOKENS),
                ("acceptedPrediction", PromptfooAttributes.USAGE_ACCEPTED_PREDICTION_TOKENS),
                ("rejectedPrediction", PromptfooAttributes.USAGE_REJECTED_PREDICTION_TOKENS),
                ("cacheReadInputTokens", GenAIAttributes.USAGE_CACHE_READ_INPUT_TOKENS),
                ("cacheCreationInputTokens", GenAIAttributes.USAGE_CACHE_CREATION_INPUT_TOKENS),
            ):
                if details.get(key) is not None:
                    span.set_attribute(attribute, details[key])

    if result.get("responseModel"):
        span.set_attribute(GenAIAttributes.RESPONSE_MODEL, result["responseModel"])
    if result.get("responseId"):
        span.set_attribute(GenAIAttributes.RESPONSE_ID, result["responseId"])
    finish_reasons = result.get("finishReasons")
    if finish_reasons and len(finish_reasons) > 0:
        span.set_attribute(GenAIAttributes.RESPONSE_FINISH_REASONS, finish_reasons)

    if result.get("cacheHit") is not None:
        span.set_attribute(PromptfooAttributes.CACHE_HIT, result["cacheHit"])
    if result.get("responseBody"):
        span.set_attribute(
            PromptfooAttributes.RESPONSE_BODY, truncate_body(result["responseBody"], sanitize)
        )

    # Provider-specific additional attributes, sanitized like the bodies.
    for key, value in (result.get("additionalAttributes") or {}).items():
        if value is None:
            continue
        if isinstance(value, str):
            span.set_attribute(key, truncate_body(value, sanitize))
        else:
            span.set_attribute(key, value)


def get_traceparent() -> str | None:
    """The W3C traceparent of the active span, if there is one."""
    active_span = trace.get_active_span()
    if active_span is None:
        return None
    ctx = active_span.span_context()
    trace_flags = format(ctx.trace_flags, "x").rjust(2, "0")
    return f"00-{ctx.trace_id}-{ctx.span_id}-{trace_flags}"


def build_chat_span_context(
    *,
    system: str,
    model: str,
    provider_id: str,
    prompt: str,
    context: Mapping[str, Any] | None = None,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A `chat` span context from the fields every provider shares."""
    ctx = context or {}
    test = ctx.get("test") or {}
    test_index = ctx.get("testIdx")
    if test_index is None:
        test_index = (test.get("vars") or {}).get("__testIdx")
    return {
        "system": system,
        "operationName": "chat",
        "model": model,
        "providerId": provider_id,
        "evalId": ctx.get("evaluationId") or (test.get("metadata") or {}).get("evaluationId"),
        "testIndex": test_index,
        "promptLabel": (ctx.get("prompt") or {}).get("label"),
        "traceparent": ctx.get("traceparent"),
        "requestBody": prompt,
        **(request or {}),
    }


def get_test_index(context: Mapping[str, Any] | None) -> Any:
    """`context?.testIdx ?? context?.test?.vars?.__testIdx`, shared by the provider spans."""
    ctx = context or {}
    if ctx.get("testIdx") is not None:
        return ctx["testIdx"]
    return ((ctx.get("test") or {}).get("vars") or {}).get("__testIdx")


def extract_provider_response_attributes(response: Mapping[str, Any]) -> dict[str, Any]:
    """The standard response attributes of a ProviderResponse; each only when present."""
    result: dict[str, Any] = {}
    if response.get("tokenUsage"):
        # The full usage object, completionDetails included.
        result["tokenUsage"] = dict(response["tokenUsage"])
    if response.get("finishReason"):
        result["finishReasons"] = [response["finishReason"]]
    if response.get("cached") is not None:
        result["cacheHit"] = response["cached"]
    if "output" in response and response["output"] is not None:
        output = response["output"]
        result["responseBody"] = output if isinstance(output, str) else json_stringify(output)
    return result


# --- per-turn spans for streaming agent providers -------------------------------------------


def open_turn_span(state: dict[str, Any], opts: Mapping[str, Any]) -> None:
    """Open a `gen_ai.turn N` span on `state`, closing a still-open prior turn as ERROR."""
    active = state.get("activeTurnSpan")
    if active is not None:
        try:
            active.set_status(
                SpanStatusCode.ERROR, "Turn span not properly closed before next turn started"
            )
            active.end(opts["eventTime"])
        except Exception:
            pass
        state["activeTurnSpan"] = None
    index = state.get("turnCount", 0) + 1
    try:
        span = opts["tracer"].start_span(
            f"gen_ai.turn {index}",
            SpanOptions(
                kind=SpanKind.INTERNAL,
                start_time=opts["eventTime"],
                attributes=add_active_span_role_attribute(
                    {
                        "gen_ai.turn.index": index,
                        GenAIAttributes.PROVIDER_NAME: get_provider_name(opts["system"]),
                        **(opts.get("attributes") or {}),
                    }
                ),
            ),
        )
        state["turnCount"] = index
        state["activeTurnIndex"] = index
        state["activeTurnSpan"] = span
    except Exception as err:
        logger.warning(
            "[%s] Failed to start turn span: %s", opts.get("logLabel") or "TurnSpan", err
        )
        state["activeTurnSpan"] = None


def close_turn_span(state: dict[str, Any], opts: Mapping[str, Any] | None = None) -> None:
    """Close the active `gen_ai.turn` span, applying usage attributes and an OK/ERROR status."""
    opts = opts or {}
    span = state.get("activeTurnSpan")
    if span is None:
        return
    try:
        for key, value in (opts.get("attributes") or {}).items():
            if value is not None:
                span.set_attribute(key, value)
        if opts.get("errorMessage"):
            span.set_status(SpanStatusCode.ERROR, opts["errorMessage"])
        else:
            span.set_status(SpanStatusCode.OK)
        span.end(opts.get("eventTime"))
    except Exception as err:
        logger.warning("[%s] Failed to end turn span: %s", opts.get("logLabel") or "TurnSpan", err)
    state["activeTurnSpan"] = None


def emit_turn_marker_span(opts: Mapping[str, Any]) -> None:
    """Emit a fire-and-forget `gen_ai.turn N` marker span between the given timestamps."""
    try:
        span = opts["tracer"].start_span(
            f"gen_ai.turn {opts['index']}",
            SpanOptions(
                kind=SpanKind.INTERNAL,
                start_time=opts["startTime"],
                attributes=add_active_span_role_attribute(dict(opts["attributes"])),
            ),
        )
        if opts.get("errorMessage"):
            span.set_status(SpanStatusCode.ERROR, opts["errorMessage"])
        else:
            span.set_status(SpanStatusCode.OK)
        span.end(opts["endTime"])
    except Exception as err:
        logger.warning("[%s] Failed to emit turn span: %s", opts.get("logLabel") or "TurnSpan", err)
