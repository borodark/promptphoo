"""Ports test/tracing/genaiTracer.test.ts.

Upstream mocks the OTel tracer; here a BasicTracerProvider with an in-memory exporter is
registered and the finished spans are inspected.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from promptphoo.tracing.genai_tracer import (
    MAX_BODY_LENGTH,
    GenAIAttributes,
    PromptfooAttributes,
    build_chat_span_context,
    extract_genai_response,
    extract_provider_response_attributes,
    get_test_index,
    get_traceparent,
    sanitize_body,
    set_genai_response_attributes,
    truncate_body,
    with_genai_span,
    with_genai_tool_span,
)
from promptphoo.tracing.otel_api import SpanKind, SpanStatusCode, propagation, trace
from promptphoo.tracing.otel_trace_sdk import (
    BasicTracerProvider,
    InMemorySpanExporter,
    SimpleSpanProcessor,
)
from promptphoo.tracing.span_roles import with_span_role

BASE: dict[str, Any] = {
    "system": "openai",
    "operationName": "chat",
    "model": "gpt-4",
    "providerId": "openai:gpt-4",
}


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    exp = InMemorySpanExporter()
    BasicTracerProvider(span_processors=[SimpleSpanProcessor(exp)]).register()
    yield exp
    trace.disable()
    propagation.disable()


def _only(exporter: InMemorySpanExporter) -> Any:
    [span] = exporter.get_finished_spans()
    return span


async def _ok(_span: Any = None) -> dict[str, Any]:
    return {"output": "test"}


def _detached_span() -> Any:
    return trace.get_tracer("test").start_span("detached")


# --- attribute names ------------------------------------------------------------------------


def test_genai_attribute_names():
    assert GenAIAttributes.SYSTEM == "gen_ai.system"
    assert GenAIAttributes.PROVIDER_NAME == "gen_ai.provider.name"
    assert GenAIAttributes.OPERATION_NAME == "gen_ai.operation.name"
    assert GenAIAttributes.AGENT_ID == "gen_ai.agent.id"
    assert GenAIAttributes.AGENT_NAME == "gen_ai.agent.name"
    assert GenAIAttributes.REQUEST_MODEL == "gen_ai.request.model"
    assert GenAIAttributes.USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
    assert GenAIAttributes.USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
    assert GenAIAttributes.USAGE_REASONING_OUTPUT_TOKENS == "gen_ai.usage.reasoning.output_tokens"
    assert GenAIAttributes.USAGE_CACHE_READ_INPUT_TOKENS == "gen_ai.usage.cache_read.input_tokens"
    assert (
        GenAIAttributes.USAGE_CACHE_CREATION_INPUT_TOKENS
        == "gen_ai.usage.cache_creation.input_tokens"
    )


def test_promptfoo_attribute_names():
    assert PromptfooAttributes.PROVIDER_ID == "promptfoo.provider.id"
    assert PromptfooAttributes.EVAL_ID == "promptfoo.eval.id"
    assert PromptfooAttributes.TEST_INDEX == "promptfoo.test.index"
    assert PromptfooAttributes.PROMPT_LABEL == "promptfoo.prompt.label"
    assert PromptfooAttributes.USAGE_TOTAL_TOKENS == "promptfoo.usage.total_tokens"
    assert PromptfooAttributes.USAGE_CACHED_RESPONSE_TOKENS == (
        "promptfoo.usage.cached_response_tokens"
    )


@pytest.mark.parametrize("finish_reason", [False, True])
def test_extract_genai_response_copies_usage(finish_reason: bool):
    response = {
        "output": "body",
        "cached": True,
        "finishReason": "stop",
        "tokenUsage": {"prompt": 2, "completion": 3, "total": 5, "cached": 4},
    }
    result = extract_genai_response(response, finish_reason)
    assert result == {
        "tokenUsage": response["tokenUsage"],
        **({"finishReasons": ["stop"]} if finish_reason else {}),
    }
    assert result["tokenUsage"] is not response["tokenUsage"]
    assert extract_genai_response({}) == {}


def test_extract_provider_response_attributes():
    assert extract_provider_response_attributes(
        {"output": {"a": 1}, "cached": False, "finishReason": "length",
         "tokenUsage": {"prompt": 1, "completionDetails": {"reasoning": 2}}}
    ) == {
        "tokenUsage": {"prompt": 1, "completionDetails": {"reasoning": 2}},
        "finishReasons": ["length"],
        "cacheHit": False,
        "responseBody": '{"a":1}',
    }  # fmt: skip
    assert extract_provider_response_attributes({"output": "hi"}) == {"responseBody": "hi"}
    assert extract_provider_response_attributes({}) == {}


# --- withGenAISpan --------------------------------------------------------------------------


async def test_span_name_kind_and_required_attributes(exporter):
    await with_genai_span(BASE, _ok)
    span = _only(exporter)
    assert span.name == "chat gpt-4"
    assert span.kind == SpanKind.CLIENT
    assert span.attributes == {
        GenAIAttributes.PROVIDER_NAME: "openai",
        GenAIAttributes.OPERATION_NAME: "chat",
        PromptfooAttributes.PROVIDER_ID: "openai:gpt-4",
        GenAIAttributes.REQUEST_MODEL: "gpt-4",
    }
    assert GenAIAttributes.SYSTEM not in span.attributes
    assert span.status.code == SpanStatusCode.OK


async def test_agent_invocation_does_not_treat_agent_name_as_model(exporter):
    ctx = {**BASE, "operationName": "invoke_agent", "agentName": "Support Agent",
           "model": "Support Agent"}  # fmt: skip
    await with_genai_span(ctx, _ok)
    span = _only(exporter)
    assert span.name == "invoke_agent Support Agent"
    assert span.attributes[GenAIAttributes.AGENT_NAME] == "Support Agent"
    assert GenAIAttributes.REQUEST_MODEL not in span.attributes


async def test_agent_invocation_keeps_explicit_model(exporter):
    await with_genai_span({**BASE, "operationName": "invoke_agent", "agentName": "A"}, _ok)
    span = _only(exporter)
    assert span.attributes[GenAIAttributes.AGENT_NAME] == "A"
    assert span.attributes[GenAIAttributes.REQUEST_MODEL] == "gpt-4"


async def test_hosted_agent_ids_are_separate(exporter):
    ctx = {**BASE, "operationName": "invoke_agent", "agentId": "asst_123", "model": "asst_123"}
    await with_genai_span(ctx, _ok)
    span = _only(exporter)
    assert span.name == "invoke_agent"
    assert span.attributes[GenAIAttributes.AGENT_ID] == "asst_123"
    assert GenAIAttributes.AGENT_NAME not in span.attributes
    assert GenAIAttributes.REQUEST_MODEL not in span.attributes


async def test_openai_api_type_only_for_openai(exporter):
    await with_genai_span({**BASE, "openaiApiType": "responses"}, _ok)
    await with_genai_span({**BASE, "system": "azure", "openaiApiType": "responses"}, _ok)
    first, second = exporter.get_finished_spans()
    assert first.attributes["openai.api.type"] == "responses"
    assert "openai.api.type" not in second.attributes


@pytest.mark.parametrize(
    ("system", "provider_name"),
    [
        ("alibaba", "alibaba_cloud"),
        ("bedrock", "aws.bedrock"),
        ("azure", "azure.ai.openai"),
        ("vertex:anthropic", "gcp.vertex_ai"),
        ("mistral", "mistral_ai"),
        ("watsonx", "ibm.watsonx.ai"),
        ("xai", "x_ai"),
        ("custom-provider", "custom-provider"),
    ],
)
async def test_normalizes_provider_names(exporter, system, provider_name):
    await with_genai_span({**BASE, "system": system}, _ok)
    assert _only(exporter).attributes[GenAIAttributes.PROVIDER_NAME] == provider_name


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("completion", "text_completion"),
        ("embedding", "embeddings"),
        ("text_completion", "text_completion"),
        ("embeddings", "embeddings"),
    ],
)
async def test_normalizes_operation_names(exporter, operation, expected):
    await with_genai_span({**BASE, "operationName": operation}, _ok)
    span = _only(exporter)
    assert span.name == f"{expected} gpt-4"
    assert span.attributes[GenAIAttributes.OPERATION_NAME] == expected


async def test_optional_request_and_promptfoo_attributes(exporter):
    ctx = {**BASE, "maxTokens": 1000, "temperature": 0.7, "topP": 0.9, "topK": 3,
           "stopSequences": ["END"], "frequencyPenalty": 0.1, "presencePenalty": 0.2,
           "evalId": "eval-123", "testIndex": 5, "promptLabel": "test-prompt"}  # fmt: skip
    await with_genai_span(ctx, _ok)
    attrs = _only(exporter).attributes
    assert attrs[GenAIAttributes.REQUEST_MAX_TOKENS] == 1000
    assert attrs[GenAIAttributes.REQUEST_TEMPERATURE] == 0.7
    assert attrs[GenAIAttributes.REQUEST_TOP_P] == 0.9
    assert attrs[GenAIAttributes.REQUEST_TOP_K] == 3
    assert attrs[GenAIAttributes.REQUEST_STOP_SEQUENCES] == ["END"]
    assert attrs[GenAIAttributes.REQUEST_FREQUENCY_PENALTY] == 0.1
    assert attrs[GenAIAttributes.REQUEST_PRESENCE_PENALTY] == 0.2
    assert attrs[PromptfooAttributes.EVAL_ID] == "eval-123"
    assert attrs[PromptfooAttributes.TEST_INDEX] == 5
    assert attrs[PromptfooAttributes.PROMPT_LABEL] == "test-prompt"


async def test_empty_stop_sequences_are_omitted(exporter):
    await with_genai_span({**BASE, "stopSequences": []}, _ok)
    assert GenAIAttributes.REQUEST_STOP_SEQUENCES not in _only(exporter).attributes


async def test_active_span_role_is_recorded(exporter):
    await with_span_role("grader", lambda: with_genai_span(BASE, _ok))
    assert _only(exporter).attributes["promptfoo.span.role"] == "grader"


async def test_returns_the_wrapped_result(exporter):
    expected = {"output": "Hello, world!"}

    async def fn(_span):
        return expected

    assert await with_genai_span(BASE, fn) is expected
    assert _only(exporter).end_time != (0, 0)


async def test_error_status_and_exception_on_failure(exporter):
    async def fail(_span):
        raise RuntimeError("API call failed")

    with pytest.raises(RuntimeError, match="API call failed"):
        await with_genai_span(BASE, fail)
    span = _only(exporter)
    assert span.status.code == SpanStatusCode.ERROR
    assert span.status.message == "API call failed"
    # Python's exception class name stands in for JS's `Error`.
    assert span.attributes["error.type"] == "RuntimeError"
    [event] = span.events
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "RuntimeError"
    assert event.attributes["exception.message"] == "API call failed"
    assert span.end_time != (0, 0)


async def test_provider_error_response_is_an_error(exporter):
    async def fn(_span):
        return {"error": "provider unavailable"}

    await with_genai_span(BASE, fn)
    span = _only(exporter)
    assert span.attributes["error.type"] == "provider_error"
    assert span.status.code == SpanStatusCode.ERROR
    assert span.status.message == "provider unavailable"


async def test_result_extractor_sets_response_attributes(exporter):
    seen = []

    def extractor(value):
        seen.append(value)
        return {"tokenUsage": {"prompt": 100, "completion": 50}, "responseId": "resp-123"}

    await with_genai_span(BASE, _ok, extractor)
    assert seen == [{"output": "test"}]
    attrs = _only(exporter).attributes
    assert attrs[GenAIAttributes.USAGE_INPUT_TOKENS] == 100
    assert attrs[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 50
    assert attrs[GenAIAttributes.RESPONSE_ID] == "resp-123"


async def test_cache_status_from_response_when_extractor_omits_it(exporter):
    async def fn(_span):
        return {"output": "cached output", "cached": True,
                "tokenUsage": {"cached": 150, "total": 150}}  # fmt: skip

    await with_genai_span(BASE, fn, lambda r: {"tokenUsage": r["tokenUsage"]})
    attrs = _only(exporter).attributes
    assert attrs[PromptfooAttributes.CACHE_HIT] is True
    assert attrs[PromptfooAttributes.USAGE_CACHED_RESPONSE_TOKENS] == 150
    assert GenAIAttributes.USAGE_CACHE_READ_INPUT_TOKENS not in attrs


async def test_explicit_cache_classification_wins(exporter):
    async def fn(_span):
        return {"output": "x", "cached": True, "tokenUsage": {"cached": 20}}

    await with_genai_span(BASE, fn, lambda r: {"tokenUsage": r["tokenUsage"], "cacheHit": False})
    attrs = _only(exporter).attributes
    assert attrs[PromptfooAttributes.CACHE_HIT] is False
    assert attrs[GenAIAttributes.USAGE_CACHE_READ_INPUT_TOKENS] == 20
    assert PromptfooAttributes.USAGE_CACHED_RESPONSE_TOKENS not in attrs


async def test_explicit_traceparent_sets_the_parent(exporter):
    traceparent = f"00-{'a' * 32}-{'b' * 16}-01"
    await with_genai_span({**BASE, "traceparent": traceparent}, _ok)
    span = _only(exporter)
    assert span.span_context().trace_id == "a" * 32
    assert span.parent_span_context.span_id == "b" * 16


async def test_get_traceparent_inside_span(exporter):
    seen = []

    async def fn(span):
        seen.append((get_traceparent(), span.span_context()))
        return {}

    assert get_traceparent() is None
    await with_genai_span(BASE, fn)
    [(traceparent, ctx)] = seen
    assert traceparent == f"00-{ctx.trace_id}-{ctx.span_id}-01"


# --- withGenAIToolSpan ----------------------------------------------------------------------


async def _in_parent(fn):
    async def run(span):
        try:
            return await fn()
        finally:
            span.end()

    return await trace.get_tracer("test").start_active_span("parent", run)


def _tool_span(exporter):
    return next(s for s in exporter.get_finished_spans() if s.name != "parent")


async def test_tool_span_attributes_and_sanitization(exporter):
    async def tool():
        return {"token": "secret-token-value-12345678901234567890"}

    result = await _in_parent(
        lambda: with_genai_tool_span(
            {"name": "lookup_account", "arguments": {"apiKey": "sk-abcdefghijklmnopqrstuvwxyz"},
             "callId": "call-123"},
            tool,
        )
    )  # fmt: skip
    assert result == {"token": "secret-token-value-12345678901234567890"}
    span = _tool_span(exporter)
    assert span.name == "execute_tool lookup_account"
    assert span.kind == SpanKind.INTERNAL
    assert span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert span.attributes["gen_ai.tool.name"] == "lookup_account"
    assert span.attributes["gen_ai.tool.call.id"] == "call-123"
    assert span.attributes["tool.name"] == "lookup_account"
    assert span.attributes["tool.arguments"] == '{"apiKey":"<REDACTED_API_KEY>"}'
    assert span.attributes["tool.output"] == '{"token":"<REDACTED>"}'
    assert span.status.code == SpanStatusCode.OK


async def test_no_orphan_tool_span_without_parent(exporter):
    assert await with_genai_tool_span({"name": "search"}, lambda: "found") == "found"
    assert exporter.get_finished_spans() == []


async def test_tool_span_records_whole_result_with_content(exporter):
    result = {"content": "account found", "metadata": {"destination": "audit-log", "attempt": 2}}
    out = await _in_parent(lambda: with_genai_tool_span({"name": "lookup"}, lambda: result))
    assert out is result
    assert _tool_span(exporter).attributes["tool.output"] == json.dumps(
        result, separators=(",", ":")
    )


async def test_mcp_results_record_only_content(exporter):
    result = {"content": "account found", "metadata": {"requestId": "request-123"}}
    await _in_parent(
        lambda: with_genai_tool_span({"name": "lookup", "resultFormat": "mcp"}, lambda: result)
    )
    assert _tool_span(exporter).attributes["tool.output"] == "account found"


@pytest.mark.parametrize(
    ("result", "fmt"),
    [
        ({"content": "account found", "error": "business-domain error detail"}, None),
        ({"content": "account found", "error": ""}, "mcp"),
    ],
)
async def test_error_fields_that_are_not_failures(exporter, result, fmt):
    tool = {"name": "lookup", **({"resultFormat": fmt} if fmt else {})}
    assert await _in_parent(lambda: with_genai_tool_span(tool, lambda: result)) is result
    span = _tool_span(exporter)
    assert span.attributes["tool.is_error"] is False
    assert span.status.code == SpanStatusCode.OK


async def test_mcp_error_results_are_failures(exporter):
    failure = {"content": "denied", "isError": True}
    tool = {"name": "search", "resultFormat": "mcp"}
    assert await _in_parent(lambda: with_genai_tool_span(tool, lambda: failure)) is failure
    span = _tool_span(exporter)
    assert span.attributes["tool.is_error"] is True
    assert span.status.code == SpanStatusCode.ERROR
    assert span.status.message is None


async def test_caught_mcp_sdk_failures(exporter):
    failure = {"content": "", "error": "MCP transport disconnected"}
    tool = {"name": "search", "resultFormat": "mcp"}
    assert await _in_parent(lambda: with_genai_tool_span(tool, lambda: failure)) is failure
    span = _tool_span(exporter)
    assert span.attributes["tool.is_error"] is True
    assert span.attributes["error.type"] == "tool_error"
    assert span.status.message == "MCP transport disconnected"


async def test_tool_span_thrown_errors_are_sanitized(exporter):
    secret = "sk-abcdefghijklmnopqrstuvwxyz"
    error = RuntimeError(f"Authentication failed for {secret}")

    def tool():
        raise error

    with pytest.raises(RuntimeError) as info:
        await _in_parent(lambda: with_genai_tool_span({"name": "lookup"}, tool))
    assert info.value is error
    span = _tool_span(exporter)
    assert span.status.code == SpanStatusCode.ERROR
    assert span.status.message == "Authentication failed for <REDACTED_API_KEY>"
    [event] = span.events
    assert event.attributes["exception.message"] == "Authentication failed for <REDACTED_API_KEY>"
    assert secret not in event.attributes["exception.stacktrace"]
    assert span.attributes["tool.is_error"] is True


async def test_unserializable_tool_attributes_do_not_fail(exporter):
    circular: dict[str, Any] = {}
    circular["self"] = circular
    tool = {"name": "search", "arguments": circular}
    assert await _in_parent(lambda: with_genai_tool_span(tool, lambda: circular)) is circular
    span = _tool_span(exporter)
    assert "tool.arguments" not in span.attributes
    assert "tool.output" not in span.attributes


async def test_large_tool_attributes_are_truncated(exporter):
    large = "x" * 5000
    await _in_parent(lambda: with_genai_tool_span({"name": "s", "arguments": large}, lambda: large))
    attrs = _tool_span(exporter).attributes
    assert len(attrs["tool.arguments"]) == MAX_BODY_LENGTH
    assert "[truncated]" in attrs["tool.arguments"]
    assert "[truncated]" in attrs["tool.output"]


# --- setGenAIResponseAttributes -------------------------------------------------------------


def _attrs_after(result: dict[str, Any]) -> dict[str, Any]:
    span = _detached_span()
    set_genai_response_attributes(span, result)
    return span.attributes


def test_token_usage_attributes(exporter):
    attrs = _attrs_after(
        {"cacheHit": True, "tokenUsage": {"prompt": 100, "completion": 50, "total": 150,
                                          "cached": 20}}
    )  # fmt: skip
    assert attrs[GenAIAttributes.USAGE_INPUT_TOKENS] == 100
    assert attrs[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 50
    assert attrs[PromptfooAttributes.USAGE_TOTAL_TOKENS] == 150
    assert attrs[PromptfooAttributes.USAGE_CACHED_RESPONSE_TOKENS] == 20


def test_prompt_cache_reads_are_not_response_cache_hits(exporter):
    attrs = _attrs_after(
        {"cacheHit": False, "tokenUsage": {"prompt": 100, "completion": 50, "cached": 20}}
    )
    assert attrs[GenAIAttributes.USAGE_CACHE_READ_INPUT_TOKENS] == 20
    assert PromptfooAttributes.USAGE_CACHED_RESPONSE_TOKENS not in attrs


def test_explicit_cache_read_details_win(exporter):
    attrs = _attrs_after(
        {"cacheHit": False,
         "tokenUsage": {"cached": 20, "completionDetails": {"cacheReadInputTokens": 12}}}
    )  # fmt: skip
    assert attrs[GenAIAttributes.USAGE_CACHE_READ_INPUT_TOKENS] == 12


def test_prompt_cache_counts_distinct_on_response_cache_hits(exporter):
    attrs = _attrs_after(
        {"cacheHit": True,
         "tokenUsage": {"cached": 3000, "total": 3000,
                        "completionDetails": {"cacheReadInputTokens": 500}}}
    )  # fmt: skip
    assert attrs[PromptfooAttributes.USAGE_CACHED_RESPONSE_TOKENS] == 3000
    assert attrs[GenAIAttributes.USAGE_CACHE_READ_INPUT_TOKENS] == 500


def test_completion_details_attributes(exporter):
    attrs = _attrs_after(
        {"tokenUsage": {"completionDetails": {"reasoning": 25, "acceptedPrediction": 10,
                                              "rejectedPrediction": 5}}}
    )  # fmt: skip
    assert attrs[GenAIAttributes.USAGE_REASONING_OUTPUT_TOKENS] == 25
    assert attrs[PromptfooAttributes.USAGE_ACCEPTED_PREDICTION_TOKENS] == 10
    assert attrs[PromptfooAttributes.USAGE_REJECTED_PREDICTION_TOKENS] == 5


def test_cache_token_completion_details(exporter):
    attrs = _attrs_after(
        {"tokenUsage": {"completionDetails": {"cacheReadInputTokens": 150,
                                              "cacheCreationInputTokens": 40}}}
    )  # fmt: skip
    assert attrs[GenAIAttributes.USAGE_CACHE_READ_INPUT_TOKENS] == 150
    assert attrs[GenAIAttributes.USAGE_CACHE_CREATION_INPUT_TOKENS] == 40


def test_response_metadata_attributes(exporter):
    attrs = _attrs_after(
        {"responseModel": "gpt-4-0613", "responseId": "chatcmpl-123", "finishReasons": ["stop"]}
    )
    assert attrs[GenAIAttributes.RESPONSE_MODEL] == "gpt-4-0613"
    assert attrs[GenAIAttributes.RESPONSE_ID] == "chatcmpl-123"
    assert attrs[GenAIAttributes.RESPONSE_FINISH_REASONS] == ["stop"]


def test_undefined_values_are_not_set(exporter):
    attrs = _attrs_after({"tokenUsage": {"prompt": 100}})
    assert attrs == {GenAIAttributes.USAGE_INPUT_TOKENS: 100}


def test_additional_attributes_are_sanitized(exporter):
    attrs = _attrs_after(
        {"additionalAttributes": {"x.secret": "sk-abcdefghijklmnopqrstuvwxyz", "x.n": 3,
                                  "x.none": None}}
    )  # fmt: skip
    assert attrs == {"x.secret": "<REDACTED_API_KEY>", "x.n": 3}


# --- body sanitization ----------------------------------------------------------------------


def test_capture_group_and_functional_replacements():
    encoded_secret = "a" * 20 + "/" + "b" * 19
    ordinary = "c" * 40
    assert sanitize_body(f"password=hunter2 {encoded_secret} {ordinary}") == (
        f"password=<REDACTED> <REDACTED_SECRET> {ordinary}"
    )


def test_truncate_body():
    assert truncate_body("short") == "short"
    long = "x" * 5000
    out = truncate_body(long)
    assert len(out) == MAX_BODY_LENGTH
    assert out.endswith("... [truncated]")
    assert truncate_body("sk-abcdefghijklmnopqrstuvwxyz", False) == (
        "sk-abcdefghijklmnopqrstuvwxyz"
    )
    # Lengths are UTF-16 code units, as in Node: an astral character counts twice, and the cut
    # may split a surrogate pair as String.prototype.slice does.
    cut = truncate_body("\U0001f600" * 3000)
    assert len(cut.encode("utf-16-le", "surrogatepass")) // 2 == MAX_BODY_LENGTH
    assert cut.endswith("\ud83d... [truncated]")


SANITIZED = {**BASE, "sanitizeBodies": True}


@pytest.mark.parametrize(
    ("body", "marker", "secret"),
    [
        ('{"api_key": "sk-proj-abcdefghij1234567890abcdefghij1234567890"}',
         "<REDACTED_API_KEY>", "sk-proj-"),
        ('{"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"}',
         "<REDACTED>", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"),
        ('{"credentials": "AKIAIOSFODNN7EXAMPLE"}', "<REDACTED_AWS_KEY>", "AKIAIOSFODNN7EXAMPLE"),
        ('{"password": "supersecret123"}', "<REDACTED>", "supersecret123"),
    ],
)  # fmt: skip
async def test_request_body_is_redacted(exporter, body, marker, secret):
    await with_genai_span({**SANITIZED, "requestBody": body}, _ok)
    value = _only(exporter).attributes[PromptfooAttributes.REQUEST_BODY]
    assert marker in value
    assert secret not in value


async def test_response_body_is_redacted(exporter):
    await with_genai_span(
        SANITIZED,
        _ok,
        lambda _r: {"responseBody": '{"token": "secret-token-value-12345678901234567890"}'},
    )
    value = _only(exporter).attributes[PromptfooAttributes.RESPONSE_BODY]
    assert "<REDACTED>" in value
    assert "secret-token-value-12345678901234567890" not in value


# --- shared span-context helpers ------------------------------------------------------------


def test_build_chat_span_context():
    context = {
        "evaluationId": "eval-1",
        "testIdx": 2,
        "prompt": {"label": "p"},
        "traceparent": "00-x-y-01",
    }
    assert build_chat_span_context(
        system="openai", model="gpt-4.1", provider_id="openai:responses:gpt-4.1",
        prompt="hi", context=context, request={"maxTokens": 64},
    ) == {
        "system": "openai", "operationName": "chat", "model": "gpt-4.1",
        "providerId": "openai:responses:gpt-4.1", "evalId": "eval-1", "testIndex": 2,
        "promptLabel": "p", "traceparent": "00-x-y-01", "requestBody": "hi", "maxTokens": 64,
    }  # fmt: skip
    fallback = build_chat_span_context(
        system="openai", model="m", provider_id="p", prompt="hi",
        context={"test": {"vars": {"__testIdx": 7}, "metadata": {"evaluationId": "eval-2"}}},
    )  # fmt: skip
    assert fallback["testIndex"] == 7
    assert fallback["evalId"] == "eval-2"


def test_get_test_index():
    assert get_test_index(None) is None
    assert get_test_index({"testIdx": 0, "test": {"vars": {"__testIdx": 9}}}) == 0
    assert get_test_index({"test": {"vars": {"__testIdx": 9}}}) == 9
