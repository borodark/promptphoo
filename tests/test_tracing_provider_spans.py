"""GenAI spans around provider calls (withGenAISpan in each provider's callApi), plus the
evaluator-target parenting case of test/tracing/providerInstrumentation.test.ts.

Each provider makes one faked call (respx, or a fake boto3 client for Bedrock) with a
BasicTracerProvider + in-memory exporter registered; the finished span is inspected.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx
from azure_helpers import api_options, azure_env, chat_url
from openai_helpers import CHAT_URL, openai_env

from promptphoo import cache
from promptphoo.providers.anthropic.messages import AnthropicMessagesProvider
from promptphoo.providers.azure import AzureChatCompletionProvider
from promptphoo.providers.bedrock.converse import AwsBedrockConverseProvider
from promptphoo.providers.compatible.openrouter import OpenRouterProvider
from promptphoo.providers.google.vertex import VertexChatProvider
from promptphoo.providers.ollama import OllamaChatProvider, OllamaCompletionProvider
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.completion import OpenAiCompletionProvider
from promptphoo.providers.openai.responses import OpenAiResponsesProvider
from promptphoo.tracing.genai_tracer import get_traceparent, with_genai_span
from promptphoo.tracing.otel_api import SpanKind, SpanStatusCode, propagation, trace
from promptphoo.tracing.otel_trace_sdk import (
    BasicTracerProvider,
    InMemorySpanExporter,
    SimpleSpanProcessor,
)
from promptphoo.tracing.target_tracer import with_target_span

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

TRACE_ID = "0123456789abcdef0123456789abcdef"
PARENT_ID = "0123456789abcdef"
CONTEXT: dict[str, Any] = {
    "testIdx": 3,
    "prompt": {"label": "my-prompt", "raw": "hi"},
    "traceparent": f"00-{TRACE_ID}-{PARENT_ID}-01",
    "vars": {},
}
USAGE = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    for name in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_BASE_URL", "OLLAMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(cache, "_enabled", False)
    exp = InMemorySpanExporter()
    BasicTracerProvider(span_processors=[SimpleSpanProcessor(exp)]).register()
    yield exp
    trace.disable()
    propagation.disable()


def _span(exporter: InMemorySpanExporter) -> Any:
    [span] = exporter.get_finished_spans()
    assert span.kind == SpanKind.CLIENT
    # Every provider span joins the evaluation's trace under the traceparent it was handed.
    assert span.span_context().trace_id == TRACE_ID
    assert span.parent_span_context.span_id == PARENT_ID
    assert span.attributes["promptfoo.test.index"] == 3
    assert span.attributes["promptfoo.prompt.label"] == "my-prompt"
    return span


def _usage(span: Any, prompt: int = 5, completion: int = 2, total: int | None = 7) -> None:
    assert span.attributes["gen_ai.usage.input_tokens"] == prompt
    assert span.attributes["gen_ai.usage.output_tokens"] == completion
    if total is not None:
        assert span.attributes["promptfoo.usage.total_tokens"] == total


def _ok(payload: Any) -> httpx.Response:
    return httpx.Response(200, json=payload)


# --- OpenAI ---------------------------------------------------------------------------------


@respx.mock
async def test_openai_chat_span(exporter):
    respx.post(CHAT_URL).mock(
        return_value=_ok({"choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                          "usage": USAGE})
    )  # fmt: skip
    provider = OpenAiChatCompletionProvider(
        "gpt-4o-mini", {"config": {"temperature": 0.5, "max_tokens": 50, "stop": "END"}}
    )
    result = await provider.call_api("Say hi", {**CONTEXT, "evaluationId": "eval-1"})
    assert result["output"] == "Hello"
    span = _span(exporter)
    assert span.name == "chat gpt-4o-mini"
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["openai.api.type"] == "chat_completions"
    assert span.attributes["gen_ai.request.max_tokens"] == 50
    assert span.attributes["gen_ai.request.temperature"] == 0.5
    assert span.attributes["gen_ai.request.stop_sequences"] == ["END"]
    assert span.attributes["promptfoo.eval.id"] == "eval-1"
    assert span.attributes["promptfoo.request.body"] == "Say hi"
    assert span.attributes["promptfoo.response.body"] == "Hello"
    assert span.attributes["gen_ai.response.finish_reasons"] == ["stop"]
    assert span.attributes["promptfoo.cache_hit"] is False
    _usage(span)
    assert span.status.code == SpanStatusCode.OK


async def test_openai_chat_body_failure_still_records_a_span(exporter, monkeypatch):
    provider = OpenAiChatCompletionProvider("gpt-4o-mini")

    async def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("bad response_format")

    monkeypatch.setattr(provider, "get_openai_body", broken)
    with pytest.raises(ValueError, match="bad response_format"):
        await provider.call_api("Say hi", CONTEXT)
    span = _span(exporter)
    assert span.name == "chat gpt-4o-mini"
    assert "gen_ai.request.max_tokens" not in span.attributes
    assert span.status.code == SpanStatusCode.ERROR
    assert span.status.message == "bad response_format"
    assert span.attributes["error.type"] == "ValueError"


@respx.mock
async def test_openai_completion_span(exporter):
    respx.post("https://api.openai.com/v1/completions").mock(
        return_value=_ok({"choices": [{"text": "Hi", "finish_reason": "stop"}], "usage": USAGE})
    )
    provider = OpenAiCompletionProvider(
        "gpt-3.5-turbo-instruct", {"config": {"stop": "END", "presence_penalty": 0.5}}
    )
    await provider.call_api("Say hi", CONTEXT)
    span = _span(exporter)
    assert span.name == "text_completion gpt-3.5-turbo-instruct"
    assert span.attributes["gen_ai.operation.name"] == "text_completion"
    assert span.attributes["gen_ai.request.max_tokens"] == 1024
    assert span.attributes["gen_ai.request.temperature"] == 0
    assert span.attributes["gen_ai.request.top_p"] == 1
    assert span.attributes["gen_ai.request.stop_sequences"] == ["END"]
    assert span.attributes["gen_ai.request.presence_penalty"] == 0.5
    # Upstream's completion provider returns no finishReason.
    assert "gen_ai.response.finish_reasons" not in span.attributes
    _usage(span)


@respx.mock
async def test_openai_responses_span(exporter):
    respx.post("https://api.openai.com/v1/responses").mock(
        return_value=_ok({
            "id": "resp_1", "object": "response", "model": "gpt-4.1", "status": "completed",
            "output": [{"type": "message", "id": "msg_1", "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hi", "annotations": []}]}],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        })
    )  # fmt: skip
    provider = OpenAiResponsesProvider("gpt-4.1", {"config": {"max_output_tokens": 64}})
    result = await provider.call_api("Say hi", CONTEXT)
    assert result["output"] == "Hi"
    span = _span(exporter)
    assert span.name == "chat gpt-4.1"
    assert span.attributes["openai.api.type"] == "responses"
    assert span.attributes["gen_ai.request.max_tokens"] == 64
    assert span.attributes["promptfoo.request.body"] == "Say hi"
    assert span.attributes["promptfoo.response.body"] == "Hi"
    _usage(span)


# --- Anthropic ------------------------------------------------------------------------------


@respx.mock
async def test_anthropic_messages_span(exporter):
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=_ok({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude",
            "content": [{"type": "text", "text": "Hi"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        })
    )  # fmt: skip
    provider = AnthropicMessagesProvider(
        "claude-3-5-haiku-20241022", {"config": {"max_tokens": 100, "temperature": 0.2}}
    )
    result = await provider.call_api("Say hi", CONTEXT)
    assert result["output"] == "Hi"
    span = _span(exporter)
    assert span.name == "chat claude-3-5-haiku-20241022"
    assert span.attributes["gen_ai.provider.name"] == "anthropic"
    assert span.attributes["gen_ai.request.max_tokens"] == 100
    assert span.attributes["gen_ai.request.temperature"] == 0.2
    assert span.attributes["gen_ai.response.finish_reasons"] == ["stop"]
    assert span.attributes["promptfoo.response.body"] == "Hi"
    _usage(span)


# --- Azure ----------------------------------------------------------------------------------


@respx.mock
@pytest.mark.usefixtures(azure_env.__name__)
async def test_azure_chat_span(exporter):
    respx.post(chat_url("my-deployment")).mock(
        return_value=_ok({
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {**USAGE, "completion_tokens_details": {"reasoning_tokens": 1}},
        })
    )  # fmt: skip
    provider = AzureChatCompletionProvider(
        "my-deployment", api_options(temperature=0.3, top_p=0.8, frequency_penalty=0.1)
    )
    result = await provider.call_api("Say hi", CONTEXT)
    assert result["output"] == "Hi"
    span = _span(exporter)
    assert span.name == "chat my-deployment"
    assert span.attributes["gen_ai.provider.name"] == "azure.ai.openai"
    assert span.attributes["gen_ai.request.temperature"] == 0.3
    assert span.attributes["gen_ai.request.top_p"] == 0.8
    assert span.attributes["gen_ai.request.frequency_penalty"] == 0.1
    assert span.attributes["gen_ai.response.finish_reasons"] == ["stop"]
    assert span.attributes["gen_ai.usage.reasoning.output_tokens"] == 1
    assert "promptfoo.request.body" not in span.attributes
    _usage(span)


# --- Bedrock --------------------------------------------------------------------------------


class _FakeBedrock:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def converse(self, **_kwargs: Any) -> dict[str, Any]:
        return self.response


async def test_bedrock_converse_span(exporter):
    provider = AwsBedrockConverseProvider(
        "amazon.nova-micro-v1:0",
        {"config": {"region": "us-east-1", "maxTokens": 77, "temperature": 0.4,
                    "stopSequences": ["STOP"]}},
    )  # fmt: skip
    provider.bedrock = _FakeBedrock({
        "output": {"message": {"role": "assistant", "content": [{"text": "Hi"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
    })  # fmt: skip
    result = await provider.call_api("Say hi", CONTEXT)
    assert result["output"] == "Hi"
    span = _span(exporter)
    assert span.name == "chat amazon.nova-micro-v1:0"
    assert span.attributes["gen_ai.provider.name"] == "aws.bedrock"
    assert span.attributes["gen_ai.request.max_tokens"] == 77
    assert span.attributes["gen_ai.request.temperature"] == 0.4
    assert span.attributes["gen_ai.request.stop_sequences"] == ["STOP"]
    assert span.attributes["gen_ai.response.finish_reasons"] == ["end_turn"]
    _usage(span)


# --- Vertex ---------------------------------------------------------------------------------


@respx.mock
async def test_vertex_gemini_span(exporter, monkeypatch):
    monkeypatch.setenv("VERTEX_API_KEY", "vertex-key")
    for name in ("VERTEX_REGION", "GOOGLE_CLOUD_LOCATION", "VERTEX_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)
    respx.post(
        "https://us-central1-aiplatform.googleapis.com/v1/publishers/google/models/"
        "gemini-2.5-flash:generateContent"
    ).mock(
        return_value=_ok({
            "candidates": [{"content": {"parts": [{"text": "Hi"}]}}],
            "usageMetadata": {"totalTokenCount": 7, "promptTokenCount": 5,
                              "candidatesTokenCount": 2},
        })
    )  # fmt: skip
    provider = VertexChatProvider(
        "gemini-2.5-flash", {"config": {"temperature": 0.6, "maxOutputTokens": 99}}
    )
    result = await provider.call_api("Say hi", CONTEXT)
    assert result["output"] == "Hi"
    span = _span(exporter)
    assert span.name == "chat gemini-2.5-flash"
    assert span.attributes["gen_ai.provider.name"] == "gcp.vertex_ai"
    assert span.attributes["gen_ai.request.temperature"] == 0.6
    assert span.attributes["gen_ai.request.max_tokens"] == 99
    _usage(span)


# --- Ollama ---------------------------------------------------------------------------------

OLLAMA_DONE = {"done": True, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 2}


@respx.mock
async def test_ollama_completion_span(exporter):
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=_ok({"model": "llama3", "response": "Hi", **OLLAMA_DONE})
    )
    provider = OllamaCompletionProvider("llama3", {"config": {"temperature": 0.1,
                                                              "num_predict": 20}})  # fmt: skip
    result = await provider.call_api("Say hi", CONTEXT)
    assert result["output"] == "Hi"
    span = _span(exporter)
    assert span.name == "text_completion llama3"
    assert span.attributes["gen_ai.provider.name"] == "ollama"
    assert span.attributes["gen_ai.request.temperature"] == 0.1
    assert span.attributes["gen_ai.request.max_tokens"] == 20
    assert span.attributes["gen_ai.response.finish_reasons"] == ["stop"]
    _usage(span)


@respx.mock
async def test_ollama_chat_span(exporter):
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=_ok({"model": "llama3", "message": {"role": "assistant", "content": "Hi"},
                          **OLLAMA_DONE})
    )  # fmt: skip
    provider = OllamaChatProvider("llama3", {"config": {"top_p": 0.9, "stop": ["END"]}})
    result = await provider.call_api("Say hi", CONTEXT)
    assert result["output"] == "Hi"
    span = _span(exporter)
    assert span.name == "chat llama3"
    assert span.attributes["gen_ai.request.top_p"] == 0.9
    assert span.attributes["gen_ai.request.stop_sequences"] == ["END"]
    assert span.attributes["gen_ai.response.finish_reasons"] == ["stop"]
    _usage(span)


# --- OpenRouter -----------------------------------------------------------------------------


@respx.mock
async def test_openrouter_span(exporter):
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=_ok({"choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
                          "usage": USAGE})
    )  # fmt: skip
    provider = OpenRouterProvider(
        "meta/llama", {"config": {"apiKey": "router-key", "temperature": 0.9}}
    )
    result = await provider.call_api("Say hi", CONTEXT)
    assert result["output"] == "Hi"
    span = _span(exporter)
    assert span.name == "chat meta/llama"
    assert span.attributes["gen_ai.provider.name"] == "openrouter"
    assert span.attributes["gen_ai.request.temperature"] == 0.9
    assert span.attributes["gen_ai.response.finish_reasons"] == ["stop"]
    assert "openai.api.type" not in span.attributes
    _usage(span)


# --- providerInstrumentation.test.ts --------------------------------------------------------


async def test_model_spans_parent_beneath_the_evaluator_target_span(exporter):
    async def model_call(_span: Any = None) -> Any:
        return await with_genai_span(
            {"system": "openai", "operationName": "chat", "model": "gpt-4",
             "providerId": "openai:gpt-4", "traceparent": get_traceparent()},
            lambda _s: _async({"output": "ok"}),
        )  # fmt: skip

    await with_target_span(
        {"targetType": "provider", "providerId": "openai:gpt-4",
         "traceparent": f"00-{TRACE_ID}-{PARENT_ID}-01"},
        model_call,
    )  # fmt: skip
    spans = {s.name: s for s in exporter.get_finished_spans()}
    target, model = spans["openai:gpt-4"], spans["chat gpt-4"]
    assert target.span_context().trace_id == TRACE_ID
    assert model.span_context().trace_id == TRACE_ID
    assert model.parent_span_context.span_id == target.span_context().span_id


async def _async(value: Any) -> Any:
    return value
