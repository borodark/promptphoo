"""Ports deterministic cases of test/providers/google/{ai.studio,vertex,util,factory}.test.ts."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from promptphoo import cache
from promptphoo.fetch import FetchWithCacheResult
from promptphoo.providers.google import create_provider
from promptphoo.providers.google.ai_studio import (
    AIStudioChatProvider,
    AIStudioEmbeddingProvider,
)
from promptphoo.providers.google.auth import GoogleAuthManager
from promptphoo.providers.google.util import (
    calculate_google_cost,
    gemini_format_and_system_instructions,
    maybe_coerce_to_gemini_format,
    sanitize_schema_for_gemini,
    validate_function_call,
)
from promptphoo.providers.google.vertex import VertexChatProvider, VertexEmbeddingProvider

KEY_ENV = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "PALM_API_KEY",
    "VERTEX_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "VERTEX_PROJECT_ID",
    "GOOGLE_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "VERTEX_REGION",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_API_HOST",
    "GOOGLE_API_BASE_URL",
    "VERTEX_API_HOST",
    "GOOGLE_APPLICATION_CREDENTIALS",
)
KEY = {"config": {"apiKey": "test-key"}}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in KEY_ENV:
        monkeypatch.delenv(name, raising=False)
    GoogleAuthManager.clear_cache()


class FakeFetch:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.result: Any = None

    def respond(self, data: Any, cached: bool = False, headers: dict | None = None) -> None:
        self.result = FetchWithCacheResult(
            data=data, cached=cached, status=200, status_text="OK", headers=headers or {}
        )

    async def __call__(self, url, options=None, timeout=None, format="json", bust=False, **kw):
        self.calls.append((url, options, timeout, format, bust))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.calls[-1][1]["body"])


@pytest.fixture
def fetch(monkeypatch):
    fake = FakeFetch()
    monkeypatch.setattr("promptphoo.fetch.fetch_with_cache", fake)
    return fake


def _gemini(text: str = "response text", usage: dict | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    if usage is not None:
        data["usageMetadata"] = usage
    return data


# --- AI Studio: configuration ------------------------------------------------------------------


def test_api_key_sources(monkeypatch):
    assert AIStudioChatProvider("gemini-pro", KEY).get_api_key() == "test-key"
    env_provider = AIStudioChatProvider("gemini-pro", {"env": {"GOOGLE_API_KEY": "env-key"}})
    assert env_provider.get_api_key() == "env-key"
    assert AIStudioChatProvider("gemini-pro").get_api_key() is None
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert AIStudioChatProvider("gemini-pro").get_api_key() == "gemini-key"
    templated = AIStudioChatProvider("gemini-pro", {"config": {"apiKey": "{{ 'k' ~ 1 }}"}})
    assert templated.get_api_key() == "k1"


def test_endpoints():
    provider = AIStudioChatProvider("gemini-pro", KEY)
    assert provider.get_api_endpoint("generateContent") == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    )
    custom = AIStudioChatProvider("gemini-pro", {"config": {"apiHost": "custom.host.com"}})
    assert "custom.host.com" in custom.get_api_endpoint("generateContent")
    base = AIStudioChatProvider("gemini-pro", {"config": {"apiBaseUrl": "https://base.url.com"}})
    assert base.get_api_endpoint("generateContent").startswith("https://base.url.com/v1beta/")
    both = AIStudioChatProvider(
        "gemini-pro",
        {"config": {"apiHost": "host.googleapis.com", "apiBaseUrl": "https://base.googleapis.com"}},
    )
    endpoint = both.get_api_endpoint("generateContent")
    assert "host.googleapis.com" in endpoint and "base.googleapis.com" not in endpoint
    thinking = AIStudioChatProvider("gemini-2.0-flash-thinking-exp", KEY)
    assert "/v1alpha/models/gemini-2.0-flash-thinking-exp:" in thinking.get_api_endpoint("x")
    pinned = AIStudioChatProvider("gemini-pro", {"config": {"apiVersion": "v1"}})
    assert "/v1/models/gemini-pro:" in pinned.get_api_endpoint("x")


def test_ids():
    assert AIStudioChatProvider("gemini-pro").id() == "google:gemini-pro"
    assert AIStudioChatProvider("gemini-pro", {"id": "custom"}).id() == "custom"
    assert str(AIStudioChatProvider("gemini-pro")) == (
        "[Google Google AI Studio Provider gemini-pro]"
    )


async def test_missing_api_key_raises():
    with pytest.raises(ValueError, match="Google API key is not set. Set the GOOGLE_API_KEY"):
        await AIStudioChatProvider("gemini-pro").call_api("test")


# --- AI Studio: responses ----------------------------------------------------------------------


async def test_gemini_response_with_token_usage(fetch):
    usage = {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
    fetch.respond(_gemini(usage=usage))
    response = await AIStudioChatProvider("gemini-pro", KEY).call_api("test prompt")
    assert response["output"] == "response text"
    assert response["tokenUsage"] == {"prompt": 10, "completion": 5, "total": 15, "numRequests": 1}
    assert response["cost"] == pytest.approx(0.0000125, abs=1e-12)
    assert response["cached"] is False and response["metadata"] == {}
    url, options, _timeout, fmt, bust = fetch.calls[0]
    assert url.endswith("v1beta/models/gemini-pro:generateContent")
    assert (options["method"], fmt, bust) == ("POST", "json", False)
    assert options["headers"]["x-goog-api-key"] == "test-key"
    assert "key=" not in url
    assert '"contents":[{"parts":[{"text":"test prompt"}],"role":"user"}]' in options["body"]


async def test_gemini_cached_response(fetch):
    fetch.respond(_gemini("cached response", {"totalTokenCount": 15}), cached=True)
    response = await AIStudioChatProvider("gemini-pro", KEY).call_api("p")
    assert response["output"] == "cached response" and response["cached"] is True
    assert response["tokenUsage"] == {
        "total": 15,
        "cached": 15,
        "numRequests": 0,
        "incurredTokenUsage": {},
    }


async def test_gemini_errors(fetch):
    fetch.respond({"candidates": []})
    response = await AIStudioChatProvider("gemini-pro", KEY).call_api("p")
    assert "No candidates returned in API response" in response["error"]
    assert "Got response:" in response["error"]

    fetch.respond({"candidates": [{"content": {"parts": None}}]})
    response = await AIStudioChatProvider("gemini-pro", KEY).call_api("p")
    assert response == {"error": 'Error: No output found in response: {"content":{"parts":null}}'}

    fetch.result = RuntimeError("API call failed")
    response = await AIStudioChatProvider("gemini-pro", KEY).call_api("p")
    assert response == {"error": "API call error: Error: API call failed"}


async def test_gemini_prompt_feedback_blocked(fetch):
    fetch.respond({
        "candidates": [],
        "promptFeedback": {
            "blockReason": "PROHIBITED_CONTENT",
            "safetyRatings": [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "HIGH"},
                {"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"},
            ],
        },
    })  # fmt: skip
    error = (await AIStudioChatProvider("gemini-pro", KEY).call_api("p"))["error"]
    assert "Response blocked: PROHIBITED_CONTENT" in error
    assert "Safety ratings: HARM_CATEGORY_HATE_SPEECH: HIGH)" in error


async def test_gemini_finish_reason_blocked(fetch):
    fetch.respond({
        "candidates": [{
            "content": {"parts": [{"text": ""}]},
            "finishReason": "RECITATION",
            "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                               "probability": "MEDIUM", "blocked": False}],
        }]
    })  # fmt: skip
    error = (await AIStudioChatProvider("gemini-pro", KEY).call_api("p"))["error"]
    assert "Response was blocked with finish reason: RECITATION" in error
    assert "too similar to content from the model's training data" in error
    assert "HARM_CATEGORY_DANGEROUS_CONTENT: MEDIUM" in error


async def test_gemini_accumulates_array_chunks(fetch):
    fetch.respond([
        {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": " world"}]}}]},
        {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2,
                           "totalTokenCount": 3}},
    ])  # fmt: skip
    response = await AIStudioChatProvider("gemini-pro", KEY).call_api("p")
    assert response["output"] == "Hello world"
    assert response["tokenUsage"]["total"] == 3


async def test_thinking_tokens_usage_and_cost(fetch):
    usage = {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30,
             "thoughtsTokenCount": 50}  # fmt: skip
    fetch.respond(_gemini("response with thinking", usage))
    response = await AIStudioChatProvider("gemini-2.5-flash", KEY).call_api("p")
    assert response["tokenUsage"] == {
        "prompt": 10,
        "completion": 20,
        "total": 30,
        "numRequests": 1,
        "completionDetails": {"reasoning": 50, "acceptedPrediction": 0, "rejectedPrediction": 0},
    }
    usage = {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 315,
             "thoughtsTokenCount": 300}  # fmt: skip
    fetch.respond(_gemini("x", usage))
    response = await AIStudioChatProvider("gemini-2.5-flash", KEY).call_api("p")
    # completionForCost = 5 + 300 = 305; cost = 0.3e-6 * 10 + 2.5e-6 * 305
    assert response["cost"] == pytest.approx(0.0007655, abs=1e-12)


async def test_thought_parts_are_included_in_output(fetch):
    fetch.respond({
        "candidates": [{"content": {"parts": [
            {"text": "Thinking...", "thought": True},
            {"text": "Answer", "thoughtSignature": "sig-1"},
        ]}}]
    })  # fmt: skip
    response = await AIStudioChatProvider("gemini-2.5-flash", KEY).call_api("p")
    assert response["output"] == "Thinking...Answer"
    assert response["metadata"]["thoughtSignatures"]


async def test_request_body_config(fetch):
    fetch.respond(_gemini())
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    provider = AIStudioChatProvider(
        "gemini-2.5-flash",
        {"config": {
            "apiKey": "k",
            "temperature": 0.2,
            "maxOutputTokens": 64,
            "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT",
                                "threshold": "BLOCK_MEDIUM_AND_ABOVE"}],
            "responseSchema": json.dumps(schema),
            "systemInstruction": "Be brief.",
            "tools": [{"functionDeclarations": [{"name": "get_weather"}]}],
        }},
    )  # fmt: skip
    await provider.call_api("hi")
    body = fetch.body
    assert body["generationConfig"]["temperature"] == 0.2
    assert body["generationConfig"]["maxOutputTokens"] == 64
    assert body["generationConfig"]["response_schema"] == schema
    assert body["generationConfig"]["response_mime_type"] == "application/json"
    assert body["safetySettings"] == [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    ]
    assert body["system_instruction"] == {"parts": [{"text": "Be brief."}]}
    assert body["tools"] == [{"functionDeclarations": [{"name": "get_weather"}]}]


async def test_function_call_output(fetch):
    fetch.respond({
        "candidates": [{"content": {"parts": [
            {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
        ]}}]
    })  # fmt: skip
    response = await AIStudioChatProvider("gemini-2.5-flash", KEY).call_api("p")
    assert response["output"] == [
        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
    ]


async def test_bust_cache_forwarded(fetch):
    fetch.respond(_gemini())
    await AIStudioChatProvider("gemini-pro", KEY).call_api("p", {"bustCache": True, "vars": {}})
    assert fetch.calls[0][4] is True


# --- AI Studio: embeddings ---------------------------------------------------------------------


async def test_ai_studio_embedding(fetch, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    provider = AIStudioEmbeddingProvider("gemini-embedding-001")
    assert provider.id() == "google:embedding:gemini-embedding-001"
    assert str(provider) == "[Google AI Studio Embedding Provider gemini-embedding-001]"
    assert AIStudioEmbeddingProvider("x", {"id": "custom-embed"}).id() == "custom-embed"

    fetch.respond(
        {"embedding": {"values": [0.1, 0.2, 0.3]}, "usageMetadata": {"promptTokenCount": 7}}
    )
    response = await provider.call_embedding_api("hello world")
    url, options, *_ = fetch.calls[0]
    assert url.endswith("/v1beta/models/gemini-embedding-001:embedContent")
    assert fetch.body == {"content": {"parts": [{"text": "hello world"}]}}
    assert options["headers"]["x-goog-api-key"] == "test-key"
    assert response["embedding"] == [0.1, 0.2, 0.3]
    assert response["tokenUsage"] == {"total": 7, "numRequests": 1}

    fetch.respond({"embedding": {"values": [0.1]}, "usageMetadata": {"promptTokenCount": 10_000}})
    response = await AIStudioEmbeddingProvider("embedding-2-preview").call_embedding_api("x")
    assert response["cost"] == pytest.approx(0.002, abs=1e-12)

    response = await provider.call_embedding_api(123)
    assert "Invalid input type for embedding API" in response["error"]

    fetch.respond({"error": {"message": "bad"}})
    assert "No embedding found" in (await provider.call_embedding_api("x"))["error"]

    monkeypatch.delenv("GOOGLE_API_KEY")
    response = await AIStudioEmbeddingProvider("gemini-embedding-001").call_embedding_api("x")
    assert "Google API key is not set" in response["error"]


# --- Vertex ------------------------------------------------------------------------------------


class FakeCredentials:
    def __init__(self) -> None:
        self.token = "oauth-token"
        self.valid = True
        self.quota_project_id = None

    def refresh(self, request: Any) -> None:  # pragma: no cover - token is always valid
        self.valid = True


@pytest.fixture
def adc(monkeypatch):
    google_auth = pytest.importorskip("google.auth")

    calls: list[Any] = []

    def fake_default(scopes=None, **kwargs):
        calls.append(scopes)
        return FakeCredentials(), "adc-project"

    monkeypatch.setattr(google_auth, "default", fake_default)
    return calls


# Without an API key, Vertex defaults to the global region (SDK alignment).
VERTEX_URL = (
    "https://aiplatform.googleapis.com/v1/projects/test-project/locations/"
    "global/publishers/google/models/gemini-pro:generateContent"
)


@respx.mock
async def test_vertex_oauth_gemini(adc):
    route = respx.post(VERTEX_URL).mock(
        return_value=httpx.Response(
            200,
            json=[{
                "candidates": [{"content": {"parts": [{"text": "response text"}]}}],
                "usageMetadata": {"totalTokenCount": 10, "promptTokenCount": 5,
                                  "candidatesTokenCount": 5},
            }],
        )
    )  # fmt: skip
    provider = VertexChatProvider("gemini-pro", {"config": {"projectId": "test-project"}})
    assert provider.id() == "vertex:gemini-pro"
    with cache.with_cache_enabled(False):
        response = await provider.call_api("test prompt")
    assert response == {
        "cached": False,
        "output": "response text",
        "tokenUsage": {"total": 10, "prompt": 5, "completion": 5},
        "cost": pytest.approx(0.00001, abs=1e-12),
        "metadata": {},
    }
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer oauth-token"
    assert json.loads(request.content)["contents"] == [
        {"parts": [{"text": "test prompt"}], "role": "user"}
    ]
    assert adc == [["https://www.googleapis.com/auth/cloud-platform"]]


@respx.mock
async def test_vertex_project_from_adc(adc):
    url = VERTEX_URL.replace("test-project", "adc-project")
    respx.post(url).mock(return_value=httpx.Response(200, json=_gemini("ok")))
    with cache.with_cache_enabled(False):
        response = await VertexChatProvider("gemini-pro").call_api("p")
    assert response["output"] == "ok"


@respx.mock
async def test_vertex_express_mode(monkeypatch):
    monkeypatch.setenv("VERTEX_API_KEY", "vertex-key")
    provider = VertexChatProvider("gemini-2.5-flash")
    assert provider.is_express_mode()
    # An API key without a configured region defaults to us-central1 for express mode.
    url = (
        "https://us-central1-aiplatform.googleapis.com/v1/publishers/google/models/"
        "gemini-2.5-flash:generateContent"
    )
    route = respx.post(url).mock(return_value=httpx.Response(200, json=_gemini("express")))
    with cache.with_cache_enabled(False):
        response = await provider.call_api("p")
    assert response["output"] == "express"
    assert route.calls[0].request.headers["x-goog-api-key"] == "vertex-key"

    respx.post(url).mock(return_value=httpx.Response(403, json={"error": {"message": "denied"}}))
    with cache.with_cache_enabled(False):
        response = await provider.call_api("p2")
    assert response == {"error": 'API call error: 403 Forbidden: {"error":{"message":"denied"}}'}

    disabled = VertexChatProvider("gemini-2.5-flash", {"config": {"expressMode": False}})
    assert not disabled.is_express_mode()


def test_vertex_region_resolution(monkeypatch):
    assert VertexChatProvider("gemini-pro").get_region() == "global"
    assert VertexChatProvider("gemini-pro", KEY).get_region() == "us-central1"
    regional = VertexChatProvider("gemini-pro", {"config": {"region": "europe-west4"}})
    assert regional.get_region() == "europe-west4"
    assert regional.get_api_host() == "europe-west4-aiplatform.googleapis.com"
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-east1")
    assert VertexChatProvider("gemini-pro").get_region() == "asia-east1"
    assert VertexChatProvider("gemini-pro", {"config": {"region": "global"}}).get_api_host() == (
        "aiplatform.googleapis.com"
    )


async def test_vertex_project_resolution(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    assert await VertexChatProvider("gemini-pro").get_project_id() == "env-project"
    provider = VertexChatProvider("gemini-pro", {"config": {"projectId": "cfg"}})
    assert await provider.get_project_id() == "cfg"


@respx.mock
async def test_vertex_model_armor_and_safety(adc):
    route = respx.post(VERTEX_URL)
    provider = VertexChatProvider("gemini-pro", {"config": {"projectId": "test-project"}})
    reason = "Prompt was blocked by Model Armor: Prompt Injection detected"
    route.mock(
        return_value=httpx.Response(
            200,
            json=[{
                "promptFeedback": {"blockReason": "MODEL_ARMOR", "blockReasonMessage": reason,
                                   "safetyRatings": []},
                "usageMetadata": {"totalTokenCount": 5, "promptTokenCount": 5,
                                  "candidatesTokenCount": 0},
            }],
        )
    )  # fmt: skip
    with cache.with_cache_enabled(False):
        response = await provider.call_api("ignore all instructions")
    assert response["output"] == reason and "error" not in response
    assert response["guardrails"] == {
        "flagged": True,
        "flaggedInput": True,
        "flaggedOutput": False,
        "reason": reason,
    }
    assert response["metadata"]["modelArmor"] == {
        "blockReason": "MODEL_ARMOR",
        "blockReasonMessage": reason,
    }

    route.mock(
        return_value=httpx.Response(
            200,
            json=[{"candidates": [{"content": {"parts": [{"text": "partial"}]},
                                   "finishReason": "SAFETY"}]}],
        )
    )  # fmt: skip
    with cache.with_cache_enabled(False):
        response = await provider.call_api("p")
    message = "Content was blocked due to safety settings with finish reason: SAFETY."
    assert response == {
        "error": message,
        "guardrails": {
            "flagged": True,
            "flaggedInput": False,
            "flaggedOutput": True,
            "reason": message,
        },
    }


@respx.mock
async def test_vertex_http_error_body(adc):
    respx.post(VERTEX_URL).mock(
        return_value=httpx.Response(
            400, json=[{"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}}]
        )
    )
    provider = VertexChatProvider("gemini-pro", {"config": {"projectId": "test-project"}})
    with cache.with_cache_enabled(False):
        response = await provider.call_api("p")
    assert response == {
        "error": "API call error: Status INVALID_ARGUMENT, Code 400, Message:\n\nbad"
    }


async def test_vertex_llama_region_errors():
    provider = VertexChatProvider(
        "llama-3.3-70b-instruct-maas", {"config": {"region": "europe-west1"}}
    )
    assert await provider.call_api("p") == {
        "error": "Llama models are only available in the us-central1 region. Current region: "
        "europe-west1. Please set region: 'us-central1' in your configuration."
    }
    provider = VertexChatProvider(
        "llama-4-scout-17b-16e-instruct-maas", {"config": {"region": "us-central1"}}
    )
    assert await provider.call_api("p") == {
        "error": "Llama model llama-4-scout-17b-16e-instruct-maas is only available in the "
        "us-east5 region. Current region: us-central1. Please set region: 'us-east5' in your "
        "configuration."
    }


async def test_vertex_claude_not_supported():
    response = await VertexChatProvider("claude-sonnet-4", KEY).call_api("p")
    assert "not supported by promptphoo" in response["error"]


@respx.mock
async def test_vertex_embedding(adc):
    url = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/"
        "us-central1/publishers/google/models/text-embedding-004:predict"
    )
    route = respx.post(url).mock(
        return_value=httpx.Response(
            200, json={"predictions": [{"embeddings": {"values": [0.5, 0.25],
                                                       "statistics": {"token_count": 3}}}]}
        )
    )  # fmt: skip
    provider = VertexEmbeddingProvider(
        "text-embedding-004", {"config": {"projectId": "test-project", "region": "us-central1"}}
    )
    assert provider.id() == "vertex:text-embedding-004"
    response = await provider.call_embedding_api("hello")
    assert response["embedding"] == [0.5, 0.25]
    assert json.loads(route.calls[0].request.content)["instances"][0]["content"] == "hello"


# --- util --------------------------------------------------------------------------------------


def test_maybe_coerce_to_gemini_format():
    assert maybe_coerce_to_gemini_format("test message") == {
        "contents": [{"parts": [{"text": "test message"}]}],
        "coerced": True,
        "systemInstruction": None,
    }
    chat = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    assert maybe_coerce_to_gemini_format(chat)["contents"] == [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi there"}]},
    ]
    assert maybe_coerce_to_gemini_format(chat, True)["contents"][1]["role"] == "assistant"
    mixed = [{"role": "user", "content": ["Hello", {"type": "text", "text": "World"}]}]
    assert maybe_coerce_to_gemini_format(mixed)["contents"] == [
        {"role": "user", "parts": [{"text": "Hello"}, {"text": "World"}]}
    ]
    system = [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "User message"},
    ]
    assert maybe_coerce_to_gemini_format(system) == {
        "contents": [{"role": "user", "parts": [{"text": "User message"}]}],
        "coerced": True,
        "systemInstruction": {"parts": [{"text": "System instruction"}]},
    }
    only_system = [
        {"role": "system", "content": "First instruction."},
        {"role": "system", "content": "Second instruction."},
    ]
    assert maybe_coerce_to_gemini_format(only_system)["contents"] == [
        {"role": "user", "parts": [{"text": "First instruction."}, {"text": "Second instruction."}]}
    ]
    assert maybe_coerce_to_gemini_format({"parts": [{"text": "test"}]})["contents"] == [
        {"parts": [{"text": "test"}]}
    ]
    unknown = maybe_coerce_to_gemini_format({"unknownFormat": "test"})
    assert unknown["contents"] == [] and unknown["coerced"] is False
    native = [{"role": "user", "parts": [{"text": "x"}]}]
    assert maybe_coerce_to_gemini_format(native)["contents"] == native


def test_gemini_format_and_system_instructions():
    prompt = json.dumps([
        {"role": "system", "content": "prompt system instruction"},
        {"role": "user", "content": "user message"},
    ])  # fmt: skip
    result = gemini_format_and_system_instructions(prompt, {}, "config system instruction")
    assert result["contents"] == [{"parts": [{"text": "user message"}], "role": "user"}]
    assert result["systemInstruction"] == {
        "parts": [{"text": "config system instruction"}, {"text": "prompt system instruction"}]
    }
    result = gemini_format_and_system_instructions(
        json.dumps([{"role": "user", "parts": [{"text": "m"}]}]),
        {},
        {"parts": [{"text": "system instruction"}]},
    )
    assert result["systemInstruction"] == {"parts": [{"text": "system instruction"}]}


FUNCTIONS = [{"functionDeclarations": [
    {"name": "testFunction", "parameters": {"type": "OBJECT", "properties": {
        "param1": {"type": "STRING"}}, "required": ["param1"]}},
    {"name": "emptyFunction"},
]}]  # fmt: skip


def test_validate_function_call():
    validate_function_call(
        [{"functionCall": {"name": "testFunction", "args": '{"param1": "test"}'}}], FUNCTIONS
    )
    validate_function_call([{"functionCall": {"name": "emptyFunction", "args": "{}"}}], FUNCTIONS)
    with pytest.raises(Exception, match="Google did not return a valid-looking function call"):
        validate_function_call({"invalidFormat": True}, FUNCTIONS)
    with pytest.raises(
        Exception, match='Called "nonExistentFunction", but there is no function with that name'
    ):
        validate_function_call(
            [{"functionCall": {"name": "nonExistentFunction", "args": "{}"}}], FUNCTIONS
        )
    with pytest.raises(Exception, match="does not match schema"):
        validate_function_call(
            [{"functionCall": {"name": "testFunction", "args": "{}"}}], FUNCTIONS
        )


def test_sanitize_schema_for_gemini():
    schema = {
        "type": "object",
        "$schema": "http://json-schema.org/draft-07/schema#",
        "additionalProperties": False,
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    }
    sanitized = sanitize_schema_for_gemini(schema)
    assert "$schema" not in sanitized and "additionalProperties" not in sanitized
    assert sanitized["properties"]["tags"]["items"]["type"] in ("string", "STRING")


def test_calculate_google_cost():
    assert calculate_google_cost("gemini-2.5-flash", {}, 10, 305) == pytest.approx(
        0.3e-6 * 10 + 2.5e-6 * 305
    )
    assert calculate_google_cost("unknown-model", {}, 10, 10) is None
    assert calculate_google_cost("gemini-2.5-flash", {"cost": 1e-6}, 10, 10) == pytest.approx(2e-5)


# --- factory -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,cls,model",
    [("google:gemini-2.5-flash", AIStudioChatProvider, "gemini-2.5-flash"),
     ("google:embedding:gemini-embedding-001", AIStudioEmbeddingProvider, "gemini-embedding-001"),
     ("vertex:gemini-2.5-pro", VertexChatProvider, "gemini-2.5-pro"),
     ("vertex:chat:gemini-2.5-pro", VertexChatProvider, "gemini-2.5-pro"),
     ("vertex:embedding:text-embedding-004", VertexEmbeddingProvider, "text-embedding-004")],
)  # fmt: skip
def test_factory_routes(path, cls, model):
    provider = create_provider(path, {})
    assert type(provider) is cls and provider.model_name == model


@pytest.mark.parametrize(
    "path",
    ["google:live:gemini-2.0-flash-live", "google:image:imagen-4", "google:video:veo-3",
     "vertex:video:veo-3", "google:gemini-2.5-flash-image"],
)  # fmt: skip
def test_factory_unsupported(path):
    with pytest.raises(ValueError, match="not supported by promptphoo"):
        create_provider(path, {})


def test_factory_live_only_model_requires_live_route():
    with pytest.raises(ValueError, match="requires the Gemini Live API"):
        create_provider("google:gemini-3.5-live-translate-preview", {})
