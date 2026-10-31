"""OpenAI embeddings provider. Ports deterministic cases from
test/providers/openai/embedding.test.ts, with HTTP mocked by respx."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from openai_helpers import ok, openai_env, sent_json

from promptphoo.providers.openai import OpenAiEmbeddingProvider
from promptphoo.providers.openai.embedding import decode_base64_embedding

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
GATEWAY_URL = "https://gateway.example/v1/embeddings"
COST_PER_TOKEN = 0.42 / 1e6
INVALID_BASE64_ERROR = (
    "API error: Error: Invalid base64 embedding in OpenAI embeddings API response"
)


def make_provider(**config: Any) -> OpenAiEmbeddingProvider:
    return OpenAiEmbeddingProvider(
        "text-embedding-3-large",
        {"config": {"apiKey": "test-key", "cost": COST_PER_TOKEN, **config}},
    )


async def test_rejects_codex_only_passthrough_model_before_dispatch():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(EMBEDDINGS_URL).mock(return_value=ok({}))
        provider = OpenAiEmbeddingProvider(
            "text-embedding-3-small",
            {"config": {"apiKey": "test-key", "passthrough": {"model": "gpt-5.3-codex-spark"}}},
        )
        with pytest.raises(ValueError, match="only available through openai:codex-sdk"):
            await provider.call_embedding_api("test text")
        assert route.call_count == 0


@respx.mock
async def test_call_embedding_api_successfully():
    respx.post(EMBEDDINGS_URL).mock(
        return_value=ok(
            {
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                "usage": {"total_tokens": 10, "prompt_tokens": 0, "completion_tokens": 0},
            }
        )
    )
    result = await make_provider().call_embedding_api("test text")
    assert result["embedding"] == [0.1, 0.2, 0.3]
    assert result["tokenUsage"] == {"total": 10, "prompt": 0, "completion": 0, "numRequests": 1}
    assert result["cost"] == pytest.approx(10 * COST_PER_TOKEN, abs=1e-12)


@pytest.mark.parametrize("encoding_format", ["float", "base64"])
@respx.mock
async def test_accepts_numeric_embeddings_for_encoding_format(encoding_format):
    route = respx.post(EMBEDDINGS_URL).mock(
        return_value=ok(
            {
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                "usage": {"total_tokens": 10, "prompt_tokens": 10, "completion_tokens": 0},
            }
        )
    )
    provider = OpenAiEmbeddingProvider(
        "text-embedding-3-small",
        {
            "config": {
                "apiKey": "test-key",
                "passthrough": {"dimensions": 8, "encoding_format": encoding_format},
            }
        },
    )
    result = await provider.call_embedding_api("test text")
    assert "error" not in result
    assert result["embedding"] == [0.1, 0.2, 0.3]
    request = route.calls[0].request
    assert request.headers["X-OpenAI-Originator"] == "promptfoo"
    assert request.content.decode() == json.dumps(
        {
            "input": "test text",
            "model": "text-embedding-3-small",
            "dimensions": 8,
            "encoding_format": encoding_format,
        },
        separators=(",", ":"),
    )


@pytest.mark.parametrize(
    ("embedding", "expected"),
    [
        ("AACAPw==", [1]),
        ("AAAAPwAAwL8=", [0.5, -1.5]),
        ("AACAPgAAAL8AAIA/", [0.25, -0.5, 1]),
        ("AACAPw", [1]),
        ("AAAAAAAAAIABAAAA//9/fw==", [0, -0.0, 1.401298464324817e-45, 3.4028234663852886e38]),
    ],
    ids=["two-padding", "one-padding", "no-padding", "omitted-padding", "float32-boundaries"],
)
@respx.mock
async def test_decodes_base64_embeddings(embedding, expected):
    respx.post(EMBEDDINGS_URL).mock(return_value=ok({"data": [{"embedding": embedding}]}))
    result = await make_provider().call_embedding_api("test text")
    assert "error" not in result
    assert result["embedding"] == expected


def test_decode_base64_preserves_negative_zero_sign():
    assert str(decode_base64_embedding("AAAAAAAAAIA=")[1]) == "-0.0"


@respx.mock
async def test_decodes_base64_responses_fresh_and_cached():
    route = respx.post(EMBEDDINGS_URL).mock(
        return_value=ok(
            {
                "data": [{"embedding": "AACAPgAAAL8AAIA/"}],
                "usage": {"total_tokens": 10, "prompt_tokens": 10},
            }
        )
    )
    provider = make_provider()
    fresh = await provider.call_embedding_api("test text")
    cached = await provider.call_embedding_api("test text")
    assert route.call_count == 1
    for result in (fresh, cached):
        assert result["embedding"] == [0.25, -0.5, 1]
        assert isinstance(result["latencyMs"], (int, float))
        assert result["cost"] == pytest.approx(10 * COST_PER_TOKEN, abs=1e-12)
    assert fresh["cached"] is False
    assert fresh["tokenUsage"] == {"total": 10, "prompt": 10, "completion": 0, "numRequests": 1}
    assert cached["cached"] is True
    assert cached["tokenUsage"] == {
        "total": 10,
        "prompt": 10,
        "completion": 0,
        "cached": 10,
        "numRequests": 0,
        "incurredTokenUsage": {},
    }


@pytest.mark.parametrize("item", [{}, {"embedding": None}, {"embedding": ""}])
@respx.mock
async def test_handles_missing_embedding(item):
    respx.post(EMBEDDINGS_URL).mock(return_value=ok({"data": [item]}))
    result = await make_provider().call_embedding_api("test text")
    assert result["error"] == "No embedding found in OpenAI embeddings API response"
    assert "embedding" not in result


@pytest.mark.parametrize(
    "embedding",
    [
        "AACAPw!!",
        "AQID",
        "AA=CAPw==",
        "AACAPw===",
        "AACAPw==AAAA",
        "AADAfw==",
        "AACAfw==",
        "AACA/w==",
    ],
    ids=[
        "invalid-characters",
        "truncated-float32",
        "embedded-padding",
        "excess-padding",
        "trailing-data",
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
@respx.mock
async def test_rejects_invalid_base64_and_evicts_cache_entry(embedding):
    route = respx.post(EMBEDDINGS_URL).mock(
        return_value=ok(
            {
                "data": [{"embedding": embedding}],
                "metadata": {"authorization": "Bearer response-secret-canary"},
            }
        )
    )
    provider = make_provider()
    result = await provider.call_embedding_api("test text")
    assert result["error"] == INVALID_BASE64_ERROR
    assert "embedding" not in result
    # The poisoned entry was deleted, so the next call refetches instead of hitting the cache.
    await provider.call_embedding_api("test text")
    assert route.call_count == 2


@respx.mock
async def test_bills_qualified_passthrough_model_through_custom_gateway():
    route = respx.post(GATEWAY_URL).mock(
        return_value=ok(
            {
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                "usage": {
                    "total_tokens": 1_000_000,
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 0,
                },
            }
        )
    )
    provider = OpenAiEmbeddingProvider(
        "text-embedding-3-large",
        {
            "config": {
                "apiKey": "test-key",
                "apiBaseUrl": "https://gateway.example/v1",
                "passthrough": {"model": "openai/text-embedding-3-small"},
            }
        },
    )
    result = await provider.call_embedding_api("test text")
    assert sent_json(route)["model"] == "openai/text-embedding-3-small"
    assert result["cost"] == pytest.approx(0.02)


@respx.mock
async def test_no_openai_pricing_for_other_gateway_namespace():
    route = respx.post(GATEWAY_URL).mock(
        return_value=ok(
            {
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                "usage": {
                    "total_tokens": 1_000_000,
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 0,
                },
            }
        )
    )
    provider = OpenAiEmbeddingProvider(
        "text-embedding-3-large",
        {
            "config": {
                "apiKey": "test-key",
                "apiBaseUrl": "https://gateway.example/v1",
                "passthrough": {"model": "vendor/text-embedding-3-small"},
            }
        },
    )
    result = await provider.call_embedding_api("test text")
    assert sent_json(route)["model"] == "vendor/text-embedding-3-small"
    assert "cost" not in result


@respx.mock
async def test_handles_api_call_errors():
    respx.post(EMBEDDINGS_URL).mock(side_effect=httpx.ConnectError("API error"))
    result = await make_provider(maxRetries=0).call_embedding_api("test text")
    assert result["error"].startswith("API call error: ")
    assert "API error" in result["error"]
    assert "embedding" not in result


async def test_validates_input_type():
    result = await make_provider().call_embedding_api({"message": "test"})
    assert result["error"] == (
        "Invalid input type for embedding API. Expected string, got object. "
        'Input: {"message":"test"}'
    )
    assert "embedding" not in result


@respx.mock
async def test_handles_http_error_status():
    respx.post(EMBEDDINGS_URL).mock(
        return_value=ok({"error": {"message": "Unauthorized"}}, status=401)
    )
    result = await make_provider().call_embedding_api("test text")
    assert result["error"] == 'API error: 401 Unauthorized\n{"error":{"message":"Unauthorized"}}'
    assert "embedding" not in result


async def test_validates_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    provider = OpenAiEmbeddingProvider("text-embedding-3-large", {"config": {}})
    result = await provider.call_embedding_api("test text")
    assert result["error"] == (
        "API key is not set. Set the OPENAI_API_KEY environment variable or add `apiKey` to the "
        "provider config."
    )
    assert "embedding" not in result
