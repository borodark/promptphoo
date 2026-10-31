"""OpenAI moderation provider. Ports deterministic cases from
test/providers/openai/moderation.test.ts, with HTTP mocked by respx and the real in-memory
cache (wrapped to record the keys it is asked for)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx
import pytest
import respx
from openai_helpers import ok, openai_env, sent_json

from promptphoo import cache
from promptphoo.providers.openai import OpenAiModerationProvider
from promptphoo.providers.openai.moderation import (
    format_moderation_input,
    is_image_input,
    is_text_input,
    supports_image_input,
)

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

MODERATIONS_URL = "https://api.openai.com/v1/moderations"
CACHE_KEY_RE = re.compile(
    r"^openai:moderation:omni-moderation-latest:[a-f0-9]{64}:[a-f0-9]{64}:[a-f0-9]{64}$"
)
IMAGE_INPUT = {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}


def missing_key_message(env_var: str = "OPENAI_API_KEY") -> str:
    return (
        f"API key is not set. Set the {env_var} environment variable or add `apiKey` to the "
        "provider config."
    )


def moderation_response(results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if results is None:
        results = [{"flagged": False, "categories": {}, "category_scores": {}}]
    return {"id": "modr-123", "model": "omni-moderation-latest", "results": results}


def make_provider(model: str = "omni-moderation-latest", **config: Any):
    return OpenAiModerationProvider(model, {"config": {"apiKey": "test-key", **config}})


class CacheRecorder:
    """Records the keys passed to the active cache's get/set."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.gets: list[str] = []
        self.sets: list[tuple[str, Any]] = []
        instance = cache.get_cache()
        original_get, original_set = instance.get, instance.set

        async def get(key: str, *args: Any, **kwargs: Any) -> Any:
            self.gets.append(key)
            return await original_get(key, *args, **kwargs)

        async def set_(key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
            self.sets.append((key, value))
            return await original_set(key, value, *args, **kwargs)

        monkeypatch.setattr(instance, "get", get)
        monkeypatch.setattr(instance, "set", set_)


@pytest.fixture
def recorder(openai_env: pytest.MonkeyPatch) -> CacheRecorder:
    return CacheRecorder(openai_env)


def test_registry_excludes_shut_down_text_moderation_models():
    assert OpenAiModerationProvider.MODERATION_MODEL_IDS == [
        "omni-moderation-latest",
        "omni-moderation-2024-09-26",
    ]
    for retired in ("text-moderation-latest", "text-moderation-stable", "text-moderation-007"):
        assert retired not in OpenAiModerationProvider.MODERATION_MODEL_IDS


@pytest.mark.parametrize("model", ["gpt-transcribe", "gpt-live-transcribe", "gpt-5.3-codex-spark"])
def test_rejects_unsupported_first_party_model(model):
    with pytest.raises(ValueError):
        OpenAiModerationProvider(model, {"config": {"apiKey": "test-key"}})


def test_rejects_retired_first_party_model():
    with pytest.raises(ValueError, match="has been retired"):
        OpenAiModerationProvider("text-moderation-latest", {"config": {"apiKey": "test-key"}})


@respx.mock
async def test_defaults_to_current_multimodal_model():
    route = respx.post(MODERATIONS_URL).mock(return_value=ok(moderation_response()))
    provider = OpenAiModerationProvider(options={"config": {"apiKey": "test-key"}})
    result = await provider.call_moderation_api("", "A friendly greeting")
    assert result == {"flags": []}
    assert provider.modelName == "omni-moderation-latest"
    assert route.calls[0].request.content.decode() == json.dumps(
        {
            "model": "omni-moderation-latest",
            "input": [{"type": "text", "text": "A friendly greeting"}],
        },
        separators=(",", ":"),
    )


@respx.mock
async def test_moderates_and_detects_harmful_content():
    route = respx.post(MODERATIONS_URL).mock(
        return_value=ok(
            moderation_response(
                [
                    {
                        "flagged": True,
                        "categories": {"hate": True, "hate/threatening": False},
                        "category_scores": {"hate": 0.99, "hate/threatening": 0.01},
                    }
                ]
            )
        )
    )
    result = await make_provider().call_moderation_api("user input", "assistant response")
    assert result == {"flags": [{"code": "hate", "description": "hate", "confidence": 0.99}]}
    headers = route.calls[0].request.headers
    assert headers["Content-Type"] == "application/json"
    assert headers["x-promptfoo-silent"] == "true"
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["X-OpenAI-Originator"] == "promptfoo"
    assert sent_json(route)["model"] == "omni-moderation-latest"


@respx.mock
async def test_empty_flags_for_safe_content():
    respx.post(MODERATIONS_URL).mock(
        return_value=ok(
            moderation_response(
                [
                    {
                        "flagged": False,
                        "categories": {"hate": False, "hate/threatening": False},
                        "category_scores": {"hate": 0.01, "hate/threatening": 0.01},
                    }
                ]
            )
        )
    )
    assert await make_provider().call_moderation_api("user input", "assistant response") == {
        "flags": []
    }


@respx.mock
async def test_uses_flagged_categories_when_scores_are_close():
    respx.post(MODERATIONS_URL).mock(
        return_value=ok(
            moderation_response(
                [
                    {
                        "flagged": True,
                        "categories": {"hate": True, "hate/threatening": False},
                        "category_scores": {"hate": 0.5, "hate/threatening": 0.49},
                    }
                ]
            )
        )
    )
    result = await make_provider().call_moderation_api("user input", "assistant response")
    assert result == {"flags": [{"code": "hate", "description": "hate", "confidence": 0.5}]}


@respx.mock
async def test_keeps_highest_confidence_across_results():
    respx.post(MODERATIONS_URL).mock(
        return_value=ok(
            moderation_response(
                [
                    {
                        "flagged": True,
                        "categories": {"hate": True},
                        "category_scores": {"hate": 0.4},
                    },
                    {
                        "flagged": True,
                        "categories": {"hate": True},
                        "category_scores": {"hate": 0.8},
                    },
                ]
            )
        )
    )
    result = await make_provider().call_moderation_api("user", "assistant")
    assert result == {"flags": [{"code": "hate", "description": "hate", "confidence": 0.8}]}


@respx.mock
async def test_handles_api_call_errors():
    respx.post(MODERATIONS_URL).mock(side_effect=httpx.ConnectError("API Error"))
    result = await make_provider(maxRetries=0).call_moderation_api("user input", "response")
    assert list(result) == ["error"]
    assert "API call error" in result["error"]


@respx.mock
async def test_handles_error_responses_from_api():
    respx.post(MODERATIONS_URL).mock(return_value=ok({"error": "Invalid request"}, status=400))
    result = await make_provider().call_moderation_api("user input", "assistant response")
    assert result == {"error": 'API error: 400 Bad Request: {"error":"Invalid request"}'}


async def test_checks_for_api_key(monkeypatch: pytest.MonkeyPatch, caplog):
    provider = make_provider()
    monkeypatch.setattr(provider, "get_api_key", lambda config=None: "")
    with caplog.at_level(logging.ERROR):
        result = await provider.call_moderation_api("user", "assistant")
    assert result == {"error": f"API call error: {missing_key_message()}"}
    assert any("OpenAI moderation API error" in m for m in caplog.messages)


async def test_custom_api_key_envar_in_missing_key_error(monkeypatch: pytest.MonkeyPatch):
    provider = OpenAiModerationProvider(
        "omni-moderation-latest",
        {"config": {"apiKeyEnvar": "CUSTOM_MODERATION_API_KEY"}, "env": {}},
    )
    monkeypatch.setattr(provider, "get_api_key", lambda config=None: "")
    result = await provider.call_moderation_api("user", "assistant")
    assert missing_key_message("CUSTOM_MODERATION_API_KEY") in result["error"]


@respx.mock
async def test_handles_empty_results():
    respx.post(MODERATIONS_URL).mock(return_value=ok(moderation_response([])))
    assert await make_provider().call_moderation_api("user", "assistant") == {"flags": []}


@respx.mock
async def test_handles_non_empty_results_without_flags():
    categories = [
        "hate",
        "hate/threatening",
        "harassment",
        "harassment/threatening",
        "illicit",
        "illicit/violent",
        "self-harm",
        "self-harm/intent",
        "self-harm/instructions",
        "sexual",
        "sexual/minors",
        "violence",
        "violence/graphic",
    ]
    respx.post(MODERATIONS_URL).mock(
        return_value=ok(
            moderation_response(
                [
                    {
                        "flagged": False,
                        "categories": dict.fromkeys(categories, False),
                        "category_scores": dict.fromkeys(categories, 0),
                        "category_applied_input_types": {},
                    }
                ]
            )
        )
    )
    assert await make_provider().call_moderation_api("user", "assistant") == {"flags": []}


# --- Caching --------------------------------------------------------------------------------


@respx.mock
async def test_uses_cache_when_enabled(recorder: CacheRecorder):
    header_secret = "Bearer cache-secret-header"
    route = respx.post(MODERATIONS_URL).mock(
        return_value=ok(
            moderation_response(
                [{"flagged": True, "categories": {"hate": True}, "category_scores": {"hate": 0.9}}]
            )
        )
    )
    provider = make_provider(headers={"Authorization": header_secret})
    first = await provider.call_moderation_api("user input", "assistant response")
    second = await provider.call_moderation_api("user input", "assistant response")
    flags = [{"code": "hate", "description": "hate", "confidence": 0.9}]
    assert first == {"flags": flags}
    assert second == {"flags": flags, "cached": True}
    assert route.call_count == 1
    key = recorder.gets[0]
    assert CACHE_KEY_RE.match(key)
    for secret in ("assistant response", header_secret, "test-key"):
        assert secret not in key


@respx.mock
async def test_stores_results_in_cache(recorder: CacheRecorder):
    route = respx.post(MODERATIONS_URL).mock(
        return_value=ok(
            moderation_response(
                [{"flagged": True, "categories": {"hate": True}, "category_scores": {"hate": 0.95}}]
            )
        )
    )
    await make_provider().call_moderation_api("user", "assistant")
    key, value = recorder.sets[-1]
    assert CACHE_KEY_RE.match(key)
    assert "assistant" not in key.split(":", 3)[3]
    assert value.startswith('{"flags":[{"code":"hate"')
    # Moderation always busts the HTTP-level cache; only its own result cache applies.
    await make_provider().call_moderation_api("user", "assistant")
    assert route.call_count == 1


@respx.mock
async def test_cache_keys_built_from_normalized_inputs(recorder: CacheRecorder):
    respx.post(MODERATIONS_URL).mock(return_value=ok(moderation_response()))
    provider = make_provider()
    await provider.call_moderation_api("user", "same normalized text")
    await provider.call_moderation_api("user", [{"type": "text", "text": "same normalized text"}])
    assert recorder.gets[0] == recorder.gets[1]


@respx.mock
async def test_cache_keys_isolated_by_resolved_api_key(recorder: CacheRecorder):
    respx.post(MODERATIONS_URL).mock(return_value=ok(moderation_response()))
    await OpenAiModerationProvider(
        "omni-moderation-latest", {"config": {"apiKey": "sk-moderation-tenant-a"}}
    ).call_moderation_api("user", "same sensitive text")
    await OpenAiModerationProvider(
        "omni-moderation-latest", {"config": {"apiKey": "sk-moderation-tenant-b"}}
    ).call_moderation_api("user", "same sensitive text")
    key_a, key_b = recorder.gets
    assert CACHE_KEY_RE.match(key_a)
    assert CACHE_KEY_RE.match(key_b)
    assert key_a != key_b
    assert "same sensitive text" not in key_a
    assert "sk-moderation-tenant-a" not in key_a
    assert "sk-moderation-tenant-b" not in key_b


async def test_cache_identity_is_stable_across_provider_instances(recorder: CacheRecorder):
    with respx.mock:
        respx.post(MODERATIONS_URL).mock(return_value=ok(moderation_response()))
        for _ in range(2):
            await OpenAiModerationProvider(
                "omni-moderation-latest", {"config": {"apiKey": "sk-moderation-reload"}}
            ).call_moderation_api("user", "same sensitive text")
    assert recorder.gets[0] == recorder.gets[1]
    assert "sk-moderation-reload" not in recorder.gets[0]


async def test_deduplicates_concurrent_identical_requests(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cache, "_enabled", True)
    release = asyncio.Event()
    calls = 0

    async def slow_response(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await release.wait()
        return ok(moderation_response())

    with respx.mock:
        respx.post(MODERATIONS_URL).mock(side_effect=slow_response)
        provider = make_provider()
        first = asyncio.ensure_future(provider.call_moderation_api("user", "same sensitive text"))
        second = asyncio.ensure_future(provider.call_moderation_api("user", "same sensitive text"))
        await asyncio.sleep(0.05)
        release.set()
        assert await asyncio.gather(first, second) == [{"flags": []}, {"flags": []}]
    assert calls == 1


async def test_no_dedupe_across_cache_namespaces(monkeypatch: pytest.MonkeyPatch):
    namespaces = iter(["repeat-0", "repeat-1"])
    monkeypatch.setattr(cache, "get_scoped_cache_key", lambda key: f"{next(namespaces)}:{key}")
    release = asyncio.Event()
    calls = 0

    async def slow_response(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await release.wait()
        return ok(moderation_response())

    with respx.mock:
        respx.post(MODERATIONS_URL).mock(side_effect=slow_response)
        provider = make_provider()
        first = asyncio.ensure_future(provider.call_moderation_api("user", "same sensitive text"))
        second = asyncio.ensure_future(provider.call_moderation_api("user", "same sensitive text"))
        await asyncio.sleep(0.05)
        release.set()
        assert await asyncio.gather(first, second) == [{"flags": []}, {"flags": []}]
    assert calls == 2


# --- Multi-modal ----------------------------------------------------------------------------


@respx.mock
async def test_formats_inputs_for_omni_models():
    route = respx.post(MODERATIONS_URL).mock(return_value=ok(moderation_response()))
    await make_provider().call_moderation_api("user input", "assistant response")
    assert '"type":"text"' in route.calls[0].request.content.decode()


@respx.mock
async def test_mixed_text_and_image_inputs_report_flags():
    route = respx.post(MODERATIONS_URL).mock(
        return_value=ok(
            moderation_response(
                [
                    {
                        "flagged": True,
                        "categories": {"violence": True, "sexual": True},
                        "category_scores": {"violence": 0.95, "sexual": 0.98},
                        "category_applied_input_types": {
                            "violence": ["image"],
                            "sexual": ["text", "image"],
                        },
                    }
                ]
            )
        )
    )
    inputs = [
        {"type": "text", "text": "Some text"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]
    result = await make_provider().call_moderation_api("user input", inputs)
    assert {f["code"] for f in result["flags"]} == {"violence", "sexual"}
    assert "image_url" in route.calls[0].request.content.decode()


@respx.mock
async def test_hashes_cached_multimodal_inputs_without_leaking(recorder: CacheRecorder):
    api_key = "sk-secret-moderation-key"
    header_secret = "Bearer moderation-header-secret"
    text_secret = "multimodal-secret-text"
    image_url = "https://example.com/image.png?token=image-secret-token"
    respx.post(MODERATIONS_URL).mock(return_value=ok(moderation_response()))
    provider = OpenAiModerationProvider(
        "omni-moderation-latest",
        {"config": {"apiKey": api_key, "headers": {"Authorization": header_secret}}},
    )
    await provider.call_moderation_api(
        "user input",
        [
            {"type": "text", "text": text_secret},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    )
    get_key, (set_key, _) = recorder.gets[0], recorder.sets[-1]
    assert get_key == set_key
    assert CACHE_KEY_RE.match(set_key)
    for secret in (text_secret, image_url, "image-secret-token", api_key, header_secret):
        assert secret not in set_key


def test_warns_about_unknown_models(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        OpenAiModerationProvider("unknown-model", {"config": {"apiKey": "test-key"}})
    assert any("unknown OpenAI moderation model" in m for m in caplog.messages)


@respx.mock
async def test_accepts_custom_api_headers():
    route = respx.post(MODERATIONS_URL).mock(return_value=ok(moderation_response([])))
    await make_provider(headers={"Custom-Header": "custom-value"}).call_moderation_api(
        "user", "assistant"
    )
    headers = route.calls[0].request.headers
    assert headers["Custom-Header"] == "custom-value"
    assert headers["x-promptfoo-silent"] == "true"


# --- Utility functions ----------------------------------------------------------------------


def test_input_type_guards():
    text = {"type": "text", "text": "test content"}
    assert is_text_input(text) is True
    assert is_text_input(IMAGE_INPUT) is False
    assert is_image_input(IMAGE_INPUT) is True
    assert is_image_input(text) is False


def test_supports_image_input():
    assert supports_image_input("omni-moderation-latest") is True
    assert supports_image_input("omni-moderation-2024-09-26") is True
    for model in ("text-moderation-latest", "text-moderation-stable", "text-moderation-007"):
        assert supports_image_input(model) is False
    assert supports_image_input("nonexistent-model") is False


def text(value: str) -> dict[str, str]:
    return {"type": "text", "text": value}


@pytest.mark.parametrize(
    ("content", "supports_images", "expected"),
    [
        ("test content", False, "test content"),
        ("test content", True, [text("test content")]),
        ([text("text content"), IMAGE_INPUT], True, [text("text content"), IMAGE_INPUT]),
        ([text("text content"), IMAGE_INPUT], False, "text content"),
        ([text("first text"), IMAGE_INPUT, text("second text")], False, "first text second text"),
        ([text(""), IMAGE_INPUT, text("second text")], False, " second text"),
        (
            [text("  first text  "), IMAGE_INPUT, text("second text")],
            False,
            "  first text   second text",
        ),
        ([IMAGE_INPUT, text("only text"), IMAGE_INPUT], False, "only text"),
    ],
)
def test_format_moderation_input(content, supports_images, expected):
    assert format_moderation_input(content, supports_images) == expected
