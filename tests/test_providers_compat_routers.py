"""OpenRouter, OrcaRouter and TrueFoundry providers. Ports deterministic cases from
test/providers/openrouter.test.ts, orcarouter.test.ts and truefoundry.test.ts, with HTTP
mocked by respx."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from openai_helpers import ok, openai_env, sent_json

from promptphoo import fetch
from promptphoo.providers.compatible.openrouter import (
    OpenRouterProvider,
    create_openrouter_provider,
    get_choice_error_kind,
)
from promptphoo.providers.compatible.openrouter import (
    create_provider as create_openrouter,
)
from promptphoo.providers.compatible.orcarouter import (
    OrcaRouterProvider,
    create_orcarouter_provider,
)
from promptphoo.providers.compatible.truefoundry import (
    TrueFoundryEmbeddingProvider,
    TrueFoundryProvider,
    create_truefoundry_provider,
)

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_URL = f"{OPENROUTER_API_BASE}/chat/completions"
ORCAROUTER_API_BASE = "https://api.orcarouter.ai/v1"
ORCA_URL = f"{ORCAROUTER_API_BASE}/chat/completions"
TRUEFOUNDRY_API_BASE = "https://llm-gateway.truefoundry.com"
TF_URL = f"{TRUEFOUNDRY_API_BASE}/chat/completions"
USAGE = {"total_tokens": 5, "prompt_tokens": 3, "completion_tokens": 2}


@pytest.fixture(autouse=True)
def vendor_env(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "OPENROUTER_API_KEY",
        "CUSTOM_OPENROUTER_KEY",
        "ORCAROUTER_API_KEY",
        "CUSTOM_ORCA_KEY",
        "TRUEFOUNDRY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _message_response(message: dict[str, Any], **choice: Any) -> dict[str, Any]:
    return {"choices": [{"message": message, **choice}], "usage": USAGE}


# --- OpenRouter ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("api_key_envar", "api_key", "expected"),
    [
        (None, None, "scoped-router-key"),
        ("CUSTOM_OPENROUTER_KEY", None, "scoped-custom-key"),
        ("OPENAI_API_KEY", None, "selected-openai-key"),
        ("CUSTOM_OPENROUTER_KEY", "explicit-key", "explicit-key"),
    ],
)
@respx.mock
async def test_openrouter_sends_selected_credential(vendor_env, api_key_envar, api_key, expected):
    vendor_env.setenv("OPENROUTER_API_KEY", "process-router-key")
    vendor_env.setenv("CUSTOM_OPENROUTER_KEY", "process-custom-key")
    route = respx.post("https://proxy.example.com/v1/chat/completions").mock(
        return_value=ok({"choices": [{"message": {"content": "Hello"}}], "usage": {"cost": 0.25}})
    )
    config: dict[str, Any] = {"apiBaseUrl": "https://proxy.example.com/v1"}
    if api_key_envar:
        config["apiKeyEnvar"] = api_key_envar
    if api_key:
        config["apiKey"] = api_key
    provider = OpenRouterProvider(
        "fixture/model",
        {
            "config": config,
            "env": {
                "OPENROUTER_API_KEY": "scoped-router-key",
                "CUSTOM_OPENROUTER_KEY": "scoped-custom-key",
                "OPENAI_API_KEY": "selected-openai-key",
            },
        },
    )
    result = await provider.call_api("Hello")
    assert result["output"] == "Hello"
    assert result["cost"] == 0.25
    assert route.calls[0].request.headers["Authorization"] == f"Bearer {expected}"


@pytest.mark.parametrize("api_key_envar", [None, "CUSTOM_OPENROUTER_KEY"])
def test_openrouter_missing_selected_credential(api_key_envar):
    provider = OpenRouterProvider(
        "fixture/model",
        {
            "config": {"apiKeyEnvar": api_key_envar},
            "env": {
                "OPENAI_API_KEY": "unrelated-scoped-openai-key",
                "OPENROUTER_API_KEY": "nonselected-router-key" if api_key_envar else None,
            },
        },
    )
    assert provider.get_api_key() is None


def test_openrouter_identity_and_json():
    provider = OpenRouterProvider("google/gemini-2.5-pro", {})
    assert provider.modelName == "google/gemini-2.5-pro"
    assert provider.id() == "openrouter:google/gemini-2.5-pro"
    assert str(provider) == "[OpenRouter Provider google/gemini-2.5-pro]"
    provider = OpenRouterProvider(
        "google/gemini-2.5-pro", {"config": {"temperature": 0.7, "max_tokens": 100}}
    )
    assert provider.to_json() == {
        "provider": "openrouter",
        "model": "google/gemini-2.5-pro",
        "config": {
            "temperature": 0.7,
            "max_tokens": 100,
            "apiKeyEnvar": "OPENROUTER_API_KEY",
            "apiBaseUrl": OPENROUTER_API_BASE,
            "passthrough": {},
        },
    }


def test_openrouter_base_url_and_envar_overrides(vendor_env):
    vendor_env.setenv("CUSTOM_OPENROUTER_KEY", "custom-test-key")
    provider = OpenRouterProvider(
        "m",
        {"config": {"apiBaseUrl": "https://proxy.example.com/openrouter/api/v1",
                    "apiKeyEnvar": "CUSTOM_OPENROUTER_KEY"}},
    )  # fmt: skip
    assert provider.config["apiBaseUrl"] == "https://proxy.example.com/openrouter/api/v1"
    assert provider.get_api_key() == "custom-test-key"
    for config in ({}, {"apiBaseUrl": "", "apiKeyEnvar": ""}):
        provider = OpenRouterProvider("m", {"config": config})
        assert provider.config["apiBaseUrl"] == OPENROUTER_API_BASE
        assert provider.config["apiKeyEnvar"] == "OPENROUTER_API_KEY"


@respx.mock
async def test_openrouter_passes_specific_options(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    route = respx.post(OPENROUTER_URL).mock(return_value=ok(_message_response({"content": "ok"})))
    provider = OpenRouterProvider(
        "openai/gpt-4o",
        {"config": {"transforms": ["middle-out"], "models": ["a", "b"], "route": "fallback",
                    "provider": {"order": ["x"]}, "passthrough": {"seed": 1}}},
    )  # fmt: skip
    await provider.call_api("Test prompt")
    body = sent_json(route)
    assert body["transforms"] == ["middle-out"]
    assert body["models"] == ["a", "b"]
    assert body["route"] == "fallback"
    assert body["provider"] == {"order": ["x"]}
    assert body["seed"] == 1
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"


@respx.mock
async def test_openrouter_malformed_response_is_evicted(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    route = respx.post(OPENROUTER_URL).mock(
        side_effect=[
            ok({"choices": [], "usage": USAGE, "private": "must-not-appear"}),
            ok(_message_response({"content": "Recovered"}, finish_reason="stop")),
        ]
    )
    provider = OpenRouterProvider("gpt-4o", {"config": {"inputCost": 0.001, "outputCost": 0.002}})
    malformed = await provider.call_api("Test prompt")
    assert malformed == {
        "error": "Malformed response data: expected choices[0].message",
        "tokenUsage": {"total": 5, "prompt": 3, "completion": 2, "numRequests": 1},
        "cached": False,
        "cost": pytest.approx(0.007),
    }
    recovered = await provider.call_api("Test prompt")
    assert recovered["output"] == "Recovered"
    assert route.call_count == 2


@respx.mock
async def test_openrouter_choice_level_error(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    respx.post(OPENROUTER_URL).mock(
        return_value=ok(
            {
                "choices": [
                    {
                        "message": {"content": "partial output must not be graded"},
                        "finish_reason": "error",
                        "error": {"code": 502, "message": "PRIVATE_DIAGNOSTIC"},
                    }
                ],
                "usage": USAGE,
            }
        )
    )
    provider = OpenRouterProvider("gpt-4o", {"config": {"inputCost": 0.001, "outputCost": 0.002}})
    failed = await provider.call_api("Test prompt")
    assert failed["error"] == "API error: PRIVATE_DIAGNOSTIC"
    assert "output" not in failed
    assert failed["finishReason"] == "error"
    assert failed["cost"] == pytest.approx(0.007)
    assert failed["metadata"]["retryableErrorKind"] == "transient_availability"


@respx.mock
async def test_openrouter_finish_reason_error(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    respx.post(OPENROUTER_URL).mock(
        return_value=ok(
            {"choices": [{"message": {"content": "partial"}, "finish_reason": "error"}]}
        )
    )
    result = await OpenRouterProvider("gpt-4o", {}).call_api("Test prompt")
    assert result == {
        "error": "API error: OpenRouter provider returned a generation error",
        "tokenUsage": {"numRequests": 1},
        "cached": False,
        "finishReason": "error",
    }


def test_choice_error_kind():
    assert get_choice_error_kind({"code": 429}) == {"rateLimitKind": "rate_limit"}
    assert get_choice_error_kind({"code": "503"}) == {
        "retryableErrorKind": "transient_availability"
    }
    assert get_choice_error_kind({"code": 429, "metadata": {"error_type": "timeout"}}) == {
        "retryableErrorKind": "transient_availability"
    }
    assert get_choice_error_kind({"code": 429, "metadata": {"error_type": "refusal"}}) is None
    assert get_choice_error_kind({"code": 400}) is None


@respx.mock
async def test_openrouter_reported_cost_and_billing_metadata(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    respx.post(OPENROUTER_URL).mock(
        return_value=ok(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {**USAGE, "cost": 0.0042, "is_byok": False,
                          "cost_details": {"upstream_inference_cost": 0.004}},
            }
        )
    )  # fmt: skip
    result = await OpenRouterProvider("openai/gpt-4o", {}).call_api("Test prompt")
    assert result["cost"] == 0.0042
    assert result["metadata"] == {
        "openrouter": {
            "accountCharge": 0.0042,
            "reportedUpstreamInferenceCost": 0.004,
            "isByok": False,
        }
    }
    assert result["finishReason"] == "stop"


@respx.mock
async def test_openrouter_http_error(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    respx.post(OPENROUTER_URL).mock(return_value=ok({"error": {"message": "bad"}}, status=400))
    result = await OpenRouterProvider("openai/gpt-4o", {}).call_api("Test prompt")
    assert result["error"].startswith("API error: 400 Bad Request\n")
    assert result["metadata"]["http"]["status"] == 400
    assert "cost" not in result


@respx.mock
async def test_openrouter_network_error(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    respx.post(OPENROUTER_URL).mock(side_effect=httpx.ConnectError("boom"))
    with fetch.with_fetch_retry_context(0):
        result = await OpenRouterProvider("openai/gpt-4o", {}).call_api("Test prompt")
    assert result["error"].startswith("API call error:")


@pytest.mark.parametrize(
    ("message", "show_thinking", "expected"),
    [
        ({"content": "Answer", "reasoning": "Think"}, None, "Thinking: Think\n\nAnswer"),
        ({"content": "Answer", "reasoning": "Think"}, False, "Answer"),
        ({"content": "", "reasoning": "Only reasoning"}, None, "Only reasoning"),
        ({"content": None, "reasoning": "Only reasoning"}, False, ""),
        ({"content": "Answer", "reasoning": ""}, None, "Answer"),
        ({"content": "Answer"}, None, "Answer"),
    ],
)
@respx.mock
async def test_openrouter_thinking(vendor_env, message, show_thinking, expected):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    respx.post(OPENROUTER_URL).mock(return_value=ok(_message_response(message)))
    config = {} if show_thinking is None else {"showThinking": show_thinking}
    result = await OpenRouterProvider("google/gemini-2.5-pro", {"config": config}).call_api("x")
    assert result["output"] == expected


TOOL_CALLS = [
    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
]


@pytest.mark.parametrize("content", ["Some content", "", "   ", None])
@respx.mock
async def test_openrouter_tool_calls_take_priority(vendor_env, content):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    respx.post(OPENROUTER_URL).mock(
        return_value=ok(
            _message_response({"content": content, "reasoning": "Think", "tool_calls": TOOL_CALLS})
        )
    )
    result = await OpenRouterProvider("qwen/qwen3", {"config": {"showThinking": True}}).call_api(
        "x"
    )
    assert result["output"] == TOOL_CALLS


@respx.mock
async def test_openrouter_function_call_whitespace_content(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    function_call = {"name": "get_weather", "arguments": '{"city":"Paris"}'}
    respx.post(OPENROUTER_URL).mock(
        return_value=ok(_message_response({"content": "  ", "function_call": function_call}))
    )
    result = await OpenRouterProvider("m", {}).call_api("x")
    assert result["output"] == function_call


@respx.mock
async def test_openrouter_tool_call_without_type_discriminator(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    call = {"id": "c", "function": {"name": "f", "arguments": "{}"}}
    respx.post(OPENROUTER_URL).mock(
        return_value=ok(_message_response({"content": None, "tool_calls": [call]}))
    )
    result = await OpenRouterProvider("m", {}).call_api("x")
    assert result["output"] == [call]


@respx.mock
async def test_openrouter_structured_array_content(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    parts = [
        {"type": "text", "text": "Here is an image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    respx.post(OPENROUTER_URL).mock(return_value=ok(_message_response({"content": parts})))
    result = await OpenRouterProvider("google/gemini-2.5-flash-image", {}).call_api("x")
    assert result["output"] == parts


@respx.mock
async def test_openrouter_audio_message(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    audio = {"id": "a1", "data": "UklGR", "transcript": "Hello there", "expires_at": 1}
    respx.post(OPENROUTER_URL).mock(
        return_value=ok(_message_response({"content": None, "audio": audio}))
    )
    result = await OpenRouterProvider("openai/gpt-audio", {}).call_api("x")
    assert result["output"] == "Hello there"
    assert result["audio"] == {
        "id": "a1",
        "expiresAt": 1,
        "data": "UklGR",
        "transcript": "Hello there",
        "format": "wav",
    }


@respx.mock
async def test_openrouter_refusal(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    respx.post(OPENROUTER_URL).mock(
        return_value=ok(_message_response({"content": None, "refusal": "I can't help"}))
    )
    result = await OpenRouterProvider("m", {}).call_api("x")
    assert result["output"] == "I can't help"
    assert result["isRefusal"] is True
    assert result["guardrails"] == {"flagged": True}


@respx.mock
async def test_openrouter_empty_and_missing_choices(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    respx.post(OPENROUTER_URL).mock(side_effect=[ok({"choices": []}), ok({"id": "x"})])
    provider = OpenRouterProvider("m", {})
    for _ in range(2):
        result = await provider.call_api("x")
        assert result["error"] == "Malformed response data: expected choices[0].message"


@respx.mock
async def test_openrouter_error_body(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    respx.post(OPENROUTER_URL).mock(
        return_value=ok({"error": {"message": "Invalid API key", "type": "auth", "code": "401"}})
    )
    result = await OpenRouterProvider("m", {}).call_api("x")
    assert result["error"].startswith("API error: Invalid API key, Type: auth, Code: 401")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"name":"John","age":30}', {"name": "John", "age": 30}),
        ("not json", "not json"),
    ],
)
@respx.mock
async def test_openrouter_json_schema(vendor_env, content, expected):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    respx.post(OPENROUTER_URL).mock(return_value=ok(_message_response({"content": content})))
    provider = OpenRouterProvider(
        "openai/gpt-4o",
        {"config": {"response_format": {"type": "json_schema", "json_schema": {"name": "x",
                                                                                "schema": {}}}}},
    )  # fmt: skip
    result = await provider.call_api("x")
    assert result["output"] == expected


@respx.mock
async def test_openrouter_json_schema_with_reasoning(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    respx.post(OPENROUTER_URL).mock(
        return_value=ok(_message_response({"content": '{"a":1}', "reasoning": "Think"}))
    )
    provider = OpenRouterProvider(
        "m", {"config": {"response_format": {"type": "json_schema", "json_schema": {}}}}
    )
    assert (await provider.call_api("x"))["output"] == {"a": 1}


@respx.mock
async def test_openrouter_json_object_not_parsed(vendor_env):
    vendor_env.setenv("OPENROUTER_API_KEY", "test-key")
    respx.post(OPENROUTER_URL).mock(return_value=ok(_message_response({"content": '{"a":1}'})))
    provider = OpenRouterProvider("m", {"config": {"response_format": {"type": "json_object"}}})
    assert (await provider.call_api("x"))["output"] == '{"a":1}'


def test_openrouter_factory_shapes():
    provider = create_openrouter(
        "openrouter:openai/gpt-4o",
        {"config": {"temperature": 0.1}, "env": {"OPENROUTER_API_KEY": "k"}, "id": "custom"},
    )
    assert isinstance(provider, OpenRouterProvider)
    # configuredProviderFactory hands the whole ProviderOptions over as `config`.
    assert provider.id() == "custom"
    assert provider.config["temperature"] == 0.1
    assert provider.get_api_key() == "k"
    provider = create_openrouter_provider(
        "openrouter:a:b", {"id": "outer", "env": {"OPENROUTER_API_KEY": "outer-key"}}
    )
    assert provider.model_name == "a:b"
    assert provider.id() == "outer"
    assert provider.get_api_key() == "outer-key"


# --- OrcaRouter ----------------------------------------------------------------------------


def _ok_chat() -> httpx.Response:
    return ok(
        {
            "choices": [{"message": {"content": "Test output"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
        }
    )


def test_orcarouter_identity_and_json():
    provider = OrcaRouterProvider("openai/gpt-5.5", {})
    assert provider.id() == "orcarouter:openai/gpt-5.5"
    assert str(provider) == "[OrcaRouter Provider openai/gpt-5.5]"
    assert provider.get_gen_ai_system() == "orcarouter"
    assert provider.to_json() == {
        "provider": "orcarouter",
        "model": "openai/gpt-5.5",
        "config": {
            "apiBaseUrl": ORCAROUTER_API_BASE,
            "apiKeyEnvar": "ORCAROUTER_API_KEY",
            "headers": {"HTTP-Referer": "https://promptfoo.dev/", "X-Title": "promptfoo"},
            "passthrough": {},
        },
    }
    redacted = OrcaRouterProvider("m", {"config": {"apiKey": "sk-secret"}}).to_json()
    assert "apiKey" not in redacted["config"]


def test_orcarouter_config_defaults_and_overrides(vendor_env):
    provider = OrcaRouterProvider("m", {"config": {"headers": {"X-Title": "custom"}}})
    assert provider.config["headers"] == {
        "HTTP-Referer": "https://promptfoo.dev/",
        "X-Title": "custom",
    }
    provider = OrcaRouterProvider("m", {"config": {"apiBaseUrl": "", "apiKeyEnvar": ""}})
    assert provider.config["apiBaseUrl"] == ORCAROUTER_API_BASE
    assert provider.config["apiKeyEnvar"] == "ORCAROUTER_API_KEY"
    # OPENAI_API_KEY is set by the openai_env fixture.
    assert provider.get_api_key() is None
    vendor_env.setenv("CUSTOM_ORCA_KEY", "custom-test-key")
    provider = OrcaRouterProvider(
        "m",
        {"config": {"apiBaseUrl": "https://proxy.example.com/orcarouter/v1",
                    "apiKeyEnvar": "CUSTOM_ORCA_KEY"}},
    )  # fmt: skip
    assert provider.get_api_key() == "custom-test-key"
    vendor_env.setenv("ORCAROUTER_API_KEY", "process-key")
    provider = OrcaRouterProvider("m", {"env": {"ORCAROUTER_API_KEY": "provider-key"}})
    assert provider.get_api_key() == "provider-key"
    provider.config["apiBaseUrl"] = None
    assert provider.get_api_url() == ORCAROUTER_API_BASE


async def test_orcarouter_missing_key_error():
    with pytest.raises(ValueError, match="ORCAROUTER_API_KEY"):
        await OrcaRouterProvider("openai/gpt-5.5", {}).call_api("Test prompt")


@respx.mock
async def test_orcarouter_request_headers_and_no_organization(vendor_env):
    vendor_env.setenv("ORCAROUTER_API_KEY", "default-test-key")
    vendor_env.setenv("OPENAI_ORGANIZATION", "org-leak")
    vendor_env.setenv("OPENAI_BASE_URL", "https://wrong.example.com/v1")
    route = respx.post(ORCA_URL).mock(return_value=_ok_chat())
    provider = OrcaRouterProvider("openai/gpt-5.5", {})
    assert provider.get_organization() is None
    result = await provider.call_api("Test prompt")
    assert result["output"] == "Test output"
    headers = route.calls[0].request.headers
    assert headers["Authorization"] == "Bearer default-test-key"
    assert headers["HTTP-Referer"] == "https://promptfoo.dev/"
    assert headers["X-Title"] == "promptfoo"
    assert "OpenAI-Organization" not in headers


@respx.mock
async def test_orcarouter_no_authorization_when_key_not_required():
    route = respx.post(ORCA_URL).mock(return_value=_ok_chat())
    await OrcaRouterProvider("openai/gpt-5.5", {"config": {"apiKeyRequired": False}}).call_api(
        "Test prompt"
    )
    assert "Authorization" not in route.calls[0].request.headers


@pytest.mark.parametrize(
    ("model", "config", "has_temperature"),
    [
        ("anthropic/claude-opus-4", {"temperature": 0.5}, False),
        ("deepseek/deepseek-reasoner", {"temperature": 0.5}, False),
        ("anthropic/claude-opus-4", {"passthrough": {"temperature": 0.5}}, False),
        ("anthropic/claude-opus-3", {"temperature": 0.5}, True),
        ("anthropic/claude-opus-25", {"temperature": 0.5}, False),
        ("openai/gpt-4o", {"temperature": 0.5, "models": ["anthropic/claude-opus-4.1"]}, False),
        ("openai/gpt-4o", {"temperature": 0.5, "models": ["deepseek/deepseek-r2"]}, False),
    ],
)
async def test_orcarouter_temperature(vendor_env, model, config, has_temperature):
    vendor_env.setenv("ORCAROUTER_API_KEY", "k")
    body = (await OrcaRouterProvider(model, {"config": config}).get_openai_body("x"))["body"]
    assert ("temperature" in body) is has_temperature
    if model.startswith("anthropic"):
        assert "max_tokens" in body


async def test_orcarouter_routing_options():
    provider = OrcaRouterProvider(
        "openai/gpt-4o", {"config": {"models": ["a", "b"], "route": "fallback"}}
    )
    body = (await provider.get_openai_body("x"))["body"]
    assert body["models"] == ["a", "b"]
    assert body["route"] == "fallback"
    body = (
        await provider.get_openai_body(
            "x", {"vars": {}, "prompt": {"config": {"models": ["c"], "route": "other"}}}
        )
    )["body"]
    assert body["models"] == ["c"]
    assert body["route"] == "other"


async def test_orcarouter_renders_templates():
    provider = OrcaRouterProvider(
        "openai/gpt-5.5",
        {"config": {"passthrough": {"models": ["{{ fallback }}"], "route": "{{ route }}"},
                    "reasoning_effort": "{{ effort }}"}},
    )  # fmt: skip
    body = (
        await provider.get_openai_body(
            "x", {"vars": {"fallback": "anthropic/claude-sonnet-4", "route": "fallback",
                           "effort": "high"}}
        )
    )["body"]  # fmt: skip
    assert body["models"] == ["anthropic/claude-sonnet-4"]
    assert body["route"] == "fallback"
    assert body["reasoning_effort"] == "high"


async def test_orcarouter_reasoning_effort_for_non_openai_upstream():
    provider = OrcaRouterProvider(
        "anthropic/claude-sonnet-4", {"config": {"reasoning_effort": "low"}}
    )
    assert (await provider.get_openai_body("x"))["body"]["reasoning_effort"] == "low"


@respx.mock
async def test_orcarouter_non_2xx(vendor_env):
    vendor_env.setenv("ORCAROUTER_API_KEY", "k")
    respx.post(ORCA_URL).mock(return_value=ok({"error": {"message": "nope"}}, status=400))
    result = await OrcaRouterProvider("openai/gpt-4o", {}).call_api("x")
    assert result["error"].startswith("API error: 400")


@respx.mock
async def test_orcarouter_reasoning_content(vendor_env):
    vendor_env.setenv("ORCAROUTER_API_KEY", "k")
    respx.post(ORCA_URL).mock(
        return_value=ok(_message_response({"content": "Answer", "reasoning_content": "Hmm"}))
    )
    result = await OrcaRouterProvider("deepseek/deepseek-r1", {}).call_api("x")
    assert result["output"] == "Thinking: Hmm\n\nAnswer"


def test_orcarouter_factory():
    assert create_orcarouter_provider("orcarouter:openai/gpt-5.5").id() == (
        "orcarouter:openai/gpt-5.5"
    )
    provider = create_orcarouter_provider(
        "orcarouter:openai/gpt-5.5",
        {"id": "custom-orca", "env": {"ORCAROUTER_API_KEY": "provider-test-key"}},
    )
    assert provider.id() == "custom-orca"
    assert provider.get_api_key() == "provider-test-key"
    provider = create_orcarouter_provider(
        "orcarouter:openai/gpt-5.5",
        {
            "id": "ignored-id",
            "env": {"ORCAROUTER_API_KEY": "ignored-key"},
            "config": {"id": "configured-id", "env": {"ORCAROUTER_API_KEY": "configured-key"}},
        },
    )
    assert provider.id() == "configured-id"
    assert provider.get_api_key() == "configured-key"


# --- TrueFoundry ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "reasoning"),
    [
        ("openai/gpt-5-nano", True),
        ("gpt-5", True),
        ("openai/o1-mini", True),
        ("o1", True),
        ("openai/gpt-4", False),
        ("anthropic/claude-3-5-sonnet", False),
        ("google/gemini-pro", False),
    ],
)
def test_truefoundry_reasoning_detection(model, reasoning):
    assert TrueFoundryProvider(model, {}).is_reasoning_model() is reasoning


def test_truefoundry_identity_and_json():
    provider = TrueFoundryProvider("openai/gpt-4", {"config": {"temperature": 0.7,
                                                                "max_tokens": 100}})  # fmt: skip
    assert provider.id() == "truefoundry:openai/gpt-4"
    assert str(provider) == "[TrueFoundry Provider openai/gpt-4]"
    assert provider.to_json() == {
        "provider": "truefoundry",
        "model": "openai/gpt-4",
        "config": {
            "temperature": 0.7,
            "max_tokens": 100,
            "apiKeyEnvar": "TRUEFOUNDRY_API_KEY",
            "apiBaseUrl": TRUEFOUNDRY_API_BASE,
        },
    }
    redacted = TrueFoundryProvider("m", {"config": {"apiKey": "secret-api-key"}}).to_json()
    assert "apiKey" not in redacted["config"]
    custom = TrueFoundryProvider("m", {"config": {"apiBaseUrl": "https://custom.example.com"}})
    assert custom.config["apiBaseUrl"] == "https://custom.example.com"


@respx.mock
async def test_truefoundry_call_api(vendor_env):
    vendor_env.setenv("TRUEFOUNDRY_API_KEY", "test-key")
    route = respx.post(TF_URL).mock(return_value=_ok_chat())
    result = await TrueFoundryProvider("openai/gpt-4", {}).call_api("Test prompt")
    assert sent_json(route) == {
        "model": "openai/gpt-4",
        "messages": [{"role": "user", "content": "Test prompt"}],
        "max_tokens": 1024,
        "temperature": 0,
    }
    headers = route.calls[0].request.headers
    assert headers["Authorization"] == "Bearer test-key"
    assert result["output"] == "Test output"
    assert result["tokenUsage"] == {"total": 10, "prompt": 5, "completion": 5, "numRequests": 1}
    assert result["cost"] == pytest.approx(0.00045)
    assert result["guardrails"] == {"flagged": False}


@pytest.mark.parametrize(
    ("model", "config", "expected_cost"),
    [
        ("openai-main/gpt-4o", {}, 0.0000625),
        ("production-east/gpt-4o", {"openaiAccountNames": ["production-east"]}, 0.0000625),
        ("openai-main/gpt-4o-mini", {"passthrough": {"model": "openai-main/gpt-4o"}}, 0.0000625),
        ("vendor/gpt-4", {}, None),
        ("vendor/openai-main/gpt-4", {}, None),
        ("openai-prod/gpt-4", {}, None),
        (
            "openai-main/gpt-4o",
            {"openaiAccountNames": ["production-east"],
             "passthrough": {"model": "vendor/production-east/gpt-4"}},
            None,
        ),
    ],
)  # fmt: skip
@respx.mock
async def test_truefoundry_billing_namespaces(vendor_env, model, config, expected_cost):
    vendor_env.setenv("TRUEFOUNDRY_API_KEY", "test-key")
    respx.post(TF_URL).mock(return_value=_ok_chat())
    result = await TrueFoundryProvider(model, {"config": config}).call_api("Test prompt")
    if expected_cost is None:
        assert "cost" not in result
    else:
        assert result["cost"] == pytest.approx(expected_cost)


@respx.mock
async def test_truefoundry_headers_and_body(vendor_env):
    vendor_env.setenv("TRUEFOUNDRY_API_KEY", "test-key")
    route = respx.post(TF_URL).mock(return_value=_ok_chat())
    metadata = {"user_id": "u1", "environment": "test"}
    logging_config = {"enabled": True}
    servers = [{"integration_fqn": "x:y", "enable_all_tools": True}]
    provider = TrueFoundryProvider(
        "openai/gpt-4",
        {"config": {"metadata": metadata, "loggingConfig": logging_config,
                    "mcp_servers": servers, "iteration_limit": 3}},
    )  # fmt: skip
    await provider.call_api("Test prompt")
    headers = route.calls[0].request.headers
    assert json.loads(headers["X-TFY-METADATA"]) == metadata
    assert json.loads(headers["X-TFY-LOGGING-CONFIG"]) == logging_config
    body = sent_json(route)
    assert "metadata" not in body
    assert body["mcp_servers"] == servers
    assert body["iteration_limit"] == 3


def _bad_request(payload: dict[str, Any]) -> httpx.Response:
    return ok(payload, status=400)


@pytest.mark.parametrize(
    ("payload", "output", "guardrails"),
    [
        (
            {"error": {"message": "Guardrail violation detected",
                       "type": "guardrail_checks_failed"},
             "guardrail_checks": {"llm_input_guardrails": [{"name": "safety/prompt-injection"}]}},
            "Guardrail violation detected",
            {"flagged": True, "flaggedInput": True, "flaggedOutput": False,
             "reason": "Guardrail violation detected"},
        ),
        (
            {"status": "failure",
             "error": {"message": "azure-foundry error: Response content blocked by label 'X'.",
                       "type": "APIError", "code": "400"}},
            "azure-foundry error: Response content blocked by label 'X'.",
            {"flagged": True, "flaggedInput": False, "flaggedOutput": True,
             "reason": "azure-foundry error: Response content blocked by label 'X'."},
        ),
        (
            {"error": {"message": "Safety policy rejected request", "code": "content_filter",
                       "innererror": {"code": "ResponsibleAIPolicyViolation"}}},
            "Safety policy rejected request",
            {"flagged": True, "reason": "Safety policy rejected request"},
        ),
        (
            {"error": {"message": "Unable to determine whether the response was filtered because "
                                  "the content filtering system timed out",
                       "code": "content_filter_error",
                       "content_filter_results": {"hate": {"filtered": True}}}},
            "Unable to determine whether the response was filtered because the content filtering "
            "system timed out",
            {"flagged": True, "flaggedInput": False, "flaggedOutput": True,
             "reason": "Unable to determine whether the response was filtered because the "
                       "content filtering system timed out"},
        ),
    ],
)  # fmt: skip
@respx.mock
async def test_truefoundry_guardrail_normalization(vendor_env, payload, output, guardrails):
    vendor_env.setenv("TRUEFOUNDRY_API_KEY", "test-key")
    respx.post(TF_URL).mock(return_value=_bad_request(payload))
    result = await TrueFoundryProvider("openai/gpt-4", {}).call_api("Test prompt")
    assert "error" not in result
    assert result["output"] == output
    assert result["isRefusal"] is True
    assert result["guardrails"] == guardrails
    assert result["metadata"]["http"]["status"] == 400


@pytest.mark.parametrize(
    "error",
    [
        {"message": "Content filtering system is down", "code": "content_filter_error"},
        {"message": "Responsible AI policy check did not complete because the content filtering "
                    "system timed out", "code": "content_filter_error"},
        {"message": "Unable to determine whether the response was filtered", "code": "400",
         "innererror": {"code": "content_filter_error"}},
        {"message": "Invalid request", "type": "invalid_request_error"},
    ],
)  # fmt: skip
@respx.mock
async def test_truefoundry_preserves_other_400s(vendor_env, error):
    vendor_env.setenv("TRUEFOUNDRY_API_KEY", "test-key")
    respx.post(TF_URL).mock(return_value=_bad_request({"error": error}))
    result = await TrueFoundryProvider("openai/gpt-4", {}).call_api("Test prompt")
    assert "400 Bad Request" in result["error"]
    assert "isRefusal" not in result
    assert "guardrails" not in result


@respx.mock
async def test_truefoundry_network_error(vendor_env):
    vendor_env.setenv("TRUEFOUNDRY_API_KEY", "test-key")
    respx.post(TF_URL).mock(side_effect=httpx.ConnectError("Network error"))
    provider = TrueFoundryProvider("openai/gpt-4", {"config": {"maxRetries": 0}})
    result = await provider.call_api("Test prompt")
    assert "Network error" in result["error"]


EMBED_URL = f"{TRUEFOUNDRY_API_BASE}/embeddings"


def _embedding_response() -> httpx.Response:
    return ok({"data": [{"embedding": [0.1, 0.2, 0.3]}],
               "usage": {"total_tokens": 10, "prompt_tokens": 10}})  # fmt: skip


def test_truefoundry_embedding_identity():
    provider = TrueFoundryEmbeddingProvider("openai/text-embedding-3-large", {})
    assert provider.id() == "truefoundry:openai/text-embedding-3-large"
    assert str(provider) == "[TrueFoundry Embedding Provider openai/text-embedding-3-large]"
    assert provider.to_json() == {
        "provider": "truefoundry",
        "model": "openai/text-embedding-3-large",
        "config": {"apiKeyEnvar": "TRUEFOUNDRY_API_KEY", "apiBaseUrl": TRUEFOUNDRY_API_BASE},
    }


@respx.mock
async def test_truefoundry_embedding_call_and_headers(vendor_env):
    vendor_env.setenv("TRUEFOUNDRY_API_KEY", "test-key")
    route = respx.post(EMBED_URL).mock(return_value=_embedding_response())
    provider = TrueFoundryEmbeddingProvider(
        "openai-main/text-embedding-3-small",
        {"config": {"metadata": {"a": 1}, "loggingConfig": {"enabled": True}}},
    )
    result = await provider.call_embedding_api("Test text")
    assert result["embedding"] == [0.1, 0.2, 0.3]
    assert result["cost"] == pytest.approx(10 * 0.02 / 1e6)
    headers = route.calls[0].request.headers
    assert headers["X-TFY-METADATA"] == '{"a":1}'
    assert headers["X-TFY-LOGGING-CONFIG"] == '{"enabled":true}'
    assert "X-TFY-METADATA" not in (provider.config.get("headers") or {})


@respx.mock
async def test_truefoundry_embedding_error(vendor_env):
    vendor_env.setenv("TRUEFOUNDRY_API_KEY", "test-key")
    respx.post(EMBED_URL).mock(return_value=_bad_request({"error": {"message": "bad"}}))
    result = await TrueFoundryEmbeddingProvider("m-embedding", {}).call_embedding_api("x")
    assert "400" in result["error"]


def test_truefoundry_factory():
    def create(path: str, **config: Any):
        return create_truefoundry_provider(path, {"config": {"config": config}})

    assert isinstance(create("truefoundry:openai/gpt-4"), TrueFoundryProvider)
    assert isinstance(
        create("truefoundry:openai/text-embedding-3-large"), TrueFoundryEmbeddingProvider
    )
    assert isinstance(
        create("truefoundry:cohere-main/embed-english-v3.0", task="embedding"),
        TrueFoundryEmbeddingProvider,
    )
    chat = create("truefoundry:embedding-account/gpt-4", task="chat")
    assert isinstance(chat, TrueFoundryProvider)
    assert chat.model_name == "embedding-account/gpt-4"
    assert create("truefoundry:openai/gpt-4:latest").model_name == "openai/gpt-4:latest"
    for task in ("", "embeddings", "image", None, False):
        with pytest.raises(ValueError, match='TrueFoundry config.task must be "chat" or'):
            create("truefoundry:x", task=task)
    provider = create_truefoundry_provider(
        "truefoundry:openai/gpt-4",
        {"config": {"config": {"temperature": 0.5}}, "env": {"TRUEFOUNDRY_API_KEY": "env-key"}},
    )
    assert provider.config["temperature"] == 0.5
    assert provider.get_api_key() == "env-key"
