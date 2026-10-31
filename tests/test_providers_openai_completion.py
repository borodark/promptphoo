"""OpenAI legacy Completions provider. Ports deterministic cases from
test/providers/openai/completion.test.ts, with HTTP mocked by respx."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest
import respx
from openai_helpers import ok, openai_env, sent_json

from promptphoo.providers.openai import OpenAiCompletionProvider

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

COMPLETIONS_URL = "https://api.openai.com/v1/completions"
GATEWAY_URL = "https://gateway.example/v1/completions"


def missing_key_message(env_var: str = "OPENAI_API_KEY") -> str:
    return (
        f"API key is not set. Set the {env_var} environment variable or add `apiKey` to the "
        "provider config."
    )


def completion_response(
    text: str = "Test output", prompt: int = 5, completion: int = 5
) -> dict[str, Any]:
    return {
        "choices": [{"text": text}],
        "usage": {
            "total_tokens": prompt + completion,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        },
    }


async def test_rejects_codex_only_passthrough_model_before_dispatch():
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(COMPLETIONS_URL).mock(return_value=ok(completion_response()))
        provider = OpenAiCompletionProvider(
            "gpt-3.5-turbo-instruct",
            {"config": {"apiKey": "test-key", "passthrough": {"model": "gpt-5.3-codex-spark"}}},
        )
        with pytest.raises(ValueError, match="only available through openai:codex-sdk"):
            await provider.call_api("Test prompt")
        assert route.call_count == 0


@respx.mock
async def test_call_api_successfully_with_text_completion():
    route = respx.post(COMPLETIONS_URL).mock(return_value=ok(completion_response()))
    result = await OpenAiCompletionProvider("text-davinci-003").call_api("Test prompt")
    assert route.call_count == 1
    assert result["output"] == "Test output"
    assert result["tokenUsage"] == {"total": 10, "prompt": 5, "completion": 5, "numRequests": 1}
    assert sent_json(route) == {
        "model": "text-davinci-003",
        "prompt": "Test prompt",
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "best_of": 1,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }


@respx.mock
async def test_passthrough_overrides_completion_request_parameters():
    route = respx.post(COMPLETIONS_URL).mock(return_value=ok(completion_response()))
    provider = OpenAiCompletionProvider(
        "text-davinci-003",
        {
            "config": {
                "max_tokens": 50,
                "temperature": 0.2,
                "stop": ["configured-stop"],
                "presence_penalty": 0.1,
                "frequency_penalty": 0.2,
                "passthrough": {
                    "max_tokens": 120,
                    "temperature": 0.8,
                    "top_p": 0.6,
                    "stop": ["passthrough-stop"],
                    "presence_penalty": 0.7,
                    "frequency_penalty": 0.9,
                },
            }
        },
    )
    await provider.call_api("Test prompt")
    body = sent_json(route)
    assert {k: body[k] for k in ("max_tokens", "temperature", "top_p", "stop")} == {
        "max_tokens": 120,
        "temperature": 0.8,
        "top_p": 0.6,
        "stop": ["passthrough-stop"],
    }
    assert body["presence_penalty"] == 0.7
    assert body["frequency_penalty"] == 0.9


@respx.mock
async def test_completion_settings_resolved_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_STOP", '["environment-stop"]')
    monkeypatch.setenv("OPENAI_PRESENCE_PENALTY", "0.35")
    monkeypatch.setenv("OPENAI_FREQUENCY_PENALTY", "0.65")
    route = respx.post(COMPLETIONS_URL).mock(return_value=ok(completion_response()))
    await OpenAiCompletionProvider(
        "text-davinci-003", {"config": {"stop": ["configured-stop"]}}
    ).call_api("Test prompt")
    body = sent_json(route)
    assert body["stop"] == ["environment-stop"]
    assert body["presence_penalty"] == 0.35
    assert body["frequency_penalty"] == 0.65


@respx.mock
async def test_string_and_mixed_stop_passthrough_sent_verbatim():
    route = respx.post(COMPLETIONS_URL).mock(return_value=ok(completion_response()))
    await OpenAiCompletionProvider(
        "text-davinci-003", {"config": {"passthrough": {"stop": "passthrough-stop"}}}
    ).call_api("Test prompt")
    assert sent_json(route)["stop"] == "passthrough-stop"
    await OpenAiCompletionProvider(
        "text-davinci-003", {"config": {"passthrough": {"stop": ["valid-stop", 123]}}}
    ).call_api("Other prompt")
    assert sent_json(route)["stop"] == ["valid-stop", 123]


@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate"),
    [
        ("babbage-002", 0.4, 0.4),
        ("davinci-002", 2, 2),
        ("ft:babbage-002:company::model", 1.6, 1.6),
        ("ft:davinci-002:company::model", 12, 12),
    ],
)
@respx.mock
async def test_call_and_price_supported_completions_models(model, input_rate, output_rate):
    route = respx.post(COMPLETIONS_URL).mock(
        return_value=ok(completion_response(prompt=2_000, completion=1_000))
    )
    result = await OpenAiCompletionProvider(model).call_api("Test prompt")
    body = sent_json(route)
    assert body["model"] == model
    assert body["prompt"] == "Test prompt"
    assert result["cost"] == pytest.approx((2_000 * input_rate + 1_000 * output_rate) / 1e6)


@respx.mock
async def test_bills_qualified_passthrough_model_through_custom_gateway():
    route = respx.post(GATEWAY_URL).mock(
        return_value=ok(completion_response(prompt=1_000_000, completion=1_000_000))
    )
    provider = OpenAiCompletionProvider(
        "davinci-002",
        {
            "config": {
                "apiBaseUrl": "https://gateway.example/v1",
                "passthrough": {"model": "openai/babbage-002"},
            }
        },
    )
    result = await provider.call_api("Test prompt")
    assert sent_json(route)["model"] == "openai/babbage-002"
    assert result["cost"] == pytest.approx(0.8)


@respx.mock
async def test_no_openai_pricing_for_other_gateway_namespace():
    route = respx.post(GATEWAY_URL).mock(
        return_value=ok(completion_response(prompt=1_000_000, completion=1_000_000))
    )
    provider = OpenAiCompletionProvider(
        "davinci-002",
        {
            "config": {
                "apiBaseUrl": "https://gateway.example/v1",
                "passthrough": {"model": "vendor/babbage-002"},
            }
        },
    )
    result = await provider.call_api("Test prompt")
    assert sent_json(route)["model"] == "vendor/babbage-002"
    assert "cost" not in result


@pytest.mark.parametrize(
    ("config", "prompt_config", "expected_tier", "expected_cost"),
    [
        ({"service_tier": "fast"}, None, "fast", (1_000 * 0.45 + 100 * 3.6) / 1e6),
        (
            {"service_tier": "flex", "passthrough": {"service_tier": "priority"}},
            None,
            "priority",
            (1_000 * 0.45 + 100 * 3.6) / 1e6,
        ),
        (
            {"passthrough": {"service_tier": "priority"}},
            {"passthrough": {}},
            None,
            (1_000 * 0.25 + 100 * 2) / 1e6,
        ),
        (
            {"service_tier": "default", "passthrough": {"service_tier": "priority"}},
            {"service_tier": "flex"},
            "flex",
            (1_000 * 0.125 + 100 * 1) / 1e6,
        ),
    ],
    ids=["fast", "passthrough-priority", "prompt-clears-passthrough", "prompt-direct-flex"],
)
@respx.mock
async def test_gateway_service_tier_request_and_billing(
    config, prompt_config, expected_tier, expected_cost
):
    route = respx.post(GATEWAY_URL).mock(
        return_value=ok(completion_response("Output", prompt=1_000, completion=100))
    )
    provider = OpenAiCompletionProvider(
        "gpt-5-mini", {"config": {"apiBaseUrl": "https://gateway.example/v1", **config}}
    )
    context = {"prompt": {"config": prompt_config}} if prompt_config is not None else None
    result = await provider.call_api("Answer", context)
    body = sent_json(route)
    if expected_tier is None:
        assert "service_tier" not in body
    else:
        assert body["service_tier"] == expected_tier
    assert result["cost"] == pytest.approx(expected_cost)


@pytest.mark.parametrize(
    "api_base_url",
    [
        None,
        "https://us.api.openai.com/v1",
        "https://eu.api.openai.com/v1",
        "https://au.api.openai.com/v1",
        "https://ca.api.openai.com/v1",
        "https://jp.api.openai.com/v1",
        "https://in.api.openai.com/v1",
        "https://sg.api.openai.com/v1",
        "https://kr.api.openai.com/v1",
        "https://gb.api.openai.com/v1",
        "https://ae.api.openai.com/v1",
    ],
)
@respx.mock
async def test_omits_service_tiers_from_first_party_legacy_completion_requests(api_base_url):
    url = f"{api_base_url or 'https://api.openai.com/v1'}/completions"
    route = respx.post(url).mock(return_value=ok(completion_response()))
    config: dict[str, Any] = {
        "service_tier": "default",
        "passthrough": {"service_tier": "priority"},
    }
    if api_base_url is not None:
        config["apiBaseUrl"] = api_base_url
    provider = OpenAiCompletionProvider("babbage-002", {"config": config})
    result = await provider.call_api(
        "Use the legacy endpoint", {"prompt": {"config": {"service_tier": "flex"}}}
    )
    assert "service_tier" not in sent_json(route)
    assert result["cost"] == pytest.approx((5 * 0.4 + 5 * 0.4) / 1e6)


@respx.mock
async def test_handles_api_errors():
    respx.post(COMPLETIONS_URL).mock(
        return_value=ok({"error": {"message": "Test error", "type": "test_error"}}, status=400)
    )
    result = await OpenAiCompletionProvider("text-davinci-003").call_api("Test prompt")
    assert "Test error" in result["error"]


@respx.mock
async def test_handles_fetch_errors():
    respx.post(COMPLETIONS_URL).mock(side_effect=httpx.ConnectError("Network error"))
    provider = OpenAiCompletionProvider("text-davinci-003", {"config": {"maxRetries": 0}})
    result = await provider.call_api("Test prompt")
    assert result["error"].startswith("API call error:")
    assert "Network error" in result["error"]


@pytest.mark.parametrize("data", [None, {}, {"choices": []}])
@respx.mock
async def test_clean_error_on_missing_completion_choice(data):
    respx.post(COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, content=json.dumps(data).encode())
    )
    result = await OpenAiCompletionProvider("text-davinci-003").call_api("Test prompt")
    assert "Malformed response data" in result["error"]
    assert "TypeError" not in result["error"]


async def test_missing_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    provider = OpenAiCompletionProvider(
        "text-davinci-003", {"config": {"apiKeyRequired": True}, "env": {}}
    )
    with pytest.raises(ValueError) as excinfo:
        await provider.call_api("Test prompt")
    assert str(excinfo.value) == missing_key_message()


async def test_custom_api_key_envar_in_missing_key_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("CUSTOM_OPENAI_KEY", raising=False)
    provider = OpenAiCompletionProvider(
        "text-davinci-003", {"config": {"apiKeyEnvar": "CUSTOM_OPENAI_KEY"}}
    )
    with pytest.raises(ValueError) as excinfo:
        await provider.call_api("Test prompt")
    assert str(excinfo.value) == missing_key_message("CUSTOM_OPENAI_KEY")


def test_warns_about_unknown_model(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        OpenAiCompletionProvider("unknown-model")
    assert "FYI: Using unknown OpenAI completion model: unknown-model" in caplog.messages


@respx.mock
async def test_handles_cached_responses():
    route = respx.post(COMPLETIONS_URL).mock(return_value=ok(completion_response()))
    provider = OpenAiCompletionProvider("text-davinci-003")
    first = await provider.call_api("Test prompt")
    second = await provider.call_api("Test prompt")
    assert route.call_count == 1
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["output"] == "Test output"
    assert second["tokenUsage"] == {"total": 10, "cached": 10}


@respx.mock
async def test_handles_responses_without_usage():
    respx.post(COMPLETIONS_URL).mock(return_value=ok({"choices": [{"text": "Test output"}]}))
    result = await OpenAiCompletionProvider("text-davinci-003").call_api("Test prompt")
    assert result["output"] == "Test output"
    assert result["tokenUsage"] == {}


@respx.mock
async def test_passes_custom_headers_from_config():
    route = respx.post(COMPLETIONS_URL).mock(return_value=ok(completion_response()))
    await OpenAiCompletionProvider(
        "text-davinci-003", {"config": {"headers": {"X-Test-Header": "test-value"}}}
    ).call_api("Test prompt")
    headers = route.calls[0].request.headers
    assert headers["Content-Type"] == "application/json"
    assert headers["X-OpenAI-Originator"] == "promptfoo"
    assert headers["X-Test-Header"] == "test-value"


@respx.mock
async def test_passes_passthrough_fields_in_body():
    route = respx.post(COMPLETIONS_URL).mock(return_value=ok(completion_response()))
    await OpenAiCompletionProvider(
        "text-davinci-003", {"config": {"passthrough": {"logprobs": 3}}}
    ).call_api("Test prompt")
    assert sent_json(route)["logprobs"] == 3


async def test_invalid_openai_stop_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_STOP", "{invalid json}")
    provider = OpenAiCompletionProvider("text-davinci-003", {"config": {"apiKey": "test-api-key"}})
    with pytest.raises(ValueError, match="OPENAI_STOP is not a valid JSON string"):
        await provider.call_api("test")
