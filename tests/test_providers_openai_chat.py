"""OpenAI Chat Completions provider. Ports deterministic cases from
test/providers/openai/chat.test.ts, with HTTP mocked by respx."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from openai_helpers import CHAT_URL, chat_response, ok, openai_env, sent_json

from promptphoo import cli_state
from promptphoo.providers.function_callbacks import (
    UnsupportedCallbackError,
    load_callback_from_file_url,
)
from promptphoo.providers.openai import OpenAiChatCompletionProvider

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

PROMPT = json.dumps([{"role": "user", "content": "Test prompt"}])


def _provider(model: str = "gpt-4o-mini", **config) -> OpenAiChatCompletionProvider:
    return OpenAiChatCompletionProvider(model, {"config": config})


async def _body(model: str, **config) -> dict:
    return (await _provider(model, **config).get_openai_body("Test prompt"))["body"]


@respx.mock
async def test_call_api_successfully():
    route = respx.post(CHAT_URL).mock(return_value=ok(chat_response()))
    result = await _provider().call_api(PROMPT)
    assert route.call_count == 1
    headers = route.calls[0].request.headers
    assert headers["X-OpenAI-Originator"] == "promptfoo"
    assert headers["Authorization"] == "Bearer test-api-key"
    assert result["output"] == "Test output"
    assert result["tokenUsage"] == {"total": 10, "prompt": 5, "completion": 5, "numRequests": 1}
    assert result["guardrails"] == {"flagged": False}
    assert sent_json(route) == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Test prompt"}],
        "max_tokens": 1024,
        "temperature": 0,
    }


@respx.mock
async def test_cost_and_http_metadata():
    headers = {"x-request-id": "test-request-123"}
    respx.post(CHAT_URL).mock(return_value=ok(chat_response(), headers=headers))
    result = await _provider().call_api(PROMPT)
    assert result["cost"] == pytest.approx(5 * 0.15e-6 + 5 * 0.6e-6, abs=1e-12)
    http = result["metadata"]["http"]
    assert http["status"] == 200
    assert http["statusText"] == "OK"
    assert http["headers"]["x-request-id"] == "test-request-123"
    assert "choices" not in result["metadata"]


@respx.mock
async def test_cache_hit_counts_cached_tokens_and_zero_cost():
    route = respx.post(CHAT_URL).mock(return_value=ok(chat_response("Test output 2")))
    provider = _provider()
    first = await provider.call_api(PROMPT)
    second = await provider.call_api(PROMPT)
    assert route.call_count == 1
    assert first["tokenUsage"] == {"total": 10, "prompt": 5, "completion": 5, "numRequests": 1}
    assert second["output"] == "Test output 2"
    assert second["cached"] is True
    assert second["tokenUsage"] == {"total": 10, "cached": 10}
    assert second["cost"] == 0


@respx.mock
async def test_all_choices_in_metadata_when_n_gt_1():
    choices = [{"message": {"content": f"r{i}"}, "index": i} for i in range(3)]
    respx.post(CHAT_URL).mock(return_value=ok({"choices": choices, "usage": {"total_tokens": 30}}))
    result = await _provider().call_api(PROMPT)
    assert result["output"] == "r0"
    assert result["metadata"]["choices"] == choices


@respx.mock
async def test_non_refusal_400_is_error():
    payload = {
        "error": {
            "message": "Invalid request: 'messages' field is required",
            "type": "invalid_request_error",
            "param": "messages",
            "code": None,
        }
    }
    respx.post(CHAT_URL).mock(return_value=ok(payload, status=400))
    result = await _provider().call_api("Invalid request format")
    assert result["error"].startswith("API error: 400 Bad Request\n")
    assert "'messages' field is required" in result["error"]
    assert "output" not in result
    assert result["metadata"]["http"]["status"] == 400


@respx.mock
async def test_invalid_prompt_400_is_refusal():
    payload = {
        "error": {
            "message": "Invalid prompt: we've limited access to this content for safety reasons.",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_prompt",
        }
    }
    respx.post(CHAT_URL).mock(return_value=ok(payload, status=400))
    result = await _provider().call_api(PROMPT)
    assert "error" not in result
    assert "limited access to this content for safety reasons" in result["output"]
    assert "400 Bad Request" in result["output"]
    assert result["isRefusal"] is True
    assert result["guardrails"] == {"flagged": True, "flaggedInput": True}


@respx.mock
async def test_network_error_string():
    respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("boom"))
    result = await _provider(maxRetries=0).call_api(PROMPT)
    assert result["error"].startswith("API call error: ")
    assert result["metadata"]["http"]["status"] == 0
    assert result["metadata"]["http"]["statusText"] == "Error"


@respx.mock
async def test_refusal_message():
    respx.post(CHAT_URL).mock(
        return_value=ok(
            {
                "choices": [
                    {"message": {"refusal": "Content policy violation"}, "finish_reason": "stop"}
                ],
                "usage": {"total_tokens": 5, "prompt_tokens": 5, "completion_tokens": 0},
            }
        )
    )
    result = await _provider().call_api(PROMPT)
    assert result["output"] == "Content policy violation"
    assert result["isRefusal"] is True
    assert result["guardrails"] == {"flagged": True}
    assert result["tokenUsage"] == {"total": 5, "prompt": 5, "completion": 0, "numRequests": 1}


@respx.mock
async def test_content_filter_finish_reason():
    respx.post(CHAT_URL).mock(
        return_value=ok(
            {
                "choices": [{"message": {"content": None}, "finish_reason": "content_filter"}],
                "usage": {"total_tokens": 10, "prompt_tokens": 10, "completion_tokens": 0},
            }
        )
    )
    result = await _provider().call_api(PROMPT)
    assert result["output"] == "Content filtered by provider"
    assert result["isRefusal"] is True
    assert result["finishReason"] == "content_filter"
    assert result["guardrails"] == {"flagged": True}


@pytest.mark.parametrize(
    ("raw", "expected"), [("function_call", "tool_calls"), ("LENGTH", "length"), ("", None)]
)
@respx.mock
async def test_finish_reason_normalization(raw, expected):
    respx.post(CHAT_URL).mock(return_value=ok(chat_response("done", finish_reason=raw)))
    result = await _provider().call_api("hi")
    assert result.get("finishReason") == expected


@respx.mock
async def test_tool_calls_output():
    tool_calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    route = respx.post(CHAT_URL).mock(
        return_value=ok(chat_response({"content": None, "tool_calls": tool_calls}))
    )
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    result = await _provider(tools=tools, tool_choice="auto").call_api("call f")
    assert result["output"] == tool_calls
    body = sent_json(route)
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


def _external_tool_response() -> dict:
    call = {"function": {"name": "external_function", "arguments": '{"param": "test_value"}'}}
    return {
        "choices": [{"message": {"content": None, "tool_calls": [call]}}],
        "usage": {"total_tokens": 15, "prompt_tokens": 10, "completion_tokens": 5},
    }


_EXTERNAL_TOOL = {
    "type": "function",
    "function": {
        "name": "external_function",
        "parameters": {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"],
        },
    },
}


@respx.mock
async def test_python_function_tool_callback(tmp_path):
    callbacks = tmp_path / "callbacks.py"
    callbacks.write_text("def test_function(args):\n    return 'External result: ' + args\n")
    respx.post(CHAT_URL).mock(return_value=ok(_external_tool_response()))
    provider = _provider(
        tools=[_EXTERNAL_TOOL],
        functionToolCallbacks={"external_function": "file://callbacks.py:test_function"},
    )
    with cli_state.with_base_path(str(tmp_path)):
        result = await provider.call_api("Call external function")
    assert result["output"] == 'External result: {"param": "test_value"}'
    assert result["tokenUsage"] == {"total": 15, "prompt": 10, "completion": 5, "numRequests": 1}


@respx.mock
async def test_inline_python_callable_callback():
    respx.post(CHAT_URL).mock(return_value=ok(_external_tool_response()))
    provider = _provider(
        tools=[_EXTERNAL_TOOL],
        functionToolCallbacks={"external_function": lambda args: {"echo": json.loads(args)}},
    )
    result = await provider.call_api("Call external function")
    assert result["output"] == '{"echo":{"param":"test_value"}}'


@respx.mock
async def test_js_function_tool_callback_rejected():
    respx.post(CHAT_URL).mock(return_value=ok(_external_tool_response()))
    provider = _provider(
        tools=[_EXTERNAL_TOOL],
        functionToolCallbacks={"external_function": "file://callbacks.js:fn"},
    )
    with pytest.raises(UnsupportedCallbackError, match="Use a Python file instead"):
        load_callback_from_file_url("file://callbacks.js:fn")
    with pytest.raises(UnsupportedCallbackError, match="not supported by promptphoo"):
        await provider.execute_function_callback(
            "external_function", "{}", {"functionToolCallbacks": {"external_function": "x => x"}}
        )
    # A failing callback falls back to the raw tool calls, as upstream does.
    result = await provider.call_api("Call external function")
    assert result["output"] == _external_tool_response()["choices"][0]["message"]["tool_calls"]


@respx.mock
async def test_prompt_response_format_wins_and_json_schema_parsed():
    schema_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "test_schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"key2": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    }
    route = respx.post(CHAT_URL).mock(return_value=ok(chat_response('{"key2": "value2"}')))
    provider = _provider(response_format={"type": "json_object"})
    prompt = {"raw": "Test prompt", "label": "t", "config": {"response_format": schema_format}}
    result = await provider.call_api("Test prompt", {"vars": {}, "prompt": prompt})
    assert sent_json(route)["response_format"] == schema_format
    assert result["output"] == {"key2": "value2"}


@pytest.mark.parametrize(
    ("model", "kept"), [("gpt-5.4-mini", False), ("o3", False), ("gpt-4o", True)]
)
@respx.mock
async def test_passthrough_max_tokens_stripping(model, kept):
    route = respx.post(CHAT_URL).mock(return_value=ok(chat_response()))
    await _provider(model, passthrough={"max_tokens": 2000}).call_api("Test prompt")
    assert ("max_tokens" in sent_json(route)) is kept


async def test_reasoning_model_body():
    body = await _body("o1", reasoning_effort="high")
    assert body["reasoning_effort"] == "high"
    assert "temperature" not in body
    assert "max_tokens" not in body
    assert "reasoning_effort" not in await _body("gpt-4", reasoning_effort="high")
    o4 = await _body("o4-mini", reasoning_effort="medium", service_tier="premium")
    assert o4["reasoning_effort"] == "medium"
    assert o4["service_tier"] == "premium"


@pytest.mark.parametrize("model", ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
async def test_max_reasoning_effort_for_gpt56(model):
    body = await _body(model, reasoning_effort="max")
    assert body["reasoning_effort"] == "max"
    assert "temperature" not in body


async def test_max_completion_tokens_seed_stop():
    body = await _body("o3-mini", max_completion_tokens=100, seed=7, stop=["x"])
    assert body["max_completion_tokens"] == 100
    assert body["seed"] == 7
    assert body["stop"] == ["x"]


async def test_prompt_cache_options():
    body = await _body(
        "gpt-4o", prompt_cache_key="shared-prefix", prompt_cache_retention="in_memory"
    )
    assert body["prompt_cache_key"] == "shared-prefix"
    assert body["prompt_cache_retention"] == "in_memory"
    gpt56 = await _body("gpt-5.6", prompt_cache_options={"mode": "explicit", "ttl": "30m"})
    assert gpt56["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}


async def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="API key is not set. Set the OPENAI_API_KEY"):
        await _provider().call_api("Test prompt")


def test_config_not_shared():
    shared = {"max_tokens": 16000}
    a = OpenAiChatCompletionProvider("gpt-4.1-mini", {"config": shared})
    b = OpenAiChatCompletionProvider("gpt-5.4-mini", {"config": shared})
    a.config["max_tokens"] = 999
    assert b.config["max_tokens"] == 16000
    assert shared["max_tokens"] == 16000


@respx.mock
async def test_cached_prompt_tokens_details_and_cost():
    respx.post(CHAT_URL).mock(
        return_value=ok(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 2000,
                    "completion_tokens": 1000,
                    "total_tokens": 3000,
                    "prompt_tokens_details": {"cached_tokens": 500},
                    "completion_tokens_details": {"reasoning_tokens": 40},
                },
            }
        )
    )
    result = await _provider("gpt-4o").call_api(PROMPT)
    assert result["tokenUsage"] == {
        "numRequests": 1,
        "total": 3000,
        "prompt": 2000,
        "completion": 1000,
        "completionDetails": {"reasoning": 40, "cacheReadInputTokens": 500},
    }
    # gpt-4o: $2.5/M input, $1.25/M cached input, $10/M output.
    assert result["cost"] == pytest.approx((1500 * 2.5 + 500 * 1.25 + 1000 * 10) / 1e6, abs=1e-12)


@respx.mock
async def test_custom_api_base_url_and_headers():
    route = respx.post("https://gateway.example/v1/chat/completions").mock(
        return_value=ok(chat_response())
    )
    config = {"apiBaseUrl": "https://gateway.example/v1", "apiKey": "k", "headers": {"X-C": "1"}}
    provider = OpenAiChatCompletionProvider("my-model", {"config": config})
    result = await provider.call_api("hi")
    assert result["output"] == "Test output"
    headers = route.calls[0].request.headers
    assert headers["Authorization"] == "Bearer k"
    assert headers["X-C"] == "1"
    assert "X-OpenAI-Originator" not in headers
    assert provider.id() == "my-model"
