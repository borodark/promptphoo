"""OpenAI Responses API provider. Ports deterministic cases from
test/providers/openai/responses/*.test.ts (request, errors, refusals, metadata, reasoning,
responseFormat, tools, callbacks, models, daybreak), with HTTP mocked by respx."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from openai_helpers import ok, openai_env, sent_json

from promptphoo import fetch
from promptphoo.providers.openai import OpenAiResponsesProvider

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

RESPONSES_URL = "https://api.openai.com/v1/responses"
GATEWAY = "https://gateway.example/v1"


def _provider(model: str = "gpt-4o", **config: Any) -> OpenAiResponsesProvider:
    return OpenAiResponsesProvider(model, {"config": {"apiKey": "test-key", **config}})


async def _body(model: str = "gpt-4o", context: Any = None, **config: Any) -> dict[str, Any]:
    return (await _provider(model, **config).get_openai_body("Test prompt", context))["body"]


def _prompt_context(**config: Any) -> dict[str, Any]:
    return {"vars": {}, "prompt": {"raw": "Test prompt", "label": "p", "config": config}}


def _message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _completed(text: str = "Test response", **extra: Any) -> dict[str, Any]:
    return {
        "id": "resp_abc123",
        "status": "completed",
        "model": "gpt-4o",
        "output": [_message(text)],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        **extra,
    }


def _mock(payload: Any, status: int = 200, url: str = RESPONSES_URL, **kw: Any) -> respx.Route:
    return respx.post(url).mock(return_value=ok(payload, status, **kw))


# -- request building ---------------------------------------------------------------------------


@respx.mock
async def test_formats_and_calls_the_responses_api():
    route = _mock(_completed("This is a test response"))
    result = await _provider().call_api("Test prompt")
    headers = route.calls[0].request.headers
    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["X-OpenAI-Originator"] == "promptfoo"
    assert sent_json(route) == {
        "model": "gpt-4o",
        "input": "Test prompt",
        "max_output_tokens": 1024,
        "temperature": 0,
        "text": {"format": {"type": "text"}},
    }
    assert "error" not in result
    assert result["output"] == "This is a test response"
    assert result["tokenUsage"] == {"total": 30, "prompt": 10, "completion": 20, "numRequests": 1}
    assert result["cost"] == pytest.approx(10 * 2.5e-6 + 20 * 10e-6, abs=1e-12)
    assert result["metadata"]["responseId"] == "resp_abc123"
    assert result["metadata"]["http"]["status"] == 200


async def test_instructions_temperature_top_p_store_include_and_cache_options():
    body = await _body(
        instructions="You are a helpful assistant",
        temperature=0.7,
        top_p=0.9,
        store=True,
        include=["web_search_call.results", "reasoning.encrypted_content"],
        prompt_cache_key="shared-prefix",
        prompt_cache_retention="24h",
        previous_response_id="resp_prev",
        truncation="auto",
        metadata={"run": "1"},
        parallel_tool_calls=0,
        max_tool_calls=3,
        user="u1",
        safety_identifier="sid",
    )
    assert body == {
        "model": "gpt-4o",
        "input": "Test prompt",
        "max_output_tokens": 1024,
        "temperature": 0.7,
        "instructions": "You are a helpful assistant",
        "top_p": 0.9,
        "max_tool_calls": 3,
        "include": ["web_search_call.results", "reasoning.encrypted_content"],
        "previous_response_id": "resp_prev",
        "text": {"format": {"type": "text"}},
        "truncation": "auto",
        "metadata": {"run": "1"},
        "parallel_tool_calls": False,
        "store": True,
        "user": "u1",
        "safety_identifier": "sid",
        "prompt_cache_key": "shared-prefix",
        "prompt_cache_retention": "24h",
    }
    gpt56 = await _body("gpt-5.6", prompt_cache_options={"mode": "explicit", "ttl": "30m"})
    assert gpt56["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}


async def test_zero_temperature_and_max_output_tokens_are_sent():
    body = await _body(temperature=0, max_output_tokens=0)
    assert body["temperature"] == 0
    assert body["max_output_tokens"] == 0


async def test_omit_defaults(monkeypatch):
    body = await _body(omitDefaults=True)
    assert "temperature" not in body and "max_output_tokens" not in body
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.5")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "2048")
    body = await _body(omitDefaults=True)
    assert body["temperature"] == 0.5
    assert body["max_output_tokens"] == 2048


async def test_strips_max_tokens_from_passthrough():
    body = await _body(max_output_tokens=512, passthrough={"max_tokens": 16000})
    assert "max_tokens" not in body
    assert body["max_output_tokens"] == 512


async def test_structured_and_object_inputs():
    provider = _provider()
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    ]
    body = (await provider.get_openai_body(json.dumps(messages)))["body"]
    assert body["input"] == [
        messages[0],
        {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
    ]
    object_input = json.dumps({"query": "What is the weather?", "context": "San Francisco"})
    assert (await provider.get_openai_body(object_input))["body"]["input"] == object_input


async def test_service_tier_precedence():
    provider = _provider("gpt-5.6", service_tier="flex", passthrough={"service_tier": "priority"})
    assert (await provider.get_openai_body("x"))["body"]["service_tier"] == "priority"
    prompt = await provider.get_openai_body("x", {"prompt": {"config": {"service_tier": "flex"}}})
    assert prompt["body"]["service_tier"] == "flex"
    assert prompt["config"]["service_tier"] == "flex"
    passthrough = await provider.get_openai_body(
        "x",
        {"prompt": {"config": {"service_tier": "flex", "passthrough": {"service_tier": "fast"}}}},
    )
    assert passthrough["body"]["service_tier"] == "priority"
    assert passthrough["config"]["service_tier"] == "fast"


@pytest.mark.parametrize(
    "access_programs",
    [{"cyber": "standard"}, {"cyber": "daybreak_blue"}, {"cyber": "daybreak_red"}, {}],
)
async def test_daybreak_access_programs(access_programs):
    body = await _body("gpt-6-sol", access_programs=access_programs)
    assert body["access_programs"] == access_programs


async def test_daybreak_access_program_overrides():
    assert "access_programs" not in await _body("gpt-6-sol")
    provider = _provider("gpt-6-sol", access_programs={"cyber": "daybreak_blue"})
    body = (await provider.get_openai_body("x", _prompt_context(access_programs={"cyber": "s"})))[
        "body"
    ]
    assert body["access_programs"] == {"cyber": "s"}
    assert provider.config["access_programs"] == {"cyber": "daybreak_blue"}
    body = await _body(
        "gpt-6-sol",
        access_programs={"cyber": "daybreak_blue"},
        passthrough={"access_programs": {"cyber": "standard"}},
    )
    assert body["access_programs"] == {"cyber": "standard"}


@respx.mock
async def test_lowercase_authorization_and_content_type_headers_win():
    route = _mock(_completed())
    await _provider(
        headers={
            "authorization": "Bearer gateway-key",
            "content-type": "application/json; charset=utf-8",
        }
    ).call_api("x")
    raw = route.calls[0].request.headers.raw
    names = [k.decode().lower() for k, _ in raw]
    assert names.count("authorization") == 1 and names.count("content-type") == 1
    headers = route.calls[0].request.headers
    assert headers["authorization"] == "Bearer gateway-key"
    assert headers["content-type"] == "application/json; charset=utf-8"


async def test_codex_only_prompt_override_rejected():
    with pytest.raises(ValueError, match="only available through openai:codex-sdk"):
        await _provider("gpt-4.1").get_openai_body(
            "x", _prompt_context(passthrough={"model": "openai/gpt-5.3-codex-spark"})
        )


@pytest.mark.parametrize("model", ["gpt-live-transcribe", "gpt-live-1"])
async def test_voice_models_rejected_before_request(model):
    with pytest.raises(ValueError):
        await _provider("gpt-4.1").call_api("Hi", _prompt_context(passthrough={"model": model}))


# -- response format ----------------------------------------------------------------------------


async def test_json_object_format():
    body = await _body(response_format={"type": "json_object"})
    assert body["text"] == {"format": {"type": "json_object"}}


async def test_json_schema_format_keeps_property_order():
    schema = {
        "type": "object",
        "properties": {"z_last": {"type": "string"}, "a_first": {"type": "string"}},
        "required": ["z_last", "a_first"],
        "additionalProperties": False,
    }
    body = await _body(
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "TestSchema", "strict": True, "schema": schema},
        }
    )
    assert body["text"] == {
        "format": {"type": "json_schema", "name": "TestSchema", "schema": schema, "strict": True}
    }
    assert list(body["text"]["format"]["schema"]["properties"]) == ["z_last", "a_first"]


async def test_json_schema_default_name_and_top_level_schema():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    body = await _body(response_format={"type": "json_schema", "schema": schema})
    assert body["text"]["format"]["name"] == "response_schema"
    assert body["text"]["format"]["schema"] == schema
    body = await _body(
        response_format={"type": "json_schema", "name": "Top", "json_schema": {"schema": schema}}
    )
    assert body["text"]["format"]["name"] == "Top"


async def test_text_format_and_verbosity():
    assert (await _body(response_format={"type": "text"}))["text"] == {"format": {"type": "text"}}
    assert (await _body("gpt-5-mini", verbosity="low"))["text"] == {
        "format": {"type": "text"},
        "verbosity": "low",
    }
    assert "verbosity" not in (await _body("gpt-4o", verbosity="low"))["text"]


async def test_response_format_loaded_from_file(tmp_path, monkeypatch):
    schema = {"type": "object", "properties": {"x": {"type": "number"}}}
    (tmp_path / "format.json").write_text(
        json.dumps({"type": "json_schema", "name": "FromFile", "schema": schema})
    )
    monkeypatch.chdir(tmp_path)
    body = await _body(response_format="file://format.json")
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "FromFile",
        "schema": schema,
        "strict": True,
    }


@respx.mock
async def test_json_schema_output_is_parsed():
    _mock(_completed('{"result": "success"}'))
    result = await _provider(
        response_format={"type": "json_schema", "json_schema": {"name": "S", "schema": {}}}
    ).call_api("x")
    assert result["output"] == {"result": "success"}


# -- reasoning ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "effort", "max_output_tokens"),
    [
        ("gpt-5.6", "max", 4000),
        ("gpt-5.6-luna", "max", 4000),
        ("o3", "high", 2000),
        ("o3-pro", "high", 2000),
        ("o4-mini", "medium", 1000),
        ("gpt-5-codex-mini", "medium", 1000),
    ],
)
async def test_reasoning_models(model, effort, max_output_tokens):
    body = await _body(model, reasoning_effort=effort, max_output_tokens=max_output_tokens)
    assert body["model"] == model
    assert body["reasoning"] == {"effort": effort}
    assert body["max_output_tokens"] == max_output_tokens
    assert "temperature" not in body


async def test_reasoning_summary_merges_with_effort():
    body = await _body("o3", reasoning_effort="high", reasoning={"summary": "auto"})
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    body = await _body("o3", reasoning={"effort": "{{ e }}"}, context={"vars": {"e": "low"}})
    assert body["reasoning"] == {"effort": "low"}


async def test_reasoning_ignored_for_non_reasoning_models():
    body = await _body("gpt-4o", reasoning_effort="high", reasoning={"summary": "auto"})
    assert "reasoning" not in body
    assert body["temperature"] == 0


async def test_gpt56_persisted_reasoning_and_pro_mode():
    body = await _body("gpt-5.6-sol", reasoning={"effort": "max", "context": "all_turns",
                                                 "mode": "pro"})  # fmt: skip
    assert body["reasoning"] == {"effort": "max", "context": "all_turns", "mode": "pro"}


@pytest.mark.parametrize("model", ["gpt-daybreak-blue-latest", "gpt-daybreak-red-latest"])
async def test_daybreak_reasoning_through_overrides(model):
    direct = await _body(model, reasoning_effort="high")
    assert direct["reasoning"] == {"effort": "high"}
    assert "temperature" not in direct and "max_output_tokens" not in direct
    for context, passthrough in ((None, {"model": model}), (_prompt_context(
            passthrough={"model": model}), None)):  # fmt: skip
        config: dict[str, Any] = {"reasoning_effort": "high"}
        if passthrough:
            config["passthrough"] = passthrough
        body = await _body("gpt-4.1", context, **config)
        assert body["model"] == model
        assert body["reasoning"] == {"effort": "high"}
        assert "temperature" not in body and "max_output_tokens" not in body


async def test_reasoning_max_output_tokens_env_defaults(monkeypatch):
    assert "max_output_tokens" not in await _body("o1")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "2048")
    assert (await _body("o1"))["max_output_tokens"] == 2048
    monkeypatch.setenv("OPENAI_MAX_COMPLETION_TOKENS", "4096")
    assert (await _body("o1"))["max_output_tokens"] == 4096


async def test_fine_tuned_o_series_is_reasoning():
    body = await _body("ft:o4-mini-2025-04-16:company::model")
    assert "max_output_tokens" not in body and "temperature" not in body


async def test_codex_mini_latest_is_reasoning():
    assert _provider("codex-mini-latest").is_reasoning_model()
    # Retired on the first-party API; gateways may still serve it.
    assert "temperature" not in await _body("codex-mini-latest", apiBaseUrl=GATEWAY)


async def test_top_p_only_without_reasoning_effort(monkeypatch):
    assert "top_p" not in await _body("o3", reasoning_effort="high", top_p=0.5)
    monkeypatch.setenv("OPENAI_TOP_P", "0.8")
    assert (await _body("gpt-4o"))["top_p"] == 0.8


@respx.mock
async def test_reasoning_output_tokens_and_summary():
    payload = {
        "id": "resp_r",
        "status": "completed",
        "model": "o1-pro",
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Thinking"}]},
            _message("Answer"),
        ],
        "usage": {
            "input_tokens": 15,
            "output_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 100},
            "input_tokens_details": {"cached_tokens": 5},
            "total_tokens": 45,
        },
    }
    route = _mock(payload)
    result = await _provider("o1-pro", reasoning_effort="medium").call_api("x")
    assert sent_json(route)["reasoning"] == {"effort": "medium"}
    assert result["output"] == "Reasoning: Thinking\nAnswer"
    assert result["tokenUsage"] == {
        "total": 45,
        "prompt": 15,
        "completion": 30,
        "numRequests": 1,
        "completionDetails": {"reasoning": 100, "cacheReadInputTokens": 5},
    }


@respx.mock
async def test_empty_reasoning_summary_is_skipped():
    _mock(
        {
            "output": [{"type": "reasoning", "summary": []}, _message("Final")],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    assert (await _provider("o3").call_api("x"))["output"] == "Final"


# -- deep research ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tools",
    [
        [{"type": "code_interpreter"}],
        [{"type": "file_search"}],
        [{"type": "file_search", "vector_store_ids": []}],
    ],
)
@respx.mock
async def test_deep_research_requires_a_data_source(tools):
    route = respx.post(f"{GATEWAY}/responses")
    result = await _provider("o3-deep-research", apiBaseUrl=GATEWAY, tools=tools).call_api("x")
    assert "requires at least one data source" in result["error"]
    assert "o3-deep-research" in result["error"]
    assert route.call_count == 0


@respx.mock
async def test_deep_research_mcp_requires_never_approval():
    tools = [{"type": "mcp", "server_label": "s", "server_url": "http://t", "require_approval":
              "auto"}]  # fmt: skip
    result = await _provider("o3-deep-research", apiBaseUrl=GATEWAY, tools=tools).call_api("x")
    assert "requires MCP tools to have require_approval: 'never'" in result["error"]


@pytest.mark.parametrize(
    "tools",
    [
        [{"type": "web_search_preview"}],
        [{"type": "web_search"}],
        [{"type": "file_search", "vector_store_ids": ["vs_123"]}],
        [{"type": "mcp", "server_label": "s", "server_url": "https://e/mcp",
          "require_approval": "never"}],
    ],
)  # fmt: skip
@respx.mock
async def test_deep_research_accepts_data_sources(tools):
    route = _mock(
        {"output": [_message("Research complete")], "usage": {"input_tokens": 1}},
        url=f"{GATEWAY}/responses",
    )
    result = await _provider("o3-deep-research", apiBaseUrl=GATEWAY, tools=tools).call_api("x")
    assert result["output"] == "Research complete"
    assert sent_json(route)["tools"] == tools


@pytest.mark.parametrize(
    ("model", "env", "expected"),
    [
        ("o3-deep-research", None, 600_000),
        ("gpt-5-pro", None, 600_000),
        ("gpt-5.2-pro", None, 600_000),
        ("gpt-5.5-pro", "120000", 120_000),
        ("gpt-4o", None, None),
    ],
)
@respx.mock
async def test_long_running_model_timeout(monkeypatch, model, env, expected):
    if env:
        monkeypatch.setenv("PROMPTFOO_EVAL_TIMEOUT_MS", env)
    seen: list[Any] = []
    original = fetch.fetch_with_cache

    async def spy(url, options=None, timeout=None, *args, **kwargs):
        seen.append(timeout)
        return await original(url, options, timeout, *args, **kwargs)

    monkeypatch.setattr(fetch, "fetch_with_cache", spy)
    _mock({"output": [_message("ok")]}, url=f"{GATEWAY}/responses")
    tools = [{"type": "web_search"}]
    await _provider(model, apiBaseUrl=GATEWAY, tools=tools).call_api("x")
    if expected is None:
        assert seen[0] != 600_000
    else:
        assert seen == [expected]


# -- errors and refusals ------------------------------------------------------------------------


@respx.mock
async def test_api_error_status():
    error = {"error": {"message": "Invalid request", "type": "invalid_request_error",
                       "code": "invalid_api_key"}}  # fmt: skip
    _mock(error, 400, headers={"x-request-id": "r1"})
    result = await _provider().call_api("x")
    serialized = json.dumps(error, separators=(",", ":"))
    assert result["error"] == f"API error: 400 Bad Request\n{serialized}"
    assert "output" not in result
    assert result["metadata"]["http"]["status"] == 400
    assert result["metadata"]["http"]["headers"]["x-request-id"] == "r1"


@respx.mock
async def test_invalid_prompt_is_a_refusal():
    error = {"error": {"message": "some random error message", "type": "invalid_request_error",
                       "param": None, "code": "invalid_prompt"}}  # fmt: skip
    _mock(error, 400)
    result = await _provider().call_api("How do I create harmful content?")
    assert "error" not in result
    assert "some random error message" in result["output"]
    assert "400 Bad Request" in result["output"]
    assert result["isRefusal"] is True


@respx.mock
async def test_error_in_200_payload():
    payload = {"error": {"message": "Content policy violation",
                         "type": "content_policy_violation", "code": "content_filter"}}  # fmt: skip
    _mock(payload)
    result = await _provider().call_api("x")
    assert result["error"].startswith(
        "API error: Content policy violation, Type: content_policy_violation, Code: content_filter"
    )
    assert result["metadata"]["http"]["status"] == 200


@respx.mock
async def test_missing_and_non_array_output():
    _mock({"id": "r", "status": "completed", "usage": {"input_tokens": 1}})
    result = await _provider().call_api("x")
    assert "Invalid response format: Missing output array" in result["error"]
    respx.post(RESPONSES_URL).mock(return_value=ok({"output": "not-an-array"}))
    result = await _provider().call_api("y")
    assert "Invalid response format" in result["error"]
    assert "output" not in result


@respx.mock
async def test_null_message_content_gives_empty_output():
    payload = {"id": "r", "status": "completed", "model": "gpt-4o",
               "output": [{"type": "message", "role": "assistant", "content": None}],
               "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}}  # fmt: skip
    _mock(payload)
    result = await _provider().call_api("x")
    assert result["output"] == ""
    assert result["raw"] == payload
    assert "error" not in result


@respx.mock
async def test_network_error():
    respx.post(RESPONSES_URL).mock(side_effect=httpx.ConnectError("Network error"))
    result = await _provider(maxRetries=0).call_api("x")
    assert result["error"].startswith("API call error:")
    assert "Network error" in result["error"]
    assert result["metadata"]["http"]["status"] == 0


@respx.mock
async def test_rate_limit_error_metadata():
    respx.post(RESPONSES_URL).mock(
        return_value=httpx.Response(
            429, json={"error": {"message": "Rate limit exceeded"}}, headers={"retry-after": "0"}
        )
    )
    result = await _provider(maxRetries=0).call_api("x")
    assert result["error"]
    assert result["metadata"]["http"]["status"] == 429


async def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        await OpenAiResponsesProvider("gpt-4o").call_api("x")
    provider = OpenAiResponsesProvider("gpt-4o", {"config": {"apiKeyEnvar": "CUSTOM_KEY"}})
    with pytest.raises(ValueError, match="CUSTOM_KEY"):
        await provider.call_api("x")


@respx.mock
async def test_explicit_refusals():
    payload = _completed()
    payload["output"] = [
        {"type": "message", "role": "assistant",
         "content": [{"type": "refusal", "refusal": "I cannot fulfill this request."}]}
    ]  # fmt: skip
    _mock(payload)
    result = await _provider().call_api("x")
    assert result["isRefusal"] is True
    assert result["output"] == "I cannot fulfill this request."
    payload["output"] = [{"type": "message", "role": "assistant", "refusal": "No."}]
    respx.post(RESPONSES_URL).mock(return_value=ok(payload))
    result = await _provider().call_api("y")
    assert result["isRefusal"] is True
    assert result["output"] == "No."


@respx.mock
async def test_gateway_policy_refusal():
    payload = {
        "error": {"message": "Blocked by policy", "code": "bio_policy",
                  "metadata": {"error_type": "refusal"}},
        "usage": {"input_tokens": 3, "output_tokens": 0, "total_tokens": 3},
    }  # fmt: skip
    _mock(payload, 400, url=f"{GATEWAY}/responses")
    result = await _provider(apiBaseUrl=GATEWAY).call_api("x")
    assert result["isRefusal"] is True
    assert result["output"] == "Blocked by policy"
    assert result["guardrails"] == {
        "flagged": True,
        "flaggedInput": True,
        "reason": "Blocked by policy",
    }
    assert result["metadata"]["providerPolicy"] == {"code": "bio_policy"}
    assert result["metadata"]["http"]["status"] == 400
    assert result["tokenUsage"]["total"] == 3


# -- metadata, incomplete status, background ----------------------------------------------------


@respx.mock
async def test_http_metadata_and_status():
    headers = {"x-request-id": "test-request-123", "x-litellm-model-group": "gpt-4o"}
    _mock(_completed(), headers=headers)
    result = await _provider().call_api("x")
    http = result["metadata"]["http"]
    assert (http["status"], http["statusText"]) == (200, "OK")
    assert http["headers"]["x-request-id"] == "test-request-123"
    assert result["metadata"]["responseStatus"] == "completed"
    assert "incompleteReason" not in result["metadata"]
    assert "finishReason" not in result


@respx.mock
async def test_incomplete_output_and_cache_hit():
    payload = {
        "id": "resp_incomplete",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "model": "gpt-4o",
        "output": [_message("Partial answer")],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    }
    route = _mock(payload)
    provider = _provider()
    for cached in (False, True):
        result = await provider.call_api("Test prompt")
        assert "error" not in result
        assert result["output"] == "Partial answer"
        assert result["raw"] == payload
        assert result["finishReason"] == "length"
        assert result["metadata"]["responseStatus"] == "incomplete"
        assert result["metadata"]["incompleteReason"] == "max_output_tokens"
        assert result["cached"] is cached
        assert result["tokenUsage"] == (
            {"cached": 30, "total": 30, "numRequests": 1}
            if cached
            else {"prompt": 10, "completion": 20, "total": 30, "numRequests": 1}
        )
    assert route.call_count == 1


@respx.mock
async def test_truncation_info_kept_in_raw():
    _mock(_completed(truncation={"tokens_truncated": 100}))
    result = await _provider().call_api("x")
    assert result["raw"]["truncation"]["tokens_truncated"] == 100


@pytest.mark.parametrize(
    ("data", "error"),
    [
        ({"id": "resp_c", "status": "cancelled", "error": None, "output": []},
         "Background response resp_c was cancelled."),
        ({"id": "resp_f", "status": "failed", "error": None, "output": []},
         "Background response resp_f was failed."),
        ({"id": "resp_f", "status": "failed", "output": [],
          "error": {"code": "server_error", "message": "internal error"}},
         "Background response resp_f failed: internal error"),
        ({"status": "queued", "output": []}, "Background response is missing its response ID."),
    ],
)  # fmt: skip
@respx.mock
async def test_background_terminal_states(data, error):
    route = _mock(data)
    provider = _provider("gpt-4.1", background=True)
    result = await provider.call_api("Background task")
    assert result["error"].startswith(error)
    assert sent_json(route)["background"] is True
    # The failed response was evicted, so a retry calls the API again.
    await provider.call_api("Background task")
    assert route.call_count == 2


@respx.mock
async def test_background_polling_not_supported():
    _mock({"id": "resp_q", "status": "queued", "output": []})
    result = await _provider("gpt-4.1", background=True).call_api("x")
    assert "polling background responses is not supported by promptphoo" in result["error"]


@respx.mock
async def test_background_completed_synchronously():
    _mock(_completed("done"))
    result = await _provider("gpt-4.1", background=True).call_api("x")
    assert result["output"] == "done"


async def test_streaming_rejected():
    with pytest.raises(ValueError, match="Streaming Responses API requests"):
        await _provider(stream=True).call_api("x")


# -- tools and callbacks ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "web_search", "search_context_size": "low",
         "filters": {"allowed_domains": ["developers.openai.com"]},
         "user_location": {"type": "approximate", "country": "US"},
         "external_web_access": False, "return_token_budget": "unlimited"},
        {"type": "web_search_preview", "search_context_size": "high"},
        {"type": "file_search", "vector_store_ids": ["vs_1"]},
        {"type": "code_interpreter", "container": {"type": "auto"}},
        {"type": "mcp", "server_label": "s", "server_url": "https://e/mcp",
         "require_approval": "never"},
    ],
)  # fmt: skip
@respx.mock
async def test_hosted_tools_pass_through(tool):
    route = _mock({"id": "resp_search", "status": "completed",
                   "output": [_message("Found documentation.")]})  # fmt: skip
    result = await _provider("gpt-5.6-sol", tools=[tool]).call_api("Find docs")
    assert sent_json(route)["tools"] == [tool]
    assert result["output"] == "Found documentation."


async def test_function_tools_and_tool_choice_are_flattened():
    tool = {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
    body = await _body(
        tools=[tool], tool_choice={"type": "function", "function": {"name": "get_weather"}}
    )
    assert body["tools"] == [
        {"type": "function", "name": "get_weather", "parameters": {"type": "object"}}
    ]
    assert body["tool_choice"] == {"type": "function", "name": "get_weather"}
    assert await _body(tool_choice="auto") == {**(await _body()), "tool_choice": "auto"}


async def test_tools_loaded_from_file(tmp_path, monkeypatch):
    tools = [{"type": "function", "name": "f", "parameters": {"type": "object"}}]
    (tmp_path / "tools.json").write_text(json.dumps(tools))
    monkeypatch.chdir(tmp_path)
    prepared = await _provider(tools="file://tools.json").get_openai_body("x")
    assert prepared["body"]["tools"] == tools
    assert prepared["config"]["tools"] == tools
    assert (await _provider().get_openai_body("x"))["config"]["tools"] is None


@respx.mock
async def test_function_callbacks():
    def add_numbers(args: str) -> str:
        values = json.loads(args)
        return json.dumps(values["a"] + values["b"])

    payload = _completed()
    payload["output"] = [
        {"type": "function_call", "name": "addNumbers", "id": "call_123",
         "arguments": '{"a": 5, "b": 6}'},
        {"type": "function_call", "status": "completed", "name": "addNumbers",
         "arguments": "{}", "call_id": "call_123"},
    ]  # fmt: skip
    _mock(payload)
    provider = _provider(functionToolCallbacks={"addNumbers": add_numbers})
    result = await provider.call_api("Add 5 and 6")
    first, second = result["output"].split("\n", 1)
    assert first == "11"
    assert json.loads(second)["status"] == "no_arguments_provided"


@respx.mock
async def test_function_callback_failure_falls_back_to_call():
    payload = _completed()
    call = {"type": "function_call", "name": "addNumbers", "id": "c", "arguments": "invalid json"}
    payload["output"] = [call]
    _mock(payload)
    provider = _provider(functionToolCallbacks={"addNumbers": lambda args: json.loads(args)})
    result = await provider.call_api("x")
    assert "error" not in result
    assert json.loads(result["output"]) == call


@respx.mock
async def test_function_call_without_callback_is_stringified():
    payload = _completed()
    call = {"type": "function_call", "name": "get_weather", "arguments": '{"city":"SF"}'}
    payload["output"] = [call]
    _mock(payload)
    assert json.loads((await _provider().call_api("x"))["output"]) == call


@respx.mock
async def test_callback_in_message_content():
    payload = _completed()
    payload["output"] = [
        {"type": "message", "role": "assistant", "content": [
            {"type": "function_call", "name": "lookup", "arguments": '{"q":"x"}'}]}
    ]  # fmt: skip
    _mock(payload)
    provider = _provider(functionToolCallbacks={"lookup": lambda args: "found"})
    assert (await provider.call_api("x"))["output"] == "found"


@respx.mock
async def test_tool_result_items_are_stringified():
    payload = _completed()
    payload["output"] = [{"type": "tool_result", "id": "call_123", "output": "Sunny"}]
    _mock(payload)
    parsed = json.loads((await _provider().call_api("x"))["output"])
    assert parsed == {"type": "tool_result", "id": "call_123", "output": "Sunny"}


# -- billing ------------------------------------------------------------------------------------

USAGE = {"input_tokens": 1_000, "output_tokens": 100, "total_tokens": 1_100}


@respx.mock
async def test_observable_web_and_file_search_fees():
    _mock({"id": "r", "status": "completed", "model": "gpt-5-mini", "usage": USAGE, "output": [
        {"type": "web_search_call", "action": {"type": "search", "query": "p"},
         "status": "completed"},
        {"type": "file_search_call", "status": "completed"}]})  # fmt: skip
    result = await _provider("gpt-5-mini", tools=[{"type": "web_search_preview"}]).call_api("x")
    assert result["cost"] == pytest.approx((1_000 * 0.25 + 100 * 2) / 1e6 + 0.0125, abs=1e-10)


@respx.mock
async def test_web_search_fee_from_effective_passthrough_tools():
    _mock({"id": "r", "status": "completed", "model": "gpt-4o",
           "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
           "output": [{"type": "web_search_call", "action": {"type": "search", "query": "p"},
                       "status": "completed"}]})  # fmt: skip
    result = await _provider(
        tools=[{"type": "web_search_preview"}], passthrough={"tools": [{"type": "web_search"}]}
    ).call_api("x")
    assert result["cost"] == pytest.approx(0.01, abs=1e-10)


@respx.mock
async def test_cost_uses_returned_service_tier():
    _mock({"id": "r", "status": "completed", "model": "gpt-5-mini", "service_tier": "priority",
           "output": [_message("Done")], "usage": USAGE})  # fmt: skip
    result = await _provider("gpt-5-mini", service_tier="flex").call_api("x")
    assert result["cost"] == pytest.approx((1_000 * 0.45 + 100 * 3.6) / 1e6, abs=1e-10)


@respx.mock
async def test_no_code_interpreter_session_fee():
    _mock({"id": "r", "status": "completed", "model": "gpt-5-mini", "usage": USAGE,
           "output": [{"type": "code_interpreter_call", "container_id": "c",
                       "status": "completed"}]})  # fmt: skip
    result = await _provider("gpt-5-mini", tools=[{"type": "code_interpreter"}]).call_api("x")
    assert result["cost"] == pytest.approx((1_000 * 0.25 + 100 * 2) / 1e6, abs=1e-10)


BIG_USAGE = {
    "input_tokens": 1_000_000,
    "input_tokens_details": {"cache_write_tokens": 0},
    "output_tokens": 1_000_000,
    "total_tokens": 2_000_000,
}


@respx.mock
async def test_effective_passthrough_model_for_capabilities_and_billing():
    route = _mock({"id": "r", "status": "completed", "model": "gpt-5.6-luna",
                   "output": [_message("ok")], "usage": BIG_USAGE})  # fmt: skip
    result = await _provider(
        "computer-use-preview",
        omitDefaults=True,
        passthrough={"model": "gpt-5.6-luna"},
        reasoning_effort="high",
        temperature=0,
    ).call_api("x")
    body = sent_json(route)
    assert body["model"] == "gpt-5.6-luna"
    assert body["reasoning"] == {"effort": "high"}
    assert "temperature" not in body
    assert result["cost"] == pytest.approx(2.2, abs=1e-10)


@pytest.mark.parametrize(
    ("model", "cost"), [("openai/gpt-5.6-luna", 2.2), ("vendor/gpt-5.6-luna", None)]
)
@respx.mock
async def test_gateway_passthrough_model_billing(model, cost):
    route = _mock({"id": "r", "status": "completed", "output": [_message("ok")],
                   "usage": BIG_USAGE}, url=f"{GATEWAY}/responses")  # fmt: skip
    result = await _provider(
        "computer-use-preview", apiBaseUrl=GATEWAY, passthrough={"model": model}
    ).call_api("x")
    assert sent_json(route)["model"] == model
    if cost is None:
        assert "cost" not in result
    else:
        assert result["cost"] == pytest.approx(cost, abs=1e-10)


# -- models and identity ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ["codex-mini-latest", "computer-use-preview", "gpt-5-chat-latest", "o1-mini",
     "o3-deep-research", "o4-mini-deep-research-2025-06-26"],
)  # fmt: skip
def test_retired_models_not_advertised(model):
    assert model not in OpenAiResponsesProvider.OPENAI_RESPONSES_MODEL_NAMES


def test_supported_model_names():
    names = OpenAiResponsesProvider.OPENAI_RESPONSES_MODEL_NAMES
    for model in ("o1-pro", "o3-pro", "gpt-4o", "o3-mini", "gpt-4.1", "gpt-5", "gpt-5-mini",
                  "gpt-5.6-sol", "gpt-6-sol", "gpt-6.1-sol", "gpt-5.5-pro", "gpt-5.4-nano",
                  "o3", "o4-mini-2025-04-16"):  # fmt: skip
        assert model in names


def test_id_label_and_to_json():
    provider = OpenAiResponsesProvider("gpt-5-mini", {"label": "Mini"})
    assert provider.id() == "openai:gpt-5-mini"
    assert provider.to_json() == {"id": "openai:gpt-5-mini", "label": "Mini"}
    assert str(provider) == "[OpenAI Provider gpt-5-mini]"
    custom = OpenAiResponsesProvider("gpt-5-mini", {"config": {"apiBaseUrl": GATEWAY}})
    assert custom.id() == "gpt-5-mini"
    assert OpenAiResponsesProvider("gpt-5-mini", {"id": "mine"}).id() == "mine"
