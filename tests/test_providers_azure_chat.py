"""Azure chat, completion and embedding providers against respx-mocked endpoints. Ports the
deterministic cases of test/providers/azure/{chat,completion,embedding}.test.ts and
configTests.ts and sharedChecks.ts."""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest
import respx
from azure_helpers import (
    API_VERSION,
    HOST,
    api_options,
    azure_env,
    chat_response,
    chat_url,
    ok,
    sent_json,
)

from promptphoo import fetch
from promptphoo.fetch.errors import HttpRateLimitError
from promptphoo.providers.azure import (
    AzureChatCompletionProvider,
    AzureCompletionProvider,
    AzureEmbeddingProvider,
)

pytestmark = pytest.mark.usefixtures(azure_env.__name__)

USAGE = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
REQUIRED_SCHEMA = {
    "type": "object",
    "properties": {"test": {"type": "string"}},
    "required": ["test"],
    "additionalProperties": False,
}
SAMPLED = {"max_tokens": 512, "temperature": 0.5, "top_p": 0.9}


def chat(deployment: str = "test-deployment", **config: Any) -> AzureChatCompletionProvider:
    return AzureChatCompletionProvider(deployment, api_options(**config))


async def body_of(provider: AzureChatCompletionProvider, context: Any = None) -> dict[str, Any]:
    return (await provider.get_openai_body("test prompt", context))["body"]


def mock_chat(payload: Any, status: int = 200, deployment: str = "test-deployment") -> Any:
    return respx.post(chat_url(deployment)).mock(return_value=ok(payload, status))


def fail_fetch(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    async def raise_(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(fetch, "fetch_with_cache", raise_)


# --- config merging (configTests.ts) -----------------------------------------------------------


@pytest.fixture
def merging() -> AzureChatCompletionProvider:
    return chat(
        functions=[{"name": "provider_func", "parameters": {}}], max_tokens=100, temperature=0.5
    )


def _ctx(config: Any = None, vars: Any = None, **prompt: Any) -> dict[str, Any]:
    p: dict[str, Any] = {"label": "test prompt", "raw": "test prompt", **prompt}
    if config is not None:
        p["config"] = config
    return {"prompt": p, "vars": vars or {}}


@pytest.mark.parametrize("config", [None, {}])
async def test_uses_provider_config(merging, config):
    body = await body_of(merging, _ctx(config))
    assert body["functions"] == [{"name": "provider_func", "parameters": {}}]
    assert (body["max_tokens"], body["temperature"]) == (100, 0.5)


async def test_merges_prompt_config(merging):
    body = await body_of(
        merging,
        _ctx({"functions": [{"name": "prompt_func", "parameters": {}}], "temperature": 0.7}),
    )
    assert body["functions"] == [{"name": "prompt_func", "parameters": {}}]
    assert (body["max_tokens"], body["temperature"]) == (100, 0.7)


async def test_nested_config_merging(merging):
    tool_choice = {"function": {"name": "test"}, "type": "function"}
    body = await body_of(
        merging, _ctx({"response_format": {"type": "json_object"}, "tool_choice": tool_choice})
    )
    assert body["response_format"] == {"type": "json_object"}
    assert body["tool_choice"] == tool_choice


async def test_json_schema_response_format_and_vars(merging):
    fmt = {
        "type": "json_schema",
        "json_schema": {"name": "s", "strict": True, "schema": REQUIRED_SCHEMA},
    }
    assert (await body_of(merging, _ctx({"response_format": fmt})))["response_format"] == fmt
    templated = {"type": "json_schema", "json_schema": {"name": "{{schemaName}}", "schema": {}}}
    body = await body_of(
        merging, _ctx({"response_format": templated}, vars={"schemaName": "dynamic_schema"})
    )
    assert body["response_format"]["json_schema"]["name"] == "dynamic_schema"


async def test_default_body():
    body = await body_of(chat())
    assert body == {
        "model": "test-deployment",
        "messages": [{"role": "user", "content": "test prompt"}],
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }


async def test_omit_defaults():
    body = await body_of(chat(omitDefaults=True))
    for key in ("max_tokens", "temperature", "top_p", "presence_penalty", "frequency_penalty"):
        assert key not in body


async def test_omit_defaults_uses_env(azure_env):
    for name, value in [
        ("OPENAI_TEMPERATURE", "0.5"),
        ("OPENAI_MAX_TOKENS", "2048"),
        ("OPENAI_TOP_P", "0.9"),
        ("OPENAI_PRESENCE_PENALTY", "0.1"),
        ("OPENAI_FREQUENCY_PENALTY", "0.2"),
    ]:
        azure_env.setenv(name, value)
    body = await body_of(chat(omitDefaults=True))
    assert (body["max_tokens"], body["temperature"], body["top_p"]) == (2048, 0.5, 0.9)
    assert (body["presence_penalty"], body["frequency_penalty"]) == (0.1, 0.2)


async def test_system_prompt_injection():
    body = await body_of(chat(systemPrompt="Be brief"))
    assert body["messages"][0] == {"role": "system", "content": "Be brief"}
    provider = chat(systemPrompt="Replaced")
    prompt = '[{"role": "system", "content": "orig"}, {"role": "user", "content": "hi"}]'
    messages = (await provider.get_openai_body(prompt))["body"]["messages"]
    assert messages == [
        {"role": "system", "content": "Replaced"},
        {"role": "user", "content": "hi"},
    ]


async def test_data_sources_and_passthrough_in_body():
    sources = [{"type": "azure_search", "parameters": {"endpoint": "https://s", "index_name": "i"}}]
    body = await body_of(
        chat(data_sources=sources, deployment_id="dep", passthrough={"seed": 7, "user": "u"})
    )
    assert body["data_sources"] == sources
    assert body["deployment_id"] == "dep"
    assert (body["seed"], body["user"]) == (7, "u")


# --- reasoning models ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config", [{"o1": True}, {"isReasoningModel": True}, {"o1": False, "isReasoningModel": True}]
)
def test_reasoning_flags(config):
    assert AzureChatCompletionProvider("test-deployment", {"config": config}).is_reasoning_model()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mai-thinking-1", True),
        ("MAI-Thinking-1", True),
        ("mai-ds-r1", True),
        ("prod-mai-reasoning", True),
        ("mai-code-1-flash", False),
        ("o3-mini", True),
        ("prod-o1-mini", True),
        ("gpt-5-mini", True),
        ("gpt-4o", False),
        ("DeepSeek-R1", True),
        ("grok-4-reasoning", True),
        ("claude-opus-4-7", False),
    ],
)
def test_reasoning_detection(name, expected):
    assert AzureChatCompletionProvider(name, {"config": {}}).is_reasoning_model() is expected


@pytest.mark.parametrize(
    ("name", "deprecated"),
    [
        ("claude-opus-4-7", True),
        ("claude-opus-4-8", True),
        ("claude-opus-4-6-20260205", False),
        ("claude-opus-4-70", False),
        ("claude-opus-4-80", False),
        ("claude-fable-5", True),
    ],
)
def test_sampling_params_deprecated_claude(name, deprecated):
    provider = AzureChatCompletionProvider(name, {"config": {}})
    assert provider.is_sampling_params_deprecated_claude_model() is deprecated


async def test_reasoning_body_rules():
    body = await body_of(
        chat(isReasoningModel=True, max_completion_tokens=2000, max_tokens=1000, temperature=0.7)
    )
    assert body["max_completion_tokens"] == 2000
    assert "max_tokens" not in body and "temperature" not in body
    assert body["reasoning_effort"] == "medium"
    assert (await body_of(chat(isReasoningModel=True, reasoning_effort="high")))[
        "reasoning_effort"
    ] == "high"
    rendered = await body_of(
        chat(isReasoningModel=True, reasoning_effort="{{effort}}"),
        _ctx(vars={"effort": "high"}),
    )
    assert rendered["reasoning_effort"] == "high"


async def test_reasoning_omit_defaults(azure_env):
    body = await body_of(chat(isReasoningModel=True, omitDefaults=True))
    assert "max_completion_tokens" not in body and "reasoning_effort" not in body
    azure_env.setenv("OPENAI_MAX_TOKENS", "2048")
    assert (await body_of(chat(isReasoningModel=True, omitDefaults=True)))[
        "max_completion_tokens"
    ] == 2048
    azure_env.setenv("OPENAI_MAX_COMPLETION_TOKENS", "4096")
    body = await body_of(chat(isReasoningModel=True, omitDefaults=True))
    assert body["max_completion_tokens"] == 4096 and "max_tokens" not in body


@pytest.mark.parametrize("deployment", ["gpt-chat-latest", "gpt-chat-latest-2026-06-24"])
async def test_fixed_reasoning_alias(deployment):
    body = await body_of(
        chat(
            deployment,
            max_tokens=1_000,
            max_completion_tokens=2_000,
            reasoning_effort="high",
            temperature=0.7,
        )
    )
    assert body["max_completion_tokens"] == 2_000
    for key in ("reasoning_effort", "max_tokens", "temperature"):
        assert key not in body


async def test_fixed_reasoning_via_prompt_model_name():
    provider = AzureChatCompletionProvider(
        "opaque-deployment", {"config": {"modelName": "gpt-4.1"}}
    )
    passthrough = {"reasoning_effort": "high"}
    body = await body_of(
        provider, _ctx({"modelName": "gpt-chat-latest", "passthrough": passthrough})
    )
    assert body["model"] == "opaque-deployment"
    assert "reasoning_effort" not in body and "temperature" not in body
    assert passthrough == {"reasoning_effort": "high"}


@pytest.mark.parametrize(
    "deployment", ["claude-opus-4-7", "claude-opus-4-8", "claude-fable-5", "claude-mythos-5-1"]
)
async def test_claude_sampling_params_omitted(deployment):
    body = await body_of(chat(deployment, **SAMPLED))
    assert body["max_tokens"] == 512
    for key in ("temperature", "top_p", "reasoning_effort", "max_completion_tokens"):
        assert key not in body


@pytest.mark.parametrize(
    "deployment", ["prod-claude-deployment", "claude-prod-5", "claude-team-blue-12", "gpt-4o"]
)
async def test_sampling_params_kept(deployment):
    body = await body_of(chat(deployment, **SAMPLED))
    assert (body["max_tokens"], body["temperature"], body["top_p"]) == (512, 0.5, 0.9)


async def test_explicit_claude_opus_flag():
    body = await body_of(chat("prod-claude-deployment", isClaudeOpus47OrLater=True, **SAMPLED))
    assert body["max_tokens"] == 512
    assert "temperature" not in body and "top_p" not in body


TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
]


@pytest.mark.parametrize(
    "deployment", ["claude-fable-5-1", "claude-mythos-5-1", "claude-opus-5-5", "claude-sonnet-5-5"]
)
async def test_forced_tool_choice_omitted(deployment):
    body = await body_of(chat(deployment, tools=TOOLS, tool_choice="required"))
    assert len(body["tools"]) == 1
    assert "tool_choice" not in body


@pytest.mark.parametrize(
    ("model_name", "tool_choice", "expected"),
    [
        ("claude-fable-5-1", "required", None),
        ("claude-mythos-5-1", {"type": "function", "function": {"name": "get_weather"}}, None),
        ("claude-fable-5-1", "auto", "auto"),
        ("claude-mythos-5-1", "none", "none"),
        ("claude-fable-5", "required", "required"),
    ],
)
async def test_aliased_claude_deployment(model_name, tool_choice, expected):
    body = await body_of(
        chat("prod-claude", modelName=model_name, tools=TOOLS, tool_choice=tool_choice, **SAMPLED)
    )
    assert body["model"] == "prod-claude"
    assert body.get("tool_choice") == expected
    if model_name != "claude-fable-5":
        assert "temperature" not in body and "top_p" not in body


async def test_grok4_drops_rejected_params():
    body = await body_of(chat("grok-4.6", presence_penalty=0.5, stop=["x"], reasoning_effort="low"))
    for key in ("presence_penalty", "frequency_penalty", "stop"):
        assert key not in body
    assert body["reasoning_effort"] == "low"
    assert "max_tokens" in body
    grok3 = await body_of(chat("grok-3", stop=["x"]))
    assert grok3["stop"] == ["x"] and grok3["presence_penalty"] == 0


# --- response handling ---------------------------------------------------------------------------


@respx.mock
async def test_chat_url_headers_and_body():
    route = mock_chat(chat_response())
    provider = chat(headers={"X-Test-Header": "test-value", "api-key": "override"})
    result = await provider.call_api("test prompt")
    assert result["output"] == "test response"
    request = route.calls[0].request
    assert str(request.url) == f"{chat_url()}?api-version={API_VERSION}"
    assert request.headers["X-Test-Header"] == "test-value"
    assert request.headers["api-key"] == "override"
    assert request.headers["Content-Type"] == "application/json"
    assert sent_json(route)["model"] == "test-deployment"
    assert result["tokenUsage"] == {"total": 10, "prompt": 5, "completion": 5}
    assert result["guardrails"] == {"flagged": False, "flaggedInput": False, "flaggedOutput": False}
    assert result["cached"] is False


@respx.mock
async def test_api_base_url_and_version(azure_env):
    azure_env.setenv("AZURE_API_KEY", "env-key")
    route = respx.post(
        "https://proxy.example.com/azure/openai/deployments/d/chat/completions"
    ).mock(return_value=ok(chat_response()))
    provider = AzureChatCompletionProvider(
        "d",
        {"config": {"apiBaseUrl": "https://proxy.example.com/azure/", "apiVersion": "2025-01-01"}},
    )
    await provider.call_api("hi")
    assert route.calls[0].request.url.params["api-version"] == "2025-01-01"
    assert route.calls[0].request.headers["api-key"] == "env-key"


@respx.mock
async def test_response_is_cached():
    route = mock_chat(chat_response())
    provider = chat()
    await provider.call_api("same prompt")
    second = await provider.call_api("same prompt")
    assert route.call_count == 1
    assert second["cached"] is True
    assert second["tokenUsage"] == {"cached": 10, "total": 10}


@respx.mock
async def test_legacy_data_sources_uses_extensions_url():
    route = respx.post(chat_url(path="extensions/chat/completions")).mock(
        return_value=ok(chat_response())
    )
    await chat().call_api(
        "test prompt", _ctx({"dataSources": [{"type": "test"}], "apiVersion": "2024-custom"})
    )
    assert route.calls[0].request.url.params["api-version"] == "2024-custom"
    assert sent_json(route)["dataSources"] == [{"type": "test"}]


@respx.mock
async def test_data_sources_uses_chat_url_and_picks_assistant_choice():
    payload = {
        "choices": [
            {"message": {"role": "tool", "content": "citations"}},
            {
                "message": {"role": "assistant", "content": "grounded answer"},
                "finish_reason": "stop",
            },
        ],
        "usage": USAGE,
    }
    route = mock_chat(payload)
    result = await chat().call_api(
        "test prompt", _ctx({"data_sources": [{"type": "test"}], "apiVersion": "2024-custom"})
    )
    assert "extensions" not in str(route.calls[0].request.url)
    assert route.calls[0].request.url.params["api-version"] == "2024-custom"
    assert result["output"] == "grounded answer"


@respx.mock
async def test_cost_with_audio_tokens():
    mock_chat(
        {
            "choices": [
                {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 1_000,
                "prompt_tokens_details": {"audio_tokens": 200, "cached_tokens": 0},
                "completion_tokens": 500,
                "completion_tokens_details": {"audio_tokens": 100},
                "total_tokens": 1_500,
            },
        },
        deployment="gpt-audio-1.5-2026-02-23",
    )
    result = await chat("gpt-audio-1.5-2026-02-23").call_api("test prompt")
    assert result["cost"] == pytest.approx(
        (800 * 2.5 + 200 * 32 + 400 * 10 + 100 * 64) / 1e6, abs=1e-12
    )
    assert result["tokenUsage"]["completionDetails"] == {}


@respx.mock
async def test_prompt_cache_usage_and_cost():
    mock_chat(
        {
            "choices": [
                {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 2_000,
                "prompt_tokens_details": {"cached_tokens": 500},
                "completion_tokens": 1_000,
                "total_tokens": 3_000,
            },
        },
        deployment="gpt-5.6",
    )
    result = await chat("gpt-5.6").call_api("test prompt")
    assert result["tokenUsage"] == {
        "total": 3_000,
        "prompt": 2_000,
        "completion": 1_000,
        "cached": 500,
    }
    assert result["cost"] == pytest.approx((1_500 * 5 + 500 * 0.5 + 1_000 * 30) / 1e6, abs=1e-12)
    assert result["finishReason"] == "stop"


@pytest.mark.parametrize("model_name", ["claude-fable-5-1", "claude-mythos-5-1"])
@respx.mock
async def test_model_name_cost_without_changing_deployment(model_name):
    route = mock_chat(
        {
            "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            "usage": {
                "prompt_tokens": 1_000,
                "prompt_tokens_details": {"cached_tokens": 500},
                "completion_tokens": 500,
                "total_tokens": 1_500,
            },
        },
        deployment="prod-claude",
    )
    result = await chat("prod-claude", modelName=model_name).call_api("test prompt")
    assert result["cost"] == pytest.approx(0.030125, abs=1e-8)
    body = sent_json(route)
    assert body["model"] == "prod-claude" and "modelName" not in body


@respx.mock
async def test_unknown_model_has_no_cost():
    mock_chat(chat_response())
    assert "cost" not in await chat().call_api("x")


@respx.mock
async def test_json_schema_output_parsed():
    mock_chat(chat_response('{"test": "value"}', finish_reason="stop"))
    fmt = {
        "type": "json_schema",
        "json_schema": {"name": "t", "strict": True, "schema": REQUIRED_SCHEMA},
    }
    result = await chat(response_format=fmt).call_api("test prompt")
    assert result["output"] == {"test": "value"}


@respx.mock
async def test_prompt_json_object_parsed_and_invalid_kept():
    route = mock_chat(chat_response('{"result": 42, "explanation": "test"}'))
    ctx = _ctx({"response_format": {"type": "json_object"}})
    assert (await chat().call_api("a", ctx))["output"] == {"result": 42, "explanation": "test"}
    route.mock(return_value=ok(chat_response("Invalid JSON response")))
    assert (await chat().call_api("b", ctx))["output"] == "Invalid JSON response"


async def test_fetch_exception(monkeypatch):
    fail_fetch(monkeypatch, RuntimeError("API Error"))
    assert (await chat().call_api("x")) == {"error": "API call error: API Error"}


async def test_rate_limit_quota(monkeypatch):
    fail_fetch(monkeypatch, HttpRateLimitError(status=429, code="insufficient_quota"))
    result = await chat().call_api("x")
    assert "Quota exceeded" in result["error"]
    assert "insufficient_quota" in result["error"]
    assert "Retries will not help" in result["error"]
    assert result["metadata"]["rateLimitKind"] == "quota"
    assert result["metadata"]["http"]["status"] == 429


async def test_rate_limit_per_window(monkeypatch):
    fail_fetch(
        monkeypatch,
        HttpRateLimitError(status=429, code="rate_limit_exceeded", retry_after_ms=7000),
    )
    result = await chat().call_api("x")
    assert "Rate limit exceeded" in result["error"]
    assert "retry after 7s" in result["error"]
    assert "Retries will not help" not in result["error"]
    assert result["metadata"]["rateLimitKind"] == "rate_limit"


async def test_invalid_json_response(monkeypatch):
    from promptphoo.cache import FetchWithCacheResult

    async def text_response(*args: Any, **kwargs: Any) -> Any:
        return FetchWithCacheResult(data="invalid json", cached=False, status=200, status_text="OK")

    monkeypatch.setattr(fetch, "fetch_with_cache", text_response)
    result = await chat().call_api("x")
    assert result["error"].startswith(
        "API returned invalid JSON response (status 200): invalid json"
    )
    assert "Request body: {\n" in result["error"]


@respx.mock
async def test_unparseable_body_from_transport():
    respx.post(chat_url()).mock(return_value=httpx.Response(200, text="invalid json"))
    result = await chat().call_api("x")
    assert result["error"].startswith("API call error: Error parsing response")


@respx.mock
async def test_api_error_response():
    mock_chat({"error": {"code": "DeploymentNotFound", "message": "No such deployment"}}, 404)
    result = await chat().call_api("x")
    assert result == {"error": "API response error: DeploymentNotFound No such deployment"}


async def test_missing_host_raises():
    provider = AzureChatCompletionProvider("d", {"config": {"apiKey": "k"}})
    with pytest.raises(ValueError, match="Azure API host must be set."):
        await provider.call_api("x")


@respx.mock
async def test_tool_calls_returned_raw():
    calls = [{"type": "function", "function": {"name": "test", "arguments": "{}"}}]
    mock_chat(
        {
            "choices": [
                {"message": {"role": "assistant", "tool_calls": calls}, "finish_reason": "stop"}
            ],
            "usage": USAGE,
        }
    )
    assert (await chat().call_api("x"))["output"] == calls


@respx.mock
async def test_logprobs():
    route = mock_chat(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "a b"},
                    "logprobs": {
                        "content": [
                            {"token": "a", "logprob": -0.1},
                            {"token": "b", "logprob": -0.2},
                        ]
                    },
                }
            ],
            "usage": USAGE,
        }
    )
    result = await chat().call_api("x", None, {"includeLogProbs": True})
    assert result["logProbs"] == [-0.1, -0.2]
    assert sent_json(route)["logprobs"] is True


# --- guardrails and content filtering ------------------------------------------------------------


def _filter(filtered: bool, severity: str = "safe") -> dict[str, Any]:
    return {"filtered": filtered, "severity": severity}


@respx.mock
async def test_output_content_filter():
    mock_chat(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Some content"},
                    "finish_reason": "content_filter",
                    "content_filter_results": {
                        "hate": _filter(False),
                        "violence": _filter(True, "medium"),
                    },
                }
            ],
            "usage": USAGE,
        }
    )
    result = await chat().call_api("x")
    assert result["guardrails"] == {"flagged": True, "flaggedInput": False, "flaggedOutput": True}
    assert result["finishReason"] == "content_filter"
    assert result["output"] == "Some content"


@respx.mock
async def test_content_filtered_null_output_is_empty_string():
    mock_chat(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "content_filter",
                }
            ]
        }
    )
    result = await chat().call_api("x")
    assert result["output"] == ""
    assert result["guardrails"]["flaggedOutput"] is True
    assert "error" not in result


@respx.mock
async def test_refusal_with_null_content():
    refusal = "I cannot help with that request."
    mock_chat(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": None, "refusal": refusal},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
        }
    )
    fmt = {"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object"}}}
    result = await chat(response_format=fmt).call_api("x")
    assert result["output"] == refusal
    assert result["isRefusal"] is True
    assert result["guardrails"]["flagged"] is True


_BILLED_PROMPT = {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}


@pytest.mark.parametrize(
    "payload", [{"choices": [], "usage": _BILLED_PROMPT}, {"usage": _BILLED_PROMPT}]
)
@respx.mock
async def test_no_choices_is_not_an_error(payload):
    mock_chat(payload)
    result = await chat().call_api("x")
    assert "error" not in result and "logProbs" not in result and "output" not in result
    assert result["tokenUsage"] == {"total": 10, "prompt": 10, "completion": 0}


INPUT_FILTERED = (
    "The response was filtered due to the prompt triggering Azure OpenAI's content management "
    "policy. Please modify your prompt and retry."
)


@respx.mock
async def test_input_content_filter():
    mock_chat(
        {
            "error": {
                "message": INPUT_FILTERED,
                "type": None,
                "param": "prompt",
                "code": "content_filter",
                "status": 400,
                "innererror": {"code": "ResponsibleAIPolicyViolation"},
            }
        },
        400,
    )
    result = await chat().call_api("x")
    assert result["guardrails"] == {"flagged": True, "flaggedInput": True, "flaggedOutput": False}
    assert result["output"] == INPUT_FILTERED
    assert result["finishReason"] == "content_filter"


@respx.mock
async def test_content_filter_system_error(caplog):
    mock_chat(
        {
            "choices": [
                {
                    "content_filter_results": {
                        "error": {
                            "code": "content_filter_error",
                            "message": "The contents are not filtered",
                        }
                    },
                    "finish_reason": "stop",
                    "message": {"content": "generated text", "refusal": None, "role": "assistant"},
                }
            ],
            "usage": {"completion_tokens": 37, "prompt_tokens": 19, "total_tokens": 56},
        }
    )
    with caplog.at_level(logging.WARNING):
        result = await chat().call_api("x")
    assert result["guardrails"] == {"flagged": False, "flaggedInput": False, "flaggedOutput": False}
    assert result["output"] == "generated text"
    assert (
        "Content filtering system is down or otherwise unable to complete the request in time: "
        "content_filter_error The contents are not filtered"
    ) in caplog.messages


# --- function tool callbacks ---------------------------------------------------------------------


def _tool_call(name: str, arguments: str, call_id: str = "call_123") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _calls_response(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": None, **message}}],
        "usage": USAGE,
    }


async def _add(args: str) -> str:
    import json

    a = json.loads(args)
    return json.dumps(a["a"] + a["b"])


async def _mul(args: str) -> str:
    import json

    a = json.loads(args)
    return json.dumps(a["a"] * a["b"])


@respx.mock
async def test_function_callback():
    mock_chat(_calls_response({"tool_calls": [_tool_call("addNumbers", '{"a": 5, "b": 6}')]}))
    result = await chat(functionToolCallbacks={"addNumbers": _add}).call_api("Add 5 and 6")
    assert "error" not in result
    assert result["output"] == "11"


@respx.mock
async def test_multiple_function_callbacks():
    mock_chat(
        _calls_response(
            {
                "tool_calls": [
                    _tool_call("addNumbers", '{"a": 5, "b": 6}'),
                    _tool_call("multiplyNumbers", '{"a": 3, "b": 4}', "call_456"),
                ]
            }
        )
    )
    provider = chat(functionToolCallbacks={"addNumbers": _add, "multiplyNumbers": _mul})
    assert (await provider.call_api("x"))["output"] == "11\n12"


@respx.mock
async def test_failed_callback_returns_raw_call():
    mock_chat(
        _calls_response({"function_call": {"name": "addNumbers", "arguments": "invalid json"}})
    )
    result = await chat(functionToolCallbacks={"addNumbers": _add}).call_api("x")
    assert "error" not in result
    assert result["output"] == '{"name":"addNumbers","arguments":"invalid json"}'


@respx.mock
async def test_legacy_function_call_callback():
    import json

    async def calc(args: str) -> str:
        return json.dumps(sum(json.loads(args)["numbers"]))

    mock_chat(
        _calls_response(
            {"function_call": {"name": "calculateSum", "arguments": '{"numbers": [1, 2, 3, 4, 5]}'}}
        )
    )
    assert (await chat(functionToolCallbacks={"calculateSum": calc}).call_api("x"))[
        "output"
    ] == "15"


# --- completion.test.ts --------------------------------------------------------------------------


COMPLETION_URL = f"https://{HOST}/openai/deployments/test/completions"


def completion(**config: Any) -> AzureCompletionProvider:
    return AzureCompletionProvider("test", api_options(**config))


def mock_completion(payload: Any, status: int = 200) -> Any:
    return respx.post(COMPLETION_URL).mock(return_value=ok(payload, status))


@respx.mock
async def test_completion_basic_and_cached():
    route = mock_completion(
        {
            "choices": [{"text": "hello"}],
            "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
        }
    )
    provider = completion()
    first = await provider.call_api("test prompt")
    second = await provider.call_api("test prompt")
    assert (first["output"], second["output"]) == ("hello", "hello")
    assert first["tokenUsage"] == {"total": 10, "prompt": 5, "completion": 5}
    assert second["tokenUsage"] == {"cached": 10, "total": 10}
    assert route.call_count == 1
    assert str(route.calls[0].request.url).endswith(f"?api-version={API_VERSION}")
    assert sent_json(route) == {
        "model": "test",
        "prompt": "test prompt",
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "best_of": 1,
    }


@respx.mock
async def test_completion_headers_override_and_passthrough():
    route = mock_completion(
        {
            "choices": [{"text": "o"}],
            "usage": {"total_tokens": 2, "prompt_tokens": 1, "completion_tokens": 1},
        }
    )
    provider = completion(
        headers={"api-key": "override-key", "Extra": "foo"},
        passthrough={"logprobs": 3},
        stop=["\n"],
    )
    await provider.call_api("test prompt")
    request = route.calls[0].request
    assert request.headers["api-key"] == "override-key"
    assert request.headers["Extra"] == "foo"
    assert sent_json(route)["logprobs"] == 3
    assert sent_json(route)["stop"] == ["\n"]


@respx.mock
async def test_completion_cache_usage_and_cost():
    respx.post(f"https://{HOST}/openai/deployments/gpt-5.6/completions").mock(
        return_value=ok(
            {
                "choices": [{"text": "hello", "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 2_000,
                    "prompt_tokens_details": {"cached_tokens": 500},
                    "completion_tokens": 1_000,
                    "total_tokens": 3_000,
                },
            }
        )
    )
    result = await AzureCompletionProvider("gpt-5.6", api_options()).call_api("x")
    assert result["tokenUsage"] == {
        "total": 3_000,
        "prompt": 2_000,
        "completion": 1_000,
        "cached": 500,
    }
    assert result["cost"] == pytest.approx((1_500 * 5 + 500 * 0.5 + 1_000 * 30) / 1e6, abs=1e-12)
    assert result["finishReason"] == "stop"


@respx.mock
async def test_completion_content_filter_output():
    mock_completion(
        {"choices": [{"text": None, "finish_reason": "content_filter"}], "usage": USAGE}
    )
    result = await completion().call_api("x")
    assert result["output"] == (
        "The generated content was filtered due to triggering Azure OpenAI Service's content "
        "filtering system."
    )
    assert result["guardrails"] == {"flagged": True, "flaggedInput": False, "flaggedOutput": True}


@respx.mock
async def test_completion_prompt_filter_results():
    mock_completion(
        {
            "choices": [{"text": "ok", "finish_reason": "stop"}],
            "prompt_filter_results": [
                {"prompt_index": 0, "content_filter_results": {"hate": _filter(True, "high")}}
            ],
            "usage": USAGE,
        }
    )
    result = await completion().call_api("x")
    assert result["guardrails"] == {"flagged": True, "flaggedInput": True, "flaggedOutput": False}


@respx.mock
async def test_completion_input_filter_error():
    mock_completion({"error": {"code": "content_filter", "status": 400, "message": "blocked"}}, 400)
    result = await completion().call_api("x")
    assert result == {
        "output": "blocked",
        "guardrails": {"flagged": True, "flaggedInput": True, "flaggedOutput": False},
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [], "usage": {"total_tokens": 5, "prompt_tokens": 5, "completion_tokens": 0}},
        {"usage": {"total_tokens": 5, "prompt_tokens": 5, "completion_tokens": 0}},
        {"choices": [{"text": "", "finish_reason": "stop"}], "usage": USAGE},
        {"choices": [{"text": None, "finish_reason": "stop"}], "usage": USAGE},
    ],
)
@respx.mock
async def test_completion_graceful_empty_output(payload):
    mock_completion(payload)
    result = await completion().call_api("x")
    assert "error" not in result
    assert result["output"] == ""


@respx.mock
async def test_completion_parse_exception():
    mock_completion({})
    result = await completion().call_api("x")
    assert result["error"].startswith("API response error:")
    assert result["tokenUsage"] == {}


async def test_completion_fetch_error(monkeypatch):
    fail_fetch(monkeypatch, RuntimeError("API Error"))
    assert (await completion().call_api("x"))["error"] == "API call error: Error: API Error"


async def test_completion_invalid_openai_stop(azure_env):
    azure_env.setenv("OPENAI_STOP", "{invalid json}")
    with pytest.raises(ValueError, match="OPENAI_STOP is not a valid JSON string"):
        await completion().call_api("x")


async def test_completion_missing_host():
    with pytest.raises(ValueError, match="Azure API host must be set."):
        await AzureCompletionProvider("test", {"config": {"apiKey": "k"}}).call_api("x")


# --- embedding.test.ts ---------------------------------------------------------------------------


EMBED_URL = f"https://{HOST}/openai/deployments/emb/embeddings"


def embedder(**config: Any) -> AzureEmbeddingProvider:
    return AzureEmbeddingProvider("emb", api_options(**config))


@pytest.mark.parametrize("dimensions", [None, 256])
@respx.mock
async def test_embedding_body_and_result(dimensions):
    route = respx.post(EMBED_URL).mock(
        return_value=ok(
            {"data": [{"embedding": [0.1, 0.2]}], "usage": {"total_tokens": 3, "prompt_tokens": 3}}
        )
    )
    provider = embedder(**({"dimensions": dimensions} if dimensions else {}))
    result = await provider.call_embedding_api("hello")
    expected = {"input": "hello", "model": "emb"}
    if dimensions:
        expected["dimensions"] = dimensions
    assert sent_json(route) == expected
    assert result == {
        "embedding": [0.1, 0.2],
        "cached": False,
        "tokenUsage": {"total": 3, "prompt": 3, "numRequests": 1},
    }
    cached = await provider.call_embedding_api("hello")
    assert cached["cached"] is True
    assert cached["tokenUsage"] == {"cached": 3, "total": 3, "numRequests": 1}


@respx.mock
async def test_embedding_missing():
    respx.post(EMBED_URL).mock(return_value=ok({"data": [], "usage": {"total_tokens": 1}}))
    result = await embedder().call_embedding_api("x")
    assert result["error"] == (
        'API response error: Error: No embedding returned: {"data":[],"usage":{"total_tokens":1}}'
    )
    assert result["tokenUsage"] == {"total": 1, "numRequests": 1}


async def test_embedding_fetch_error(monkeypatch):
    fail_fetch(monkeypatch, RuntimeError("boom"))
    assert await embedder().call_embedding_api("x") == {
        "error": "API call error: Error: boom",
        "tokenUsage": {"total": 0, "prompt": 0, "completion": 0, "numRequests": 1},
    }


async def test_embedding_missing_host():
    with pytest.raises(ValueError, match="Azure API host must be set."):
        await AzureEmbeddingProvider("e", {"config": {"apiKey": "k"}}).call_embedding_api("x")
