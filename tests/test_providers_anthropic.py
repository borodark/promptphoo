"""Anthropic providers. Ports deterministic cases from test/providers/anthropic/*.test.ts.

HTTP is mocked with respx; the SDK's `messages.create(params, options)` call maps to a POST of
`params` to /v1/messages with `options.headers` merged into the request headers.
"""

import asyncio
import copy
import json

import httpx
import pytest
import respx

from promptphoo import cache as cache_mod
from promptphoo.providers.anthropic import create_provider
from promptphoo.providers.anthropic.completion import AnthropicCompletionProvider
from promptphoo.providers.anthropic.generic import hash_anthropic_cache_value
from promptphoo.providers.anthropic.messages import AnthropicMessagesProvider
from promptphoo.providers.anthropic.util import (
    calculate_anthropic_cost,
    get_token_usage,
    output_from_message,
    parse_messages,
)

MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_ENV = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_MAX_TOKENS",
    "ANTHROPIC_TEMPERATURE",
    "ANTHROPIC_STOP",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ANTHROPIC_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.setattr(cache_mod, "_instance", cache_mod.Cache(None, 60_000))
    cache_mod.disable_cache()
    yield
    cache_mod.enable_cache()


def make_provider(model, options=None):
    options = dict(options or {})
    options.setdefault("label", f"test:{model}")
    return AnthropicMessagesProvider(model, options)


def message(text="Test response", **extra):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        **extra,
    }


def mock_messages(*responses):
    route = respx.post(MESSAGES_URL)
    route.side_effect = [
        r if isinstance(r, httpx.Response) else httpx.Response(200, json=r) for r in responses
    ]
    return route


def sent(route, index=0):
    request = route.calls[index].request
    return json.loads(request.content), request.headers


def run(coro):
    return asyncio.run(coro)


TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }
]


@respx.mock
def test_default_request_and_response():
    route = mock_messages(message("Hello"))
    provider = make_provider("claude-3-5-sonnet-20241022", {"config": {"tools": TOOLS}})
    result = run(provider.call_api("What is the forecast in San Francisco?"))
    body, headers = sent(route)
    assert body == {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the forecast in San Francisco?"}],
            }
        ],
        "tools": TOOLS,
        "temperature": 0,
        "stream": False,
    }
    assert headers["x-api-key"] == "test-api-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "anthropic-beta" not in headers
    # Upstream sends this request through the Anthropic SDK, not fetchWithProxy, so it carries no
    # x-promptfoo-version header (found comparing live requests from both runtimes).
    assert "x-promptfoo-version" not in headers
    assert result["output"] == "Hello"
    assert result["tokenUsage"] == {"total": 15, "prompt": 10, "completion": 5}
    assert result["finishReason"] == "stop"
    assert result["cost"] == pytest.approx(10 * 3e-6 + 5 * 15e-6)


@respx.mock
def test_tool_choice_and_extra_body():
    route = mock_messages(message(), message(), message())
    tool_choice = {"name": "get_weather", "type": "tool"}
    run(
        make_provider(
            "claude-3-5-sonnet-20241022", {"config": {"tool_choice": tool_choice}}
        ).call_api("q")
    )
    assert sent(route, 0)[0]["tool_choice"] == tool_choice
    extra = {"top_p": 0.9, "custom_param": "test_value"}
    run(
        make_provider("claude-3-5-sonnet-20241022", {"config": {"extra_body": extra}}).call_api(
            "Test prompt"
        )
    )
    assert sent(route, 1)[0] == {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}],
        "temperature": 0,
        "stream": False,
        "top_p": 0.9,
        "custom_param": "test_value",
    }
    run(make_provider("claude-3-5-sonnet-20241022", {"config": {"extra_body": "x"}}).call_api("p"))
    assert "custom_param" not in sent(route, 2)[0]


@respx.mock
def test_beta_headers():
    route = mock_messages(message(), message())
    run(
        make_provider(
            "claude-3-7-sonnet-20250219", {"config": {"beta": ["output-128k-2025-02-19"]}}
        ).call_api("p")
    )
    assert sent(route, 0)[1]["anthropic-beta"] == "output-128k-2025-02-19"
    betas = ["output-128k-2025-02-19", "another-beta-feature"]
    run(make_provider("claude-3-7-sonnet-20250219", {"config": {"beta": betas}}).call_api("p"))
    assert sent(route, 1)[1]["anthropic-beta"] == "output-128k-2025-02-19,another-beta-feature"


@respx.mock
def test_structured_output_format():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    route = mock_messages(message('{"name":"Alice","age":30}'))
    provider = make_provider(
        "claude-sonnet-4-5-20250929",
        {"config": {"output_format": {"type": "json_schema", "schema": schema}}},
    )
    result = run(provider.call_api("Extract the person data"))
    body, headers = sent(route)
    assert body["output_config"] == {"format": {"type": "json_schema", "schema": schema}}
    assert headers["anthropic-beta"] == "structured-outputs-2025-11-13"
    assert result["output"] == {"name": "Alice", "age": 30}
    assert "fileReferences" not in (result.get("metadata") or {})


@respx.mock
def test_thinking_configuration():
    route = mock_messages(
        message(
            content=[
                {
                    "type": "thinking",
                    "thinking": "Let me analyze this step by step...",
                    "signature": "test-signature",
                },
                {"type": "text", "text": "Final answer"},
            ]
        ),
        message(
            content=[
                {"type": "redacted_thinking", "data": "encrypted-data"},
                {"type": "text", "text": "Final answer"},
            ]
        ),
    )
    provider = make_provider(
        "claude-3-7-sonnet-20250219",
        {"config": {"thinking": {"type": "enabled", "budget_tokens": 2048}}},
    )
    result = run(provider.call_api("Complex question"))
    body, _ = sent(route)
    assert body["max_tokens"] == 3072
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in body
    assert result["output"] == (
        "Thinking: Let me analyze this step by step...\nSignature: test-signature\n\nFinal answer"
    )
    redacted = run(provider.call_api("Another question"))
    assert redacted["output"] == "Redacted Thinking: encrypted-data\n\nFinal answer"


@respx.mock
def test_api_errors():
    mock_messages(
        httpx.Response(
            400,
            json={
                "type": "error",
                "error": {"type": "invalid_params", "message": "Invalid request parameters"},
            },
        )
    )
    result = run(make_provider("claude-3-5-sonnet-20241022").call_api("p"))
    assert result == {
        "error": "API call error: Invalid request parameters, status 400, type invalid_params"
    }


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ValueError, match="Anthropic API key is not set.*apiKeyRequired: false"):
        run(make_provider("claude-3-5-sonnet-20241022").call_api("p"))


@respx.mock
def test_cost_override_and_finish_reasons():
    route = mock_messages(
        message(usage={"input_tokens": 50, "output_tokens": 50}),
        message(stop_reason="max_tokens"),
        message(stop_reason="tool_use"),
        message(stop_reason=None),
    )
    provider = make_provider("claude-3-5-sonnet-20241022", {"config": {"cost": 0.015}})
    result = run(provider.call_api("a"))
    assert result["cost"] == pytest.approx(1.5)
    assert result["tokenUsage"] == {"total": 100, "prompt": 50, "completion": 50}
    plain = make_provider("claude-3-5-sonnet-20241022")
    assert run(plain.call_api("b"))["finishReason"] == "length"
    assert run(plain.call_api("c"))["finishReason"] == "tool_calls"
    assert "finishReason" not in run(plain.call_api("d"))
    assert route.call_count == 4


def expected_cache_key(model, params, label=None):
    namespace = {"providerId": f"anthropic:{model}", "providerLabel": label or f"test:{model}"}
    return (
        f"anthropic:messages:{model}:{hash_anthropic_cache_value({})}:"
        f"{hash_anthropic_cache_value(namespace)}:{hash_anthropic_cache_value(params)}"
    )


@respx.mock
def test_labeled_cache_uses_hashed_key_and_ignores_metadata():
    cache_mod.enable_cache()
    route = mock_messages(message("fresh"))
    model = "claude-3-5-sonnet-20241022"
    provider = make_provider(model, {"config": {"metadata": {"user_id": "u1"}}})
    first = run(provider.call_api("Hi"))
    second = run(provider.call_api("Hi"))
    assert route.call_count == 1
    assert first["output"] == "fresh"
    assert second["cached"] is True
    assert second["output"] == "fresh"
    params = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        "temperature": 0,
        "stream": False,
    }
    key = expected_cache_key(model, params)
    assert run(cache_mod.get_cache().get(key)) is not None


@respx.mock
def test_cached_plain_string_response():
    cache_mod.enable_cache()
    model = "claude-3-5-sonnet-20241022"
    params = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}],
        "temperature": 0,
        "stream": False,
    }
    run(cache_mod.get_cache().set(expected_cache_key(model, params), "Test output"))
    result = run(make_provider(model).call_api("What is 2+2?"))
    assert (result["output"], result["cached"]) == ("Test output", True)
    assert "tokenUsage" in result


@respx.mock
def test_unlabeled_cache_is_ephemeral_and_cleared():
    cache_mod.enable_cache()
    route = mock_messages(message("one"), message("two"))
    provider = AnthropicMessagesProvider("claude-3-5-sonnet-20241022")
    assert run(provider.call_api("Hi"))["output"] == "one"
    assert run(provider.call_api("Hi"))["cached"] is True
    assert cache_mod.get_cache()._data == {}
    run(cache_mod.clear_cache())
    assert run(provider.call_api("Hi"))["output"] == "two"
    assert route.call_count == 2


@respx.mock
def test_custom_headers_bypass_cache():
    cache_mod.enable_cache()
    route = mock_messages(message(), message())
    provider = make_provider(
        "claude-3-5-sonnet-20241022", {"config": {"headers": {"Authorization": "Bearer x"}}}
    )
    run(provider.call_api("Hi"))
    run(provider.call_api("Hi"))
    assert route.call_count == 2
    assert sent(route)[1]["authorization"] == "Bearer x"


@respx.mock
def test_pause_turn_resume():
    paused = message(
        content=[
            {"type": "text", "text": "Searching for sources."},
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": "solar"},
            },
        ],
        stop_reason="pause_turn",
        usage={"input_tokens": 1000, "output_tokens": 100},
    )
    finished = message(
        content=[
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
            {"type": "text", "text": "Solar leads new capacity."},
        ],
        usage={"input_tokens": 1500, "output_tokens": 200},
    )
    route = mock_messages(paused, finished)
    provider = make_provider(
        "claude-sonnet-4-6",
        {"config": {"tools": [{"type": "web_search_20260209", "name": "web_search"}]}},
    )
    result = run(provider.call_api("What leads new power capacity?"))
    assert route.call_count == 2
    request, resume = sent(route, 0)[0], sent(route, 1)[0]
    assert resume == {
        **request,
        "messages": [*request["messages"], {"role": "assistant", "content": paused["content"]}],
    }
    assert "Searching for sources." in result["output"]
    assert "Solar leads new capacity." in result["output"]
    assert result["finishReason"] == "stop"
    usage = result["tokenUsage"]
    assert (usage["prompt"], usage["completion"], usage["total"]) == (2500, 300, 2800)
    assert result["cost"] == pytest.approx(0.012, abs=1e-10)


@respx.mock
def test_custom_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/anthropic")
    route = respx.post("https://gateway.example/anthropic/v1/messages").mock(
        return_value=httpx.Response(200, json=message())
    )
    run(make_provider("claude-3-5-sonnet-20241022").call_api("p"))
    assert route.call_count == 1


@respx.mock
def test_long_non_streaming_request_refused():
    result = run(
        make_provider("claude-3-7-sonnet-20250219", {"config": {"max_tokens": 64000}}).call_api("p")
    )
    assert result["error"].startswith("API call error: Streaming is required for operations")


# --- util ------------------------------------------------------------------------------------


def test_hash_matches_node():
    # Golden values from Node (hashAnthropicCacheValue in src/providers/anthropic/generic.ts).
    value = {"b": [{"z": 1, "a": "é"}], "a": {"B": 2, "a": 1, "é": 3, "Z": 0, "😀": 4, "Ａ": 5}}
    assert hash_anthropic_cache_value(value) == (
        "c3e6662806e7bedd6897a561206de03072d05ae6ef394920d03aadd73ce34069"
    )
    assert hash_anthropic_cache_value("plain") == (
        "9034d54852402387f1dec69f6734a841d1a376cf59af96ee8fb12fafa48ea8a7"
    )


def test_parse_messages():
    assert parse_messages("system: Be nice\nuser: Hi\nassistant: Hello") == {
        "system": [{"type": "text", "text": "Be nice"}],
        "extractedMessages": [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
        ],
    }
    prompt = json.dumps([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    parsed = parse_messages(prompt)
    assert parsed["system"] == [{"type": "text", "text": "S"}]
    assert parsed["extractedMessages"] == [
        {"role": "user", "content": [{"type": "text", "text": "U"}]}
    ]


def test_output_from_message_tool_use():
    msg = {
        "content": [
            {"type": "text", "text": "Calling"},
            {"type": "tool_use", "id": "t", "name": "f", "input": {"a": 1}},
        ]
    }
    assert output_from_message(msg, True) == (
        'Calling\n\n{"type":"tool_use","id":"t","name":"f","input":{"a":1}}'
    )


def test_token_usage_and_cost_helpers():
    data = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 200,
        }
    }
    usage = get_token_usage(data, False)
    assert (usage["prompt"], usage["completion"], usage["total"]) == (1200, 50, 1250)
    assert calculate_anthropic_cost(
        "claude-3-5-sonnet-20241022", {}, 1_000_000, 0
    ) == pytest.approx(3)
    assert calculate_anthropic_cost("unknown-model", {}, 10, 10) is None


# --- completion + factory ----------------------------------------------------------------------


@respx.mock
def test_completion_provider():
    route = respx.post("https://api.anthropic.com/v1/complete").mock(
        return_value=httpx.Response(200, json={"completion": "Test output", "stop_reason": "stop"})
    )
    provider = AnthropicCompletionProvider("claude-2.1")
    result = run(provider.call_api("Test prompt"))
    assert result["output"] == "Test output"
    body = json.loads(route.calls[0].request.content)
    assert body == {
        "model": "claude-2.1",
        "prompt": "\n\nHuman: Test prompt \n\nAssistant:",
        "max_tokens_to_sample": 1024,
        "temperature": 0,
        "stop_sequences": ["<|im_end|>", "<|endoftext|>"],
    }


def test_factory_routes():
    messages = create_provider("anthropic:messages:claude-sonnet-4-5", {})
    assert isinstance(messages, AnthropicMessagesProvider)
    assert messages.model_name == "claude-sonnet-4-5"
    assert isinstance(create_provider("anthropic:claude-opus-4-1", {}), AnthropicMessagesProvider)
    assert isinstance(
        create_provider("anthropic:completion:claude-2.1", {}), AnthropicCompletionProvider
    )
    with pytest.raises(ValueError, match="not supported by promptphoo"):
        create_provider("anthropic:claude-agent-sdk", {})
    with pytest.raises(ValueError):
        create_provider("anthropic:bogus:model", {})


# --- end to end --------------------------------------------------------------------------------


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    from promptphoo import db, paths

    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path))
    paths.set_config_directory_path(None)
    db.close_db()
    db.run_db_migrations()
    yield tmp_path
    db.close_db()


@respx.mock
def test_end_to_end_eval(config_dir):
    from promptphoo.evaluator.run import evaluate
    from promptphoo.models.eval import Eval
    from promptphoo.providers import load_api_provider

    def reply(request):
        body = json.loads(request.content)
        text = body["messages"][0]["content"][0]["text"]
        return httpx.Response(200, json=message(f"Echo: {text}"))

    route = respx.post(MESSAGES_URL).mock(side_effect=reply)
    provider = run(load_api_provider("anthropic:messages:claude-sonnet-4-5"))
    suite = {
        "providers": [provider],
        "prompts": [{"raw": "Hello {{name}}", "label": "greet"}],
        "tests": [
            {"vars": {"name": "World"}, "assert": [{"type": "contains", "value": "World"}]},
            {"vars": {"name": "Bob"}, "assert": [{"type": "equals", "value": "nope"}]},
        ],
    }
    record = Eval({"tests": suite["tests"]}, author=None)
    run(evaluate(copy.deepcopy(suite), record, {"cache": False}))
    assert route.call_count == 2
    results = sorted(record.results, key=lambda r: r.test_idx)
    assert [r.success for r in results] == [True, False]
    assert results[0].response["output"] == "Echo: Hello World"
    assert results[0].response["tokenUsage"]["total"] == 15
    assert results[0].cost == pytest.approx(10 * 3e-6 + 5 * 15e-6)
    summary = record.to_evaluate_summary()
    assert summary["stats"]["successes"] == 1
    assert summary["stats"]["tokenUsage"]["total"] == 30


@respx.mock
@pytest.mark.parametrize(
    "prompt",
    [
        json.dumps([{"role": "system", "content": "Grade this: yes or no?"}]),
        "system: Grade this: yes or no?",
    ],
)
def test_system_only_prompt_is_sent_as_the_user_turn(prompt):
    # D68: the API rejects a system prompt with no messages (closed QA, select-best).
    route = mock_messages(message("Y"))
    run(make_provider("claude-sonnet-4-5").call_api(prompt))
    body, _ = sent(route)
    assert "system" not in body
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Grade this: yes or no?"}]}
    ]


@respx.mock
def test_system_and_user_prompt_keeps_the_system_prompt():
    route = mock_messages(message("ok"))
    prompt = json.dumps(
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
    )
    run(make_provider("claude-sonnet-4-5").call_api(prompt))
    body, _ = sent(route)
    assert body["system"] == [{"type": "text", "text": "Be brief."}]
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
