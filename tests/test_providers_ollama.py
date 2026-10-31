"""Ports the deterministic cases of test/providers/ollama.test.ts."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest
import respx

from promptphoo import cache
from promptphoo.fetch import FetchWithCacheResult
from promptphoo.providers.ollama import (
    OllamaChatProvider,
    OllamaCompletionProvider,
    OllamaEmbeddingProvider,
    create_provider,
)

WEATHER_TOOLS = {
    "config": {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
    }
}

CHAT_HI = '{"message":{"role":"assistant","content":"hi"},"done":true}\n'


class FakeFetch:
    """Stands in for fetch_with_cache, recording calls like vi.mocked(fetchWithCache)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.result: Any = None
        self.deleted = 0

    def respond(self, data: Any, status: int = 200, status_text: str = "OK", cached=False):
        async def delete() -> None:
            self.deleted += 1

        self.result = FetchWithCacheResult(
            data=data,
            cached=cached,
            status=status,
            status_text=status_text,
            headers={},
            delete_from_cache=delete,
        )

    async def __call__(self, url, options=None, timeout=None, format="json", bust=False, **kw):
        self.calls.append((url, options, timeout, format, bust))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.calls[0][1]["body"])


@pytest.fixture
def fetch(monkeypatch):
    fake = FakeFetch()
    monkeypatch.setattr("promptphoo.fetch.fetch_with_cache", fake)
    return fake


# --- completion --------------------------------------------------------------------------------


def test_completion_construct_with_options():
    provider = OllamaCompletionProvider(
        "llama3.3", {"id": "custom-id", "config": {"temperature": 0.7}}
    )
    assert provider.model_name == "llama3.3"
    assert provider.config["temperature"] == 0.7
    assert provider.id() == "custom-id"


def test_completion_default_id_and_str():
    provider = OllamaCompletionProvider("llama3.3")
    assert provider.id() == "ollama:completion:llama3.3"
    assert str(provider) == "[Ollama Completion Provider llama3.3]"


async def test_completion_basic_and_chunks(fetch):
    fetch.respond('{"response":"test response","done":true}\n')
    assert await OllamaCompletionProvider("llama3.3").call_api("test prompt") == {
        "output": "test response",
        "cached": False,
    }
    fetch.respond('{"response":"test response","done":false}\n{"response":" more","done":true}\n')
    result = await OllamaCompletionProvider("llama3.3").call_api("test prompt")
    assert result == {"output": "test response more", "cached": False}
    url, options, timeout, fmt, bust = fetch.calls[0]
    assert url == "http://localhost:11434/api/generate"
    assert fmt == "text" and bust is False and timeout == 300_000
    assert json.loads(options["body"]) == {
        "model": "llama3.3",
        "prompt": "test prompt",
        "stream": False,
        "options": {},
    }


async def test_completion_api_errors(fetch):
    fetch.result = RuntimeError("API error")
    result = await OllamaCompletionProvider("llama3.3").call_api("test prompt")
    assert "API call error: Error: API error" in result["error"]


async def test_completion_error_field_evicts_cache(fetch):
    fetch.respond('{"error":"transient failure"}')
    result = await OllamaCompletionProvider("llama3.3").call_api("test prompt")
    assert result == {"error": "Ollama error: transient failure"}
    assert fetch.deleted == 1


@pytest.mark.parametrize(
    "status,status_text,msg",
    [
        (404, "Not Found", "model 'llama3.3' not found"),
        (500, "Internal Server Error", 'option "num_predict" must be of type integer'),
        (401, "Unauthorized", "Unauthorized"),
    ],
)
async def test_completion_http_error_body(fetch, status, status_text, msg):
    fetch.respond(json.dumps({"error": msg}), status, status_text)
    result = await OllamaCompletionProvider("llama3.3").call_api("test prompt")
    assert result == {"error": f"Ollama API error: {status} {status_text}: {msg}"}


@pytest.mark.parametrize(
    "data,status,status_text,expected",
    [
        ("<html><body>502 Bad Gateway</body></html>", 502, "Bad Gateway",
         "Ollama API error: 502 Bad Gateway: <html><body>502 Bad Gateway</body></html>"),
        ("", 300, "Multiple Choices", "Ollama API error: 300 Multiple Choices"),
        ("", 503, "Service Unavailable", "Ollama API error: 503 Service Unavailable"),
    ],
)  # fmt: skip
async def test_completion_non_2xx(fetch, data, status, status_text, expected):
    fetch.respond(data, status, status_text)
    result = await OllamaCompletionProvider("llama3.3").call_api("test prompt")
    assert result == {"error": expected}


async def test_completion_invalid_json(fetch):
    fetch.respond("invalid json")
    result = await OllamaCompletionProvider("llama3.3").call_api("test prompt")
    assert result["error"].startswith("Ollama API response error:")


async def test_completion_thinking_and_finish_reason(fetch):
    fetch.respond(
        '{"response":"","thinking":"Let me ","done":false}\n'
        '{"response":"","thinking":"reason.","done":true,"done_reason":"length","eval_count":32}\n'
    )
    result = await OllamaCompletionProvider("qwen3:0.6b").call_api("test prompt")
    assert result["output"] == "Thinking: Let me reason."
    assert result["finishReason"] == "length"

    fetch.respond('{"response":"Hi!","thinking":"Short.","done":true,"done_reason":"stop"}\n')
    result = await OllamaCompletionProvider("qwen3:0.6b").call_api("test prompt")
    assert result["output"] == "Thinking: Short.\n\nHi!"
    assert result["finishReason"] == "stop"


async def test_completion_show_thinking_false_not_sent(fetch):
    fetch.respond('{"response":"Hi!","thinking":"Short.","done":true,"done_reason":"stop"}\n')
    provider = OllamaCompletionProvider("qwen3:0.6b", {"config": {"showThinking": False}})
    result = await provider.call_api("test prompt")
    assert result["output"] == "Hi!"
    assert "showThinking" not in fetch.body and "showThinking" not in fetch.body["options"]


async def test_completion_cache_hit_usage(fetch):
    fetch.respond(
        '{"response":"hi","done":true,"prompt_eval_count":10,"eval_count":20}\n', cached=True
    )
    result = await OllamaCompletionProvider("llama3.3").call_api("test prompt")
    assert result["cached"] is True
    assert result["tokenUsage"] == {
        "prompt": 10,
        "completion": 20,
        "cached": 30,
        "total": 30,
        "numRequests": 0,
        "incurredTokenUsage": {},
    }


async def test_completion_format_and_zero_cached_count(fetch):
    schema = {"type": "object", "properties": {"capital": {"type": "string"}}}
    fetch.respond(
        '{"response":"hi","done":true,"prompt_eval_count":10,"prompt_eval_cached_count":0,'
        '"eval_count":5}\n'
    )
    result = await OllamaCompletionProvider("qwen3", {"config": {"format": schema}}).call_api("p")
    assert fetch.body["format"] == schema and "format" not in fetch.body["options"]
    assert result["tokenUsage"]["completionDetails"] == {"cacheReadInputTokens": 0}


@pytest.mark.parametrize(
    "key,value",
    [("suffix", "return result"), ("system", "You are terse."), ("template", "{{ .Prompt }}"),
     ("raw", True)],
)  # fmt: skip
async def test_completion_generate_only_params(fetch, key, value):
    fetch.respond('{"response":"ok","done":true}\n')
    await OllamaCompletionProvider("qwen2.5", {"config": {key: value}}).call_api("test prompt")
    assert fetch.body[key] == value and key not in fetch.body["options"]


@pytest.mark.parametrize(
    "fragment,expected",
    [('"thinking":{"a":1}', "hi"), ('"thinking":[1,2],"response":""', "")],
)
async def test_completion_degrades_gracefully(fetch, fragment, expected):
    fetch.respond(f'{{"response":"hi",{fragment},"done":true}}\n')
    result = await OllamaCompletionProvider("llama3.3").call_api("test prompt")
    assert "error" not in result and result["output"] == expected


async def test_completion_non_string_response_is_empty(fetch):
    fetch.respond('{"response":{"a":1},"done":true}\n')
    result = await OllamaCompletionProvider("llama3.3").call_api("test prompt")
    assert result["output"] == ""


@pytest.mark.parametrize(
    "extra,expected", [({"bustCache": True}, True), ({"debug": True}, True), ({}, False)]
)
async def test_completion_forwards_bust_cache(fetch, extra, expected):
    fetch.respond('{"response":"hi","done":true}\n')
    context = {"prompt": {"raw": "test prompt", "label": "test"}, "vars": {}, **extra}
    await OllamaCompletionProvider("llama3.3").call_api("test prompt", context)
    assert fetch.calls[0][4] is expected


@pytest.mark.parametrize(
    "data,usage",
    [
        ('{"response":"test response","done":false,"prompt_eval_count":26}\n'
         '{"response":"","done":true,"prompt_eval_count":26,"eval_count":259}\n',
         {"prompt": 26, "completion": 259, "total": 285, "numRequests": 1}),
        ('{"response":"test response","done":true,"prompt_eval_count":26}\n',
         {"prompt": 26, "completion": 0, "total": 26, "numRequests": 1}),
        ('{"response":"test response","done":true,"eval_count":259}\n',
         {"prompt": 0, "completion": 259, "total": 259, "numRequests": 1}),
    ],
)  # fmt: skip
async def test_completion_token_usage(fetch, data, usage):
    fetch.respond(data)
    result = await OllamaCompletionProvider("llama3.3").call_api("test prompt")
    assert result == {"output": "test response", "tokenUsage": usage, "cached": False}


async def test_completion_omits_finish_reason_without_done_reason(fetch):
    fetch.respond('{"response":"Hi!","done":true}\n')
    result = await OllamaCompletionProvider("llama3.3").call_api("test prompt")
    assert result["output"] == "Hi!" and "finishReason" not in result


# --- chat --------------------------------------------------------------------------------------


def test_chat_default_id_and_str():
    provider = OllamaChatProvider("llama3.3")
    assert provider.id() == "ollama:chat:llama3.3"
    assert str(provider) == "[Ollama Chat Provider llama3.3]"


async def test_chat_basic(fetch):
    fetch.respond(CHAT_HI)
    assert await OllamaChatProvider("llama3.3").call_api("test prompt") == {
        "output": "hi",
        "cached": False,
    }
    assert fetch.calls[0][0] == "http://localhost:11434/api/chat"
    assert fetch.body == {
        "model": "llama3.3",
        "messages": [{"role": "user", "content": "test prompt"}],
        "options": {},
    }


async def test_chat_error_field_and_error_mid_stream(fetch):
    fetch.respond('{"error":"chat error occurred"}')
    assert (await OllamaChatProvider("llama3.3").call_api("p")) == {
        "error": "Ollama error: chat error occurred"
    }
    fetch.respond(
        '{"message":{"role":"assistant","content":"Partial"},"done":false}\n'
        '{"error":"an unexpected error was encountered while running the model"}\n'
    )
    result = await OllamaChatProvider("llama3.3").call_api("p")
    assert result == {
        "error": "Ollama error: an unexpected error was encountered while running the model"
    }


@pytest.mark.parametrize(
    "status,status_text,msg",
    [(404, "Not Found", "model 'llama3.3' not found"),
     (400, "Bad Request", '"all-minilm" does not support chat'),
     (401, "Unauthorized", "Unauthorized")],
)  # fmt: skip
async def test_chat_http_error_body(fetch, status, status_text, msg):
    fetch.respond(json.dumps({"error": msg}), status, status_text)
    result = await OllamaChatProvider("llama3.3").call_api("p")
    assert result == {"error": f"Ollama API error: {status} {status_text}: {msg}"}


async def test_chat_ndjson_stream_not_error(fetch):
    fetch.respond(
        '{"message":{"role":"assistant","content":"Hello"},"done":false}\n'
        '{"message":{"role":"assistant","content":" there"},"done":true,"prompt_eval_count":3,'
        '"eval_count":2}\n'
    )
    result = await OllamaChatProvider("llama3.3").call_api("p")
    assert result["output"] == "Hello there" and "error" not in result


async def test_chat_invalid_json(fetch):
    fetch.respond("invalid json")
    result = await OllamaChatProvider("llama3.3").call_api("p")
    assert result["error"].startswith("Ollama API response error:")


async def test_chat_cache_hit_usage(fetch):
    fetch.respond(
        '{"message":{"role":"assistant","content":"hi"},"done":true,"prompt_eval_count":10,'
        '"eval_count":20}\n',
        cached=True,
    )
    result = await OllamaChatProvider("llama3.3").call_api("p")
    assert result["cached"] is True
    assert result["tokenUsage"] == {
        "prompt": 10,
        "completion": 20,
        "cached": 30,
        "total": 30,
        "numRequests": 0,
        "incurredTokenUsage": {},
    }


async def test_chat_passthrough_options_merge(fetch):
    fetch.respond(CHAT_HI)
    provider = OllamaChatProvider(
        "llama3.3",
        {
            "config": {
                "temperature": 0.5,
                "num_predict": 64,
                "passthrough": {"options": {"min_p": 0.1}},
            }
        },
    )
    await provider.call_api("p")
    assert fetch.body["options"] == {"temperature": 0.5, "num_predict": 64, "min_p": 0.1}


async def test_chat_forwards_valid_and_drops_invalid_options(fetch):
    fetch.respond(CHAT_HI)
    config = {"min_p": 0.05, "draft_num_predict": 4, "keep_alive": "5m", "mirostat": 2,
              "tfs_z": 1, "useNUMA": True, "max_tokens": 99}  # fmt: skip
    await OllamaChatProvider("llama3.3", {"config": config}).call_api("p")
    body = fetch.body
    assert body["options"] == {"min_p": 0.05, "draft_num_predict": 4, "mirostat": 2, "tfs_z": 1}
    assert body["keep_alive"] == "5m" and "max_tokens" not in body


@pytest.mark.parametrize("level", ["low", "medium", "high", "max", True, False])
async def test_chat_think_top_level(fetch, level):
    fetch.respond(CHAT_HI)
    await OllamaChatProvider("qwen3", {"config": {"think": level}}).call_api("p")
    assert fetch.body["think"] == level and "think" not in fetch.body["options"]


async def test_chat_prompt_eval_cached_count(fetch):
    fetch.respond(
        '{"message":{"role":"assistant","content":"hi"},"done":true,"prompt_eval_count":54,'
        '"prompt_eval_cached_count":53,"eval_count":25}\n'
    )
    result = await OllamaChatProvider("qwen3").call_api("p")
    assert result["tokenUsage"] == {
        "prompt": 54,
        "completion": 25,
        "total": 79,
        "completionDetails": {"cacheReadInputTokens": 53},
        "numRequests": 1,
    }
    assert result["cached"] is False


async def test_chat_warns_on_completion_only_keys(fetch, caplog):
    fetch.respond(CHAT_HI)
    with caplog.at_level(logging.WARNING, logger="promptphoo.providers.ollama"):
        provider = OllamaChatProvider(
            "qwen3", {"config": {"suffix": "X", "system": "t", "raw": True}}
        )
        await provider.call_api("p")
    for key in ("suffix", "system", "raw"):
        assert key not in fetch.body
    assert "chat endpoint does not accept: suffix, system, raw" in caplog.text


async def test_chat_internal_keys_not_reported(fetch, caplog):
    fetch.respond(CHAT_HI)
    with caplog.at_level(logging.DEBUG, logger="promptphoo.providers.ollama"):
        config = {"temperature": 0, "basePath": "/x", "showThinking": False}
        await OllamaChatProvider("llama3.3", {"config": config}).call_api("p")
    assert "Ignoring unsupported config keys" not in caplog.text
    assert fetch.body["options"] == {"temperature": 0}


async def test_chat_think_and_passthrough_not_nested(fetch):
    fetch.respond(CHAT_HI)
    config = {"temperature": 0.5, "think": True, "passthrough": {"format": "json"}}
    await OllamaChatProvider("llama3.3", {"config": config}).call_api("p")
    assert fetch.body["options"] == {"temperature": 0.5}
    assert fetch.body["think"] is True and fetch.body["format"] == "json"


def _tool_call(args: Any) -> list[Any]:
    return [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": args}}]}]


@pytest.mark.parametrize(
    "value,expected",
    [
        ([], None),
        ([{"role": "user", "content": "hi"}], None),
        ([{"role": "assistant", "tool_calls": None}], None),
        ([{"role": "assistant", "tool_calls": "nope"}], None),
        ([{"role": "assistant", "tool_calls": [None]}], None),
        ([{"role": "assistant", "tool_calls": [{"function": None}]}], None),
        (_tool_call({"a": 1}), None),
        (_tool_call("[1,2]"), None),
        (_tool_call("null"), None),
        (_tool_call("{oops"), None),
        ([None], None),
        (["hello"], None),
        ({"role": "user", "content": "hi"}, None),
        ("plain", None),
        (5, None),
        (_tool_call('{"city":"Paris"}'), _tool_call({"city": "Paris"})),
    ],
)
async def test_chat_normalizes_outgoing_messages(fetch, value, expected):
    fetch.respond('{"message":{"role":"assistant","content":"ok"},"done":true}\n')
    await OllamaChatProvider("llama3.3").call_api(json.dumps(value))
    assert fetch.body["messages"] == (value if expected is None else expected)


async def test_chat_tools_config_and_debug_bust(fetch):
    fetch.respond(CHAT_HI)
    provider = OllamaChatProvider("llama3.3", {"config": {"tools": [{"name": "test-tool"}]}})
    context = {"prompt": {"raw": "p", "label": "t"}, "vars": {"test": "value"}, "debug": True}
    await provider.call_api("test prompt", context)
    assert fetch.body["tools"] == [{"name": "test-tool"}]
    assert fetch.calls[0][4] is True


async def test_chat_tool_calls(fetch):
    args = '{\\"location\\":\\"Amsterdam\\",\\"unit\\":\\"celsius\\"}'
    call = f'{{"function":{{"name":"get_weather","arguments":"{args}"}}}}'
    amsterdam = '{"location":"Amsterdam","unit":"celsius"}'
    expected = [{"function": {"name": "get_weather", "arguments": amsterdam}}]
    fetch.respond(
        f'{{"message":{{"role":"assistant","content":"","tool_calls":[{call}]}},"done":true}}\n'
    )
    result = await OllamaChatProvider("llama3.3", WEATHER_TOOLS).call_api("weather?")
    assert result["output"] == expected
    assert fetch.body["tools"] == WEATHER_TOOLS["config"]["tools"]

    fetch.respond(
        '{"message":{"role":"assistant","content":"Let me check the weather for you.",'
        f'"tool_calls":[{call}]}},"done":true}}\n'
    )
    result = await OllamaChatProvider("llama3.3", WEATHER_TOOLS).call_api("weather?")
    assert result["output"] == {
        "content": "Let me check the weather for you.",
        "tool_calls": expected,
    }


@pytest.mark.parametrize("with_content", [False, True])
async def test_chat_collects_tool_calls_across_chunks(fetch, with_content):
    chunks = [
        {"message": {"role": "assistant", "content": "Checking " if with_content else "",
                     "tool_calls": [{"function": {"name": "get_weather",
                                                  "arguments": {"city": "Amsterdam"}}}]},
         "done": False},
        {"message": {"role": "assistant", "content": "weather." if with_content else "",
                     "tool_calls": [{"function": {"name": "get_weather",
                                                  "arguments": '{"city":"Paris"}'}}]},
         "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True,
         "prompt_eval_count": 10, "eval_count": 20},
    ]  # fmt: skip
    fetch.respond("\n".join(json.dumps(c) for c in chunks) + "\n")
    result = await OllamaChatProvider("llama3.3").call_api("Compare the weather")
    tool_calls = [
        {"function": {"name": "get_weather", "arguments": '{"city":"Amsterdam"}'}},
        {"function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}},
    ]
    assert result["output"] == (
        {"content": "Checking weather.", "tool_calls": tool_calls} if with_content else tool_calls
    )
    assert result["tokenUsage"] == {"prompt": 10, "completion": 20, "total": 30, "numRequests": 1}


async def test_chat_thinking(fetch):
    fetch.respond(
        '{"message":{"role":"assistant","content":"","thinking":"Let me "},"done":false}\n'
        '{"message":{"role":"assistant","content":"","thinking":"reason."},"done":false}\n'
        '{"message":{"role":"assistant","content":""},"done":true,"done_reason":"length",'
        '"eval_count":32}\n'
    )
    result = await OllamaChatProvider("qwen3:0.6b").call_api("p")
    assert result["output"] == "Thinking: Let me reason." and result["finishReason"] == "length"

    data = (
        '{"message":{"role":"assistant","content":"","thinking":"Short."},"done":false}\n'
        '{"message":{"role":"assistant","content":"Hi!"},"done":true,"done_reason":"stop"}\n'
    )
    fetch.respond(data)
    result = await OllamaChatProvider("qwen3:0.6b").call_api("p")
    assert result["output"] == "Thinking: Short.\n\nHi!" and result["finishReason"] == "stop"
    fetch.respond(data)
    provider = OllamaChatProvider("qwen3:0.6b", {"config": {"showThinking": False}})
    assert (await provider.call_api("p"))["output"] == "Hi!"


async def test_chat_tool_call_output_untouched_by_thinking(fetch):
    fetch.respond(
        '{"message":{"role":"assistant","content":"","thinking":"Deciding."},"done":false}\n'
        '{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":'
        '"get_weather","arguments":{"city":"Paris"}}}]},"done":false}\n'
        '{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}\n'
    )
    result = await OllamaChatProvider("qwen3:0.6b").call_api("p")
    assert result["output"] == [
        {"function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}
    ]


@pytest.mark.parametrize(
    "fragment",
    ['"tool_calls":{"a":1}', '"tool_calls":[null]', '"tool_calls":[{}]',
     '"tool_calls":[{"function":null}]', '"tool_calls":[{"function":{"arguments":{"a":1}}}]',
     '"thinking":{"a":1}'],
)  # fmt: skip
async def test_chat_degrades_gracefully(fetch, fragment):
    fetch.respond(f'{{"message":{{"role":"assistant","content":"hi",{fragment}}},"done":true}}\n')
    result = await OllamaChatProvider("llama3.3").call_api("p")
    assert "error" not in result and result["output"] == "hi"


async def test_chat_malformed_tool_calls(fetch):
    fetch.respond('{"message":{"role":"assistant","content":{"a":1}},"done":true}\n')
    assert (await OllamaChatProvider("llama3.3").call_api("p"))["output"] == ""
    fetch.respond(
        '{"message":{"role":"assistant","content":"","tool_calls":[null,{"function":{"name":"f",'
        '"arguments":{"a":1}}}]},"done":true}\n'
    )
    result = await OllamaChatProvider("llama3.3").call_api("p")
    assert result["output"] == [{"function": {"name": "f", "arguments": '{"a":1}'}}]
    for fn in ('{"name":"f"}', '{"name":"f","arguments":null}'):
        fetch.respond(
            f'{{"message":{{"role":"assistant","content":"","tool_calls":[{{"function":{fn}}}]}},'
            '"done":true}\n'
        )
        result = await OllamaChatProvider("llama3.3").call_api("p")
        assert result["output"] == [{"function": {"name": "f", "arguments": "{}"}}]


# --- endpoint key matrix / think ---------------------------------------------------------------

_VALUES = {
    "think": False,
    "keep_alive": "5m",
    "format": "json",
    "truncate": False,
    "suffix": "S",
    "system": "SYS",
    "template": "T",
    "raw": True,
    "dimensions": 128,
}


@pytest.mark.parametrize(
    "kind,keys",
    [("chat", ["think", "keep_alive", "format", "truncate"]),
     ("completion", ["think", "keep_alive", "format", "truncate", "suffix", "system", "template",
                     "raw"]),
     ("embedding", ["keep_alive", "truncate", "dimensions"])],
)  # fmt: skip
async def test_every_accepted_key_reaches_the_wire(fetch, caplog, kind, keys):
    config = {k: _VALUES[k] for k in keys}
    with caplog.at_level(logging.WARNING, logger="promptphoo.providers.ollama"):
        if kind == "embedding":
            fetch.respond({"embeddings": [[0.1]]})
            await OllamaEmbeddingProvider("all-minilm", {"config": config}).call_embedding_api("t")
        elif kind == "chat":
            fetch.respond(CHAT_HI)
            await OllamaChatProvider("m", {"config": config}).call_api("p")
        else:
            fetch.respond('{"response":"hi","done":true}\n')
            await OllamaCompletionProvider("m", {"config": config}).call_api("p")
    for key in keys:
        assert key in fetch.body and key not in fetch.body["options"]
    assert "does not accept" not in caplog.text


@pytest.mark.parametrize("cls,endpoint", [(OllamaCompletionProvider, "/api/generate"),
                                          (OllamaChatProvider, "/api/chat")])  # fmt: skip
@pytest.mark.parametrize("think", [None, False, True])
async def test_think_forwarded_exactly(fetch, cls, endpoint, think):
    fetch.respond("")
    await cls("llama3.3", {"config": {} if think is None else {"think": think}}).call_api("p")
    assert fetch.calls[0][0].endswith(endpoint)
    assert fetch.body.get("think", "unset") == ("unset" if think is None else think)


# --- embeddings --------------------------------------------------------------------------------


async def test_embedding_warns_for_unaccepted_key(fetch, caplog):
    fetch.respond({"embeddings": [[0.1]]})
    with caplog.at_level(logging.WARNING, logger="promptphoo.providers.ollama"):
        provider = OllamaEmbeddingProvider(
            "all-minilm", {"config": {"format": "json", "suffix": "X"}}
        )
        await provider.call_embedding_api("test text")
    assert "format" not in fetch.body
    assert "embedding endpoint does not accept: format, suffix" in caplog.text


async def test_embedding_basic_and_cached(fetch):
    fetch.respond({"embeddings": [[0.1, 0.2, 0.3]], "prompt_eval_count": 4})
    result = await OllamaEmbeddingProvider("all-minilm").call_embedding_api("test text")
    assert result == {
        "embedding": [0.1, 0.2, 0.3],
        "tokenUsage": {"prompt": 4, "total": 4, "numRequests": 1},
        "cached": False,
    }
    assert fetch.calls[0][0] == "http://localhost:11434/api/embed"
    assert fetch.calls[0][3] == "json"
    assert fetch.body == {"model": "all-minilm", "input": "test text", "truncate": False,
                          "options": {}}  # fmt: skip

    fetch.respond({"embeddings": [[0.1, 0.2]], "prompt_eval_count": 7}, cached=True)
    result = await OllamaEmbeddingProvider("all-minilm").call_embedding_api("test text")
    assert result["cached"] is True
    assert result["tokenUsage"] == {"prompt": 7, "cached": 7, "total": 7, "numRequests": 0,
                                    "incurredTokenUsage": {}}  # fmt: skip


async def test_embedding_threads_config(fetch):
    fetch.respond({"embeddings": [[0.1]]})
    config = {"num_ctx": 2048, "truncate": True, "dimensions": 128, "keep_alive": "5m"}
    await OllamaEmbeddingProvider("all-minilm", {"config": config}).call_embedding_api("t")
    body = fetch.body
    assert body["options"] == {"num_ctx": 2048}
    assert (body["truncate"], body["dimensions"], body["keep_alive"]) == (True, 128, "5m")


async def test_embedding_errors(fetch):
    fetch.respond({"error": "the input length exceeds the context length"}, 400, "Bad Request")
    result = await OllamaEmbeddingProvider("all-minilm").call_embedding_api("a" * 5000)
    assert "the input length exceeds the context length" in result["error"]
    assert "num_ctx" in result["error"] and "truncate" in result["error"]
    assert "embedding" not in result

    fetch.respond({"error": "model 'llama3.3' not found"}, 404, "Not Found")
    result = await OllamaEmbeddingProvider("llama3.3").call_embedding_api("t")
    assert result == {"error": "Ollama API error: 404 Not Found: model 'llama3.3' not found"}

    fetch.respond({"message": "invalid token"}, 401, "Unauthorized")
    result = await OllamaEmbeddingProvider("llama3.3").call_embedding_api("t")
    assert result == {"error": 'Ollama API error: 401 Unauthorized: {"message":"invalid token"}'}

    fetch.result = RuntimeError("API error")
    result = await OllamaEmbeddingProvider("llama3.3").call_embedding_api("t")
    assert result == {"error": "API call error: Error: API error"}

    fetch.respond({})
    result = await OllamaEmbeddingProvider("llama3.3").call_embedding_api("t")
    assert "No embedding found in Ollama embeddings API response" in result["error"]

    fetch.respond("invalid json")
    result = await OllamaEmbeddingProvider("llama3.3").call_embedding_api("t")
    assert result["error"].startswith("API response error:")


# --- factory and environment -------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,cls,model",
    [("ollama:chat:llama3", OllamaChatProvider, "llama3"),
     ("ollama:completion:llama3:8b", OllamaCompletionProvider, "llama3:8b"),
     ("ollama:embedding:nomic", OllamaEmbeddingProvider, "nomic"),
     ("ollama:embeddings:nomic", OllamaEmbeddingProvider, "nomic"),
     ("ollama:llama3.3:70b", OllamaCompletionProvider, "llama3.3:70b")],
)  # fmt: skip
def test_factory(path, cls, model):
    provider = create_provider(path, {})
    assert type(provider) is cls and provider.model_name == model


@respx.mock
async def test_base_url_and_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:1234")
    monkeypatch.setenv("OLLAMA_API_KEY", "secret")
    route = respx.post("http://ollama.internal:1234/api/chat").mock(
        return_value=httpx.Response(200, text=CHAT_HI)
    )
    with cache.with_cache_enabled(False):
        result = await OllamaChatProvider("llama3").call_api("hello")
    assert result == {"output": "hi", "cached": False}
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer secret"
    assert request.headers["content-type"] == "application/json"

    # Provider env wins over the process env.
    route2 = respx.post("http://other:1/api/chat").mock(
        return_value=httpx.Response(200, text=CHAT_HI)
    )
    provider = OllamaChatProvider(
        "llama3", {"env": {"OLLAMA_BASE_URL": "http://other:1", "OLLAMA_API_KEY": "k2"}}
    )
    with cache.with_cache_enabled(False):
        await provider.call_api("hello")
    assert route2.calls[0].request.headers["authorization"] == "Bearer k2"
