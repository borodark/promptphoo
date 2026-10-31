"""Shared provider helpers. Ports test/providers/shared.test.ts plus cache-metadata and
cache-bust cases."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from promptphoo.providers.shared import (
    calculate_cost,
    clamp_cached_tokens,
    get_request_timeout_ms,
    is_openai_tool_array,
    is_openai_tool_choice,
    is_promptfoo_sample_target,
    model_name_from_provider_path,
    openai_tool_choice_to_anthropic,
    openai_tool_choice_to_bedrock,
    openai_tool_choice_to_google,
    openai_tools_to_anthropic,
    openai_tools_to_bedrock,
    openai_tools_to_google,
    parse_chat_prompt,
    resolve_direct_test_variable,
    should_bust_provider_cache,
    to_title_case,
    transform_tool_choice,
    transform_tools,
    with_response_cache_metadata,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PROMPTFOO_REQUIRE_JSON_PROMPTS", "REQUEST_TIMEOUT_MS"):
        monkeypatch.delenv(name, raising=False)


# --- modelNameFromProviderPath --------------------------------------------------------------


def test_model_name_keeps_colons_in_model_id():
    path = "anthropic:messages:anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert model_name_from_provider_path(path, 2) == "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_model_name_two_segment_paths():
    assert model_name_from_provider_path("voyage:voyage-3-large", 1) == "voyage-3-large"
    assert model_name_from_provider_path("voyage:some:model:v2", 1) == "some:model:v2"


def test_model_name_empty_when_missing():
    assert model_name_from_provider_path("anthropic:messages", 2) == ""
    assert model_name_from_provider_path("voyage", 1) == ""
    assert model_name_from_provider_path("anthropic:messages:", 2) == ""


def test_model_name_keeps_empty_trailing_segment():
    assert model_name_from_provider_path("anthropic:messages:model:", 2) == "model:"


# --- getRequestTimeoutMs --------------------------------------------------------------------


def test_request_timeout_read_lazily(monkeypatch: pytest.MonkeyPatch):
    assert get_request_timeout_ms() == 300_000
    monkeypatch.setenv("REQUEST_TIMEOUT_MS", "12345")
    assert get_request_timeout_ms() == 12_345


# --- parseChatPrompt ------------------------------------------------------------------------


def test_parse_yaml_prompt():
    yaml_prompt = """
        - role: user
          content: Hello
        - role: assistant
          content: Hi there!
      """
    assert parse_chat_prompt(yaml_prompt, []) == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]


def test_parse_json_prompt():
    messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
    assert parse_chat_prompt(json.dumps(messages), []) == messages


def test_default_for_plain_text_prompt():
    default = [{"role": "user", "content": "Default"}]
    assert parse_chat_prompt("Just a regular string", default) == default


def test_invalid_yaml_raises():
    with pytest.raises(ValueError, match="Chat Completion prompt is not a valid YAML string"):
        parse_chat_prompt("- role: user\n  content: :\n", [])


def test_invalid_json_raises_when_json_prompts_required(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PROMPTFOO_REQUIRE_JSON_PROMPTS", "true")
    with pytest.raises(ValueError, match="Chat Completion prompt is not a valid JSON string"):
        parse_chat_prompt('"role": "user", "content": "Hello" }', [])


@pytest.mark.parametrize("prompt", ['{ "invalid: "json" }', '[{ "invalid": }]'])
def test_invalid_json_raises_when_prompt_looks_like_json(prompt):
    with pytest.raises(ValueError) as excinfo:
        parse_chat_prompt(prompt, [])
    message = str(excinfo.value)
    assert message.startswith("Chat Completion prompt is not a valid JSON string: SyntaxError: ")
    assert message.endswith(f"\n\n{prompt}")


def test_default_for_plain_text_wrapped_in_brackets():
    default = [{"role": "user", "content": "Default"}]
    prompt = "[INST] <<SYS>>Be terse.<</SYS>>\nTell me about yourself [/INST]"
    assert parse_chat_prompt(prompt, default) == default


# --- toTitleCase ----------------------------------------------------------------------------


def test_to_title_case():
    assert to_title_case("hello world") == "Hello World"
    assert to_title_case("UPPERCASE STRING") == "Uppercase String"
    assert to_title_case("mixed CASE string") == "Mixed Case String"
    assert to_title_case("") == ""
    assert to_title_case("word") == "Word"


# --- calculateCost --------------------------------------------------------------------------

MODELS = [
    {"id": "model1", "cost": {"input": 0.001, "output": 0.002}},
    {"id": "model2", "cost": {"input": 0.003, "output": 0.004}},
]
LONG_CONTEXT_MODELS = [
    {
        "id": "model1",
        "cost": {
            "input": 0.001,
            "output": 0.002,
            "longContext": {"threshold": 1_000, "input": 0.003, "output": 0.004},
        },
    }
]


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, 2),
        ({"cost": 0.005}, 7.5),
        ({"inputCost": 0.003, "outputCost": 0.007}, 6.5),
        ({"cost": 0.005, "inputCost": 0.003, "outputCost": 0.007}, 6.5),
        ({"cost": 0.005, "inputCost": 0.003}, 5.5),
        ({"cost": 0.005, "outputCost": 0.007}, 8.5),
    ],
)
def test_calculate_cost(config, expected):
    assert calculate_cost("model1", config, 1000, 500, MODELS) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("config", "expected"),
    [({}, 5.003), ({"cost": 0.005}, 7.505), ({"inputCost": 0.005, "outputCost": 0.006}, 8.005)],
)
def test_calculate_cost_long_context(config, expected):
    assert calculate_cost("model1", config, 1_001, 500, LONG_CONTEXT_MODELS) == pytest.approx(
        expected
    )


def test_calculate_cost_unknown_model_or_bad_tokens():
    assert calculate_cost("nonexistent", {}, 1000, 500, MODELS) is None
    assert calculate_cost("model1", {}, math.nan, 500, MODELS) is None
    assert calculate_cost("model1", {}, 1000, math.inf, MODELS) is None
    assert calculate_cost("model1", {}, None, 500, MODELS) is None
    assert calculate_cost("model1", {}, 1000, None, MODELS) is None


# --- clampCachedTokens ----------------------------------------------------------------------


def test_clamp_cached_tokens():
    assert clamp_cached_tokens(300, 1000) == 300
    assert clamp_cached_tokens(0, 1000) == 0
    assert clamp_cached_tokens(1000, 1000) == 1000
    assert clamp_cached_tokens(1500, 1000) == 1000
    assert clamp_cached_tokens(-500, 1000) == 0
    assert clamp_cached_tokens(None, 1000) == 0
    assert clamp_cached_tokens(math.nan, 1000) == 0
    assert clamp_cached_tokens(math.inf, 1000) == 0


# --- isPromptfooSampleTarget ----------------------------------------------------------------


def test_is_promptfoo_sample_target():
    def target(config):
        return SimpleNamespace(config=config)

    assert is_promptfoo_sample_target(target({"url": "https://api.promptfoo.app/v1"})) is True
    assert is_promptfoo_sample_target(target({"url": "https://api.promptfoo.dev/v1"})) is True
    assert is_promptfoo_sample_target(target({"url": "https://api.other-domain.com/v1"})) is False
    assert is_promptfoo_sample_target(target(None)) is False
    assert is_promptfoo_sample_target(target({})) is False


# --- Tool choice ----------------------------------------------------------------------------

SPECIFIC_CHOICE = {"type": "function", "function": {"name": "my_tool"}}


def test_is_openai_tool_choice():
    for valid in ("auto", "none", "required", SPECIFIC_CHOICE):
        assert is_openai_tool_choice(valid) is True
    for invalid in (
        None,
        "invalid",
        {},
        {"type": "function"},
        {"type": "function", "function": {}},
        {"mode": "auto"},
        {"mode": "tool", "toolName": "foo"},
    ):
        assert is_openai_tool_choice(invalid) is False


def test_tool_choice_to_anthropic():
    assert openai_tool_choice_to_anthropic("auto") == {"type": "auto"}
    assert openai_tool_choice_to_anthropic("none") == {"type": "none"}
    assert openai_tool_choice_to_anthropic("required") == {"type": "any"}
    assert openai_tool_choice_to_anthropic(SPECIFIC_CHOICE) == {"type": "tool", "name": "my_tool"}


def test_tool_choice_to_bedrock():
    assert openai_tool_choice_to_bedrock("auto") == {"auto": {}}
    assert openai_tool_choice_to_bedrock("none") is None
    assert openai_tool_choice_to_bedrock("required") == {"any": {}}
    assert openai_tool_choice_to_bedrock(SPECIFIC_CHOICE) == {"tool": {"name": "my_tool"}}


def test_tool_choice_to_google():
    assert openai_tool_choice_to_google("auto") == {"functionCallingConfig": {"mode": "AUTO"}}
    assert openai_tool_choice_to_google("none") == {"functionCallingConfig": {"mode": "NONE"}}
    assert openai_tool_choice_to_google("required") == {"functionCallingConfig": {"mode": "ANY"}}
    assert openai_tool_choice_to_google({"type": "function", "function": {"name": "my_func"}}) == {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["my_func"]}
    }


def test_transform_tool_choice():
    assert transform_tool_choice({"type": "auto"}, "anthropic") == {"type": "auto"}
    assert transform_tool_choice({"auto": {}}, "bedrock") == {"auto": {}}
    assert transform_tool_choice("auto", "openai") == "auto"
    assert transform_tool_choice("required", "openai") == "required"
    foo = {"type": "function", "function": {"name": "foo"}}
    assert transform_tool_choice(foo, "openai") == foo
    assert transform_tool_choice("auto", "anthropic") == {"type": "auto"}
    assert transform_tool_choice("none", "anthropic") == {"type": "none"}
    assert transform_tool_choice("required", "anthropic") == {"type": "any"}
    assert transform_tool_choice("auto", "bedrock") == {"auto": {}}
    assert transform_tool_choice("required", "bedrock") == {"any": {}}
    assert transform_tool_choice("auto", "google") == {"functionCallingConfig": {"mode": "AUTO"}}
    assert transform_tool_choice("none", "google") == {"functionCallingConfig": {"mode": "NONE"}}
    assert transform_tool_choice("auto", "unknown") == "auto"  # type: ignore[arg-type]


# --- Tool transforms ------------------------------------------------------------------------

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    },
]


def test_is_openai_tool_array():
    assert is_openai_tool_array(SAMPLE_TOOLS) is True
    assert is_openai_tool_array([{"type": "function", "function": {"name": "simple_tool"}}]) is True
    for invalid in (
        [{"function": {"name": "test"}}],
        [{"type": "other", "function": {"name": "test"}}],
        [{"type": "function", "function": {}}],
        [{"type": "function"}],
        [],
        None,
        "string",
        {"type": "function", "function": {"name": "tool"}},
        [{"name": "test", "input_schema": {"type": "object"}}],
        [{"toolSpec": {"name": "test"}}],
        [{"functionDeclarations": [{"name": "test"}]}],
    ):
        assert is_openai_tool_array(invalid) is False


def test_tools_to_anthropic():
    result = openai_tools_to_anthropic(SAMPLE_TOOLS)
    assert result == [
        {
            "name": "get_weather",
            "description": "Get the current weather",
            "input_schema": SAMPLE_TOOLS[0]["function"]["parameters"],
        },
        {
            "name": "search",
            "description": "Search the web",
            "input_schema": SAMPLE_TOOLS[1]["function"]["parameters"],
        },
    ]
    assert openai_tools_to_anthropic([{"type": "function", "function": {"name": "minimal"}}]) == [
        {"name": "minimal", "input_schema": {"type": "object", "properties": {}}}
    ]
    assert openai_tools_to_anthropic(
        [{"type": "function", "function": {"name": "no_desc", "parameters": {"type": "object"}}}]
    ) == [{"name": "no_desc", "input_schema": {"type": "object"}}]


def test_tools_to_bedrock():
    result = openai_tools_to_bedrock(SAMPLE_TOOLS)
    assert len(result) == 2
    assert result[0] == {
        "toolSpec": {
            "name": "get_weather",
            "description": "Get the current weather",
            "inputSchema": {"json": SAMPLE_TOOLS[0]["function"]["parameters"]},
        }
    }
    assert openai_tools_to_bedrock([{"type": "function", "function": {"name": "minimal"}}]) == [
        {
            "toolSpec": {
                "name": "minimal",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }
        }
    ]


def test_tools_to_google():
    result = openai_tools_to_google(SAMPLE_TOOLS)
    assert len(result) == 1
    declarations = result[0]["functionDeclarations"]
    assert [d["name"] for d in declarations] == ["get_weather", "search"]
    params = declarations[0]["parameters"]
    assert params["type"] == "OBJECT"
    assert params["properties"]["location"]["type"] == "STRING"
    # The input schema is not mutated.
    assert SAMPLE_TOOLS[0]["function"]["parameters"]["type"] == "object"


def test_tools_to_google_removes_additional_properties():
    result = openai_tools_to_google(
        [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "parameters": {
                        "type": "object",
                        "properties": {"foo": {"type": "string"}},
                        "additionalProperties": False,
                    },
                },
            }
        ]
    )
    assert "additionalProperties" not in result[0]["functionDeclarations"][0]["parameters"]


def test_tools_to_google_without_parameters():
    result = openai_tools_to_google([{"type": "function", "function": {"name": "minimal"}}])
    assert result[0]["functionDeclarations"][0] == {"name": "minimal"}


def test_transform_tools():
    anthropic_tools = [{"name": "test", "input_schema": {"type": "object"}}]
    assert transform_tools(anthropic_tools, "anthropic") == anthropic_tools
    bedrock_tools = [{"toolSpec": {"name": "test"}}]
    assert transform_tools(bedrock_tools, "bedrock") == bedrock_tools
    assert transform_tools(SAMPLE_TOOLS, "openai") == SAMPLE_TOOLS
    anthropic = transform_tools(SAMPLE_TOOLS, "anthropic")
    assert anthropic[0]["name"] == "get_weather"
    assert "input_schema" in anthropic[0]
    assert transform_tools(SAMPLE_TOOLS, "bedrock")[0]["toolSpec"]["name"] == "get_weather"
    google = transform_tools(SAMPLE_TOOLS, "google")
    assert google[0]["functionDeclarations"][0]["name"] == "get_weather"
    assert transform_tools(SAMPLE_TOOLS, "unknown") == SAMPLE_TOOLS  # type: ignore[arg-type]


# --- Cache helpers --------------------------------------------------------------------------


def test_should_bust_provider_cache():
    assert should_bust_provider_cache() is False
    assert should_bust_provider_cache({"debug": True}) is True
    assert should_bust_provider_cache({"bustCache": False, "debug": True}) is False
    assert should_bust_provider_cache({"bustCache": True}) is True


def test_with_response_cache_metadata_fresh_response_untouched():
    response = {"output": "x", "tokenUsage": {"total": 10, "prompt": 4, "completion": 6}}
    assert with_response_cache_metadata(response, False) == {**response, "cached": False}


def test_with_response_cache_metadata_cached_response():
    response = {
        "output": "x",
        "cost": 0.5,
        "tokenUsage": {"total": 10, "prompt": 4, "completion": 6, "numRequests": 1},
    }
    result = with_response_cache_metadata(response, True)
    assert result == {
        "output": "x",
        "cost": 0.5,
        "cached": True,
        "tokenUsage": {
            "total": 10,
            "prompt": 4,
            "completion": 6,
            "numRequests": 0,
            "cached": 10,
            "incurredTokenUsage": {},
        },
    }
    # As with Node's object spread, existing keys keep their positions.
    assert list(result) == ["output", "cost", "tokenUsage", "cached"]
    reordered = with_response_cache_metadata({"cached": False, "tokenUsage": {}, "x": 1}, True)
    assert list(reordered) == ["cached", "tokenUsage", "x"]
    # An empty usage object is truthy in JS, so it still gets the cache markers.
    assert reordered["tokenUsage"] == {"numRequests": 0, "incurredTokenUsage": {}}


def test_with_response_cache_metadata_preserves_provider_cached_reads():
    response = {"tokenUsage": {"total": 10, "prompt": 8, "cached": 3}}
    assert with_response_cache_metadata(response, True)["tokenUsage"] == {
        "total": 10,
        "prompt": 8,
        "completionDetails": {"cacheReadInputTokens": 3},
        "cached": 10,
        "numRequests": 0,
        "incurredTokenUsage": {},
    }


def test_with_response_cache_metadata_without_usage():
    assert with_response_cache_metadata({"output": "x"}, True) == {"output": "x", "cached": True}


def test_resolve_direct_test_variable():
    assert resolve_direct_test_variable("{{ name }}", {"name": 42}) == 42
    assert resolve_direct_test_variable("{{name}}", {}) == "{{name}}"
    assert resolve_direct_test_variable("{{ a.b }}", {"a": {"b": 1}}) == "{{ a.b }}"
    assert resolve_direct_test_variable(7, {"x": 1}) == 7
