# ruff: noqa: E501
"""Ports test/assertions/{refusal,openai,toolCallF1,toolCallF1.responses,skill}.test.ts."""

from __future__ import annotations

import json
from typing import Any

import pytest

from promptphoo.assertions.base import AssertionParams, AssertionValueError
from promptphoo.assertions.refusal import handle_is_refusal, is_basic_refusal
from promptphoo.assertions.tools import (
    HANDLERS,
    InvalidToolCallError,
    handle_is_valid_function_call,
    handle_is_valid_openai_tools_call,
    handle_skill_used,
    handle_tool_call_f1,
    validate_openai_function_call,
)
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.base import ApiProvider


def _params(
    assertion: dict[str, Any],
    output: Any,
    *,
    inverse: bool = False,
    provider: Any = None,
    provider_response: dict[str, Any] | None = None,
    test: dict[str, Any] | None = None,
    rendered_value: Any = None,
    output_string: Any = ...,
) -> AssertionParams:
    if output_string is ...:
        output_string = output if isinstance(output, str) else json_stringify(output)
    return AssertionParams(
        assertion=assertion,
        base_type=assertion["type"].removeprefix("not-"),
        inverse=inverse,
        output=output,
        output_string=output_string,
        rendered_value=rendered_value,
        provider=provider,
        provider_response=provider_response if provider_response is not None else {},
        test=test if test is not None else {},
    )


# --- is-refusal --------------------------------------------------------------------------------

REFUSAL = {"type": "is-refusal"}


def _refusal(output: Any, inverse: bool = False, **kw: Any) -> dict[str, Any]:
    return handle_is_refusal(_params(REFUSAL, output, inverse=inverse, **kw))


@pytest.mark.parametrize(
    "output",
    [
        "This content was flagged for possible biological risk.",
        "Content filtered by the model provider.",
        "",
    ],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_refusal_structured_provider_flag(output: str, inverse: bool) -> None:
    r = _refusal(output, inverse, provider_response={"isRefusal": True})
    assert r["pass"] is (not inverse)
    assert r["score"] == (0 if inverse else 1)
    assert r["reason"] == (
        "Expected output to not be a refusal" if inverse else "Provider reported a refusal"
    )


@pytest.mark.parametrize("flag", [None, False])
def test_refusal_text_detection_without_provider_flag(flag: Any) -> None:
    r = _refusal("I cannot assist with that request.", provider_response={"isRefusal": flag})
    assert (r["pass"], r["score"]) == (True, 1)


def test_refusal_transform_disables_provider_flag() -> None:
    r = handle_is_refusal(
        _params(
            {"type": "is-refusal", "transform": "output[0]"},
            "The result is 42.",
            provider_response={"isRefusal": True},
        )
    )
    assert r["pass"] is False


@pytest.mark.parametrize(
    ("output", "inverse", "passed", "score", "reason"),
    [
        ("I apologize, but I cannot assist with that request.", False, True, 1, "Assertion passed"),
        (
            "Here's how to accomplish that task...",
            False,
            False,
            0,
            "Expected output to be a refusal",
        ),
        ("Here's how to accomplish that task...", True, True, 1, "Assertion passed"),
        (
            "I apologize, but I cannot assist with that request.",
            True,
            False,
            0,
            "Expected output to not be a refusal",
        ),
        (None, False, True, 0, "No value provided, assumed to be a refusal"),
        (None, True, False, 0, "Expected output to not be a refusal"),
        ("", False, True, 0, "Empty response treated as refusal"),
        ("   \n\t  ", False, True, 0, "Empty response treated as refusal"),
        ("", True, False, 0, "Expected output to not be a refusal"),
        ("I am not able to help with that request", False, True, 1, "Assertion passed"),
        ("As an AI, I cannot assist with that request", False, True, 1, "Assertion passed"),
        ("That request is not ethical and I cannot assist", False, True, 1, "Assertion passed"),
        (
            "The generated text has been blocked by our content filters",
            False,
            True,
            1,
            "Assertion passed",
        ),
        ("i Am NoT able TO help with that Request", False, True, 1, "Assertion passed"),
        ("I CANNOT ASSIST WITH THAT REQUEST", False, True, 1, "Assertion passed"),
    ],
)
def test_refusal_cases(output: Any, inverse: bool, passed: bool, score: int, reason: str) -> None:
    assert _refusal(output, inverse) == {
        "pass": passed,
        "score": score,
        "reason": reason,
        "assertion": REFUSAL,
    }


@pytest.mark.parametrize(
    "output",
    [
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "delete_files", "arguments": '{"path":"/"}'},
            }
        ],
        {"answer": "Here is the script"},
    ],
)
def test_refusal_structured_output_is_graded(output: Any) -> None:
    assert _refusal(output)["pass"] is False
    assert _refusal(output, True)["pass"] is True


def test_refusal_text_inside_structured_output() -> None:
    assert _refusal({"message": "I cannot assist with that request."})["pass"] is True


def test_refusal_structured_output_without_output_string() -> None:
    r = _refusal({"tool_calls": [{"function": {"name": "run"}}]}, output_string=None)
    assert (r["pass"], r["score"]) == (False, 0)


def test_basic_refusal_word_boundaries_and_apostrophes() -> None:
    assert not is_basic_refusal("As an airbnb host, here are tips")
    assert is_basic_refusal("I’m sorry, no.")
    assert is_basic_refusal("Well, I can’t help with that.")


# --- is-valid-function-call / is-valid-openai-tools-call ---------------------------------------


class FakeOpenAiProvider(ApiProvider):
    """Stands in for OpenAiChatCompletionProvider's validateFunctionToolCall."""

    def default_id(self) -> str:
        return "openai:chat:test"

    def validate_function_tool_call(self, output: Any, vars: Any = None) -> None:
        validate_openai_function_call(output, self.config.get("functions"), vars)


class PlainProvider(ApiProvider):
    def default_id(self) -> str:
        return "mock"


def _temperature_function() -> dict[str, Any]:
    return {
        "name": "getCurrentTemperature",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["Celsius", "Fahrenheit"]},
            },
            "required": ["location", "unit"],
        },
    }


def _add_function() -> dict[str, Any]:
    return {
        "name": "add",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"],
        },
    }


FAHRENHEIT_CALL = {
    "name": "getCurrentTemperature",
    "arguments": '{"location": "San Francisco, CA", "unit": "Fahrenheit"}',
}
FUNCTION_ASSERTION = {"type": "is-valid-openai-function-call"}
TOOLS_ASSERTION = {"type": "is-valid-openai-tools-call"}
MOCK_PROVIDER = FakeOpenAiProvider(
    {
        "config": {
            "tools": [{"type": "function", "function": _temperature_function()}],
            "functions": [_temperature_function()],
        }
    }
)


def _fn(output: Any, provider: Any = MOCK_PROVIDER, **kw: Any) -> dict[str, Any]:
    kw.setdefault("test", {"vars": {}})
    return handle_is_valid_function_call(
        _params(kw.pop("assertion", FUNCTION_ASSERTION), output, provider=provider, **kw)
    )


async def _tools(output: Any, provider: Any = MOCK_PROVIDER, **kw: Any) -> dict[str, Any]:
    kw.setdefault("test", {"vars": {}})
    return await handle_is_valid_openai_tools_call(
        _params(kw.pop("assertion", TOOLS_ASSERTION), output, provider=provider, **kw)
    )


def _add_provider() -> FakeOpenAiProvider:
    return FakeOpenAiProvider({"config": {"functions": [_add_function()]}})


@pytest.mark.parametrize(
    "output",
    [
        {"arguments": '{"x": 10, "y": 20}', "name": "add"},
        {"function_call": {"arguments": '{"x": 10, "y": 20}', "name": "add"}},
    ],
)
def test_function_call_valid(output: Any) -> None:
    r = _fn(output, _add_provider())
    assert (r["pass"], r["reason"]) == (True, "Assertion passed")


def test_function_call_invalid_arguments() -> None:
    r = _fn({"arguments": '{"x": "10", "y": 20}', "name": "add"}, _add_provider())
    assert r["pass"] is False
    assert r["reason"].startswith('Call to "add" does not match schema: ')
    errors = json.loads(r["reason"].split("schema: ", 1)[1])
    assert errors[0]["instancePath"] == "/x"
    assert errors[0]["message"] == "must be number"


def test_function_call_without_validator() -> None:
    r = _fn({"arguments": '{"x": 10}', "name": "add"}, PlainProvider())
    assert r == {
        "pass": False,
        "score": 0,
        "reason": "Provider does not have functionality for checking function call.",
        "assertion": FUNCTION_ASSERTION,
    }
    assert _fn({"name": "add"}, None)["pass"] is False


def test_function_call_delegates_to_structural_provider() -> None:
    calls: list[Any] = []

    class Custom:
        def validate_function_tool_call(self, output: Any, vars: Any) -> None:
            calls.append((output, vars))

    r = _fn({"name": "x", "arguments": "{}"}, Custom(), test={"vars": {"a": 1}})
    assert r["reason"] == "Assertion passed"
    assert calls == [({"name": "x", "arguments": "{}"}, {"a": 1})]


def test_function_call_matches_schema() -> None:
    assert _fn(FAHRENHEIT_CALL)["pass"] is True


def test_function_call_loads_functions_from_file(tmp_path: Any) -> None:
    path = tmp_path / "weather_functions.yaml"
    path.write_text(
        "- name: getCurrentTemperature\n  parameters:\n    type: object\n    properties:\n"
        "      location:\n        type: string\n      unit:\n        type: string\n"
        "        enum: [Celsius, Fahrenheit]\n    required: [location, unit]\n"
    )
    validate_openai_function_call(FAHRENHEIT_CALL, f"file://{path}", {})


def test_function_call_renders_vars_in_definitions() -> None:
    fn = _temperature_function()
    fn["parameters"]["properties"]["unit"]["enum"] = ["{{unit}}"]
    provider = FakeOpenAiProvider({"config": {"functions": [fn]}})
    call = {
        "name": "getCurrentTemperature",
        "arguments": '{"location": "San Francisco, CA", "unit": "custom_unit"}',
    }
    r = _fn(call, provider, test={"vars": {"unit": "custom_unit"}})
    assert r == {
        "pass": True,
        "score": 1,
        "reason": "Assertion passed",
        "assertion": FUNCTION_ASSERTION,
    }


def test_function_call_no_functions_defined() -> None:
    r = _fn(
        {"name": "getCurrentTemperature", "arguments": '{"location": "San Francisco, CA"}'},
        FakeOpenAiProvider({"config": {}}),
    )
    assert r == {
        "pass": False,
        "score": 0,
        "reason": 'Called "getCurrentTemperature", but there is no function with that name',
        "assertion": FUNCTION_ASSERTION,
    }


def test_function_call_output_not_object() -> None:
    r = _fn("not an object")
    assert r["pass"] is False
    assert r["reason"] == 'OpenAI did not return a valid-looking function call: "not an object"'


def test_function_call_missing_required() -> None:
    r = _fn({"name": "getCurrentTemperature", "arguments": '{"location": "San Francisco, CA"}'})
    assert r["pass"] is False
    assert "must have required property 'unit'" in r["reason"]


@pytest.mark.parametrize(
    "output",
    [
        [{"id": "call_123", "type": "function", "function": FAHRENHEIT_CALL}],
        {"tool_calls": [{"id": "call_123", "type": "function", "function": FAHRENHEIT_CALL}]},
    ],
)
async def test_tools_call_valid(output: Any) -> None:
    assert await _tools(output) == {
        "pass": True,
        "score": 1,
        "reason": "Assertion passed",
        "assertion": TOOLS_ASSERTION,
    }


async def test_tools_call_parallel_and_invalid() -> None:
    provider = FakeOpenAiProvider(
        {"config": {"tools": [{"type": "function", "function": _add_function()}]}}
    )
    ok = [
        {"type": "function", "function": {"name": "add", "arguments": '{"x": 1, "y": 2}'}},
        {"type": "function", "function": {"name": "add", "arguments": '{"x": 3, "y": 4}'}},
    ]
    assert (await _tools(ok, provider))["pass"] is True
    bad = [{"type": "function", "function": {"name": "add", "arguments": '{"x": "10", "y": 20}'}}]
    r = await _tools(bad, provider)
    assert r["pass"] is False
    assert 'Call to "add" does not match schema' in r["reason"]


async def test_tools_call_loads_tools_from_file_and_renders_vars(tmp_path: Any) -> None:
    fn = _temperature_function()
    fn["parameters"]["properties"]["unit"]["enum"] = ["{{unit}}"]
    path = tmp_path / "tools.json"
    path.write_text(json.dumps([{"type": "function", "function": fn}]))
    provider = FakeOpenAiProvider({"config": {"tools": f"file://{path}"}})
    call = {"name": "getCurrentTemperature", "arguments": '{"location": "SF", "unit": "K"}'}
    r = await _tools(
        [{"type": "function", "function": call}], provider, test={"vars": {"unit": "K"}}
    )
    assert r["reason"] == "Assertion passed"


async def test_tools_call_loads_tools_from_python(tmp_path: Any) -> None:
    path = tmp_path / "tools.py"
    path.write_text(
        "def get_tools():\n"
        "    return [{'type': 'function', 'function': {'name': 'add', 'parameters': "
        "{'type': 'object', 'properties': {'x': {'type': 'number'}}}}}]\n"
    )
    provider = FakeOpenAiProvider({"config": {"tools": f"file://{path}:get_tools"}})
    out = [{"type": "function", "function": {"name": "add", "arguments": '{"x": 1}'}}]
    assert (await _tools(out, provider))["pass"] is True


async def test_tools_call_no_tools() -> None:
    out = [{"type": "function", "function": FAHRENHEIT_CALL}]
    r = await _tools(out, FakeOpenAiProvider({"config": {}}))
    assert r == {
        "pass": False,
        "score": 0,
        "reason": "No tools configured in provider, but output contains tool calls",
        "assertion": TOOLS_ASSERTION,
    }


@pytest.mark.parametrize(
    "output", ["not an array", [{"type": "function"}], [{"type": "function", "function": None}], []]
)
async def test_tools_call_not_valid_looking(output: Any) -> None:
    r = await _tools(output)
    assert r["pass"] is False
    assert r["reason"] == (
        f"OpenAI did not return a valid-looking tools response: {json_stringify(output)}"
    )


async def test_tools_call_missing_required() -> None:
    call = {"name": "getCurrentTemperature", "arguments": '{"location": "San Francisco, CA"}'}
    r = await _tools([{"type": "function", "function": call}])
    assert r["pass"] is False
    assert "must have required property" in r["reason"]


@pytest.mark.parametrize(
    ("output", "passed", "reason"),
    [
        (
            "MCP Tool Result (ask_question): The answer",
            True,
            "MCP tool call succeeded for ask_question",
        ),
        (
            "MCP Tool Error (ask_question): Repository not found",
            False,
            "MCP tool call failed for ask_question: Repository not found",
        ),
        (
            "MCP Tool Result (read_wiki_structure): ok",
            True,
            "MCP tool call succeeded for read_wiki_structure",
        ),
        (
            "MCP Tool Error (stripe_payment): Authentication failed: Invalid API key provided",
            False,
            "MCP tool call failed for stripe_payment: Authentication failed: Invalid API key provided",
        ),
        (
            "Text\n\nMCP Tool Result (ask_question): x\n\nmore",
            True,
            "MCP tool call succeeded for ask_question",
        ),
        (
            "MCP Tool Result (ask_question): a\nMCP Tool Error (read_wiki_structure): Failed to read structure.",
            False,
            "MCP tool call failed for read_wiki_structure: Failed to read structure.",
        ),
        ("MCP Tool Error: something", False, "MCP tool call failed for unknown: unknown error"),
        ("MCP Tool Result: something", True, "MCP tool call succeeded for unknown"),
        (
            {"content": "MCP Tool Result (ask_question): data"},
            True,
            "MCP tool call succeeded for ask_question",
        ),
    ],
)
async def test_tools_call_mcp(output: Any, passed: bool, reason: str) -> None:
    r = await _tools(output)
    assert (r["pass"], r["score"], r["reason"]) == (passed, int(passed), reason)


TOOL_CALL = {
    "name": "getCurrentTemperature",
    "arguments": '{"location": "Paris", "unit": "Celsius"}',
}
BAD_ARGS_CALL = {"name": "getCurrentTemperature", "arguments": '{"location": "Paris"}'}


@pytest.mark.parametrize(
    ("kind", "output", "passed", "reason"),
    [
        ("tools", "It is sunny.", True, "Assertion passed"),
        (
            "tools",
            [{"type": "function", "function": TOOL_CALL}],
            False,
            "Expected output to not be a valid OpenAI tools call",
        ),
        ("tools", [{"type": "function", "function": BAD_ARGS_CALL}], True, "Assertion passed"),
        ("function", "It is sunny.", True, "Assertion passed"),
        ("function", TOOL_CALL, False, "Expected output to not be a valid function call"),
    ],
)
async def test_negated_verdicts(kind: str, output: Any, passed: bool, reason: str) -> None:
    if kind == "tools":
        r = await _tools(output, assertion={"type": "not-is-valid-openai-tools-call"}, inverse=True)
    else:
        r = _fn(output, assertion={"type": "not-is-valid-function-call"}, inverse=True)
    assert (r["pass"], r["score"], r["reason"]) == (passed, int(passed), reason)


async def test_negated_config_errors_stay_failures() -> None:
    out = [{"type": "function", "function": TOOL_CALL}]
    r = await _tools(out, FakeOpenAiProvider({"config": {}}), inverse=True)
    assert r["pass"] is False
    assert r["reason"] == "No tools configured in provider, but output contains tool calls"

    bad_schema = FakeOpenAiProvider(
        {
            "config": {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "getCurrentTemperature",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "not-a-type"}},
                            },
                        },
                    }
                ]
            }
        }
    )
    r = await _tools(out, bad_schema, inverse=True)
    assert (r["pass"], r["score"]) == (False, 0)
    assert "schema is invalid" in r["reason"]

    missing = FakeOpenAiProvider({"config": {"functions": "file:///nonexistent/functions.json"}})
    r = _fn(TOOL_CALL, missing, inverse=True)
    assert (r["pass"], r["score"]) == (False, 0)
    assert "/nonexistent/functions.json" in r["reason"]

    not_array = FakeOpenAiProvider({"config": {"functions": {"name": "getCurrentTemperature"}}})
    r = _fn(TOOL_CALL, not_array, inverse=True)
    assert (r["pass"], r["score"], r["reason"]) == (
        False,
        0,
        "Expected the loaded functions to be an array",
    )

    r = _fn(TOOL_CALL, PlainProvider(), inverse=True)
    assert r["reason"] == "Provider does not have functionality for checking function call."


@pytest.mark.parametrize(
    "definition", [None, {}, {"name": 42}, {"name": "getCurrentTemperature", "parameters": None}]
)
def test_negated_malformed_function_definitions(definition: Any) -> None:
    provider = FakeOpenAiProvider({"config": {"functions": [definition]}})
    r = _fn(TOOL_CALL, provider, inverse=True)
    assert (r["pass"], r["score"]) == (False, 0)


def test_zod_style_issue_message() -> None:
    provider = FakeOpenAiProvider({"config": {"functions": [None, {"name": " "}]}})
    r = _fn(TOOL_CALL, provider)
    assert json.loads(r["reason"]) == [
        {
            "expected": "object",
            "code": "invalid_type",
            "path": [0],
            "message": "Invalid input: expected object, received null",
        },
        {"code": "custom", "path": [1, "name"], "message": "Expected a nonempty function name"},
    ]


@pytest.mark.parametrize(
    "tools",
    [
        {},
        [None],
        [{"type": "function", "function": None}],
        [
            {"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}},
            {"type": "function"},
        ],
    ],
)
async def test_negated_malformed_tools(tools: Any) -> None:
    out = [{"type": "function", "function": BAD_ARGS_CALL}]
    r = await _tools(out, FakeOpenAiProvider({"config": {"tools": tools}}), inverse=True)
    assert (r["pass"], r["score"]) == (False, 0)


def test_validator_errors() -> None:
    class Unavailable:
        def validate_function_tool_call(self, output: Any, vars: Any) -> None:
            raise RuntimeError("validator unavailable")

    r = _fn(TOOL_CALL, Unavailable(), inverse=True)
    assert r == {
        "pass": False,
        "score": 0,
        "reason": "validator unavailable",
        "assertion": FUNCTION_ASSERTION,
    }

    class Custom:
        def validate_function_tool_call(self, output: Any, vars: Any) -> None:
            raise InvalidToolCallError("Unknown function")

    r = _fn(TOOL_CALL, Custom())
    assert (r["pass"], r["score"], r["reason"]) == (False, 0, "Unknown function")
    r = _fn(TOOL_CALL, Custom(), inverse=True)
    assert (r["pass"], r["score"]) == (True, 1)


# --- tool-call-f1 ------------------------------------------------------------------------------


def _f1(
    output: Any, expected: Any, threshold: float | None = None, inverse: bool = False
) -> dict[str, Any]:
    assertion: dict[str, Any] = {"type": "tool-call-f1", "value": expected}
    if threshold is not None:
        assertion["threshold"] = threshold
    return handle_tool_call_f1(_params(assertion, output, inverse=inverse, rendered_value=expected))


def _calls(*names: str) -> dict[str, Any]:
    return {"tool_calls": [{"function": {"name": n, "arguments": "{}"}} for n in names]}


def _tool_use(name: str, **extra: Any) -> str:
    return json.dumps({"type": "tool_use", "name": name, **extra}, separators=(",", ":"))


def _pretty(value: Any) -> str:
    return json.dumps(value, indent=2)


WEATHER_FC = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "get_weather",
    "arguments": '{"city":"NYC"}',
}


def test_f1_exact_match() -> None:
    r = _f1(_calls("get_weather", "book_flight"), ["get_weather", "book_flight"])
    assert (r["pass"], r["score"]) == (True, 1)
    assert r["reason"] == (
        "Tool Call F1: 1.000 (precision=1.000, recall=1.000). "
        "Expected: [book_flight, get_weather], Called: [book_flight, get_weather]"
    )


def test_f1_no_match() -> None:
    r = _f1(_calls("search_web"), ["get_weather", "book_flight"])
    assert (r["pass"], r["score"]) == (False, 0)
    assert r["reason"] == (
        "Tool Call F1 score 0.000 is below threshold 1. Expected: [book_flight, get_weather], "
        "Called: [search_web]. Precision=0.000, Recall=0.000"
    )


def test_f1_partial() -> None:
    r = _f1(_calls("get_weather", "book_flight", "extra_tool"), ["get_weather", "book_flight"])
    assert r["score"] == pytest.approx(0.8)
    assert "Precision=0.667" in r["reason"] and "Recall=1.000" in r["reason"]
    r = _f1(_calls("get_weather"), ["get_weather", "book_flight"])
    assert r["score"] == pytest.approx(2 / 3)
    assert "Precision=1.000" in r["reason"] and "Recall=0.500" in r["reason"]


@pytest.mark.parametrize("output", [{"tool_calls": []}, None, "This is just a text response"])
def test_f1_nothing_called(output: Any) -> None:
    r = _f1(output, ["get_weather"])
    assert (r["pass"], r["score"]) == (False, 0)
    assert "Called: [(none)]" in r["reason"]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            {
                "tool_calls": [
                    {"type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ]
            },
            ["get_weather"],
        ),
        (json.dumps(WEATHER_FC), ["get_weather"]),
        (WEATHER_FC, ["get_weather"]),
        (
            "Thinking about the request.\n"
            '{"type":"function_call","id":"fc_1","name":"get_weather","arguments":"{\\"city\\":\\"NYC\\"}"}\n'
            '{"type":"function_call","id":"fc_2","name":"book_flight","arguments":"{}"}',
            ["get_weather", "book_flight"],
        ),
        (
            [{"function": {"name": "get_weather"}}, {"function": {"name": "book_flight"}}],
            ["get_weather", "book_flight"],
        ),
        ([{"name": "get_weather"}, {"name": "book_flight"}], ["get_weather", "book_flight"]),
        (
            [{"function": {"name": "get_weather"}}, {"name": "book_flight"}],
            ["get_weather", "book_flight"],
        ),
        ({"type": "tool_use", "id": "t", "name": "get_weather", "input": {}}, ["get_weather"]),
        (
            [
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "name": "get_weather"},
                {"type": "tool_use", "name": "book_flight"},
            ],
            ["get_weather", "book_flight"],
        ),
        ({"functionCall": {"name": "get_weather", "args": {}}}, ["get_weather"]),
        (
            [{"functionCall": {"name": "get_weather"}}, {"functionCall": {"name": "book_flight"}}],
            ["get_weather", "book_flight"],
        ),
        (
            {"toolCall": {"functionCalls": [{"name": "get_weather"}, {"name": "book_flight"}]}},
            ["get_weather", "book_flight"],
        ),
        (json.dumps(_calls("get_weather")), ["get_weather"]),
        (json.dumps([{"type": "tool_use", "name": "get_weather"}]), ["get_weather"]),
        (json.dumps({"functionCall": {"name": "get_weather", "args": {}}}), ["get_weather"]),
        (
            "Let me check the weather for you.\n\n"
            '{"type":"tool_use","id":"toolu_01ABC123","name":"get_weather","input":{"city":"NYC"}}\n\n'
            '{"type":"tool_use","id":"toolu_01DEF456","name":"book_flight","input":{"dest":"LA"}}',
            ["get_weather", "book_flight"],
        ),
        (
            "\n\n".join(
                [
                    "Let me check that. Braces in prose {not JSON} are ignored.",
                    _pretty(
                        {"type": "tool_use", "name": "get_weather", "input": {"query": "a } b"}}
                    ),
                    "And then book the flight.",
                    _pretty([{"type": "tool_use", "name": "book_flight", "input": {"d": "LA"}}]),
                ]
            ),
            ["get_weather", "book_flight"],
        ),
        (
            "I'll help you with that.\n\n" + _tool_use("search_web", input={"query": "test"}),
            ["search_web"],
        ),
        ("{\nnot_json\n" + _tool_use("get_weather") + "\n}", ["get_weather"]),
        ("{\n" * 10_000 + _tool_use("get_weather"), ["get_weather"]),
        (_calls("get_weather", "get_weather"), ["get_weather"]),
        (_calls("get_weather"), ["get_weather", "get_weather"]),
        (_calls("get_weather", "book_flight"), "get_weather, book_flight"),
    ],
)
def test_f1_output_formats_full_match(output: Any, expected: Any) -> None:
    r = _f1(output, expected)
    assert (r["pass"], r["score"]) == (True, 1), r["reason"]


@pytest.mark.parametrize(
    "output",
    [
        json.dumps({"type": "response_metadata", "name": "get_weather"}),
        'Here is a partial call: {"type":"tool_use","name":"get_weather"',
        'The tool payload would be {"type":"tool_use","name":"get_weather"}',
        '{"type":"tool_use","name":"get_weather"} is an example payload.',
        '{\n"input":\n{"type":"tool_use","name":"get_weather"}\n} is an example payload.',
        'Example:\n```json\n{"type":"tool_use","name":"get_weather"}',
        '```json\n    ```\n{"type":"tool_use","name":"get_weather"}\n```',
        {"name": "get_weather", "type": "message"},
        {"name": "get_weather"},
        {"type": "function_call"},
        {"type": "function_call", "name": 42},
        '{"type":"function_call","name":',
    ],
)
def test_f1_not_counted(output: Any) -> None:
    r = _f1(output, ["get_weather"])
    assert (r["pass"], r["score"]) == (False, 0)


@pytest.mark.parametrize("prefix", ["{", "[", '{"broken":', "{]", '{"broken":"unfinished\\'])
def test_f1_recovers_after_invalid_candidate(prefix: str) -> None:
    output = f"{prefix}\n" + _pretty({"type": "tool_use", "name": "get_weather", "input": {}})
    assert _f1(output, ["get_weather"])["score"] == 1


def test_f1_unindented_nested_arguments() -> None:
    body = _pretty(
        {
            "type": "tool_use",
            "name": "get_weather",
            "input": [{"type": "tool_use", "name": "delete_account", "query": 'a \\" } [ b'}],
        }
    )
    output = "Calling the weather tool.\n" + "\n".join(
        line.lstrip(" ") for line in body.split("\n")
    )
    assert _f1(output, ["get_weather"])["score"] == 1


@pytest.mark.parametrize(
    ("opening", "closing"),
    [
        ("```json", "```"),
        ("~~~json", "~~~"),
        ("```json", " ```"),
        ("   ```json", "   ````"),
        ("   ```json", "```"),
        ("````json", "```\n~~~\n````"),
        ("```json", "~~~\n```"),
        ("```json", "```json\n```"),
    ],
)
def test_f1_ignores_fenced_examples(opening: str, closing: str) -> None:
    output = "\r\n".join(
        ["Example:", opening, _tool_use("delete_account"), closing, _tool_use("get_weather")]
    )
    assert _f1(output, ["get_weather"])["score"] == 1


@pytest.mark.parametrize(
    "example", ['```json {"example":true} ```', "```info`invalid", "- ```info`invalid"]
)
def test_f1_invalid_backtick_fence_opener(example: str) -> None:
    output = "\n".join([_tool_use("get_weather"), example, _tool_use("delete_account")])
    r = _f1(output, ["get_weather"])
    assert r["pass"] is False
    assert r["score"] == pytest.approx(2 / 3)
    assert "Called: [delete_account, get_weather]" in r["reason"]


@pytest.mark.parametrize(
    ("opening", "closing"), [('Example: {\n"payload":', "}"), ("Example: [", "]")]
)
def test_f1_ignores_inline_example_wrapper(opening: str, closing: str) -> None:
    output = "\n".join([_tool_use("get_weather"), opening, _tool_use("delete_account"), closing])
    assert _f1(output, ["get_weather"])["score"] == 1


@pytest.mark.parametrize("indent", ["    ", "      ", "\t"])
def test_f1_fence_in_markdown_list(indent: str) -> None:
    output = "\n".join(
        [
            "  - Example:",
            f"{indent}```json",
            f"{indent}{_tool_use('delete_account')}",
            f"{indent}```",
            _tool_use("get_weather"),
        ]
    )
    assert _f1(output, ["get_weather"])["score"] == 1


@pytest.mark.parametrize(
    ("opening", "marker", "indent", "closing"),
    [
        ("```json", "    ```", "", "```"),
        ("  - Example:\n    ```json", "        ```", "    ", "    ```"),
    ],
)
def test_f1_over_indented_markers(opening: str, marker: str, indent: str, closing: str) -> None:
    output = "\n".join(
        [
            opening,
            marker,
            f"{indent}{_tool_use('delete_account')}",
            closing,
            _tool_use("get_weather"),
        ]
    )
    assert _f1(output, ["get_weather"])["score"] == 1


@pytest.mark.parametrize("lst", ["- ", "12. "])
def test_f1_fence_after_list_marker(lst: str) -> None:
    indent = " " * len(lst)
    output = "\n".join(
        [
            f"{lst}```json",
            f"{indent}{_tool_use('delete_account')}",
            f"{indent}```",
            _tool_use("get_weather"),
        ]
    )
    assert _f1(output, ["get_weather"])["score"] == 1


@pytest.mark.parametrize(
    ("opening", "closing"),
    [("Actual call: {", "}"), ("Actual call: [\ninvalid", "]"), ("{\n" + "x" * 2000, "}")],
)
def test_f1_recovers_from_malformed_wrapper(opening: str, closing: str) -> None:
    output = "\n".join([opening, _tool_use("delete_account"), closing, _tool_use("get_weather")])
    r = _f1(output, ["get_weather"])
    assert r["pass"] is False
    assert r["score"] == pytest.approx(2 / 3)
    assert "Called: [delete_account, get_weather]" in r["reason"]


def test_f1_work_limit() -> None:
    output = "{\n" * 1000 + "}\n" * 1000
    r = _f1(output, ["get_weather"])
    assert (r["pass"], r["score"]) == (False, 0)
    assert "work limit" in r["reason"]


@pytest.mark.parametrize("inverse", [False, True])
def test_f1_work_limit_exhausted(inverse: bool) -> None:
    output = _tool_use("get_weather") + "\n" + "{\n" * 1000 + "}\n" * 1000
    for out in (output, json.dumps(output)):
        r = _f1(out, ["get_weather"], threshold=0, inverse=inverse)
        assert (r["pass"], r["score"]) == (False, 0)
        assert "work limit" in r["reason"]


@pytest.mark.parametrize("delimiter", ["{", "[", "}", "]"])
def test_f1_delimiter_limit(delimiter: str) -> None:
    output = _tool_use("get_weather") + "\n" + delimiter * 200_000
    for inverse in (False, True):
        for out in (output, json.dumps(output)):
            r = _f1(out, ["get_weather"], threshold=0, inverse=inverse)
            assert (r["pass"], r["score"]) == (False, 0)
            assert r["reason"] == "Tool Call F1 exceeded its delimiter limit (65536)"


def test_f1_many_blocks() -> None:
    for prefix in ("", "{\n"):
        output = (
            prefix + _tool_use("book_flight") + "\n" + "{}\n" * 50_000 + _tool_use("get_weather")
        )
        assert _f1(output, ["book_flight", "get_weather"])["score"] == 1


def test_f1_threshold_and_inverse() -> None:
    assert _f1(_calls("get_weather", "extra_tool"), ["get_weather"])["pass"] is False
    assert _f1(_calls("get_weather", "extra_tool"), ["get_weather"], threshold=0.5)["pass"] is True
    r = _f1(_calls("get_weather"), ["get_weather"], inverse=True)
    assert (r["pass"], r["score"]) == (False, 1)
    assert r["reason"].startswith("Tool Call F1 score 1.000 is above threshold 1.")
    r = _f1(_calls("wrong_tool"), ["get_weather"], inverse=True)
    assert (r["pass"], r["score"]) == (True, 0)


def test_f1_trims_comma_separated() -> None:
    r = _f1(_calls("get_weather"), "  get_weather  ,  book_flight  ")
    assert r["score"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("expected", [[], "", "  ", " , , ", [" ", ""]])
def test_f1_rejects_empty_expected(expected: Any) -> None:
    with pytest.raises(
        AssertionValueError,
        match='"tool-call-f1" assertion requires at least one expected tool name',
    ):
        _f1({}, expected)


def test_f1_rejects_missing_value() -> None:
    with pytest.raises(AssertionValueError, match='"tool-call-f1" assertion requires a value'):
        _f1(_calls("get_weather"), None)


# --- skill-used --------------------------------------------------------------------------------

SKILL_RESPONSE = {
    "output": "Done",
    "metadata": {
        "skillCalls": [
            {
                "name": "token-skill",
                "path": ".agents/skills/token-skill/SKILL.md",
                "source": "heuristic",
            },
            {"name": "project-standards:standards-check", "source": "tool"},
        ]
    },
}


def _skill(
    value: Any, inverse: bool = False, response: dict[str, Any] | None = None
) -> dict[str, Any]:
    assertion = {"type": "not-skill-used" if inverse else "skill-used", "value": value}
    return handle_skill_used(
        _params(
            assertion,
            "Done",
            inverse=inverse,
            rendered_value=value,
            provider_response=response or SKILL_RESPONSE,
            test={"vars": {}},
        )
    )


def test_skill_exact_name() -> None:
    r = _skill("token-skill")
    assert r["pass"] is True
    assert r["reason"] == (
        "Observed required skill(s): token-skill. Actual skills: token-skill "
        "(heuristic, .agents/skills/token-skill/SKILL.md), project-standards:standards-check (tool)"
    )


def test_skill_ignores_errored_calls() -> None:
    response = {
        "output": "Done",
        "metadata": {"skillCalls": [{"name": "missing-skill", "source": "tool", "is_error": True}]},
    }
    r = _skill("missing-skill", response=response)
    assert r["pass"] is False
    assert r["reason"] == "Missing required skill(s): missing-skill. Actual skills: (none)"


def test_skill_list_and_patterns() -> None:
    assert _skill(["token-skill", "project-standards:standards-check"])["pass"] is True
    r = _skill({"pattern": "project-*:*", "min": 1, "max": 1})
    assert r["pass"] is True
    assert r["reason"] == (
        'Matched skill "project-*:*" 1 time(s) (expected 1-1). '
        "Matches: project-standards:standards-check (tool)"
    )
    r = _skill({"pattern": " project-*:* ", "min": 1})
    assert r["pass"] is True
    assert 'Matched skill "project-*:*" 1 time(s) (expected at least 1)' in r["reason"]


def test_skill_inverse() -> None:
    assert _skill("forbidden-skill", True)["pass"] is True
    assert "Missing required skill(s): missing-skill" in _skill("missing-skill")["reason"]
    r = _skill("token-skill", True)
    assert r["pass"] is False
    assert "Forbidden skill(s) were used: token-skill" in r["reason"]
    for value in ({"pattern": "token-*"}, {"pattern": "token-*", "max": 0}):
        r = _skill(value, True)
        assert r["pass"] is False
        assert r["reason"].startswith('Forbidden skill "token-*" was used 1 time(s). Matches: ')
    r = _skill({"pattern": "missing-*", "max": 0}, True)
    assert r["pass"] is True
    assert 'Forbidden skill "missing-*" was not used' in r["reason"]


@pytest.mark.parametrize(
    ("value", "inverse", "message"),
    [
        (
            {"pattern": "token-*", "max": 1},
            True,
            "not-skill-used object assertions only support name/pattern with no count bounds, or max: 0",
        ),
        (
            {"pattern": "token-*", "min": -1},
            False,
            "skill-used assertion object min must be a finite non-negative integer",
        ),
        (
            {"pattern": "token-*", "min": 1.5},
            False,
            "skill-used assertion object min must be a finite non-negative integer",
        ),
        (
            {"pattern": "token-*", "min": 2, "max": 1},
            False,
            "skill-used assertion object max must be greater than or equal to min",
        ),
        ({"min": 1}, False, "skill-used assertion object must include a name or pattern property"),
        (42, False, "skill-used assertion must have a string, string array, or object value"),
    ],
)
def test_skill_errors(value: Any, inverse: bool, message: str) -> None:
    with pytest.raises(AssertionValueError) as exc:
        _skill(value, inverse)
    assert str(exc.value) == message


def test_skill_empty_skill_calls() -> None:
    r = _skill("token-skill", response={"output": "Done", "metadata": {"skillCalls": []}})
    assert r["pass"] is False
    assert r["reason"] == "Missing required skill(s): token-skill. Actual skills: (none)"


def test_handlers_registered() -> None:
    assert set(HANDLERS) == {
        "is-valid-function-call",
        "is-valid-openai-function-call",
        "is-valid-openai-tools-call",
        "tool-call-f1",
        "skill-used",
    }
