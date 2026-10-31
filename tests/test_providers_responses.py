"""Shared Responses API code: input normalization, output processing and function-call
handling. Ports test/providers/responses/input.test.ts and processor.test.ts."""

from __future__ import annotations

import json
from typing import Any

import pytest

from promptphoo.providers.function_callbacks import FunctionCallbackHandler
from promptphoo.providers.responses import (
    ResponsesProcessor,
    get_responses_output_text,
    normalize_responses_input,
)

# -- normalize_responses_input ------------------------------------------------------------------


def test_non_array_input_unchanged():
    assert normalize_responses_input("a plain string prompt") == "a plain string prompt"
    assert normalize_responses_input(None) is None


def test_chat_text_parts_become_input_text():
    assert normalize_responses_input(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    ) == [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]


def test_assistant_turns_use_output_text():
    assert normalize_responses_input(
        [{"role": "assistant", "content": [{"type": "text", "text": "prior reply"}]}]
    ) == [{"role": "assistant", "content": [{"type": "output_text", "text": "prior reply"}]}]


def test_nested_image_url_flattens_to_input_image():
    url = "data:image/png;base64,AAAA"
    assert normalize_responses_input(
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]
    ) == [{"role": "user", "content": [{"type": "input_image", "image_url": url}]}]


def test_image_detail_preserved():
    part = {"type": "image_url", "image_url": {"url": "https://x/y.png", "detail": "low"}}
    assert normalize_responses_input([{"role": "user", "content": [part]}]) == [
        {
            "role": "user",
            "content": [{"type": "input_image", "image_url": "https://x/y.png", "detail": "low"}],
        }
    ]


def test_flat_image_url_string_accepted():
    part = {"type": "image_url", "image_url": "https://x/y.png"}
    assert normalize_responses_input([{"role": "user", "content": [part]}]) == [
        {"role": "user", "content": [{"type": "input_image", "image_url": "https://x/y.png"}]}
    ]


def test_responses_parts_untouched():
    items = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "hi"},
                {"type": "input_image", "image_url": "https://x/y.png"},
                {"type": "input_file", "file_id": "file_123"},
            ],
        }
    ]
    assert normalize_responses_input(items) == items


def test_typed_message_items_normalized_other_typed_items_kept():
    items = [
        {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "done"},
        {"type": "message", "role": "user", "content": [{"type": "text", "text": "hello"}]},
    ]
    assert normalize_responses_input(items) == [
        items[0],
        items[1],
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
    ]


def test_string_content_and_unknown_parts_left_alone():
    items = [
        {"role": "user", "content": "plain string content"},
        {"role": "user", "content": [{"type": "something_new", "value": 1}]},
        {"role": "user", "content": [{"type": "text"}]},
        {"role": "user"},
        "not an object",
        None,
    ]
    assert normalize_responses_input(items) == items


def test_get_responses_output_text():
    assert get_responses_output_text({"output_text": "direct"}) == "direct"
    assert (
        get_responses_output_text(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "a"},
                            {"type": "text", "text": "b"},
                        ],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "output_text", "text": "x"}],
                    },
                    {"type": "output_message", "content": [{"type": "output_text", "text": "c"}]},
                ]
            }
        )  # fmt: skip
        == "ab\nc"
    )
    assert get_responses_output_text({"output": [{"type": "reasoning"}]}) is None


# -- ResponsesProcessor -------------------------------------------------------------------------


class FakeHandler(FunctionCallbackHandler):
    def __init__(self, result: Any = None) -> None:
        super().__init__()
        self.result = result
        self.calls: list[tuple[Any, Any]] = []

    async def process_calls(self, calls, callbacks=None, context=None):
        self.calls.append((calls, callbacks))
        return self.result


def _processor(handler: FunctionCallbackHandler | None = None, cost: Any = 0.001):
    return ResponsesProcessor(
        model_name="gpt-4.1",
        provider_type="openai",
        function_callback_handler=handler or FakeHandler(),
        cost_calculator=lambda *_: cost,
    )


def _message(text: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, **extra}],
    }


@pytest.mark.parametrize(
    ("reason", "finish_reason"),
    [
        ("max_output_tokens", "length"),
        ("content_filter", "content_filter"),
        ("max_messages", "max_messages"),
        (None, None),
    ],
)
async def test_incomplete_reason_exposed_with_partial_text(reason, finish_reason):
    data = {
        "status": "incomplete",
        "incomplete_details": {"reason": reason},
        "output": [{**_message("Partial text"), "status": "incomplete"}],
    }
    result = await _processor().process_response_output(data, {}, False)
    assert result["output"] == "Partial text"
    assert "error" not in result and "isRefusal" not in result
    assert result.get("finishReason") == finish_reason
    assert result["metadata"]["responseStatus"] == "incomplete"
    assert result["metadata"].get("incompleteReason") == reason
    assert result["raw"] is data


async def test_token_exhaustion_with_no_visible_text():
    data = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "reasoning", "id": "rs_incomplete", "summary": []}],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 20},
        },
    }
    result = await _processor().process_response_output(data, {}, False)
    assert result["output"] == ""
    assert result["finishReason"] == "length"
    assert result["tokenUsage"] == {
        "total": 30,
        "prompt": 10,
        "completion": 20,
        "numRequests": 1,
        "completionDetails": {"reasoning": 20},
    }


@pytest.mark.parametrize("cached", [False, True])
async def test_incomplete_diagnostics_retained_for_empty_output(cached):
    data = {
        "id": "resp_empty",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    }
    result = await _processor().process_response_output(data, {}, cached)
    assert "Invalid response format: Missing output array" in result["error"]
    assert "output" not in result
    assert result["finishReason"] == "length"
    assert result["metadata"] == {
        "responseId": "resp_empty",
        "responseStatus": "incomplete",
        "incompleteReason": "max_output_tokens",
    }
    assert result["tokenUsage"] == (
        {"cached": 30, "total": 30, "numRequests": 1}
        if cached
        else {"prompt": 10, "completion": 20, "total": 30, "numRequests": 1}
    )
    assert result["cached"] is cached
    assert result["raw"] is data


async def test_parse_error_for_malformed_completed_response():
    data = {
        "status": "completed",
        "incomplete_details": None,
        "output": [],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    }
    result = await _processor().process_response_output(data, {}, False)
    assert result == {
        "error": "Error parsing response: Error: Invalid response format: Missing output array"
        f"\nResponse: {json.dumps(data, separators=(',', ':'))}"
    }


async def test_simple_text_output():
    data = {
        "id": "resp_test123",
        "model": "gpt-5-2025-08-07",
        "output": [_message("Hello, world!")],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    result = await _processor().process_response_output(data, {}, False)
    assert result["output"] == "Hello, world!"
    assert result["cached"] is False
    assert result["cost"] == 0.001
    assert result["metadata"] == {"responseId": "resp_test123", "model": "gpt-5-2025-08-07"}


async def test_output_token_details_in_usage():
    data = {
        "output": [_message("Hello with usage details")],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 21,
            "cache_write_input_tokens": 2,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {
                "reasoning_tokens": 4,
                "accepted_prediction_tokens": 2,
                "rejected_prediction_tokens": 1,
            },
        },
    }
    result = await _processor().process_response_output(data, {}, False)
    assert result["tokenUsage"] == {
        "total": 21,
        "prompt": 11,
        "completion": 7,
        "numRequests": 1,
        "completionDetails": {
            "reasoning": 4,
            "acceptedPrediction": 2,
            "rejectedPrediction": 1,
            "cacheReadInputTokens": 3,
            "cacheCreationInputTokens": 2,
        },
    }


async def test_function_calls_go_through_the_handler():
    handler = FakeHandler("Function executed successfully")
    item = {
        "type": "function_call",
        "name": "test_function",
        "arguments": '{"param": "value"}',
        "status": "completed",
    }
    data = {"output": [item], "usage": {"input_tokens": 10, "output_tokens": 8}}
    result = await _processor(handler).process_response_output(data, {}, False)
    assert result["output"] == "Function executed successfully"
    assert handler.calls == [(item, None)]


async def test_empty_function_arguments_report_the_call():
    handler = FakeHandler("unused")
    item = {"type": "function_call", "name": "empty_function", "arguments": "{}"}
    data = {"output": [{**item, "status": "completed"}], "usage": {}}
    result = await _processor(handler).process_response_output(data, {}, False)
    assert "no_arguments_provided" in result["output"]
    assert "empty_function" in result["output"]
    assert handler.calls == []


async def test_reasoning_summary_and_suppression():
    data = {
        "output": [
            {"type": "reasoning", "summary": [{"text": "Step 1"}, {"text": "Step 2"}]},
            _message("Final answer"),
        ],
        "usage": {"input_tokens": 15, "output_tokens": 20},
    }
    processor = _processor()
    result = await processor.process_response_output(data, {}, False)
    assert result["output"] == "Reasoning: Step 1\nStep 2\nFinal answer"
    suppressed = await processor.process_response_output(
        data, {}, False, suppress_reasoning_output=True
    )
    assert suppressed["output"] == "Final answer"


async def test_web_search_and_code_interpreter_items():
    data = {
        "output": [
            {"type": "web_search_call", "action": {"type": "search", "query": "test query"},
             "status": "completed"},
            {"type": "web_search_call", "action": {"type": "open_page",
             "url": "https://example.com"}, "status": "completed"},
            {"type": "web_search_call", "action": {"type": "find_in_page", "query": "q"}},
            {"type": "web_search_call", "action": {"type": "scroll"}},
            {"type": "web_search_call", "status": "failed", "error": "boom"},
            {"type": "web_search_call"},
            {"type": "code_interpreter_call", "code": 'print("Hello, world!")',
             "status": "completed"},
            {"type": "code_interpreter_call", "status": "failed", "error": "bad"},
        ]
    }  # fmt: skip
    result = await _processor().process_response_output(data, {}, False)
    assert result["output"].split("\n") == [
        'Web Search: "test query"',
        "Opening page: https://example.com",
        'Finding in page: "q"',
        "Web action: scroll",
        "Web Search Call (status: failed) (Error: boom)",
        "Web Search Call (status: unknown)",
        'Code Interpreter: print("Hello, world!")',
        "Code Interpreter: Running code... (Error: bad)",
    ]


async def test_hosted_mcp_items_and_tool_results():
    tool_result = {"type": "tool_result", "id": "t1", "output": "Sunny"}
    data = {
        "output": [
            {"type": "mcp_list_tools", "server_label": "srv", "tools": [{"name": "t"}]},
            {"type": "mcp_call", "name": "t", "output": "Tool result", "status": "completed"},
            {"type": "mcp_call", "name": "t", "error": "Denied"},
            {"type": "mcp_approval_request", "server_label": "srv", "name": "t",
             "arguments": '{"x":1}'},
            tool_result,
        ]
    }  # fmt: skip
    result = await _processor().process_response_output(data, {}, False)
    assert result["output"] == "\n".join(
        [
            'MCP Tools from srv: [\n  {\n    "name": "t"\n  }\n]',
            "MCP Tool Result (t): Tool result",
            "MCP Tool Error (t): Denied",
            'MCP Approval Required for srv.t: {"x":1}',
            json.dumps(tool_result, separators=(",", ":")),
        ]
    )


async def test_refusal():
    data = {
        "id": "resp_refusal456",
        "model": "gpt-4.1",
        "status": "incomplete",
        "incomplete_details": {"reason": "content_filter"},
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "I cannot help with that request."}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 8},
    }
    result = await _processor().process_response_output(data, {}, False)
    assert result["output"] == "I cannot help with that request."
    assert result["isRefusal"] is True
    assert result["finishReason"] == "content_filter"
    assert result["metadata"] == {
        "responseId": "resp_refusal456",
        "model": "gpt-4.1",
        "responseStatus": "incomplete",
        "incompleteReason": "content_filter",
    }


async def test_mixed_output_types():
    data = {
        "output": [
            {"type": "reasoning", "summary": [{"text": "Let me think about this..."}]},
            {"type": "function_call", "name": "search", "arguments": '{"query": "test"}'},
            _message("Here is the result."),
        ]
    }
    result = await _processor(FakeHandler("Function result")).process_response_output(
        data, {}, False
    )
    assert result["output"] == (
        "Reasoning: Let me think about this...\nFunction result\nHere is the result."
    )


async def test_json_schema_output_parsed():
    data = {"output": [_message('{"result": "success"}')]}
    config = {"response_format": {"type": "json_schema"}}
    result = await _processor().process_response_output(data, config, False)
    assert result["output"] == {"result": "success"}


async def test_errors_and_missing_cost():
    result = await _processor().process_response_output({"output": "bad"}, {}, False)
    assert "Invalid response format: Missing output array" in result["error"]
    result = await _processor(cost=None).process_response_output(
        {"output": [_message("Unpriced response")]}, {}, False
    )
    assert result["output"] == "Unpriced response"
    assert "cost" not in result


async def test_error_payload_formatted():
    result = await _processor().process_response_output(
        {"error": {"message": "nope", "type": "server_error"}}, {}, False
    )
    assert result["error"].startswith("API error: nope, Type: server_error\n\n")


async def test_annotations_in_metadata_and_raw():
    data = {
        "id": "resp_multi123",
        "model": "o3-deep-research",
        "output": [
            _message("First citation", annotations=[{"url": "https://s1"}, {"url": "https://s2"}]),
            _message("Second citation", annotations=[{"url": "https://s3"}]),
        ],
    }
    result = await _processor().process_response_output(data, {}, False)
    annotations = [{"url": "https://s1"}, {"url": "https://s2"}, {"url": "https://s3"}]
    assert result["output"] == "First citation\nSecond citation"
    assert result["metadata"] == {
        "responseId": "resp_multi123",
        "model": "o3-deep-research",
        "annotations": annotations,
    }
    assert result["raw"]["annotations"] == annotations


async def test_metadata_ignores_empty_and_mistyped_fields():
    for data in (
        {"id": "", "model": ""},
        {"id": {"nested": "object"}, "model": 12345},
    ):
        result = await _processor().process_response_output(
            {**data, "output": [_message("x", annotations=[])]}, {}, False
        )
        assert result["metadata"] == {}
    result = await _processor().process_response_output(
        {"id": "r", "output": [_message("x", annotations="not-an-array")]}, {}, False
    )
    assert result["metadata"] == {"responseId": "r"}


# -- FunctionCallbackHandler --------------------------------------------------------------------


async def test_handler_runs_callbacks_and_falls_back():
    handler = FunctionCallbackHandler()

    def add(args: str) -> str:
        values = json.loads(args)
        return str(values["a"] + values["b"])

    callbacks = {"add": add}
    call = {"type": "function_call", "name": "add", "arguments": '{"a": 5, "b": 6}'}
    assert await handler.process_calls(call, callbacks) == "11"
    bad = {**call, "arguments": "invalid json"}
    assert json.loads(await handler.process_calls(bad, callbacks))["arguments"] == "invalid json"
    other = {"type": "function_call", "name": "unknown", "arguments": "{}"}
    assert await handler.process_calls([call, other], callbacks) == (
        '11\n{"type":"function_call","name":"unknown","arguments":"{}"}'
    )
    assert await handler.process_calls([other], callbacks) == [other]
    chat_style = {"type": "function", "function": {"name": "add", "arguments": '{"a":1,"b":2}'}}
    assert await handler.process_calls(chat_style, callbacks) == "3"


async def test_handler_rejects_inline_javascript():
    handler = FunctionCallbackHandler()
    call = {"name": "f", "arguments": "{}"}
    result = await handler.process_calls(call, {"f": "(args) => 1"})
    assert result == '{"name":"f","arguments":"{}"}'


async def test_handler_loads_python_file_callbacks(tmp_path, monkeypatch):
    (tmp_path / "callbacks.py").write_text("def double(args):\n    return {'x': 2}\n")
    monkeypatch.chdir(tmp_path)
    handler = FunctionCallbackHandler()
    call = {"name": "double", "arguments": "{}"}
    assert await handler.process_calls(call, {"double": "file://callbacks.py:double"}) == (
        '{"x":2}'
    )
