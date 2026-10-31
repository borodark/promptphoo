"""Ports test/assertions/trajectory.test.ts (summarizeTrajectoryForJudge and the model-graded
trajectory:goal-success cases are not ported)."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from promptphoo.assertions.base import AssertionParams, AssertionValueError
from promptphoo.assertions.trajectory import (
    HANDLERS,
    extract_trajectory_steps,
    handle_trajectory_goal_success,
    handle_trajectory_step_count,
    handle_trajectory_tool_args_match,
    handle_trajectory_tool_sequence,
    handle_trajectory_tool_used,
    matches_pattern,
)

MOCK_TRACE: dict[str, Any] = {
    "traceId": "test-trace-id",
    "evaluationId": "test-evaluation-id",
    "testCaseId": "test-test-case-id",
    "metadata": {"test": "value"},
    "spans": [
        {
            "spanId": "span-1",
            "name": "chat gpt-5",
            "startTime": 1000,
            "endTime": 1800,
            "attributes": {"promptfoo.provider.id": "openai:gpt-5"},
        },
        {
            "spanId": "span-2",
            "name": "tool.call",
            "startTime": 1100,
            "endTime": 1200,
            "attributes": {
                "tool.name": "search_orders",
                "tool.arguments": '{"order_id":"123","include_history":false}',
            },
        },
        {
            "spanId": "span-3",
            "name": "mcp inventory/search_inventory",
            "startTime": 1250,
            "endTime": 1350,
            "attributes": {
                "codex.item.type": "mcp_tool_call",
                "codex.mcp.server": "inventory",
                "codex.mcp.tool": "search_inventory",
                "codex.mcp.input": '{"query":"quantum computing","limit":3}',
            },
        },
        {
            "spanId": "span-4",
            "name": "tool.call",
            "startTime": 1400,
            "endTime": 1500,
            "attributes": {
                "tool.name": "compose_reply",
                "tool.args": {"tone": "friendly", "citations": ["doc_1", "doc_2"]},
            },
        },
        {
            "spanId": "span-5",
            "name": "exec ls",
            "startTime": 1550,
            "endTime": 1600,
            "attributes": {"codex.item.type": "command_execution", "codex.command": "ls -la"},
        },
        {
            "spanId": "span-6",
            "name": "reasoning_plan",
            "startTime": 1650,
            "endTime": 1700,
            "attributes": {"reasoning.step": "plan"},
        },
    ],
}

ACTUAL_TOOLS = "tool:search_orders, tool:search_inventory, tool:compose_reply"


def span(name: str, attributes: dict[str, Any], start: int = 1000, span_id: str = "s") -> dict:
    return {
        "spanId": span_id,
        "name": name,
        "startTime": start,
        "endTime": start + 100,
        "attributes": attributes,
    }


def trace_with(*spans: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict:
    t = copy.deepcopy(MOCK_TRACE)
    t["spans"] = list(spans)
    if metadata is not None:
        t["metadata"] = metadata
    return t


def tool_trace(name: str, arguments: Any) -> dict[str, Any]:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return trace_with(span("tool.call", {"tool.name": name, "tool.arguments": args}))


def make_params(
    base_type: str,
    value: Any,
    *,
    inverse: bool = False,
    trace: Any = MOCK_TRACE,
) -> AssertionParams:
    context: dict[str, Any] = {"vars": {}, "test": {}, "prompt": "test prompt"}
    if trace is not None:
        context["trace"] = trace
    return AssertionParams(
        assertion={"type": f"not-{base_type}" if inverse else base_type, "value": value},
        base_type=base_type,
        inverse=inverse,
        output="test output",
        output_string="test output",
        rendered_value=value,
        provider_response={"output": "test output"},
        test={},
        assertion_value_context=context,
    )


def expect_result(result: dict, passed: bool, reason: str, params: AssertionParams) -> None:
    assert result == {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": reason,
        "assertion": params.assertion,
    }


# --- trajectory utilities ----------------------------------------------------------------------


def test_matches_pattern_glob():
    assert matches_pattern("search_orders", "search*")
    assert matches_pattern("SEARCH_orders", "search_*")
    assert matches_pattern("v1", "v?")
    assert not matches_pattern("v12", "v?")
    assert matches_pattern("a.b", "a.b")
    assert not matches_pattern("axb", "a.b")


def test_configured_shell_tool_names_are_commands():
    t = trace_with(
        span("tool.call", {"tool.name": "bash", "tool.arguments": '{"cmd": "pytest tests/"}'}),
        metadata={},
    )
    assert extract_trajectory_steps(t)[0]["type"] == "tool"
    steps = extract_trajectory_steps({**t, "metadata": {"commandToolNames": ["bash"]}})
    assert steps[0]["type"] == "command"
    assert steps[0]["name"] == "pytest tests/"


def test_configured_command_tool_names_case_insensitive_and_invalid_entries():
    t = trace_with(
        span("tool.call", {"tool.name": "bash", "tool.arguments": '{"cmd": "ls -la"}'}),
    )
    upper = extract_trajectory_steps({**t, "metadata": {"commandToolNames": ["BASH"]}})
    assert upper[0]["type"] == "command"
    assert upper[0]["name"] == "ls -la"
    mixed = extract_trajectory_steps(
        {**t, "metadata": {"commandToolNames": ["bash", 123, "", "  "]}}
    )
    assert mixed[0]["type"] == "command"


def test_extracts_normalized_steps():
    steps = extract_trajectory_steps(MOCK_TRACE)
    assert [{"type": s["type"], "name": s["name"]} for s in steps] == [
        {"type": "span", "name": "chat gpt-5"},
        {"type": "tool", "name": "search_orders"},
        {"type": "tool", "name": "search_inventory"},
        {"type": "tool", "name": "compose_reply"},
        {"type": "command", "name": "ls -la"},
        {"type": "reasoning", "name": "reasoning_plan"},
    ]
    assert "ls" in steps[4]["aliases"]
    assert "mcp inventory/search_inventory" in steps[2]["aliases"]
    assert steps[1]["args"] == {"order_id": "123", "include_history": False}
    assert steps[2]["args"] == {"query": "quantum computing", "limit": 3}
    assert steps[3]["args"] == {"tone": "friendly", "citations": ["doc_1", "doc_2"]}


def test_chat_span_with_tool_definitions_is_not_a_tool():
    definitions = json.dumps(
        [
            {"type": "function", "name": "orders", "description": "List orders"},
            {"type": "function", "name": "order_rows", "description": "List order rows"},
        ]
    )
    steps = extract_trajectory_steps(
        trace_with(
            span("chat gpt-4o-mini", {"gen_ai.tool.definitions": definitions}, 1000),
            span("execute_tool", {"gen_ai.tool.name": "order_rows"}, 1200),
        )
    )
    assert [(s["type"], s["name"]) for s in steps] == [
        ("span", "chat gpt-4o-mini"),
        ("tool", "order_rows"),
    ]


def test_plain_string_generic_tool_attribute():
    steps = extract_trajectory_steps(trace_with(span("invoke", {"app.tool": "lookup_customer"})))
    assert [(s["type"], s["name"]) for s in steps] == [("tool", "lookup_customer")]


def test_generic_query_is_search_only_for_search_like_spans():
    steps = extract_trajectory_steps(
        trace_with(
            span("sql.query", {"query": "SELECT * FROM users"}, 1000),
            span("document_search", {"query": "refund policy"}, 1200),
        )
    )
    assert [(s["type"], s["name"]) for s in steps] == [
        ("span", "sql.query"),
        ("search", "refund policy"),
    ]


def test_exec_command_tool_spans_are_commands():
    steps = extract_trajectory_steps(
        trace_with(
            span(
                "tool exec_command",
                {
                    "tool.name": "exec_command",
                    "tool.arguments": json.dumps(
                        {"cmd": "cat repo/tickets/TICKET-014.md", "workdir": "/tmp/ws"}
                    ),
                },
            )
        )
    )
    assert len(steps) == 1
    assert steps[0]["type"] == "command"
    assert steps[0]["name"] == "cat repo/tickets/TICKET-014.md"
    assert steps[0]["args"] == {"cmd": "cat repo/tickets/TICKET-014.md", "workdir": "/tmp/ws"}
    assert {"tool exec_command", "exec_command", "cat"} <= set(steps[0]["aliases"])


def test_local_shell_uses_command_arg():
    steps = extract_trajectory_steps(
        trace_with(
            span(
                "tool local_shell",
                {
                    "tool.name": "local_shell",
                    "tool.arguments": '{"command":"ls -la /tmp","timeout":30}',
                },
            )
        )
    )
    assert steps[0]["type"] == "command"
    assert steps[0]["name"] == "ls -la /tmp"
    assert {"tool local_shell", "local_shell", "ls"} <= set(steps[0]["aliases"])


def test_shell_tool_spans_are_commands():
    steps = extract_trajectory_steps(
        trace_with(
            span("tool shell", {"tool.name": "shell", "tool.arguments": '{"cmd":"echo hello"}'})
        )
    )
    assert steps[0]["type"] == "command"
    assert steps[0]["name"] == "echo hello"
    assert {"shell", "echo"} <= set(steps[0]["aliases"])


def test_shell_command_arrays_join_into_one_step():
    commands = [
        "cat skills/discount-review/SKILL.md",
        "python3 skills/discount-review/scripts/analyze_discount_policy.py skill_fixture/repo",
    ]
    steps = extract_trajectory_steps(
        trace_with(
            span(
                "tool shell",
                {"tool.name": "shell", "tool.arguments": json.dumps({"commands": commands})},
            )
        )
    )
    assert len(steps) == 1
    assert steps[0]["type"] == "command"
    assert steps[0]["name"] == "; ".join(commands)
    assert {"shell", "cat"} <= set(steps[0]["aliases"])


def test_array_format_command_arguments():
    steps = extract_trajectory_steps(
        trace_with(
            span(
                "tool exec_command",
                {
                    "tool.name": "exec_command",
                    "tool.arguments": '{"cmd":["python","-m","pytest"],"workdir":"/ws"}',
                },
            )
        )
    )
    assert steps[0]["type"] == "command"
    assert steps[0]["name"] == "python -m pytest"
    assert {"exec_command", "python -m pytest", "python"} <= set(steps[0]["aliases"])


def test_exec_command_without_cmd_falls_back_to_tool():
    steps = extract_trajectory_steps(
        trace_with(
            span(
                "tool exec_command",
                {"tool.name": "exec_command", "tool.arguments": '{"workdir":"/tmp"}'},
            )
        )
    )
    assert steps[0]["type"] == "tool"
    assert steps[0]["name"] == "exec_command"


def test_command_tool_names_case_insensitive_with_trimming():
    steps = extract_trajectory_steps(
        trace_with(span("tool Shell", {"tool.name": " Shell ", "tool.arguments": '{"cmd":"pwd"}'}))
    )
    assert steps[0]["type"] == "command"
    assert steps[0]["name"] == "pwd"


@pytest.mark.parametrize(
    "arg_key", ["ai.toolCall.args", "ai.toolCall.arguments", "ai.toolCall.input"]
)
def test_vercel_ai_sdk_tool_spans(arg_key: str):
    steps = extract_trajectory_steps(
        trace_with(
            span(
                "ai.toolCall",
                {
                    "ai.toolCall.name": "search_orders",
                    arg_key: '{"order_id":"123","include_history":false}',
                    "ai.toolCall.result": '{"status":"ok"}',
                },
            )
        )
    )
    assert len(steps) == 1
    assert steps[0]["type"] == "tool"
    assert steps[0]["name"] == "search_orders"
    assert {"ai.toolCall", "search_orders"} <= set(steps[0]["aliases"])
    assert steps[0]["args"] == {"order_id": "123", "include_history": False}


def test_preserves_span_order_when_timestamps_tie():
    steps = extract_trajectory_steps(
        trace_with(
            span("tool.call", {"tool.name": "compose_reply"}),
            span("tool.call", {"tool.name": "search_orders"}),
            span("tool.call", {"tool.name": "finalize"}),
        )
    )
    assert [s["name"] for s in steps] == ["compose_reply", "search_orders", "finalize"]


def test_steps_sorted_by_start_then_end_time():
    a = {**span("tool.call", {"tool.name": "late"}, 2000)}
    b = {**span("tool.call", {"tool.name": "long"}, 1000), "endTime": 1500}
    c = {**span("tool.call", {"tool.name": "short"}, 1000), "endTime": 1200}
    assert [s["name"] for s in extract_trajectory_steps(trace_with(a, b, c))] == [
        "short",
        "long",
        "late",
    ]


# --- trajectory:tool-used ----------------------------------------------------------------------


@pytest.mark.parametrize("base_type", sorted(HANDLERS))
def test_throws_when_trace_data_is_not_available(base_type: str):
    params = make_params(base_type, "search_orders", trace=None)
    with pytest.raises(
        AssertionValueError, match=f"No trace data available for {base_type} assertion"
    ):
        HANDLERS[base_type](params)


def test_tool_used_passes_when_required_tool_present():
    params = make_params("trajectory:tool-used", "search_orders")
    expect_result(
        handle_trajectory_tool_used(params),
        True,
        f"Observed required tool(s): search_orders. Actual tools: {ACTUAL_TOOLS}",
        params,
    )


def test_tool_used_count_matching_with_patterns():
    params = make_params("trajectory:tool-used", {"pattern": "search*", "min": 2, "max": 2})
    expect_result(
        handle_trajectory_tool_used(params),
        True,
        'Matched tool "search*" 2 time(s) (expected 2-2). '
        "Matches: tool:search_orders, tool:search_inventory",
        params,
    )


def test_tool_used_minimum_only_count():
    params = make_params("trajectory:tool-used", {"pattern": "search*", "min": 1})
    expect_result(
        handle_trajectory_tool_used(params),
        True,
        'Matched tool "search*" 2 time(s) (expected at least 1). '
        "Matches: tool:search_orders, tool:search_inventory",
        params,
    )


def test_tool_used_inverse():
    params = make_params("trajectory:tool-used", "compose_reply", inverse=True)
    expect_result(
        handle_trajectory_tool_used(params),
        False,
        f"Forbidden tool(s) were used: compose_reply. Actual tools: {ACTUAL_TOOLS}",
        params,
    )


def test_tool_used_inverse_list_fails_when_any_forbidden_present():
    params = make_params("trajectory:tool-used", ["missing_tool", "compose_reply"], inverse=True)
    expect_result(
        handle_trajectory_tool_used(params),
        False,
        f"Forbidden tool(s) were used: compose_reply. Actual tools: {ACTUAL_TOOLS}",
        params,
    )


def test_tool_used_inverse_list_passes_when_absent():
    params = make_params("trajectory:tool-used", ["missing_tool"], inverse=True)
    expect_result(
        handle_trajectory_tool_used(params),
        True,
        "Forbidden tool(s) were not used: missing_tool",
        params,
    )


def test_tool_used_inverse_count_not_satisfied():
    params = make_params("trajectory:tool-used", {"name": "missing_tool", "min": 1}, inverse=True)
    expect_result(
        handle_trajectory_tool_used(params),
        True,
        'Tool "missing_tool" did not satisfy the forbidden match condition',
        params,
    )


def test_tool_used_inverse_count_satisfied():
    params = make_params("trajectory:tool-used", {"name": "compose_reply"}, inverse=True)
    expect_result(
        handle_trajectory_tool_used(params),
        False,
        'Tool "compose_reply" matched 1 time(s), which violates the inverse assertion',
        params,
    )


def test_tool_used_fails_when_required_tools_missing():
    params = make_params("trajectory:tool-used", ["search_orders", "missing_tool"])
    expect_result(
        handle_trajectory_tool_used(params),
        False,
        f"Missing required tool(s): missing_tool. Actual tools: {ACTUAL_TOOLS}",
        params,
    )


def test_tool_used_actual_tools_none():
    params = make_params(
        "trajectory:tool-used", "search_orders", trace=trace_with(span("chat", {}))
    )
    expect_result(
        handle_trajectory_tool_used(params),
        False,
        "Missing required tool(s): search_orders. Actual tools: (none)",
        params,
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (123, "trajectory:tool-used assertion must have a string, string array, or object value"),
        (
            {"min": 1},
            "trajectory:tool-used assertion object must include a name or pattern property",
        ),
        ([], "trajectory:tool-used assertion requires at least one expected tool"),
    ],
)
def test_tool_used_rejects_invalid_values(value: Any, message: str):
    with pytest.raises(AssertionValueError) as exc:
        handle_trajectory_tool_used(make_params("trajectory:tool-used", value))
    assert str(exc.value) == message


# --- trajectory:tool-sequence ------------------------------------------------------------------

SEARCH_THEN_REPLY = {"mode": "exact", "steps": ["search_orders", "compose_reply"]}


def test_tool_sequence_in_order():
    params = make_params("trajectory:tool-sequence", ["search_orders", "compose_reply"])
    expect_result(
        handle_trajectory_tool_sequence(params),
        True,
        "Observed tool sequence in order: tool:search_orders, tool:compose_reply. "
        f"Actual tools: {ACTUAL_TOOLS}",
        params,
    )


def test_tool_sequence_exact_fails_with_additional_tools():
    params = make_params("trajectory:tool-sequence", SEARCH_THEN_REPLY)
    expect_result(
        handle_trajectory_tool_sequence(params),
        False,
        "Expected exact tool sequence of search_orders, compose_reply, but actual tools were "
        f"{ACTUAL_TOOLS}",
        params,
    )


def test_tool_sequence_exact_passes():
    t = trace_with(
        span("tool.call", {"tool.name": "search_orders"}, 1000),
        span("tool.call", {"tool.name": "compose_reply"}, 1200),
    )
    params = make_params("trajectory:tool-sequence", SEARCH_THEN_REPLY, trace=t)
    expect_result(
        handle_trajectory_tool_sequence(params),
        True,
        "Observed exact tool sequence: tool:search_orders, tool:compose_reply",
        params,
    )


def test_tool_sequence_not_in_order():
    params = make_params("trajectory:tool-sequence", ["compose_reply", "search_orders"])
    expect_result(
        handle_trajectory_tool_sequence(params),
        False,
        f'Expected tool "search_orders" was not observed in order. Actual tools: {ACTUAL_TOOLS}',
        params,
    )


def test_tool_sequence_inverse_present():
    params = make_params(
        "trajectory:tool-sequence", ["search_orders", "compose_reply"], inverse=True
    )
    expect_result(
        handle_trajectory_tool_sequence(params),
        False,
        f"Forbidden tool sequence was observed. Actual tools: {ACTUAL_TOOLS}",
        params,
    )


def test_tool_sequence_inverse_absent():
    params = make_params(
        "trajectory:tool-sequence", ["compose_reply", "search_orders"], inverse=True
    )
    expect_result(
        handle_trajectory_tool_sequence(params),
        True,
        "Forbidden tool sequence was not observed",
        params,
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (
            [{"type": "tool"}, "compose_reply"],
            "trajectory:tool-sequence assertion step 1 must include a name or pattern property",
        ),
        ("search_orders", "trajectory:tool-sequence assertion must have an array or object value"),
        ({"steps": []}, "trajectory:tool-sequence assertion requires at least one expected step"),
    ],
)
def test_tool_sequence_rejects_invalid_values(value: Any, message: str):
    with pytest.raises(AssertionValueError) as exc:
        handle_trajectory_tool_sequence(make_params("trajectory:tool-sequence", value))
    assert str(exc.value) == message


# --- trajectory:tool-args-match ----------------------------------------------------------------

ARGS = "trajectory:tool-args-match"
ORDERS_PAGINATION = {
    "name": "orders",
    "mode": "exact",
    "args": {"status": "Q"},
    "defaults": {"page": 1, "page_size": 5},
}
ORDERS_PAGE = {"name": "orders", "mode": "exact", "args": {"status": "Q"}, "defaults": {"page": 1}}


def search_history(mode: str, include_history: bool) -> dict[str, Any]:
    return {
        "name": "search_orders",
        "mode": mode,
        "args": {"order_id": "123"},
        "defaults": {"include_history": include_history},
    }


def test_args_partial_subset():
    params = make_params(ARGS, {"name": "search_orders", "args": {"order_id": "123"}})
    expect_result(
        handle_trajectory_tool_args_match(params),
        True,
        'Tool "search_orders" matched expected arguments (partial) on tool:search_orders. '
        'Args: {"order_id":"123","include_history":false}',
        params,
    )


def test_args_partial_array_matching():
    params = make_params(ARGS, {"name": "compose_reply", "args": {"citations": ["doc_1", "doc_2"]}})
    expect_result(
        handle_trajectory_tool_args_match(params),
        True,
        'Tool "compose_reply" matched expected arguments (partial) on tool:compose_reply. '
        'Args: {"tone":"friendly","citations":["doc_1","doc_2"]}',
        params,
    )


def test_args_exact_mode():
    value = {
        "pattern": "compose_*",
        "mode": "exact",
        "arguments": {"tone": "friendly", "citations": ["doc_1", "doc_2"]},
    }
    params = make_params(ARGS, value)
    expect_result(
        handle_trajectory_tool_args_match(params),
        True,
        'Tool "compose_*" matched expected arguments (exact) on tool:compose_reply. '
        'Args: {"tone":"friendly","citations":["doc_1","doc_2"]}',
        params,
    )


def test_args_vercel_spans_end_to_end():
    t = trace_with(
        span(
            "ai.toolCall",
            {
                "ai.toolCall.name": "lookup_customer",
                "ai.toolCall.args": '{"customer_id":"cust_1234","include_history":true}',
                "ai.toolCall.result": '{"plan":"pro"}',
            },
        )
    )
    params = make_params(
        ARGS, {"name": "lookup_customer", "args": {"customer_id": "cust_1234"}}, trace=t
    )
    result = handle_trajectory_tool_args_match(params)
    assert result["pass"] is True
    assert result["score"] == 1


def test_args_no_match():
    params = make_params(ARGS, {"name": "search_orders", "args": {"order_id": "999"}})
    expect_result(
        handle_trajectory_tool_args_match(params),
        False,
        'No call to tool "search_orders" matched expected arguments (partial): {"order_id":"999"}. '
        'Observed args: {"order_id":"123","include_history":false}',
        params,
    )


def test_args_no_tool_matches():
    params = make_params(ARGS, {"name": "missing_tool", "args": {"order_id": "123"}})
    expect_result(
        handle_trajectory_tool_args_match(params),
        False,
        f'No tool call matched "missing_tool". Actual tools: {ACTUAL_TOOLS}',
        params,
    )


def test_args_tool_without_captured_args():
    t = trace_with(span("tool.call", {"tool.name": "search_orders"}))
    params = make_params(ARGS, {"name": "search_orders", "args": {"order_id": "123"}}, trace=t)
    expect_result(
        handle_trajectory_tool_args_match(params),
        False,
        'Tool "search_orders" was observed but no arguments were captured. '
        "Actual tools: tool:search_orders",
        params,
    )


def test_args_inverse_observed():
    params = make_params(ARGS, {"name": "search_orders", "args": {"order_id": "123"}}, inverse=True)
    expect_result(
        handle_trajectory_tool_args_match(params),
        False,
        'Forbidden argument match for tool "search_orders" was observed on tool:search_orders. '
        'Args: {"order_id":"123","include_history":false}',
        params,
    )


def test_args_inverse_no_tool_matched():
    params = make_params(ARGS, {"name": "missing_tool", "args": {"order_id": "123"}}, inverse=True)
    expect_result(
        handle_trajectory_tool_args_match(params),
        True,
        'Forbidden argument match for tool "missing_tool" was not observed because no tool call '
        "matched it",
        params,
    )


def test_args_inverse_not_observed_with_args():
    params = make_params(ARGS, {"name": "search_orders", "args": {"order_id": "9"}}, inverse=True)
    expect_result(
        handle_trajectory_tool_args_match(params),
        True,
        'Forbidden argument match for tool "search_orders" was not observed. '
        'Observed args: {"order_id":"123","include_history":false}',
        params,
    )


DEFAULTS_ERROR = (
    "trajectory:tool-args-match assertion defaults must be an object mapping argument names to "
    "default values"
)
IGNORE_ERROR = (
    "trajectory:tool-args-match assertion ignore must be a non-empty string or an array of "
    "non-empty strings"
)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (
            {"name": "search_orders"},
            "trajectory:tool-args-match assertion must include an args or arguments property",
        ),
        (
            {"name": "search_orders", "mode": "excat", "args": {"order_id": "123"}},
            'trajectory:tool-args-match assertion mode must be "partial" or "exact"',
        ),
        ("search_orders", "trajectory:tool-args-match assertion must have an object value"),
        (
            {"args": {"order_id": "123"}},
            "trajectory:tool-args-match assertion object must include a name or pattern property",
        ),
        ({"name": "search_orders", "args": {}, "defaults": ["include_history"]}, DEFAULTS_ERROR),
        ({"name": "search_orders", "args": {}, "defaults": None}, DEFAULTS_ERROR),
        ({"name": "search_orders", "args": {}, "defaults": "include_history"}, DEFAULTS_ERROR),
        ({"name": "search_orders", "args": {}, "defaults": 1}, DEFAULTS_ERROR),
        ({"name": "search_orders", "args": {}, "ignore": [5]}, IGNORE_ERROR),
        ({"name": "search_orders", "args": {}, "ignore": [""]}, IGNORE_ERROR),
        ({"name": "search_orders", "args": {}, "ignore": [None]}, IGNORE_ERROR),
        ({"name": "search_orders", "args": {}, "ignore": [{"request_id": True}]}, IGNORE_ERROR),
    ],
)
def test_args_rejects_invalid_values(value: Any, message: str):
    with pytest.raises(AssertionValueError) as exc:
        handle_trajectory_tool_args_match(make_params(ARGS, value))
    assert str(exc.value) == message


def test_args_null_args_is_a_valid_expectation():
    t = tool_trace("search_orders", "null")
    result = handle_trajectory_tool_args_match(
        make_params(ARGS, {"name": "search_orders", "args": None}, trace=t)
    )
    assert result["pass"] is True
    assert result["reason"].endswith("Args: null")


# defaults


def test_defaults_exact_mode_strips_matching_default():
    params = make_params(ARGS, search_history("exact", False))
    expect_result(
        handle_trajectory_tool_args_match(params),
        True,
        'Tool "search_orders" matched expected arguments (exact) on tool:search_orders. '
        'Args: {"order_id":"123","include_history":false}. '
        "Ignored default argument(s): include_history",
        params,
    )


def test_defaults_exact_mode_fails_when_value_differs():
    result = handle_trajectory_tool_args_match(make_params(ARGS, search_history("exact", True)))
    assert result["pass"] is False
    assert result["score"] == 0
    assert 'No call to tool "search_orders" matched expected arguments' in result["reason"]


@pytest.mark.parametrize(
    ("arguments", "value", "passed"),
    [
        ('{"status":"Q","page":1,"page_size":5}', ORDERS_PAGINATION, True),
        ('{"status":"Q"}', ORDERS_PAGINATION, True),
        ('{"status":"Q","page":1,"delete_database":true}', ORDERS_PAGE, False),
        (
            '{"status":"Q","sort":{"field":"created","order":"desc"}}',
            {**ORDERS_PAGE, "defaults": {"sort": {"field": "created", "order": "desc"}}},
            True,
        ),
        ('{"status":"Q","page":1}', ORDERS_PAGINATION, True),
        ('{"status":"Q","page":2}', ORDERS_PAGINATION, False),
        ('{"status":"Q","cursor":null}', {**ORDERS_PAGE, "defaults": {"cursor": None}}, True),
        (
            '{"status":"Q","page":1}',
            {**ORDERS_PAGE, "args": {"status": "Q", "page": 2}},
            False,
        ),
        ('{"status":"Q","pagination":{"page":1}}', ORDERS_PAGE, False),
        ('{"status":"Q","__proto__":{"polluted":true}}', ORDERS_PAGE, False),
        (
            '{"status":"Q","page":1}',
            {"name": "orders", "args": {"status": "Q", "page": 1}, "defaults": {"page": 1}},
            False,
        ),
        ("[1,2,3]", {**ORDERS_PAGE, "args": [1, 2, 3]}, True),
    ],
)
def test_defaults_table(arguments: str, value: dict[str, Any], passed: bool):
    result = handle_trajectory_tool_args_match(
        make_params(ARGS, value, trace=tool_trace("orders", arguments))
    )
    assert result["pass"] is passed
    if "delete_database" in arguments:
        assert "delete_database" in result["reason"]


def test_defaults_empty_same_as_none():
    value = {"name": "search_orders", "mode": "exact", "args": {"order_id": "123"}, "defaults": {}}
    assert handle_trajectory_tool_args_match(make_params(ARGS, value))["pass"] is False


def test_defaults_partial_mode_still_passes():
    value = {
        "name": "search_orders",
        "args": {"order_id": "123"},
        "defaults": {"include_history": False},
    }
    assert handle_trajectory_tool_args_match(make_params(ARGS, value))["pass"] is True


def test_defaults_inverse():
    result = handle_trajectory_tool_args_match(
        make_params(ARGS, search_history("exact", False), inverse=True)
    )
    assert result["pass"] is False
    assert "Forbidden argument match" in result["reason"]


def test_defaults_multiple_stripped_keys_in_reason():
    t = tool_trace("orders", '{"status":"Q","page":1,"page_size":5}')
    result = handle_trajectory_tool_args_match(make_params(ARGS, ORDERS_PAGINATION, trace=t))
    assert result["pass"] is True
    assert "Ignored default argument(s): page, page_size" in result["reason"]


def test_defaults_no_note_when_nothing_stripped():
    result = handle_trajectory_tool_args_match(make_params(ARGS, search_history("partial", True)))
    assert result["pass"] is True
    assert "Ignored default" not in result["reason"]


def test_defaults_arguments_alias():
    value = {
        "name": "search_orders",
        "mode": "exact",
        "arguments": {"order_id": "123"},
        "defaults": {"include_history": False},
    }
    assert handle_trajectory_tool_args_match(make_params(ARGS, value))["pass"] is True


def test_defaults_literal_star_is_not_a_wildcard():
    value = {
        "name": "search_orders",
        "mode": "exact",
        "args": {"status": "Q"},
        "defaults": {"fields": "*"},
    }

    def run(fields: str) -> dict:
        t = tool_trace("search_orders", {"status": "Q", "fields": fields})
        return handle_trajectory_tool_args_match(make_params(ARGS, value, trace=t))

    stripped = run("*")
    assert stripped["pass"] is True
    assert "Ignored default argument(s): fields" in stripped["reason"]
    assert run("ssn,password")["pass"] is False


# ignore


def ignore_run(tool_args: Any, value: dict[str, Any], inverse: bool = False) -> dict:
    return handle_trajectory_tool_args_match(
        make_params(ARGS, value, inverse=inverse, trace=tool_trace("search_orders", tool_args))
    )


EXACT_Q = {"name": "search_orders", "mode": "exact", "args": {"status": "Q"}}


def test_ignore_drops_key_regardless_of_value():
    value = {**EXACT_Q, "ignore": ["request_id"]}
    assert ignore_run({"status": "Q", "request_id": "a"}, value)["pass"] is True
    assert ignore_run({"status": "Q", "request_id": "z"}, value)["pass"] is True


@pytest.mark.parametrize("cursor", ["eyJvZmZzZXQiOjQyfQ==", None, 0, {"page": 3}])
def test_ignore_across_value_types(cursor: Any):
    value = {**EXACT_Q, "ignore": ["cursor"]}
    assert ignore_run({"status": "Q", "cursor": cursor}, value)["pass"] is True


@pytest.mark.parametrize(
    ("tool_args", "value", "passed"),
    [
        ({"status": "Q", "request_id": "abc"}, {**EXACT_Q, "ignore": "request_id"}, True),
        (
            {"status": "Q", "request_id": "abc", "delete_database": True},
            {**EXACT_Q, "ignore": ["request_id"]},
            False,
        ),
        (
            {"status": "Q"},
            {
                **EXACT_Q,
                "args": {"status": "Q", "request_id": "expected"},
                "ignore": ["request_id"],
            },
            True,
        ),
        (
            {"status": "Q", "request_id": "abc"},
            {**EXACT_Q, "mode": "partial", "ignore": ["request_id"]},
            True,
        ),
        (
            '{"status":"Q","__proto__":{"polluted":true},"request_id":"abc"}',
            {**EXACT_Q, "ignore": ["request_id"]},
            False,
        ),
        ('["a","b"]', {**EXACT_Q, "args": ["a", "b"], "ignore": ["request_id"]}, True),
        (
            {"status": "Q", "request_id": "a1", "delete_database": True},
            {**EXACT_Q, "ignore": ["*_id"]},
            False,
        ),
        ({"status": "Q", "v1": "x"}, {**EXACT_Q, "ignore": ["v?"]}, True),
        ({"status": "Q", "request_id": "a1"}, {**EXACT_Q, "ignore": ["request"]}, False),
    ],
)
def test_ignore_table(tool_args: Any, value: dict[str, Any], passed: bool):
    assert ignore_run(tool_args, value)["pass"] is passed


def test_ignore_composes_with_defaults():
    value = {**EXACT_Q, "defaults": {"page": 1}, "ignore": ["request_id"]}
    result = ignore_run({"status": "Q", "page": 1, "request_id": "abc"}, value)
    assert result["pass"] is True
    assert "Ignored argument(s): request_id" in result["reason"]
    assert "Ignored default argument(s): page" in result["reason"]


def test_ignore_inverts():
    value = {**EXACT_Q, "ignore": ["request_id"]}
    assert ignore_run({"status": "Q", "request_id": "abc"}, value, inverse=True)["pass"] is False


def test_ignore_glob_patterns():
    result = ignore_run(
        {"status": "Q", "request_id": "a1", "order_id": "b2"}, {**EXACT_Q, "ignore": ["*_id"]}
    )
    assert result["pass"] is True
    assert "Ignored argument(s): request_id, order_id" in result["reason"]


# --- trajectory:step-count ---------------------------------------------------------------------

STEPS = "trajectory:step-count"


def test_step_count_by_type():
    params = make_params(STEPS, {"type": "command", "min": 1, "max": 1})
    expect_result(
        handle_trajectory_step_count(params),
        True,
        "Matched 1 trajectory step(s) for type=command (expected 1-1). Matches: command:ls -la",
        params,
    )


def test_step_count_by_pattern():
    params = make_params(STEPS, {"pattern": "reasoning*", "min": 1})
    expect_result(
        handle_trajectory_step_count(params),
        True,
        "Matched 1 trajectory step(s) for pattern=reasoning* (expected at least 1). "
        "Matches: reasoning:reasoning_plan",
        params,
    )


def test_step_count_max_only():
    params = make_params(STEPS, {"type": "tool", "max": 3})
    expect_result(
        handle_trajectory_step_count(params),
        True,
        f"Matched 3 trajectory step(s) for type=tool (expected at most 3). Matches: {ACTUAL_TOOLS}",
        params,
    )


def test_step_count_multiple_types_and_failure():
    params = make_params(STEPS, {"type": ["tool", "command"], "max": 1})
    result = handle_trajectory_step_count(params)
    assert result["pass"] is False
    assert result["reason"].startswith(
        "Matched 4 trajectory step(s) for type=tool|command (expected at most 1)"
    )


def test_step_count_inverse():
    params = make_params(STEPS, {"type": "tool", "min": 4}, inverse=True)
    expect_result(
        handle_trajectory_step_count(params),
        True,
        "Trajectory step count did not satisfy the forbidden range",
        params,
    )


def test_step_count_inverse_satisfied():
    params = make_params(STEPS, {"type": "tool", "min": 1}, inverse=True)
    expect_result(
        handle_trajectory_step_count(params),
        False,
        "Trajectory step count satisfied the forbidden range",
        params,
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"type": "tool"}, "trajectory:step-count assertion must include a min or max property"),
        ("tool", "trajectory:step-count assertion must have an object value"),
    ],
)
def test_step_count_rejects_invalid_values(value: Any, message: str):
    with pytest.raises(AssertionValueError) as exc:
        handle_trajectory_step_count(make_params(STEPS, value))
    assert str(exc.value) == message


# --- trajectory:goal-success -------------------------------------------------------------------


def test_goal_success_is_not_implemented_yet():
    with pytest.raises(NotImplementedError, match="Phase 4"):
        handle_trajectory_goal_success(make_params("trajectory:goal-success", "Refund the order"))


def test_goal_success_without_trace():
    with pytest.raises(
        AssertionValueError,
        match="No trace data available for trajectory:goal-success assertion",
    ):
        handle_trajectory_goal_success(
            make_params("trajectory:goal-success", "Refund the order", trace=None)
        )


def test_goal_success_rejects_invalid_value():
    with pytest.raises(AssertionValueError) as exc:
        handle_trajectory_goal_success(make_params("trajectory:goal-success", {"goal": "  "}))
    assert str(exc.value) == (
        "trajectory:goal-success assertion must have a string value or an object with a goal "
        "property"
    )


def test_handlers_registered():
    assert set(HANDLERS) == {
        "trajectory:goal-success",
        "trajectory:tool-args-match",
        "trajectory:step-count",
        "trajectory:tool-sequence",
        "trajectory:tool-used",
    }
