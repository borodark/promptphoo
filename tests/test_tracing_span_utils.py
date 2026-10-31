"""Ports test/tracing/spanFilter.test.ts and test/tracing/toolAttributes.test.ts, plus the
context-based port of src/tracing/spanRoles.ts."""

from __future__ import annotations

import pytest

from promptphoo.tracing.otel_api import ROOT_CONTEXT, SpanContext, context, trace
from promptphoo.tracing.span_filter import is_relevant_span, matches_span_filter
from promptphoo.tracing.span_roles import (
    SPAN_ROLE_ATTRIBUTE,
    get_active_span_role,
    get_active_traceparent,
    with_span_role,
)
from promptphoo.tracing.tool_attributes import (
    TOOL_ARGUMENT_ATTRIBUTE_KEYS,
    TOOL_NAME_ATTRIBUTE_KEYS,
    get_normalized_tool_attributes,
    get_tool_name_from_attributes,
)

# --- spanFilter.test.ts -----------------------------------------------------------------------

RELEVANT = [
    {"gen_ai.operation.name": "chat", "gen_ai.request.model": "gpt-4.1-mini"},
    {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search_knowledge_base"},
    {"ai.toolCall.name": "lookup_customer"},
    {"ai.model.id": "gpt-4.1-mini"},
    {"llm.model": "gpt-4"},
    {"guardrails.decision": "blocked"},
    *({a: "git status"} for a in ("codex.command", "command", "command.name", "command_name")),
    *({a: "customer records"} for a in ("codex.search.query", "search.query", "search_query")),
]


@pytest.mark.parametrize("attributes", RELEVANT)
def test_includes_relevant_activity_regardless_of_span_name(attributes):
    assert is_relevant_span({"attributes": attributes}) is True


def test_includes_errors_without_genai_attributes():
    assert is_relevant_span({"attributes": {}, "statusCode": 2}) is True


def test_excludes_grader_activity():
    assert (
        is_relevant_span(
            {"attributes": {"gen_ai.operation.name": "chat", "promptfoo.span.role": "grader"}}
        )
        is False
    )
    assert (
        is_relevant_span({"attributes": {"promptfoo.span.role": "grader"}, "statusCode": 2})
        is False
    )


@pytest.mark.parametrize(
    "attributes",
    [
        {"http.request.method": "POST"},
        {"url.full": "https://example.com/chat"},
        {"otel.span.kind": "internal"},
        {"command.output": "git status output"},
        {"search.results": "customer records"},
        {"command": "  "},
        {"search.query": ""},
        {},
    ],
)
def test_excludes_framework_and_http_spans(attributes):
    assert is_relevant_span({"attributes": attributes}) is False


@pytest.mark.parametrize(
    ("span_name", "filters"),
    [
        ("llm.generate", ["llm.*"]),
        ("chat gpt-4.1-mini", ["chat*"]),
        ("execute_tool search_knowledge_base", ["*tool*"]),
        ("guardrail.check", ["guardrail.?heck"]),
        ("CHAT GPT-4.1-MINI", ["chat*"]),
    ],
)
def test_matches_wildcard_filters(span_name, filters):
    assert matches_span_filter(span_name, filters) is True


def test_plain_filters_are_case_insensitive_substrings():
    assert matches_span_filter("execute_tool search_knowledge_base", ["KNOWLEDGE"]) is True


def test_regex_punctuation_is_literal():
    assert matches_span_filter("target.call", ["target.(call)"]) is False
    assert matches_span_filter("target.call", ["target.*"]) is True


def test_any_filter_may_match():
    assert matches_span_filter("chat gpt-4.1-mini", ["guardrail*", "chat*"]) is True


# --- toolAttributes.test.ts -------------------------------------------------------------------


def test_normalizes_structured_tool_arguments():
    assert get_normalized_tool_attributes("search", {"query": "customer orders"}) == {
        "gen_ai.tool.name": "search",
        "tool.name": "search",
        "tool.arguments": '{"query":"customer orders"}',
    }


def test_preserves_string_arguments_and_omits_absent_ones():
    assert (
        get_normalized_tool_attributes("search", '{"query":"orders"}')["tool.arguments"]
        == '{"query":"orders"}'
    )
    assert "tool.arguments" not in get_normalized_tool_attributes("search")


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        (None, None),
        ({"foo": "bar"}, None),
        ({"ai.toolCall.name": "lookup_customer"}, "lookup_customer"),
        ({"tool.name": "search_orders", "ai.toolCall.name": "lookup_customer"}, "search_orders"),
        ({"function.name": "compose_reply"}, "compose_reply"),
        ({"tool.name": "  trimmed_tool  "}, "trimmed_tool"),
        ({"tool.name": "   ", "ai.toolCall.name": "fallback_tool"}, "fallback_tool"),
        ({"tool.name": 42, "ai.toolCall.name": "fallback_tool"}, "fallback_tool"),
        ({"workflow.tool": "should_not_match", "tool.description": "not a tool name"}, None),
    ],
)
def test_get_tool_name_from_attributes(attributes, expected):
    assert get_tool_name_from_attributes(attributes) == expected


def test_tool_name_key_table():
    assert sorted(TOOL_NAME_ATTRIBUTE_KEYS) == [
        "agent.tool", "agent.toolName", "agent.tool_name", "ai.toolCall.name", "codex.mcp.tool",
        "function.name", "function_name", "gen_ai.tool.name", "tool", "tool.name", "tool_name",
    ]  # fmt: skip


def test_tool_argument_key_table():
    assert sorted(TOOL_ARGUMENT_ATTRIBUTE_KEYS) == [
        "agent.tool.args", "agent.tool.arguments", "agent.tool.input", "ai.toolCall.args",
        "ai.toolCall.arguments", "ai.toolCall.input", "args", "arguments", "codex.mcp.args",
        "codex.mcp.arguments", "codex.mcp.input", "function.args", "function.arguments",
        "function.input", "function_args", "function_arguments", "gen_ai.tool.args",
        "gen_ai.tool.arguments", "gen_ai.tool.call.args", "gen_ai.tool.call.arguments",
        "gen_ai.tool.input", "input", "tool.args", "tool.arguments", "tool.input", "tool_args",
        "tool_arguments", "tool_input",
    ]  # fmt: skip


def test_key_tables_have_no_duplicates():
    assert len(set(TOOL_NAME_ATTRIBUTE_KEYS)) == len(TOOL_NAME_ATTRIBUTE_KEYS)
    assert len(set(TOOL_ARGUMENT_ATTRIBUTE_KEYS)) == len(TOOL_ARGUMENT_ATTRIBUTE_KEYS)


# --- spanRoles.ts -----------------------------------------------------------------------------


async def test_span_role_is_scoped():
    assert SPAN_ROLE_ATTRIBUTE == "promptfoo.span.role"
    assert get_active_span_role() is None
    assert await with_span_role("grader", get_active_span_role) == "grader"
    assert get_active_span_role() is None


def test_active_traceparent_requires_a_valid_span_context():
    assert get_active_traceparent() is None
    valid = trace.set_span_context(ROOT_CONTEXT, SpanContext("a" * 32, "b" * 16, 1))
    with context.use(valid):
        assert get_active_traceparent() == f"00-{'a' * 32}-{'b' * 16}-01"
    invalid = trace.set_span_context(ROOT_CONTEXT, SpanContext("0" * 32, "b" * 16, 1))
    with context.use(invalid):
        assert get_active_traceparent() is None
