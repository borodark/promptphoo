"""Ports test/tracing/sanitizeAttributes.test.ts, plus the UTF-16 truncation edge cases."""

from __future__ import annotations

from promptphoo.tracing.sanitize_attributes import js_length, js_slice, sanitize_trace_attributes


def test_preserves_token_metrics_and_redacts_credentials_recursively():
    tokens = {
        "gen_ai.usage.reasoning.output_tokens": 3,
        "gen_ai.usage.cache_read.input_tokens": 4,
        "gen_ai.usage.cache_creation.input_tokens": 5,
        "gen_ai.usage.total_tokens": 12,
        "promptfoo.usage.total_tokens": 12,
        "promptfoo.usage.cached_response_tokens": 6,
        "promptfoo.usage.accepted_prediction_tokens": 7,
        "promptfoo.usage.rejected_prediction_tokens": 8,
        "llm.usage.prompt_tokens": 9,
        "llm.usage.completion_tokens": 10,
        "llm.usage.total_tokens": 19,
    }
    result = sanitize_trace_attributes(
        {
            **tokens,
            "X-API-Key": "secret",
            "client.secret": "private",
            "login.pass_word": "private",
            "pass.phrase": "private",
            "nested": {"access_token": "secret"},
            "customer_email": "private@example.com",
        },
        {"redactAttributes": ["email"]},
    )
    assert result == {
        **tokens,
        "X-API-Key": "<redacted>",
        "client.secret": "<redacted>",
        "login.pass_word": "<redacted>",
        "pass.phrase": "<redacted>",
        "nested": {"access_token": "<redacted>"},
        "customer_email": "[REDACTED]",
    }


def test_explicit_redactions_apply_without_generic_sanitization():
    assert sanitize_trace_attributes(
        {"authorization": "visible", "private_field": "secret"},
        {"redactAttributes": ["private"], "sanitizeSensitiveAttributes": False},
    ) == {"authorization": "visible", "private_field": "[REDACTED]"}


def test_explicit_redactions_without_truncation():
    long_args = "argument-value " * 40
    assert sanitize_trace_attributes(
        {
            "authorization": "visible",
            "private_field": "secret",
            "gen_ai.tool.call.arguments": long_args,
            "nested": {"customer_email": "private@example.com", "full_input": long_args},
        },
        {
            "redactAttributes": ["private", "email"],
            "sanitizeSensitiveAttributes": False,
            "truncateValues": False,
        },
    ) == {
        "authorization": "visible",
        "private_field": "[REDACTED]",
        "gen_ai.tool.call.arguments": long_args,
        "nested": {"customer_email": "[REDACTED]", "full_input": long_args},
    }


def test_long_values_are_truncated_by_utf16_units():
    result = sanitize_trace_attributes({"plain": "a" * 401, "emoji": "😀" * 201, "ok": "b" * 400})
    assert result["plain"] == "a" * 400 + "…"
    assert result["ok"] == "b" * 400
    # 201 emoji are 402 UTF-16 units; JS slices to 400 units, i.e. 200 whole emoji.
    assert result["emoji"] == "😀" * 200 + "…"


def test_js_length_and_slice():
    assert js_length("a😀") == 3
    assert js_slice("a😀b", 2) == "a\ud83d"
    assert js_slice("abc", 2) == "ab"


def test_none_and_empty_attributes():
    assert sanitize_trace_attributes(None) is None or sanitize_trace_attributes(None) == {}
    assert sanitize_trace_attributes({}) == {}
