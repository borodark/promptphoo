"""Finish-reason normalization. Ports src/util/finishReason.ts."""

from __future__ import annotations

from typing import Any

FINISH_REASON_MAP: dict[str, str] = {
    # OpenAI - standard values (pass through unchanged)
    "stop": "stop",
    "length": "length",
    "content_filter": "content_filter",
    "tool_calls": "tool_calls",
    # OpenAI - legacy mappings
    "function_call": "tool_calls",
    # OpenAI - Responses API incomplete reason
    "max_output_tokens": "length",
    # Anthropic - Messages API mappings
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
    "pause_turn": "pause_turn",
    # Vercel AI SDK - unified values use hyphens
    "tool-calls": "tool_calls",
    "content-filter": "content_filter",
}


def normalize_finish_reason(raw: Any = None) -> str | None:
    """Normalize a provider finish/stop reason; unknown values are trimmed and lowercased."""
    if raw is None or not isinstance(raw, str):
        return None
    trimmed = raw.strip()
    if trimmed == "":
        return None
    key = trimmed.lower()
    return FINISH_REASON_MAP.get(key, key)
