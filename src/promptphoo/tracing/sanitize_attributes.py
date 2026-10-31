# SHIM: replaced by the trace-store branch at merge
"""Ports src/tracing/sanitizeAttributes.ts (sanitizeTraceAttributes)."""

from __future__ import annotations

import re
from typing import Any

# Match normalized names; the cookie pattern also covers set-cookie.
_SENSITIVE_ATTRIBUTE_PATTERN = re.compile(
    r"authorization|cookie|token|apikey|secret|password|passphrase"
)
_NON_ALNUM = re.compile(r"[^a-z0-9]")

SAFE_TOKEN_ATTRIBUTE_KEYS = frozenset({
    "gen_ai.request.max_tokens", "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens",
    "gen_ai.usage.reasoning.output_tokens", "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.cache_creation.input_tokens", "promptfoo.usage.total_tokens",
    "promptfoo.usage.cached_response_tokens", "promptfoo.usage.accepted_prediction_tokens",
    "promptfoo.usage.rejected_prediction_tokens", "llm.usage.prompt_tokens",
    "llm.usage.completion_tokens", "llm.usage.total_tokens", "gen_ai.usage.total_tokens",
    "gen_ai.usage.cached_tokens", "gen_ai.usage.reasoning_tokens",
    "gen_ai.usage.accepted_prediction_tokens", "gen_ai.usage.rejected_prediction_tokens",
    "gen_ai.usage.cache_read_input_tokens", "gen_ai.usage.cache_creation_input_tokens",
})  # fmt: skip


def _utf16_len(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _utf16_slice(value: str, end: int) -> str:
    return value.encode("utf-16-le")[: end * 2].decode("utf-16-le", errors="ignore")


def sanitize_trace_attributes(
    attributes: dict[str, Any] | None, options: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not attributes:
        return {}
    options = options or {}
    redact_attributes = options.get("redactAttributes") or []
    sanitize_sensitive = options.get("sanitizeSensitiveAttributes")
    if sanitize_sensitive is None:
        sanitize_sensitive = True
    truncate_values = options.get("truncateValues")
    if truncate_values is None:
        truncate_values = True
    custom_patterns = list(
        dict.fromkeys(
            p.strip().lower() for p in redact_attributes if isinstance(p, str) and p.strip()
        )
    )

    def sanitize_value(value: Any) -> Any:
        if isinstance(value, str):
            if truncate_values and _utf16_len(value) > 400:
                return f"{_utf16_slice(value, 400)}…"
            return value
        if isinstance(value, list):
            return [sanitize_value(v) for v in value]
        if isinstance(value, dict):
            return sanitize_trace_attributes(value, options)
        return value

    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        lower_key = str(key).lower()
        if any(pattern in lower_key for pattern in custom_patterns):
            sanitized[key] = "[REDACTED]"
            continue
        if (
            sanitize_sensitive
            and lower_key not in SAFE_TOKEN_ATTRIBUTE_KEYS
            and _SENSITIVE_ATTRIBUTE_PATTERN.search(_NON_ALNUM.sub("", lower_key))
        ):
            sanitized[key] = "<redacted>"
            continue
        sanitized[key] = sanitize_value(value)
    return sanitized
