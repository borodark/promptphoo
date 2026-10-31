"""Trace attribute redaction and truncation. Ports src/tracing/sanitizeAttributes.ts."""

from __future__ import annotations

import re
from typing import Any

# Match normalized names; the cookie pattern also covers set-cookie.
SENSITIVE_ATTRIBUTE_PATTERN = re.compile(
    r"authorization|cookie|token|apikey|secret|password|passphrase"
)

SAFE_TOKEN_ATTRIBUTE_KEYS = frozenset({
    "gen_ai.request.max_tokens",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.reasoning.output_tokens",
    "gen_ai.usage.cache_read.input_tokens",
    "gen_ai.usage.cache_creation.input_tokens",
    "promptfoo.usage.total_tokens",
    "promptfoo.usage.cached_response_tokens",
    "promptfoo.usage.accepted_prediction_tokens",
    "promptfoo.usage.rejected_prediction_tokens",
    # Preserve token counts from externally instrumented LLM applications.
    "llm.usage.prompt_tokens",
    "llm.usage.completion_tokens",
    "llm.usage.total_tokens",
    # Keep historical span attributes readable after upgrading.
    "gen_ai.usage.total_tokens",
    "gen_ai.usage.cached_tokens",
    "gen_ai.usage.reasoning_tokens",
    "gen_ai.usage.accepted_prediction_tokens",
    "gen_ai.usage.rejected_prediction_tokens",
    "gen_ai.usage.cache_read_input_tokens",
    "gen_ai.usage.cache_creation_input_tokens",
})  # fmt: skip

_NON_ALNUM = re.compile(r"[^a-z0-9]")
MAX_ATTRIBUTE_VALUE_LENGTH = 400


def js_length(value: str) -> int:
    """String.prototype.length: UTF-16 code units."""
    if value.isascii():
        return len(value)
    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def js_slice(value: str, end: int) -> str:
    """value.slice(0, end) in UTF-16 code units; a split surrogate pair keeps its lone half."""
    if value.isascii():
        return value[:end]
    units = value.encode("utf-16-le", errors="surrogatepass")[: end * 2]
    return units.decode("utf-16-le", errors="surrogatepass")


def _js_lower(value: str) -> str:
    """String.prototype.toLowerCase (Python's lower() maps 'İ' to two code points, as JS does)."""
    return value.lower()


def sanitize_trace_attributes(
    attributes: dict[str, Any] | None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """sanitizeTraceAttributes. Options use upstream's keys: `redactAttributes`,
    `sanitizeSensitiveAttributes` (default true) and `truncateValues` (default true)."""
    if not attributes:
        return {}
    options = options or {}
    redact_attributes = options.get("redactAttributes")
    if redact_attributes is None:
        redact_attributes = []
    sanitize_sensitive = options.get("sanitizeSensitiveAttributes")
    sanitize_sensitive = True if sanitize_sensitive is None else sanitize_sensitive
    truncate_values = options.get("truncateValues")
    truncate_values = True if truncate_values is None else truncate_values
    custom_patterns = list(
        dict.fromkeys(
            pattern
            for pattern in (
                _js_lower(p.strip()) if isinstance(p, str) else "" for p in redact_attributes
            )
            if pattern
        )
    )

    def sanitize_value(value: Any) -> Any:
        if isinstance(value, str):
            if truncate_values and js_length(value) > MAX_ATTRIBUTE_VALUE_LENGTH:
                return f"{js_slice(value, MAX_ATTRIBUTE_VALUE_LENGTH)}…"
            return value
        if isinstance(value, list):
            return [sanitize_value(v) for v in value]
        if isinstance(value, dict):
            return sanitize_trace_attributes(value, options)
        return value

    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        lower_key = _js_lower(key)
        if any(pattern in lower_key for pattern in custom_patterns):
            sanitized[key] = "[REDACTED]"
            continue
        if (
            sanitize_sensitive
            and lower_key not in SAFE_TOKEN_ATTRIBUTE_KEYS
            and SENSITIVE_ATTRIBUTE_PATTERN.search(_NON_ALNUM.sub("", lower_key))
        ):
            sanitized[key] = "<redacted>"
            continue
        sanitized[key] = sanitize_value(value)
    return sanitized
