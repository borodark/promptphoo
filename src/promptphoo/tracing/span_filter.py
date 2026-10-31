"""Ports src/tracing/spanFilter.ts: which spans matter to graders, and span-name filters."""

from __future__ import annotations

import functools
import re
from typing import Any

from promptphoo.tracing.span_roles import SPAN_ROLE_ATTRIBUTE
from promptphoo.tracing.tool_attributes import (
    COMMAND_ATTRIBUTE_KEYS,
    SEARCH_ATTRIBUTE_KEYS,
    get_first_string_attribute,
    get_tool_name_from_attributes,
)

_NOT_LT = "[^\n\r\u2028\u2029]"  # JS `.` without the s flag
_FILTER_ESCAPE = re.compile(r"[|\\{}()\[\]^$+?.]")


def is_relevant_span(span: dict[str, Any]) -> bool:
    """Identifies spans that describe model, tool, command, search, guardrail, or error
    activity. `span` is `{attributes?, statusCode?}`."""
    attributes = span.get("attributes")
    if isinstance(attributes, dict) and attributes.get(SPAN_ROLE_ATTRIBUTE) == "grader":
        return False
    if (
        span.get("statusCode") == 2
        or get_tool_name_from_attributes(attributes)
        or get_first_string_attribute(attributes, COMMAND_ATTRIBUTE_KEYS)
        or get_first_string_attribute(attributes, SEARCH_ATTRIBUTE_KEYS)
    ):
        return True
    for attribute in attributes or {}:
        normalized = str(attribute).lower()
        if (
            normalized.startswith(("gen_ai.", "llm.", "guardrail.", "guardrails."))
            or normalized == "ai.model.id"
        ):
            return True
    return False


@functools.lru_cache(maxsize=256)
def _filter_regex(pattern: str) -> re.Pattern[str]:
    escaped = _FILTER_ESCAPE.sub(lambda m: "\\" + m.group(0), pattern)
    # JS: escaped.replace(/\*/g, '.*').replace(/\\\?/g, '.')
    glob = escaped.replace("*", f"{_NOT_LT}*").replace("\\?", _NOT_LT)
    return re.compile(f"^{glob}\\Z", re.IGNORECASE)


def matches_span_filter(span_name: str, filters: list[str]) -> bool:
    """Matches optional glob filters while preserving existing substring-filter behavior."""
    for flt in filters:
        if "*" not in flt and "?" not in flt:
            if flt.lower() in span_name.lower():
                return True
            continue
        if _filter_regex(flt).search(span_name):
            return True
    return False
