"""Ports src/tracing/toolAttributes.ts: the attribute keys that name tools, commands and
searches on a span."""

from __future__ import annotations

from typing import Any

from promptphoo.jsvalues import json_stringify

COMMAND_ATTRIBUTE_KEYS: tuple[str, ...] = (
    "codex.command",
    "command",
    "command.name",
    "command_name",
)
SEARCH_ATTRIBUTE_KEYS: tuple[str, ...] = ("codex.search.query", "search.query", "search_query")

_TOOL_ATTRIBUTE_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "prefix": "tool",
        "nameSuffixes": [".name", "_name"],
        "argSuffixes": [".args", ".arguments", ".input", "_args", "_arguments", "_input"],
        "extraNameKeys": ["tool"],
    },
    {
        "prefix": "function",
        "nameSuffixes": [".name", "_name"],
        "argSuffixes": [".args", ".arguments", ".input", "_args", "_arguments"],
    },
    {
        "prefix": "ai.toolCall",
        "nameSuffixes": [".name"],
        "argSuffixes": [".args", ".arguments", ".input"],
    },
    {
        "prefix": "gen_ai.tool",
        "nameSuffixes": [".name"],
        "argSuffixes": [".args", ".arguments", ".input"],
        "extraArgKeys": ["gen_ai.tool.call.args", "gen_ai.tool.call.arguments"],
    },
    {
        "prefix": "agent.tool",
        "nameSuffixes": ["_name"],
        "argSuffixes": [".args", ".arguments", ".input"],
        "extraNameKeys": ["agent.tool", "agent.toolName"],
    },
    {
        "prefix": "codex.mcp",
        "argSuffixes": [".args", ".arguments", ".input"],
        "extraNameKeys": ["codex.mcp.tool"],
    },
)

TOOL_NAME_ATTRIBUTE_KEYS: tuple[str, ...] = tuple(
    key
    for f in _TOOL_ATTRIBUTE_FAMILIES
    for key in [f["prefix"] + s for s in f.get("nameSuffixes", [])] + f.get("extraNameKeys", [])
)
TOOL_ARGUMENT_ATTRIBUTE_KEYS: tuple[str, ...] = (
    *(
        key
        for f in _TOOL_ATTRIBUTE_FAMILIES
        for key in [f["prefix"] + s for s in f.get("argSuffixes", [])] + f.get("extraArgKeys", [])
    ),
    "args",
    "arguments",
    "input",
)

# String.prototype.trim's whitespace set (ECMAScript WhiteSpace and LineTerminator).
_JS_WHITESPACE = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009"
    "\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def _js_trim(value: str) -> str:
    return value.strip(_JS_WHITESPACE)


def get_normalized_tool_attributes(name: str, args: Any = None) -> dict[str, Any]:
    """Normalize imported tool calls to the attributes used by trace filtering and trajectory
    grading. `args=None` is JS `undefined` (the key is omitted)."""
    out: dict[str, Any] = {"gen_ai.tool.name": name, "tool.name": name}
    if args is not None:
        out["tool.arguments"] = args if isinstance(args, str) else json_stringify(args)
    return out


def get_first_string_attribute(attributes: Any, keys: tuple[str, ...] | list[str]) -> str | None:
    if not isinstance(attributes, dict):
        return None
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, str) and _js_trim(value):
            return _js_trim(value)
    return None


def get_tool_name_from_attributes(attributes: Any) -> str | None:
    return get_first_string_attribute(attributes, TOOL_NAME_ATTRIBUTE_KEYS)
