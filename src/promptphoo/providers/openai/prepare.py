"""Load and render OpenAI request fragments (response_format, functions, tools, reasoning).
Ports src/providers/openai/prepare.ts.

Validation mirrors the zod schemas' structure checks without stripping vendor fields; failures
raise ValueError carrying zod-v4-shaped issue JSON.
"""

from __future__ import annotations

import json
import math
from typing import Any

from promptphoo.assertions.tools import (
    maybe_load_tools_from_external_file,
    prepare_openai_functions,
)
from promptphoo.files import maybe_load_from_external_file
from promptphoo.templating import render_vars_in_object

__all__ = [
    "is_openai_json_schema_format",
    "prepare_openai_functions",
    "prepare_openai_reasoning",
    "prepare_openai_response_format",
    "prepare_openai_tools",
]


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _is_file_reference(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("file://")


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "NaN" if isinstance(value, float) and math.isnan(value) else "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _issue(expected: str, path: list[Any], value: Any) -> dict[str, Any]:
    return {
        "expected": expected,
        "code": "invalid_type",
        "path": path,
        "message": f"Invalid input: expected {expected}, received {_type_name(value)}",
    }


def _raise_issues(issues: list[dict[str, Any]]) -> None:
    if issues:
        raise ValueError(json.dumps(issues, indent=2, ensure_ascii=False))


def _load_and_render(value: Any, vars: dict[str, Any]) -> Any:
    """Render authored values once. Loaded files are new authored input, not rendered output."""
    if _is_file_reference(value):
        reference = render_vars_in_object(value, vars)
        return render_vars_in_object(maybe_load_from_external_file(reference), vars)
    return render_vars_in_object(value, vars)


def _parse_response_format(value: Any) -> dict[str, Any]:
    if not _is_record(value):
        _raise_issues([_issue("object", [], value)])
    issues: list[dict[str, Any]] = []
    if not isinstance(value.get("type"), str):
        issues.append(_issue("string", ["type"], value.get("type")))
    if "name" in value and not isinstance(value["name"], str):
        issues.append(_issue("string", ["name"], value["name"]))
    if "schema" in value and not _is_record(value["schema"]):
        issues.append(_issue("record", ["schema"], value["schema"]))
    nested = value.get("json_schema")
    if "json_schema" in value:
        if not _is_record(nested):
            issues.append(_issue("object", ["json_schema"], nested))
        else:
            if "name" in nested and not isinstance(nested["name"], str):
                issues.append(_issue("string", ["json_schema", "name"], nested["name"]))
            if "schema" in nested and not _is_record(nested["schema"]):
                issues.append(_issue("record", ["json_schema", "schema"], nested["schema"]))
            strict = nested.get("strict")
            if strict is not None and not isinstance(strict, bool):
                issues.append(_issue("boolean", ["json_schema", "strict"], strict))
    _raise_issues(issues)
    return value


def prepare_openai_response_format(value: Any, vars: dict[str, Any] | None = None) -> Any:
    vars = vars or {}
    loaded = _load_and_render(value, vars)
    if loaded is None:
        return None
    if _is_record(loaded) and loaded.get("type") == "json_schema":
        nested = loaded["json_schema"] if _is_record(loaded.get("json_schema")) else None
        schema = loaded.get("schema")
        if schema is None:
            schema = (nested or {}).get("schema")
        # The format's single pass already rendered the path. Only render the newly loaded
        # document here; inline schema values must never be rendered twice.
        if _is_file_reference(schema):
            resolved = render_vars_in_object(maybe_load_from_external_file(schema), vars)
            if "schema" not in loaded:
                return _parse_response_format(
                    {**loaded, "json_schema": {**(nested or {}), "schema": resolved}}
                )
            return _parse_response_format({**loaded, "schema": resolved})
    return _parse_response_format(loaded)


async def _load_tool_definitions(value: Any, vars: dict[str, Any]) -> Any:
    if _is_file_reference(value):
        reference = render_vars_in_object(value, vars)
        # The loader must not render its output; this pass belongs to the fresh definitions.
        loaded = await maybe_load_tools_from_external_file(reference)
        return render_vars_in_object(loaded, vars)
    if isinstance(value, list):
        loaded_items = [await _load_tool_definitions(entry, vars) for entry in value]
        if all(isinstance(item, list) for item in loaded_items):
            return [x for item in loaded_items for x in item]
        return loaded_items
    return render_vars_in_object(value, vars)


async def prepare_openai_tools(value: Any, vars: dict[str, Any] | None = None) -> Any:
    if value is None:
        return None
    loaded = await _load_tool_definitions(value, vars or {})
    if loaded is None:
        return None
    if not isinstance(loaded, list):
        _raise_issues([_issue("array", [], loaded)])
    issues: list[dict[str, Any]] = []
    for i, tool in enumerate(loaded):
        if not _is_record(tool):
            issues.append(_issue("object", [i], tool))
        elif not isinstance(tool.get("type"), str):
            issues.append(_issue("string", [i, "type"], tool.get("type")))
    _raise_issues(issues)
    return loaded


def prepare_openai_reasoning(value: Any, vars: dict[str, Any] | None = None) -> Any:
    rendered = render_vars_in_object(value, vars or {})
    if rendered is None:
        return None
    if not _is_record(rendered):
        _raise_issues([_issue("record", [], rendered)])
    return rendered


def is_openai_json_schema_format(value: Any) -> bool:
    """Inspect the effective wire format, including a passthrough override."""
    return _is_record(value) and value.get("type") == "json_schema"
