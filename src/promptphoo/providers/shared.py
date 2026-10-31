"""Helpers shared by providers. Ports parts of src/providers/shared.ts."""

from __future__ import annotations

from typing import Any

from promptphoo import envars


def get_request_timeout_ms() -> int:
    """The default timeout for API requests in milliseconds."""
    return envars.get_env_int("REQUEST_TIMEOUT_MS", 300_000)


# --- tool_choice conversion ---------------------------------------------------------------------


def is_openai_tool_choice(obj: Any) -> bool:
    if isinstance(obj, str):
        return obj in ("auto", "none", "required")
    if isinstance(obj, dict):
        fn = obj.get("function")
        return (
            obj.get("type") == "function"
            and isinstance(fn, dict)
            and isinstance(fn.get("name"), str)
        )
    return False


def openai_tool_choice_to_anthropic(choice: Any) -> dict[str, Any]:
    if isinstance(choice, str):
        return {"auto": {"type": "auto"}, "none": {"type": "none"}, "required": {"type": "any"}}[
            choice
        ]
    return {"type": "tool", "name": choice["function"]["name"]}


def openai_tool_choice_to_bedrock(choice: Any) -> dict[str, Any] | None:
    if isinstance(choice, str):
        return {"auto": {"auto": {}}, "none": None, "required": {"any": {}}}[choice]
    return {"tool": {"name": choice["function"]["name"]}}


def openai_tool_choice_to_google(choice: Any) -> dict[str, Any] | None:
    if isinstance(choice, str):
        mode = {"auto": "AUTO", "none": "NONE", "required": "ANY"}[choice]
        return {"functionCallingConfig": {"mode": mode}}
    return {
        "functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": [choice["function"]["name"]],
        }
    }


def transform_tool_choice(tool_choice: Any, format: str) -> Any:
    if not is_openai_tool_choice(tool_choice):
        return tool_choice
    if format == "anthropic":
        return openai_tool_choice_to_anthropic(tool_choice)
    if format == "bedrock":
        return openai_tool_choice_to_bedrock(tool_choice)
    if format == "google":
        return openai_tool_choice_to_google(tool_choice)
    return tool_choice


# --- tools conversion ---------------------------------------------------------------------------


def is_openai_tool_array(tools: Any) -> bool:
    if not isinstance(tools, list) or len(tools) == 0:
        return False
    first = tools[0]
    if not isinstance(first, dict):
        return False
    fn = first.get("function")
    return (
        first.get("type") == "function" and isinstance(fn, dict) and isinstance(fn.get("name"), str)
    )


def _description(fn: dict[str, Any]) -> dict[str, Any]:
    return {"description": fn["description"]} if fn.get("description") else {}


def openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": t["function"]["name"],
            **_description(t["function"]),
            "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}},
        }
        for t in tools
    ]


def openai_tools_to_bedrock(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "toolSpec": {
                "name": t["function"]["name"],
                **_description(t["function"]),
                "inputSchema": {
                    "json": t["function"].get("parameters") or {"type": "object", "properties": {}}
                },
            }
        }
        for t in tools
    ]


def _sanitize_schema_for_google(schema: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("additionalProperties", "$schema", "default", "$id", "$ref"):
            continue
        if key == "type" and isinstance(value, str):
            result[key] = value.upper()
        elif key == "properties" and isinstance(value, dict):
            result[key] = {
                k: _sanitize_schema_for_google(v) if isinstance(v, dict) else v
                for k, v in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            result[key] = _sanitize_schema_for_google(value)
        else:
            result[key] = value
    return result


def openai_tools_to_google(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    declarations = [
        {
            "name": t["function"]["name"],
            **_description(t["function"]),
            **(
                {"parameters": _sanitize_schema_for_google(t["function"]["parameters"])}
                if t["function"].get("parameters")
                else {}
            ),
        }
        for t in tools
    ]
    return [{"functionDeclarations": declarations}]


def transform_tools(tools: Any, format: str) -> Any:
    if not is_openai_tool_array(tools):
        return tools
    if format == "anthropic":
        return openai_tools_to_anthropic(tools)
    if format == "bedrock":
        return openai_tools_to_bedrock(tools)
    if format == "google":
        return openai_tools_to_google(tools)
    return tools
