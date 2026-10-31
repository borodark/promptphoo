"""Helpers shared by the network providers. Ports src/providers/shared.ts.

Tool and tool-choice converters take OpenAI's format as canonical and translate it for
Anthropic, Bedrock Converse and Google; anything not in OpenAI format passes through unchanged.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from promptphoo import envars

LONG_RUNNING_MODEL_TIMEOUT_MS = 600_000  # 10 minutes


def should_bust_provider_cache(context: dict[str, Any] | None = None) -> bool:
    """An explicit bustCache setting takes precedence over the legacy debug fallback."""
    context = context or {}
    if context.get("bustCache") is not None:
        return bool(context["bustCache"])
    if context.get("debug") is not None:
        return bool(context["debug"])
    return False


def with_response_cache_metadata(response: dict[str, Any], cached: bool) -> dict[str, Any]:
    """Mark cached responses without changing reported usage or cost."""
    if not cached or not response.get("tokenUsage"):
        return {**response, "cached": cached}
    usage = {k: v for k, v in response["tokenUsage"].items() if k != "cached"}
    reported_cached = response["tokenUsage"].get("cached")
    details = usage.get("completionDetails") or {}
    provider_cached = details.get("cacheReadInputTokens")
    if provider_cached is None:
        provider_cached = None if response.get("cached") is True else reported_cached
    token_usage: dict[str, Any] = {**usage}
    if provider_cached is not None:
        token_usage["completionDetails"] = {
            **(usage.get("completionDetails") or {}),
            "cacheReadInputTokens": provider_cached,
        }
    if usage.get("total") is not None:
        token_usage["cached"] = usage["total"]
    token_usage["numRequests"] = 0
    token_usage["incurredTokenUsage"] = {}
    return {**response, "cached": cached, "tokenUsage": token_usage}


def model_name_from_provider_path(provider_path: str, segments: int) -> str:
    """Returns the complete model suffix after the given number of provider/type segments."""
    return ":".join(provider_path.split(":")[segments:])


def get_request_timeout_ms() -> int:
    return envars.get_env_int("REQUEST_TIMEOUT_MS", 300_000)


def resolve_direct_test_variable(value: Any, vars: dict[str, Any] | None = None) -> Any:
    """Read a simple eval variable without evaluating template expressions."""
    if not isinstance(value, str) or envars.get_env_bool("PROMPTFOO_DISABLE_TEMPLATING"):
        return value
    m = re.match(r"^\{\{\s*([A-Za-z_]\w*)\s*\}\}$", value)
    variable = m.group(1) if m else None
    return vars[variable] if variable and vars and variable in vars else value


def get_openai_chat_output_limit_from_env() -> int | None:
    limit = get_openai_completion_token_limit_from_env()
    return limit if limit is not None else envars.get_env_int("OPENAI_MAX_TOKENS")


def get_openai_completion_token_limit_from_env() -> int | None:
    return envars.get_env_int("OPENAI_MAX_COMPLETION_TOKENS")


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and not (isinstance(value, float) and not math.isfinite(value))
    )


def calculate_cost(
    model_name: str,
    config: dict[str, Any],
    prompt_tokens: float | None,
    completion_tokens: float | None,
    models: list[dict[str, Any]],
) -> float | None:
    if not _is_finite_number(prompt_tokens) or not _is_finite_number(completion_tokens):
        return None
    assert prompt_tokens is not None and completion_tokens is not None
    model = next((m for m in models if m.get("id") == model_name), None)
    if not model or not model.get("cost"):
        return None
    cost = model["cost"]
    long_context = cost.get("longContext")
    if not (long_context and prompt_tokens > long_context["threshold"]):
        long_context = None

    def pick(*values: Any) -> Any:
        return next((v for v in values if v is not None), None)

    input_cost = pick(
        config.get("inputCost"),
        config.get("cost"),
        long_context and long_context.get("input"),
        cost["input"],
    )
    output_cost = pick(
        config.get("outputCost"),
        config.get("cost"),
        long_context and long_context.get("output"),
        cost["output"],
    )
    return input_cost * prompt_tokens + output_cost * completion_tokens


def clamp_cached_tokens(cached_tokens: float | None, prompt_tokens: float) -> float:
    """Clamp reported cached prompt tokens to [0, promptTokens] for cost billing."""
    if not _is_finite_number(cached_tokens):
        return 0
    assert cached_tokens is not None
    return min(max(cached_tokens, 0), prompt_tokens)


def _looks_like_json(prompt: str) -> bool:
    trimmed = prompt.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return True
    if trimmed.startswith("[") and trimmed.endswith("]"):
        after = trimmed[1:].lstrip()
        return (
            after.startswith(('"', "{", "["))
            or re.match(r"^[\d-]", after) is not None
            or re.match(r"^(true|false|null)", after) is not None
        )
    return False


def parse_chat_prompt(prompt: str, default_value: Any) -> Any:
    """Parse a chat prompt (JSON, or legacy YAML starting with `- role:`)."""
    trimmed = prompt.strip()
    if trimmed.startswith("- role:"):
        from promptphoo.yamlload import load_yaml

        try:
            return load_yaml(prompt)
        except Exception as err:
            raise ValueError(
                f"Chat Completion prompt is not a valid YAML string: {err}\n\n{prompt}"
            ) from err
    try:
        return json.loads(prompt)
    except ValueError as err:
        if envars.get_env_bool("PROMPTFOO_REQUIRE_JSON_PROMPTS") or _looks_like_json(trimmed):
            raise ValueError(
                f"Chat Completion prompt is not a valid JSON string: SyntaxError: {err}\n\n{prompt}"
            ) from err
        return default_value


def to_title_case(s: str) -> str:
    return re.sub(r"\w\S*", lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(), s)


def is_promptfoo_sample_target(provider: Any) -> bool:
    url = (getattr(provider, "config", None) or {}).get("url")
    return isinstance(url, str) and ("promptfoo.app" in url or "promptfoo.dev" in url)


# --- Tool choice (OpenAI format is canonical) -----------------------------------------------


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
        # Bedrock has no 'none'; returning None omits toolChoice.
        return {"auto": {"auto": {}}, "none": None, "required": {"any": {}}}[choice]
    return {"tool": {"name": choice["function"]["name"]}}


def openai_tool_choice_to_google(choice: Any) -> dict[str, Any]:
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
    """Translate an OpenAI tool choice to `format`; non-OpenAI values pass through."""
    if not is_openai_tool_choice(tool_choice):
        return tool_choice
    if format == "anthropic":
        return openai_tool_choice_to_anthropic(tool_choice)
    if format == "bedrock":
        return openai_tool_choice_to_bedrock(tool_choice)
    if format == "google":
        return openai_tool_choice_to_google(tool_choice)
    return tool_choice


# --- Tool definitions -------------------------------------------------------------------------


def is_openai_tool_array(tools: Any) -> bool:
    if not isinstance(tools, list) or not tools:
        return False
    first = tools[0]
    if not isinstance(first, dict):
        return False
    fn = first.get("function")
    return (
        first.get("type") == "function" and isinstance(fn, dict) and isinstance(fn.get("name"), str)
    )


def _default_schema(fn: dict[str, Any]) -> Any:
    params = fn.get("parameters")
    return params if params else {"type": "object", "properties": {}}


def openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool["function"]["name"],
            **(
                {"description": tool["function"]["description"]}
                if tool["function"].get("description")
                else {}
            ),
            "input_schema": _default_schema(tool["function"]),
        }
        for tool in tools
    ]


def openai_tools_to_bedrock(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "toolSpec": {
                "name": tool["function"]["name"],
                **(
                    {"description": tool["function"]["description"]}
                    if tool["function"].get("description")
                    else {}
                ),
                "inputSchema": {"json": _default_schema(tool["function"])},
            }
        }
        for tool in tools
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
            "name": tool["function"]["name"],
            **(
                {"description": tool["function"]["description"]}
                if tool["function"].get("description")
                else {}
            ),
            **(
                {"parameters": _sanitize_schema_for_google(tool["function"]["parameters"])}
                if tool["function"].get("parameters")
                else {}
            ),
        }
        for tool in tools
    ]
    return [{"functionDeclarations": declarations}]


def transform_tools(tools: Any, format: str) -> Any:
    """Translate OpenAI-format tools to `format`; anything else passes through."""
    if not is_openai_tool_array(tools):
        return tools
    if format == "anthropic":
        return openai_tools_to_anthropic(tools)
    if format == "bedrock":
        return openai_tools_to_bedrock(tools)
    if format == "google":
        return openai_tools_to_google(tools)
    return tools
