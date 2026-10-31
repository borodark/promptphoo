"""Helpers shared by the network providers. Ports src/providers/shared.ts.

Abort signals have no Python equivalent (asyncio cancellation covers them), so
`throw_if_aborted` and `wait_for_promise_with_abort` are thin shims kept for name parity.
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Literal, TypeVar

from promptphoo import envars
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import is_truthy
from promptphoo.yamlload import load_yaml

T = TypeVar("T")


def should_bust_provider_cache(context: dict[str, Any] | None = None) -> bool:
    """An explicit bustCache setting takes precedence over the legacy debug fallback."""
    context = context or {}
    if context.get("bustCache") is not None:
        return context["bustCache"]
    if context.get("debug") is not None:
        return context["debug"]
    return False


def with_response_cache_metadata(response: dict[str, Any], cached: bool) -> dict[str, Any]:
    """Mark cached responses without changing reported usage or cost. The evaluator uses the
    cache marker to calculate incurred usage; missing counts stay unknown."""
    if not cached or not is_truthy(response.get("tokenUsage")):
        return {**response, "cached": cached}
    usage = {k: v for k, v in response["tokenUsage"].items() if k != "cached"}
    reported_cached = response["tokenUsage"].get("cached")
    completion_details = usage.get("completionDetails")
    provider_cached = (completion_details or {}).get("cacheReadInputTokens")
    if provider_cached is None:
        provider_cached = None if response.get("cached") is True else reported_cached
    token_usage = dict(usage)
    if provider_cached is not None:
        token_usage["completionDetails"] = {
            **(completion_details or {}),
            "cacheReadInputTokens": provider_cached,
        }
    if usage.get("total") is not None:
        token_usage["cached"] = usage["total"]
    token_usage["numRequests"] = 0
    token_usage["incurredTokenUsage"] = {}
    # Node spreads the whole response first, so existing keys keep their positions.
    return {**response, "cached": cached, "tokenUsage": token_usage}


def throw_if_aborted(signal: Any = None) -> None:
    """Abort signals map to asyncio cancellation; a signal-like object with `aborted` is
    still honoured so callers can share one code path."""
    if signal is not None and getattr(signal, "aborted", False):
        raise getattr(signal, "reason", None) or RuntimeError("This operation was aborted")


async def wait_for_promise_with_abort(promise: Awaitable[T], signal: Any = None) -> T:
    throw_if_aborted(signal)
    return await promise


def model_name_from_provider_path(provider_path: str, segments: int) -> str:
    """Returns the complete model suffix after the given number of provider/type segments."""
    return ":".join(provider_path.split(":")[segments:])


def get_request_timeout_ms() -> int:
    """The default timeout for API requests in milliseconds."""
    return envars.get_env_int("REQUEST_TIMEOUT_MS", 300_000)


_DIRECT_VARIABLE = re.compile(r"^\{\{\s*([A-Za-z_]\w*)\s*\}\}$")


def resolve_direct_test_variable(value: Any, vars: dict[str, Any] | None = None) -> Any:
    """Read a simple eval variable without evaluating template expressions."""
    if not isinstance(value, str) or envars.get_env_bool("PROMPTFOO_DISABLE_TEMPLATING"):
        return value
    match = _DIRECT_VARIABLE.match(value)
    variable = match.group(1) if match else None
    return vars[variable] if variable and vars and variable in vars else value


def get_openai_chat_output_limit_from_env() -> int | None:
    """Match OpenAI-compatible output-limit environment precedence."""
    limit = get_openai_completion_token_limit_from_env()
    return limit if limit is not None else envars.get_env_int("OPENAI_MAX_TOKENS")


def get_openai_completion_token_limit_from_env() -> int | None:
    return envars.get_env_int("OPENAI_MAX_COMPLETION_TOKENS")


# Extended timeout for long-running models (deep research, gpt-5-pro, etc.) in milliseconds.
LONG_RUNNING_MODEL_TIMEOUT_MS = 600_000


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _first(*values: Any) -> Any:
    """JS `a ?? b ?? c`."""
    for value in values:
        if value is not None:
            return value
    return None


def calculate_cost(
    model_name: str,
    config: Mapping[str, Any],
    prompt_tokens: Any,
    completion_tokens: Any,
    models: Sequence[Mapping[str, Any]],
) -> float | None:
    """Calculates the cost of an API call from the model table and token usage."""
    if not _is_finite_number(prompt_tokens) or not _is_finite_number(completion_tokens):
        return None
    model = next((m for m in models if m.get("id") == model_name), None)
    if not model or not model.get("cost"):
        return None
    cost = model["cost"]
    long_context = cost.get("longContext")
    long_context_cost = (
        long_context if long_context and prompt_tokens > long_context["threshold"] else None
    )
    input_cost = _first(
        config.get("inputCost"),
        config.get("cost"),
        (long_context_cost or {}).get("input"),
        cost.get("input"),
    )
    output_cost = _first(
        config.get("outputCost"),
        config.get("cost"),
        (long_context_cost or {}).get("output"),
        cost.get("output"),
    )
    return input_cost * prompt_tokens + output_cost * completion_tokens


def clamp_cached_tokens(cached_tokens: Any, prompt_tokens: float) -> float:
    """Clamp reported cached prompt tokens to [0, promptTokens] for cost billing."""
    return min(max(cached_tokens, 0), prompt_tokens) if _is_finite_number(cached_tokens) else 0


def _looks_like_json(prompt: str) -> bool:
    """Checks if a string looks like it's attempting to be JSON."""
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


def _js_error_text(err: Exception) -> str:
    """`${err}` for a thrown Error: 'SyntaxError: message'."""
    kind = "SyntaxError" if isinstance(err, ValueError) else "Error"
    return f"{kind}: {err}"


def parse_chat_prompt(prompt: str, default_value: T) -> T | Any:
    """Parses a chat prompt (JSON, or legacy YAML starting with `- role:`) into messages."""
    trimmed = prompt.strip()
    if trimmed.startswith("- role:"):
        try:
            return load_yaml(prompt)
        except Exception as err:
            raise ValueError(
                f"Chat Completion prompt is not a valid YAML string: YAMLException: {err}"
                f"\n\n{prompt}"
            ) from err
    try:
        return js_json_parse(prompt)
    except ValueError as err:
        if envars.get_env_bool("PROMPTFOO_REQUIRE_JSON_PROMPTS") or _looks_like_json(trimmed):
            raise ValueError(
                f"Chat Completion prompt is not a valid JSON string: {_js_error_text(err)}"
                f"\n\n{prompt}"
            ) from err
        return default_value


def to_title_case(text: str) -> str:
    return re.sub(r"\w\S*", lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(), text)


def is_promptfoo_sample_target(provider: Any) -> bool:
    url = (getattr(provider, "config", None) or {}).get("url")
    return isinstance(url, str) and ("promptfoo.app" in url or "promptfoo.dev" in url)


# ==================
# OpenAI Tool Choice (Canonical Format)
# ==================

ToolChoiceFormat = Literal["openai", "anthropic", "bedrock", "google"]
ToolFormat = ToolChoiceFormat


def is_openai_tool_choice(obj: Any) -> bool:
    """'auto' | 'none' | 'required' | {type: 'function', function: {name}}."""
    if isinstance(obj, str):
        return obj in ("auto", "none", "required")
    if isinstance(obj, dict) and obj.get("type") == "function":
        fn = obj.get("function")
        return isinstance(fn, dict) and isinstance(fn.get("name"), str)
    return False


def openai_tool_choice_to_anthropic(choice: Any) -> dict[str, Any]:
    if isinstance(choice, str):
        return {"auto": {"type": "auto"}, "none": {"type": "none"}, "required": {"type": "any"}}[
            choice
        ]
    return {"type": "tool", "name": choice["function"]["name"]}


def openai_tool_choice_to_bedrock(choice: Any) -> dict[str, Any] | None:
    if isinstance(choice, str):
        # Bedrock doesn't have 'none'; None omits toolChoice.
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


def transform_tool_choice(tool_choice: Any, format: ToolChoiceFormat) -> Any:
    """Transforms an OpenAI tool choice to `format`; non-OpenAI values pass through."""
    if not is_openai_tool_choice(tool_choice):
        return tool_choice
    if format == "anthropic":
        return openai_tool_choice_to_anthropic(tool_choice)
    if format == "bedrock":
        return openai_tool_choice_to_bedrock(tool_choice)
    if format == "google":
        return openai_tool_choice_to_google(tool_choice)
    return tool_choice


# ==================
# Tool Format Transformation
# ==================


def is_openai_tool_array(tools: Any) -> bool:
    """True if the first tool has `type: 'function'` and `function.name`."""
    if not isinstance(tools, list) or not tools:
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


_EMPTY_SCHEMA = {"type": "object", "properties": {}}


def openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool["function"]["name"],
            **_description(tool["function"]),
            "input_schema": tool["function"].get("parameters") or dict(_EMPTY_SCHEMA),
        }
        for tool in tools
    ]


def openai_tools_to_bedrock(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "toolSpec": {
                "name": tool["function"]["name"],
                **_description(tool["function"]),
                "inputSchema": {"json": tool["function"].get("parameters") or dict(_EMPTY_SCHEMA)},
            }
        }
        for tool in tools
    ]


def sanitize_schema_for_google(schema: dict[str, Any]) -> dict[str, Any]:
    """Uppercase types, drop unsupported keywords, recurse into properties and items."""
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("additionalProperties", "$schema", "default", "$id", "$ref"):
            continue
        if key == "type" and isinstance(value, str):
            result[key] = value.upper()
        elif key == "properties" and isinstance(value, dict):
            result[key] = {
                k: sanitize_schema_for_google(v) if isinstance(v, dict) else v
                for k, v in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            result[key] = sanitize_schema_for_google(value)
        else:
            result[key] = value
    return result


def openai_tools_to_google(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    declarations = [
        {
            "name": tool["function"]["name"],
            **_description(tool["function"]),
            **(
                {"parameters": sanitize_schema_for_google(tool["function"]["parameters"])}
                if tool["function"].get("parameters")
                else {}
            ),
        }
        for tool in tools
    ]
    return [{"functionDeclarations": declarations}]


def transform_tools(tools: Any, format: ToolFormat) -> Any:
    """Transforms OpenAI-format tools to `format`; anything else passes through."""
    if not is_openai_tool_array(tools):
        return tools
    if format == "anthropic":
        return openai_tools_to_anthropic(tools)
    if format == "bedrock":
        return openai_tools_to_bedrock(tools)
    if format == "google":
        return openai_tools_to_google(tools)
    return tools


# Helpers other providers import from here. finish_reason and credentials own them
# (ports of util/finishReason.ts and providers/env.ts); they are re-exported for those callers.
from promptphoo.finish_reason import FINISH_REASON_MAP as FINISH_REASON_MAP  # noqa: E402
from promptphoo.finish_reason import normalize_finish_reason as normalize_finish_reason  # noqa: E402
from promptphoo.providers.credentials import (  # noqa: E402
    resolve_provider_env as resolve_provider_env,
)


def js_error_string(err: BaseException) -> str:
    """String(err) for a thrown Error: `Error: message`."""
    message = str(err)
    name = getattr(err, "js_name", None) or "Error"
    return f"{name}: {message}" if message else name
