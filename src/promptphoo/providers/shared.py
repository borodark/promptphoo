"""Helpers shared by the network providers. Ports src/providers/shared.ts, plus
resolveProviderEnv (src/providers/env.ts) and normalizeFinishReason (src/util/finishReason.ts).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from promptphoo import envars
from promptphoo.jsvalues import UNDEFINED

# --- shared.ts: cache / timeout helpers --------------------------------------------------------


def should_bust_provider_cache(context: Mapping[str, Any] | None = None) -> bool:
    """An explicit bustCache setting takes precedence over the legacy debug fallback."""
    context = context or {}
    for key in ("bustCache", "debug"):
        value = context.get(key)
        if value is not None:
            return bool(value)
    return False


def with_response_cache_metadata(response: dict[str, Any], cached: bool) -> dict[str, Any]:
    """Mark cached responses without changing reported usage or cost."""
    token_usage = response.get("tokenUsage")
    if not cached or token_usage is None:
        return {**response, "cached": cached}
    usage = {k: v for k, v in token_usage.items() if k != "cached"}
    reported_cached = token_usage.get("cached")
    completion_details = usage.get("completionDetails") or {}
    provider_cached = completion_details.get("cacheReadInputTokens")
    if provider_cached is None:
        provider_cached = None if response.get("cached") is True else reported_cached
    new_usage: dict[str, Any] = {**usage}
    if provider_cached is not None:
        new_usage["completionDetails"] = {
            **(usage.get("completionDetails") or {}),
            "cacheReadInputTokens": provider_cached,
        }
    if usage.get("total") is not None:
        new_usage["cached"] = usage["total"]
    new_usage["numRequests"] = 0
    new_usage["incurredTokenUsage"] = {}
    return {**response, "cached": cached, "tokenUsage": new_usage}


def model_name_from_provider_path(provider_path: str, segments: int) -> str:
    """Returns the complete model suffix after the given number of provider/type segments."""
    return ":".join(provider_path.split(":")[segments:])


def get_request_timeout_ms() -> int:
    return envars.get_env_int("REQUEST_TIMEOUT_MS", 300_000)


LONG_RUNNING_MODEL_TIMEOUT_MS = 600_000


def js_error_string(err: BaseException) -> str:
    """String(err) for a thrown Error: `Error: message`."""
    message = str(err)
    name = getattr(err, "js_name", None) or "Error"
    return f"{name}: {message}" if message else name


# --- shared.ts: cost ---------------------------------------------------------------------------


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _first_defined(*values: Any) -> Any:
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
    input_cost = _first_defined(
        config.get("inputCost"),
        config.get("cost"),
        long_context_cost and long_context_cost.get("input"),
        cost["input"],
    )
    output_cost = _first_defined(
        config.get("outputCost"),
        config.get("cost"),
        long_context_cost and long_context_cost.get("output"),
        cost["output"],
    )
    return input_cost * prompt_tokens + output_cost * completion_tokens


def clamp_cached_tokens(cached_tokens: Any, prompt_tokens: float) -> float:
    """Clamp reported cached prompt tokens to [0, promptTokens] for cost billing."""
    return min(max(cached_tokens, 0), prompt_tokens) if _is_finite_number(cached_tokens) else 0


# --- shared.ts: parseChatPrompt ----------------------------------------------------------------


def _looks_like_json(prompt: str) -> bool:
    trimmed = prompt.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return True
    if trimmed.startswith("[") and trimmed.endswith("]"):
        after = trimmed[1:].lstrip()
        return bool(
            after.startswith(('"', "{", "["))
            or re.match(r"^[\d-]", after)
            or re.match(r"^(true|false|null)", after)
        )
    return False


def parse_chat_prompt(prompt: str, default_value: Any) -> Any:
    from promptphoo.jsonutil import js_json_parse
    from promptphoo.yamlload import load_yaml

    trimmed = prompt.strip()
    if trimmed.startswith("- role:"):
        try:
            return load_yaml(prompt)
        except Exception as err:
            raise ValueError(
                f"Chat Completion prompt is not a valid YAML string: {err}\n\n{prompt}"
            ) from err
    try:
        return js_json_parse(prompt)
    except ValueError as err:
        if envars.get_env_bool("PROMPTFOO_REQUIRE_JSON_PROMPTS") or _looks_like_json(trimmed):
            raise ValueError(
                f"Chat Completion prompt is not a valid JSON string: SyntaxError: {err}\n\n{prompt}"
            ) from err
        return default_value


def to_title_case(text: str) -> str:
    return re.sub(r"\w\S*", lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(), text)


# --- shared.ts: tool choice --------------------------------------------------------------------


def is_openai_tool_choice(obj: Any) -> bool:
    if isinstance(obj, str):
        return obj in ("auto", "none", "required")
    if isinstance(obj, dict):
        fn = obj.get("function")
        if obj.get("type") == "function" and isinstance(fn, dict):
            return isinstance(fn.get("name"), str)
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
    """Transforms an OpenAI tool choice to a provider format; other values pass through."""
    if not is_openai_tool_choice(tool_choice):
        return tool_choice
    if format == "anthropic":
        return openai_tool_choice_to_anthropic(tool_choice)
    if format == "bedrock":
        return openai_tool_choice_to_bedrock(tool_choice)
    if format == "google":
        return openai_tool_choice_to_google(tool_choice)
    return tool_choice


# --- shared.ts: tool format --------------------------------------------------------------------


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


def _description(fn: Mapping[str, Any]) -> dict[str, Any]:
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


def sanitize_schema_for_google(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Uppercase types and drop properties Gemini rejects, recursively."""
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
    declarations = []
    for t in tools:
        fn = t["function"]
        declarations.append(
            {
                "name": fn["name"],
                **_description(fn),
                **(
                    {"parameters": sanitize_schema_for_google(fn["parameters"])}
                    if fn.get("parameters")
                    else {}
                ),
            }
        )
    return [{"functionDeclarations": declarations}]


def transform_tools(tools: Any, format: str) -> Any:
    """Transforms OpenAI-format tools to a provider format; other values pass through."""
    if not is_openai_tool_array(tools):
        return tools
    if format == "anthropic":
        return openai_tools_to_anthropic(tools)
    if format == "bedrock":
        return openai_tools_to_bedrock(tools)
    if format == "google":
        return openai_tools_to_google(tools)
    return tools


# --- providers/env.ts: resolveProviderEnv ------------------------------------------------------


def resolve_provider_env(
    env: Mapping[str, Any] | None,
    names: Sequence[str],
    ambient_names: Sequence[str] | None = None,
) -> dict[str, str] | None:
    """Resolve aliases within each scope; an empty value masks only that variable.

    Scopes, highest first: the provider's env, the registered provider layer, the suite env,
    the --env-file layer, then the process environment.
    """
    ambient_names = names if ambient_names is None else ambient_names
    masked: set[str] = set()
    ambient = {name: envars.get_env_string(name) for name in names}
    layers: list[Mapping[str, Any] | None] = [
        env,
        envars.get_env_overrides("provider"),
        envars.get_env_overrides("suite"),
        envars.get_env_overrides("file"),
        ambient,
    ]
    for layer in layers:
        for name in ambient_names if layer is ambient else names:
            value = layer.get(name) if layer else None
            if name in masked or value is None or value is UNDEFINED:
                continue
            masked.add(name)
            if value:
                return {"name": name, "value": str(value)}
    return None


# --- util/finishReason.ts ----------------------------------------------------------------------

FINISH_REASON_MAP: dict[str, str] = {
    "stop": "stop",
    "length": "length",
    "content_filter": "content_filter",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "max_output_tokens": "length",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
    "pause_turn": "pause_turn",
    "tool-calls": "tool_calls",
    "content-filter": "content_filter",
}


def normalize_finish_reason(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    trimmed = raw.strip()
    if not trimmed:
        return None
    key = trimmed.lower()
    return FINISH_REASON_MAP.get(key, key)


# Ported in the Anthropic/Bedrock branch.


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


def is_promptfoo_sample_target(provider: Any) -> bool:
    url = (getattr(provider, "config", None) or {}).get("url")
    return isinstance(url, str) and ("promptfoo.app" in url or "promptfoo.dev" in url)
