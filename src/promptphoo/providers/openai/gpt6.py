"""GPT-6 request rules shared by Chat Completions and Responses.
Ports src/providers/openai/gpt6.ts.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

REASONING_EFFORTS = frozenset(["low", "medium", "high", "xhigh", "max"])
# JS symbols: sentinels that never equal a configured value.
UNKNOWN_PERSISTED_EFFORT = object()
COMPACTED_EFFORT = object()

Gpt6Variant = Literal["astra", "sol", "luna", "6.1-sol"]
Render = Callable[[Any], Any]

_MISSING = object()


def _get_gpt6_base_model_name(model_name: str) -> str:
    model = model_name[model_name.rfind("/") + 1 :]
    parts = model.split(":")
    name = parts[0]
    fine_tune_base = parts[1] if len(parts) > 1 else None
    return (fine_tune_base or "") if name == "ft" else name


def get_gpt6_variant(model_name: Any) -> Gpt6Variant | None:
    if not isinstance(model_name, str):
        return None
    base = _get_gpt6_base_model_name(model_name)
    if re.search(r"(?:^|[.-])gpt-6\.1-sol(?:[-:]|$)", base):
        return "6.1-sol"
    m = re.search(r"(?:^|[.-])gpt-6-(astra|sol|luna)(?:[-:]|$)", base)
    return m.group(1) if m else None  # type: ignore[return-value]


def is_gpt6_model(model_name: Any) -> bool:
    if not isinstance(model_name, str):
        return False
    base = _get_gpt6_base_model_name(model_name)
    return get_gpt6_variant(base) is not None or bool(re.search(r"(?:^|[/-])gpt-6(?:[.-]|$)", base))


def _get_object(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _get(obj: dict[str, Any] | None, key: str) -> Any:
    """`obj?.[key]` distinguishing undefined (`_MISSING`) from null (None)."""
    if obj is None or key not in obj:
        return _MISSING
    return obj[key]


def _first_defined(*values: Any) -> Any:
    return next((v for v in values if v is not _MISSING), _MISSING)


def _get_gpt6_chat_output_cap(config: dict[str, Any] | None, is_openrouter: bool) -> Any:
    passthrough = _get_object((config or {}).get("passthrough"))
    candidates = [
        _get(passthrough, "max_completion_tokens"),
        *(
            [_get(passthrough, "max_tokens"), _get(config, "max_completion_tokens")]
            if is_openrouter
            else [_get(config, "max_completion_tokens"), _get(passthrough, "max_tokens")]
        ),
        _get(config, "max_tokens"),
    ]
    return _first_defined(*candidates)


def resolve_gpt6_chat_output_cap(
    provider_config: dict[str, Any],
    prompt_config: dict[str, Any] | None,
    is_openrouter: bool,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
) -> Any:
    cap = _first_defined(
        _get_gpt6_chat_output_cap(prompt_config, is_openrouter),
        _get_gpt6_chat_output_cap(provider_config, is_openrouter),
        _MISSING if max_completion_tokens is None else max_completion_tokens,
        _MISSING if max_tokens is None else max_tokens,
    )
    # `cap ?? undefined`: an explicit null clears the cap.
    return None if cap is _MISSING else cap


def get_gpt6_chat_reasoning_effort(
    provider_config: dict[str, Any], prompt_config: dict[str, Any] | None, render: Render
) -> Any:
    candidates = [
        _get(prompt_config, "reasoning_effort"),
        _get(_get_object((prompt_config or {}).get("passthrough")), "reasoning_effort"),
        _get(provider_config, "reasoning_effort"),
        _get(_get_object(provider_config.get("passthrough")), "reasoning_effort"),
    ]
    for candidate in candidates:
        if candidate is not _MISSING:
            effort = render(candidate)
            if effort is not _MISSING:
                return effort
    return _MISSING


def _render_responses_reasoning(value: Any, render: Render, resolve_effort: bool) -> Any:
    obj = _get_object(value)
    if not resolve_effort and obj is not None:
        return render({k: v for k, v in obj.items() if k != "effort"})
    result = render(value)
    rendered = _get_object(result)
    if not resolve_effort and rendered is not None:
        return {k: v for k, v in rendered.items() if k != "effort"}
    return result


def _get_responses_reasoning(
    config: dict[str, Any] | None, render: Render, resolve_effort: bool = True
) -> dict[str, Any]:
    typed = _render_responses_reasoning(_get(config, "reasoning"), render, resolve_effort)
    typed_effort = None if typed is None else _get(_get_object(typed), "effort")
    passthrough = _render_responses_reasoning(
        _get(_get_object((config or {}).get("passthrough")), "reasoning"),
        render,
        resolve_effort and typed_effort is _MISSING,
    )
    for value in (typed, passthrough):
        if value is not None and value is not _MISSING and _get_object(value) is None:
            raise ValueError(
                "GPT-6 Responses reasoning must be an object or null. Use { effort: low }."
            )
    options = {**(_get_object(passthrough) or {}), **(_get_object(typed) or {})}
    nested_effort = _first_defined(
        typed_effort, None if passthrough is None else _get(_get_object(passthrough), "effort")
    )
    effort = (
        render(_get(config, "reasoning_effort"))
        if nested_effort is _MISSING and resolve_effort
        else nested_effort
    )
    return {
        "options": options,
        "effort": effort,
        "clearOptions": typed is None or passthrough is None,
    }


def get_gpt6_responses_reasoning(
    provider_config: dict[str, Any], prompt_config: dict[str, Any] | None, render: Render
) -> dict[str, Any] | None:
    """`render` must map `_MISSING` (JS undefined) to itself; use `make_render`."""
    prompt = _get_responses_reasoning(prompt_config, render)
    provider = (
        None
        if prompt["clearOptions"]
        else _get_responses_reasoning(provider_config, render, prompt["effort"] is _MISSING)
    )
    options = {**((provider or {}).get("options") or {}), **prompt["options"]}
    effort = _first_defined(prompt["effort"], (provider or {}).get("effort", _MISSING))
    if effort is None:
        options.pop("effort", None)
    elif effort is not _MISSING:
        options["effort"] = effort
    return options or None


def make_render(render: Callable[[Any], Any]) -> Render:
    """Wrap a value renderer so JS `undefined` passes through untouched."""
    return lambda value: value if value is _MISSING else render(value)


def is_missing(value: Any) -> bool:
    return value is _MISSING


def _normalize_reasoning(body: dict[str, Any]) -> Any:
    if body.get("reasoning_effort") is None or body.get("reasoning_effort") == "":
        body.pop("reasoning_effort", None)
    reasoning = body.get("reasoning")
    effort = _get(_get_object(reasoning), "effort")
    if effort is not None and effort != "":
        return reasoning
    normalized = dict(reasoning or {})
    normalized.pop("effort", None)
    if normalized:
        body["reasoning"] = normalized
    else:
        body.pop("reasoning", None)
    return body.get("reasoning")


def _get_responses_effort_updates(input: Any) -> list[Any]:
    if not isinstance(input, list):
        return []
    efforts: list[Any] = []
    for item in input:
        item_obj = item if isinstance(item, dict) else {}
        if isinstance(item_obj.get("id"), str) and (
            item_obj.get("type") == "item_reference"
            or (item_obj.get("type") is None and all(k in ("id", "type") for k in item_obj))
        ):
            efforts.append(UNKNOWN_PERSISTED_EFFORT)
            continue
        if item_obj.get("type") == "compaction":
            efforts.append(COMPACTED_EFFORT)
            continue
        if item_obj.get("type") == "configuration_update":
            effort = (_get_object(item_obj.get("reasoning")) or {}).get("effort")
            if effort is not None and effort != "":
                efforts.append(effort)
    return efforts


def _get_openrouter_chat_effort_updates(messages: Any) -> list[Any]:
    if not isinstance(messages, list):
        return []
    efforts: list[Any] = []
    for message in messages:
        message = message if isinstance(message, dict) else {}
        if message.get("role") in ("system", "developer") and message.get("content") == "":
            update = _get_object(message.get("configuration_update")) or {}
            effort = (_get_object(update.get("reasoning")) or {}).get("effort")
            if effort is not None and effort != "":
                efforts.append(effort)
    return efforts


def _validate_reasoning_effort(effort: Any, variant: Gpt6Variant, model_label: str) -> None:
    requires_reasoning = variant in ("astra", "6.1-sol")
    if (
        effort is None
        or effort is _MISSING
        or effort == ""
        or (
            isinstance(effort, str)
            and (effort in REASONING_EFFORTS or (not requires_reasoning and effort == "none"))
        )
    ):
        return
    raise ValueError(
        f"{model_label} supports reasoning effort low, medium, high, xhigh, or max. Use low "
        "instead of none or minimal."
        if requires_reasoning
        else f"{model_label} supports reasoning effort none, low, medium, high, xhigh, or max."
    )


def _validate_chat_tools(
    body: dict[str, Any],
    variant: Gpt6Variant,
    model_label: str,
    effort: Any,
    allow_chat_tools: bool,
) -> None:
    if variant != "astra":
        for key in ("tools", "functions"):
            if isinstance(body.get(key), list) and not body[key]:
                del body[key]
    if allow_chat_tools or not any(
        body.get(k) is not None for k in ("tools", "tool_choice", "functions", "function_call")
    ):
        return
    if variant in ("astra", "6.1-sol"):
        model = "gpt-6.1-sol" if variant == "6.1-sol" else "gpt-6-astra"
        raise ValueError(
            f"{model_label} tool calling requires the Responses API. Use "
            f"openai:responses:{model} or azure:responses:<deployment>."
        )
    for selector, definitions in (("tool_choice", "tools"), ("function_call", "functions")):
        if body.get(selector) is not None and body.get(definitions) is None:
            raise ValueError(
                f"{model_label} Chat Completions {selector} requires a non-empty {definitions} "
                f"list. Omit {selector} when no {definitions} are configured."
            )
    if effort != "none":
        raise ValueError(
            f"{model_label} Chat Completions function calling requires reasoning_effort: none. "
            f"Set reasoning_effort to none or use openai:responses:gpt-6-{variant}."
        )


def _get_sampling_effort(
    body: dict[str, Any],
    api: Literal["chat", "responses"],
    effort: Any,
    variant: Gpt6Variant,
    model_label: str,
    is_openrouter: bool,
    supports_persisted_effort_updates: bool,
) -> Any:
    if (api == "chat" and not is_openrouter) or (
        api == "responses" and not supports_persisted_effort_updates
    ):
        return effort
    updates = (
        _get_openrouter_chat_effort_updates(body.get("messages"))
        if api == "chat"
        else _get_responses_effort_updates(body.get("input"))
    )
    for update in updates:
        if update is not UNKNOWN_PERSISTED_EFFORT and update is not COMPACTED_EFFORT:
            _validate_reasoning_effort(update, variant, model_label)
    if updates:
        latest = updates[-1]
        return effort if latest is COMPACTED_EFFORT else latest
    # A linked response or stored conversation can carry an effort we cannot see locally.
    if api == "responses" and (body.get("previous_response_id") or body.get("conversation")):
        return UNKNOWN_PERSISTED_EFFORT
    return effort


def apply_gpt6_request_rules(
    body: dict[str, Any],
    model_name: Any,
    api: Literal["chat", "responses"],
    is_openrouter: bool = False,
    default_responses_temperature: float | None = None,
    supports_persisted_effort_updates: bool = True,
) -> None:
    """Apply GPT-6 request restrictions after config and passthrough have been merged."""
    variant = get_gpt6_variant(model_name)
    if not variant:
        return
    model_label = "GPT-6.1 Sol" if variant == "6.1-sol" else f"GPT-6 {variant.capitalize()}"

    reasoning = _get_object(_normalize_reasoning(body))
    if api == "responses" and "reasoning_effort" in body:
        raise ValueError(
            f"{model_label} Responses requests use reasoning.effort. Configure reasoning or "
            "reasoning_effort instead of passthrough.reasoning_effort."
        )
    if (
        api == "chat"
        and variant != "astra"
        and not is_openrouter
        and body.get("reasoning") is not None
    ):
        raise ValueError(
            f"{model_label} Chat Completions requests use reasoning_effort. Configure "
            "reasoning_effort instead of passthrough.reasoning."
        )
    if api == "chat":
        effort = body.get("reasoning_effort")
        if effort is None:
            effort = (reasoning or {}).get("effort")
        if effort is None and is_openrouter and (reasoning or {}).get("enabled") is False:
            effort = "none"
    else:
        effort = (reasoning or {}).get("effort")
    _validate_reasoning_effort(effort, variant, model_label)

    if api == "chat":
        _validate_chat_tools(body, variant, model_label, effort, is_openrouter)
    sampling_effort = _get_sampling_effort(
        body,
        api,
        effort,
        variant,
        model_label,
        is_openrouter,
        supports_persisted_effort_updates
        and (reasoning or {}).get("mode") != "pro"
        and (_get_object(body.get("multi_agent")) or {}).get("enabled") is not True,
    )
    if (
        api == "responses"
        and sampling_effort == "none"
        and "temperature" not in body
        and default_responses_temperature is not None
    ):
        body["temperature"] = default_responses_temperature

    if variant in ("astra", "6.1-sol") or (
        sampling_effort is not UNKNOWN_PERSISTED_EFFORT and sampling_effort != "none"
    ):
        for key in ("temperature", "top_p", "logprobs", "top_logprobs"):
            body.pop(key, None)
        if isinstance(body.get("include"), list):
            body["include"] = [i for i in body["include"] if i != "message.output_text.logprobs"]
    body.pop("max_tokens", None)
    body.pop("max_output_tokens" if api == "chat" else "max_completion_tokens", None)
