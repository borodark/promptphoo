"""Claude model catalogue, capability rules, message parsing, output, usage and cost.
Ports src/providers/anthropic/util.ts (and parseDataUrl from src/util/dataUrl.ts)."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.providers.shared import calculate_cost as calculate_cost_base


def _models_with_cost(ids: list[str], input: float, output: float) -> list[dict[str, Any]]:
    return [{"id": id, "cost": {"input": input, "output": output}} for id in ids]


# Model definitions with cost information (USD per token).
ANTHROPIC_MODELS: list[dict[str, Any]] = [
    *_models_with_cost(
        ["claude-fable-5-1", "claude-mythos-5-1", "claude-fable-5", "claude-mythos-5"],
        10 / 1e6,
        50 / 1e6,
    ),
    *_models_with_cost(["claude-opus-5-5"], 4 / 1e6, 20 / 1e6),
    *_models_with_cost(["claude-opus-5"], 5 / 1e6, 25 / 1e6),
    *_models_with_cost(["claude-sonnet-5-5"], 2 / 1e6, 10 / 1e6),
    *_models_with_cost(["claude-sonnet-5"], 2 / 1e6, 10 / 1e6),
    *_models_with_cost(["claude-mythos-preview"], 25 / 1e6, 125 / 1e6),
    *_models_with_cost(["claude-opus-4-8"], 5 / 1e6, 25 / 1e6),
    *_models_with_cost(["claude-opus-4-7"], 5 / 1e6, 25 / 1e6),
    *_models_with_cost(["claude-sonnet-4-6"], 3 / 1e6, 15 / 1e6),
    *_models_with_cost(["claude-opus-4-6"], 5 / 1e6, 25 / 1e6),
    *_models_with_cost(["claude-opus-4-5", "claude-opus-4-5-20251101"], 5 / 1e6, 25 / 1e6),
    *_models_with_cost(
        [
            "claude-opus-4-1",
            "claude-opus-4-1-20250805",
            "claude-opus-4-20250514",
            "claude-opus-4-0",
        ],
        15 / 1e6,
        75 / 1e6,
    ),
    *_models_with_cost(
        [
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-5-latest",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-0",
        ],
        3 / 1e6,
        15 / 1e6,
    ),
    *_models_with_cost(["claude-haiku-4-5", "claude-haiku-4-5-20251001"], 1 / 1e6, 5 / 1e6),
    *_models_with_cost(["claude-2.0"], 0.008 / 1000, 0.024 / 1000),
    *_models_with_cost(["claude-2.1"], 0.008 / 1000, 0.024 / 1000),
    *_models_with_cost(["claude-3-haiku-20240307"], 0.00025 / 1000, 0.00125 / 1000),
    *_models_with_cost(["claude-3-opus-20240229"], 0.015 / 1000, 0.075 / 1000),
    *_models_with_cost(["claude-3-5-haiku-20241022"], 0.8 / 1e6, 4 / 1e6),
    *_models_with_cost(
        ["claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219"],
        3 / 1e6,
        15 / 1e6,
    ),
]

# Aliases promptfoo used to accept that Anthropic does not publish; kept out of ANTHROPIC_MODELS.
ANTHROPIC_COMPATIBILITY_ALIAS_MODELS: list[dict[str, Any]] = [
    *_models_with_cost(["claude-opus-4-6-latest", "claude-opus-4-5-latest"], 5 / 1e6, 25 / 1e6),
    *_models_with_cost(["claude-sonnet-4-6-latest"], 3 / 1e6, 15 / 1e6),
    *_models_with_cost(["claude-haiku-4-5-latest"], 1 / 1e6, 5 / 1e6),
    *_models_with_cost(["claude-opus-4-latest"], 15 / 1e6, 75 / 1e6),
    *_models_with_cost(["claude-sonnet-4-latest"], 3 / 1e6, 15 / 1e6),
]

ANTHROPIC_SHORTHAND_MODEL_IDS = {m["id"] for m in ANTHROPIC_MODELS} | {
    m["id"] for m in ANTHROPIC_COMPATIBILITY_ALIAS_MODELS
}

_B = r"(^|[^a-z0-9])"
CLAUDE_FABLE_MYTHOS_5_PATTERN = re.compile(_B + r"claude-(?:fable|mythos)-5(?![a-z0-9])", re.I)
CLAUDE_MYTHOS_PREVIEW_RE = re.compile(_B + r"claude-mythos-preview(?![a-z0-9])", re.I)
CLAUDE_FABLE_MYTHOS_51_PATTERN = re.compile(_B + r"claude-(?:fable|mythos)-5-1(?![a-z0-9])", re.I)
CLAUDE_OPUS_55_PATTERN = re.compile(_B + r"claude-opus-5-5(?![a-z0-9])", re.I)
CLAUDE_OPUS_5_PATTERN = re.compile(_B + r"claude-opus-5(?!-[0-9](?![0-9]))(?![a-z0-9])", re.I)
CLAUDE_SONNET_55_PATTERN = re.compile(_B + r"claude-sonnet-5-5(?![a-z0-9])", re.I)
CLAUDE_SONNET_5_PATTERN = re.compile(_B + r"claude-sonnet-5(?!-[0-9](?![0-9]))(?![a-z0-9])", re.I)
CLAUDE_OPUS_48_PATTERN = re.compile(_B + r"claude-opus-4-8(?![0-9])", re.I)
CLAUDE_OPUS_47_PATTERN = re.compile(_B + r"claude-opus-4-7(?![0-9])", re.I)
CLAUDE_5_OR_LATER_PATTERN = re.compile(
    _B + r"claude-[a-z][a-z0-9]*(?:-[a-z][a-z0-9]*){0,4}-(?:[5-9]|[1-9][0-9])(?![a-z0-9])", re.I
)
CLAUDE_4_5_AND_4_6_REGIONAL_PREMIUM_PATTERN = re.compile(
    _B + r"claude-(?:opus|sonnet|haiku)-4-(?:5|6)(?![0-9])", re.I
)

# Single source of truth for Claude model capabilities (CLAUDE_MODEL_FAMILIES).
CLAUDE_MODEL_FAMILIES: list[dict[str, Any]] = [
    {
        "match": CLAUDE_MYTHOS_PREVIEW_RE,
        "warningName": "Claude Mythos Preview",
        "samplingParamsDeprecated": True,
        "alwaysOnAdaptiveThinking": True,
    },
    {
        "match": CLAUDE_FABLE_MYTHOS_51_PATTERN,
        "warningName": "Claude Fable 5.1 and Claude Mythos 5.1",
        "samplingParamsDeprecated": True,
        "alwaysOnAdaptiveThinking": True,
        "forcedToolChoiceUnsupported": True,
        "regionalPremium": True,
        "cacheReadMultiplier": 0.025,
    },
    {
        "match": CLAUDE_FABLE_MYTHOS_5_PATTERN,
        "warningName": "Claude Fable 5 and Claude Mythos 5",
        "samplingParamsDeprecated": True,
        "alwaysOnAdaptiveThinking": True,
        "regionalPremium": True,
    },
    {
        "match": CLAUDE_OPUS_55_PATTERN,
        "warningName": "Claude Opus 5.5",
        "samplingParamsDeprecated": True,
        "alwaysOnAdaptiveThinking": True,
        "forcedToolChoiceUnsupported": True,
        "regionalPremium": True,
        "cacheReadMultiplier": 0.05,
    },
    {
        "match": CLAUDE_OPUS_5_PATTERN,
        "warningName": "Claude Opus 5",
        "samplingParamsDeprecated": True,
        "thinkingOnByDefault": True,
        "disabledThinkingEffortCapped": True,
        "regionalPremium": True,
    },
    {
        "match": CLAUDE_SONNET_55_PATTERN,
        "warningName": "Claude Sonnet 5.5",
        "samplingParamsDeprecated": True,
        "thinkingOnByDefault": True,
        "disabledThinkingEffortCapped": True,
        "betweenToolsReplacesDisabledThinking": True,
        "forcedToolChoiceUnsupported": True,
        "regionalPremium": True,
    },
    {
        "match": CLAUDE_SONNET_5_PATTERN,
        "warningName": "Claude Sonnet 5",
        "samplingParamsDeprecated": True,
        "thinkingOnByDefault": True,
        "regionalPremium": True,
    },
    {
        "match": CLAUDE_OPUS_48_PATTERN,
        "warningName": "Claude Opus 4.7 and 4.8",
        "samplingParamsDeprecated": True,
        "regionalPremium": True,
    },
    {
        "match": CLAUDE_OPUS_47_PATTERN,
        "warningName": "Claude Opus 4.7 and 4.8",
        "samplingParamsDeprecated": True,
        "regionalPremium": True,
    },
    {"match": CLAUDE_4_5_AND_4_6_REGIONAL_PREMIUM_PATTERN, "regionalPremium": True},
]


def _has_claude_capability(model_id: str, capability: str) -> bool:
    return any(f.get(capability) and f["match"].search(model_id) for f in CLAUDE_MODEL_FAMILIES)


def is_claude_opus55_model(model_id: str) -> bool:
    return CLAUDE_OPUS_55_PATTERN.search(model_id) is not None


def is_claude_opus5_model(model_id: str) -> bool:
    return CLAUDE_OPUS_5_PATTERN.search(model_id) is not None


def is_claude_fable_or_mythos5_model(model_id: str) -> bool:
    return CLAUDE_FABLE_MYTHOS_5_PATTERN.search(model_id) is not None


def is_claude_sonnet55_model(model_id: str) -> bool:
    return CLAUDE_SONNET_55_PATTERN.search(model_id) is not None


def is_claude_sonnet5_model(model_id: str) -> bool:
    return CLAUDE_SONNET_5_PATTERN.search(model_id) is not None


def is_forced_tool_choice_unsupported_claude_model(model_id: str) -> bool:
    return _has_claude_capability(model_id, "forcedToolChoiceUnsupported")


def get_claude_model_warning_name(model_id: str) -> str | None:
    for family in CLAUDE_MODEL_FAMILIES:
        if family.get("warningName") and family["match"].search(model_id):
            return family["warningName"]
    return None


def is_claude_regional_premium_model(model_id: str) -> bool:
    return _has_claude_capability(model_id, "regionalPremium")


def is_always_on_adaptive_thinking_claude_model(model_id: str) -> bool:
    return _has_claude_capability(model_id, "alwaysOnAdaptiveThinking")


def is_between_tools_lowest_thinking_claude_model(model_id: str) -> bool:
    return _has_claude_capability(model_id, "betweenToolsReplacesDisabledThinking")


def is_thinking_on_by_default_claude_model(model_id: str) -> bool:
    return _has_claude_capability(model_id, "thinkingOnByDefault")


def claude_thinking_consumes_tokens(model_id: str, resolved_thinking: dict | None) -> bool:
    thinking_type = (resolved_thinking or {}).get("type")
    return (
        is_always_on_adaptive_thinking_claude_model(model_id)
        or thinking_type in ("enabled", "adaptive")
        or (resolved_thinking is None and is_thinking_on_by_default_claude_model(model_id))
    )


def is_disabled_thinking_rejected_at_effort(model_id: str, effort: str | None) -> bool:
    return _has_claude_capability(model_id, "disabledThinkingEffortCapped") and effort in (
        "xhigh",
        "max",
    )


def looks_like_claude_model_id(model_id: str) -> bool:
    return re.match(r"^claude-[a-z0-9]", model_id, re.I) is not None


def clamp_max_tokens_for_thinking_budget(max_tokens: Any, thinking: dict | None) -> Any:
    """Raise `max_tokens` above a manual thinking budget (the API rejects <= budget)."""
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return max_tokens
    budget: Any = thinking.get("budget_tokens")
    if not is_truthy(budget):
        return max_tokens
    return budget + 1024 if max_tokens <= budget else max_tokens


def normalize_anthropic_model_name(model_name: str) -> str:
    return re.sub(r"^(?:(?:global|us|eu|jp|au)\.)?anthropic\.", "", model_name, count=1)


def is_sampling_params_deprecated_claude_model(
    model_id: str, allow_generation_fallback: bool | None = None
) -> bool:
    is_app_profile_arn = model_id.startswith("arn:") and ":application-inference-profile/" in (
        model_id
    )
    return _has_claude_capability(model_id, "samplingParamsDeprecated") or (
        allow_generation_fallback is not False
        and not is_app_profile_arn
        and CLAUDE_5_OR_LATER_PATTERN.search(model_id) is not None
    )


def is_claude_thinking_enabled(thinking: Any) -> bool:
    return isinstance(thinking, dict) and thinking.get("type") in ("enabled", "adaptive")


def _num(value: Any) -> str:
    from promptphoo.jsvalues import js_string

    return js_string(value)


def resolve_claude_sampling_params(
    params: dict[str, Any],
    *,
    thinking_enabled: bool,
    sampling_params_deprecated: bool,
    default_temperature: float | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return (sampling params to send, warnings for dropped values)."""
    temperature = params.get("temperature")
    top_p = params.get("top_p")
    top_k = params.get("top_k")
    if sampling_params_deprecated:
        return {}, []
    warnings: list[str] = []
    if thinking_enabled:
        if top_k is not None:
            warnings.append(
                "top_k is incompatible with extended thinking and will be omitted. Remove top_k "
                "from your config or disable thinking."
            )
        if temperature is not None:
            warnings.append(
                "temperature is incompatible with extended thinking and will be omitted. Remove "
                "temperature from your config or disable thinking."
            )
        if top_p is None:
            return {}, warnings
        if top_p < 0.95 or top_p > 1:
            warnings.append(
                f"top_p must be between 0.95 and 1.0 with extended thinking (got {_num(top_p)}). "
                "Clamping to valid range."
            )
        return {"top_p": max(0.95, min(1, top_p))}, warnings
    if top_p is not None and temperature is not None:
        warnings.append(
            "temperature is incompatible with top_p on Claude and will be omitted. Remove one of "
            "these parameters."
        )
    resolved_temperature = (
        (temperature if temperature is not None else default_temperature) if top_p is None else None
    )
    sampling: dict[str, Any] = {}
    if resolved_temperature is not None:
        sampling["temperature"] = resolved_temperature
    if top_p is not None:
        sampling["top_p"] = top_p
    if top_k is not None:
        sampling["top_k"] = top_k
    return sampling, warnings


def normalize_claude_thinking_config(
    model_id: str,
    thinking: Any,
    effort: str | None,
    allow_generation_fallback: bool | None = None,
) -> Any:
    """Reconcile a thinking config with what the model accepts (see util.ts)."""
    thinking_type = thinking.get("type") if isinstance(thinking, dict) else None
    if thinking_type == "enabled" and (
        is_sampling_params_deprecated_claude_model(model_id, allow_generation_fallback)
        or is_always_on_adaptive_thinking_claude_model(model_id)
    ):
        out: dict[str, Any] = {"type": "adaptive"}
        if thinking.get("display"):
            out["display"] = thinking["display"]
        return out
    if thinking_type == "disabled":
        if is_always_on_adaptive_thinking_claude_model(
            model_id
        ) or is_disabled_thinking_rejected_at_effort(model_id, effort):
            return None
        if is_between_tools_lowest_thinking_claude_model(model_id):
            return {"type": "between_tools"}
    if (
        thinking_type == "between_tools"
        and is_between_tools_lowest_thinking_claude_model(model_id)
        and is_disabled_thinking_rejected_at_effort(model_id, effort)
    ):
        return None
    return thinking


CLAUDE_REGIONAL_ENDPOINT_PREMIUM = 1.1
CLAUDE_US_INFERENCE_GEO_MULTIPLIER = 1.1
CLAUDE_46_OR_LATER_MODEL_PATTERN = re.compile(
    r"^claude-(?:(?:opus|sonnet)-4-(?:6|7|8)(?:-|$)|(?:fable|mythos|opus|sonnet)-5(?:-|$))"
)


def apply_claude_regional_premium(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    if (
        not is_claude_regional_premium_model(model_name)
        or config.get("cost") is not None
        or config.get("inputCost") is not None
        or config.get("outputCost") is not None
    ):
        return config
    return {**config, "regionalPremiumMultiplier": CLAUDE_REGIONAL_ENDPOINT_PREMIUM}


def get_file_references(block: dict[str, Any]) -> list[dict[str, Any]]:
    block_type = block.get("type")
    if block_type == "container_upload":
        return [{"type": block_type, "file_id": block.get("file_id")}]
    content = block.get("content")
    if (
        block_type in ("code_execution_tool_result", "bash_code_execution_tool_result")
        and isinstance(content, dict)
        and "content" in content
    ):
        return [
            {"type": item.get("type"), "file_id": item.get("file_id")}
            for item in content["content"]
        ]
    return []


def output_from_message(message: dict[str, Any], show_thinking: bool) -> str:
    segments: list[str] = []
    previous_block_was_text = False
    stop_reason = message.get("stop_reason")
    for block in message.get("content") or []:
        block_type = block.get("type")
        if block_type == "text":
            # Citation boundaries can split a sentence or table row across text blocks.
            if previous_block_was_text:
                segments[-1] += block.get("text", "")
            else:
                segments.append(block.get("text", ""))
            previous_block_was_text = True
            continue
        previous_block_was_text = False
        if block_type == "thinking" and show_thinking and block.get("thinking", "").strip() != "":
            segments.append(f"Thinking: {block['thinking']}\nSignature: {block.get('signature')}")
        elif block_type == "redacted_thinking" and show_thinking:
            segments.append(f"Redacted Thinking: {block.get('data')}")
        elif block_type == "tool_use" or (
            stop_reason == "pause_turn" and block_type not in ("thinking", "redacted_thinking")
        ):
            segments.append(json_stringify(block))
        else:
            # Keep generated files without including execution logs or encrypted payloads.
            for file in get_file_references(block):
                segments.append(json_stringify(file))
    return "\n\n".join(s for s in segments if s != "")


def parse_data_url(data_url: Any) -> dict[str, str] | None:
    """Ports parseDataUrl from src/util/dataUrl.ts."""
    if not isinstance(data_url, str) or not data_url.startswith("data:") or len(data_url) <= 5:
        return None
    m = re.match(r"^data:([^;,]+)(?:;[^,]*)?;base64,(.+)$", data_url, re.S)
    if not m:
        return None
    return {"mimeType": m.group(1).strip(), "base64Data": m.group(2).strip()}


def _process_anthropic_image_content(content: list[Any]) -> list[Any]:
    out = []
    for item in content:
        source = item.get("source") if isinstance(item, dict) else None
        if (
            isinstance(item, dict)
            and item.get("type") == "image"
            and isinstance(source, dict)
            and source.get("type") == "base64"
        ):
            parsed = parse_data_url(source.get("data"))
            if parsed:
                item = {
                    **item,
                    "source": {
                        **source,
                        "media_type": source.get("media_type") or parsed["mimeType"],
                        "data": parsed["base64Data"],
                    },
                }
        out.append(item)
    return out


def _js_trim(s: str) -> str:
    return s.strip(" \t\n\r\v\f   ﻿")


def parse_messages(messages: str) -> dict[str, Any]:
    """Split a prompt into {system?, extractedMessages, thinking?} for the Messages API."""
    try:
        parsed = json.loads(messages)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        try:
            return _parse_message_array(parsed)
        except (AttributeError, TypeError):
            pass  # JS would have thrown inside the try; fall back to plain text.

    lines = [line for line in (_js_trim(raw) for raw in messages.split("\n")) if line]
    system: list[dict[str, Any]] | None = None
    thinking: Any = None
    extracted: list[dict[str, Any]] = []
    current_role: str | None = None
    current_content: list[str] = []

    def push_message() -> None:
        nonlocal current_content
        if current_role and current_content:
            extracted.append(
                {
                    "role": current_role,
                    "content": [{"type": "text", "text": "\n".join(current_content)}],
                }
            )
            current_content = []

    for line in lines:
        if line.startswith("system:"):
            system = [{"type": "text", "text": _js_trim(line[7:])}]
        elif line.startswith("thinking:"):
            try:
                thinking = json.loads(_js_trim(line[9:]))
            except ValueError:
                pass  # Invalid thinking config, ignore
        elif line.startswith(("user:", "assistant:")):
            push_message()
            current_role = "user" if line.startswith("user:") else "assistant"
            current_content.append(_js_trim(line[line.index(":") + 1 :]))
        elif current_role:
            current_content.append(line)
        else:
            current_role = "user"
            current_content.append(line)
    push_message()

    if not extracted and not system:
        extracted.append(
            {"role": "user", "content": [{"type": "text", "text": _js_trim(messages)}]}
        )

    result: dict[str, Any] = {"extractedMessages": extracted}
    if system is not None:
        result["system"] = system
    if thinking is not None:
        result["thinking"] = thinking
    return result


def _parse_message_array(parsed: list[Any]) -> dict[str, Any]:
    system_message = next((m for m in parsed if m.get("role") == "system"), None)
    thinking_message = next((m for m in parsed if is_truthy(m.get("thinking"))), None)
    extracted = [
        {
            "role": m["role"],
            "content": _process_anthropic_image_content(m.get("content"))
            if isinstance(m.get("content"), list)
            else [{"type": "text", "text": m.get("content")}],
        }
        for m in parsed
        if is_truthy(m.get("role")) and m.get("role") != "system"
    ]
    result: dict[str, Any] = {"extractedMessages": extracted}
    if system_message is not None:
        content = system_message.get("content")
        result["system"] = (
            content if isinstance(content, list) else [{"type": "text", "text": content}]
        )
    if thinking_message is not None:
        result["thinking"] = thinking_message["thinking"]
    return result


def calculate_cache_input_cost(
    base_input_rate: float,
    uncached_input_tokens: float,
    cache_read: float,
    cache_creation: float,
    model_id: str = "",
    cache_creation_1h: float = 0,
) -> float:
    """Anthropic cache pricing: reads at the family multiplier (default 0.1x), 5-minute writes
    at 1.25x and one-hour writes at 2x the base input rate."""
    multiplier = next(
        (
            f["cacheReadMultiplier"]
            for f in CLAUDE_MODEL_FAMILIES
            if f.get("cacheReadMultiplier") is not None and f["match"].search(model_id)
        ),
        0.1,
    )
    one_hour = min(max(cache_creation_1h, 0), cache_creation)
    five_minute = max(cache_creation - one_hour, 0)
    return (
        uncached_input_tokens * base_input_rate
        + cache_read * base_input_rate * multiplier
        + five_minute * base_input_rate * 1.25
        + one_hour * base_input_rate * 2
    )


def _finite(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and not (isinstance(value, float) and not math.isfinite(value))
    )


def calculate_anthropic_cost(
    model_name: str,
    config: dict[str, Any],
    prompt_tokens: float | None = None,
    completion_tokens: float | None = None,
    cache_read_tokens: float | None = None,
    cache_creation_tokens: float | None = None,
    cache_creation_1h_tokens: float | None = None,
    reported_inference_geo: str | None = None,
) -> float | None:
    pricing_model_name = normalize_anthropic_model_name(model_name)
    has_explicit_pricing = (
        config.get("cost") is not None
        or config.get("inputCost") is not None
        or config.get("outputCost") is not None
    )
    registered = next((m for m in ANTHROPIC_MODELS if m["id"] == pricing_model_name), None)
    if registered is None and has_explicit_pricing:
        registered = next(
            (m for m in ANTHROPIC_COMPATIBILITY_ALIAS_MODELS if m["id"] == pricing_model_name),
            None,
        )
    if pricing_model_name != model_name and pricing_model_name == "claude-sonnet-5":
        model_info: dict[str, Any] | None = {
            "id": pricing_model_name,
            "cost": {"input": 3 / 1e6, "output": 15 / 1e6},
        }
    else:
        model_info = registered
    uses_regional_bedrock_pricing = pricing_model_name != model_name and not model_name.startswith(
        "global."
    )
    effective = (
        apply_claude_regional_premium(model_name, config)
        if uses_regional_bedrock_pricing
        else config
    )
    regional_multiplier = effective.get("regionalPremiumMultiplier")
    if regional_multiplier is None:
        regional_multiplier = 1
    inference_geo = reported_inference_geo
    if inference_geo is None:
        extra_body = effective.get("extra_body")
        inference_geo = extra_body.get("inference_geo") if isinstance(extra_body, dict) else None
    uses_us_geo = (
        pricing_model_name == model_name
        and inference_geo == "us"
        and CLAUDE_46_OR_LATER_MODEL_PATTERN.search(pricing_model_name) is not None
        and effective.get("cost") is None
        and effective.get("inputCost") is None
        and effective.get("outputCost") is None
    )
    geo_multiplier = CLAUDE_US_INFERENCE_GEO_MULTIPLIER if uses_us_geo else 1

    def with_multipliers(cost: float | None) -> float | None:
        return None if cost is None else cost * regional_multiplier * geo_multiplier

    uses_flat_cost = (
        effective.get("cost") is not None
        and effective.get("inputCost") is None
        and effective.get("outputCost") is None
    )
    cache_read = cache_read_tokens or 0
    cache_creation = cache_creation_tokens or 0
    cache_creation_1h = cache_creation_1h_tokens or 0

    if (
        not uses_flat_cost
        and model_info
        and (cache_read or cache_creation)
        and _finite(prompt_tokens)
        and _finite(completion_tokens)
    ):
        assert prompt_tokens is not None and completion_tokens is not None

        def pick(*values: Any) -> Any:
            return next(v for v in values if v is not None)

        input_cost = pick(
            effective.get("inputCost"), effective.get("cost"), model_info["cost"]["input"]
        )
        output_cost = pick(
            effective.get("outputCost"), effective.get("cost"), model_info["cost"]["output"]
        )
        return with_multipliers(
            calculate_cache_input_cost(
                input_cost,
                prompt_tokens,
                cache_read,
                cache_creation,
                pricing_model_name,
                cache_creation_1h,
            )
            + completion_tokens * output_cost
        )

    return with_multipliers(
        calculate_cost_base(
            pricing_model_name,
            effective,
            prompt_tokens,
            completion_tokens,
            [model_info] if model_info else [],
        )
    )


def get_refusal_details(data: dict[str, Any]) -> str | None:
    """Human-readable refusal summary from `stop_details`, or None."""
    details = data.get("stop_details")
    if data.get("stop_reason") != "refusal" or not details:
        return None
    parts = ["Content refused by Anthropic safety filters"]
    if details.get("category"):
        parts.append(f"category: {details['category']}")
    if details.get("explanation"):
        parts.append(f"explanation: {details['explanation']}")
    return " — ".join(parts)


def _to_token_count(value: Any) -> float:
    if isinstance(value, str):
        stripped = _js_trim(value)
        try:
            parsed: Any = float(stripped) if stripped else 0
        except ValueError:
            return 0
        if isinstance(parsed, float) and parsed.is_integer():
            parsed = int(parsed)
        value = parsed
    return value if _finite(value) else 0


def get_token_usage(data: dict[str, Any], cached: bool) -> dict[str, Any]:
    usage = data.get("usage") if isinstance(data, dict) else None
    if not is_truthy(usage):
        return {}
    assert isinstance(usage, dict)
    cache_read = _to_token_count(usage.get("cache_read_input_tokens"))
    cache_creation = _to_token_count(usage.get("cache_creation_input_tokens"))
    all_input = _to_token_count(usage.get("input_tokens")) + cache_read + cache_creation
    total = all_input + _to_token_count(usage.get("output_tokens"))
    if cached:
        return {"cached": total, "total": total}
    result: dict[str, Any] = {
        "total": total,
        "prompt": all_input,
        "completion": _to_token_count(usage.get("output_tokens")),
    }
    details = usage.get("output_tokens_details")
    thinking_tokens = details.get("thinking_tokens") if isinstance(details, dict) else None
    has_cache_details = (
        usage.get("cache_read_input_tokens") is not None
        or usage.get("cache_creation_input_tokens") is not None
    )
    if thinking_tokens is not None or has_cache_details:
        completion_details: dict[str, Any] = {}
        if thinking_tokens is not None:
            completion_details["reasoning"] = _to_token_count(thinking_tokens)
        if has_cache_details:
            completion_details["cacheReadInputTokens"] = cache_read
            completion_details["cacheCreationInputTokens"] = cache_creation
        result["completionDetails"] = completion_details
    return result


# Fields forwarded to each server-tool version.
_WEB_FETCH_FIELDS = (
    "allowed_callers",
    "max_uses",
    "allowed_domains",
    "blocked_domains",
    "citations",
    "max_content_tokens",
    "cache_control",
    "defer_loading",
    "strict",
    "url_sources",
)
_WEB_FETCH_20260309_FIELDS = (*_WEB_FETCH_FIELDS, "use_cache")
_WEB_FETCH_20260318_FIELDS = (*_WEB_FETCH_20260309_FIELDS, "response_inclusion")
_WEB_SEARCH_FIELDS = (
    "allowed_callers",
    "allowed_domains",
    "blocked_domains",
    "cache_control",
    "defer_loading",
    "max_uses",
    "strict",
    "user_location",
)
_WEB_SEARCH_20260318_FIELDS = (*_WEB_SEARCH_FIELDS, "response_inclusion")

SERVER_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "web_fetch_20250910": {
        "name": "web_fetch",
        "fields": _WEB_FETCH_FIELDS,
        "betaFeature": "web-fetch-2025-09-10",
    },
    "web_fetch_20260209": {"name": "web_fetch", "fields": _WEB_FETCH_FIELDS},
    "web_fetch_20260309": {"name": "web_fetch", "fields": _WEB_FETCH_20260309_FIELDS},
    "web_fetch_20260318": {"name": "web_fetch", "fields": _WEB_FETCH_20260318_FIELDS},
    "web_search_20250305": {"name": "web_search", "fields": _WEB_SEARCH_FIELDS},
    "web_search_20260209": {"name": "web_search", "fields": _WEB_SEARCH_FIELDS},
    "web_search_20260318": {"name": "web_search", "fields": _WEB_SEARCH_20260318_FIELDS},
}


def process_anthropic_tools(tools: list[Any] | None = None) -> dict[str, list[Any]]:
    """Rebuild server tools (web fetch/search) from their specs and collect beta features.

    Returns {"processedTools": [...], "requiredBetaFeatures": [...]}.
    """
    processed: list[Any] = []
    required: list[str] = []

    def add(feature: str) -> None:
        if feature not in required:
            required.append(feature)

    for tool in tools or []:
        tool_type = tool.get("type") if isinstance(tool, dict) else None
        spec = SERVER_TOOL_SPECS.get(tool_type) if isinstance(tool_type, str) else None
        if spec:
            built: dict[str, Any] = {"type": tool_type, "name": spec["name"]}
            for f in spec["fields"]:
                if f in tool:
                    built[f] = tool[f]
            processed.append(built)
            if spec.get("betaFeature"):
                add(spec["betaFeature"])
        else:
            processed.append(tool)
        if isinstance(tool, dict) and tool.get("strict") is True:
            add("structured-outputs-2025-11-13")
    return {"processedTools": processed, "requiredBetaFeatures": required}
