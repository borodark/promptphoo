"""Shared Gemini request/response helpers. Ports src/providers/google/util.ts.

Covers safety settings, tool config and tool loading, message conversion
(maybeCoerceToGeminiFormat / geminiFormatAndSystemInstructions, inline media detection),
response parsing (candidates, blocked/safety finish reasons, grounding, thought signatures,
audio), function-call validation, schema sanitizing and the cost model.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import logging
import math
import ntpath
import os
import re
import struct
import time
from collections.abc import Mapping
from typing import Any

from promptphoo.jsvalues import UNDEFINED, is_truthy, json_stringify
from promptphoo.providers.google.models import GEMINI_FLASH_MODELS, GOOGLE_MODELS
from promptphoo.providers.shared import (
    calculate_cost,
    clamp_cached_tokens,
    parse_chat_prompt,
    resolve_provider_env,
    transform_tool_choice,
    transform_tools,
)

logger = logging.getLogger(__name__)

VALID_SCHEMA_TYPES = ("TYPE_UNSPECIFIED", "STRING", "NUMBER", "INTEGER", "BOOLEAN", "ARRAY",
                      "OBJECT")  # fmt: skip


def _render_template(text: str, context: Mapping[str, Any] | None = None) -> str:
    from promptphoo.templating import render_string

    return render_string(text, context or {})


def _render_vars(value: Any, vars: Mapping[str, Any] | None) -> Any:
    from promptphoo.templating import render_vars_in_object

    return render_vars_in_object(value, vars)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_finite(value: Any) -> bool:
    return _is_number(value) and math.isfinite(value)


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


# --- safety settings / service tier ------------------------------------------------------------


def normalize_safety_settings(safety_settings: Any) -> list[dict[str, Any]] | None:
    """Maps the legacy `probability` field to the API's `threshold`."""
    if not is_truthy(safety_settings):
        return None
    return [
        {
            "category": s.get("category"),
            "threshold": s.get("threshold") or s.get("probability") or "",
        }
        for s in safety_settings
    ]


def normalize_google_service_tier(service_tier: Any) -> str | None:
    if not isinstance(service_tier, str):
        return None
    normalized = re.sub(r"^service_tier_", "", service_tier.lower())
    if normalized not in ("standard", "priority", "flex"):
        return service_tier
    return normalized


VERTEX_SHARED_REQUEST_TYPE_HEADER = "X-Vertex-AI-LLM-Shared-Request-Type"


def get_vertex_service_tier_headers(
    service_tier: Any, headers: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Add the documented Vertex PayGo tier without overriding an explicit custom header."""
    headers = dict(headers or {})
    name = VERTEX_SHARED_REQUEST_TYPE_HEADER
    if any(h.lower() == name.lower() for h in headers):
        return headers
    normalized = normalize_google_service_tier(service_tier)
    return {name: normalized, **headers} if normalized in ("priority", "flex") else headers


def get_google_response_service_tier(
    headers: Any, usage_metadata: Any = None, vertexai: bool = False
) -> str | None:
    """Read the actual processing tier before estimating costs for a downgraded request."""
    header_value = None
    if isinstance(headers, Mapping):
        header_value = next(
            (v for k, v in headers.items() if k.lower() == "x-gemini-service-tier"), None
        )
    metadata = usage_metadata if isinstance(usage_metadata, Mapping) else {}
    if header_value is None and vertexai and metadata.get("trafficType") is not None:
        return {
            "ON_DEMAND": "standard",
            "ON_DEMAND_PRIORITY": "priority",
            "ON_DEMAND_FLEX": "flex",
        }.get(metadata["trafficType"])
    normalized = normalize_google_service_tier(
        _coalesce(header_value, metadata.get("serviceTier"), metadata.get("service_tier"))
    )
    return normalized if normalized in ("standard", "priority", "flex") else None


# --- tool config -------------------------------------------------------------------------------


def _normalize_google_tool_mode(mode: Any) -> str | None:
    if not isinstance(mode, str):
        return None
    upper = mode.upper()
    return upper if upper in ("MODE_UNSPECIFIED", "AUTO", "VALIDATED", "ANY", "NONE") else None


def _normalize_camel_case_google_tool_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_fcc: Any = config.get("functionCallingConfig")
    normalized = {k: v for k, v in config.items() if k != "functionCallingConfig"}
    if is_truthy(raw_fcc):
        rest = {k: v for k, v in raw_fcc.items() if k not in ("mode", "allowedFunctionNames")}
        mode = _normalize_google_tool_mode(raw_fcc.get("mode"))
        allowed = raw_fcc.get("allowedFunctionNames")
        fcc = {
            **rest,
            **({"mode": mode} if mode else {}),
            **({"allowedFunctionNames": allowed} if allowed else {}),
        }
        if fcc:
            normalized["functionCallingConfig"] = fcc
    return normalized or None


def _normalize_snake_case_google_tool_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_fcc: Any = config.get("function_calling_config")
    retrieval: Any = config.get("retrieval_config")
    include = config.get("include_server_side_tool_invocations")
    normalized: dict[str, Any] = {}
    if is_truthy(retrieval):
        normalized["retrievalConfig"] = {
            **({"latLng": retrieval["lat_lng"]} if retrieval.get("lat_lng") else {}),
            **(
                {"languageCode": retrieval["language_code"]}
                if retrieval.get("language_code")
                else {}
            ),
        }
    if include is not None:
        normalized["includeServerSideToolInvocations"] = include
    if is_truthy(raw_fcc):
        mode = _normalize_google_tool_mode(raw_fcc.get("mode"))
        allowed = raw_fcc.get("allowed_function_names")
        stream = raw_fcc.get("stream_function_call_arguments")
        fcc = {
            **({"mode": mode} if mode else {}),
            **({"allowedFunctionNames": allowed} if allowed else {}),
            **({"streamFunctionCallArguments": stream} if stream is not None else {}),
        }
        if fcc:
            normalized["functionCallingConfig"] = fcc
    return normalized or None


def normalize_google_tool_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    if is_truthy(config.get("toolConfig")):
        return _normalize_camel_case_google_tool_config(config["toolConfig"])
    if is_truthy(config.get("tool_config")):
        return _normalize_snake_case_google_tool_config(config["tool_config"])
    return None


def resolve_google_tool_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Returns {toolConfig?, toolsDisabled}."""
    explicit = normalize_google_tool_config(config) or {}
    passthrough: Any = config.get("passthrough")
    passthrough_config = (
        normalize_google_tool_config(passthrough) if is_truthy(passthrough) else None
    ) or {}
    transformed = transform_tool_choice(config.get("tool_choice"), "google")
    tool_choice_config = transformed if isinstance(transformed, dict) else {}
    tool_config = {**tool_choice_config, **explicit, **passthrough_config}
    fcc = {
        **(tool_choice_config.get("functionCallingConfig") or {}),
        **(explicit.get("functionCallingConfig") or {}),
        **(passthrough_config.get("functionCallingConfig") or {}),
    }
    if fcc:
        tool_config["functionCallingConfig"] = fcc
    return {
        **({"toolConfig": tool_config} if tool_config else {}),
        "toolsDisabled": fcc.get("mode") == "NONE",
    }


def merge_google_request_tools(configured_tools: list[Any], passthrough_tools: Any) -> Any:
    """Merge configured tools with single-object or array passthrough tools."""
    if passthrough_tools is None:
        return configured_tools if configured_tools else None
    extra = passthrough_tools if isinstance(passthrough_tools, list) else [passthrough_tools]
    return [*configured_tools, *extra]


def get_google_api_base_url(config: Mapping[str, Any], env: Mapping[str, Any] | None) -> str:
    """Resolve native Gemini endpoints consistently across text and image adapters."""
    if config.get("apiHost"):
        return f"https://{_render_template(config['apiHost'])}"
    if config.get("apiBaseUrl"):
        return config["apiBaseUrl"]
    endpoint = resolve_provider_env(
        env, ["GOOGLE_API_HOST", "PALM_API_HOST", "GOOGLE_API_BASE_URL"]
    )
    if endpoint and endpoint["name"] == "GOOGLE_API_BASE_URL":
        return endpoint["value"]
    host = (endpoint or {}).get("value") or "generativelanguage.googleapis.com"
    return f"https://{_render_template(host)}"


def merge_google_completion_options(
    base_config: Mapping[str, Any], prompt_config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    merged: dict[str, Any] = {**base_config, **(prompt_config or {})}
    prompt_config = prompt_config or {}
    # `undefined` means "not set"; an explicit null still replaces the base policy.
    has_tool_choice = "tool_choice" in prompt_config
    has_tool_config = "toolConfig" in prompt_config
    has_snake_tool_config = "tool_config" in prompt_config

    if has_tool_choice or has_tool_config or has_snake_tool_config:
        for key in ("tool_choice", "toolConfig", "tool_config"):
            merged.pop(key, None)
        if has_tool_choice:
            merged["tool_choice"] = prompt_config["tool_choice"]
        if has_tool_config:
            merged["toolConfig"] = prompt_config["toolConfig"]
        if has_snake_tool_config:
            merged["tool_config"] = prompt_config["tool_config"]

        if has_tool_choice and not has_tool_config and not has_snake_tool_config:
            base_tool_config = normalize_google_tool_config(base_config)
            if base_tool_config:
                non_function = {
                    k: v for k, v in base_tool_config.items() if k != "functionCallingConfig"
                }
                if non_function:
                    merged["toolConfig"] = non_function

        # A provider-level passthrough policy must not override a prompt's tool choice.
        if "passthrough" not in prompt_config and is_truthy(base_config.get("passthrough")):
            passthrough = dict(base_config["passthrough"])
            inherited = normalize_google_tool_config(passthrough)
            passthrough.pop("toolConfig", None)
            passthrough.pop("tool_config", None)
            if inherited:
                non_function = {k: v for k, v in inherited.items() if k != "functionCallingConfig"}
                if non_function:
                    passthrough["toolConfig"] = non_function
            merged["passthrough"] = passthrough
    return merged


def remove_google_function_declarations(tools: Any) -> list[Any]:
    tool_list = tools if isinstance(tools, list) else [tools]
    result = []
    for raw in tool_list:
        if not isinstance(raw, dict):
            continue
        tool = {k: v for k, v in raw.items() if k not in ("functionDeclarations",
                                                            "function_declarations")}  # fmt: skip
        declares = is_truthy(raw.get("functionDeclarations")) or is_truthy(
            raw.get("function_declarations")
        )
        if declares and not tool:
            continue
        result.append(tool)
    return result


_DEPRECATED_FLASH_PARAMS = (
    "temperature", "topP", "top_p", "topK", "top_k", "candidateCount", "candidate_count",
    "presencePenalty", "presence_penalty", "frequencyPenalty", "frequency_penalty",
)  # fmt: skip


def remove_deprecated_gemini_generation_params(model_name: str, generation_config: Any) -> Any:
    """Current Gemini Flash models ignore sampling parameters and reject penalties."""
    if (
        not any(model_name.startswith(m["id"]) for m in GEMINI_FLASH_MODELS)
        and model_name != "gemini-flash-latest"
        and model_name != "gemini-flash-lite-latest"
    ):
        return generation_config
    if not isinstance(generation_config, dict):
        return generation_config
    sanitized = {k: v for k, v in generation_config.items() if k not in _DEPRECATED_FLASH_PARAMS}
    if (
        model_name.startswith("gemini-3.8-flash")
        or model_name.startswith("gemini-3.7-flash")
        or model_name == "gemini-flash-latest"
    ):
        for thinking_config in (sanitized.get("thinkingConfig"), sanitized.get("thinking_config")):
            if not isinstance(thinking_config, dict):
                continue
            if (
                thinking_config.get("thinkingBudget") is not None
                or thinking_config.get("thinking_budget") is not None
            ):
                raise ValueError(
                    f"{model_name} does not support thinkingBudget. Use thinkingLevel "
                    "(LOW, MEDIUM, or HIGH)."
                )
            level = _coalesce(
                thinking_config.get("thinkingLevel"), thinking_config.get("thinking_level")
            )
            if isinstance(level, str) and level.upper() == "MINIMAL":
                raise ValueError(
                    f"{model_name} does not support MINIMAL thinking. Use LOW, MEDIUM, or HIGH."
                )
    return sanitized


def _strip_executable_tool_file_references_from_value(tools: Any) -> Any:
    from promptphoo.files import is_javascript_file, parse_file_url

    if isinstance(tools, str) and tools.startswith("file://"):
        file_path, _ = parse_file_url(tools)
        if file_path.endswith(".py") or is_javascript_file(file_path):
            return UNDEFINED
    if isinstance(tools, list):
        out = []
        for tool in tools:
            stripped = _strip_executable_tool_file_references_from_value(tool)
            if stripped is not UNDEFINED:
                out.append(stripped)
        return out
    return tools


def strip_executable_tool_file_references(tools: Any, vars: Mapping[str, Any] | None) -> Any:
    stripped = _strip_executable_tool_file_references_from_value(_render_vars(tools, vars))
    return None if stripped is UNDEFINED else stripped


async def load_google_tools(
    tools: Any,
    vars: Mapping[str, Any] | None = None,
    skip_executable_tool_files: bool = False,
) -> list[Any]:
    """Load and normalize configured tools using the shared file and variable handling."""
    from promptphoo.assertions.tools import maybe_load_tools_from_external_file
    from promptphoo.jsvalues import is_truthy

    request_tools = (
        strip_executable_tool_file_references(tools, vars) if skip_executable_tool_files else tools
    )
    loaded = (
        await maybe_load_tools_from_external_file(
            request_tools, None if skip_executable_tool_files else dict(vars or {})
        )
        if is_truthy(request_tools)
        else []
    )
    if isinstance(loaded, list):
        transformed = transform_tools(loaded, "google")
    elif is_truthy(loaded):
        transformed = [loaded]
    else:
        transformed = []
    return normalize_tools(transformed)


# --- cost --------------------------------------------------------------------------------------


def _get_service_tier_cache_read(cost: Mapping[str, Any], tier: Any, audio: bool = False):
    if tier == "priority":
        if audio:
            return _coalesce(cost.get("priorityCacheReadAudio"), cost.get("priorityCacheRead"))
        return cost.get("priorityCacheRead")
    if tier == "flex":
        if audio:
            return _coalesce(cost.get("flexCacheReadAudio"), cost.get("flexCacheRead"))
        return cost.get("flexCacheRead")
    return None


def _resolve_service_tier_cache_cost(default: float, exact: Any, multiplier: float) -> float:
    return default if exact is None else exact / multiplier


def _resolve_google_service_tier_costs(
    cost: Mapping[str, Any],
    tier: Any,
    cached_input_cost: float,
    cached_audio_input_cost: float,
    cached_image_input_cost: float,
    audio_input_cost: float,
) -> dict[str, float]:
    multiplier = 1
    if tier == "priority":
        multiplier = _coalesce(cost.get("priorityMultiplier"), 1)
    elif tier == "flex":
        multiplier = _coalesce(cost.get("flexMultiplier"), 1)
    tier_cache_read = _get_service_tier_cache_read(cost, tier)
    tier_audio_cache_read = _get_service_tier_cache_read(cost, tier, True)
    tier_audio_input = audio_input_cost
    if tier == "priority" and cost.get("priorityAudioInput") is not None:
        tier_audio_input = cost["priorityAudioInput"] / multiplier
    elif tier == "flex" and cost.get("flexAudioInput") is not None:
        tier_audio_input = cost["flexAudioInput"] / multiplier
    return {
        "multiplier": multiplier,
        "cachedInput": _resolve_service_tier_cache_cost(
            cached_input_cost, tier_cache_read, multiplier
        ),
        "cachedAudioInput": _resolve_service_tier_cache_cost(
            cached_audio_input_cost, tier_audio_cache_read, multiplier
        ),
        "cachedImageInput": _resolve_service_tier_cache_cost(
            cached_image_input_cost, tier_cache_read, multiplier
        ),
        "audioInput": tier_audio_input,
    }


_SCALED_COST_FIELDS = (
    "cacheRead", "cacheReadAudio", "audioInput", "audioOutput", "imageInput",
    "videoInputPerSecond", "videoOutput", "priorityCacheRead", "priorityCacheReadAudio",
    "priorityAudioInput", "flexCacheRead", "flexCacheReadAudio", "flexAudioInput",
)  # fmt: skip


def _apply_google_regional_premium(cost: Mapping[str, Any], multiplier: float) -> dict[str, Any]:
    scaled = {**cost, "input": cost["input"] * multiplier, "output": cost["output"] * multiplier}
    for field in _SCALED_COST_FIELDS:
        value = cost.get(field)
        scaled[field] = None if value is None else value * multiplier
    return scaled


def _get_google_catalog_cost(
    model: Mapping[str, Any] | None, prompt_tokens: float, is_vertex: Any, non_global: Any
) -> Mapping[str, Any] | None:
    if not model:
        return None
    tiered = model.get("tieredCost")
    if tiered and prompt_tokens > tiered["threshold"]:
        return tiered["above"]
    if non_global and model.get("vertexRegionalCost"):
        return model["vertexRegionalCost"]
    if is_vertex and model.get("vertexCost"):
        return model["vertexCost"]
    return model.get("cost")


# 2026-10-08T00:00:00Z: the Gemini API redirects 3.5 Flash to 3.6 Flash from this date.
_GEMINI_3_5_FLASH_REDIRECT_MS = 1_791_417_600_000


def calculate_google_cost(
    model_name: str,
    config: Mapping[str, Any],
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    is_vertex_mode: Any = None,
    audio_prompt_tokens: Any = None,
    audio_completion_tokens: Any = None,
    video_completion_tokens: Any = None,
    image_prompt_tokens: Any = None,
    cached_prompt_tokens: Any = None,
    cached_audio_prompt_tokens: Any = None,
    cached_image_prompt_tokens: Any = None,
    actual_service_tier: str | None = None,
    vertex_region: str | None = None,
    requested_service_tier: Any = None,
    pricing_time: float | None = None,
) -> float | None:
    """Cost of a Google call, handling tiered, Vertex, regional, modality and tier pricing."""
    now_ms = time.time() * 1000
    timestamp: float = (
        pricing_time if pricing_time is not None and _is_finite(pricing_time) else now_ms
    )
    pricing_model_name = (
        "gemini-3.6-flash"
        if not is_vertex_mode
        and model_name == "gemini-3.5-flash"
        and timestamp >= _GEMINI_3_5_FLASH_REDIRECT_MS
        else model_name
    )
    model = next((m for m in GOOGLE_MODELS if m["id"] == pricing_model_name), None)

    if not _is_finite(prompt_tokens) or not _is_finite(completion_tokens):
        return calculate_cost(model_name, config, prompt_tokens, completion_tokens, GOOGLE_MODELS)

    effective_region = _coalesce(vertex_region, config.get("region"))
    non_global = (
        bool(is_vertex_mode) and effective_region is not None and (effective_region != "global")
    )
    base_cost = _get_google_catalog_cost(model, prompt_tokens, is_vertex_mode, non_global)
    if not base_cost:
        return None
    model = model or {}

    regional_multiplier = (
        model["vertexRegionalPremium"]
        if non_global
        and model.get("vertexRegionalCost") is None
        and model.get("vertexRegionalPremium") is not None
        else 1
    )
    intro = model.get("introductoryPricing")
    catalog_multiplier = intro["multiplier"] if intro and timestamp < intro["expiresAt"] else 1

    def catalog(rate: Any) -> Any:
        return None if rate is None else rate * catalog_multiplier

    model_cost = (
        base_cost
        if regional_multiplier == 1
        else _apply_google_regional_premium(base_cost, regional_multiplier)
    )
    passthrough = config.get("passthrough") or {}
    service_tier = normalize_google_service_tier(
        _coalesce(
            actual_service_tier,
            requested_service_tier if is_vertex_mode else None,
            passthrough.get("service_tier") if isinstance(passthrough, Mapping) else None,
            passthrough.get("serviceTier") if isinstance(passthrough, Mapping) else None,
            config.get("service_tier"),
        )
    )
    catalog_input = model_cost["input"] * catalog_multiplier
    catalog_output = model_cost["output"] * catalog_multiplier
    catalog_audio_input = _coalesce(catalog(model_cost.get("audioInput")), catalog_input)
    catalog_image_input = _coalesce(catalog(model_cost.get("imageInput")), catalog_input)
    catalog_cache_read = catalog(model_cost.get("cacheRead"))
    tier = _resolve_google_service_tier_costs(
        _apply_google_regional_premium(model_cost, catalog_multiplier),
        service_tier,
        _coalesce(catalog_cache_read, catalog_input),
        _coalesce(catalog(model_cost.get("cacheReadAudio")), catalog_cache_read,
                  catalog_audio_input),
        _coalesce(catalog_cache_read, catalog_image_input),
        catalog_audio_input,
    )  # fmt: skip
    m = tier["multiplier"]
    c = config
    input_cost = _coalesce(c.get("inputCost"), c.get("cost"), catalog_input * m)
    output_cost = _coalesce(c.get("outputCost"), c.get("cost"), catalog_output * m)

    audio_input_tokens = clamp_cached_tokens(audio_prompt_tokens, prompt_tokens)
    image_input_tokens = clamp_cached_tokens(
        image_prompt_tokens, max(prompt_tokens - audio_input_tokens, 0)
    )
    text_input_tokens = max(prompt_tokens - audio_input_tokens - image_input_tokens, 0)
    cached_tokens = clamp_cached_tokens(cached_prompt_tokens, prompt_tokens)
    cached_audio = clamp_cached_tokens(
        cached_audio_prompt_tokens, min(cached_tokens, audio_input_tokens)
    )
    cached_image = clamp_cached_tokens(
        cached_image_prompt_tokens,
        min(max(cached_tokens - cached_audio, 0), image_input_tokens),
    )
    if cached_audio == 0 and cached_image == 0:
        cached_non_text = max(cached_tokens - text_input_tokens, 0)
        if image_input_tokens == 0:
            cached_audio = min(cached_non_text, audio_input_tokens)
        elif audio_input_tokens == 0:
            cached_image = min(cached_non_text, image_input_tokens)
    cached_text = min(max(cached_tokens - cached_audio - cached_image, 0), text_input_tokens)
    audio_output_tokens = clamp_cached_tokens(audio_completion_tokens, completion_tokens)
    video_output_tokens = clamp_cached_tokens(
        video_completion_tokens, max(completion_tokens - audio_output_tokens, 0)
    )

    audio_input_cost = _coalesce(
        c.get("audioInputCost"), c.get("audioCost"), c.get("inputCost"), c.get("cost"),
        tier["audioInput"] * m,
    )  # fmt: skip
    audio_output_cost = _coalesce(
        c.get("audioOutputCost"), c.get("audioCost"), c.get("outputCost"), c.get("cost"),
        _coalesce(catalog(model_cost.get("audioOutput")), catalog_output) * m,
    )  # fmt: skip
    video_output_cost = _coalesce(
        c.get("videoOutputCost"), c.get("outputCost"), c.get("cost"),
        _coalesce(catalog(model_cost.get("videoOutput")), catalog_output) * m,
    )  # fmt: skip
    image_input_cost = _coalesce(
        c.get("imageInputCost"), c.get("inputCost"), c.get("cost"), catalog_image_input * m
    )
    cached_input_cost = _coalesce(c.get("inputCost"), c.get("cost"), tier["cachedInput"] * m)
    cached_audio_input_cost = _coalesce(
        c.get("audioInputCost"), c.get("audioCost"), c.get("inputCost"), c.get("cost"),
        tier["cachedAudioInput"] * m,
    )  # fmt: skip
    cached_image_input_cost = _coalesce(
        c.get("imageInputCost"), c.get("inputCost"), c.get("cost"), tier["cachedImageInput"] * m
    )
    return (
        (text_input_tokens - cached_text) * input_cost
        + cached_text * cached_input_cost
        + (audio_input_tokens - cached_audio) * audio_input_cost
        + cached_audio * cached_audio_input_cost
        + (image_input_tokens - cached_image) * image_input_cost
        + cached_image * cached_image_input_cost
        + (completion_tokens - audio_output_tokens - video_output_tokens) * output_cost
        + audio_output_tokens * audio_output_cost
        + video_output_tokens * video_output_cost
    )


def _get_google_modality_token_count(details: Any, modalities: tuple[str, ...]) -> float:
    if not isinstance(details, list):
        return 0
    total = 0
    for detail in details:
        detail = detail if isinstance(detail, Mapping) else {}
        token_count = _coalesce(detail.get("tokenCount"), detail.get("token_count"))
        if detail.get("modality") in modalities and _is_finite(token_count):
            total += max(token_count, 0)
    return total


_IMAGE_MODALITIES = ("IMAGE", "VIDEO", "DOCUMENT")


def calculate_google_cost_from_usage(
    model_name: str,
    config: Mapping[str, Any],
    prompt_tokens: Any,
    completion_tokens: Any,
    is_vertex_mode: bool,
    usage_metadata: Any,
    response_service_tier: Any = None,
    vertex_region: str | None = None,
    requested_service_tier: Any = None,
    pricing_time: float | None = None,
) -> float | None:
    u = usage_metadata if isinstance(usage_metadata, Mapping) else {}
    prompt_details = _coalesce(u.get("promptTokensDetails"), u.get("prompt_tokens_details"))
    tool_prompt_details = _coalesce(
        u.get("toolUsePromptTokensDetails"), u.get("tool_use_prompt_tokens_details")
    )
    response_details = _coalesce(
        u.get("candidatesTokensDetails"),
        u.get("responseTokensDetails"),
        u.get("candidates_tokens_details"),
        u.get("response_tokens_details"),
    )
    cache_details = _coalesce(u.get("cacheTokensDetails"), u.get("cache_tokens_details"))
    tool_prompt_tokens = _coalesce(
        u.get("toolUsePromptTokenCount"), u.get("tool_use_prompt_token_count"), 0
    )
    prompt_tokens_for_cost = (
        prompt_tokens + max(tool_prompt_tokens, 0)
        if _is_number(prompt_tokens) and _is_finite(tool_prompt_tokens)
        else prompt_tokens
    )
    audio_prompt = _get_google_modality_token_count(
        prompt_details, ("AUDIO",)
    ) + _get_google_modality_token_count(tool_prompt_details, ("AUDIO",))
    image_prompt = _get_google_modality_token_count(
        prompt_details, _IMAGE_MODALITIES
    ) + _get_google_modality_token_count(tool_prompt_details, _IMAGE_MODALITIES)
    return calculate_google_cost(
        model_name,
        config,
        prompt_tokens_for_cost,
        completion_tokens,
        is_vertex_mode,
        audio_prompt,
        _get_google_modality_token_count(response_details, ("AUDIO",)),
        _get_google_modality_token_count(response_details, ("VIDEO",)),
        image_prompt,
        _coalesce(u.get("cachedContentTokenCount"), u.get("cached_content_token_count")),
        _get_google_modality_token_count(cache_details, ("AUDIO",)),
        _get_google_modality_token_count(cache_details, _IMAGE_MODALITIES),
        get_google_response_service_tier(
            {"x-gemini-service-tier": response_service_tier} if response_service_tier else None,
            usage_metadata,
            is_vertex_mode,
        ),
        vertex_region,
        requested_service_tier,
        pricing_time,
    )


# --- message conversion ------------------------------------------------------------------------


def _parse_gemini_format(contents: Any) -> list[dict[str, Any]] | None:
    """zod GeminiFormatSchema.safeParse: an array of {role?: user|model, parts: Part[]}.
    Content objects keep only role/parts (zod strips unknown keys); parts pass through."""
    if not isinstance(contents, list):
        return None
    parsed = []
    for item in contents:
        if not isinstance(item, dict):
            return None
        role = item.get("role", UNDEFINED)
        if role is not UNDEFINED and role not in ("user", "model"):
            return None
        parts = item.get("parts")
        if not isinstance(parts, list):
            return None
        for part in parts:
            if not isinstance(part, dict):
                return None
            if "text" in part and part["text"] is not None and not isinstance(part["text"], str):
                return None
            inline = part.get("inline_data")
            if inline is not None and not (
                isinstance(inline, dict)
                and isinstance(inline.get("mime_type"), str)
                and isinstance(inline.get("data"), str)
            ):
                return None
        parsed.append({**({"role": role} if role is not UNDEFINED else {}), "parts": parts})
    return parsed


def maybe_coerce_to_gemini_format(
    contents: Any, use_assistant_role: bool | None = None
) -> dict[str, Any]:
    """Returns {contents, coerced, systemInstruction}."""
    coerced = False
    parsed = _parse_gemini_format(contents)
    if parsed is not None:
        return {"contents": parsed, "coerced": False, "systemInstruction": None}

    if isinstance(contents, dict) and "system_instruction" in contents:
        return {
            "contents": contents.get("contents", []),
            "coerced": True,
            "systemInstruction": contents["system_instruction"],
        }

    target_role = "assistant" if use_assistant_role else "model"

    def mapped_role(item: Mapping[str, Any]) -> Any:
        return target_role if item.get("role") == "assistant" else item.get("role")

    def with_role(item: Mapping[str, Any], parts: list[Any]) -> dict[str, Any]:
        role = mapped_role(item)
        return {**({"role": role} if role is not None else {}), "parts": parts}

    coerced_contents: list[Any]
    if isinstance(contents, str):
        coerced_contents = [{"parts": [{"text": contents}]}]
        coerced = True
    elif isinstance(contents, list) and all(
        isinstance(item, dict) and isinstance(item.get("content"), str) for item in contents
    ):
        coerced_contents = [with_role(item, [{"text": item["content"]}]) for item in contents]
        coerced = True
    elif isinstance(contents, list) and all(
        isinstance(item, dict) and item.get("role") and item.get("content") for item in contents
    ):
        coerced_contents = []
        for item in contents:
            content = item["content"]
            if isinstance(content, list):
                parts = []
                for content_item in content:
                    if isinstance(content_item, str):
                        parts.append({"text": content_item})
                    elif isinstance(content_item, dict) and content_item.get("type") == "text":
                        parts.append({"text": content_item.get("text")})
                    else:
                        parts.append(content_item)
                coerced_contents.append(with_role(item, parts))
            elif isinstance(content, dict):
                coerced_contents.append(with_role(item, [content]))
            else:
                coerced_contents.append(with_role(item, [{"text": content}]))
        coerced = True
    elif isinstance(contents, dict) and "parts" in contents:
        coerced_contents = [contents]
        coerced = True
    else:
        logger.warning("Unknown format for Gemini: %s", json_stringify(contents))
        return {
            "contents": contents if isinstance(contents, list) else [],
            "coerced": False,
            "systemInstruction": None,
        }

    system_parts: list[Any] = []
    kept = []
    for message in coerced_contents:
        if message.get("role") == "system" and message.get("parts"):
            system_parts.extend(
                p
                for p in message["parts"]
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            )
            continue
        kept.append(message)
    coerced_contents = kept

    # Gemini does not support execution with systemInstruction only.
    if not coerced_contents and system_parts:
        coerced_contents = [{"role": "user", "parts": system_parts}]
        coerced = True
        system_parts = []

    return {
        "contents": coerced_contents,
        "coerced": coerced,
        "systemInstruction": {"parts": system_parts} if system_parts else None,
    }


# --- response parsing --------------------------------------------------------------------------


def _flagged(ratings: list[Mapping[str, Any]]) -> list[str]:
    return [
        f"{r.get('category')}: {r.get('probability')}"
        for r in ratings
        if r.get("probability") != "NEGLIGIBLE"
    ]


def get_candidate(data: Any) -> dict[str, Any]:
    candidates = data.get("candidates") if isinstance(data, dict) else None
    if not candidates:
        error_details = "No candidates returned in API response."
        feedback = (data or {}).get("promptFeedback") if isinstance(data, dict) else None
        feedback = feedback or {}
        if feedback.get("blockReason"):
            error_details = f"Response blocked: {feedback['blockReason']}"
            if feedback.get("safetyRatings"):
                flagged = _flagged(feedback["safetyRatings"])
                if flagged:
                    error_details += f" (Safety ratings: {', '.join(flagged)})"
        elif feedback.get("safetyRatings"):
            flagged = _flagged(feedback["safetyRatings"])
            if flagged:
                error_details = (
                    "Response may have been blocked due to safety filters: " + ", ".join(flagged)
                )
        error_details += f"\n\nGot response: {json_stringify(data)}"
        raise ValueError(error_details)
    if len(candidates) > 1:
        logger.debug(
            "Expected one candidate in AI Studio API response, but got %d", len(candidates)
        )
    candidate = candidates[0]
    if not candidate:
        raise ValueError(
            f"No candidates returned in API response.\n\nGot response: {json_stringify(data)}"
        )
    return candidate


def is_non_candidate_stream_chunk(datum: Mapping[str, Any]) -> bool:
    feedback = datum.get("promptFeedback")
    return not datum.get("candidates") and (
        (bool(datum.get("usageMetadata")) and not feedback)
        or (bool((feedback or {}).get("safetyRatings")) and not (feedback or {}).get("blockReason"))
    )


def get_last_prompt_safety_ratings(data: list[Mapping[str, Any]]) -> Any:
    ratings = None
    for datum in data:
        feedback = datum.get("promptFeedback") or {}
        if feedback.get("safetyRatings"):
            ratings = feedback["safetyRatings"]
    return ratings


def get_google_guardrails(prompt_ratings: Any, candidate_ratings: Any) -> dict[str, Any] | None:
    """Normalize Google's prompt and candidate ratings for guardrails assertions."""
    if not prompt_ratings and not candidate_ratings:
        return None
    flagged_input = (
        None
        if prompt_ratings is None
        else any(r.get("probability") != "NEGLIGIBLE" for r in prompt_ratings)
    )
    flagged_output = (
        None
        if candidate_ratings is None
        else any(r.get("probability") != "NEGLIGIBLE" for r in candidate_ratings)
    )
    result: dict[str, Any] = {}
    if flagged_input is not None:
        result["flaggedInput"] = flagged_input
    if flagged_output is not None:
        result["flaggedOutput"] = flagged_output
    flagged = flagged_input or flagged_output
    if flagged is not None:
        result["flagged"] = flagged
    return result


def collect_thought_signatures(data: list[Mapping[str, Any]]) -> list[str]:
    return [
        part["thoughtSignature"]
        for datum in data
        for candidate in (datum.get("candidates") or [])
        for part in ((candidate.get("content") or {}).get("parts") or [])
        if isinstance(part.get("thoughtSignature"), str)
    ]


_GROUNDING_LIST_KEYS = ("groundingChunks", "groundingSupports", "webSearchQueries")


def collect_grounding_metadata(data: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate grounding signals across every candidate-bearing chunk."""
    candidates = [get_candidate(d) for d in data if d.get("candidates")]
    flat = {
        key: [x for c in candidates for x in (c.get(key) or [])] for key in _GROUNDING_LIST_KEYS
    }
    metas = [c["groundingMetadata"] for c in candidates if c.get("groundingMetadata")]

    grounding_metadata = None
    if len(metas) == 1:
        grounding_metadata = metas[0]
    elif len(metas) > 1:
        merged: dict[str, Any] = {}
        for meta in metas:
            for key, value in meta.items():
                if key in _GROUNDING_LIST_KEYS:
                    continue
                if value is not None:
                    merged[key] = value
        for key in _GROUNDING_LIST_KEYS:
            inner = [x for meta in metas for x in (meta.get(key) or [])]
            if inner:
                merged[key] = inner
        grounding_metadata = merged

    return {
        **({"groundingMetadata": grounding_metadata} if grounding_metadata else {}),
        **{key: value for key, value in flat.items() if value},
    }


BLOCKED_FINISH_REASONS = ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII")


def format_candidate_contents(candidate: Mapping[str, Any]) -> Any:
    finish_reason = candidate.get("finishReason")
    if finish_reason and finish_reason in BLOCKED_FINISH_REASONS:
        message = f"Response was blocked with finish reason: {finish_reason}"
        ratings = candidate.get("safetyRatings")
        if ratings:
            flagged = [
                f"{r.get('category')}: {r.get('probability')}"
                + (" (BLOCKED)" if r.get("blocked") else "")
                for r in ratings
                if r.get("probability") != "NEGLIGIBLE" or r.get("blocked")
            ]
            if flagged:
                message += f"\nSafety ratings: {', '.join(flagged)}"
        if finish_reason == "RECITATION":
            message += (
                "\n\nThis typically occurs when the response is too similar to content from the "
                "model's training data."
            )
        elif finish_reason == "SAFETY":
            message += (
                "\n\nThe response was blocked due to safety filters. Consider adjusting safety "
                "settings or modifying your prompt."
            )
        raise ValueError(message)

    parts = (candidate.get("content") or {}).get("parts")
    if parts:
        output = ""
        is_text = True
        for part in parts:
            if "text" in part:
                output += _js_concat(part["text"])
            else:
                is_text = False
        return output if is_text else parts
    if parts is not None:
        # An empty parts array is truthy in JS and concatenates to ''.
        return ""
    raise ValueError(f"No output found in response: {json_stringify(candidate)}")


def _js_concat(value: Any) -> str:
    from promptphoo.jsvalues import js_string

    return value if isinstance(value, str) else js_string(value)


def merge_parts(parts1: Any, parts2: Any) -> Any:
    if parts1 is None:
        return parts2 if isinstance(parts2, str) else list(parts2)
    if isinstance(parts1, str) and isinstance(parts2, str):
        return parts1 + parts2
    array1 = [{"text": parts1}] if isinstance(parts1, str) else parts1
    array2 = [{"text": parts2}] if isinstance(parts2, str) else parts2
    array1.extend(array2)
    return array1


def normalize_gemini_audio(output: Any) -> dict[str, Any] | None:
    if not isinstance(output, list):
        return None
    audio_parts = [
        p
        for p in output
        if isinstance(p, dict)
        and str(((p.get("inlineData") or {}).get("mimeType")) or "").startswith("audio/")
    ]
    if not audio_parts:
        return None
    mime_type = audio_parts[0]["inlineData"]["mimeType"]
    audio = b"".join(base64.b64decode(p["inlineData"].get("data") or "") for p in audio_parts)
    if not re.match(r"^audio/(?:L16|pcm)(?:;|$)", mime_type, re.IGNORECASE):
        return {
            "data": base64.b64encode(audio).decode(),
            "format": re.split(r"[;/]", mime_type)[1],
        }
    rate = re.search(r"(?:^|;)\s*rate=(\d+)", mime_type, re.IGNORECASE)
    sample_rate = int(rate.group(1)) if rate else 24_000
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(audio))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(audio))
    )
    return {
        "data": base64.b64encode(header + audio).decode(),
        "format": "wav",
        "sampleRate": sample_rate,
        "channels": 1,
    }


# --- tools -------------------------------------------------------------------------------------


def normalize_tools(tools: list[Any]) -> list[Any]:
    """Canonicalize snake_case aliases and sanitize function declaration schemas."""
    canonical_names = {
        d.get("name")
        for tool in tools
        if isinstance(tool, dict)
        for d in (tool.get("functionDeclarations") or [])
    }
    seen: set[Any] = set()
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            result.append(tool)
            continue
        normalized = {
            k: v
            for k, v in tool.items()
            if k not in ("code_execution", "google_search", "google_search_retrieval")
        }
        for snake, camel in (
            ("code_execution", "codeExecution"),
            ("google_search", "googleSearch"),
            ("google_search_retrieval", "googleSearchRetrieval"),
        ):
            if normalized.get(camel) is None and tool.get(snake) is not None:
                normalized[camel] = tool[snake]

        canonical = tool.get("functionDeclarations")
        legacy = tool.get("function_declarations")
        if canonical or legacy:
            merged = [
                *(canonical or []),
                *(d for d in (legacy or []) if d.get("name") not in canonical_names),
            ]
            declarations = []
            for d in merged:
                if d.get("name") in seen:
                    continue
                seen.add(d.get("name"))
                declarations.append(d)
            normalized["functionDeclarations"] = declarations
            normalized.pop("function_declarations", None)
            if not declarations and (canonical or legacy):
                del normalized["functionDeclarations"]
                if not normalized:
                    continue

        if normalized.get("functionDeclarations"):
            normalized["functionDeclarations"] = [
                {
                    **fd,
                    "parameters": sanitize_schema_for_gemini(fd["parameters"])
                    if fd.get("parameters")
                    else UNDEFINED,
                }
                for fd in normalized["functionDeclarations"]
            ]
            for fd in normalized["functionDeclarations"]:
                if fd["parameters"] is UNDEFINED:
                    del fd["parameters"]
        result.append(normalized)
    return result


def load_file(config_var: Any, context_vars: Mapping[str, Any] | None) -> Any:
    """Load tools/instructions given inline, as file:// references or via rendered vars."""
    from promptphoo.files import maybe_load_from_external_file
    from promptphoo.jsonutil import js_json_parse

    file_contents = maybe_load_from_external_file(_render_vars(config_var, context_vars))
    if isinstance(file_contents, str):
        try:
            parsed = js_json_parse(file_contents)
        except ValueError as err:
            logger.debug("ERROR: failed to convert file contents to JSON: %s", err)
            return file_contents
        return normalize_tools(parsed) if isinstance(parsed, list) else parsed
    if isinstance(file_contents, list):
        return normalize_tools(file_contents)
    return file_contents


# --- inline media detection --------------------------------------------------------------------

_ASF_HEADER_GUID = bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c")
_ASF_STREAM_PROPERTIES_GUID = bytes.fromhex("9107dcb7b7a9cf118ee600c00c205365")
_ASF_AUDIO_STREAM_GUID = bytes.fromhex("409e69f84d5bcf11a8fd00805f5c442b")
_ASF_VIDEO_STREAM_GUID = bytes.fromhex("c0ef19bc4d5bcf11a8fd00805f5c442b")
_EBML_DOCTYPE_ID = bytes([0x42, 0x82])
_MAX_MEDIA_SNIFF_BYTES = 65_536
SUPPORTED_INLINE_MEDIA_MIME_TYPES = frozenset([
    "application/pdf", "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif",
    "audio/wav", "audio/mpeg", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac", "audio/mp4",
    "video/mp4", "video/mpeg", "video/quicktime", "video/avi", "video/x-flv", "video/webm",
    "video/wmv", "video/3gpp",
])  # fmt: skip


def _latin1(data: bytes) -> str:
    return data.decode("latin-1")


def _mime_from_ftyp_brand(brand: str) -> str | None:
    if brand in ("avif", "avis", "crx "):
        return None
    if brand in ("M4A ", "M4B ", "M4P ", "F4A ", "F4B "):
        return "audio/mp4"
    if brand in ("heic", "heix", "hevc", "hevx"):
        return "image/heic"
    if brand in ("mif1", "msf1"):
        return "image/heif"
    if brand == "qt  ":
        return "video/quicktime"
    if brand.startswith("3g"):
        return "video/3gpp"
    return "video/mp4"


def _asf_mime_type(data: bytes) -> str | None:
    offset = data.find(_ASF_STREAM_PROPERTIES_GUID, len(_ASF_HEADER_GUID))
    has_audio = False
    while offset != -1:
        type_offset = offset + len(_ASF_STREAM_PROPERTIES_GUID) + 8
        stream_type = data[type_offset : type_offset + 16]
        if stream_type == _ASF_VIDEO_STREAM_GUID:
            return "video/wmv"
        if stream_type == _ASF_AUDIO_STREAM_GUID:
            has_audio = True
        offset = data.find(_ASF_STREAM_PROPERTIES_GUID, type_offset)
    return "audio/x-ms-wma" if has_audio else None


def _ebml_mime_type(data: bytes) -> str | None:
    doctype_offset = data.find(_EBML_DOCTYPE_ID, 4)
    if doctype_offset < 0:
        return None
    size_offset = doctype_offset + len(_EBML_DOCTYPE_ID)
    if size_offset >= len(data) or data[size_offset] == 0:
        return None
    encoded = data[size_offset]
    width, marker = 1, 0x80
    while encoded & marker == 0:
        marker >>= 1
        width += 1
    if size_offset + width > len(data):
        return None
    length = encoded & (marker - 1)
    for index in range(1, width):
        length = length * 256 + data[size_offset + index]
        if length > 4:
            return None
    start = size_offset + width
    if length != 4 or start + length > len(data):
        return None
    return "video/webm" if data[start : start + length] == b"webm" else None


def _ogg_mime_type(data: bytes) -> str | None:
    offset = 0
    has_audio = False
    while offset + 27 <= len(data):
        if data[offset : offset + 4] != b"OggS" or data[offset + 4] != 0:
            return None
        flags = data[offset + 5]
        if flags & 2 == 0:
            return "audio/ogg" if has_audio else None
        segment_count = data[offset + 26]
        payload_offset = offset + 27 + segment_count
        if flags & 1 != 0 or segment_count == 0 or payload_offset > len(data):
            return None
        segments = data[offset + 27 : payload_offset]
        packet_end = next((i for i, length in enumerate(segments) if length < 255), -1)
        if packet_end < 0:
            return None
        packet_length = sum(segments[: packet_end + 1])
        page_end = payload_offset + sum(segments)
        if page_end > len(data):
            return None
        packet = data[payload_offset : payload_offset + packet_length]
        if len(packet) >= 64 and packet[:8] == b"fishead\0":
            offset = page_end
            continue
        if packet[:7] == b"\x80theora":
            return "video/ogg"
        if not any(packet.startswith(h) for h in (b"OpusHead", b"\x01vorbis", b"Speex   ",
                                                    b"\x7fFLAC")):  # fmt: skip
            return None
        has_audio = True
        offset = page_end
    return "audio/ogg" if offset == len(data) and has_audio else None


def _byte(data: bytes, index: int) -> int:
    return data[index] if index < len(data) else -1


def _mime_from_media_bytes(data: bytes) -> str | None:
    riff_type = _latin1(data[8:12])
    if data[:4] == b"RIFF":
        return {"WEBP": "image/webp", "WAVE": "audio/wav", "AVI ": "video/avi"}.get(riff_type)
    if data[:4] == b"FORM" and riff_type in ("AIFF", "AIFC"):
        return "audio/aiff"
    if data[4:8] == b"ftyp":
        return _mime_from_ftyp_brand(_latin1(data[8:12]))
    if data[4:8] in (b"moov", b"mdat", b"wide"):
        return "video/quicktime"
    if data[:4].hex() == "1a45dfa3":
        return _ebml_mime_type(data)
    if data[:4].hex() in ("000001ba", "000001b3"):
        return "video/mpeg"
    if data[:3] == b"FLV":
        return "video/x-flv"
    if data[: len(_ASF_HEADER_GUID)] == _ASF_HEADER_GUID:
        return _asf_mime_type(data)
    b0, b1 = _byte(data, 0), _byte(data, 1)
    if b0 == 0xFF and b1 >= 0 and (b1 & 0xF6) == 0xF0:
        return "audio/aac"
    if data[:3] == b"ID3" or (b0 == 0xFF and b1 >= 0 and (b1 & 0xE0) == 0xE0):
        return "audio/mpeg"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if data[:4] == b"OggS":
        return _ogg_mime_type(data)
    return None


def is_data_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("data:") and len(value) > 5


def parse_data_url(value: str) -> dict[str, str] | None:
    if not is_data_url(value):
        return None
    match = re.match(r"^data:([^;,]+)(?:;[^,]*)?;base64,(.+)$", value, re.DOTALL)
    if not match:
        return None
    return {"mimeType": match.group(1).strip(), "base64Data": match.group(2).strip()}


def extract_base64_from_data_url(value: str) -> str:
    parsed = parse_data_url(value)
    return parsed["base64Data"] if parsed else value


_MIME_ALIASES = {
    "audio/mp3": "audio/mpeg",
    "audio/m4a": "audio/mp4",
    "video/mpg": "video/mpeg",
    "video/mov": "video/quicktime",
}


def _b64decode_loose(data: str) -> bytes:
    """Buffer.from(data, 'base64'): tolerant of missing padding."""
    data = data.rstrip("=")
    data += "=" * (-len(data) % 4)
    try:
        return base64.b64decode(data)
    except ValueError:
        return base64.b64decode(data[: len(data) // 4 * 4] or "")


def get_mime_type_from_base64(data: str) -> str | None:
    parsed = parse_data_url(data)
    base64_data = parsed["base64Data"] if parsed else data
    if (
        not base64_data
        or len(base64_data) < 20
        or not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", base64_data)
    ):
        return None
    if parsed:
        mime_type = parsed["mimeType"].lower()
        mime_type = _MIME_ALIASES.get(mime_type, mime_type)
        return mime_type if mime_type in SUPPORTED_INLINE_MEDIA_MIME_TYPES else None
    if base64_data.startswith("/9j/"):
        return "image/jpeg"
    if base64_data.startswith("iVBORw0KGgo"):
        return "image/png"
    if base64_data.startswith("JVBER"):
        return "application/pdf"
    sniff_length = math.ceil(_MAX_MEDIA_SNIFF_BYTES / 3) * 4
    inferred = _mime_from_media_bytes(_b64decode_loose(base64_data[:sniff_length]))
    return inferred if inferred and inferred in SUPPORTED_INLINE_MEDIA_MIME_TYPES else None


def process_images_in_contents(contents: Any, context_vars: Mapping[str, Any] | None) -> Any:
    """Split text parts whose lines are base64 media from test vars into inlineData parts."""
    if not context_vars:
        return contents
    if not isinstance(contents, list):
        logger.warning("[Google] contents is not an array in processImagesInContents")
        return []
    base64_to_mime: dict[str, str] = {}
    for value in context_vars.values():
        if isinstance(value, str):
            mime_type = get_mime_type_from_base64(value)
            if mime_type and (mime_type != "video/mp4" or base64_to_mime.get(value) != "audio/mp4"):
                base64_to_mime[value] = mime_type

    result = []
    for content in contents:
        if not isinstance(content, dict) or not content.get("parts"):
            result.append(content)
            continue
        new_parts: list[Any] = []
        for part in content["parts"]:
            text = part.get("text") if isinstance(part, dict) else None
            if not text:
                new_parts.append(part)
                continue
            found = False
            block = ""
            processed: list[Any] = []
            for line in text.split("\n"):
                trimmed = line.strip()
                mime_type = base64_to_mime.get(trimmed)
                if mime_type:
                    found = True
                    if block:
                        processed.append({"text": block})
                        block = ""
                    processed.append({
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": extract_base64_from_data_url(trimmed)
                            if is_data_url(trimmed)
                            else trimmed,
                        }
                    })  # fmt: skip
                else:
                    if block:
                        block += "\n"
                    block += line
            if block:
                processed.append({"text": block})
            new_parts.extend(processed if found else [part])
        result.append({**content, "parts": new_parts})
    return result


# --- system instruction / response schema ------------------------------------------------------


def resolve_google_config_file_reference(reference: str, base_path: str | None = None) -> str:
    """Resolve only relative Google config file references against their owning base path."""
    from promptphoo.files import parse_file_url

    if not base_path or not reference.startswith("file://"):
        return reference
    file_path, _ = parse_file_url(reference)
    is_absolute = os.path.isabs(file_path) or ntpath.isabs(file_path)
    resolved = file_path if is_absolute else os.path.abspath(os.path.join(base_path, file_path))
    return f"file://{resolved}"


def parse_config_system_instruction(
    config_system_instruction: Any,
    context_vars: Mapping[str, Any] | None = None,
    base_path: str | None = None,
) -> Any:
    """Load, convert and render config-level systemInstruction."""
    if not config_system_instruction:
        return None
    instruction = copy.deepcopy(config_system_instruction)
    render_parts = True
    if isinstance(config_system_instruction, str):
        rendered = _render_vars(config_system_instruction, context_vars)
        reference = resolve_google_config_file_reference(rendered, base_path)
        instruction = load_file(reference, None)
        render_parts = reference.startswith("file://")
    if isinstance(instruction, str):
        instruction = {"parts": [{"text": instruction}]}
    if render_parts and context_vars and instruction:
        for part in instruction.get("parts") or []:
            if part.get("text"):
                try:
                    part["text"] = _render_template(part["text"], context_vars)
                except Exception as err:
                    raise ValueError(
                        f"Unable to render nunjucks in systemInstruction: Error: {err}"
                    ) from err
    return instruction


def parse_config_response_schema(
    config_response_schema: Any,
    context_vars: Mapping[str, Any] | None = None,
    base_path: str | None = None,
) -> Any:
    from promptphoo.files import maybe_load_from_external_file
    from promptphoo.jsonutil import js_json_parse

    rendered = _render_vars(config_response_schema, context_vars)
    reference = (
        resolve_google_config_file_reference(rendered, base_path)
        if isinstance(rendered, str)
        else rendered
    )
    schema = maybe_load_from_external_file(reference)
    if isinstance(schema, str):
        try:
            schema = js_json_parse(schema)
        except ValueError as error:
            raise ValueError(f"Invalid JSON in responseSchema: SyntaxError: {error}") from error
    if isinstance(reference, str) and reference.startswith("file://"):
        return _render_vars(schema, context_vars)
    return schema


def gemini_format_and_system_instructions(
    prompt: str,
    context_vars: Mapping[str, Any] | None = None,
    config_system_instruction: Any = None,
    base_path: str | None = None,
    use_assistant_role: bool | None = None,
) -> dict[str, Any]:
    """Returns {contents, systemInstruction}."""
    contents = parse_chat_prompt(prompt, [{"parts": [{"text": prompt}], "role": "user"}])
    coerced = maybe_coerce_to_gemini_format(contents, use_assistant_role)
    if coerced["coerced"]:
        logger.debug("Coerced JSON prompt to Gemini format: %s", json_stringify(contents))
    # Node keeps the zod-parsed copy only when coerced; the parsed copy differs from the input
    # only by stripped unknown content keys, which Gemini rejects anyway.
    contents = coerced["contents"] if coerced["coerced"] else contents
    system_instruction = coerced["systemInstruction"]

    parsed_config = parse_config_system_instruction(
        config_system_instruction, context_vars, base_path
    )
    if parsed_config:
        system_instruction = (
            {"parts": [*parsed_config["parts"], *system_instruction["parts"]]}
            if system_instruction
            else parsed_config
        )
    contents = process_images_in_contents(contents, context_vars)
    return {"contents": contents, "systemInstruction": system_instruction}


# --- function call validation ------------------------------------------------------------------


def _normalize_schema_types(node: Any) -> Any:
    if isinstance(node, list):
        return [_normalize_schema_types(v) for v in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "type":
            if isinstance(value, str) and value in VALID_SCHEMA_TYPES:
                out[key] = value.lower()
            elif isinstance(value, list):
                out[key] = [
                    t.lower() if isinstance(t, str) and t in VALID_SCHEMA_TYPES else t
                    for t in value
                ]
            else:
                out[key] = _normalize_schema_types(value)
        else:
            out[key] = _normalize_schema_types(value)
    return out


def parse_string_object(value: Any) -> Any:
    from promptphoo.jsonutil import js_json_parse

    return js_json_parse(value) if isinstance(value, str) else value


def validate_function_call(output: Any, functions: Any = None, vars: Any = None) -> None:
    from promptphoo.ajv import compile_schema
    from promptphoo.assertions.tools import InvalidToolCallError, InvalidToolSchemaError

    try:
        parsed = parse_string_object(output)
        if isinstance(parsed, dict) and "toolCall" in parsed:
            function_calls = (parsed["toolCall"] or {}).get("functionCalls")
        elif isinstance(parsed, list):
            function_calls = [
                obj["functionCall"]
                for obj in parsed
                if isinstance(obj, dict) and "functionCall" in obj
            ]
        else:
            raise ValueError("Unrecognized function call format")
        if (
            not isinstance(function_calls, list)
            or not function_calls
            or any(
                not isinstance(c, dict) or not isinstance(c.get("name"), str)
                for c in function_calls
            )
        ):
            raise ValueError("Expected at least one function call")
    except Exception as err:
        raise InvalidToolCallError(
            f"Google did not return a valid-looking function call: {json_stringify(output)}"
        ) from err

    try:
        interpolated = load_file(functions, vars)
    except Exception as err:
        raise InvalidToolSchemaError(str(err)) from err
    if interpolated is not None and not isinstance(interpolated, list):
        raise InvalidToolSchemaError("Expected the loaded functions to be an array")
    if interpolated and any(
        not isinstance(tool, dict)
        or (
            tool.get("functionDeclarations") is not None
            and (
                not isinstance(tool["functionDeclarations"], list)
                or any(
                    not isinstance(d, dict)
                    or not isinstance(d.get("name"), str)
                    or not d["name"].strip()
                    for d in tool["functionDeclarations"]
                )
            )
        )
        for tool in interpolated
    ):
        raise InvalidToolSchemaError("Expected valid function declarations in each tool")
    declarations = [
        d for tool in (interpolated or []) for d in (tool.get("functionDeclarations") or [])
    ]
    if not declarations:
        raise InvalidToolSchemaError(
            f'Called "{function_calls[0]["name"]}", but there is no function with that name'
        )

    for call in function_calls:
        name = call["name"]
        try:
            args = (
                {}
                if call.get("args", UNDEFINED) is UNDEFINED
                else parse_string_object(call["args"])
            )
        except Exception as err:
            raise InvalidToolCallError(str(err)) from err
        schema = next((d for d in declarations if d.get("name") == name), None)
        if not schema:
            raise InvalidToolCallError(f'Called "{name}", but there is no function with that name')
        if schema.get("parameters"):
            parameter_schema = _normalize_schema_types(schema["parameters"])
            try:
                validate = compile_schema(parameter_schema)
            except Exception as err:
                raise InvalidToolSchemaError(
                    f"Tool schema doesn't compile with ajv: Error: {err}. If this is a valid tool "
                    "schema you may need to reformulate your assertion without "
                    "is-valid-function-call."
                ) from err
            if not validate(args):
                raise InvalidToolCallError(
                    f'Call to "{name}":\n{json_stringify(call)}\ndoes not match schema:\n'
                    f"{json_stringify(validate.errors)}"
                )
        elif json_stringify(args) != "{}":
            raise InvalidToolCallError(
                f'Call to "{name}":\n{json_stringify(call)}\ndoes not match schema:\n'
                f"{json_stringify(schema)}"
            )


# --- schema sanitizing / cache discriminator ---------------------------------------------------

GEMINI_SUPPORTED_SCHEMA_PROPERTIES = frozenset([
    "type", "format", "description", "nullable", "enum", "maxItems", "minItems", "properties",
    "required", "propertyOrdering", "items",
])  # fmt: skip

JSON_SCHEMA_TYPE_MAP = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
    "null": "STRING",  # Gemini doesn't support null type, fall back to STRING
}


def sanitize_schema_for_gemini(schema: Any) -> Any:
    """Drop unsupported JSON Schema properties and uppercase types, recursively."""
    if not isinstance(schema, dict):
        return schema
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in GEMINI_SUPPORTED_SCHEMA_PROPERTIES:
            continue
        if key == "type":
            result[key] = (
                JSON_SCHEMA_TYPE_MAP.get(value.lower()) or value.upper()
                if isinstance(value, str)
                else value
            )
        elif key == "properties" and isinstance(value, dict):
            result[key] = {
                name: sanitize_schema_for_gemini(prop) if isinstance(prop, dict) else prop
                for name, prop in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            result[key] = sanitize_schema_for_gemini(value)
        else:
            result[key] = value
    return result


_AUTH_HEADER_NAMES = ("authorization", "x-goog-api-key", "x-api-key", "api-key",
                      "x-goog-user-project")  # fmt: skip


def create_auth_cache_discriminator(headers: Mapping[str, str]) -> str:
    """A short hash of auth headers so different credentials don't share cached responses."""
    values = [f"{name}:{headers[name]}" for name in _AUTH_HEADER_NAMES if headers.get(name)]
    if not values:
        return ""
    return hashlib.sha256("|".join(values).encode()).hexdigest()[:16]
