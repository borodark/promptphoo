"""Azure helpers: configuration errors and cost. Ports src/providers/azure/util.ts.

The pricing tables live in `tables.py`, generated from upstream by tools/gen_azure_tables.mjs.
"""

from __future__ import annotations

import math
from typing import Any, NoReturn

from promptphoo.providers.azure.tables import (
    AZURE_CACHE_READ_RATES,
    AZURE_LONG_CONTEXT_CACHE_READ_RATES,
    AZURE_MODELS,
    AZURE_PRIORITY_MULTIPLIERS,
    AZURE_SHORT_CONTEXT_PRIORITY_MULTIPLIERS,
)
from promptphoo.providers.shared import clamp_cached_tokens

_MODELS_BY_ID: dict[str, dict[str, Any]] = {}
for _model in AZURE_MODELS:
    _MODELS_BY_ID.setdefault(_model["id"], _model)


class AzureConfigurationError(ValueError):
    """`throwConfigurationError`: a plain Error whose message ends with the docs link."""


def throw_configuration_error(message: str) -> NoReturn:
    raise AzureConfigurationError(
        f"{message}\n\n"
        "See https://www.promptfoo.dev/docs/providers/azure/ to learn more about Azure "
        "configuration."
    )


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def calculate_azure_cost(
    model_name: str,
    config: dict[str, Any] | None,
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    cached_prompt_tokens: Any = None,
    audio_prompt_tokens: Any = None,
    audio_completion_tokens: Any = None,
    image_prompt_tokens: Any = None,
    cached_audio_prompt_tokens: Any = None,
    cached_image_prompt_tokens: Any = None,
    image_completion_tokens: Any = None,
) -> float | None:
    """Cost from the Azure price table, or None when the model or token counts are unknown."""
    if not _is_finite_number(prompt_tokens) or not _is_finite_number(completion_tokens):
        return None
    model = _MODELS_BY_ID.get(model_name)
    if not model:
        return None
    cost: dict[str, Any] = model["cost"]
    long_context_cost = cost.get("longContext")
    long_context = (
        long_context_cost
        if long_context_cost and prompt_tokens > long_context_cost["threshold"]
        else None
    )
    input_cost = _first((long_context or {}).get("input"), cost["input"])
    output_cost = _first((long_context or {}).get("output"), cost["output"])
    default_cache_read_cost = _first(
        cost.get("cacheRead"), AZURE_CACHE_READ_RATES.get(model_name), input_cost
    )
    cache_read_cost = _first(
        (long_context or {}).get("cacheRead"),
        AZURE_LONG_CONTEXT_CACHE_READ_RATES.get(model_name) if long_context else None,
        default_cache_read_cost,
    )
    cached_tokens = clamp_cached_tokens(cached_prompt_tokens, prompt_tokens)
    audio_input_tokens = clamp_cached_tokens(audio_prompt_tokens, prompt_tokens)
    image_input_tokens = clamp_cached_tokens(
        image_prompt_tokens, max(prompt_tokens - audio_input_tokens, 0)
    )
    text_input_tokens = max(prompt_tokens - audio_input_tokens - image_input_tokens, 0)
    cached_audio_tokens = clamp_cached_tokens(
        cached_audio_prompt_tokens, min(cached_tokens, audio_input_tokens)
    )
    cached_image_tokens = clamp_cached_tokens(
        cached_image_prompt_tokens,
        min(max(cached_tokens - cached_audio_tokens, 0), image_input_tokens),
    )
    if cached_audio_tokens == 0 and cached_image_tokens == 0:
        cached_non_text_tokens = max(cached_tokens - text_input_tokens, 0)
        if image_input_tokens == 0:
            cached_audio_tokens = min(cached_non_text_tokens, audio_input_tokens)
        elif audio_input_tokens == 0:
            cached_image_tokens = min(cached_non_text_tokens, image_input_tokens)
    cached_text_tokens = min(
        max(cached_tokens - cached_audio_tokens - cached_image_tokens, 0), text_input_tokens
    )
    audio_output_tokens = clamp_cached_tokens(audio_completion_tokens, completion_tokens)
    image_output_tokens = clamp_cached_tokens(
        image_completion_tokens, max(completion_tokens - audio_output_tokens, 0)
    )
    # Cost follows what is sent on the wire: Azure chat/completion only send
    # `passthrough.service_tier`.
    passthrough = (config or {}).get("passthrough")
    service_tier = passthrough.get("service_tier") if isinstance(passthrough, dict) else None
    priority_multiplier = (
        _first(
            cost.get("priorityMultiplier"),
            None if long_context else AZURE_SHORT_CONTEXT_PRIORITY_MULTIPLIERS.get(model_name),
            AZURE_PRIORITY_MULTIPLIERS.get(model_name),
            1,
        )
        if service_tier == "priority"
        else 1
    )
    return (
        (text_input_tokens - cached_text_tokens) * input_cost
        + cached_text_tokens * cache_read_cost
        + (audio_input_tokens - cached_audio_tokens) * _first(cost.get("audioInput"), input_cost)
        + cached_audio_tokens * _first(cost.get("cacheReadAudio"), cache_read_cost)
        + (image_input_tokens - cached_image_tokens) * _first(cost.get("imageInput"), input_cost)
        + cached_image_tokens * _first(cost.get("cacheReadImage"), cache_read_cost)
        + (completion_tokens - audio_output_tokens - image_output_tokens) * output_cost
        + audio_output_tokens * _first(cost.get("audioOutput"), output_cost)
        + image_output_tokens * _first(cost.get("imageOutput"), output_cost)
    ) * priority_multiplier
