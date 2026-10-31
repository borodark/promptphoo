"""OpenAI usage-ledger billing: processing tiers, cached/cache-write input, audio and image
modalities. Ports src/providers/openai/billing.ts (rate tables are in `tables.py`).
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from promptphoo.jsvalues import is_truthy
from promptphoo.providers.openai import tables
from promptphoo.providers.openai.gpt6 import get_gpt6_variant, is_gpt6_model
from promptphoo.providers.openai.urls import url_hostname, url_pathname
from promptphoo.providers.openai.util import (
    GPT_LONG_CONTEXT_THRESHOLD,
    OPENAI_BILLING_MODELS,
    OPENAI_DAYBREAK_ALIASES,
    get_openai_cache_write_input_tokens,
    is_azure_openai_endpoint,
)

OpenAIProcessingTier = Literal["standard", "batch", "flex", "fast", "priority", "ultrafast"]

PER_MILLION = 1 / 1_000_000


def per_million(value: float) -> float:
    return value * PER_MILLION


_STANDARD_CACHED_INPUT_RATES: dict[str, float] = tables.STANDARD_CACHED_INPUT_RATES
_FINE_TUNED_TEXT_RATES: dict[str, dict[str, float]] = tables.FINE_TUNED_TEXT_RATES
_FINE_TUNED_BATCH_OVERRIDES: dict[str, dict[str, float]] = tables.FINE_TUNED_BATCH_OVERRIDES
_LONG_CONTEXT_CACHED_INPUT_RATES: dict[str, float] = tables.LONG_CONTEXT_CACHED_INPUT_RATES
_FLEX_SUPPORTED_TEXT_MODELS: frozenset[str] = tables.FLEX_SUPPORTED_TEXT_MODELS
_FAST_TEXT_RATES: dict[str, dict[str, float]] = tables.FAST_TEXT_RATES
_IMAGE_MODEL_RATES: dict[str, dict[str, Any]] = tables.IMAGE_MODEL_RATES
_REALTIME_MODAL_RATES: dict[str, dict[str, Any]] = tables.REALTIME_MODAL_RATES
_EMBEDDING_RATES: dict[str, dict[str, float]] = tables.EMBEDDING_RATES
_TEXT_MODELS_BY_ID: dict[str, dict[str, Any]] = {}
for _model in OPENAI_BILLING_MODELS:
    _TEXT_MODELS_BY_ID[_model["id"]] = _model  # Map semantics: the last duplicate wins.
_GPT_5_6_MODELS: frozenset[str] = tables.GPT_5_6_MODELS
_BEDROCK_MANTLE_TEXT_RATES: dict[str, dict[str, float]] = tables.BEDROCK_MANTLE_TEXT_RATES
_CACHE_WRITE_MODELS: frozenset[str] = tables.CACHE_WRITE_MODELS
_OPENAI_REGIONAL_PROCESSING_MODEL = re.compile(
    r"^(?:gpt-5\.[456]|gpt-6-(?:astra|sol|luna)|gpt-6\.1-sol)(?:-|$)"
)
OPENAI_REGIONAL_PROCESSING_MULTIPLIER = 1.1
_OPENAI_REGIONAL_PROCESSING_HOSTNAMES = frozenset(["us.api.openai.com", "eu.api.openai.com"])
# AWS's GPT-5.6 Terra/Luna model cards price GovCloud 20% above commercial In-Region rates.
_BEDROCK_GOVCLOUD_MODELS: frozenset[str] = tables.BEDROCK_GOVCLOUD_MODELS
_BEDROCK_GOVCLOUD_MULTIPLIER = 1.2
_BEDROCK_GOVCLOUD_REGION = re.compile(r"^us-gov-(?:east|west)-1$")
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)


def _first(*values: Any) -> Any:
    """JS `a ?? b ?? c`."""
    return next((v for v in values if v is not None), None)


def _uses_bedrock_govcloud_pricing(
    model_name: str,
    config: dict[str, Any],
    api_url: str | None,
    provider: str | None,
    region: str | None,
) -> bool:
    if model_name not in _BEDROCK_GOVCLOUD_MODELS:
        return False
    hostname = url_hostname(api_url or config.get("apiBaseUrl") or "")
    if hostname is not None:
        endpoint = re.match(
            r"^bedrock-(mantle|runtime(?:-fips)?)\.([a-z0-9-]+)\.(?:amazonaws\.com|api\.aws)$",
            hostname,
        )
        if endpoint:
            return endpoint.group(1) == "mantle" and bool(
                _BEDROCK_GOVCLOUD_REGION.match(endpoint.group(2))
            )
    # The resolved region still identifies the target behind a custom proxy.
    return provider == "bedrock" and bool(
        _BEDROCK_GOVCLOUD_REGION.match(region or config.get("region") or "")
    )


def _apply_regional_processing_rates(
    model_name: str, model_rates: dict[str, Any], config: dict[str, Any], options: dict[str, Any]
) -> dict[str, Any]:
    if not _OPENAI_REGIONAL_PROCESSING_MODEL.search(model_name) or not (
        options.get("regionalProcessing")
        or _uses_openai_regional_processing(config, options.get("apiUrl"))
    ):
        return model_rates
    multiplier = OPENAI_REGIONAL_PROCESSING_MULTIPLIER * (
        _BEDROCK_GOVCLOUD_MULTIPLIER
        if _uses_bedrock_govcloud_pricing(
            model_name,
            config,
            options.get("apiUrl"),
            options.get("provider"),
            options.get("region"),
        )
        else 1
    )
    return {**model_rates, "text": _apply_rate_multiplier(model_rates["text"], multiplier)}


def uses_azure_openai_billing(
    config: dict[str, Any], resolved_api_url: str | None, provider: str | None
) -> bool:
    if provider in ("azure", "azure-openai"):
        return True
    endpoint = resolved_api_url or config.get("apiHost") or config.get("apiBaseUrl")
    if not endpoint:
        return False
    if is_azure_openai_endpoint(endpoint):
        return True
    url = endpoint if _SCHEME.match(endpoint) else f"https://{endpoint}"
    hostname = url_hostname(url)
    pathname = url_pathname(url)
    return (
        hostname == "gateway.ai.cloudflare.com"
        and pathname is not None
        and re.match(r"^/v1/[^/]+/[^/]+/azure-openai(?:/|$)", pathname) is not None
    )


def _uses_openai_regional_processing(config: dict[str, Any], resolved_api_url: str | None) -> bool:
    endpoint = resolved_api_url or config.get("apiHost") or config.get("apiBaseUrl")
    if not endpoint:
        return False
    hostname = url_hostname(endpoint if _SCHEME.match(endpoint) else f"https://{endpoint}")
    if hostname is None:
        return False
    hostname = hostname.lower()
    return hostname in _OPENAI_REGIONAL_PROCESSING_HOSTNAMES or bool(
        re.match(r"^bedrock-mantle\.[a-z0-9-]+\.api\.aws$", hostname)
    )


def _get_bedrock_mantle_text_rates(
    model_name: str,
    config: dict[str, Any],
    options: dict[str, Any],
    tier: str,
    total_input_tokens: float,
) -> dict[str, float] | None:
    has_marker = model_name.startswith("bedrock:")
    billing_model = re.sub(r"^bedrock:", "", model_name)
    if tier != "standard":
        # The Mantle rates in this table cover Standard processing only.
        return None
    endpoint = _first(options.get("apiUrl"), config.get("apiBaseUrl"))
    hostname = url_hostname(endpoint) if endpoint else None
    hostname = hostname.lower() if hostname else None
    if not has_marker and not re.match(r"^bedrock-mantle\.[a-z0-9-]+\.api\.aws$", hostname or ""):
        return None
    catalog = _BEDROCK_MANTLE_TEXT_RATES.get(billing_model)
    base = (
        _apply_rate_multiplier(catalog, _BEDROCK_GOVCLOUD_MULTIPLIER)
        if catalog
        and _uses_bedrock_govcloud_pricing(
            billing_model,
            config,
            options.get("apiUrl"),
            "bedrock" if has_marker else options.get("provider"),
            options.get("region"),
        )
        else catalog
    )
    if base and billing_model in _GPT_5_6_MODELS and total_input_tokens > 272_000:
        return {**_apply_rate_multiplier(base, 2), "output": (base.get("output") or 0) * 1.5}
    return base


def _apply_rate_multiplier(rates: dict[str, float], multiplier: float) -> dict[str, float]:
    out = {"input": rates["input"] * multiplier}
    for key in ("cachedInput", "cacheWriteInput", "output"):
        if rates.get(key) is not None:
            out[key] = rates[key] * multiplier
    return out


def _get_openai_usage_parts(raw_usage: Any) -> dict[str, Any]:
    usage = _first((raw_usage or {}).get("usage") if isinstance(raw_usage, dict) else None,
                   raw_usage, {})  # fmt: skip
    input_details = _first(
        usage.get("prompt_tokens_details"),
        usage.get("input_tokens_details"),
        usage.get("input_token_details"),
        {},
    )
    output_details = _first(
        usage.get("completion_tokens_details"),
        usage.get("output_tokens_details"),
        usage.get("output_token_details"),
        {},
    )
    return {
        "usage": usage,
        "inputDetails": input_details,
        "outputDetails": output_details,
        "hasOutputBreakdown": any(
            _is_number(output_details.get(k))
            for k in ("text_tokens", "image_tokens", "audio_tokens")
        ),
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def calculate_openai_usage_cost_from_token_usage(
    model_name: str | None, token_usage: dict[str, Any] | None
) -> float | None:
    if not model_name or not token_usage:
        return None
    billing_model = re.sub(r"^openai\.", "", model_name)
    # These models have no published Bedrock rates.
    if model_name.startswith("openai.") and billing_model in ("gpt-6-astra", "gpt-6.1-sol"):
        return None
    cache_write = (token_usage.get("completionDetails") or {}).get("cacheCreationInputTokens")
    details: dict[str, Any] = {"cached_tokens": _first(token_usage.get("cached"), 0)}
    if cache_write is not None:
        details["cache_write_tokens"] = cache_write
    cost = calculate_openai_usage_cost(
        billing_model,
        {},
        {
            "prompt_tokens": token_usage.get("prompt"),
            "completion_tokens": token_usage.get("completion"),
            "prompt_tokens_details": details,
        },
    )
    return (
        cost * OPENAI_REGIONAL_PROCESSING_MULTIPLIER
        if cost is not None and model_name.startswith("openai.")
        else cost
    )


def _normalize_service_tier(service_tier: Any) -> str:
    if service_tier == "priority":
        return "fast"
    if service_tier in ("batch", "flex", "fast", "ultrafast"):
        return service_tier
    return "standard"


def _get_base_text_rates(model_name: str, total_input_tokens: float) -> dict[str, Any] | None:
    model = _TEXT_MODELS_BY_ID.get(model_name)
    if not model or not model.get("cost"):
        return None
    cost = model["cost"]
    lc = cost.get("longContext")
    long_context = lc if lc and total_input_tokens > lc["threshold"] else None
    input_rate = _first((long_context or {}).get("input"), cost.get("input"))
    rates: dict[str, Any] = {
        "input": input_rate,
        "cachedInput": _first(
            _LONG_CONTEXT_CACHED_INPUT_RATES.get(model_name) if long_context else None,
            _STANDARD_CACHED_INPUT_RATES.get(model_name),
        ),
    }
    if model_name in _CACHE_WRITE_MODELS:
        rates["cacheWriteInput"] = input_rate * 1.25
    rates["output"] = _first((long_context or {}).get("output"), cost.get("output"))
    return rates


def _get_daybreak_model_rates(
    model_name: str, total_input_tokens: float, options: dict[str, Any]
) -> dict[str, Any] | None:
    underlying = OPENAI_DAYBREAK_ALIASES.get(model_name)
    if (
        not underlying
        or not options.get("apiUrl")
        or options.get("regionalProcessing")
        or options.get("serviceTier") not in (None, "default", "standard")
    ):
        return None
    hostname = url_hostname(options["apiUrl"])
    if hostname is None or hostname.lower() != "api.openai.com":
        return None
    if underlying == "gpt-5.6-sol":
        text = _get_base_text_rates(underlying, total_input_tokens)
        return {"text": text} if text else None
    # Red's model card caps input at 272K; its pricing table has no long-context rates.
    if total_input_tokens > 272_000:
        return None
    return {
        "text": {
            "input": per_million(12.5),
            "cachedInput": per_million(1.25),
            "cacheWriteInput": per_million(15.625),
            "output": per_million(75),
        }
    }


def _get_fine_tuned_model_rates(model_name: str, tier: str) -> dict[str, Any] | None:
    base = next(
        (c for c in _FINE_TUNED_TEXT_RATES if model_name == c or model_name.startswith(f"{c}:")),
        None,
    )
    if not base or tier in ("flex", "fast"):
        return None
    text = _FINE_TUNED_TEXT_RATES[base]
    if tier != "batch":
        return {"text": text}
    override = _FINE_TUNED_BATCH_OVERRIDES.get(base)
    if override is not None:
        return {"text": override}
    halved = {"input": text["input"] * 0.5}
    for key in ("cachedInput", "output"):
        if text.get(key) is not None:
            halved[key] = text[key] * 0.5
    return {"text": halved}


def _get_complete_text_cost_overrides(config: dict[str, Any]) -> dict[str, Any] | None:
    input_rate = _first(config.get("inputCost"), config.get("cost"))
    output_rate = _first(config.get("outputCost"), config.get("cost"))
    return (
        None
        if input_rate is None or output_rate is None
        else {
            "input": input_rate,
            "output": output_rate,
        }
    )


def _get_complete_audio_cost_overrides(
    config: dict[str, Any], usage: dict[str, Any]
) -> dict[str, Any] | None:
    input_rate = _first(config.get("audioInputCost"), config.get("audioCost"))
    output_rate = _first(config.get("audioOutputCost"), config.get("audioCost"))
    if (
        (input_rate is None and output_rate is None)
        or (usage["audioInputTokens"] > 0 and input_rate is None)
        or (usage["audioOutputTokens"] > 0 and output_rate is None)
    ):
        return None
    return {"input": input_rate, "output": output_rate}


def _get_fast_text_rates(model_name: str, total_input_tokens: float) -> dict[str, Any] | None:
    rates = _FAST_TEXT_RATES[model_name]
    long_context = ((_TEXT_MODELS_BY_ID.get(model_name) or {}).get("cost") or {}).get("longContext")
    if not long_context or total_input_tokens <= long_context["threshold"]:
        return rates
    if model_name not in _CACHE_WRITE_MODELS:
        return None
    out = _apply_rate_multiplier(rates, 2)
    if rates.get("output") is not None:
        out["output"] = rates["output"] * 1.5
    return out


def _get_ultrafast_model_rates(model_name: str, total_input_tokens: float) -> dict[str, Any] | None:
    # Only Astra has published Ultrafast rates, at 6x each standard rate.
    text = (
        _get_base_text_rates(model_name, total_input_tokens)
        if model_name == "gpt-6-astra"
        else None
    )
    return {"text": _apply_rate_multiplier(text, 6)} if text else None


def _get_model_rates(
    model_name: str, tier: str, total_input_tokens: float
) -> dict[str, Any] | None:
    if tier == "ultrafast":
        return _get_ultrafast_model_rates(model_name, total_input_tokens)
    if model_name.startswith("ft:"):
        return _get_fine_tuned_model_rates(model_name, tier)
    if model_name.startswith("gpt-image-1.5-"):
        return _IMAGE_MODEL_RATES["gpt-image-1.5"]
    if model_name.startswith("gpt-image-1-mini-"):
        return _IMAGE_MODEL_RATES["gpt-image-1-mini"]
    if model_name.startswith("gpt-image-1-"):
        return _IMAGE_MODEL_RATES["gpt-image-1"]
    if model_name in _IMAGE_MODEL_RATES:
        return _IMAGE_MODEL_RATES[model_name]
    if model_name in _REALTIME_MODAL_RATES:
        return _REALTIME_MODAL_RATES[model_name]
    if model_name in _EMBEDDING_RATES:
        text = _EMBEDDING_RATES[model_name]
        return {"text": {"input": text["input"] * 0.5} if tier == "batch" else text}

    # The current pricing table leaves GPT-5.5 Pro long-context Batch/Flex blank.
    if (
        model_name in ("gpt-5.5-pro", "gpt-5.5-pro-2026-04-23")
        and total_input_tokens > GPT_LONG_CONTEXT_THRESHOLD
        and tier in ("batch", "flex")
    ):
        return None

    model = _TEXT_MODELS_BY_ID.get(model_name)
    if tier == "fast" and model_name in _FAST_TEXT_RATES:
        fast = _get_fast_text_rates(model_name, total_input_tokens)
        return {"text": fast} if fast else None

    text = _get_base_text_rates(model_name, total_input_tokens)
    if not text:
        return None
    discounted = text
    if tier == "batch" or (tier == "flex" and model_name in _FLEX_SUPPORTED_TEXT_MODELS):
        discounted = {"input": text["input"] * 0.5}
        for key in ("cachedInput", "cacheWriteInput", "output"):
            if text.get(key) is not None:
                discounted[key] = text[key] * 0.5
    if tier == "flex" and discounted is text:
        return None
    cost = (model or {}).get("cost") or {}
    result: dict[str, Any] = {"text": discounted}
    if cost.get("audioInput") or cost.get("audioOutput"):
        result["audio"] = {"input": cost.get("audioInput"), "output": cost.get("audioOutput")}
    return result


def _get_numeric_value(value: Any) -> float:
    return value if _is_number(value) and math.isfinite(value) else 0


def extract_openai_billing_usage(raw_usage: Any) -> dict[str, Any]:
    parts = _get_openai_usage_parts(raw_usage)
    usage, input_details, output_details = (
        parts["usage"],
        parts["inputDetails"],
        parts["outputDetails"],
    )
    total_output = _get_numeric_value(
        _first(usage.get("completion_tokens"), usage.get("output_tokens"))
    )
    reported_input = _get_numeric_value(
        _first(usage.get("prompt_tokens"), usage.get("input_tokens"))
    )
    total_input = (
        _get_numeric_value(usage.get("total_tokens"))
        if reported_input == 0 and total_output == 0
        else reported_input
    )
    audio_input = _get_numeric_value(
        _first(
            input_details.get("audio_tokens"),
            usage.get("audio_prompt_tokens"),
            usage.get("audio_input_tokens"),
        )
    )
    image_input = _get_numeric_value(input_details.get("image_tokens"))
    text_input = _get_numeric_value(
        _first(input_details.get("text_tokens"), max(total_input - audio_input - image_input, 0))
    )
    audio_output = _get_numeric_value(
        _first(
            output_details.get("audio_tokens"),
            usage.get("audio_completion_tokens"),
            usage.get("audio_output_tokens"),
        )
    )
    image_output = _get_numeric_value(output_details.get("image_tokens"))
    text_output = _get_numeric_value(
        _first(
            output_details.get("text_tokens"), max(total_output - audio_output - image_output, 0)
        )
    )
    cached_details = _first(input_details.get("cached_tokens_details"), {})
    return {
        "totalInputTokens": total_input,
        "cachedInputTokens": _get_numeric_value(
            _first(input_details.get("cached_tokens"), usage.get("cached_input_tokens"))
        ),
        "cacheWriteInputTokens": _first(get_openai_cache_write_input_tokens(usage), 0),
        "cachedTextInputTokens": _get_numeric_value(cached_details.get("text_tokens")),
        "cachedAudioInputTokens": _get_numeric_value(cached_details.get("audio_tokens")),
        "cachedImageInputTokens": _get_numeric_value(cached_details.get("image_tokens")),
        "textInputTokens": text_input,
        "audioInputTokens": audio_input,
        "imageInputTokens": image_input,
        "totalOutputTokens": total_output,
        "textOutputTokens": text_output,
        "audioOutputTokens": audio_output,
        "imageOutputTokens": image_output,
    }


def _split_cached_input_tokens(usage: dict[str, Any]) -> dict[str, float]:
    if usage["cachedInputTokens"] <= 0:
        return {"textInputTokens": 0, "audioInputTokens": 0, "imageInputTokens": 0}
    explicit = (
        usage["cachedTextInputTokens"]
        + usage["cachedAudioInputTokens"]
        + usage["cachedImageInputTokens"]
    )
    if explicit > 0:
        return {
            "textInputTokens": min(usage["cachedTextInputTokens"], usage["textInputTokens"]),
            "audioInputTokens": min(usage["cachedAudioInputTokens"], usage["audioInputTokens"]),
            "imageInputTokens": min(usage["cachedImageInputTokens"], usage["imageInputTokens"]),
        }
    cached = usage["cachedInputTokens"]
    if usage["audioInputTokens"] == 0 and usage["imageInputTokens"] == 0:
        return {
            "textInputTokens": min(cached, usage["textInputTokens"]),
            "audioInputTokens": 0,
            "imageInputTokens": 0,
        }
    if usage["textInputTokens"] == 0 and usage["imageInputTokens"] == 0:
        return {
            "textInputTokens": 0,
            "audioInputTokens": min(cached, usage["audioInputTokens"]),
            "imageInputTokens": 0,
        }
    if usage["textInputTokens"] == 0 and usage["audioInputTokens"] == 0:
        return {
            "textInputTokens": 0,
            "audioInputTokens": 0,
            "imageInputTokens": min(cached, usage["imageInputTokens"]),
        }
    # Older payloads expose only a total cached-token count; attribute mixed input to text.
    return {
        "textInputTokens": min(cached, usage["textInputTokens"]),
        "audioInputTokens": 0,
        "imageInputTokens": 0,
    }


def _calculate_text_cost(
    rates: dict[str, Any],
    usage: dict[str, Any],
    cached_text_input_tokens: float,
    config: dict[str, Any],
) -> float:
    text_input_cost = _first(config.get("inputCost"), config.get("cost"), rates.get("input"))
    cached_input_cost = _first(
        config.get("inputCost"), config.get("cost"), rates.get("cachedInput"), text_input_cost
    )
    cache_write_cost = _first(
        config.get("inputCost"), config.get("cost"), rates.get("cacheWriteInput"), text_input_cost
    )
    output_cost = _first(config.get("outputCost"), config.get("cost"), rates.get("output"), 0)
    cache_write_tokens = min(
        usage["cacheWriteInputTokens"], max(usage["textInputTokens"] - cached_text_input_tokens, 0)
    )
    uncached = max(usage["textInputTokens"] - cached_text_input_tokens - cache_write_tokens, 0)
    return (
        uncached * text_input_cost
        + cached_text_input_tokens * cached_input_cost
        + cache_write_tokens * cache_write_cost
        + usage["textOutputTokens"] * output_cost
    )


def _calculate_modal_cost(
    rates: dict[str, Any] | None,
    input_tokens: float,
    cached_input_tokens: float,
    output_tokens: float,
    overrides: dict[str, Any],
) -> float:
    if not rates:
        return 0
    input_cost = _first(overrides.get("input"), rates.get("input"), 0)
    cached_cost = _first(overrides.get("cachedInput"), rates.get("cachedInput"), input_cost)
    output_cost = _first(overrides.get("output"), rates.get("output"), 0)
    uncached = max(input_tokens - cached_input_tokens, 0)
    return uncached * input_cost + cached_input_tokens * cached_cost + output_tokens * output_cost


def calculate_custom_usage_cost(
    usage: dict[str, Any], config: dict[str, Any], cached_response: bool | None
) -> float | None:
    token_rates = [
        (
            max(usage["totalInputTokens"] - usage["audioInputTokens"], 0),
            _first(config.get("inputCost"), config.get("cost")),
        ),
        (
            max(usage["totalOutputTokens"] - usage["audioOutputTokens"], 0),
            _first(config.get("outputCost"), config.get("cost")),
        ),
        (usage["audioInputTokens"], _first(config.get("audioInputCost"), config.get("audioCost"))),
        (
            usage["audioOutputTokens"],
            _first(config.get("audioOutputCost"), config.get("audioCost")),
        ),
    ]
    if all(rate is None for _, rate in token_rates) or any(
        tokens > 0 and rate is None for tokens, rate in token_rates
    ):
        return None
    if cached_response:
        return 0
    total: float = 0
    for tokens, rate in token_rates:
        total = total + tokens * (rate if rate is not None else 0)
    return total


def calculate_openai_usage_cost(
    model_name: str,
    config: dict[str, Any],
    raw_usage: Any,
    service_tier: Any = None,
    cached_response: bool | None = None,
    api_url: str | None = None,
    regional_processing: bool | None = None,
    provider: str | None = None,
    region: str | None = None,
) -> float | None:
    """Cost of one response from its usage ledger. Keyword options mirror upstream's
    `options` object (serviceTier, cachedResponse, apiUrl, regionalProcessing, provider,
    region)."""
    if not is_truthy(raw_usage):
        return None
    config = config or {}
    options = {
        "serviceTier": service_tier,
        "cachedResponse": cached_response,
        "apiUrl": api_url,
        "regionalProcessing": regional_processing,
        "provider": provider,
        "region": region,
    }
    usage_parts = _get_openai_usage_parts(raw_usage)
    usage = extract_openai_billing_usage(raw_usage)
    variant = get_gpt6_variant(model_name)
    if variant == "6.1-sol" and provider == "bedrock":
        return calculate_custom_usage_cost(usage, config, cached_response)
    if variant in ("sol", "6.1-sol", "luna") and uses_azure_openai_billing(
        config, api_url, provider
    ):
        input_rate = _first(config.get("inputCost"), config.get("cost"))
        output_rate = _first(config.get("outputCost"), config.get("cost"))
        if (
            (input_rate is None and output_rate is None)
            or (usage["totalInputTokens"] > 0 and input_rate is None)
            or (usage["totalOutputTokens"] > 0 and output_rate is None)
        ):
            return None
        if cached_response:
            return 0
        return usage["totalInputTokens"] * (input_rate or 0) + usage["totalOutputTokens"] * (
            output_rate or 0
        )
    tier = _normalize_service_tier(service_tier)
    mantle_rates = _get_bedrock_mantle_text_rates(
        model_name, config, options, tier, usage["totalInputTokens"]
    )
    if mantle_rates:
        catalog_rates: dict[str, Any] | None = {"text": mantle_rates}
    elif model_name in OPENAI_DAYBREAK_ALIASES:
        catalog_rates = _get_daybreak_model_rates(model_name, usage["totalInputTokens"], options)
    else:
        catalog_rates = _get_model_rates(model_name, tier, usage["totalInputTokens"])
    explicit_text = _get_complete_text_cost_overrides(config)
    explicit_audio = _get_complete_audio_cost_overrides(config, usage)
    # Complete explicit rates are authoritative when no catalog entry applies.
    model_rates: dict[str, Any] | None
    if catalog_rates:
        model_rates = {**catalog_rates}
        if not catalog_rates.get("audio") and explicit_audio:
            model_rates["audio"] = explicit_audio
    elif explicit_text:
        model_rates = {
            "text": explicit_text,
            **({"audio": explicit_audio} if explicit_audio else {}),
        }
    else:
        model_rates = None
    if not model_rates:
        return calculate_custom_usage_cost(usage, config, cached_response)
    rates = (
        model_rates
        if mantle_rates
        else _apply_regional_processing_rates(model_name, model_rates, config, options)
    )

    if cached_response:
        return 0
    if not rates.get("audio") and (usage["audioInputTokens"] > 0 or usage["audioOutputTokens"] > 0):
        return None

    text_rates = rates["text"]
    text_input_cost = _first(config.get("inputCost"), config.get("cost"), text_rates.get("input"))
    cache_write_cost = _first(
        config.get("inputCost"),
        config.get("cost"),
        text_rates.get("cacheWriteInput"),
        text_input_cost,
    )
    if (
        cache_write_cost != text_input_cost
        and get_openai_cache_write_input_tokens(usage_parts["usage"]) is None
    ):
        return None

    has_output_breakdown = usage_parts["hasOutputBreakdown"]
    image_rates = rates.get("image")
    if (
        (image_rates or {}).get("output")
        and text_rates.get("output")
        and usage["totalOutputTokens"] > 0
        and usage["imageOutputTokens"] == 0
        and not has_output_breakdown
    ):
        return None
    if (
        (image_rates or {}).get("output")
        and text_rates.get("output") is None
        and usage["totalOutputTokens"] > 0
        and usage["imageOutputTokens"] == 0
        and usage["audioOutputTokens"] == 0
        and usage["textOutputTokens"] == usage["totalOutputTokens"]
    ):
        usage["imageOutputTokens"] = usage["totalOutputTokens"]
        usage["textOutputTokens"] = 0
    if (
        not catalog_rates
        and not image_rates
        and (usage["imageInputTokens"] > 0 or usage["imageOutputTokens"] > 0)
    ):
        return None
    if not image_rates:
        usage["textInputTokens"] += usage["imageInputTokens"]
        usage["cachedTextInputTokens"] += usage["cachedImageInputTokens"]
        usage["imageInputTokens"] = 0
        usage["cachedImageInputTokens"] = 0

    cached_input = _split_cached_input_tokens(usage)
    text_cost = _calculate_text_cost(text_rates, usage, cached_input["textInputTokens"], config)
    audio_override = _first(config.get("audioInputCost"), config.get("audioCost"))
    audio_cost = _calculate_modal_cost(
        rates.get("audio"),
        usage["audioInputTokens"],
        cached_input["audioInputTokens"],
        usage["audioOutputTokens"],
        {
            "input": audio_override,
            "cachedInput": audio_override,
            "output": _first(config.get("audioOutputCost"), config.get("audioCost")),
        },
    )
    image_cost = _calculate_modal_cost(
        image_rates,
        usage["imageInputTokens"],
        cached_input["imageInputTokens"],
        usage["imageOutputTokens"],
        {},
    )
    return text_cost + audio_cost + image_cost


def _is_reasoning_model(model_name: str) -> bool:
    model = re.sub(r"(^|/)ft:", r"\1", re.sub(r"^bedrock:", "", model_name))
    return (
        model.startswith(("gpt-5", "o1", "o3", "o4"))
        or re.search(r"(^|/)gpt-daybreak-(?:blue|red)-latest$", model) is not None
        or is_gpt6_model(model)
        or "deep-research" in model
    )


def _get_web_search_tool_type(config: dict[str, Any]) -> str | None:
    tools = config.get("tools")
    if not isinstance(tools, list):
        return None
    types = [t.get("type") for t in tools if isinstance(t, dict)]
    if "web_search" in types:
        return "web_search"
    if "web_search_preview" in types:
        return "web_search_preview"
    return None


def calculate_observable_openai_tool_cost(
    data: Any, model_name: str, config: dict[str, Any] | None = None
) -> float:
    output = data.get("output") if isinstance(data, dict) else None
    if not isinstance(output, list):
        return 0
    web_search_type = _get_web_search_tool_type(config or {})
    total: float = 0
    for item in output:
        item = item if isinstance(item, dict) else {}
        if item.get("type") == "web_search_call" and (item.get("action") or {}).get("type") == (
            "search"
        ):
            total += (
                0.01
                if web_search_type == "web_search" or _is_reasoning_model(model_name)
                else 0.025
            )
        elif item.get("type") == "file_search_call":
            total += 0.0025
    return total
