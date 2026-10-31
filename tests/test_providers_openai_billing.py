"""OpenAI usage billing. Ports the text, embedding, cache, tier, regional and custom-rate
cases from test/providers/openai/billing.test.ts (image and realtime-only cases are skipped)."""

from __future__ import annotations

from typing import Any

import pytest

from promptphoo.providers.openai.billing import (
    calculate_observable_openai_tool_cost,
    calculate_openai_usage_cost,
    calculate_openai_usage_cost_from_token_usage,
    extract_openai_billing_usage,
)

_OPTION_NAMES = {
    "serviceTier": "service_tier",
    "cachedResponse": "cached_response",
    "apiUrl": "api_url",
    "regionalProcessing": "regional_processing",
    "provider": "provider",
    "region": "region",
}

TIERS = [("default", 1), ("batch", 0.5), ("flex", 0.5), ("fast", 2), ("priority", 2)]
API_URL = "https://api.openai.com/v1"
US_EU = ["https://us.api.openai.com/v1", "https://eu.api.openai.com/v1"]


def cost(model: str, config: dict[str, Any], usage: Any, options: dict[str, Any] | None = None):
    """calculateOpenAIUsageCost with upstream's camelCase options object."""
    kwargs = {_OPTION_NAMES[k]: v for k, v in (options or {}).items()}
    return calculate_openai_usage_cost(model, config, usage, **kwargs)


def close(expected: float, digits: int = 10):
    """vitest toBeCloseTo(expected, digits)."""
    return pytest.approx(expected, rel=0, abs=0.5 * 10**-digits)


def cached_responses_usage() -> dict[str, Any]:
    return {
        "input_tokens": 2000,
        "output_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 500, "cache_write_tokens": 250},
    }


def cached_chat_usage() -> dict[str, Any]:
    return {
        "prompt_tokens": 2_000,
        "completion_tokens": 1_000,
        "prompt_tokens_details": {"cached_tokens": 500},
    }


def cached_input_usage() -> dict[str, Any]:
    return {
        "input_tokens": 1_000,
        "output_tokens": 100,
        "input_tokens_details": {"cached_tokens": 400},
    }


# --- Daybreak aliases -----------------------------------------------------------------------

DAYBREAK_USAGE = cached_responses_usage()


@pytest.mark.parametrize(
    ("model", "input_tokens", "inp", "cached", "write", "out"),
    [
        ("gpt-daybreak-blue-latest", 272_000, 4, 0.4, 5, 20),
        ("gpt-daybreak-blue-latest", 272_001, 8, 0.8, 10, 30),
        ("gpt-daybreak-red-latest", 272_000, 12.5, 1.25, 15.625, 75),
    ],
)
def test_daybreak_pricing(model, input_tokens, inp, cached, write, out):
    usage = {**DAYBREAK_USAGE, "input_tokens": input_tokens}
    assert cost(model, {}, usage, {"apiUrl": API_URL, "serviceTier": "default"}) == close(
        ((input_tokens - 750) * inp + 500 * cached + 250 * write + 1000 * out) / 1e6
    )


def test_daybreak_red_not_extrapolated():
    usage = {**DAYBREAK_USAGE, "input_tokens": 272_001}
    assert cost("gpt-daybreak-red-latest", {}, usage, {"apiUrl": API_URL}) is None


@pytest.mark.parametrize("model", ["gpt-daybreak-blue-latest", "gpt-daybreak-red-latest"])
def test_daybreak_limited_to_established_tiers_endpoints_and_usage(model):
    for tier in (None, "default", "standard"):
        assert isinstance(
            cost(model, {}, DAYBREAK_USAGE, {"apiUrl": API_URL, "serviceTier": tier}), float
        )
    for tier in ("auto", "batch", "flex", "priority", "fast", "future-tier"):
        assert cost(model, {}, DAYBREAK_USAGE, {"apiUrl": API_URL, "serviceTier": tier}) is None
    for endpoint in (
        None,
        "https://gateway.example/v1",
        *US_EU,
        "https://api.openai.com.example/v1",
        "https://gateway.example/api.openai.com",
    ):
        assert cost(model, {}, DAYBREAK_USAGE, {"apiUrl": endpoint}) is None
    assert cost(model, {}, DAYBREAK_USAGE, {"apiUrl": API_URL, "regionalProcessing": True}) is None
    assert (
        cost(model, {}, {"input_tokens": 2000, "output_tokens": 1000}, {"apiUrl": API_URL}) is None
    )
    assert cost(model, {}, None, {"apiUrl": API_URL}) is None
    audio_usage = {
        **DAYBREAK_USAGE,
        "input_tokens_details": {**DAYBREAK_USAGE["input_tokens_details"], "audio_tokens": 10},
    }
    assert cost(model, {}, audio_usage, {"apiUrl": API_URL}) is None
    assert cost(model, {}, DAYBREAK_USAGE, {"apiUrl": API_URL, "cachedResponse": True}) == 0


def test_daybreak_partial_overrides_and_explicit_rates():
    red = "gpt-daybreak-red-latest"
    assert cost(red, {"inputCost": 0}, DAYBREAK_USAGE, {"apiUrl": API_URL}) == close(0.075)
    gateway = {"apiUrl": "https://gateway.example/v1", "serviceTier": "flex"}
    assert cost(red, {"inputCost": 0}, DAYBREAK_USAGE, gateway) is None
    assert cost(red, {"inputCost": 2 / 1e6, "outputCost": 3 / 1e6}, DAYBREAK_USAGE, gateway) == (
        close(0.007)
    )
    assert cost(red, {"cost": 0}, DAYBREAK_USAGE, gateway) == 0


def test_daybreak_no_arbitrary_name_resolution():
    for model in (
        "openai/gpt-daybreak-blue-latest",
        "bedrock:gpt-daybreak-blue-latest",
        "gpt-daybreak-red-latest-custom",
    ):
        assert cost(model, {}, DAYBREAK_USAGE, {"apiUrl": API_URL}) is None
    assert (
        calculate_openai_usage_cost_from_token_usage(
            "openai.gpt-daybreak-blue-latest",
            {
                "prompt": 2000,
                "completion": 1000,
                "completionDetails": {"cacheCreationInputTokens": 0},
            },
        )
        is None
    )


# --- GPT-6 family ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "input_tokens", "inp", "cached", "write", "out"),
    [
        ("gpt-6.1-sol", 272_000, 2, 0.1, 2.5, 10),
        ("gpt-6.1-sol", 272_001, 4, 0.2, 5, 15),
        ("gpt-6-astra", 272_000, 10, 1, 12.5, 50),
        ("gpt-6-astra", 272_001, 20, 2, 25, 75),
    ],
)
def test_gpt6_all_tiers(model, input_tokens, inp, cached, write, out):
    usage = {**cached_responses_usage(), "input_tokens": input_tokens}
    standard = ((input_tokens - 750) * inp + 500 * cached + 250 * write + 1000 * out) / 1e6
    for tier, multiplier in TIERS:
        assert cost(model, {}, usage, {"serviceTier": tier}) == close(standard * multiplier)


def test_gpt61_sol_regional_normalized_and_custom():
    usage = cached_responses_usage()
    expected = (1250 * 2 + 500 * 0.1 + 250 * 2.5 + 1000 * 10) / 1e6
    for api_url in US_EU:
        assert cost("gpt-6.1-sol", {}, usage, {"apiUrl": api_url}) == close(expected * 1.1)
        assert cost("gpt-6.1-sol", {}, usage, {"apiUrl": api_url, "cachedResponse": True}) == 0
        assert cost("gpt-6.1-sol", {"inputCost": 3 / 1e6}, usage, {"apiUrl": api_url}) == close(
            (2000 * 3 + 1000 * 11) / 1e6
        )
    assert calculate_openai_usage_cost_from_token_usage(
        "gpt-6.1-sol",
        {"prompt": 2000, "completion": 1000, "cached": 500,
         "completionDetails": {"cacheCreationInputTokens": 250}},
    ) == close(expected)  # fmt: skip
    assert cost("gpt-6.1-sol", {"cost": 3 / 1e6}, usage) == close(0.009)
    assert cost("gpt-6.1-sol-unpublished", {}, usage) is None


@pytest.mark.parametrize("provider", ["azure", "azure-openai", "bedrock"])
def test_gpt61_sol_requires_explicit_rates_off_platform(provider):
    usage = {"input_tokens": 1000, "output_tokens": 100}
    options = {"provider": provider}
    assert cost("gpt-6.1-sol", {}, usage, options) is None
    assert cost("gpt-6.1-sol", {"inputCost": 2 / 1e6}, usage, options) is None
    assert cost("gpt-6.1-sol", {"inputCost": 2 / 1e6, "outputCost": 3 / 1e6}, usage, options) == (
        close(0.0023)
    )


def test_no_bedrock_inference_from_normalized_usage():
    assert (
        calculate_openai_usage_cost_from_token_usage(
            "openai.gpt-6.1-sol", {"prompt": 1000, "completion": 100}
        )
        is None
    )
    assert (
        calculate_openai_usage_cost_from_token_usage(
            "openai.gpt-6-astra", {"prompt": 2000, "completion": 1000, "cached": 500}
        )
        is None
    )


@pytest.mark.parametrize(
    ("input_tokens", "inp", "cached", "write", "out"),
    [(272_000, 60, 6, 75, 300), (272_001, 120, 12, 150, 450)],
)
def test_ultrafast_astra_rates(input_tokens, inp, cached, write, out):
    usage = {**cached_responses_usage(), "input_tokens": input_tokens}
    assert cost("gpt-6-astra", {}, usage, {"serviceTier": "ultrafast"}) == close(
        ((input_tokens - 750) * inp + 500 * cached + 250 * write + 1000 * out) / 1e6
    )


def test_ultrafast_explicit_costs_and_local_cache():
    usage = {"input_tokens": 1000, "output_tokens": 100}
    options = {"serviceTier": "ultrafast"}
    assert cost("gpt-6-astra", {"inputCost": 2 / 1e6}, usage, options) == close(0.032)
    assert cost("gpt-6-astra", {}, usage, {**options, "cachedResponse": True}) == 0


@pytest.mark.parametrize(
    "model", ["gpt-5.6-sol", "gpt-6.1-sol", "gpt-6-sol", "gpt-4o", "gpt-image-1"]
)
def test_ultrafast_not_substituted_for_unpublished_models(model):
    usage = {"input_tokens": 1000, "output_tokens": 100}
    options = {"serviceTier": "ultrafast"}
    assert cost(model, {}, usage, options) is None
    assert cost(model, {"inputCost": 2 / 1e6}, usage, options) is None
    assert cost(model, {"inputCost": 2 / 1e6, "outputCost": 3 / 1e6}, usage, options) == close(
        0.0023
    )


@pytest.mark.parametrize("api_url", US_EU)
def test_astra_regional_text_image_and_cache(api_url):
    usage = {
        "input_tokens": 2000,
        "output_tokens": 1000,
        "input_tokens_details": {
            "text_tokens": 1500,
            "image_tokens": 500,
            "cached_tokens": 500,
            "cached_tokens_details": {"image_tokens": 500},
            "cache_write_tokens": 250,
        },
    }
    assert cost("gpt-6-astra", {}, usage, {"apiUrl": api_url}) == close(
        ((1250 * 10 + 500 * 1 + 250 * 12.5 + 1000 * 50) / 1e6) * 1.1
    )
    assert cost("gpt-6-astra", {}, usage, {"apiUrl": api_url, "cachedResponse": True}) == 0


def test_astra_explicit_overrides_and_unpublished_ids():
    assert cost(
        "gpt-6-astra", {"inputCost": 2 / 1e6, "outputCost": 3 / 1e6}, cached_responses_usage()
    ) == close(0.007)
    assert (
        cost(
            "gpt-6-astra-unpublished",
            {},
            {"input_tokens": 2000, "output_tokens": 1000},
            {"apiUrl": "https://us.api.openai.com/v1"},
        )
        is None
    )


SOL_LUNA = [("gpt-6-sol", 2, 0.2, 2.5, 10), ("gpt-6-luna", 0.1, 0.01, 0.125, 0.5)]


@pytest.mark.parametrize(("model", "inp", "cached", "write", "out"), SOL_LUNA)
@pytest.mark.parametrize("input_tokens", [272_000, 272_001])
def test_sol_luna_every_tier(model, inp, cached, write, out, input_tokens):
    input_multiplier = 2 if input_tokens > 272_000 else 1
    output_multiplier = 1.5 if input_tokens > 272_000 else 1
    usage = {**cached_responses_usage(), "input_tokens": input_tokens}
    standard = (
        ((input_tokens - 750) * inp + 500 * cached + 250 * write) * input_multiplier
        + 1000 * out * output_multiplier
    ) / 1e6
    for tier, multiplier in TIERS:
        assert cost(model, {}, usage, {"serviceTier": tier}) == close(standard * multiplier)


@pytest.mark.parametrize(("model", "inp", "cached", "write", "out"), SOL_LUNA)
def test_sol_luna_regional_normalized_and_bedrock(model, inp, cached, write, out):
    usage = cached_responses_usage()
    expected = (1250 * inp + 500 * cached + 250 * write + 1000 * out) / 1e6
    for api_url in US_EU:
        assert cost(model, {}, usage, {"apiUrl": api_url}) == close(expected * 1.1)
        assert cost(model, {}, usage, {"apiUrl": api_url, "cachedResponse": True}) == 0
    assert calculate_openai_usage_cost_from_token_usage(
        model,
        {"prompt": 2000, "completion": 1000, "cached": 500,
         "completionDetails": {"cacheCreationInputTokens": 250}},
    ) == close(expected)  # fmt: skip
    assert cost(model, {"inputCost": 2 / 1e6, "outputCost": 3 / 1e6}, usage) == close(0.007)
    assert calculate_openai_usage_cost_from_token_usage(
        f"openai.{model}",
        {"prompt": 2000, "completion": 1000, "completionDetails": {"cacheCreationInputTokens": 0}},
    ) == close(((2000 * inp + 1000 * out) / 1e6) * 1.1)
    assert cost(model, {}, usage, {"provider": "bedrock"}) == close(expected)
    for options in (
        {"provider": "bedrock", "apiUrl": "https://bedrock-mantle.us-east-1.api.aws/openai/v1"},
        {"provider": "bedrock", "regionalProcessing": True},
    ):
        assert cost(model, {}, usage, options) == close(expected * 1.1)
    assert cost(model, {"inputCost": 2 / 1e6}, usage, {"provider": "bedrock"}) == close(
        (2000 * 2 + 1000 * out) / 1e6
    )
    assert cost(
        model, {"inputCost": 2 / 1e6, "outputCost": 3 / 1e6}, usage, {"provider": "bedrock"}
    ) == close(0.007)
    assert cost(f"{model}-unpublished", {}, usage) is None


@pytest.mark.parametrize(("model", "inp", "cached", "write", "out"), SOL_LUNA)
def test_sol_luna_no_direct_prices_for_azure(model, inp, cached, write, out):
    usage = {"input_tokens": 1000, "output_tokens": 100, "cache_write_input_tokens": 0}
    for options in (
        {"provider": "azure-openai"},
        {"provider": "azure"},
        {"provider": "openai", "apiUrl": "https://example.openai.azure.com/openai/v1"},
        {"provider": "openai", "apiUrl": "https://example.services.ai.azure.com/openai/v1"},
        {"apiUrl": "https://example.services.ai.azure.com/api/projects/project/openai/v1"},
        {
            "apiUrl": "https://gateway.ai.cloudflare.com/v1/account/gateway/azure-openai/"
            "resource/deployment"
        },
    ):
        assert cost(model, {}, usage, options) is None
        assert cost(model, {"inputCost": 2 / 1e6}, usage, options) is None
        assert cost(model, {"inputCost": 2 / 1e6, "outputCost": 3 / 1e6}, usage, options) == (
            close(0.0023)
        )
        assert cost(model, {"cost": 2 / 1e6}, usage, options) == close(0.0022)
        assert cost(model, {"cost": 2 / 1e6}, usage, {**options, "cachedResponse": True}) == 0
        assert cost(
            model, {"inputCost": 2 / 1e6}, {"input_tokens": 1000, "output_tokens": 0}, options
        ) == close(0.002)
    assert cost(model, {"apiHost": "example.openai.azure.com"}, usage) is None
    assert cost(model, {"apiHost": "example.services.ai.azure.com"}, usage) is None
    for api_url in (
        "https://example.openai.azure.com.invalid/openai/v1",
        "https://example.services.ai.azure.com.invalid/openai/v1",
        "https://services.ai.azure.com.attacker.test/api/projects/project/openai/v1",
        "https://nonazure-services.ai.azure.example/openai/v1",
        "https://gateway.ai.cloudflare.com/v1/account/gateway/openai",
        "https://gateway.ai.cloudflare.com.invalid/v1/account/gateway/azure-openai/resource/x",
        API_URL,
    ):
        assert cost(model, {}, usage, {"apiUrl": api_url}) == close((1000 * inp + 100 * out) / 1e6)


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-luna"])
def test_reasoning_model_web_search_preview_pricing(model):
    data = {"output": [{"type": "web_search_call", "action": {"type": "search"}}]}
    assert (
        calculate_observable_openai_tool_cost(
            data, model, {"tools": [{"type": "web_search_preview"}]}
        )
        == 0.01
    )


# --- usage extraction -----------------------------------------------------------------------


def test_extracts_multimodal_responses_usage():
    assert extract_openai_billing_usage(
        {
            "input_tokens": 220,
            "output_tokens": 460,
            "input_tokens_details": {
                "text_tokens": 20,
                "image_tokens": 194,
                "audio_tokens": 6,
                "cached_tokens": 12,
                "cache_write_tokens": 5,
            },
            "output_tokens_details": {"text_tokens": 180, "image_tokens": 272, "audio_tokens": 8},
        }
    ) == {
        "totalInputTokens": 220,
        "cachedInputTokens": 12,
        "cacheWriteInputTokens": 5,
        "cachedTextInputTokens": 0,
        "cachedAudioInputTokens": 0,
        "cachedImageInputTokens": 0,
        "textInputTokens": 20,
        "audioInputTokens": 6,
        "imageInputTokens": 194,
        "totalOutputTokens": 460,
        "textOutputTokens": 180,
        "audioOutputTokens": 8,
        "imageOutputTokens": 272,
    }


def test_extracts_multimodal_realtime_usage():
    assert extract_openai_billing_usage(
        {
            "input_tokens": 30,
            "output_tokens": 23,
            "input_token_details": {
                "text_tokens": 21,
                "audio_tokens": 9,
                "cached_tokens": 4,
                "cached_tokens_details": {"text_tokens": 3, "audio_tokens": 1, "image_tokens": 0},
            },
            "output_token_details": {"text_tokens": 16, "audio_tokens": 7},
        }
    ) == {
        "totalInputTokens": 30,
        "cachedInputTokens": 4,
        "cacheWriteInputTokens": 0,
        "cachedTextInputTokens": 3,
        "cachedAudioInputTokens": 1,
        "cachedImageInputTokens": 0,
        "textInputTokens": 21,
        "audioInputTokens": 9,
        "imageInputTokens": 0,
        "totalOutputTokens": 23,
        "textOutputTokens": 16,
        "audioOutputTokens": 7,
        "imageOutputTokens": 0,
    }


# --- standard text models -------------------------------------------------------------------


def test_cached_input_discount_for_standard_text():
    usage = {
        "prompt_tokens": 1_000,
        "completion_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 400},
    }
    assert cost("gpt-5-mini", {}, usage, {}) == close((600 * 0.25 + 400 * 0.025 + 100 * 2) / 1e6)


@pytest.mark.parametrize("model", ["gpt-5-search-api", "gpt-5-search-api-2025-10-14"])
def test_search_model_cached_input(model):
    assert cost(model, {}, cached_chat_usage()) == close(
        (1_500 * 1.25 + 500 * 0.125 + 1_000 * 10) / 1e6
    )


@pytest.mark.parametrize(
    ("model", "inp", "cached", "out"),
    [
        ("ft:babbage-002:company::model", 1.6, None, 1.6),
        ("ft:davinci-002:company::model", 12, None, 12),
        ("ft:gpt-3.5-turbo:company::model", 3, None, 6),
        ("ft:gpt-3.5-turbo-0125:company::model", 3, None, 6),
        ("ft:gpt-3.5-turbo-0613:company::model", 3, None, 6),
        ("ft:gpt-3.5-turbo-1106:company::model", 3, None, 6),
        ("ft:gpt-4-0613:company::model", 30, None, 60),
        ("ft:gpt-4.1:company::model", 3, 0.75, 12),
        ("ft:gpt-4.1-2025-04-14:company::model", 3, 0.75, 12),
        ("ft:gpt-4.1-mini:company::model", 0.8, 0.2, 3.2),
        ("ft:gpt-4.1-mini-2025-04-14:company::model", 0.8, 0.2, 3.2),
        ("ft:gpt-4.1-nano:company::model", 0.2, 0.05, 0.8),
        ("ft:gpt-4.1-nano-2025-04-14:company::model", 0.2, 0.05, 0.8),
        ("ft:gpt-4o:company::model", 3.75, 1.875, 15),
        ("ft:gpt-4o-2024-08-06:company::model", 3.75, 1.875, 15),
        ("ft:gpt-4o-2024-11-20:company::model", 3.75, None, 15),
        ("ft:gpt-4o-mini:company::model", 0.3, 0.15, 1.2),
        ("ft:gpt-4o-mini-2024-07-18:company::model", 0.3, 0.15, 1.2),
        ("ft:o4-mini:company::model", 4, 1, 16),
        ("ft:o4-mini-2025-04-16:company::model", 4, 1, 16),
    ],
)
def test_fine_tuned_inference_rates(model, inp, cached, out):
    cached_input = 0 if cached is None else 500
    usage = {
        "prompt_tokens": 2_000,
        "completion_tokens": 1_000,
        "prompt_tokens_details": {"cached_tokens": cached_input},
    }
    assert cost(model, {}, usage) == close(
        ((2_000 - cached_input) * inp + cached_input * (inp if cached is None else cached)
         + 1_000 * out) / 1e6
    )  # fmt: skip


def test_fine_tuned_batch_and_unsupported_tiers():
    model = "ft:gpt-4.1-mini-2025-04-14:company::model"
    usage = cached_chat_usage()
    assert cost(model, {}, usage, {"serviceTier": "batch"}) == close(
        ((1_500 * 0.8 + 500 * 0.2 + 1_000 * 3.2) / 1e6) * 0.5
    )
    assert cost(model, {}, usage, {"serviceTier": "flex"}) is None
    assert cost(model, {}, usage, {"serviceTier": "priority"}) is None


def test_fine_tuned_unknown_base_unpriced():
    usage = {"prompt_tokens": 2_000, "completion_tokens": 1_000}
    assert cost("ft:gpt-99-unknown:company::model", {}, usage) is None


def test_fine_tuned_cached_input_falls_back_to_input_rate():
    usage = cached_chat_usage()
    assert cost("ft:gpt-4o-mini:company::model", {}, usage) == close(
        (1_500 * 0.3 + 500 * 0.15 + 1_000 * 1.2) / 1e6
    )
    assert cost("ft:gpt-4o-2024-11-20:company::model", {}, usage) == close(
        (1_500 * 3.75 + 500 * 3.75 + 1_000 * 15) / 1e6
    )


@pytest.mark.parametrize(
    ("model", "inp", "cached", "out"),
    [
        ("ft:babbage-002:company::model", 0.8, None, 0.9),
        ("ft:gpt-4.1-2025-04-14:company::model", 1.5, 0.5, 6),
        ("ft:gpt-4o-2024-08-06:company::model", 2.225, 0.9, 12.5),
    ],
)
def test_fine_tuned_published_batch_rates(model, inp, cached, out):
    cached_input = 0 if cached is None else 500
    usage = {
        "prompt_tokens": 2_000,
        "completion_tokens": 1_000,
        "prompt_tokens_details": {"cached_tokens": cached_input},
    }
    assert cost(model, {}, usage, {"serviceTier": "batch"}) == close(
        ((2_000 - cached_input) * inp + cached_input * (inp if cached is None else cached)
         + 1_000 * out) / 1e6
    )  # fmt: skip


def test_gpt53_codex_priced_and_spark_unpriced():
    assert cost("gpt-5.3-codex", {}, cached_chat_usage()) == close(
        (1_500 * 1.75 + 500 * 0.175 + 1_000 * 14) / 1e6
    )
    assert cost("gpt-5.3-codex-spark", {}, cached_chat_usage()) is None


GPT_56 = [
    ("gpt-5.6", 4, 0.4, 20),
    ("gpt-5.6-sol", 4, 0.4, 20),
    ("gpt-5.6-terra", 2, 0.2, 12),
    ("gpt-5.6-luna", 0.2, 0.02, 1.2),
]


@pytest.mark.parametrize(("model", "inp", "cached", "out"), GPT_56)
def test_gpt56_cached_input_discount(model, inp, cached, out):
    usage = {
        "prompt_tokens": 2_000,
        "completion_tokens": 1_000,
        "prompt_tokens_details": {"cached_tokens": 500, "cache_write_tokens": 0},
    }
    assert cost(model, {}, usage) == close((1_500 * inp + 500 * cached + 1_000 * out) / 1e6)


@pytest.mark.parametrize(("model", "inp", "cached", "out"), GPT_56)
def test_gpt56_image_input_at_text_rate(model, inp, cached, out):
    usage = {
        "input_tokens": 1_000,
        "output_tokens": 100,
        "input_tokens_details": {"image_tokens": 200, "cache_write_tokens": 0},
    }
    assert cost(model, {}, usage) == close((1_000 * inp + 100 * out) / 1e6)


def test_gpt56_cached_image_input_at_cached_text_rate():
    usage = {
        "input_tokens": 1_000,
        "output_tokens": 100,
        "input_tokens_details": {
            "text_tokens": 800,
            "image_tokens": 200,
            "cached_tokens": 200,
            "cached_tokens_details": {"image_tokens": 200},
            "cache_write_tokens": 0,
        },
    }
    assert cost("gpt-5.6", {}, usage) == close((800 * 4 + 200 * 0.4 + 100 * 20) / 1e6)


@pytest.mark.parametrize(
    ("model", "inp", "cached", "write", "out"),
    [
        ("gpt-5.6", 4, 0.4, 5, 20),
        ("gpt-5.6-sol", 4, 0.4, 5, 20),
        ("gpt-5.6-terra", 2, 0.2, 2.5, 12),
        ("gpt-5.6-luna", 0.2, 0.02, 0.25, 1.2),
    ],
)
def test_gpt56_cache_writes_at_125x(model, inp, cached, write, out):
    assert cost(model, {}, cached_responses_usage()) == close(
        (1_250 * inp + 500 * cached + 250 * write + 1_000 * out) / 1e6
    )


@pytest.mark.parametrize(("model", "inp", "cached", "out"), GPT_56)
def test_gpt56_unknown_without_cache_write_tokens(model, inp, cached, out):
    usage = {
        "input_tokens": 2_000,
        "output_tokens": 1_000,
        "input_tokens_details": {"cached_tokens": 500},
    }
    assert cost(model, {}, usage) is None


def test_gpt56_custom_input_cost_without_cache_writes():
    usage = {
        "input_tokens": 2_000,
        "output_tokens": 1_000,
        "input_tokens_details": {"cached_tokens": 500},
    }
    assert cost("gpt-5.6", {"inputCost": 2 / 1e6}, usage) == close((2_000 * 2 + 1_000 * 20) / 1e6)


def test_gpt56_explicit_top_level_zero_cache_write():
    usage = {"input_tokens": 2_000, "output_tokens": 1_000, "cache_write_input_tokens": 0}
    assert cost("gpt-5.6-terra", {}, usage) == close((2_000 * 2 + 1_000 * 12) / 1e6)


@pytest.mark.parametrize(
    ("config", "options"),
    [
        ({"apiHost": "eu.api.openai.com"}, {}),
        ({"apiBaseUrl": "https://us.api.openai.com/v1"}, {}),
        ({}, {"apiUrl": "https://eu.api.openai.com/v1"}),
    ],
)
def test_gpt56_regional_uplift(config, options):
    assert cost("gpt-5.6", config, cached_responses_usage(), options) == close(
        ((1_250 * 4 + 500 * 0.4 + 250 * 5 + 1_000 * 20) / 1e6) * 1.1
    )


@pytest.mark.parametrize(
    ("model", "inp", "out", "long_in", "long_out"),
    [("gpt-5.6-terra", 2, 12, 4, 18), ("gpt-5.6-luna", 0.2, 1.2, 0.4, 1.8)],
)
def test_bedrock_govcloud_rates(model, inp, out, long_in, long_out):
    short_usage = cached_responses_usage()
    short_base = (1_250 * inp + 500 * inp * 0.1 + 250 * inp * 1.25 + 1_000 * out) / 1e6
    long_usage = {"input_tokens": 300_000, "output_tokens": 1_000, "cache_write_input_tokens": 0}
    long_base = (300_000 * long_in + 1_000 * long_out) / 1e6
    for region in ("us-gov-east-1", "us-gov-west-1"):
        api_url = f"https://bedrock-mantle.{region}.api.aws/openai/v1"
        assert cost(model, {}, short_usage, {"apiUrl": api_url}) == close(short_base * 1.1 * 1.2)
        assert cost(model, {}, long_usage, {"apiUrl": api_url}) == close(long_base * 1.1 * 1.2)
        for runtime_host in (
            f"bedrock-runtime.{region}.amazonaws.com",
            f"bedrock-runtime.{region}.api.aws",
            f"bedrock-runtime-fips.{region}.amazonaws.com",
        ):
            options = {
                "apiUrl": f"https://{runtime_host}/openai/v1",
                "provider": "bedrock",
                "region": region,
                "regionalProcessing": True,
            }
            assert cost(model, {"region": region}, short_usage, options) == close(short_base * 1.1)
    proxy = "https://proxy.example.test/openai/v1"
    assert cost(
        model, {"region": "us-gov-east-1"}, short_usage,
        {"provider": "bedrock", "regionalProcessing": True, "apiUrl": proxy},
    ) == close(short_base * 1.1 * 1.2)  # fmt: skip
    assert cost(
        model, {}, short_usage,
        {"provider": "bedrock", "region": "us-gov-west-1", "regionalProcessing": True,
         "apiUrl": proxy},
    ) == close(short_base * 1.1 * 1.2)  # fmt: skip
    assert cost(
        model, {"region": "us-gov-east-1"}, short_usage,
        {"provider": "bedrock", "apiUrl": "https://bedrock-mantle.us-east-1.api.aws/openai/v1"},
    ) == close(short_base * 1.1)  # fmt: skip
    assert cost(
        model, {}, short_usage,
        {"apiUrl": "https://bedrock-runtime.us-gov-west-1.amazonaws.com/openai/v1",
         "provider": "bedrock", "regionalProcessing": False},
    ) == close(short_base)  # fmt: skip
    assert cost(
        model, {"inputCost": 7 / 1e6}, short_usage,
        {"apiUrl": "https://bedrock-mantle.us-gov-east-1.api.aws/openai/v1"},
    ) == close((2_000 * 7 + 1_000 * out * 1.1 * 1.2) / 1e6)  # fmt: skip


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.4",
        "gpt-5.4-2026-03-05",
        "gpt-5.4-mini",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5.4-nano",
        "gpt-5.4-nano-2026-03-17",
        "gpt-5.4-pro",
        "gpt-5.4-pro-2026-03-05",
        "gpt-5.5",
        "gpt-5.5-2026-04-23",
        "gpt-5.5-pro",
        "gpt-5.5-pro-2026-04-23",
        "gpt-5.6",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    ],
)
def test_regional_uplift_for_eligible_models(model):
    usage = {
        "input_tokens": 2_000,
        "output_tokens": 1_000,
        "input_tokens_details": {"cached_tokens": 500, "cache_write_tokens": 0},
    }
    standard = cost(model, {}, usage)
    assert standard is not None
    assert cost(model, {"apiHost": "eu.api.openai.com"}, usage) == close(standard * 1.1)


def test_no_regional_uplift_before_cutoff():
    usage = {"input_tokens": 2_000, "output_tokens": 1_000}
    standard = cost("gpt-5.2", {}, usage)
    assert standard is not None
    assert cost("gpt-5.2", {"apiHost": "eu.api.openai.com"}, usage) == close(standard)


def test_gpt56_custom_cost_with_regional_uplift():
    config = {"apiHost": "eu.api.openai.com", "inputCost": 2 / 1e6}
    assert cost("gpt-5.6", config, cached_responses_usage()) == close(
        (2_000 * 2 + 1_000 * 20 * 1.1) / 1e6
    )


@pytest.mark.parametrize("api_host", ["proxy.api.openai.com", "au.api.openai.com"])
def test_gpt56_no_regional_uplift_for_other_hosts(api_host):
    assert cost("gpt-5.6", {"apiHost": api_host}, cached_responses_usage()) == close(
        (1_250 * 4 + 500 * 0.4 + 250 * 5 + 1_000 * 20) / 1e6
    )


@pytest.mark.parametrize(
    ("model", "inp", "cached", "out"),
    [
        *GPT_56,
        ("openai.gpt-5.6-sol", 4.4, 0.44, 22),
        ("openai.gpt-5.6-terra", 2.2, 0.22, 13.2),
        ("openai.gpt-5.6-luna", 0.22, 0.022, 1.32),
        ("openai.gpt-5.5", 5.5, 0.55, 33),
        ("openai.gpt-5.4", 2.75, 0.275, 16.5),
    ],
)
def test_summarized_usage_only_when_cache_write_rates_match(model, inp, cached, out):
    result = calculate_openai_usage_cost_from_token_usage(
        model, {"prompt": 2_000, "completion": 1_000, "cached": 500}
    )
    if "gpt-5.6" in model:
        assert result is None
    else:
        assert result == close((1_500 * inp + 500 * cached + 1_000 * out) / 1e6)


def test_gpt56_summarized_usage_with_known_cache_writes():
    assert calculate_openai_usage_cost_from_token_usage(
        "gpt-5.6-sol",
        {"prompt": 2_000, "completion": 1_000, "cached": 500,
         "completionDetails": {"cacheCreationInputTokens": 250}},
    ) == close((1_250 * 4 + 500 * 0.4 + 250 * 5 + 1_000 * 20) / 1e6)  # fmt: skip


def test_gpt56_flex_and_fast_long_context():
    usage = {
        "input_tokens": 300_000,
        "output_tokens": 1_000,
        "input_tokens_details": {"cached_tokens": 100_000, "cache_write_tokens": 50_000},
    }
    assert cost("gpt-5.6-terra", {}, usage) == close(
        (150_000 * 4 + 100_000 * 0.4 + 50_000 * 5 + 1_000 * 18) / 1e6
    )
    assert cost("gpt-5.6-terra", {}, usage, {"serviceTier": "flex"}) == close(
        (150_000 * 2 + 100_000 * 0.2 + 50_000 * 2.5 + 1_000 * 9) / 1e6
    )
    assert cost("gpt-5.6-terra", {}, usage, {"serviceTier": "fast"}) == close(
        (150_000 * 8 + 100_000 * 0.8 + 50_000 * 10 + 1_000 * 36) / 1e6
    )


@pytest.mark.parametrize(
    ("prompt", "inp", "cached", "write", "out"),
    [(272_000, 4.4, 0.44, 5.5, 22), (272_001, 8.8, 0.88, 11, 33)],
)
def test_bedrock_sol_summarized_boundary(prompt, inp, cached, write, out):
    assert calculate_openai_usage_cost_from_token_usage(
        "openai.gpt-5.6-sol",
        {"prompt": prompt, "completion": 1_000, "cached": 100_000,
         "completionDetails": {"cacheCreationInputTokens": 50_000}},
    ) == close(
        ((prompt - 150_000) * inp + 100_000 * cached + 50_000 * write + 1_000 * out) / 1e6
    )  # fmt: skip


@pytest.mark.parametrize(
    ("model", "inp", "cached", "write", "out"),
    [
        ("gpt-5.6-sol", 8, 0.8, 10, 40),
        ("gpt-5.6-terra", 4, 0.4, 5, 24),
        ("gpt-5.6-luna", 0.4, 0.04, 0.5, 2.4),
    ],
)
def test_gpt56_fast_rates(model, inp, cached, write, out):
    assert cost(model, {}, cached_responses_usage(), {"serviceTier": "fast"}) == close(
        (1_250 * inp + 500 * cached + 250 * write + 1_000 * out) / 1e6
    )


@pytest.mark.parametrize(
    ("model", "tier", "input_tokens", "cached_tokens", "write_tokens", "rates"),
    [
        ("gpt-5.6-terra", "batch", 2_000, 500, 250, (1, 0.1, 1.25, 6)),
        ("gpt-5.6-terra", "flex", 2_000, 500, 250, (1, 0.1, 1.25, 6)),
        ("gpt-5.6-terra", "batch", 300_000, 100_000, 50_000, (2, 0.2, 2.5, 9)),
        ("gpt-5.6-terra", "flex", 300_000, 100_000, 50_000, (2, 0.2, 2.5, 9)),
        ("gpt-5.6-luna", "batch", 2_000, 500, 250, (0.1, 0.01, 0.125, 0.6)),
        ("gpt-5.6-luna", "flex", 2_000, 500, 250, (0.1, 0.01, 0.125, 0.6)),
        ("gpt-5.6-luna", "batch", 300_000, 100_000, 50_000, (0.2, 0.02, 0.25, 0.9)),
        ("gpt-5.6-luna", "flex", 300_000, 100_000, 50_000, (0.2, 0.02, 0.25, 0.9)),
    ],
)
def test_gpt56_batch_flex_rates(model, tier, input_tokens, cached_tokens, write_tokens, rates):
    inp, cached, write, out = rates
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": 1_000,
        "input_tokens_details": {
            "cached_tokens": cached_tokens,
            "cache_write_tokens": write_tokens,
        },
    }
    uncached = input_tokens - cached_tokens - write_tokens
    assert cost(model, {}, usage, {"serviceTier": tier}) == close(
        (uncached * inp + cached_tokens * cached + write_tokens * write + 1_000 * out) / 1e6
    )


@pytest.mark.parametrize(
    ("model", "inp", "cached", "write", "out"),
    [("gpt-5.6-terra", 4, 0.4, 5, 24), ("gpt-5.6-luna", 0.4, 0.04, 0.5, 2.4)],
)
def test_gpt56_fast_across_threshold(model, inp, cached, write, out):
    usage = {
        "input_tokens": 272_000,
        "output_tokens": 1_000,
        "input_tokens_details": {"cached_tokens": 100_000, "cache_write_tokens": 50_000},
    }
    for tier in ("fast", "priority"):
        assert cost(model, {}, usage, {"serviceTier": tier}) == close(
            (122_000 * inp + 100_000 * cached + 50_000 * write + 1_000 * out) / 1e6
        )
        assert cost(model, {}, {**usage, "input_tokens": 272_001}, {"serviceTier": tier}) == close(
            (122_001 * inp * 2 + 100_000 * cached * 2 + 50_000 * write * 2 + 1_000 * out * 1.5)
            / 1e6
        )


def test_gpt56_explicit_cache_writes():
    assert cost("gpt-5.6", {}, cached_responses_usage()) == close(
        (1_250 * 4 + 500 * 0.4 + 250 * 5 + 1_000 * 20) / 1e6
    )


def test_chat_latest_cached_input():
    assert cost("chat-latest", {}, cached_chat_usage()) == close(
        (1_500 * 5 + 500 * 0.5 + 1_000 * 30) / 1e6
    )


def test_returned_service_tiers_for_flex_and_fast():
    usage = cached_input_usage()
    assert cost("gpt-5-mini", {}, usage, {"serviceTier": "flex"}) == close(
        (600 * 0.125 + 400 * 0.0125 + 100 * 1) / 1e6
    )
    for tier in ("priority", "fast"):
        assert cost("gpt-5-mini", {}, usage, {"serviceTier": tier}) == close(
            (600 * 0.45 + 400 * 0.045 + 100 * 3.6) / 1e6
        )


@pytest.mark.parametrize("tier", ["batch", "flex"])
def test_no_estimate_for_gpt55_pro_long_context_discount_tiers(tier):
    usage = {"input_tokens": 300_000, "output_tokens": 1_000}
    assert cost("gpt-5.5-pro", {}, usage, {"serviceTier": tier}) is None


@pytest.mark.parametrize(
    ("tier", "config", "expected"),
    [
        ("batch", {"cost": 2 / 1e6}, (301_000 * 2) / 1e6),
        ("flex", {"cost": 2 / 1e6}, (301_000 * 2) / 1e6),
        ("batch", {"inputCost": 2 / 1e6, "outputCost": 7 / 1e6}, (300_000 * 2 + 1_000 * 7) / 1e6),
        ("flex", {"inputCost": 2 / 1e6, "outputCost": 7 / 1e6}, (300_000 * 2 + 1_000 * 7) / 1e6),
    ],
)
def test_explicit_costs_for_gpt55_pro_long_context(tier, config, expected):
    usage = {
        "input_tokens": 300_000,
        "output_tokens": 1_000,
        "input_tokens_details": {"cached_tokens": 100_000},
    }
    assert cost("gpt-5.5-pro", config, usage, {"serviceTier": tier}) == close(expected)


def test_no_flex_pricing_for_unsupported_models():
    usage = {"input_tokens": 1_000, "output_tokens": 100}
    assert cost("gpt-4.1", {}, usage, {"serviceTier": "flex"}) is None


def test_batch_discount_on_standard_rates():
    assert cost("gpt-5-mini", {}, cached_input_usage(), {"serviceTier": "batch"}) == close(
        (600 * 0.125 + 400 * 0.0125 + 100 * 1) / 1e6
    )
    assert cost(
        "text-embedding-3-large",
        {},
        {"prompt_tokens": 10, "total_tokens": 10},
        {"serviceTier": "batch"},
    ) == close((10 * 0.065) / 1e6, 12)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"inputCost": 2 / 1e6, "outputCost": 7 / 1e6}, (2_000 * 2 + 1_000 * 7) / 1e6),
        ({"cost": 3 / 1e6}, (2_000 * 3 + 1_000 * 3) / 1e6),
    ],
)
def test_unknown_namespaced_gateway_model_with_explicit_rates(config, expected):
    assert cost("third-party/gpt-4o", config, cached_chat_usage()) == close(expected)


def test_cached_responses_for_unknown_models_unpriced():
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    assert cost("third-party/model", {}, usage, {"cachedResponse": True}) is None


MULTIMODAL_CHAT_USAGE = {
    "prompt_tokens": 30,
    "completion_tokens": 23,
    "prompt_tokens_details": {"text_tokens": 21, "audio_tokens": 9},
    "completion_tokens_details": {"text_tokens": 16, "audio_tokens": 7},
}


def test_unknown_gateway_multimodal_usage_needs_audio_rates():
    gateway = {"apiUrl": "https://gateway.example/v1"}
    text_rates = {"inputCost": 2 / 1e6, "outputCost": 7 / 1e6}
    assert cost("third-party/gpt-4o", text_rates, MULTIMODAL_CHAT_USAGE, gateway) is None
    all_rates = {**text_rates, "audioInputCost": 11 / 1e6, "audioOutputCost": 19 / 1e6}
    assert cost("third-party/gpt-4o", all_rates, MULTIMODAL_CHAT_USAGE, gateway) == close(
        (21 * 2 + 9 * 11 + 16 * 7 + 7 * 19) / 1e6, 12
    )


def test_explicit_audio_rates_with_known_text_model():
    config = {"audioInputCost": 11 / 1e6, "audioOutputCost": 19 / 1e6}
    assert cost("gpt-4o", config, MULTIMODAL_CHAT_USAGE) == close(
        (21 * 2.5 + 9 * 11 + 16 * 10 + 7 * 19) / 1e6, 12
    )


@pytest.mark.parametrize("direction", ["input", "output"])
def test_unknown_gateway_image_tokens_not_priced_as_text(direction):
    usage = {
        "prompt_tokens": 30,
        "completion_tokens": 23,
        "prompt_tokens_details": {"image_tokens": 10 if direction == "input" else 0},
        "completion_tokens_details": {"image_tokens": 10 if direction == "output" else 0},
    }
    config = {"inputCost": 2 / 1e6, "outputCost": 7 / 1e6}
    assert cost("third-party/image-model", config, usage) is None


@pytest.mark.parametrize(
    ("region", "input_tokens", "inp", "cached", "write", "out"),
    [
        ("us-gov-west-1", 272_000, 2.64, 0.264, 3.3, 15.84),
        ("us-gov-west-1", 272_001, 5.28, 0.528, 6.6, 23.76),
        ("us-gov-east-1", 272_000, 2.64, 0.264, 3.3, 15.84),
    ],
)
def test_terra_govcloud_explicit_bedrock_marker(region, input_tokens, inp, cached, write, out):
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 100_000, "cache_write_tokens": 50_000},
    }
    options = {"apiUrl": f"https://bedrock-mantle.{region}.api.aws/v1"}
    assert cost("bedrock:gpt-5.6-terra", {}, usage, options) == close(
        ((input_tokens - 150_000) * inp + 100_000 * cached + 50_000 * write + 1000 * out) / 1e6
    )


# --- custom rates for unknown models --------------------------------------------------------

CUSTOM_USAGE = {
    "prompt_tokens": 2000,
    "completion_tokens": 1000,
    "prompt_tokens_details": {"cached_tokens": 500, "cache_write_tokens": 250},
}


@pytest.mark.parametrize("tier", ["standard", "batch", "flex", "priority"])
def test_custom_rates_without_assumed_discounts(tier):
    config = {"inputCost": 2 / 1e6, "outputCost": 3 / 1e6}
    assert cost("custom-text-model", config, CUSTOM_USAGE, {"serviceTier": tier}) == close(0.007)


def test_custom_shared_rate_with_direction_overrides():
    assert cost("custom-text-model", {"cost": 2 / 1e6}, CUSTOM_USAGE) == close(0.006)
    config = {"cost": 2 / 1e6, "inputCost": 0, "outputCost": 3 / 1e6}
    assert cost("custom-text-model", config, CUSTOM_USAGE) == close(0.003)


@pytest.mark.parametrize("config", [{}, {"inputCost": 2 / 1e6}, {"outputCost": 3 / 1e6}])
def test_custom_rates_not_invented(config):
    assert cost("custom-text-model", config, CUSTOM_USAGE) is None


def test_custom_rate_required_only_for_used_directions():
    assert cost(
        "custom-embedding-model",
        {"inputCost": 2 / 1e6},
        {"prompt_tokens": 2000, "completion_tokens": 0},
    ) == close(0.004)
    assert cost(
        "custom-output-model", {"outputCost": 3 / 1e6}, {"input_tokens": 0, "output_tokens": 1000}
    ) == close(0.003)


def test_custom_rates_not_charged_on_local_cache_hit():
    config = {"inputCost": 2 / 1e6, "outputCost": 3 / 1e6}
    assert cost("custom-text-model", config, CUSTOM_USAGE, {"cachedResponse": True}) == 0


def test_custom_text_and_audio_priced_separately():
    usage = {
        "prompt_tokens": 30,
        "completion_tokens": 23,
        "prompt_tokens_details": {"text_tokens": 21, "audio_tokens": 9, "cached_tokens": 5},
        "completion_tokens_details": {"text_tokens": 16, "audio_tokens": 7},
    }
    config = {
        "inputCost": 2 / 1e6,
        "outputCost": 3 / 1e6,
        "audioInputCost": 20 / 1e6,
        "audioOutputCost": 30 / 1e6,
    }
    assert cost("custom-audio-model", config, usage) == close(
        (21 * 2 + 9 * 20 + 16 * 3 + 7 * 30) / 1e6
    )
    assert cost("custom-audio-model", config, usage, {"cachedResponse": True}) == 0


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"audioCost": 20 / 1e6}, (9 * 20 + 7 * 20) / 1e6),
        ({"audioCost": 20 / 1e6, "audioInputCost": 0, "audioOutputCost": 30 / 1e6}, (7 * 30) / 1e6),
    ],
)
def test_custom_audio_only_usage(config, expected):
    usage = {
        "input_tokens": 9,
        "output_tokens": 7,
        "input_tokens_details": {"audio_tokens": 9},
        "output_tokens_details": {"audio_tokens": 7},
    }
    assert cost("custom-audio-model", config, usage) == close(expected)


AUDIO_INPUT_USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 500,
    "prompt_tokens_details": {
        "text_tokens": 250,
        "audio_tokens": 750,
        "cached_tokens": 100,
        "cached_tokens_details": {"audio_tokens": 100},
    },
    "completion_tokens_details": {"text_tokens": 500, "audio_tokens": 0},
}
AUDIO_OUTPUT_USAGE = {
    "input_tokens": 1000,
    "output_tokens": 500,
    "input_tokens_details": {"text_tokens": 1000, "audio_tokens": 0},
    "output_tokens_details": {"text_tokens": 100, "audio_tokens": 400},
}


@pytest.mark.parametrize(
    ("config", "usage", "expected"),
    [
        ({"cost": 2 / 1e6, "audioInputCost": 10 / 1e6}, AUDIO_INPUT_USAGE, 0.009),
        ({"cost": 2 / 1e6, "audioOutputCost": 20 / 1e6}, AUDIO_OUTPUT_USAGE, 0.0102),
        ({"cost": 2 / 1e6, "audioInputCost": 0}, AUDIO_INPUT_USAGE, 0.0015),
        ({"cost": 2 / 1e6, "audioOutputCost": 0}, AUDIO_OUTPUT_USAGE, 0.0022),
        ({"cost": 0, "audioInputCost": 0}, AUDIO_INPUT_USAGE, 0),
    ],
    ids=["audio-input", "audio-output", "zero-audio-input", "zero-audio-output", "all-zero"],
)
def test_custom_audio_rates_for_used_directions(config, usage, expected):
    assert cost("gateway/custom-audio", config, usage) == close(expected, 12)


@pytest.mark.parametrize(
    ("config", "usage"),
    [
        ({"cost": 2 / 1e6, "audioOutputCost": 20 / 1e6}, AUDIO_INPUT_USAGE),
        ({"cost": 2 / 1e6, "audioInputCost": 10 / 1e6}, AUDIO_OUTPUT_USAGE),
    ],
    ids=["input", "output"],
)
def test_custom_audio_missing_used_rate(config, usage):
    assert cost("gateway/custom-audio", config, usage) is None


def test_custom_audio_cached_and_no_audio_cases():
    config = {"cost": 2 / 1e6, "audioInputCost": 10 / 1e6}
    assert cost("gateway/custom-audio", config, AUDIO_INPUT_USAGE, {"cachedResponse": True}) == 0
    assert cost("gateway/custom-audio", config, CUSTOM_USAGE) == close(0.006, 12)


@pytest.mark.parametrize(
    "config",
    [
        {"inputCost": 2 / 1e6, "outputCost": 3 / 1e6},
        {"cost": 2 / 1e6, "audioInputCost": 20 / 1e6},
        {"audioCost": 20 / 1e6},
    ],
)
def test_custom_modality_rates_not_invented(config):
    assert cost("custom-audio-model", config, MULTIMODAL_CHAT_USAGE) is None


def test_audio_preview_text_and_audio_priced_separately():
    usage = {
        "prompt_tokens": 30,
        "completion_tokens": 23,
        "prompt_tokens_details": {"text_tokens": 21, "audio_tokens": 9},
        "completion_tokens_details": {"text_tokens": 16, "audio_tokens": 7},
    }
    assert cost("gpt-4o-mini-audio-preview", {}, usage, {}) == close(
        (21 * 0.15 + 9 * 10 + 16 * 0.6 + 7 * 20) / 1e6
    )


# --- embeddings, cache hits and tool fees ---------------------------------------------------


def test_embeddings_priced_from_prompt_tokens():
    assert cost(
        "text-embedding-3-large", {}, {"prompt_tokens": 10, "total_tokens": 10}, {}
    ) == close((10 * 0.13) / 1e6, 12)


def test_empty_usage_object_is_still_priced():
    # `!rawUsage` is false for `{}` in JS, so an empty ledger prices as zero tokens.
    assert cost("gpt-4o", {}, {}) == 0
    assert cost("gpt-4o", {}, None) is None


def test_promptfoo_cache_hits_not_billed_again():
    usage = {"prompt_tokens": 1_000, "completion_tokens": 100}
    assert cost("gpt-5-mini", {}, usage, {"cachedResponse": True}) == 0


def test_observable_tool_fees():
    preview = {"tools": [{"type": "web_search_preview"}]}
    search = {"type": "web_search_call", "action": {"type": "search"}}
    assert calculate_observable_openai_tool_cost(
        {"output": [search, search, {"type": "file_search_call"}]}, "gpt-5-mini", preview
    ) == close(0.0225)
    assert calculate_observable_openai_tool_cost({"output": [search]}, "gpt-4o", preview) == close(
        0.025
    )
    assert calculate_observable_openai_tool_cost(
        {"output": [search]}, "gpt-4o", {"tools": [{"type": "web_search"}]}
    ) == close(0.01)
    assert calculate_observable_openai_tool_cost(
        {"output": [search]}, "ft:o4-mini-2025-04-16:company::model", preview
    ) == close(0.01)


@pytest.mark.parametrize("model", ["gpt-daybreak-blue-latest", "gpt-daybreak-red-latest"])
def test_daybreak_reasoning_web_search_pricing(model):
    response = {"output": [{"type": "web_search_call", "action": {"type": "search"}}]}
    for config in (
        {"tools": [{"type": "web_search_preview"}]},
        {},
        {"tools": [{"type": "web_search"}]},
    ):
        assert calculate_observable_openai_tool_cost(response, model, config) == close(0.01)


def test_non_search_web_actions_are_free():
    data = {
        "output": [
            {"type": "web_search_call", "action": {"type": "open_page"}},
            {"type": "web_search_call", "action": {"type": "find_in_page"}},
        ]
    }
    assert (
        calculate_observable_openai_tool_cost(
            data, "o3", {"tools": [{"type": "web_search_preview"}]}
        )
        == 0
    )
