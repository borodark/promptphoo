"""OpenAI util helpers: model catalogs, pricing, token usage, errors and validation. Ports
test/providers/openai/util.test.ts (the SDK APIError case uses a duck-typed error)."""

from __future__ import annotations

import json
import sys

import pytest

from promptphoo.providers.openai.util import (
    OPENAI_BILLING_MODELS,
    OPENAI_CHAT_MODELS,
    OPENAI_CODEX_ONLY_MODELS,
    OPENAI_COMPLETION_MODELS,
    OPENAI_DEEP_RESEARCH_MODELS,
    OPENAI_REALTIME_MODELS,
    OPENAI_RESPONSES_ONLY_MODELS,
    OPENAI_TRANSCRIPTION_MODELS,
    OPENAI_TTS_MODELS,
    RETIRED_OPENAI_MODEL_IDS,
    append_openai_api_path,
    assert_openai_api_model,
    calculate_openai_cost,
    calculate_safe_openai_cost,
    fail_api_call,
    format_openai_error,
    get_chat_completion_refusal,
    get_openai_effective_service_tier,
    get_token_usage,
    get_token_usage_with_request_count,
    is_openai_error_only_response,
    normalize_openai_service_tier_for_wire,
    validate_chat_completion_message,
    validate_function_call,
)

MAX_SAFE_INTEGER = 2**53 - 1
REGIONAL_HOSTS = ["us", "eu", "au", "ca", "jp", "in", "sg", "kr", "gb", "ae"]


def ids(models):
    return {m["id"] for m in models}


RETIRED_CHAT_MODEL_IDS = [
    "chatgpt-4o-latest",
    "o1-preview",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-4-turbo-preview",
    "gpt-4-0125-preview",
    "gpt-4-1106-vision-preview",
    "gpt-4-vision-preview",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4o-search-preview-2025-03-11",
    "gpt-4o-mini-search-preview-2025-03-11",
    "gpt-4o-audio-preview",
    "gpt-4o-audio-preview-2024-10-01",
    "gpt-4o-audio-preview-2024-12-17",
    "gpt-4o-audio-preview-2025-06-03",
    "gpt-4o-mini-audio-preview",
    "gpt-5-chat-latest",
    "codex-mini-latest",
    "gpt-5.1-chat-latest",
    "gpt-audio-mini-2025-10-06",
]
RETIRED_RESPONSES_MODEL_IDS = [
    "computer-use-preview",
    "computer-use-preview-2025-03-11",
    "gpt-5-codex",
    "gpt-5.1-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5.2-codex",
    "o3-deep-research",
    "o3-deep-research-2025-06-26",
    "o4-mini-deep-research",
    "o4-mini-deep-research-2025-06-26",
]
OTHER_RETIRED_MODEL_IDS = [
    "gpt-4o-realtime-preview",
    "gpt-4o-realtime-preview-2024-10-01",
    "gpt-4o-realtime-preview-2024-12-17",
    "gpt-4o-realtime-preview-2025-06-03",
    "gpt-4o-mini-realtime-preview",
    "gpt-realtime-mini-2025-10-06",
    "text-moderation-007",
    "text-moderation-latest",
    "text-moderation-stable",
]


# --- service tiers --------------------------------------------------------------------------


def test_effective_service_tier_layer_precedence():
    provider_config = {"service_tier": "default", "passthrough": {"service_tier": "priority"}}
    assert get_openai_effective_service_tier(provider_config) == "priority"
    assert get_openai_effective_service_tier(provider_config, {"service_tier": "flex"}) == "flex"
    assert (
        get_openai_effective_service_tier(
            provider_config, {"service_tier": "flex", "passthrough": {"service_tier": "fast"}}
        )
        == "fast"
    )
    assert (
        get_openai_effective_service_tier(
            provider_config, {"passthrough": {"model": "gpt-5.6-luna"}}
        )
        == "default"
    )


def test_normalize_service_tier_for_wire():
    assert normalize_openai_service_tier_for_wire("fast") == "priority"
    for host in ["api.openai.com", *(f"{r}.api.openai.com" for r in REGIONAL_HOSTS)]:
        assert normalize_openai_service_tier_for_wire("fast", f"https://{host}/v1") == "priority"
    assert normalize_openai_service_tier_for_wire("fast", "https://gateway.example/v1") == "fast"
    assert normalize_openai_service_tier_for_wire("priority") == "priority"
    assert normalize_openai_service_tier_for_wire("flex") == "flex"
    assert normalize_openai_service_tier_for_wire(None) is None


# --- failApiCall ----------------------------------------------------------------------------


class FakeApiError(Exception):
    def __init__(self, status: int, error_type: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.type = error_type


def test_fail_api_call_formats_sdk_errors():
    assert fail_api_call(FakeApiError(400, "invalid_request_error", "Bad request")) == {
        "error": "API error: invalid_request_error 400 Bad request"
    }


def test_fail_api_call_formats_generic_errors():
    assert fail_api_call(Exception("Network error")) == {"error": "API error: Error: Network error"}


# --- getTokenUsage --------------------------------------------------------------------------


def test_token_usage_fresh():
    data = {"usage": {"total_tokens": 100, "prompt_tokens": 40, "completion_tokens": 60}}
    assert get_token_usage(data, False) == {
        "total": 100,
        "prompt": 40,
        "completion": 60,
        "numRequests": 1,
    }


def test_token_usage_cached():
    assert get_token_usage({"usage": {"total_tokens": 100}}, True) == {"cached": 100, "total": 100}


def test_token_usage_missing():
    assert get_token_usage({}, False) == {}


def test_token_usage_completion_details():
    data = {
        "usage": {
            "total_tokens": 100,
            "prompt_tokens": 40,
            "completion_tokens": 60,
            "completion_tokens_details": {
                "reasoning_tokens": 20,
                "accepted_prediction_tokens": 30,
                "rejected_prediction_tokens": 10,
            },
        }
    }
    assert get_token_usage(data, False) == {
        "total": 100,
        "prompt": 40,
        "completion": 60,
        "numRequests": 1,
        "completionDetails": {"reasoning": 20, "acceptedPrediction": 30, "rejectedPrediction": 10},
    }


def test_token_usage_reads_output_tokens_details():
    data = {
        "usage": {
            "total_tokens": 45,
            "prompt_tokens": 15,
            "completion_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 100},
        }
    }
    assert get_token_usage(data, False) == {
        "total": 45,
        "prompt": 15,
        "completion": 30,
        "numRequests": 1,
        "completionDetails": {"reasoning": 100},
    }


def test_token_usage_prefers_completion_tokens_details():
    data = {
        "usage": {
            "total_tokens": 45,
            "prompt_tokens": 15,
            "completion_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 7},
            "output_tokens_details": {"reasoning_tokens": 100},
        }
    }
    assert get_token_usage(data, False)["completionDetails"] == {"reasoning": 7}


def test_token_usage_preserves_cache_read_and_write():
    data = {
        "usage": {
            "total_tokens": 100,
            "prompt_tokens": 40,
            "completion_tokens": 60,
            "prompt_tokens_details": {"cached_tokens": 32, "cache_write_tokens": 4},
        }
    }
    assert get_token_usage(data, False) == {
        "total": 100,
        "prompt": 40,
        "completion": 60,
        "numRequests": 1,
        "completionDetails": {"cacheReadInputTokens": 32, "cacheCreationInputTokens": 4},
    }


def test_token_usage_drops_malformed_fields():
    data = {
        "usage": {
            "total_tokens": sys.float_info.max,
            "prompt_tokens": -1,
            "completion_tokens": 2,
            "completion_tokens_details": {
                "reasoning_tokens": "private",
                "accepted_prediction_tokens": 1,
            },
        }
    }
    assert get_token_usage(data, False) == {
        "prompt": 0,
        "completion": 2,
        "numRequests": 1,
        "completionDetails": {"acceptedPrediction": 1},
    }


def nested(depth: int = 5_000):
    value: object = 1
    for _ in range(depth):
        value = [value]
    return value


def test_token_usage_bounds_adversarial_nesting():
    data = {"usage": {"prompt_tokens_details": {"cached_tokens": nested()}}}
    assert get_token_usage(data, False) == {"prompt": 0, "completion": 0, "numRequests": 1}


def test_token_usage_reads_top_level_cached_tokens():
    data = {
        "usage": {
            "total_tokens": 100,
            "prompt_tokens": 40,
            "completion_tokens": 60,
            "cached_tokens": 32,
        }
    }
    assert get_token_usage(data, False) == {
        "total": 100,
        "prompt": 40,
        "completion": 60,
        "numRequests": 1,
        "completionDetails": {"cacheReadInputTokens": 32},
    }


def test_token_usage_prefers_nested_cached_tokens():
    data = {
        "usage": {
            "total_tokens": 100,
            "prompt_tokens": 40,
            "completion_tokens": 60,
            "cached_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 32},
        }
    }
    assert get_token_usage(data, False)["completionDetails"] == {"cacheReadInputTokens": 32}


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        ({"cache_write_input_tokens": 4}, 4),
        ({"prompt_tokens_details": {"cache_write_tokens": 0}}, 0),
    ],
    ids=["top-level-positive", "nested-explicit-zero"],
)
def test_token_usage_cache_write(details, expected):
    data = {"usage": {"total_tokens": 100, "prompt_tokens": 40, "completion_tokens": 60, **details}}
    assert get_token_usage(data, False)["completionDetails"] == {
        "cacheCreationInputTokens": expected
    }


def test_token_usage_with_request_count():
    data = {"usage": {"total_tokens": 100, "prompt_tokens": 40, "completion_tokens": 60}}
    assert get_token_usage_with_request_count(data, False) == {
        "total": 100,
        "prompt": 40,
        "completion": 60,
        "numRequests": 1,
    }
    assert get_token_usage_with_request_count({}, True) == {"numRequests": 0}


@pytest.mark.parametrize(
    "usage",
    [
        {
            "total_tokens": {"private": "must-not-appear"},
            "prompt_tokens": {"private": "must-not-appear"},
            "completion_tokens": {"private": "must-not-appear"},
        },
        {"prompt_tokens_details": {"cached_tokens": nested()}},
    ],
    ids=["objects", "adversarial-nesting"],
)
def test_token_usage_with_request_count_drops_malformed(usage):
    assert get_token_usage_with_request_count({"usage": usage}, False) == {
        "prompt": 0,
        "completion": 0,
        "numRequests": 1,
    }


@pytest.mark.parametrize(
    "usage",
    [
        {"total_tokens": sys.float_info.max, "prompt_tokens": -1, "completion_tokens": 2},
        {
            "total_tokens": MAX_SAFE_INTEGER,
            "prompt_tokens": MAX_SAFE_INTEGER,
            "completion_tokens": 2,
        },
    ],
    ids=["negative-and-unsafe", "absurd-safe-integers"],
)
def test_token_usage_with_request_count_drops_unsafe_counts(usage):
    assert get_token_usage_with_request_count({"usage": usage}, False) == {
        "prompt": 0,
        "completion": 2,
        "numRequests": 1,
    }


# --- calculateSafeOpenAICost ----------------------------------------------------------------

SAFE_CONFIG = {"inputCost": 0.001, "outputCost": 0.002}


def test_safe_cost_valid_usage():
    data = {"usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    assert calculate_safe_openai_cost("gpt-4o", SAFE_CONFIG, data) == pytest.approx(0.007)


def test_safe_cost_prefers_provider_reported_cost_without_overrides():
    data = {"usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.0042}}
    assert calculate_safe_openai_cost("openai/gpt-4o", {}, data) == 0.0042


@pytest.mark.parametrize("model", ["gpt-4o", "openai/gpt-4o"])
def test_safe_cost_local_overrides_beat_reported_cost(model):
    data = {"usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 99}}
    assert calculate_safe_openai_cost(model, SAFE_CONFIG, data) == pytest.approx(0.007)


def test_safe_cost_no_fallback_when_override_unsafe():
    data = {"usage": {"prompt_tokens": -1, "completion_tokens": 2, "cost": 99}}
    assert calculate_safe_openai_cost("openai/gpt-4o", SAFE_CONFIG, data) is None


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": -2, "completion_tokens": -1},
        {"prompt_tokens": 1e308, "completion_tokens": 1e308},
        {"prompt_tokens": 3, "completion_tokens": "2"},
    ],
    ids=["negative", "unsafe", "mixed-type"],
)
def test_safe_cost_omitted_for_bad_counts(usage):
    assert calculate_safe_openai_cost("gpt-4o", SAFE_CONFIG, {"usage": usage}) is None


@pytest.mark.parametrize("cost", [-1, float("inf"), "0.01"])
def test_safe_cost_rejects_unsafe_reported_cost(cost):
    assert calculate_safe_openai_cost("openai/unknown-model", {}, {"usage": {"cost": cost}}) is None


# --- validateChatCompletionMessage ----------------------------------------------------------


def test_validate_message_accepts_custom_tool_call():
    tool_call = {"id": "call_custom", "type": "custom", "custom": {"name": "shell", "input": "ls"}}
    message = validate_chat_completion_message({"tool_calls": [tool_call]})
    assert message is not None
    assert message["toolCalls"] == [tool_call]


def test_validate_message_requires_usable_payload():
    assert validate_chat_completion_message({"content": None}) is None
    assert validate_chat_completion_message({"content": ""}) is None
    assert validate_chat_completion_message({"content": []}, allow_structured_content=True) is None


def test_validate_message_audio_only_when_allowed():
    audio = {"id": "audio-1", "data": "UklGRg==", "transcript": "Hi"}
    assert validate_chat_completion_message({"content": None, "audio": audio}) is None
    message = validate_chat_completion_message({"content": None, "audio": audio}, allow_audio=True)
    assert message is not None
    assert message["audio"] == audio
    assert (
        validate_chat_completion_message({"content": None, "audio": {}}, allow_audio=True) is None
    )


def test_validate_message_structured_content_parts():
    assert (
        validate_chat_completion_message(
            {"content": [{"type": "text"}]}, allow_structured_content=True
        )
        is None
    )
    message = validate_chat_completion_message(
        {"content": [{"type": "text", "text": "Hello"}]}, allow_structured_content=True
    )
    assert message is not None
    assert message["structuredContent"] == [{"type": "text", "text": "Hello"}]


def test_refusal_and_content_filter_normalization():
    refusal = validate_chat_completion_message({"refusal": "Cannot comply"})
    filtered = validate_chat_completion_message({"content": None}, finish_reason="content_filter")
    assert refusal is not None
    assert filtered is not None
    assert get_chat_completion_refusal(refusal, "stop") == {
        "output": "Cannot comply",
        "isRefusal": True,
        "guardrails": {"flagged": True},
    }
    assert get_chat_completion_refusal(filtered, "content_filter") == {
        "output": "Content filtered by provider",
        "isRefusal": True,
        "guardrails": {"flagged": True},
    }


# --- calculateOpenAICost --------------------------------------------------------------------


def test_cost_rejects_incomplete_audio_usage():
    assert calculate_openai_cost("gpt-4o-audio-preview", {}, 1000, 500, 0) is None
    assert calculate_openai_cost("gpt-4o-audio-preview", {}, 1000, 500, None, 0) is None
    assert calculate_openai_cost("gpt-4o-audio-preview", {}, 1000, 500, None, 100) is None


def test_cost_long_context_rates_with_audio_usage():
    text_cost = calculate_openai_cost("gpt-6-astra", {}, 300_000, 500)
    assert calculate_openai_cost("gpt-6-astra", {}, 300_000, 500, 1, 0) == text_cost


@pytest.mark.parametrize("model", ["gpt-4o-mini-tts", "gpt-4o-mini-tts-2025-12-15"])
def test_tts_models_priced_but_not_chat(model):
    assert model in ids(OPENAI_TTS_MODELS)
    assert model not in ids(OPENAI_CHAT_MODELS)
    assert calculate_openai_cost(model, {}, 1000, 0, 0, 500) == pytest.approx(
        (1000 * 0.6 + 500 * 12) / 1e6, abs=1e-6
    )


@pytest.mark.parametrize("model", ["tts-1", "tts-1-1106", "tts-1-hd", "tts-1-hd-1106"])
def test_legacy_speech_models_not_chat(model):
    assert model in ids(OPENAI_TTS_MODELS)
    assert model not in ids(OPENAI_CHAT_MODELS)


def test_codex_model_classification():
    assert "gpt-5-codex-mini" not in ids(OPENAI_CHAT_MODELS)
    assert "gpt-5-codex-mini" in ids(OPENAI_RESPONSES_ONLY_MODELS)
    assert "gpt-5-codex-mini" not in ids(OPENAI_CODEX_ONLY_MODELS)
    assert calculate_openai_cost("gpt-5-codex-mini", {}, 1000, 500) == pytest.approx(
        (1000 * 0.5 + 500 * 2) / 1e6, abs=1e-6
    )
    assert "gpt-5.3-codex-spark" not in ids(OPENAI_CHAT_MODELS)
    assert "gpt-5.3-codex-spark" not in ids(OPENAI_RESPONSES_ONLY_MODELS)
    assert "gpt-5.3-codex-spark" in ids(OPENAI_CODEX_ONLY_MODELS)


@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate"),
    [
        ("gpt-4o-search-preview", 2.5, 10),
        ("gpt-4o-search-preview-2025-03-11", 2.5, 10),
        ("gpt-4o-mini-search-preview", 0.15, 0.6),
        ("gpt-5-search-api", 1.25, 10),
        ("gpt-5-search-api-2025-10-14", 1.25, 10),
        ("computer-use-preview", 3, 12),
        ("gpt-4-1106-vision-preview", 10, 30),
        ("gpt-4o-realtime-preview-2024-10-01", 5, 20),
        ("gpt-4o-realtime-preview-2024-12-17", 5, 20),
        ("gpt-4o-mini-realtime-preview-2024-12-17", 0.6, 2.4),
        ("gpt-realtime", 4, 16),
        ("gpt-realtime-1.5", 4, 16),
        ("gpt-realtime-2025-08-28", 4, 16),
        ("gpt-realtime-2", 4, 24),
        ("chat-latest", 5, 30),
        ("gpt-4", 30, 60),
        ("gpt-4.1", 2, 8),
        ("gpt-3.5-turbo", 0.5, 1.5),
        ("o4-mini", 1.1, 4.4),
        ("codex-mini-latest", 1.5, 6.0),
        ("gpt-5", 1.25, 10),
        ("gpt-5-chat-latest", 1.25, 10),
        ("gpt-5-chat", 1.25, 10),
        ("gpt-5.1-chat-latest", 1.25, 10),
        ("gpt-5.2-chat-latest", 1.75, 14),
        ("gpt-5.3-chat-latest", 1.75, 14),
        ("gpt-5.5", 5, 30),
        ("gpt-6.1-sol", 2, 10),
        ("gpt-5.6", 4, 20),
        ("gpt-5.6-sol", 4, 20),
        ("gpt-5.6-terra", 2, 12),
        ("gpt-5.6-luna", 0.2, 1.2),
        ("gpt-5.5-2026-04-23", 5, 30),
        ("gpt-5.4", 2.5, 15),
        ("gpt-5.4-2026-03-05", 2.5, 15),
        ("gpt-5.4-mini", 0.75, 4.5),
        ("gpt-5.4-mini-2026-03-17", 0.75, 4.5),
        ("gpt-5.4-nano", 0.2, 1.25),
        ("gpt-5.4-nano-2026-03-17", 0.2, 1.25),
        ("gpt-5.2-codex", 1.75, 14),
        ("gpt-5.2-pro", 21, 168),
        ("gpt-5.5-pro", 30, 180),
        ("gpt-5.5-pro-2026-04-23", 30, 180),
        ("gpt-5.4-pro", 30, 180),
        ("gpt-5.4-pro-2026-03-05", 30, 180),
        ("gpt-5-nano", 0.05, 0.4),
        ("gpt-5-mini", 0.25, 2),
        ("babbage-002", 0.4, 0.4),
        ("davinci-002", 2, 2),
        ("gpt-3.5-turbo-instruct", 1.5, 2),
        ("gpt-3.5-turbo-instruct-0914", 1.5, 2),
        ("o1-pro", 150, 600),
        ("o1-pro-2025-03-19", 150, 600),
        ("o3-pro", 20, 80),
        ("o3-pro-2025-06-10", 20, 80),
        ("o3", 2, 8),
        ("o3-2025-04-16", 2, 8),
        ("gpt-4-0314", 30, 60),
        ("gpt-4-32k-0314", 60, 120),
        ("gpt-4-32k-0613", 60, 120),
        ("gpt-4-vision-preview", 10, 30),
        ("gpt-3.5-turbo-0301", 1.5, 2),
        ("gpt-3.5-turbo-16k", 3, 4),
        ("gpt-3.5-turbo-16k-0613", 3, 4),
        ("gpt-4o-audio-preview-2025-06-03", 2.5, 10),
        ("o3-deep-research", 10, 40),
        ("o3-deep-research-2025-06-26", 10, 40),
        ("o4-mini-deep-research", 2, 8),
        ("o4-mini-deep-research-2025-06-26", 2, 8),
    ],
)
def test_text_cost(model, input_rate, output_rate):
    assert calculate_openai_cost(model, {}, 1000, 500) == pytest.approx(
        (1000 * input_rate + 500 * output_rate) / 1e6, abs=1e-6
    )


@pytest.mark.parametrize(
    ("model", "base_in", "base_out", "long_in", "long_out"),
    [
        ("gpt-6.1-sol", 2, 10, 4, 15),
        ("gpt-5.6", 4, 20, 8, 30),
        ("gpt-5.6-sol", 4, 20, 8, 30),
        ("gpt-5.6-terra", 2, 12, 4, 18),
        ("gpt-5.6-luna", 0.2, 1.2, 0.4, 1.8),
    ],
)
def test_long_context_pricing_above_272k(model, base_in, base_out, long_in, long_out):
    assert calculate_openai_cost(model, {}, 272_000, 1_000) == pytest.approx(
        (272_000 * base_in + 1_000 * base_out) / 1e6, abs=1e-6
    )
    assert calculate_openai_cost(model, {}, 272_001, 1_000) == pytest.approx(
        (272_001 * long_in + 1_000 * long_out) / 1e6, abs=1e-6
    )


@pytest.mark.parametrize(
    ("model", "long_in", "long_out"),
    [("gpt-5.5", 10, 45), ("gpt-5.4", 5, 22.5), ("gpt-5.5-pro", 60, 270), ("gpt-5.4-pro", 60, 270)],
)
def test_long_context_cost(model, long_in, long_out):
    assert calculate_openai_cost(model, {}, 300_000, 1_000) == pytest.approx(
        (300_000 * long_in + 1_000 * long_out) / 1e6, abs=1e-6
    )


@pytest.mark.parametrize(
    ("model", "rates"),
    [
        ("gpt-4o-audio-preview", (2.5, 10, 40, 80)),
        ("gpt-audio-1.5", (2.5, 10, 32, 64)),
        ("gpt-audio", (2.5, 10, 32, 64)),
        ("gpt-audio-mini-2025-12-15", (0.6, 2.4, 10, 20)),
        ("gpt-4o-realtime-preview-2024-12-17", (5, 20, 40, 80)),
        ("gpt-4o-mini-realtime-preview-2024-12-17", (0.6, 2.4, 10, 20)),
        ("gpt-realtime", (4, 16, 32, 64)),
        ("gpt-realtime-2", (4, 24, 32, 64)),
        ("gpt-realtime-mini-2025-12-15", (0.6, 2.4, 10, 20)),
    ],
)
def test_audio_cost(model, rates):
    text_in, text_out, audio_in, audio_out = rates
    assert calculate_openai_cost(model, {}, 1000, 500, 200, 100) == pytest.approx(
        (1000 * text_in + 500 * text_out + 200 * audio_in + 100 * audio_out) / 1e6, abs=1e-6
    )


@pytest.mark.parametrize(
    ("model", "rates"),
    [("gpt-realtime-2.1", (4, 24, 32, 64)), ("gpt-realtime-2.1-mini", (0.6, 2.4, 10, 20))],
)
def test_realtime_models_recognized_and_priced(model, rates):
    assert model in ids(OPENAI_REALTIME_MODELS)
    text_in, text_out, audio_in, audio_out = rates
    assert calculate_openai_cost(model, {}, 1000, 500, 200, 100) == pytest.approx(
        (1000 * text_in + 500 * text_out + 200 * audio_in + 100 * audio_out) / 1e6, abs=1e-6
    )


def test_cost_edge_cases():
    assert calculate_openai_cost("gpt-4", {}, None, None) is None
    assert calculate_openai_cost("unknown-model", {}, 100, 50) is None
    assert calculate_openai_cost("non-existent-model-with-no-cost", {}, 1000, 500) is None
    assert calculate_openai_cost("gpt-4.1", {}, 0, 0) == 0
    assert calculate_openai_cost("gpt-4.1", {}, 1000, 0) == pytest.approx(2000 / 1e6)
    assert calculate_openai_cost("gpt-4.1", {}, 0, 1000) == pytest.approx(8000 / 1e6)
    assert calculate_openai_cost("gpt-4.1", {}, 1_000_000, 1_000_000) == pytest.approx(10)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"cost": 0.123}, 184.5),
        ({"inputCost": 0.001, "outputCost": 0.003}, 2.5),
        ({"cost": 0.123, "inputCost": 0.001, "outputCost": 0.003}, 2.5),
        ({"cost": 0.004, "outputCost": 0.001}, 4.5),
    ],
)
def test_custom_text_costs(config, expected):
    assert calculate_openai_cost("gpt-4", config, 1000, 500) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"audioCost": 0.05}, 15.0075),
        ({"audioInputCost": 0.01, "audioOutputCost": 0.03}, 5.0075),
        ({"audioCost": 0.05, "audioInputCost": 0.01, "audioOutputCost": 0.03}, 5.0075),
        ({"audioInputCost": 0.01}, 2.5e-3 + 5e-3 + 0.01 * 200 + 80 * 100 / 1e6),
        ({"audioCost": 0.02, "audioInputCost": 0.01}, 2.5e-3 + 5e-3 + 0.01 * 200 + 0.02 * 100),
    ],
)
def test_custom_audio_costs(config, expected):
    assert calculate_openai_cost(
        "gpt-4o-audio-preview", config, 1000, 500, 200, 100
    ) == pytest.approx(expected)


# --- model catalogs -------------------------------------------------------------------------


def test_gpt_5_x_catalog_membership():
    chat, responses = ids(OPENAI_CHAT_MODELS), ids(OPENAI_RESPONSES_ONLY_MODELS)
    assert {"gpt-5.5", "gpt-5.5-2026-04-23"} <= chat
    assert {"gpt-5.5-pro", "gpt-5.5-pro-2026-04-23"} <= responses
    for model in ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6.1-sol"):
        assert model in chat
        assert model not in responses
    for model in ("gpt-5.4-pro", "gpt-5.4-pro-2026-03-05"):
        assert model not in chat
        assert model in responses
    assert "gpt-5.5-pro" not in chat
    assert "gpt-5.5-pro-2026-04-23" not in chat


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5-pro",
        "gpt-5-pro-2025-10-06",
        "gpt-5.2-pro",
        "gpt-5.2-pro-2025-12-11",
        "gpt-5.3-codex",
        "o1-pro",
        "o1-pro-2025-03-19",
        "o3-pro",
        "o3-pro-2025-06-10",
    ],
)
def test_active_responses_only_models_not_chat(model):
    assert model not in ids(OPENAI_CHAT_MODELS)
    assert model in ids(OPENAI_RESPONSES_ONLY_MODELS)


def test_current_audio_and_realtime_models_routable():
    assert "gpt-audio-1.5" in ids(OPENAI_CHAT_MODELS)
    realtime = ids(OPENAI_REALTIME_MODELS)
    assert {"gpt-realtime-1.5", "gpt-realtime-2", "gpt-4o-mini-realtime-preview-2024-12-17"} <= (
        realtime
    )


def test_retired_audio_and_realtime_previews_keep_billing():
    for model in (m for m in RETIRED_CHAT_MODEL_IDS if "audio-preview" in m):
        assert model not in ids(OPENAI_CHAT_MODELS)
        assert isinstance(calculate_openai_cost(model, {}, 1_000, 500), float)
    for model in (m for m in OTHER_RETIRED_MODEL_IDS if "realtime-preview" in m):
        assert model not in ids(OPENAI_REALTIME_MODELS)
        assert isinstance(calculate_openai_cost(model, {}, 1_000, 500), float)


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o-mini-audio-preview-2024-12-17",
        "gpt-4o-mini-search-preview",
        "gpt-4o-search-preview",
        "gpt-5-chat",
    ],
)
def test_legacy_discovery_exclusions_are_not_shutdowns(model):
    assert model not in ids(OPENAI_CHAT_MODELS)
    assert model not in ids(OPENAI_RESPONSES_ONLY_MODELS)
    assert model not in RETIRED_OPENAI_MODEL_IDS
    assert isinstance(calculate_openai_cost(model, {}, 1_000, 500), float)


def test_july_23_shutdowns_excluded_but_billed():
    for model in (
        "gpt-4o-search-preview-2025-03-11",
        "gpt-4o-mini-search-preview-2025-03-11",
        "gpt-5-chat-latest",
        "gpt-5.1-chat-latest",
        "gpt-audio-mini-2025-10-06",
    ):
        assert model not in ids(OPENAI_CHAT_MODELS)
        assert isinstance(calculate_openai_cost(model, {}, 1_000, 500), float)
    assert "gpt-4o-mini-tts-2025-03-20" in ids(OPENAI_TTS_MODELS)
    assert isinstance(
        calculate_openai_cost("gpt-4o-mini-tts-2025-03-20", {}, 1_000, 0, 0, 500), float
    )
    assert "gpt-realtime-mini-2025-10-06" not in ids(OPENAI_REALTIME_MODELS)
    for model in RETIRED_RESPONSES_MODEL_IDS[:7]:
        assert model not in ids(OPENAI_RESPONSES_ONLY_MODELS)
        assert isinstance(calculate_openai_cost(model, {}, 1_000, 500), float)
    for model in RETIRED_RESPONSES_MODEL_IDS[7:]:
        assert model in ids(OPENAI_DEEP_RESEARCH_MODELS)
        assert model not in ids(OPENAI_RESPONSES_ONLY_MODELS)
        assert isinstance(calculate_openai_cost(model, {}, 1_000, 500), float)


def test_shut_down_models_excluded_from_routing_registries():
    for model in RETIRED_CHAT_MODEL_IDS:
        assert model not in ids(OPENAI_CHAT_MODELS)
    for model in RETIRED_RESPONSES_MODEL_IDS:
        assert model not in ids(OPENAI_RESPONSES_ONLY_MODELS)


def test_retired_models_rejected_only_on_first_party_api():
    for model in [*RETIRED_CHAT_MODEL_IDS, *RETIRED_RESPONSES_MODEL_IDS, *OTHER_RETIRED_MODEL_IDS]:
        assert model in RETIRED_OPENAI_MODEL_IDS
        with pytest.raises(ValueError, match=f"{model} has been retired"):
            assert_openai_api_model(model, "https://api.openai.com/v1")
        assert_openai_api_model(model, "https://gateway.example/v1")


@pytest.mark.parametrize(
    "model", ["gpt-transcribe", "vendor/gpt-transcribe", "gpt-live-transcribe"]
)
def test_custom_endpoints_may_use_own_models(model):
    assert_openai_api_model(model, "https://gateway.example/v1")


@pytest.mark.parametrize("region", REGIONAL_HOSTS)
def test_retired_models_rejected_on_regional_endpoints(region):
    with pytest.raises(ValueError, match="gpt-5-chat-latest has been retired"):
        assert_openai_api_model("gpt-5-chat-latest", f"https://{region}.api.openai.com/v1")


def test_deep_research_export_is_billing_only():
    active = (
        ids(OPENAI_CHAT_MODELS)
        | ids(OPENAI_RESPONSES_ONLY_MODELS)
        | ids(OPENAI_REALTIME_MODELS)
        | ids(OPENAI_COMPLETION_MODELS)
    )
    assert [m["id"] for m in OPENAI_DEEP_RESEARCH_MODELS] == RETIRED_RESPONSES_MODEL_IDS[7:]
    for model in OPENAI_DEEP_RESEARCH_MODELS:
        assert model["id"] not in active
        assert calculate_openai_cost(model["id"], {}, 1000, 500) > 0


ACTIVE_MODEL_IDS = (
    ids(OPENAI_CHAT_MODELS)
    | ids(OPENAI_RESPONSES_ONLY_MODELS)
    | ids(OPENAI_REALTIME_MODELS)
    | ids(OPENAI_COMPLETION_MODELS)
)


@pytest.mark.parametrize(
    "model",
    [
        "chatgpt-4o-latest",
        "codex-mini-latest",
        "computer-use-preview",
        "computer-use-preview-2025-03-11",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0125-preview",
        "gpt-4-0314",
        "gpt-4-1106-vision-preview",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-4-32k-0613",
        "gpt-4-turbo-preview",
        "gpt-4-vision-preview",
        "gpt-4o-mini-search-preview",
        "gpt-4o-mini-search-preview-2025-03-11",
        "gpt-4o-search-preview",
        "gpt-4o-search-preview-2025-03-11",
        "gpt-5-chat",
        "gpt-5-chat-latest",
        "gpt-5-codex",
        "gpt-5.1-chat-latest",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5.2-chat-latest",
        "gpt-5.2-codex",
        "gpt-5.3-chat-latest",
        "gpt-audio-mini-2025-10-06",
        "gpt-realtime-mini-2025-10-06",
        "o1-mini",
        "o1-mini-2024-09-12",
        "o1-preview",
        "o1-preview-2024-09-12",
        "o3-deep-research",
        "o3-deep-research-2025-06-26",
        "o4-mini-deep-research",
        "o4-mini-deep-research-2025-06-26",
    ],
)
def test_retired_or_legacy_models_not_advertised_but_billed(model):
    assert model not in ACTIVE_MODEL_IDS
    assert calculate_openai_cost(model, {}, 1000, 500) > 0


@pytest.mark.parametrize(
    "model",
    [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-instruct",
        "babbage-002",
        "davinci-002",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4.1-nano",
        "gpt-4o-2024-05-13",
        "o1",
        "o1-pro",
        "o3-mini",
        "o4-mini",
        "gpt-5-2025-08-07",
        "gpt-audio-mini",
        "gpt-audio-mini-2025-12-15",
        "gpt-realtime-mini",
        "gpt-realtime-mini-2025-12-15",
    ],
)
def test_models_retained_before_shutdown(model):
    assert model in ACTIVE_MODEL_IDS


@pytest.mark.parametrize(
    "model", ["babbage-002", "davinci-002", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo-instruct-0914"]
)
def test_legacy_completion_models_not_chat(model):
    assert model in ids(OPENAI_COMPLETION_MODELS)
    assert model not in ids(OPENAI_CHAT_MODELS)


def test_price_objects_independent_across_aliases():
    transcription = [m["cost"] for m in OPENAI_TRANSCRIPTION_MODELS]
    assert len({id(c) for c in transcription}) == len(transcription)
    costs = [m["cost"] for m in OPENAI_BILLING_MODELS if m.get("cost")]
    assert len({id(c) for c in costs}) == len(costs)
    long_context = [c["longContext"] for c in costs if c.get("longContext")]
    assert len({id(c) for c in long_context}) == len(long_context)


# --- validateFunctionCall / formatOpenAiError / misc ----------------------------------------

SAMPLE_FUNCTION = {
    "name": "testFunction",
    "parameters": {
        "type": "object",
        "properties": {"foo": {"type": "string"}, "bar": {"type": "number"}},
        "required": ["foo"],
    },
}


def test_validate_function_call_valid():
    call = {"name": "testFunction", "arguments": json.dumps({"foo": "test", "bar": 123})}
    validate_function_call(call, [SAMPLE_FUNCTION], {})


def test_validate_function_call_unknown_name():
    call = {"name": "nonexistentFunction", "arguments": json.dumps({"foo": "test"})}
    with pytest.raises(
        Exception, match='Called "nonexistentFunction", but there is no function with that name'
    ):
        validate_function_call(call, [SAMPLE_FUNCTION], {})


def test_validate_function_call_bad_arguments():
    call = {"name": "testFunction", "arguments": json.dumps({"bar": "not a number"})}
    with pytest.raises(Exception, match='Call to "testFunction" does not match schema'):
        validate_function_call(call, [SAMPLE_FUNCTION], {})


def test_format_openai_error_with_type_and_code():
    data = {"error": {"message": "Error message", "type": "error_type", "code": "error_code"}}
    assert format_openai_error(data) == (
        "API error: Error message, Type: error_type, Code: error_code\n\n"
        + json.dumps(data, indent=2)
    )


def test_format_openai_error_without_type_and_code():
    result = format_openai_error({"error": {"message": "Error message"}})
    assert result.startswith("API error: Error message\n\n")
    assert "Type:" not in result
    assert "Code:" not in result


def test_is_openai_error_only_response():
    assert is_openai_error_only_response({"error": {"message": "boom"}}) is True
    assert is_openai_error_only_response({"error": {"message": "boom"}, "choices": []}) is True
    assert (
        is_openai_error_only_response({"error": {"message": "boom"}, "choices": [{"x": 1}]})
        is False
    )
    assert is_openai_error_only_response({"error": "boom"}) is False


@pytest.mark.parametrize(
    ("api_url", "expected"),
    [
        ("https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"),
        ("https://api.openai.com/v1/", "https://api.openai.com/v1/chat/completions"),
    ],
)
def test_append_openai_api_path(api_url, expected):
    assert append_openai_api_path(api_url, "chat/completions") == expected
