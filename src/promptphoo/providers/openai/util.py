"""OpenAI model registries, cost, token usage and response helpers.
Ports src/providers/openai/util.ts.

The model and pricing tables themselves live in `tables.py`, generated from upstream.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Literal
from urllib.parse import unquote

from promptphoo.assertions.tools import (
    InvalidToolCallError,
    InvalidToolSchemaError,
    prepare_openai_functions,
    validate_openai_function_call,
)
from promptphoo.jsonutil import js_json_parse
from promptphoo.numeric import is_safe_cost, is_safe_token_count
from promptphoo.providers.openai import tables
from promptphoo.providers.openai.urls import url_hostname, url_pathname
from promptphoo.providers.shared import calculate_cost
from promptphoo.sanitizer import (
    is_non_credential_header,
    looks_like_secret,
    safe_json_stringify,
    sanitize_url,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InvalidToolCallError",
    "InvalidToolSchemaError",
    "prepare_openai_functions",
]

GPT_LONG_CONTEXT_THRESHOLD: int = tables.GPT_LONG_CONTEXT_THRESHOLD

# Billing relationships verified September 11, 2026. Keep the alias on the wire.
OPENAI_DAYBREAK_ALIASES: dict[str, str] = dict(tables.OPENAI_DAYBREAK_ALIASES)

_AZURE_OPENAI_HOSTNAME = re.compile(r"(?:^|\.)(?:openai\.azure\.com|services\.ai\.azure\.com)$")
_OPAQUE_CREDENTIAL_PATH_SEGMENT = re.compile(
    r"(?:^|/)(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32,}"
    r"|(?:token|key|secret|credential|auth)[-_][a-z0-9._-]{8,})(?:/|$)",
    re.I,
)
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)


def _get_record(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _is_record(value: Any) -> bool:
    return isinstance(value, dict)


def _first(*values: Any) -> Any:
    """JS `a ?? b ?? c`."""
    return next((v for v in values if v is not None), None)


def _get_openai_endpoint_hostname(value: str) -> str | None:
    return url_hostname(value if _SCHEME.match(value) else f"https://{value}")


def is_azure_openai_endpoint(value: str | None) -> bool:
    return value is not None and bool(
        _AZURE_OPENAI_HOSTNAME.search(_get_openai_endpoint_hostname(value) or "")
    )


def is_custom_openai_endpoint(value: str) -> bool:
    hostname = _get_openai_endpoint_hostname(value)
    return (
        hostname is not None
        and not re.match(r"^(?:[a-z0-9-]+\.)?api\.openai\.com$", hostname)
        and not _AZURE_OPENAI_HOSTNAME.search(hostname)
    )


def get_openai_chat_choice_error(data: Any) -> dict[str, Any] | None:
    """`{error, partialOutput?}` for a `finish_reason: "error"` choice, else None."""
    response = _get_record(data)
    choices = response.get("choices") if response else None
    choice = _get_record(choices[0]) if isinstance(choices, list) and choices else None
    if not choice or choice.get("finish_reason") != "error":
        return None
    error = _get_record(choice.get("error"))
    if not error:
        return None
    content = (_get_record(choice.get("message")) or {}).get("content")
    message = error.get("message")
    result: dict[str, Any] = {
        "error": {
            **error,
            "message": message
            if isinstance(message, str)
            else "The provider failed during generation.",
        }
    }
    if isinstance(content, (str, list)) and len(content):
        result["partialOutput"] = content
    return result


_ENTITY = r"(?:organization|account|api key|user|safety[- ]identifier)"
_ACCESS = r"(?:access|permissions?)"
_PROVIDER_TARGET = (
    r"(?:(?:these|this|the|our|openai(?:['’]s)?)\s+)?(?:models?|api|service|platform|provider)"
)
_STATE = (
    r"(?:(?:has|have)\s+been|is|are|was|were)\s+(?:(?:temporarily|permanently)\s+)?"
    r"(?:revoked|suspended|disabled|restricted)\b"
)
_POLICY_REVOKED_PATTERNS = [
    re.compile(
        rf"^(?:your|this)\s+{_ENTITY}(?:['’]s)?\s+{_ACCESS}(?:\s+to\s+{_PROVIDER_TARGET})?\s+{_STATE}",
        re.I,
    ),
    re.compile(rf"^(?:your|this)\s+{_ACCESS}\s+to\s+{_PROVIDER_TARGET}\s+{_STATE}", re.I),
    re.compile(
        rf"^{_ACCESS}\s+(?:for|of)\s+(?:(?:your|this|the)\s+)?{_ENTITY}(?:\s+[\"'][\w.-]+[\"'])?\s+{_STATE}",
        re.I,
    ),
    re.compile(rf"^(?:your|this|the)\s+{_ENTITY}\s+{_STATE}", re.I),
    re.compile(
        r"^(?:we|openai|the provider)\s+(?:have|has)\s+(?:(?:temporarily|permanently)\s+)?"
        r"(?:revoked|suspended|disabled|restricted)\s+(?:your|this)\s+"
        rf"(?:{_ENTITY}(?:['’]s)?\s+)?{_ACCESS}(?:\s+to\s+{_PROVIDER_TARGET})?"
        r"(?:$|[!,;]|\s+(?:because|due)\b)",
        re.I,
    ),
]


def _is_openai_policy_access_revoked(message: str) -> bool:
    for part in re.split(r"(?:[!?]|\.(?:\s|$)|\n)+", message):
        sentence = re.sub(r"^error:\s*", "", part.strip(), flags=re.I)
        if any(p.search(sentence) for p in _POLICY_REVOKED_PATTERNS):
            return True
    return False


def _response_root(data: Any) -> dict[str, Any] | None:
    root = _get_record(data)
    return _get_record(root.get("response") if root else None) or root


def get_openai_gateway_error_type(data: Any) -> str | None:
    response = _response_root(data)
    top_level_error = _get_record(response.get("error") if response else None)
    choice_error = None if top_level_error else get_openai_chat_choice_error(response)
    error = top_level_error or (choice_error or {}).get("error")
    metadata = _get_record((error or {}).get("metadata"))
    candidates = [
        (metadata or {}).get("error_type"),
        (error or {}).get("error_type"),
        None if choice_error else (response or {}).get("error_type"),
    ]
    return next((v for v in candidates if isinstance(v, str) and v), None)


def get_openai_gateway_provider_code(data: Any) -> str | None:
    response = _response_root(data)
    error = _get_record(response.get("error") if response else None) or (
        get_openai_chat_choice_error(response) or {}
    ).get("error")
    metadata = _get_record((error or {}).get("metadata"))
    code = _first((metadata or {}).get("provider_code"), (error or {}).get("code"))
    return code if isinstance(code, str) else None


def get_openai_partial_output(output: Any, json_schema: bool) -> Any:
    if json_schema and isinstance(output, str):
        try:
            return js_json_parse(output)
        except ValueError:
            pass  # A refusal can interrupt a JSON response before it is complete.
    return output


def get_openai_policy_refusal(
    data: Any, allow_gateway_marker: bool = False
) -> dict[str, Any] | None:
    """A gateway refusal marker distinguishes prompt blocks from native access-level errors."""
    if not allow_gateway_marker:
        return None
    response = _response_root(data)
    top_level_error = _get_record(response.get("error") if response else None)
    choice_error = None if top_level_error else get_openai_chat_choice_error(response)
    error = top_level_error or (choice_error or {}).get("error")
    if not error:
        return None
    metadata = _get_record(error.get("metadata")) or {}
    provider_code = _first(metadata.get("provider_code"), error.get("code"))
    marker = get_openai_gateway_error_type(data)
    message = error["message"] if isinstance(error.get("message"), str) else ""
    if (marker not in ("refusal", "content_policy_violation")) or (
        marker == "refusal" and _is_openai_policy_access_revoked(message)
    ):
        return None
    result: dict[str, Any] = {
        "message": message if message.strip() else "The model provider declined this request."
    }
    if isinstance(provider_code, str):
        result["code"] = provider_code
    if not choice_error and marker == "refusal" and provider_code in ("bio_policy", "cyber_policy"):
        result["flaggedInput"] = True
    if choice_error and "partialOutput" in choice_error:
        result["partialOutput"] = choice_error["partialOutput"]
    return result


def classify_openai_gateway_stream_error(data: Any) -> str | None:
    marker = get_openai_gateway_error_type(data)
    if get_openai_policy_refusal(data, True):
        return marker
    if marker:
        return "technical"
    response = _response_root(data)
    error = _get_record(response.get("error") if response else None) or {}
    code = error.get("code")
    if code in ("image_content_policy_violation", "content_policy_violation", "content_filter"):
        return "content_policy_violation"
    if code == "refusal":
        return "refusal"
    if code in ("bio_policy", "cyber_policy"):
        return "potential"
    return None


def _has_inline_secret(value: str) -> bool:
    return (
        looks_like_secret(value)
        or re.search(r"(?:^|\s)(?:Bearer|Basic)\s+\S+", value, re.I) is not None
        or re.search(r"eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+", value) is not None
        or re.search(
            r"(?:sk-(?:proj-|ant-)?[a-zA-Z0-9-_]{20,}|key-[a-zA-Z0-9]{20,}|AKIA[A-Z0-9]{16}"
            r"|AIza[a-zA-Z0-9_-]{35})",
            value,
        )
        is not None
    )


def has_sensitive_openai_cache_string(value: str) -> bool:
    urls = re.findall(r"\b(?:https?|s3|gs|az)://[^\s<>\"']+", value, re.I)
    for url in urls:
        if sanitize_url(url) != url:
            return True
        pathname = url_pathname(url)
        if pathname is None:
            return True
        try:
            decoded = unquote(pathname, errors="strict")
        except UnicodeDecodeError:
            return True
        if has_sensitive_openai_cache_path(decoded):
            return True
    return _has_inline_secret(value)


def has_sensitive_openai_cache_path(value: str) -> bool:
    return _has_inline_secret(value) or _OPAQUE_CREDENTIAL_PATH_SEGMENT.search(value) is not None


def has_openai_gateway_credentials(headers: dict[str, str] | None, api_url: str) -> bool:
    has_credential_header = any(
        bool(isinstance(value, str) and value.strip())
        and not is_non_credential_header(name)
        and not re.match(
            r"^(?:x-(?:request|correlation)-id|traceparent|tracestate|baggage)$", name, re.I
        )
        for name, value in (headers or {}).items()
    )
    return has_credential_header or has_sensitive_openai_cache_string(api_url)


DEFAULT_MAX_TOOL_ITERATIONS = 8


def resolve_max_tool_iterations(value: Any) -> int:
    """Resolve a tool-call cap from 1 to 64, falling back to 8 for missing or invalid values."""
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 1 <= value <= 64
    ):
        return math.floor(value)
    return DEFAULT_MAX_TOOL_ITERATIONS


def append_openai_api_path(api_url: str, endpoint: str, query: str | None = None) -> str:
    fragment_index = api_url.find("#")
    fragment = "" if fragment_index == -1 else api_url[fragment_index:]
    without_fragment = api_url if fragment_index == -1 else api_url[:fragment_index]
    query_index = without_fragment.find("?")
    base = without_fragment if query_index == -1 else without_fragment[:query_index]
    existing_query = "" if query_index == -1 else without_fragment[query_index:]
    appended_query = f"{'&' if existing_query else '?'}{query}" if query else ""
    return f"{base.rstrip('/')}/{endpoint.lstrip('/')}{existing_query}{appended_query}{fragment}"


# --- model registries --------------------------------------------------------------------------

RETIRED_OPENAI_MODEL_IDS: frozenset[str] = tables.RETIRED_OPENAI_MODEL_IDS
_LEGACY_OPENAI_DISCOVERY_EXCLUSIONS: frozenset[str] = tables.LEGACY_OPENAI_DISCOVERY_EXCLUSIONS


def _exclude_retired_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        m
        for m in models
        if m["id"] not in RETIRED_OPENAI_MODEL_IDS
        and m["id"] not in _LEGACY_OPENAI_DISCOVERY_EXCLUSIONS
    ]


_OPENAI_TTS_AND_RETIRED_MODELS: list[dict[str, Any]] = tables.OPENAI_TTS_AND_RETIRED_MODELS
_OPENAI_CHAT_AND_RETIRED_MODELS: list[dict[str, Any]] = tables.OPENAI_CHAT_AND_RETIRED_MODELS
_LEGACY_OPENAI_AUDIO_MODELS: list[dict[str, Any]] = tables.LEGACY_OPENAI_AUDIO_MODELS
_OPENAI_RESPONSES_ONLY_AND_RETIRED_MODELS: list[dict[str, Any]] = (
    tables.OPENAI_RESPONSES_ONLY_AND_RETIRED_MODELS
)
_OPENAI_DEEP_RESEARCH_AND_RETIRED_MODELS: list[dict[str, Any]] = (
    tables.OPENAI_DEEP_RESEARCH_AND_RETIRED_MODELS
)
_OPENAI_REALTIME_AND_RETIRED_MODELS: list[dict[str, Any]] = (
    tables.OPENAI_REALTIME_AND_RETIRED_MODELS
)
_RETIRED_OPENAI_REALTIME_MODELS: list[dict[str, Any]] = tables.RETIRED_OPENAI_REALTIME_MODELS

# Models served by /v1/audio/speech, not Chat Completions.
OPENAI_TTS_MODELS = _exclude_retired_models(_OPENAI_TTS_AND_RETIRED_MODELS)
OPENAI_CHAT_MODELS = _exclude_retired_models(
    [*_OPENAI_CHAT_AND_RETIRED_MODELS, *_LEGACY_OPENAI_AUDIO_MODELS]
)
OPENAI_CODEX_ONLY_MODELS: list[dict[str, Any]] = tables.OPENAI_CODEX_ONLY_MODELS
OPENAI_RESPONSES_ONLY_MODELS = _exclude_retired_models(_OPENAI_RESPONSES_ONLY_AND_RETIRED_MODELS)
# Historical billing metadata only; these native models retired July 23, 2026.
OPENAI_DEEP_RESEARCH_MODELS = _OPENAI_DEEP_RESEARCH_AND_RETIRED_MODELS
OPENAI_COMPLETION_MODELS: list[dict[str, Any]] = tables.OPENAI_COMPLETION_MODELS
# Realtime IDs on a different endpoint family from the conversational Realtime API.
NON_CONVERSATIONAL_REALTIME_MODELS: frozenset[str] = tables.NON_CONVERSATIONAL_REALTIME_MODELS
OPENAI_REALTIME_MODELS = _exclude_retired_models(
    [*_OPENAI_REALTIME_AND_RETIRED_MODELS, *_RETIRED_OPENAI_REALTIME_MODELS]
)
OPENAI_BILLING_MODELS: list[dict[str, Any]] = [
    *_OPENAI_CHAT_AND_RETIRED_MODELS,
    *_OPENAI_TTS_AND_RETIRED_MODELS,
    *_LEGACY_OPENAI_AUDIO_MODELS,
    *OPENAI_COMPLETION_MODELS,
    *_OPENAI_REALTIME_AND_RETIRED_MODELS,
    *_RETIRED_OPENAI_REALTIME_MODELS,
    *_OPENAI_RESPONSES_ONLY_AND_RETIRED_MODELS,
    *OPENAI_CODEX_ONLY_MODELS,
    *_OPENAI_DEEP_RESEARCH_AND_RETIRED_MODELS,
]
# Transcription models for the /v1/audio/transcriptions endpoint.
OPENAI_TRANSCRIPTION_MODELS: list[dict[str, Any]] = tables.OPENAI_TRANSCRIPTION_MODELS

_OPENAI_FIRST_PARTY_API_HOSTNAMES: frozenset[str] = tables.OPENAI_FIRST_PARTY_API_HOSTNAMES


def is_openai_first_party_api_url(api_url: str | None = None) -> bool:
    if not api_url:
        return True
    hostname = url_hostname(api_url)
    return hostname is not None and hostname.lower() in _OPENAI_FIRST_PARTY_API_HOSTNAMES


def normalize_openai_billing_model_name(model_name: str) -> str:
    if model_name.startswith("openai/"):
        return model_name[len("openai/") :]
    if model_name.startswith("github/openai/"):
        return model_name[len("github/openai/") :]
    return model_name


def _passthrough_value(config: dict[str, Any] | None, key: str) -> Any:
    passthrough = (config or {}).get("passthrough")
    return passthrough.get(key) if isinstance(passthrough, dict) else None


def get_openai_effective_service_tier(
    provider_config: dict[str, Any], prompt_config: dict[str, Any] | None = None
) -> Any:
    """The service tier a request uses, honouring prompt and passthrough overrides.

    Returns `UNDEFINED`-free values: None means unset; an explicit `null` tier is not
    distinguishable from unset in YAML-loaded dicts either.
    """
    prompt_passthrough_tier = _passthrough_value(prompt_config, "service_tier")
    provider_passthrough_tier = _passthrough_value(provider_config, "service_tier")
    if prompt_passthrough_tier is not None:
        return prompt_passthrough_tier
    if (prompt_config or {}).get("service_tier") is not None:
        return prompt_config["service_tier"]  # type: ignore[index]
    if prompt_config and "passthrough" in prompt_config:
        return provider_config.get("service_tier")
    return (
        provider_config.get("service_tier")
        if provider_passthrough_tier is None
        else provider_passthrough_tier
    )


def normalize_openai_service_tier_for_wire(service_tier: Any, api_url: str | None = None) -> Any:
    return (
        "priority"
        if service_tier == "fast" and is_openai_first_party_api_url(api_url)
        else service_tier
    )


def assert_openai_model_endpoint_compatibility(
    model: Any, allow_transcription: bool = False
) -> None:
    if not isinstance(model, str):
        return
    normalized = model.split("/")[-1]
    if normalized == "gpt-live-transcribe" or normalized.startswith("gpt-live-transcribe-"):
        raise ValueError(
            'OpenAI model "gpt-live-transcribe" requires Realtime transcription sessions, which '
            "are not yet supported by promptfoo."
        )
    if normalized.startswith("gpt-live-"):
        raise ValueError(f"Use openai:live:{model} for GPT-Live sessions.")
    if normalized == "gpt-transcribe" and not allow_transcription:
        raise ValueError(
            'OpenAI model "gpt-transcribe" is transcription-only. Use '
            "openai:transcription:gpt-transcribe (or bare openai:gpt-transcribe)."
        )


def assert_openai_api_model(
    model: Any, api_url: str | None = None, allow_transcription: bool = False
) -> None:
    if not isinstance(model, str):
        return
    if not is_openai_first_party_api_url(api_url):
        return
    assert_openai_model_endpoint_compatibility(model, allow_transcription)
    normalized = model.split("/")[-1]
    if normalized in RETIRED_OPENAI_MODEL_IDS:
        raise ValueError(
            f"OpenAI model {model} has been retired and is no longer available from OpenAI's "
            "first-party API. Use a current model or configure a custom OpenAI-compatible "
            "apiBaseUrl."
        )
    if any(candidate["id"] == normalized for candidate in OPENAI_CODEX_ONLY_MODELS):
        raise ValueError(
            f"OpenAI model {model} is only available through openai:codex-sdk with eligible "
            "Codex authentication."
        )


RetiredOpenAiModelRoute = Literal["chat", "moderation", "responses", "tts", "realtime"]


def _has_id(models: list[dict[str, Any]], model_id: str) -> bool:
    return any(m["id"] == model_id for m in models)


def get_retired_openai_model_route(model_name: str) -> RetiredOpenAiModelRoute | None:
    """The endpoint family historically used by a retired model ID, including gateway
    prefixes. Only used after assert_openai_api_model allowed a custom endpoint."""
    model_id = model_name.split("/")[-1]
    if model_id not in RETIRED_OPENAI_MODEL_IDS:
        return None
    if _has_id(_OPENAI_CHAT_AND_RETIRED_MODELS, model_id) or _has_id(
        _LEGACY_OPENAI_AUDIO_MODELS, model_id
    ):
        return "chat"
    if _has_id(_OPENAI_TTS_AND_RETIRED_MODELS, model_id):
        return "tts"
    if _has_id(_OPENAI_REALTIME_AND_RETIRED_MODELS, model_id) or _has_id(
        _RETIRED_OPENAI_REALTIME_MODELS, model_id
    ):
        return "realtime"
    if model_id.startswith("text-moderation-"):
        return "moderation"
    if _has_id(_OPENAI_RESPONSES_ONLY_AND_RETIRED_MODELS, model_id) or _has_id(
        _OPENAI_DEEP_RESEARCH_AND_RETIRED_MODELS, model_id
    ):
        return "responses"
    return None


# --- cost --------------------------------------------------------------------------------------


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def calculate_openai_cost(
    model_name: str,
    config: dict[str, Any],
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    audio_prompt_tokens: Any = None,
    audio_completion_tokens: Any = None,
) -> float | None:
    if audio_prompt_tokens is None and audio_completion_tokens is None:
        return calculate_cost(
            model_name, config, prompt_tokens, completion_tokens, OPENAI_BILLING_MODELS
        )
    if not all(
        _is_finite(v)
        for v in (prompt_tokens, completion_tokens, audio_prompt_tokens, audio_completion_tokens)
    ):
        return None
    model = next((m for m in OPENAI_BILLING_MODELS if m["id"] == model_name), None)
    if not model or not model.get("cost"):
        return None
    text_cost = calculate_cost(
        model_name, config, prompt_tokens, completion_tokens, OPENAI_BILLING_MODELS
    )
    if text_cost is None:
        return None
    total = text_cost
    cost = model["cost"]
    if "audioInput" in cost or "audioOutput" in cost:
        model_audio_input = cost["audioInput"] if _is_finite(cost.get("audioInput")) else 0
        model_audio_output = cost["audioOutput"] if _is_finite(cost.get("audioOutput")) else 0
        audio_input_cost = _first(
            config.get("audioInputCost"), config.get("audioCost"), model_audio_input
        )
        audio_output_cost = _first(
            config.get("audioOutputCost"), config.get("audioCost"), model_audio_output
        )
        total += (
            audio_input_cost * audio_prompt_tokens + audio_output_cost * audio_completion_tokens
        )
    return total


def calculate_safe_openai_cost(model_name: str, config: dict[str, Any], data: Any) -> float | None:
    """Calculate cost without trusting provider-controlled usage or cost fields."""
    usage = (data or {}).get("usage") if isinstance(data, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    safe_tokens = is_safe_token_count(prompt_tokens) and is_safe_token_count(completion_tokens)
    local_cost = (
        calculate_openai_cost(model_name, config, prompt_tokens, completion_tokens)
        if safe_tokens
        else None
    )
    has_local_override = any(config.get(k) is not None for k in ("cost", "inputCost", "outputCost"))
    if has_local_override:
        if is_safe_cost(local_cost):
            return local_cost
        input_cost = _first(config.get("inputCost"), config.get("cost"))
        output_cost = _first(config.get("outputCost"), config.get("cost"))
        if safe_tokens and is_safe_cost(input_cost) and is_safe_cost(output_cost):
            configured = input_cost * prompt_tokens + output_cost * completion_tokens
            return configured if is_safe_cost(configured) else None
        return None
    if is_safe_cost(usage.get("cost")):
        return usage["cost"]
    return local_cost if is_safe_cost(local_cost) else None


def js_error_string(err: Any) -> str:
    """`String(err)` as Node prints a thrown value: `Error: message` for errors."""
    if not isinstance(err, BaseException):
        from promptphoo.jsvalues import js_string

        return js_string(err)
    name = "TypeError" if isinstance(err, TypeError) else "Error"
    return f"{name}: {err}"


def fail_api_call(err: Any) -> dict[str, str]:
    """SDK APIError-shaped errors (`status` plus `error`/`type`) get the detailed form."""
    status = getattr(err, "status", None)
    if status is not None and (hasattr(err, "error") or hasattr(err, "type")):
        body = getattr(err, "error", None)
        body = body if isinstance(body, dict) else {}
        error_type = body.get("type") or getattr(err, "type", None) or "unknown"
        message = body.get("message") or str(err) or "Unknown error"
        return {"error": f"API error: {error_type} {status} {message}"}
    return {"error": f"API error: {js_error_string(err)}"}


def get_openai_cache_write_input_tokens(usage: Any) -> int | float | None:
    usage = usage if isinstance(usage, dict) else {}
    candidates = [
        (usage.get("prompt_tokens_details") or {}).get("cache_write_tokens")
        if isinstance(usage.get("prompt_tokens_details"), dict)
        else None,
        (usage.get("input_tokens_details") or {}).get("cache_write_tokens")
        if isinstance(usage.get("input_tokens_details"), dict)
        else None,
        (usage.get("input_token_details") or {}).get("cache_write_tokens")
        if isinstance(usage.get("input_token_details"), dict)
        else None,
        usage.get("cache_write_input_tokens"),
    ]
    for value in candidates:
        if _is_finite(value) and value >= 0:
            return value
    return None


def get_openai_completion_token_details(usage: dict[str, Any]) -> dict[str, Any] | None:
    # Some OpenAI-compatible APIs (e.g. Moonshot) report cached prompt tokens at the top level.
    cached_input = _first(
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        (usage.get("input_tokens_details") or {}).get("cached_tokens"),
        usage.get("cached_tokens"),
        0,
    )
    cache_write = get_openai_cache_write_input_tokens(usage)
    completion_details = _first(
        usage.get("completion_tokens_details"), usage.get("output_tokens_details")
    )
    if not completion_details and cached_input <= 0 and cache_write is None:
        return None
    details: dict[str, Any] = {}
    if completion_details:
        for field, key in (
            ("reasoning", "reasoning_tokens"),
            ("acceptedPrediction", "accepted_prediction_tokens"),
            ("rejectedPrediction", "rejected_prediction_tokens"),
        ):
            # `{reasoning: undefined}` serializes away, so absent keys are omitted.
            if completion_details.get(key) is not None:
                details[field] = completion_details[key]
    if cached_input > 0:
        details["cacheReadInputTokens"] = cached_input
    if cache_write is not None:
        details["cacheCreationInputTokens"] = cache_write
    return details


def get_token_usage(data: Any, cached: bool) -> dict[str, Any]:
    """TokenUsage from an OpenAI-style `usage` block. Cached hits report only `cached`/`total`."""
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return {}
    total = usage["total_tokens"] if is_safe_token_count(usage.get("total_tokens")) else None
    if cached:
        return {} if total is None else {"cached": total, "total": total}

    token_usage: dict[str, Any] = {"numRequests": 1}
    if total is not None:
        token_usage["total"] = total
    token_usage["prompt"] = (
        usage["prompt_tokens"] if is_safe_token_count(usage.get("prompt_tokens")) else 0
    )
    token_usage["completion"] = (
        usage["completion_tokens"] if is_safe_token_count(usage.get("completion_tokens")) else 0
    )
    prompt_details = (
        usage["prompt_tokens_details"]
        if _is_record(usage.get("prompt_tokens_details"))
        else usage["input_tokens_details"]
        if _is_record(usage.get("input_tokens_details"))
        else None
    )
    completion_details = (
        usage["completion_tokens_details"]
        if _is_record(usage.get("completion_tokens_details"))
        else usage["output_tokens_details"]
        if _is_record(usage.get("output_tokens_details"))
        else None
    )
    cd = completion_details or {}
    mappings = [
        ("reasoning", cd.get("reasoning_tokens")),
        ("acceptedPrediction", cd.get("accepted_prediction_tokens")),
        ("rejectedPrediction", cd.get("rejected_prediction_tokens")),
        # Nested details win over Moonshot-style top-level cached_tokens.
        (
            "cacheReadInputTokens",
            _first((prompt_details or {}).get("cached_tokens"), usage.get("cached_tokens")),
        ),
        ("cacheCreationInputTokens", get_openai_cache_write_input_tokens(usage)),
    ]
    details = {field: value for field, value in mappings if is_safe_token_count(value)}
    if details:
        token_usage["completionDetails"] = details
    return token_usage


def get_token_usage_with_request_count(data: Any, cached: bool) -> dict[str, Any]:
    return {**get_token_usage(data, cached), "numRequests": 0 if cached else 1}


# --- chat completion message validation --------------------------------------------------------


def _has_named_payload(value: Any) -> bool:
    return _is_record(value) and isinstance(value.get("name"), str) and bool(value["name"].strip())


def _has_function_payload(value: Any) -> bool:
    return _has_named_payload(value) and isinstance(value.get("arguments"), str)


def _is_tool_call(value: Any) -> bool:
    if not _is_record(value) or not isinstance(value.get("id"), str):
        return False
    # OpenRouter sometimes omits the discriminator on a complete function payload.
    return (
        value.get("type") in ("function", None) and _has_function_payload(value.get("function"))
    ) or (
        value.get("type") == "custom"
        and _has_named_payload(value.get("custom"))
        and isinstance(value["custom"].get("input"), str)
    )


def _has_url_payload(value: Any) -> bool:
    return _is_record(value) and isinstance(value.get("url"), str) and bool(value["url"].strip())


def _is_structured_content_part(value: Any) -> bool:
    if not _is_record(value):
        return False
    if value.get("type") == "text":
        return isinstance(value.get("text"), str) and bool(value["text"].strip())
    if value.get("type") == "image_url":
        return _has_url_payload(value.get("image_url"))
    return False


def validate_chat_completion_message(
    value: Any,
    allow_audio: bool = False,
    allow_structured_content: bool = False,
    finish_reason: str | None = None,
) -> dict[str, Any] | None:
    """Validate the response fields providers inspect before returning a completion.

    Returns a dict with audio, content, functionCall, reasoning, refusal, structuredContent and
    toolCalls (None where absent), plus `hasContent`, or None if the message is unusable.
    """
    if not _is_record(value):
        return None
    function_call = (
        value["function_call"] if _has_function_payload(value.get("function_call")) else None
    )
    if value.get("function_call") is not None and not function_call:
        return None
    tool_calls_raw = value.get("tool_calls")
    if tool_calls_raw is not None and (
        not isinstance(tool_calls_raw, list) or not all(_is_tool_call(t) for t in tool_calls_raw)
    ):
        return None
    tool_calls = tool_calls_raw if isinstance(tool_calls_raw, list) and tool_calls_raw else None

    has_content = "content" in value
    raw_content = value.get("content")
    content_ok = raw_content is None or isinstance(raw_content, str)
    content = raw_content if content_ok else None
    structured_content = (
        raw_content
        if allow_structured_content
        and isinstance(raw_content, list)
        and all(_is_structured_content_part(p) for p in raw_content)
        else None
    )
    if has_content and not content_ok and structured_content is None:
        return None

    audio_raw = value.get("audio")
    audio = (
        audio_raw
        if allow_audio
        and _is_record(audio_raw)
        and (isinstance(audio_raw.get("transcript"), str) or isinstance(audio_raw.get("data"), str))
        else None
    )
    reasoning_raw = value.get("reasoning")
    reasoning = reasoning_raw if isinstance(reasoning_raw, str) and reasoning_raw.strip() else None
    has_usable_content = (isinstance(content, str) and bool(content.strip())) or bool(
        structured_content
    )
    refusal_raw = value.get("refusal")
    refusal = refusal_raw if isinstance(refusal_raw, str) and refusal_raw.strip() else None
    if (
        not function_call
        and not tool_calls
        and not has_usable_content
        and audio is None
        and reasoning is None
        and not refusal
        and finish_reason != "content_filter"
    ):
        return None
    return {
        "audio": audio,
        "content": content,
        "functionCall": function_call,
        "reasoning": reasoning,
        "refusal": refusal,
        "structuredContent": structured_content,
        "toolCalls": tool_calls,
    }


def get_chat_completion_refusal(
    message: dict[str, Any], finish_reason: str | None
) -> dict[str, Any] | None:
    if message.get("refusal"):
        return {"output": message["refusal"], "isRefusal": True, "guardrails": {"flagged": True}}
    if finish_reason == "content_filter":
        content = message.get("content")
        return {
            "output": content
            if isinstance(content, str) and content.strip()
            else "Content filtered by provider",
            "isRefusal": True,
            "guardrails": {"flagged": True},
        }
    return None


def parse_chat_completion_json_output(message: dict[str, Any], output: Any, log_label: str) -> Any:
    """Parse the raw completion content as JSON, falling back to `output`."""
    content = message.get("content")
    candidate = content if isinstance(content, str) else output if isinstance(output, str) else None
    if not candidate:
        return output
    try:
        return js_json_parse(candidate)
    except ValueError as error:
        logger.warning("%s: SyntaxError: %s", log_label, error)
        return output


def validate_function_call(
    output: Any, functions: Any = None, vars: dict[str, Any] | None = None
) -> None:
    """Raise InvalidToolCallError (bad call) or InvalidToolSchemaError (bad definitions)."""
    validate_openai_function_call(output, functions, vars)


def is_openai_error_only_response(data: Any) -> bool:
    """A completed provider error without a competing nonempty choice."""
    if not isinstance(data, dict):
        return False
    error = data.get("error")
    if not isinstance(error, dict) or not isinstance(error.get("message"), str):
        return False
    choices = data.get("choices")
    return "choices" not in data or choices is None or choices == []


def format_openai_error(data: dict[str, Any]) -> str:
    error = data["error"]
    message = f"API error: {error.get('message')}"
    if error.get("type"):
        message += f", Type: {error['type']}"
    if error.get("code"):
        message += f", Code: {error['code']}"
    message += "\n\n" + (safe_json_stringify(data, True) or "")
    return message


def flatten_response_tool(tool: Any) -> Any:
    record = _get_record(tool)
    definition = _get_record((record or {}).get("function"))
    if (record or {}).get("type") != "function" or not definition:
        return tool
    rest = {k: v for k, v in record.items() if k != "function"}  # type: ignore[union-attr]
    return {**rest, **definition}
