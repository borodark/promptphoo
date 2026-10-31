"""Meta Model API (Muse Spark) providers: chat, Responses and Anthropic-compatible Messages.
Ports src/providers/meta.ts and the `meta:` entry of src/providers/registry.ts.

The Responses route subclasses the OpenAI Responses provider, which is imported lazily; until
`promptphoo.providers.openai.responses` exists, `meta:responses:*` and bare `meta:<model>` fail
with a clear error. The Messages route reuses promptphoo's Anthropic Messages provider, which
sends the request non-streaming (see that module's docstring).
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any

from promptphoo import envars
from promptphoo.providers.anthropic.generic import ANTHROPIC_VERSION
from promptphoo.providers.anthropic.messages import AnthropicMessagesProvider
from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.compatible.common import serialize_provider
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider

META_API_BASE_URL = "https://api.meta.ai/v1"
# The Anthropic-compatible Messages surface is served from the bare host.
META_MESSAGES_API_BASE_URL = "https://api.meta.ai"
META_API_KEY_ENVAR = "MODEL_API_KEY"
DEFAULT_META_MODEL = "muse-spark-1.1"
DEFAULT_META_MAX_OUTPUT_TOKENS = 131_072
UNSUPPORTED_META_SUBTYPES = frozenset([
    "assistant", "audio", "completion", "embedding", "embeddings", "image", "moderation",
    "realtime", "transcription", "video",
])  # fmt: skip
# Muse Spark rejects other values (including OpenAI's 'none') with an HTTP 400.
META_REASONING_EFFORTS = ["minimal", "low", "medium", "high", "xhigh"]

# USD per token. https://dev.meta.ai/docs/pricing-rate-limits
META_MODEL_PRICES: dict[str, dict[str, float]] = {
    "muse-spark-1.3": {"input": 1.25 / 1e6, "cachedInput": 0.15 / 1e6, "output": 4.25 / 1e6},
    "muse-spark-1.3-contributor": {
        "input": 0.1 / 1e6,
        "cachedInput": 0.002 / 1e6,
        "output": 0.2 / 1e6,
    },
    "muse-spark-1.1": {"input": 1.25 / 1e6, "cachedInput": 0.15 / 1e6, "output": 4.25 / 1e6},
}


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def _is_finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def resolve_meta_api_key(config: Mapping[str, Any], env: Any) -> str | None:
    """Meta-specific sources only: never OPENAI_API_KEY or ANTHROPIC_API_KEY."""
    api_key_envar = config.get("apiKeyEnvar") or META_API_KEY_ENVAR
    return (
        config.get("apiKey")
        or envars.get_provider_env_string(env, api_key_envar)
        or envars.get_env_string(api_key_envar)
    )


def missing_meta_api_key_message(config: Mapping[str, Any]) -> str:
    return (
        f"Meta Model API key is not set. Set the {config.get('apiKeyEnvar') or META_API_KEY_ENVAR} "
        "environment variable or add `apiKey` to the provider config."
    )


def resolve_meta_openai_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    return {
        **config,
        "apiKeyEnvar": _first(config.get("apiKeyEnvar"), META_API_KEY_ENVAR),
        "apiBaseUrl": config.get("apiBaseUrl") or META_API_BASE_URL,
    }


def resolve_meta_openai_url(config: Mapping[str, Any]) -> str:
    if config.get("apiHost"):
        return f"https://{config['apiHost'].rstrip('/')}/v1"
    return (config.get("apiBaseUrl") or META_API_BASE_URL).rstrip("/")


def meta_to_json(provider: str, model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    class _View:
        pass

    view = _View()
    view.model_name = model_name  # type: ignore[attr-defined]
    view.config = config  # type: ignore[attr-defined]
    return serialize_provider(view, provider)


def assert_supported_reasoning_effort(effort: Any) -> None:
    if effort is None:
        return
    text = effort if isinstance(effort, str) else str(effort)
    if text not in META_REASONING_EFFORTS:
        raise ValueError(
            f"Muse Spark models do not support reasoning_effort '{text}'. Omit it to let "
            "the model pick its own reasoning depth, or use one of: "
            f"{', '.join(META_REASONING_EFFORTS)}."
        )


def apply_meta_chat_output_cap(
    body: dict[str, Any], passthrough: Mapping[str, Any], value: Any
) -> None:
    if "max_completion_tokens" in passthrough:
        return
    output_cap = _first(passthrough.get("max_tokens"), value)
    if output_cap is None:
        body.pop("max_completion_tokens", None)
    else:
        body["max_completion_tokens"] = output_cap


def apply_meta_responses_output_cap(
    body: dict[str, Any], passthrough: Mapping[str, Any], value: Any
) -> None:
    body.pop("max_completion_tokens", None)
    if "max_output_tokens" in passthrough:
        return
    output_cap = _first(
        passthrough.get("max_completion_tokens"), passthrough.get("max_tokens"), value
    )
    if output_cap is None:
        body.pop("max_output_tokens", None)
    else:
        body["max_output_tokens"] = output_cap


def assert_supported_meta_request(body: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if config.get("stop") is not None or "stop" in body:
        raise ValueError(
            "Muse Spark models do not support `stop` sequences; remove `stop` from the provider "
            "config."
        )
    if "logprobs" in body:
        raise ValueError("Muse Spark models do not support logprobs.")
    include = body.get("include")
    if isinstance(include, list) and any(
        item in ("logprobs", "message.output_text.logprobs") for item in include
    ):
        raise ValueError("Muse Spark models do not support logprobs.")
    if "logit_bias" in body or config.get("logit_bias") is not None:
        raise ValueError(
            "Muse Spark models do not support `logit_bias`; remove it from the provider config."
        )
    n = body.get("n")
    if _is_finite(n) and (n or 0) > 1:
        raise ValueError(
            "Muse Spark models do not support `n > 1`; remove `n` from the provider config."
        )
    if body.get("stream") or config.get("stream"):
        raise ValueError(
            "Meta chat and Responses streaming is not supported by promptfoo; remove `stream` "
            "from the provider config."
        )


def apply_meta_sampling_hygiene(
    body: dict[str, Any],
    config: Mapping[str, Any],
    passthrough: Mapping[str, Any],
    sampling_keys: list[str],
) -> None:
    """Keep OPENAI_* sampling env vars out of Meta requests."""
    if config.get("temperature") is None and "temperature" not in passthrough:
        if config.get("omitDefaults"):
            body.pop("temperature", None)
        else:
            body["temperature"] = 0
    for key in sampling_keys:
        if key in passthrough:
            continue
        if config.get(key) is None:
            body.pop(key, None)
        else:
            body[key] = config[key]


def calculate_meta_cost(
    model_name: str,
    config: Mapping[str, Any],
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    cached_tokens: Any = None,
) -> float | None:
    if not _is_finite(prompt_tokens) or not _is_finite(completion_tokens):
        return None
    prices = META_MODEL_PRICES.get(model_name)
    input_cost = _first(config.get("inputCost"), config.get("cost"), (prices or {}).get("input"))
    output_cost = _first(config.get("outputCost"), config.get("cost"), (prices or {}).get("output"))
    if input_cost is None or output_cost is None:
        return None
    billable_cached = min(max(cached_tokens, 0), prompt_tokens) if _is_finite(cached_tokens) else 0
    uncached = prompt_tokens - billable_cached
    cache_read_cost = _first(
        config.get("cacheReadCost"),
        config.get("inputCost"),
        config.get("cost"),
        (prices or {}).get("cachedInput"),
        input_cost,
    )
    return (
        input_cost * uncached + cache_read_cost * billable_cached + output_cost * completion_tokens
    )


def apply_meta_cost(
    response: ProviderResponse,
    model_name: str,
    provider_config: Mapping[str, Any],
    context: dict[str, Any] | None = None,
) -> ProviderResponse:
    """Fill in cost for responses the OpenAI/Anthropic tables can't price."""
    if (
        not response
        or response.get("cached")
        or response.get("cost") is not None
        or not response.get("tokenUsage")
    ):
        return response
    prompt_config = ((context or {}).get("prompt") or {}).get("config") or {}
    config = {**provider_config, **prompt_config}
    usage = response["tokenUsage"]
    cost = calculate_meta_cost(
        model_name,
        config,
        usage.get("prompt"),
        usage.get("completion"),
        _first((usage.get("completionDetails") or {}).get("cacheReadInputTokens"), 0),
    )
    if cost is not None:
        response["cost"] = cost
    return response


class _MetaOpenAiMixin:
    """Overrides shared by the chat and Responses providers."""

    config: dict[str, Any]
    env: Any
    model_name: str

    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        return resolve_meta_api_key(self.config, self.env)

    def get_api_url(self, config: dict[str, Any] | None = None) -> str:
        return resolve_meta_openai_url(self.config)

    # Meta has no OpenAI organization; a stray OPENAI_ORGANIZATION must not leak onto requests.
    def get_organization(self, config: dict[str, Any] | None = None) -> str | None:
        return None

    def get_missing_api_key_error_message(self, config: dict[str, Any] | None = None) -> str:
        return missing_meta_api_key_message(self.config)

    # Muse Spark models are reasoning models that still accept temperature.
    def is_reasoning_model(self, model_name: str | None = None) -> bool:
        return True

    def is_reasoning_capability_model(self, model_name: str) -> bool:
        return True

    def supports_temperature(self, model_name: str | None = None) -> bool:
        return True

    def supports_temperature_for_capability_model(self, model_name: str) -> bool:
        return True

    def get_gen_ai_system(self) -> str:
        return "meta"


class MetaProvider(_MetaOpenAiMixin, OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        resolved = resolve_meta_openai_config(options.get("config"))
        super().__init__(model_name, {**options, "config": resolved})
        self.config = resolved

    def default_id(self) -> str:
        return f"meta:chat:{self.model_name}"

    def __str__(self) -> str:
        return f"[Meta Model API Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return meta_to_json("meta:chat", self.model_name, self.config)

    async def get_openai_body(self, prompt, context=None, call_api_options=None) -> dict[str, Any]:
        result = await super().get_openai_body(prompt, context, call_api_options)
        body, config = result["body"], result["config"]
        passthrough = config.get("passthrough") or {}
        assert_supported_reasoning_effort(body.get("reasoning_effort"))
        assert_supported_meta_request(body, config)
        apply_meta_chat_output_cap(
            body,
            passthrough,
            _first(config.get("max_completion_tokens"), config.get("max_tokens")),
        )
        apply_meta_sampling_hygiene(
            body, config, passthrough, ["top_p", "presence_penalty", "frequency_penalty"]
        )
        return result

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        response = await super().call_api(prompt, context, options)
        if not response or (
            response.get("error")
            and (
                (response.get("metadata") or {}).get("errorOrigin") != "tool"
                or not response.get("tokenUsage")
            )
        ):
            return response
        prompt_config = ((context or {}).get("prompt") or {}).get("config") or {}
        model_name = self.get_billing_model_name({**self.config, **prompt_config})
        return apply_meta_cost(response, model_name, self.config, context)


def _load_openai_responses_provider() -> type:
    # Imported lazily: the Responses provider pulls in the OpenAI chat stack.
    return importlib.import_module("promptphoo.providers.openai.responses").OpenAiResponsesProvider


def create_meta_responses_provider(model_name: str, options: dict[str, Any]) -> Any:
    base = _load_openai_responses_provider()

    class MetaResponsesProvider(_MetaOpenAiMixin, base):  # type: ignore[misc, valid-type]
        def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
            options = options or {}
            resolved = resolve_meta_openai_config(options.get("config"))
            init: Any = super().__init__
            init(model_name, {**options, "config": resolved})
            self.config = resolved

        def default_id(self) -> str:
            return f"meta:responses:{self.model_name}"

        def __str__(self) -> str:
            return f"[Meta Model API Responses Provider {self.model_name}]"

        def to_json(self) -> dict[str, Any]:
            return meta_to_json("meta:responses", self.model_name, self.config)

        async def get_openai_body(self, prompt, context=None, call_api_options=None):
            result = await super().get_openai_body(prompt, context, call_api_options)
            body, config = result["body"], result["config"]
            passthrough = config.get("passthrough") or {}
            reasoning = body.get("reasoning")
            assert_supported_reasoning_effort(
                reasoning.get("effort") if isinstance(reasoning, dict) else None
            )
            assert_supported_meta_request(body, config)
            apply_meta_responses_output_cap(
                body,
                passthrough,
                _first(
                    config.get("max_output_tokens"),
                    config.get("max_completion_tokens"),
                    config.get("max_tokens"),
                ),
            )
            apply_meta_sampling_hygiene(body, config, passthrough, ["top_p"])
            return result

        def apply_billing(self, result, data, config, cached):
            parent = getattr(super(), "apply_billing", None)
            billed = parent(result, data, config, cached) if parent else result
            if cached or billed.get("cost") is not None:
                return billed
            usage = data.get("usage") if isinstance(data, dict) else None
            usage = usage or {}
            cost = calculate_meta_cost(
                self.get_billing_model_name(config),
                config,
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                (usage.get("input_tokens_details") or {}).get("cached_tokens"),
            )
            return billed if cost is None else {**billed, "cost": cost}

    return MetaResponsesProvider(model_name, options)


class MetaMessagesProvider(AnthropicMessagesProvider):
    """Meta's Anthropic-compatible Messages endpoint, authenticated with a bearer token."""

    # Never send a local Claude Code OAuth credential to api.meta.ai.
    SUPPORTS_CLAUDE_CODE_OAUTH = False
    # Muse model ids are not Anthropic models.
    WARNS_ON_UNKNOWN_MODEL = False

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        meta_config = options.get("config") or {}
        resolved = {
            **meta_config,
            "apiBaseUrl": meta_config.get("apiBaseUrl") or META_MESSAGES_API_BASE_URL,
            "max_tokens": _first(meta_config.get("max_tokens"), DEFAULT_META_MAX_OUTPUT_TOKENS),
            "temperature": _first(meta_config.get("temperature"), 0),
            "stream": _first(meta_config.get("stream"), True),
            # Muse reasoning arrives as encrypted redacted_thinking blocks; hide it by default.
            "showThinking": _first(meta_config.get("showThinking"), False),
        }
        super().__init__(model_name, {**options, "config": resolved})

    def build_request_headers(self, request_headers: Mapping[str, str]) -> dict[str, str]:
        """Bearer auth instead of x-api-key; Anthropic-scoped ANTHROPIC_CUSTOM_HEADERS are not
        sent to Meta."""
        headers: dict[str, str] = {
            "accept": "application/json",
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for name, value in request_headers.items():
            for existing in [k for k in headers if k.lower() == name.lower()]:
                del headers[existing]
            headers[name] = value
        return headers

    def get_api_key(self) -> str | None:
        return resolve_meta_api_key(self.config, self.env)

    # ANTHROPIC_BASE_URL overrides are Anthropic-scoped.
    def get_api_base_url(self) -> str | None:
        return self.config.get("apiBaseUrl") or META_MESSAGES_API_BASE_URL

    def has_custom_headers(self) -> bool:
        return False

    def get_gen_ai_system(self) -> str:
        return "meta"

    def default_id(self) -> str:
        return f"meta:messages:{self.model_name}"

    def __str__(self) -> str:
        return f"[Meta Model API Messages Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return meta_to_json("meta:messages", self.model_name, self.config)

    def validate_authentication(self) -> None:
        if not self.api_key:
            raise ValueError(missing_meta_api_key_message(self.config))

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        if not self.api_key:
            raise ValueError(missing_meta_api_key_message(self.config))
        response = await super().call_api(prompt, context)
        return apply_meta_cost(response, self.model_name, self.config, context)


def create_meta_provider(provider_path: str, options: dict[str, Any] | None = None) -> Any:
    options = options or {}
    rest = provider_path.split(":")[1:]
    first = rest[0] if rest else None
    if first in UNSUPPORTED_META_SUBTYPES:
        raise ValueError(
            f'The Meta Model API does not expose an {first} endpoint; "{provider_path}" cannot '
            "be resolved. Use meta:<model>, meta:chat:<model>, meta:responses:<model>, or "
            "meta:messages:<model> instead."
        )
    sub_type = "responses"
    if first in ("chat", "responses", "messages"):
        sub_type = rest.pop(0)
    elif len(rest) > 1:
        raise ValueError(
            f"Unknown Meta Model API sub-type '{first}' in \"{provider_path}\". Use "
            "meta:<model>, meta:chat:<model>, meta:responses:<model>, or meta:messages:<model>."
        )
    model_name = ":".join(rest) or DEFAULT_META_MODEL
    if sub_type == "chat":
        return MetaProvider(model_name, options)
    if sub_type == "messages":
        return MetaMessagesProvider(model_name, options)
    return create_meta_responses_provider(model_name, options)


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `meta:` registry entry (`env: providerOptions.env ?? context.env`, the same here)."""
    return create_meta_provider(provider_path, options)
