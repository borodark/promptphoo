"""MiniMax chat provider. Ports src/providers/minimax.ts and the `minimax:` factory
(`configuredProviderFactory`) in src/providers/registry.ts.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from promptphoo.providers.compatible.common import (
    configured_provider_options,
    serialize_provider,
)
from promptphoo.providers.credentials import resolve_configured_api_key
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.util import get_openai_completion_token_details
from promptphoo.providers.shared import clamp_cached_tokens

logger = logging.getLogger(__name__)

MINIMAX_API_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_API_KEY_ENV_VAR = "MINIMAX_API_KEY"

MINIMAX_CHAT_MODELS: list[dict[str, Any]] = [
    {
        "id": "MiniMax-M3",
        "cost": {
            "input": 0.3 / 1e6,
            "output": 1.2 / 1e6,
            "cache_read": 0.06 / 1e6,
            "longContext": {
                "threshold": 512_000,
                "input": 0.6 / 1e6,
                "output": 2.4 / 1e6,
                "cache_read": 0.12 / 1e6,
            },
        },
    },
    {"id": "MiniMax-M2.7",
     "cost": {"input": 0.3 / 1e6, "output": 1.2 / 1e6, "cache_read": 0.06 / 1e6}},
    {
        "id": "MiniMax-M2.7-highspeed",
        "cost": {"input": 0.6 / 1e6, "output": 2.4 / 1e6, "cache_read": 0.06 / 1e6},
    },
]  # fmt: skip


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def calculate_minimax_cost(
    model_name: str,
    config: dict[str, Any],
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    cached_tokens: Any = None,
) -> float | None:
    if not _is_finite(prompt_tokens) or not _is_finite(completion_tokens):
        return None
    model = next((m for m in MINIMAX_CHAT_MODELS if m["id"] == model_name), None)
    billable_cached = clamp_cached_tokens(cached_tokens, prompt_tokens)
    uncached_prompt_tokens = prompt_tokens - billable_cached
    model_cost = model["cost"] if model else None
    long_context = model_cost.get("longContext") if model_cost else None
    if long_context and prompt_tokens > long_context["threshold"]:
        model_cost = long_context
    tier_multiplier = (
        1.5 if model_name == "MiniMax-M3" and config.get("service_tier") == "priority" else 1
    )
    input_cost = _first(
        config.get("inputCost"),
        config.get("cost"),
        model_cost["input"] * tier_multiplier if model_cost else None,
    )
    output_cost = _first(
        config.get("outputCost"),
        config.get("cost"),
        model_cost["output"] * tier_multiplier if model_cost else None,
    )
    # Explicit rates also price private deployment aliases without a model-table entry.
    if input_cost is None or output_cost is None:
        return None
    cache_read_cost = _first(
        config.get("cacheReadCost"),
        model_cost["cache_read"] * tier_multiplier if model_cost else None,
        input_cost,
    )
    input_total = input_cost * uncached_prompt_tokens
    cache_read_total = cache_read_cost * billable_cached
    output_total = output_cost * completion_tokens
    logger.debug(
        "MiniMax cost calculation for %s: promptTokens=%s, completionTokens=%s, "
        "cachedTokens=%s, inputCost=%s, cacheReadCost=%s, outputCost=%s",
        model_name,
        prompt_tokens,
        completion_tokens,
        billable_cached,
        input_total,
        cache_read_total,
        output_total,
    )
    return input_total + cache_read_total + output_total


class MiniMaxProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        outer = provider_options.get("config") or {}
        minimax_config = outer.get("config") or {}
        resolved = {
            **minimax_config,
            "apiKeyEnvar": _first(minimax_config.get("apiKeyEnvar"), MINIMAX_API_KEY_ENV_VAR),
            "apiBaseUrl": _first(minimax_config.get("apiBaseUrl"), MINIMAX_API_BASE_URL),
        }
        options: dict[str, Any] = {
            "env": _first(outer.get("env"), provider_options.get("env")),
            "config": resolved,
        }
        provider_id = _first(outer.get("id"), provider_options.get("id"))
        if provider_id is not None:
            options["id"] = provider_id
        super().__init__(model_name, options)

    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        return resolve_configured_api_key(self)

    def get_organization(self, config: dict[str, Any] | None = None) -> str | None:
        return None

    def get_api_url(self, config: dict[str, Any] | None = None) -> str:
        return _first(self.config.get("apiBaseUrl"), MINIMAX_API_BASE_URL)

    def get_missing_api_key_error_message(self, config: dict[str, Any] | None = None) -> str:
        envar = _first(self.config.get("apiKeyEnvar"), MINIMAX_API_KEY_ENV_VAR)
        return (
            f"MiniMax API key is not set. Set the {envar} "
            "environment variable or add `apiKey` to the provider config."
        )

    def default_id(self) -> str:
        return f"minimax:{self.model_name}"

    def __str__(self) -> str:
        return f"[MiniMax Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "minimax")

    async def get_openai_body(self, prompt, context=None, call_api_options=None):
        result = await super().get_openai_body(prompt, context, call_api_options)
        body, config = result["body"], result["config"]
        if "function_call" in body:
            raise ValueError(
                "MiniMax does not support function_call. Use tools and tool_choice for tool "
                "calling instead."
            )
        # Prompt-level limits beat provider-level aliases; passthrough wins within a layer.
        prompt_config = ((context or {}).get("prompt") or {}).get("config") or {}
        prompt_passthrough = prompt_config.get("passthrough") or {}
        passthrough = config.get("passthrough") or {}
        max_completion_tokens = _first(
            prompt_passthrough.get("max_completion_tokens"),
            prompt_passthrough.get("max_tokens"),
            prompt_config.get("max_completion_tokens"),
            prompt_config.get("max_tokens"),
            passthrough.get("max_completion_tokens"),
            passthrough.get("max_tokens"),
            config.get("max_completion_tokens"),
            config.get("max_tokens"),
        )
        if max_completion_tokens is None:
            body.pop("max_completion_tokens", None)
        else:
            body["max_completion_tokens"] = max_completion_tokens
        body.pop("max_tokens", None)
        # Let MiniMax apply its own sampling defaults, and keep OPENAI_* env defaults out.
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if config.get(key) is None and passthrough.get(key) is None:
                body.pop(key, None)
        return result

    def calculate_response_cost(
        self, data: Any, config: dict[str, Any], cached: bool
    ) -> float | None:
        if cached:
            return 0
        data = data if isinstance(data, dict) else {}
        passthrough = config.get("passthrough") or {}
        passthrough_model = passthrough.get("model")
        model_name = passthrough_model if isinstance(passthrough_model, str) else self.model_name
        service_tier = _first(data.get("service_tier"), config.get("service_tier"))
        cost_config = {**config, "service_tier": "priority" if service_tier == "priority" else None}
        usage = data.get("usage") or {}
        details = get_openai_completion_token_details(usage) or {}
        return calculate_minimax_cost(
            model_name,
            cost_config,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            details.get("cacheReadInputTokens"),
        )


def create_minimax_provider(provider_path: str, options: dict[str, Any] | None = None) -> Any:
    model_name = ":".join(provider_path.split(":")[1:]) or "MiniMax-M3"
    return MiniMaxProvider(model_name, options or {})


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """`configuredProviderFactory('minimax:', ...)` in src/providers/registry.ts."""
    return create_minimax_provider(provider_path, configured_provider_options(options))
