"""DeepSeek provider. Ports src/providers/deepseek.ts and the `deepseek:` registry entry."""

from __future__ import annotations

import logging
import math
from typing import Any

from promptphoo import envars
from promptphoo.providers.compatible.common import configured_provider_options, serialize_provider
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.shared import clamp_cached_tokens
from promptphoo.templating import render_vars_in_object

logger = logging.getLogger(__name__)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def get_deepseek_cached_tokens(usage: Any, prompt_tokens: Any = None) -> float:
    usage = usage if isinstance(usage, dict) else {}
    native_hits = _numeric(usage.get("prompt_cache_hit_tokens"))
    if native_hits is not None:
        return native_hits
    native_misses = _numeric(usage.get("prompt_cache_miss_tokens"))
    if native_misses is not None and _numeric(prompt_tokens) is not None:
        return prompt_tokens - native_misses
    details = usage.get("prompt_tokens_details")
    cached = _numeric(details.get("cached_tokens")) if isinstance(details, dict) else None
    return cached if cached is not None else 0


def _model(model_id: str, input: float, output: float, cache_read: float) -> dict[str, Any]:
    return {"id": model_id, "cost": {"input": input / 1e6, "output": output / 1e6,
                                     "cache_read": cache_read / 1e6}}  # fmt: skip


# Peak-hour estimates; off-peak rates are half. https://api-docs.deepseek.com/quick_start/pricing/
DEEPSEEK_CHAT_MODELS: list[dict[str, Any]] = [
    *(_model(m, 0.3, 1.2, 0.006)
      for m in ("deepseek-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp")),
    _model("deepseek-v4-pro", 1.32, 3.96, 0.044),
    # Retired models retain their historical rates.
    *(_model(m, 0.14, 0.28, 0.0028) for m in ("deepseek-chat", "deepseek-reasoner")),
]  # fmt: skip


def _valid_count(value: Any) -> bool:
    return _numeric(value) is not None and value >= 0


def _valid_rate(value: Any) -> bool:
    return _numeric(value) is not None and value >= 0


def calculate_deepseek_cost(
    model_name: str,
    config: dict[str, Any],
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    cached_tokens: Any = None,
) -> float | None:
    """Calculate DeepSeek cost based on model name and token usage."""
    if not _valid_count(prompt_tokens) or not _valid_count(completion_tokens):
        return None
    model = next((m for m in DEEPSEEK_CHAT_MODELS if m["id"] == model_name), None)
    if not model and all(
        config.get(key) is None for key in ("inputCost", "outputCost", "cost", "cacheReadCost")
    ):
        return None
    cost = model["cost"] if model else {}
    billable_cached = clamp_cached_tokens(cached_tokens, prompt_tokens)
    uncached_prompt = prompt_tokens - billable_cached
    input_cost = _first(config.get("inputCost"), config.get("cost"), cost.get("input"))
    output_cost = _first(config.get("outputCost"), config.get("cost"), cost.get("output"))
    cache_read_cost = _first(
        config.get("cacheReadCost"),
        config.get("inputCost"),
        config.get("cost"),
        cost.get("cache_read"),
        input_cost,
    )
    rates_and_tokens = [
        (input_cost, uncached_prompt),
        (cache_read_cost, billable_cached),
        (output_cost, completion_tokens),
    ]
    if any(tokens > 0 and not _valid_rate(rate) for rate, tokens in rates_and_tokens):
        return None
    input_total = input_cost * uncached_prompt if uncached_prompt > 0 else 0
    cache_read_total = cache_read_cost * billable_cached if billable_cached > 0 else 0
    output_total = output_cost * completion_tokens if completion_tokens > 0 else 0
    logger.debug(
        "DeepSeek cost calculation for %s: promptTokens=%s, completionTokens=%s, "
        "cachedTokens=%s, inputCost=%s, cacheReadCost=%s, outputCost=%s",
        model_name, prompt_tokens, completion_tokens, billable_cached,
        input_total, cache_read_total, output_total,
    )  # fmt: skip
    return input_total + cache_read_total + output_total


class DeepSeekProvider(OpenAiChatCompletionProvider):
    def __init__(
        self,
        model_name: str,
        provider_options: dict[str, Any] | None = None,
        uses_bare_model_default: bool = False,
    ) -> None:
        provider_options = provider_options or {}
        outer = provider_options.get("config") or {}
        # Extract the nested config.
        deepseek_config = outer.get("config") or {}
        api_base_url = _first(
            deepseek_config.get("apiBaseUrl"),
            outer.get("apiBaseUrl"),
            "https://api.deepseek.com/v1",
        )
        super().__init__(
            model_name,
            {
                **provider_options,
                "env": _first(outer.get("env"), provider_options.get("env")),
                "config": {
                    **outer,
                    **deepseek_config,
                    "apiKeyEnvar": "DEEPSEEK_API_KEY",
                    "apiBaseUrl": api_base_url,
                },
            },
        )
        self.uses_bare_model_default = uses_bare_model_default

    def default_id(self) -> str:
        return f"deepseek:{self.model_name}"

    def __str__(self) -> str:
        return f"[DeepSeek Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "deepseek", lambda: (self.config or {}).get("apiKey"))

    async def get_openai_body(self, prompt, context=None, call_api_options=None):
        result = await super().get_openai_body(prompt, context, call_api_options)
        body, config = result["body"], result["config"]
        passthrough = config.get("passthrough") or {}
        # Let DeepSeek choose its reasoning budget instead of the inherited 1,024-token limit.
        if (
            config.get("max_tokens") is None
            and passthrough.get("max_tokens") is None
            and envars.get_env_string("OPENAI_MAX_TOKENS") is None
        ):
            body.pop("max_tokens", None)
        if config.get("reasoning_effort") is not None:
            body["reasoning_effort"] = render_vars_in_object(
                config["reasoning_effort"], (context or {}).get("vars")
            )
        if self.uses_bare_model_default and "model" not in passthrough and "thinking" not in body:
            body["thinking"] = {"type": "disabled"}
        return result

    def calculate_response_cost(self, data, config, cached):
        if cached:
            return 0
        data = data if isinstance(data, dict) else {}
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        passthrough = config.get("passthrough") or {}
        passthrough_model = passthrough.get("model") if isinstance(passthrough, dict) else None
        model_name = passthrough_model if isinstance(passthrough_model, str) else self.model_name
        prompt_tokens = (usage or {}).get("prompt_tokens")
        return calculate_deepseek_cost(
            model_name,
            config,
            prompt_tokens,
            (usage or {}).get("completion_tokens"),
            get_deepseek_cached_tokens(usage, prompt_tokens),
        )


def create_deepseek_provider(provider_path: str, options: dict[str, Any] | None = None):
    explicit_model_name = ":".join(provider_path.split(":")[1:])
    uses_bare_model_default = len(explicit_model_name) == 0
    model_name = explicit_model_name or "deepseek-flash"
    return DeepSeekProvider(model_name, options or {}, uses_bare_model_default)


def create_provider(provider_path: str, options: dict[str, Any]):
    return create_deepseek_provider(provider_path, configured_provider_options(options))
