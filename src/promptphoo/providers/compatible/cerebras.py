"""Cerebras provider. Ports src/providers/cerebras.ts."""

from __future__ import annotations

import math
from typing import Any

from promptphoo.providers.compatible.common import (
    resolve_provider_creator_input,
    split_local_options,
)
from promptphoo.providers.openai.billing import (
    calculate_custom_usage_cost,
    extract_openai_billing_usage,
)
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.shared import calculate_cost

CEREBRAS_CHAT_MODELS: list[dict[str, Any]] = [
    {"id": "gpt-oss-120b", "cost": {"input": 0.35 / 1e6, "output": 0.75 / 1e6}},
    {"id": "gemma-4-31b", "cost": {"input": 0.99 / 1e6, "output": 1.49 / 1e6}},
    {"id": "zai-glm-4.7", "cost": {"input": 2.25 / 1e6, "output": 2.75 / 1e6}},
]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_known(model_name: str) -> bool:
    return any(model["id"] == model_name for model in CEREBRAS_CHAT_MODELS)


def calculate_cerebras_cost(
    model_name: str,
    config: dict[str, Any],
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
) -> float | None:
    if _is_known(model_name):
        return calculate_cost(
            model_name, config, prompt_tokens, completion_tokens, CEREBRAS_CHAT_MODELS
        )
    if not _finite(prompt_tokens) or not _finite(completion_tokens):
        return None
    return calculate_custom_usage_cost(
        extract_openai_billing_usage(
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        ),
        config,
        False,
    )


class CerebrasProvider(OpenAiChatCompletionProvider):
    def get_organization(self, config: dict[str, Any] | None = None) -> str | None:
        return self.config.get("organization")

    async def get_openai_body(self, prompt, context=None, call_api_options=None):
        result = await super().get_openai_body(prompt, context, call_api_options)
        body = result["body"]
        # Cerebras API doesn't support both max_tokens and max_completion_tokens.
        if body.get("max_completion_tokens"):
            body.pop("max_tokens", None)
        # Promptfoo pricing overrides are local billing metadata, not Cerebras request fields.
        for key in ("cost", "inputCost", "outputCost"):
            body.pop(key, None)
        return result

    def calculate_response_cost(self, data, config, cached):
        if cached:
            return 0
        passthrough = config.get("passthrough") or {}
        effective_config = {**passthrough, **config}
        # The request body uses the provider selector unless passthrough overrides it.
        model = passthrough.get("model")
        model_name = model if isinstance(model, str) else self.model_name
        usage = data.get("usage") if isinstance(data, dict) else None
        if usage:
            custom_cost = calculate_custom_usage_cost(
                extract_openai_billing_usage(usage), effective_config, False
            )
            if custom_cost is not None:
                return custom_cost
        if not _is_known(model_name):
            return None
        usage = usage if isinstance(usage, dict) else {}
        return calculate_cerebras_cost(
            model_name, effective_config, usage.get("prompt_tokens"), usage.get("completion_tokens")
        )


def create_cerebras_provider(provider_path: str, options: dict[str, Any] | None = None):
    provider_options = resolve_provider_creator_input(options)
    model_name = ":".join(provider_path.split(":")[1:])
    config = provider_options.get("config") or {}
    # Only genuine model parameters belong in the request body; promptfoo's own settings
    # (credentials, headers, cost overrides, basePath) stay local.
    local_options, model_parameters = split_local_options(config)
    return CerebrasProvider(
        model_name,
        {
            **provider_options,
            "config": {
                **local_options,
                "apiBaseUrl": local_options.get("apiBaseUrl") or "https://api.cerebras.ai/v1",
                "apiKeyEnvar": local_options.get("apiKeyEnvar") or "CEREBRAS_API_KEY",
                "passthrough": {**model_parameters, **(config.get("passthrough") or {})},
            },
        },
    )


def create_provider(provider_path: str, options: dict[str, Any]):
    return create_cerebras_provider(provider_path, {"providerOptions": options})
