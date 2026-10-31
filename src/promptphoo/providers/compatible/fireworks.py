"""Fireworks AI chat and embedding providers. Ports src/providers/fireworks/{chat,embedding,
shared}.ts and the `fireworks:` factory in src/providers/registry.ts.
"""

from __future__ import annotations

from typing import Any

from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.credentials import resolve_provider_api_key, resolve_provider_env
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.embedding import OpenAiEmbeddingProvider

FIREWORKS_API_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_FIREWORKS_API_KEY_ENVAR = "FIREWORKS_API_KEY"

# `embedding`/`embeddings` route to the dedicated Fireworks embedding provider.
FIREWORKS_EMBEDDING_SUBTYPES = frozenset(["embedding", "embeddings"])
# Subtypes reserved for future dedicated providers; fail fast instead of routing to chat.
FIREWORKS_RESERVED_PROVIDER_SUBTYPES = frozenset(
    ["chat", "completion", "image", "moderation", "realtime", "responses"]
)


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def build_fireworks_provider_config(
    config: dict[str, Any] | None, env: dict[str, Any] | None
) -> dict[str, Any]:
    """The caller's config plus the resolved base URL and the Fireworks key envar. Never
    consults OPENAI_API_HOST / OPENAI_API_BASE_URL / OPENAI_BASE_URL."""
    env_base_url = (resolve_provider_env(env, ["FIREWORKS_API_BASE_URL"]) or {}).get("value")
    config = config or {}
    return {
        **config,
        "apiBaseUrl": config.get("apiBaseUrl") or env_base_url or FIREWORKS_API_BASE_URL,
        "apiKeyEnvar": config.get("apiKeyEnvar") or DEFAULT_FIREWORKS_API_KEY_ENVAR,
    }


def resolve_fireworks_api_key(
    config: dict[str, Any] | None, env: dict[str, Any] | None
) -> str | None:
    """Never falls back to OPENAI_API_KEY."""
    return resolve_provider_api_key(config, env, [DEFAULT_FIREWORKS_API_KEY_ENVAR])


def resolve_fireworks_api_url(config: dict[str, Any] | None) -> str:
    config = config or {}
    if config.get("apiHost"):
        return f"https://{config['apiHost']}/v1"
    return config.get("apiBaseUrl") or FIREWORKS_API_BASE_URL


def calculate_fireworks_cost(
    config: dict[str, Any],
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    cached: bool = False,
    cached_input_tokens: float = 0,
) -> float | None:
    # Promptfoo's own response cache: nothing was sent to the provider.
    if cached:
        return 0
    input_cost = _first(config.get("inputCost"), config.get("cost"))
    output_cost = _first(config.get("outputCost"), config.get("cost"))
    if input_cost is None or output_cost is None or prompt_tokens is None:
        return None
    if completion_tokens is None:
        return None
    # Split prompt tokens into freshly billed and provider cache-hit halves.
    cache_hit_tokens = min(max(cached_input_tokens, 0), prompt_tokens)
    uncached_prompt_tokens = prompt_tokens - cache_hit_tokens
    cache_read_cost = _first(config.get("cacheReadInputCost"), input_cost)
    return (
        input_cost * uncached_prompt_tokens
        + cache_read_cost * cache_hit_tokens
        + output_cost * completion_tokens
    )


def _parse_int(value: str) -> int | None:
    """JS `Number.parseInt(value, 10)`; None stands in for NaN."""
    text = value.lstrip()
    sign = ""
    if text and text[0] in "+-":
        sign, text = text[0], text[1:]
    digits = ""
    for ch in text:
        if not ch.isdigit():
            break
        digits += ch
    return int(sign + digits) if digits else None


def read_fireworks_cached_prompt_tokens(response: ProviderResponse) -> float:
    """Prompt-cache hits, from the usage details or the `fireworks-cached-prompt-tokens`
    header, whichever is larger."""
    token_usage = response.get("tokenUsage") or {}
    from_usage = _first((token_usage.get("completionDetails") or {}).get("cacheReadInputTokens"), 0)
    metadata = response.get("metadata")
    http = metadata.get("http") if isinstance(metadata, dict) else None
    headers = http.get("headers") if isinstance(http, dict) else None
    header_value = (
        _first(
            headers.get("fireworks-cached-prompt-tokens"),
            headers.get("Fireworks-Cached-Prompt-Tokens"),
        )
        if isinstance(headers, dict)
        else None
    )
    from_header = _parse_int(header_value) if isinstance(header_value, str) else 0
    return max(from_usage, from_header if from_header is not None and from_header > 0 else 0)


class FireworksProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any]) -> None:
        super().__init__(
            model_name,
            {
                **provider_options,
                "config": build_fireworks_provider_config(
                    provider_options.get("config"), provider_options.get("env")
                ),
            },
        )

    def default_id(self) -> str:
        return f"fireworks:{self.model_name}"

    def __str__(self) -> str:
        return f"[Fireworks AI Provider {self.model_name}]"

    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        return resolve_fireworks_api_key(self.config, self.env)

    def get_organization(self, config: dict[str, Any] | None = None) -> str | None:
        return None

    def get_api_url(self, config: dict[str, Any] | None = None) -> str:
        return resolve_fireworks_api_url(self.config)

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        response = await super().call_api(prompt, context, options)
        metadata = response.get("metadata") or {}
        if response.get("error") and (
            metadata.get("errorOrigin") != "tool" or not response.get("tokenUsage")
        ):
            return response
        config = {**self.config, **(((context or {}).get("prompt") or {}).get("config") or {})}
        token_usage = response.get("tokenUsage") or {}
        cost = calculate_fireworks_cost(
            config,
            token_usage.get("prompt"),
            token_usage.get("completion"),
            bool(response.get("cached")),
            read_fireworks_cached_prompt_tokens(response),
        )
        # Only overwrite when a cost was computed.
        if cost is not None:
            response["cost"] = cost
        return response


class FireworksEmbeddingProvider(OpenAiEmbeddingProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any]) -> None:
        super().__init__(
            model_name,
            {
                **provider_options,
                "config": build_fireworks_provider_config(
                    provider_options.get("config"), provider_options.get("env")
                ),
            },
        )

    def default_id(self) -> str:
        return f"fireworks:embedding:{self.model_name}"

    def __str__(self) -> str:
        return f"[Fireworks AI Embedding Provider {self.model_name}]"

    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        return resolve_fireworks_api_key(self.config, self.env)

    def get_organization(self, config: dict[str, Any] | None = None) -> str | None:
        return None

    def get_api_url(self, config: dict[str, Any] | None = None) -> str:
        return resolve_fireworks_api_url(self.config)


def create_fireworks_provider(provider_path: str, provider_options: dict[str, Any]) -> Any:
    splits = provider_path.split(":")
    subtype = splits[1] if len(splits) > 1 else None
    constructor_options = {
        **(provider_options.get("config") or {}),
        "env": provider_options.get("env"),
    }
    if subtype in FIREWORKS_EMBEDDING_SUBTYPES:
        embedding_model = ":".join(splits[2:])
        if not embedding_model:
            raise ValueError(
                "Fireworks embedding provider needs a model identifier (e.g. "
                '"fireworks:embedding:accounts/fireworks/models/qwen3-embedding-8b"), got '
                f'"{provider_path}".'
            )
        return FireworksEmbeddingProvider(embedding_model, constructor_options)
    if subtype in FIREWORKS_RESERVED_PROVIDER_SUBTYPES:
        raise ValueError(
            f"The fireworks:{subtype}:* subtype is reserved for a future dedicated provider. "
            'Pass the model directly, e.g. "fireworks:accounts/fireworks/models/'
            'llama-v3p3-70b-instruct".'
        )
    model_name = ":".join(splits[1:])
    if not model_name:
        raise ValueError(
            "Fireworks provider needs a model identifier (e.g. "
            '"fireworks:accounts/fireworks/models/llama-v3p3-70b-instruct"), got '
            f'"{provider_path}".'
        )
    return FireworksProvider(model_name, constructor_options)


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `fireworks:` factory in src/providers/registry.ts. Python's `options["env"]` is
    already the merged context/provider env."""
    return create_fireworks_provider(provider_path, {"config": options, "env": options.get("env")})
