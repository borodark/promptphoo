"""LiteLLM proxy providers (chat, completion, embedding). Ports src/providers/litellm.ts and the
`litellm:` entry of src/providers/families/compatible.ts.
"""

from __future__ import annotations

import re
from typing import Any

from promptphoo.providers.base import ApiProvider, ProviderResponse
from promptphoo.providers.compatible.common import resolve_provider_creator_input
from promptphoo.providers.credentials import resolve_provider_env
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.completion import OpenAiCompletionProvider
from promptphoo.providers.openai.embedding import OpenAiEmbeddingProvider
from promptphoo.providers.openai.util import has_openai_gateway_credentials

_AUTH_ERROR = re.compile(
    r"\b(?:401|unauthorized|authentication error|auth_error|invalid_api_key)\b", re.I
)
AUTH_HINT = (
    "No LiteLLM API key was configured. Set LITELLM_API_KEY or the provider's apiKey or "
    "apiKeyEnvar. OPENAI_API_KEY is not used by default."
)


class LiteLLMProviderWrapper(ApiProvider):
    """Wraps an OpenAI provider while keeping the LiteLLM identity."""

    def __init__(self, provider: Any, provider_type: str, provider_id: str | None = None) -> None:
        super().__init__({"id": provider_id, "env": provider.env})
        self.provider = provider
        self.provider_type = provider_type
        self.config = provider.config

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    @property
    def modelName(self) -> str:  # noqa: N802 - upstream attribute name
        return self.model_name

    def get_api_key(self) -> str | None:
        return self.provider.get_api_key()

    def default_id(self) -> str:
        type_prefix = "" if self.provider_type == "chat" else f":{self.provider_type}"
        return f"litellm{type_prefix}:{self.model_name}"

    def __str__(self) -> str:
        type_str = "" if self.provider_type == "chat" else f" {self.provider_type}"
        return f"[LiteLLM Provider{type_str} {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        config = dict(self.config or {})
        if self.get_api_key():
            config.pop("apiKey", None)
        return {
            "provider": "litellm",
            "model": self.model_name,
            "type": self.provider_type,
            "config": config,
        }

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        prompt_config = ((context or {}).get("prompt") or {}).get("config") or {}
        headers = (
            (
                prompt_config.get("headers")
                if prompt_config.get("headers") is not None
                else self.config.get("headers")
            )
            if self.provider_type == "chat"
            else self.config.get("headers")
        )
        return self.with_auth_hint(await self.provider.call_api(prompt, context, options), headers)

    def with_auth_hint(self, response: dict[str, Any], headers: Any = None) -> dict[str, Any]:
        headers = self.config.get("headers") if headers is None else headers
        error = response.get("error")
        if (
            not error
            or self.get_api_key()
            or has_openai_gateway_credentials(headers, self.provider.get_api_url())
            or not _AUTH_ERROR.search(error.split("\n", 1)[0])
        ):
            return response
        return {**response, "error": f"{error}\n{AUTH_HINT}"}


class LiteLLMChatProvider(LiteLLMProviderWrapper):
    def __init__(self, model_name: str, options: dict[str, Any]) -> None:
        super().__init__(
            OpenAiChatCompletionProvider(model_name, options), "chat", options.get("id")
        )


class LiteLLMCompletionProvider(LiteLLMProviderWrapper):
    def __init__(self, model_name: str, options: dict[str, Any]) -> None:
        super().__init__(
            OpenAiCompletionProvider(model_name, options), "completion", options.get("id")
        )


class LiteLLMEmbeddingProvider(LiteLLMProviderWrapper):
    supports_embedding_cancellation = True

    def __init__(self, model_name: str, options: dict[str, Any]) -> None:
        self.embedding_provider = OpenAiEmbeddingProvider(model_name, options)
        super().__init__(self.embedding_provider, "embedding", options.get("id"))

    async def call_embedding_api(self, text, context=None, options=None) -> dict[str, Any]:
        return self.with_auth_hint(
            await self.embedding_provider.call_embedding_api(text, context, options)
        )


class LiteLLMProvider(LiteLLMChatProvider):
    """Backward-compatible name for the chat provider."""


def create_litellm_provider(provider_path: str, options: dict[str, Any] | None = None) -> Any:
    options = options or {}
    provider_options = resolve_provider_creator_input(
        {**options, "id": (options.get("config") or {}).get("id") or options.get("id")}
        if options.get("providerOptions") is None
        else options
    )
    splits = provider_path.split(":")
    provider_type = splits[1] if len(splits) > 1 else ""
    model_name = (
        ":".join(splits[2:])
        if provider_type in ("chat", "completion", "embedding", "embeddings")
        else ":".join(splits[1:])
    )
    config = provider_options.get("config") or {}
    resolved_env = resolve_provider_env(provider_options.get("env"), ["LITELLM_API_BASE"])
    resolved_api_base_url = (
        config.get("apiBaseUrl")
        or (resolved_env["value"] if resolved_env else None)
        or "http://0.0.0.0:4000"
    )
    # omitDefaults lets the LiteLLM proxy apply its own model-specific defaults.
    merged_config: dict[str, Any] = {
        "apiKeyEnvar": "LITELLM_API_KEY",
        "apiKeyRequired": False,
        "apiBaseUrl": resolved_api_base_url,
        "omitDefaults": True,
    }
    for key, value in config.items():
        if value is not None:
            merged_config[key] = value
    litellm_options = {**provider_options, "config": merged_config}
    if provider_type == "completion":
        return LiteLLMCompletionProvider(model_name, litellm_options)
    if provider_type in ("embedding", "embeddings"):
        return LiteLLMEmbeddingProvider(model_name, litellm_options)
    return LiteLLMProvider(model_name, litellm_options)


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `litellm:` family factory (families/compatible.ts)."""
    return create_litellm_provider(provider_path, {"providerOptions": options})
