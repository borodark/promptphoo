"""Cloudflare Workers AI through its OpenAI-compatible endpoint. Ports
src/providers/cloudflare-ai.ts and the `cloudflare-ai:` factory in src/providers/registry.ts.
"""

from __future__ import annotations

from typing import Any

from promptphoo import envars
from promptphoo.providers.compatible.common import split_local_options
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.completion import OpenAiCompletionProvider
from promptphoo.providers.openai.embedding import OpenAiEmbeddingProvider


def _env_value(env: dict[str, Any] | None, name: str) -> str | None:
    value = envars.get_provider_env_string(env, name)
    return value if value is not None else envars.get_env_string(name)


def get_cloudflare_api_config(
    config: dict[str, Any] | None = None, env: dict[str, Any] | None = None
) -> dict[str, str]:
    config = config or {}
    named_api_token = _env_value(env, config["apiKeyEnvar"]) if config.get("apiKeyEnvar") else None
    api_token = config.get("apiKey") or (
        named_api_token if named_api_token is not None else _env_value(env, "CLOUDFLARE_API_KEY")
    )
    if not api_token:
        raise ValueError(
            "Invariant failed: Cloudflare API token required. Supply it via config apiKey or "
            "apiKeyEnvar, or the CLOUDFLARE_API_KEY environment variable"
        )
    named_account_id = (
        _env_value(env, config["accountIdEnvar"]) if config.get("accountIdEnvar") else None
    )
    account_id = config.get("accountId") or (
        named_account_id
        if named_account_id is not None
        else _env_value(env, "CLOUDFLARE_ACCOUNT_ID")
    )
    if not account_id:
        raise ValueError(
            "Invariant failed: Cloudflare account ID required. Supply it via config accountId or "
            "accountIdEnvar, or the CLOUDFLARE_ACCOUNT_ID environment variable"
        )
    return {"apiToken": api_token, "accountId": account_id}


def get_api_base_url(
    config: dict[str, Any] | None = None, env: dict[str, Any] | None = None
) -> str:
    if config and config.get("apiBaseUrl"):
        return config["apiBaseUrl"]
    account_id = get_cloudflare_api_config(config, env)["accountId"]
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"


def _get_openai_provider_options(provider_options: dict[str, Any]) -> dict[str, Any]:
    config = provider_options.get("config") or {}
    api_base_url = get_api_base_url(config, provider_options.get("env"))
    # Only genuine model parameters belong in the request body.
    _, model_parameters = split_local_options(config, ["accountId", "accountIdEnvar"])
    openai_config = {
        **config,
        "apiKeyEnvar": "CLOUDFLARE_API_KEY",
        "apiBaseUrl": api_base_url,
        "passthrough": {**model_parameters, **(config.get("passthrough") or {})},
    }
    return {**provider_options, "config": openai_config}


class _CloudflareAiMixin:
    model_type = ""
    model_name: str
    config: dict[str, Any]
    env: dict[str, Any] | None
    cloudflare_config: dict[str, Any]

    def default_id(self) -> str:
        return f"cloudflare-ai:{self.model_type}:{self.model_name}"

    def __str__(self) -> str:
        return f"[Cloudflare AI {self.model_type} Provider {self.model_name}]"

    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        return get_cloudflare_api_config(self.cloudflare_config, self.env)["apiToken"]

    def to_json(self) -> dict[str, Any]:
        config = dict(self.config)
        if self.get_api_key():
            config.pop("apiKey", None)
        return {
            "provider": "cloudflare-ai",
            "model": self.model_name,
            "modelType": self.model_type,
            "config": config,
        }


class CloudflareAiChatCompletionProvider(_CloudflareAiMixin, OpenAiChatCompletionProvider):
    model_type = "chat"

    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        super().__init__(model_name, _get_openai_provider_options(provider_options))
        self.cloudflare_config = provider_options.get("config") or {}


class CloudflareAiCompletionProvider(_CloudflareAiMixin, OpenAiCompletionProvider):
    model_type = "completion"

    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        super().__init__(model_name, _get_openai_provider_options(provider_options))
        self.cloudflare_config = provider_options.get("config") or {}


class CloudflareAiEmbeddingProvider(_CloudflareAiMixin, OpenAiEmbeddingProvider):
    model_type = "embedding"

    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        super().__init__(model_name, _get_openai_provider_options(provider_options))
        self.cloudflare_config = provider_options.get("config") or {}


def create_cloudflare_ai_provider(provider_path: str, options: dict[str, Any] | None = None) -> Any:
    options = options or {}
    splits = provider_path.split(":")
    model_type = splits[1] if len(splits) > 1 else None
    model_name = ":".join(splits[2:])
    if not model_name:
        raise ValueError("Invariant failed: Model name is required")
    if model_type == "chat":
        return CloudflareAiChatCompletionProvider(model_name, options)
    if model_type == "completion":
        return CloudflareAiCompletionProvider(model_name, options)
    if model_type in ("embedding", "embeddings"):
        return CloudflareAiEmbeddingProvider(model_name, options)
    raise ValueError(f"Unknown Cloudflare AI model type: {model_type}")


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `cloudflare-ai:` registry factory: `{...providerOptions, env: context.env}`."""
    return create_cloudflare_ai_provider(provider_path, {**options, "env": options.get("env")})
