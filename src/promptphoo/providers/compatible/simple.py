"""Small OpenAI chat subclasses that only preset a URL, key env var, id or label. Ports
src/providers/atlascloud.ts, llamaApi.ts, cloudera.ts, jfrog.ts, helicone.ts and envoy.ts, with
their registry.ts entries.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from promptphoo import envars
from promptphoo.providers.compatible.common import (
    configured_provider_options,
    resolve_provider_creator_input,
    serialize_provider,
)
from promptphoo.providers.credentials import resolve_provider_env
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider

ATLASCLOUD_API_BASE = "https://api.atlascloud.ai/v1"


class AtlasCloudProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        config = provider_options.get("config") or {}
        super().__init__(
            model_name,
            {
                **provider_options,
                "config": {
                    **config,
                    "apiBaseUrl": config.get("apiBaseUrl") or ATLASCLOUD_API_BASE,
                    "apiKeyEnvar": config.get("apiKeyEnvar") or "ATLASCLOUD_API_KEY",
                },
            },
        )

    def default_id(self) -> str:
        return f"atlascloud:{self.model_name}"

    def __str__(self) -> str:
        return f"[Atlas Cloud Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "atlascloud")


def create_atlascloud_provider(provider_path: str, options: dict[str, Any] | None = None):
    options = options or {}
    outer = options.get("config") or {}
    model_name = ":".join(provider_path.split(":")[1:])
    env = outer.get("env") if outer.get("env") is not None else options.get("env")
    return AtlasCloudProvider(model_name, {**outer, "env": env})


class LlamaApiProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        config = options.get("config") or {}
        super().__init__(
            model_name,
            {
                **options,
                "config": {
                    **config,
                    "apiBaseUrl": "https://api.llama.com/compat/v1",
                    "apiKeyEnvar": "LLAMA_API_KEY",
                    "passthrough": {**(config.get("passthrough") or {})},
                },
            },
        )

    def default_id(self) -> str:
        return f"llamaapi:{self.model_name}"

    def __str__(self) -> str:
        return f"[Llama API Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        rest = {k: v for k, v in (self.config or {}).items() if k != "apiKey"}
        return {"provider": "llamaapi", "model": self.model_name, "config": rest}


def create_llama_api_provider(provider_path: str, options: dict[str, Any] | None = None):
    """Creates a Llama API provider using OpenAI-compatible endpoints."""
    options = options or {}
    splits = provider_path.split(":")
    model_name = ":".join(splits[2 if len(splits) > 1 and splits[1] == "chat" else 1 :])
    provider_options = {**(options.get("config") or {}), "env": options.get("env")}
    if options.get("id") is not None:
        provider_options["id"] = options["id"]
    else:
        provider_options.pop("id", None)
    return LlamaApiProvider(model_name, provider_options)


class ClouderaAiChatCompletionProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any]) -> None:
        config = provider_options.get("config") or {}
        # https://docs.cloudera.com/machine-learning/cloud/ai-inference/topics/ml-caii-openai-inference-protocol-using-curl.html
        domain = (
            config.get("domain")
            or envars.get_provider_env_string(provider_options.get("env"), "CDP_DOMAIN")
            or envars.get_env_string("CDP_DOMAIN")
        )
        namespace = config.get("namespace") or "serving-default"
        endpoint = config.get("endpoint") or model_name
        super().__init__(
            model_name,
            {
                **provider_options,
                "config": {
                    **config,
                    "apiKeyEnvar": config.get("apiKeyEnvar") or "CDP_TOKEN",
                    "apiBaseUrl": config.get("apiBaseUrl")
                    or f"https://{_js(domain)}/namespaces/{namespace}/endpoints/{endpoint}/v1",
                },
            },
        )


def _js(value: Any) -> str:
    return "undefined" if value is None else str(value)


class JfrogMlChatCompletionProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any]) -> None:
        config = provider_options.get("config") or {}
        base_url = re.sub(
            r"/+$", "", config.get("baseUrl") or "https://models.qwak-prod.qwak.ai/v1"
        )
        super().__init__(
            model_name,
            {
                **provider_options,
                "config": {
                    **config,
                    "apiKeyEnvar": config.get("apiKeyEnvar") or "QWAK_TOKEN",
                    "apiBaseUrl": config.get("apiBaseUrl") or f"{base_url}/{model_name}",
                },
            },
        )


class HeliconeGatewayProvider(OpenAiChatCompletionProvider):
    """Routes requests through a self-hosted Helicone AI Gateway instance."""

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        config = options.get("config") or {}
        # Default base URL to localhost Helicone AI Gateway.
        base_url = config.get("baseUrl") or "http://localhost:8080"
        # Determine endpoint based on router configuration.
        if config.get("router"):
            api_base_url = f"{base_url}/router/{config['router']}"
        else:
            api_base_url = f"{base_url}/ai"
        # Use the model from config or the provided modelName.
        model = config.get("model") or model_name
        super().__init__(model, {**options, "config": {**config, "apiBaseUrl": api_base_url}})
        self.helicone_config = config

    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        return (
            self.config.get("apiKey")
            or envars.get_provider_env_string(self.env, "HELICONE_API_KEY")
            or envars.get_env_string("HELICONE_API_KEY")
            or "placeholder-api-key"
        )

    def default_id(self) -> str:
        router = f":{self.helicone_config['router']}" if self.helicone_config.get("router") else ""
        return f"helicone-gateway{router}:{self.model_name}"

    def __str__(self) -> str:
        base_url = (
            self.config.get("apiBaseUrl")
            or self.helicone_config.get("baseUrl")
            or "http://localhost:8080"
        )
        return f"[Helicone AI Gateway {base_url}]"


def _parse_gateway_url(value: str) -> Any:
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Unsupported gateway protocol")
    return parsed


def create_envoy_provider(provider_path: str, options: dict[str, Any] | None = None):
    """Creates an Envoy AI Gateway provider using OpenAI-compatible endpoints."""
    provider_options = resolve_provider_creator_input(options)
    model_name = ":".join(provider_path.split(":")[1:])
    if not model_name:
        raise ValueError("Envoy provider requires a model name. Use format: envoy:<model_name>")
    # Filter out basePath from config to avoid passing it to the API.
    config = dict(provider_options.get("config") or {})
    config.pop("basePath", None)
    configured_base_url = config.pop("apiBaseUrl", None)
    env = provider_options.get("env")
    resolved = resolve_provider_env(env, ["ENVOY_API_BASE_URL"])
    api_base_url = (
        configured_base_url if configured_base_url is not None else (resolved or {}).get("value")
    )
    if not api_base_url:
        raise ValueError(
            "Envoy provider requires a gateway URL. Set ENVOY_API_BASE_URL environment variable "
            "or specify apiBaseUrl in config."
        )
    try:
        parsed = _parse_gateway_url(api_base_url)
    except ValueError:
        raise ValueError(
            "Envoy provider requires a valid gateway URL. Check ENVOY_API_BASE_URL or "
            "config.apiBaseUrl."
        ) from None
    if configured_base_url is None:
        base_path = re.sub(r"/+$", "", parsed.path)
        path = base_path if base_path.endswith("/v1") else f"{base_path}/v1"
        api_base_url = urlunsplit(parsed._replace(path=path))
    return OpenAiChatCompletionProvider(
        model_name,
        {**provider_options, "env": env, "config": {**config, "apiBaseUrl": api_base_url}},
    )


# --- registry entries ---------------------------------------------------------------------------


def create_atlascloud(provider_path: str, options: dict[str, Any]):
    return create_atlascloud_provider(provider_path, configured_provider_options(options))


def create_llamaapi(provider_path: str, options: dict[str, Any]):
    return create_llama_api_provider(provider_path, configured_provider_options(options))


def create_cloudera(provider_path: str, options: dict[str, Any]):
    model_name = provider_path.split(":")[1]
    return ClouderaAiChatCompletionProvider(
        model_name, {**options, "config": options.get("config") or {}}
    )


def create_jfrog(provider_path: str, options: dict[str, Any]):
    model_name = ":".join(provider_path.split(":")[1:])
    return JfrogMlChatCompletionProvider(model_name, options)


def create_helicone(provider_path: str, options: dict[str, Any]):
    # Parse helicone:model format (e.g., helicone:openai/gpt-4o).
    model = provider_path[len("helicone:") :]
    if not model:
        raise ValueError(
            "Helicone provider requires a model in format helicone:<provider/model> (e.g., "
            "helicone:openai/gpt-4o, helicone:anthropic/claude-sonnet-5)"
        )
    return HeliconeGatewayProvider(model, options)


def create_envoy(provider_path: str, options: dict[str, Any]):
    return create_envoy_provider(provider_path, {"providerOptions": options})
