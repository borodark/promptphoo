"""Cloudflare AI Gateway: routes OpenAI-compatible providers and Anthropic through
`https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/{provider}`. Ports
src/providers/cloudflare-gateway.ts and the `cloudflare-gateway:` factory in
src/providers/registry.ts.

The Anthropic route subclasses promptphoo's Messages provider. Node isolates the SDK client from
`ANTHROPIC_CUSTOM_HEADERS` (buildIsolatedAnthropicClientOptions); here that is done by ignoring
the env custom headers, so only the provider's own `x-api-key` and config headers are sent.
"""

from __future__ import annotations

import logging
from typing import Any

from promptphoo import envars
from promptphoo.providers.anthropic.messages import AnthropicMessagesProvider
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider

logger = logging.getLogger(__name__)

CLOUDFLARE_GATEWAY_BASE_URL = "https://gateway.ai.cloudflare.com/v1"

# AWS Bedrock is not supported: it needs request signing, which the gateway proxy breaks.
PROVIDER_CONFIGS: dict[str, dict[str, str]] = {
    "openai": {"apiKeyEnvar": "OPENAI_API_KEY"},
    "anthropic": {"apiKeyEnvar": "ANTHROPIC_API_KEY"},
    "groq": {"apiKeyEnvar": "GROQ_API_KEY"},
    "perplexity-ai": {"apiKeyEnvar": "PERPLEXITY_API_KEY"},
    "google-ai-studio": {"apiKeyEnvar": "GOOGLE_API_KEY"},
    "mistral": {"apiKeyEnvar": "MISTRAL_API_KEY"},
    "cohere": {"apiKeyEnvar": "COHERE_API_KEY"},
    "azure-openai": {"apiKeyEnvar": "AZURE_OPENAI_API_KEY"},
    "workers-ai": {"apiKeyEnvar": "CLOUDFLARE_API_KEY"},
    "huggingface": {"apiKeyEnvar": "HUGGINGFACE_API_KEY"},
    "replicate": {"apiKeyEnvar": "REPLICATE_API_KEY"},
    "grok": {"apiKeyEnvar": "XAI_API_KEY"},
}

_CLOUDFLARE_ONLY_KEYS = (
    "accountId", "accountIdEnvar", "gatewayId", "gatewayIdEnvar", "cfAigToken",
    "cfAigTokenEnvar", "resourceName", "deploymentName", "apiVersion",
)  # fmt: skip


def _invariant(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(f"Invariant failed: {message}")


def _get_custom_env_value(name: str, env: dict[str, Any] | None) -> str | None:
    value = envars.get_provider_env_string(env, name)
    return value if value is not None else envars.get_env_string(name)


def _get_env_id(
    config: dict[str, Any], env: dict[str, Any] | None, key: str, default_envar: str, label: str
) -> str | None:
    custom_envar = config.get(f"{key}Envar")
    custom_value = _get_custom_env_value(custom_envar, env) if custom_envar else None
    if custom_envar and custom_value is None:
        logger.warning(
            "[CloudflareGateway] Custom %s environment variable '%s' is not set. "
            "Falling back to %s.",
            label,
            custom_envar,
            default_envar,
        )
    return custom_value if custom_value is not None else _get_custom_env_value(default_envar, env)


def get_account_id(config: dict[str, Any] | None = None, env: dict[str, Any] | None = None) -> str:
    config = config or {}
    if config.get("accountId"):
        return config["accountId"]
    candidate = _get_env_id(config, env, "accountId", "CLOUDFLARE_ACCOUNT_ID", "account ID")
    _invariant(
        candidate,
        "Cloudflare account ID required. Supply it via config accountId or accountIdEnvar, or "
        "the CLOUDFLARE_ACCOUNT_ID environment variable",
    )
    assert candidate is not None
    return candidate


def get_gateway_id(config: dict[str, Any] | None = None, env: dict[str, Any] | None = None) -> str:
    config = config or {}
    if config.get("gatewayId"):
        return config["gatewayId"]
    candidate = _get_env_id(config, env, "gatewayId", "CLOUDFLARE_GATEWAY_ID", "gateway ID")
    _invariant(
        candidate,
        "Cloudflare AI Gateway ID required. Supply it via config gatewayId or gatewayIdEnvar, "
        "or the CLOUDFLARE_GATEWAY_ID environment variable",
    )
    assert candidate is not None
    return candidate


def get_cf_aig_token(
    config: dict[str, Any] | None = None, env: dict[str, Any] | None = None
) -> str | None:
    config = config or {}
    if config.get("cfAigToken"):
        return config["cfAigToken"]
    if config.get("cfAigTokenEnvar"):
        custom_value = _get_custom_env_value(config["cfAigTokenEnvar"], env)
        if custom_value is not None:
            return custom_value
    return _get_custom_env_value("CF_AIG_TOKEN", env)


def build_gateway_url(
    account_id: str,
    gateway_id: str,
    provider: str,
    config: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> str:
    base_url = f"{CLOUDFLARE_GATEWAY_BASE_URL}/{account_id}/{gateway_id}"
    config = config or {}
    if provider == "azure-openai":
        resource_name = config.get("resourceName")
        deployment_name = config.get("deploymentName")
        _invariant(
            resource_name,
            "Azure OpenAI requires resourceName in config. Example: "
            "cloudflare-gateway:azure-openai:gpt-4 with config.resourceName set",
        )
        _invariant(
            deployment_name,
            "Azure OpenAI requires deploymentName in config. Example: "
            "cloudflare-gateway:azure-openai:gpt-4 with config.deploymentName set",
        )
        return f"{base_url}/azure-openai/{resource_name}/{deployment_name}"
    # Mistral's native gateway proxy retains the upstream /v1 path.
    if provider == "mistral":
        return f"{base_url}/mistral/v1"
    if provider == "workers-ai":
        _invariant(
            model_name, "Workers AI requires a model name (e.g., @cf/meta/llama-3.1-8b-instruct)"
        )
        return f"{base_url}/workers-ai/{model_name}"
    return f"{base_url}/{provider}"


def _get_passthrough_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {k: v for k, v in (config or {}).items() if k not in _CLOUDFLARE_ONLY_KEYS}


def _gateway_headers(config: dict[str, Any], cf_aig_token: str | None) -> dict[str, str]:
    headers: dict[str, str] = {**(config.get("headers") or {})}
    if cf_aig_token:
        headers["cf-aig-authorization"] = f"Bearer {cf_aig_token}"
    return headers


class CloudflareGatewayOpenAiProvider(OpenAiChatCompletionProvider):
    """Routes OpenAI, Groq, Perplexity, Mistral, etc. through Cloudflare AI Gateway."""

    def __init__(
        self,
        underlying_provider: str,
        model_name: str,
        provider_options: dict[str, Any] | None = None,
    ) -> None:
        provider_options = provider_options or {}
        options_config = provider_options.get("config") or {}
        env = provider_options.get("env")
        account_id = get_account_id(options_config, env)
        gateway_id = get_gateway_id(options_config, env)
        gateway_url = build_gateway_url(
            account_id, gateway_id, underlying_provider, options_config, model_name
        )
        passthrough = _get_passthrough_config(options_config)
        provider_config = PROVIDER_CONFIGS.get(underlying_provider)
        cf_aig_token = get_cf_aig_token(options_config, env)
        headers = _gateway_headers(options_config, cf_aig_token)

        final_gateway_url = gateway_url
        api_key_envar: str | None
        if underlying_provider == "azure-openai":
            azure_api_key = options_config.get("apiKey") or _get_custom_env_value(
                "AZURE_OPENAI_API_KEY", env
            )
            _invariant(
                azure_api_key,
                "Azure OpenAI API key is required. Set the AZURE_OPENAI_API_KEY environment "
                "variable or add apiKey to the provider config.",
            )
            assert azure_api_key is not None
            # Azure uses the api-key header, not an Authorization bearer.
            headers["api-key"] = azure_api_key
            api_key_envar = None
            api_version = options_config.get("apiVersion") or "2024-12-01-preview"
            final_gateway_url = f"{gateway_url}?api-version={api_version}"
        else:
            api_key_envar = options_config.get("apiKeyEnvar") or (provider_config or {}).get(
                "apiKeyEnvar"
            )

        config: dict[str, Any] = {**passthrough, "apiBaseUrl": final_gateway_url}
        if api_key_envar is not None:
            config["apiKeyEnvar"] = api_key_envar
        else:
            config.pop("apiKeyEnvar", None)
        if headers:
            config["headers"] = headers
        else:
            config.pop("headers", None)
        self.underlying_provider = underlying_provider
        super().__init__(model_name, {**provider_options, "config": config})
        logger.debug(
            "[CloudflareGateway] Configured %s:%s gatewayUrl=%s hasApiKey=%s hasCfAigToken=%s",
            underlying_provider,
            model_name,
            final_gateway_url,
            bool(options_config.get("apiKey")),
            bool(cf_aig_token),
        )

    def get_gen_ai_system(self) -> str:
        return self.underlying_provider

    def default_id(self) -> str:
        return f"cloudflare-gateway:{self.underlying_provider}:{self.model_name}"

    def __str__(self) -> str:
        return f"[Cloudflare AI Gateway {self.underlying_provider} Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        # Never expose API keys in JSON output.
        return {
            "provider": "cloudflare-gateway",
            "underlyingProvider": self.underlying_provider,
            "model": self.model_name,
            "config": {k: v for k, v in self.config.items() if k != "apiKey"},
        }


def _get_anthropic_passthrough_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    if not config:
        return {}
    keys = ("apiKey", "max_tokens", "temperature", "top_p", "top_k", "cost")
    return {k: config[k] for k in keys if config.get(k) is not None}


class CloudflareGatewayAnthropicProvider(AnthropicMessagesProvider):
    """Routes Anthropic through Cloudflare AI Gateway."""

    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        options_config = provider_options.get("config") or {}
        env = provider_options.get("env")
        account_id = get_account_id(options_config, env)
        gateway_id = get_gateway_id(options_config, env)
        gateway_url = build_gateway_url(account_id, gateway_id, "anthropic")
        passthrough = _get_anthropic_passthrough_config(options_config)
        cf_aig_token = get_cf_aig_token(options_config, env)
        headers = _gateway_headers(options_config, cf_aig_token)
        config: dict[str, Any] = {**passthrough, "apiBaseUrl": gateway_url}
        if headers:
            config["headers"] = headers
        super().__init__(model_name, {**provider_options, "config": config})
        logger.debug(
            "[CloudflareGateway] Configured anthropic:%s gatewayUrl=%s hasApiKey=%s "
            "hasCfAigToken=%s",
            model_name,
            gateway_url,
            bool(options_config.get("apiKey")),
            bool(cf_aig_token),
        )

    # ANTHROPIC_CUSTOM_HEADERS is scoped to api.anthropic.com and never rides to the gateway.
    def _custom_header_value(self) -> str | None:
        return None

    def has_custom_headers(self) -> bool:
        return False

    def allows_claude_generation_fallback(self) -> bool:
        # The dedicated Anthropic gateway forwards canonical model IDs unchanged.
        return True

    def default_id(self) -> str:
        return f"cloudflare-gateway:anthropic:{self.model_name}"

    def __str__(self) -> str:
        return f"[Cloudflare AI Gateway Anthropic Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return {
            "provider": "cloudflare-gateway",
            "underlyingProvider": "anthropic",
            "model": self.model_name,
            "config": {k: v for k, v in self.config.items() if k != "apiKey"},
        }


def create_cloudflare_gateway_provider(
    provider_path: str, options: dict[str, Any] | None = None
) -> Any:
    options = options or {}
    splits = provider_path.split(":")
    if len(splits) < 3:
        raise ValueError(
            f'Invalid cloudflare-gateway provider path: "{provider_path}". '
            "Expected format: cloudflare-gateway:{provider}:{model} "
            "(e.g., cloudflare-gateway:openai:gpt-4o)"
        )
    underlying_provider = splits[1]
    model_name = ":".join(splits[2:])
    _invariant(model_name, "Model name is required for cloudflare-gateway provider")
    if underlying_provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f'Unsupported Cloudflare AI Gateway provider: "{underlying_provider}". '
            f"Supported providers: {', '.join(PROVIDER_CONFIGS)}"
        )
    if underlying_provider == "anthropic":
        return CloudflareGatewayAnthropicProvider(model_name, options)
    return CloudflareGatewayOpenAiProvider(underlying_provider, model_name, options)


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `cloudflare-gateway:` registry factory: `{...providerOptions, env: context.env}`."""
    return create_cloudflare_gateway_provider(provider_path, {**options, "env": options.get("env")})
