"""Portkey AI gateway through its OpenAI-compatible endpoint. Ports src/providers/portkey.ts
and the `portkey:` factory in src/providers/registry.ts.
"""

from __future__ import annotations

import re
from typing import Any

from promptphoo.jsvalues import js_string, json_stringify
from promptphoo.providers.credentials import resolve_provider_env
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.generic import has_header_override

PORTKEY_CONFIG_PREFIX = "portkey"
# Sets the gateway URL promptfoo calls, so it is not forwarded as a header.
PORTKEY_API_BASE_URL_KEY = "portkeyApiBaseUrl"


def to_kebab_case(value: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1-\2", value).lower()


def _to_header_value(value: Any) -> str:
    return json_stringify(value) if isinstance(value, dict | list) else js_string(value)


def get_portkey_headers(
    config: dict[str, Any] | None = None, portkey_api_key: str | None = None
) -> dict[str, str]:
    """Request headers from `portkey*` config keys, with `config.headers` applied last."""
    config = config or {}
    custom_headers = config.get("headers") if isinstance(config.get("headers"), dict) else {}
    headers: dict[str, str] = {}

    def set_generated(name: str, value: Any) -> None:
        if value is not None and not has_header_override(custom_headers, name):
            headers[name] = _to_header_value(value)

    set_generated("x-portkey-api-key", portkey_api_key)
    for key, value in config.items():
        if key.startswith(PORTKEY_CONFIG_PREFIX) and key != PORTKEY_API_BASE_URL_KEY:
            set_generated(f"x-portkey-{to_kebab_case(key[len(PORTKEY_CONFIG_PREFIX) :])}", value)
    for key, value in (custom_headers or {}).items():
        if value is not None:
            headers[key] = _to_header_value(value)
    return headers


def _canonicalize_authorization(headers: dict[str, str]) -> dict[str, str]:
    return {
        ("Authorization" if key.lower() == "authorization" else key): value
        for key, value in headers.items()
    }


def _resolve_portkey_api_key(
    config: dict[str, Any] | None = None, env: dict[str, Any] | None = None
) -> str | None:
    config = config or {}
    if config.get("portkeyApiKey"):
        return config["portkeyApiKey"]
    resolved = resolve_provider_env(env, ["PORTKEY_API_KEY"])
    return resolved["value"] if resolved else None


def _uses_managed_credentials(model_name: str, config: dict[str, Any]) -> bool:
    provider = config.get("portkeyProvider")
    return bool(
        config.get("portkeyVirtualKey")
        or (isinstance(provider, str) and provider.startswith("@"))
        or model_name.startswith("@")
    )


class PortkeyChatCompletionProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        config = provider_options.get("config") or {}
        env = provider_options.get("env")
        portkey_api_key = _resolve_portkey_api_key(config, env)
        base_url_env = resolve_provider_env(env, ["PORTKEY_API_BASE_URL"])
        api_key_required = config.get("apiKeyRequired")
        if api_key_required is None:
            api_key_required = not (
                portkey_api_key or has_header_override(config.get("headers"), "x-portkey-api-key")
            )
        super().__init__(
            model_name,
            {
                **provider_options,
                "config": {
                    **config,
                    "apiKeyEnvar": "PORTKEY_API_KEY",
                    "apiKeyRequired": api_key_required,
                    "apiBaseUrl": config.get("portkeyApiBaseUrl")
                    or (base_url_env["value"] if base_url_env else None)
                    or "https://api.portkey.ai/v1",
                    "headers": get_portkey_headers(config, portkey_api_key),
                },
            },
        )

    def get_openai_request_headers(
        self,
        custom_headers: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
        *,
        use_config_headers: bool = True,
    ) -> dict[str, str]:
        """Re-derives the `x-portkey-*` headers for each request, so a per-prompt `headers`
        block does not drop the Portkey credential."""
        if custom_headers is None:
            custom_headers = self.config.get("headers")
        return _canonicalize_authorization(
            super().get_openai_request_headers(
                get_portkey_headers(
                    {**self.config, "headers": custom_headers},
                    _resolve_portkey_api_key(self.config, self.env),
                ),
                config,
            )
        )

    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        """The `Authorization` bearer Portkey forwards to the upstream provider."""
        if _uses_managed_credentials(self.model_name, self.config):
            return None
        if self.config.get("apiKey"):
            return self.config["apiKey"]
        resolved = resolve_provider_env(self.env, ["OPENAI_API_KEY"])
        return resolved["value"] if resolved else None

    def get_missing_api_key_error_message(self, config: dict[str, Any] | None = None) -> str:
        return (
            "Portkey API key is not set. Set the PORTKEY_API_KEY environment variable or add "
            "`portkeyApiKey` to the provider config."
        )


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `portkey:` registry factory."""
    model_name = ":".join(provider_path.split(":")[1:])
    return PortkeyChatCompletionProvider(model_name, options)
