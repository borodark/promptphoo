"""OrcaRouter, an OpenAI-compatible meta-router (https://www.orcarouter.ai). Ports
src/providers/orcarouter.ts and the `orcarouter:` factory in src/providers/registry.ts.
"""

from __future__ import annotations

import re
from typing import Any

from promptphoo import envars
from promptphoo.providers.compatible.common import (
    configured_provider_options,
    serialize_provider,
)
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.templating import render_vars_in_object

DEFAULT_ORCAROUTER_API_BASE_URL = "https://api.orcarouter.ai/v1"


def reasoning_upstream_rejects_temperature(model_name: str) -> bool:
    """Claude Opus 4+ and DeepSeek Reasoner / r<N> reject the OpenAI-style `temperature`."""
    if re.search(r"(^|/)claude-opus-([4-9]|\d{2,})", model_name):
        return True
    return re.search(r"(^|/)deepseek-(reasoner|r\d+)\b", model_name) is not None


class OrcaRouterProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        config = provider_options.get("config") or {}
        passthrough: dict[str, Any] = {}
        for key in ("models", "route"):
            if config.get(key):
                passthrough[key] = config[key]
        passthrough.update(config.get("passthrough") or {})
        super().__init__(
            model_name,
            {
                **provider_options,
                "config": {
                    **config,
                    "apiBaseUrl": config.get("apiBaseUrl") or DEFAULT_ORCAROUTER_API_BASE_URL,
                    "apiKeyEnvar": config.get("apiKeyEnvar") or "ORCAROUTER_API_KEY",
                    "headers": {
                        "HTTP-Referer": "https://promptfoo.dev/",
                        "X-Title": "promptfoo",
                        **(config.get("headers") or {}),
                    },
                    "passthrough": passthrough,
                },
            },
        )

    def default_id(self) -> str:
        return f"orcarouter:{self.model_name}"

    def __str__(self) -> str:
        return f"[OrcaRouter Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "orcarouter")

    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        """No `OPENAI_API_KEY` fallback: an OpenAI key must not be sent to OrcaRouter."""
        envar = self.config.get("apiKeyEnvar") or "ORCAROUTER_API_KEY"
        return (
            self.config.get("apiKey")
            or envars.get_provider_env_string(self.env, envar)
            or envars.get_env_string(envar)
        )

    def get_organization(self, config: dict[str, Any] | None = None) -> str | None:
        return None

    def get_missing_api_key_error_message(self, config: dict[str, Any] | None = None) -> str:
        envar = self.config.get("apiKeyEnvar") or "ORCAROUTER_API_KEY"
        return (
            f"API key is not set. Set the {envar} environment variable or add `apiKey` to the "
            "provider config."
        )

    def get_api_url(self, config: dict[str, Any] | None = None) -> str:
        """Ignores OPENAI_API_HOST / OPENAI_API_BASE_URL / OPENAI_BASE_URL."""
        return self.config.get("apiBaseUrl") or DEFAULT_ORCAROUTER_API_BASE_URL

    def get_gen_ai_system(self) -> str:
        return "orcarouter"

    async def get_openai_body(self, prompt, context=None, call_api_options=None):
        result = await super().get_openai_body(prompt, context, call_api_options)
        context = context or {}
        vars = context.get("vars")
        prompt_config = (context.get("prompt") or {}).get("config") or {}
        config, body = result["config"], result["body"]
        passthrough = config.get("passthrough") or {}

        # prompt.config wins over provider config, with passthrough as the final fallback.
        models = next(
            (v for v in (prompt_config.get("models"), config.get("models"),
                         passthrough.get("models")) if v is not None),
            None,
        )  # fmt: skip
        if models is not None:
            body["models"] = render_vars_in_object(models, vars)
        route = next(
            (v for v in (prompt_config.get("route"), config.get("route"),
                         passthrough.get("route")) if v is not None),
            None,
        )  # fmt: skip
        if route is not None:
            body["route"] = render_vars_in_object(route, vars)
        # The base builder can overwrite reasoning_effort with the raw config value.
        if config.get("reasoning_effort") is not None:
            body["reasoning_effort"] = render_vars_in_object(config["reasoning_effort"], vars)

        any_fallback_rejects_temperature = isinstance(body.get("models"), list) and any(
            isinstance(m, str) and reasoning_upstream_rejects_temperature(m) for m in body["models"]
        )
        if "temperature" in body and (
            not self.supports_temperature() or any_fallback_rejects_temperature
        ):
            del body["temperature"]
        return result

    def supports_temperature(self, model_name: str | None = None) -> bool:
        """Done here (not in `is_reasoning_model`) so `max_tokens` is kept."""
        if not super().supports_temperature():
            return False
        return not reasoning_upstream_rejects_temperature(self.model_name)


def create_orcarouter_provider(
    provider_path: str, options: dict[str, Any] | None = None
) -> OrcaRouterProvider:
    options = options or {}
    model_name = ":".join(provider_path.split(":")[1:])
    provider_options: dict[str, Any] = dict(options.get("config") or {})
    if options.get("env") and not provider_options.get("env"):
        provider_options["env"] = options["env"]
    if options.get("id") and not provider_options.get("id"):
        provider_options["id"] = options["id"]
    return OrcaRouterProvider(model_name, provider_options)


def create_provider(provider_path: str, options: dict[str, Any]) -> OrcaRouterProvider:
    """The `orcarouter:` registry factory (configuredProviderFactory)."""
    return create_orcarouter_provider(provider_path, configured_provider_options(options))
