"""Abliteration chat provider. Ports src/providers/abliteration.ts and the `abliteration:` entry
of src/providers/registry.ts.
"""

from __future__ import annotations

from typing import Any

from promptphoo import envars
from promptphoo.providers.credentials import resolve_provider_env
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.templating import render_vars_in_object

ABLITERATION_API_BASE_URL = "https://api.abliteration.ai/v1"
ABLITERATION_API_BASE_URL_ENV_VAR = "ABLIT_API_BASE_URL"


def normalize_api_base_url(api_base_url: Any) -> str | None:
    trimmed = api_base_url.strip() if isinstance(api_base_url, str) else None
    return trimmed or None


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


class AbliterationProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        config = options.get("config") or {}
        resolved = resolve_provider_env(options.get("env"), [ABLITERATION_API_BASE_URL_ENV_VAR])
        super().__init__(
            model_name,
            {
                **options,
                "config": {
                    **config,
                    "apiBaseUrl": _first(
                        normalize_api_base_url(config.get("apiBaseUrl")),
                        normalize_api_base_url(resolved["value"] if resolved else None),
                        ABLITERATION_API_BASE_URL,
                    ),
                    "apiKeyEnvar": _first(config.get("apiKeyEnvar"), "ABLIT_KEY"),
                    "showThinking": _first(config.get("showThinking"), False),
                },
            },
        )

    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        api_key_envar = self.config.get("apiKeyEnvar")
        return self.config.get("apiKey") or (
            (
                envars.get_provider_env_string(self.env, api_key_envar)
                or envars.get_env_string(api_key_envar)
            )
            if api_key_envar
            else None
        )

    def get_organization(self, config: dict[str, Any] | None = None) -> str | None:
        return None

    async def get_openai_body(self, prompt, context=None, call_api_options=None) -> dict[str, Any]:
        result = await super().get_openai_body(prompt, context, call_api_options)
        body, config = result["body"], result["config"]
        # All Abliteration models take reasoning_effort while keeping max_tokens.
        if "reasoning_effort" not in (config.get("passthrough") or {}):
            reasoning_effort = render_vars_in_object(
                config.get("reasoning_effort"), (context or {}).get("vars")
            )
            if reasoning_effort:
                body["reasoning_effort"] = reasoning_effort
        return {"body": body, "config": config}

    def default_id(self) -> str:
        return f"abliteration:{self.model_name}"

    def __str__(self) -> str:
        return f"[Abliteration Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        config = {k: v for k, v in self.config.items() if k != "apiKey"}
        return {"provider": "abliteration", "model": self.model_name, "config": config}


def create_abliteration_provider(
    provider_path: str, options: dict[str, Any] | None = None
) -> AbliterationProvider:
    options = options or {}
    splits = provider_path.split(":")
    model_name = ":".join(splits[2:] if len(splits) > 1 and splits[1] == "chat" else splits[1:])
    if not model_name:
        raise ValueError(
            "Abliteration provider requires a model name. Use format: abliteration:<model_name> "
            "or abliteration:chat:<model_name>"
        )
    provider_options = _first(options.get("providerOptions"), options.get("config"), {})
    out = {**provider_options, "env": _first(provider_options.get("env"), options.get("env"))}
    provider_id = _first(options.get("id"), provider_options.get("id"))
    if provider_id is not None:
        out["id"] = provider_id
    return AbliterationProvider(model_name, out)


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `abliteration:` registry entry: `{providerOptions, env: context.env}`."""
    return create_abliteration_provider(
        provider_path, {"providerOptions": options, "env": options.get("env")}
    )
