"""Groq Chat Completions provider. Ports src/providers/groq/chat.ts, groq/util.ts and the
`groq:` factory in src/providers/registry.ts.

`groq:responses:<model>` subclasses the OpenAI Responses provider, which is imported lazily.
"""

from __future__ import annotations

import importlib
from typing import Any

from promptphoo import envars
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.compatible.common import serialize_provider
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider

GROQ_API_BASE_URL = "https://api.groq.com/openai/v1"

# Groq reasoning models: OpenAI GPT-OSS and current Qwen variants.
GROQ_REASONING_MODEL_PATTERNS = ("gpt-oss", "qwen")

GROQ_CHAT_SERVICE_TIERS = ("auto", "on_demand", "flex", "performance", None)
GROQ_RESPONSES_SERVICE_TIERS = ("auto", "default", "flex", None)

_MISSING = object()


def _assert_groq_service_tier(service_tier: Any, endpoint: str, supported: tuple) -> None:
    if service_tier is not _MISSING and service_tier not in supported:
        tiers = ", ".join(json_stringify(tier) for tier in supported)
        raise ValueError(
            f"Invalid Groq {endpoint} service_tier {json_stringify(service_tier)}. "
            f"Use one of: {tiers}."
        )


def assert_groq_chat_service_tier(service_tier: Any = _MISSING) -> None:
    _assert_groq_service_tier(service_tier, "Chat Completions", GROQ_CHAT_SERVICE_TIERS)


def assert_groq_responses_service_tier(service_tier: Any = _MISSING) -> None:
    _assert_groq_service_tier(service_tier, "Responses", GROQ_RESPONSES_SERVICE_TIERS)


def is_groq_reasoning_model(model_name: str) -> bool:
    return any(pattern in model_name for pattern in GROQ_REASONING_MODEL_PATTERNS)


def groq_supports_temperature(model_name: str) -> bool:
    return is_groq_reasoning_model(model_name)


def get_groq_provider_options(provider_options: dict[str, Any] | None) -> dict[str, Any]:
    provider_options = provider_options or {}
    config = provider_options.get("config") or {}
    return {
        **provider_options,
        "config": {
            **config,
            "apiKeyEnvar": config.get("apiKeyEnvar") or "GROQ_API_KEY",
            "apiBaseUrl": config.get("apiBaseUrl") or GROQ_API_BASE_URL,
        },
    }


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


class GroqProvider(OpenAiChatCompletionProvider):
    """`groq:<model>`: Chat Completions with Groq's reasoning parameters."""

    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        super().__init__(model_name, get_groq_provider_options(provider_options))

    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        api_key_envar = self.config.get("apiKeyEnvar") or "GROQ_API_KEY"
        return (
            self.config.get("apiKey")
            or envars.get_env_string(api_key_envar)
            or (self.env or {}).get(api_key_envar)
        )

    def is_reasoning_model(self, model_name: str | None = None) -> bool:
        model_name = model_name or self.model_name
        return is_groq_reasoning_model(model_name) or super().is_reasoning_model(model_name)

    def is_reasoning_capability_model(self, model_name: str) -> bool:
        return is_groq_reasoning_model(model_name) or super().is_reasoning_capability_model(
            model_name
        )

    def supports_temperature(self, model_name: str | None = None) -> bool:
        # Groq's reasoning models support temperature, unlike OpenAI's o1 models.
        if groq_supports_temperature(self.model_name):
            return True
        return super().supports_temperature()

    def supports_temperature_for_capability_model(self, model_name: str) -> bool:
        return (
            True
            if groq_supports_temperature(model_name)
            else super().supports_temperature_for_capability_model(model_name)
        )

    async def get_openai_body(self, prompt, context=None, call_api_options=None):
        result = await super().get_openai_body(prompt, context, call_api_options)
        body, config = result["body"], result["config"]
        if isinstance(body.get("model"), str) and self.is_reasoning_model(body["model"]):
            passthrough = config.get("passthrough") or {}
            max_completion_tokens = _first(
                passthrough.get("max_completion_tokens"),
                config.get("max_completion_tokens"),
                passthrough.get("max_tokens"),
                config.get("max_tokens"),
                body.get("max_tokens"),
            )
            if max_completion_tokens is not None:
                body["max_completion_tokens"] = max_completion_tokens
            body.pop("max_tokens", None)
        groq_config = self.config
        # Groq-specific reasoning parameters.
        if groq_config.get("reasoning_format") is not None:
            body["reasoning_format"] = groq_config["reasoning_format"]
        if groq_config.get("include_reasoning") is not None:
            body["include_reasoning"] = groq_config["include_reasoning"]
        # Compound model parameters.
        if groq_config.get("compound_custom"):
            body["compound_custom"] = groq_config["compound_custom"]
        if groq_config.get("search_settings"):
            body["search_settings"] = groq_config["search_settings"]
        assert_groq_chat_service_tier(body.get("service_tier", _MISSING))
        return {"body": body, "config": config}

    def default_id(self) -> str:
        return f"groq:{self.model_name}"

    def __str__(self) -> str:
        return f"[Groq Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "groq", lambda: self.config.get("apiKey"))


def _create_groq_responses_provider(model_name: str, options: dict[str, Any]) -> Any:
    try:
        responses = importlib.import_module("promptphoo.providers.openai.responses")
    except ImportError as err:
        raise NotImplementedError(
            f"The groq:responses provider (groq:responses:{model_name}) needs the OpenAI "
            "Responses API provider, which is not ported yet. Use groq:<model> for Chat "
            "Completions meanwhile."
        ) from err
    base: type = responses.OpenAiResponsesProvider

    class GroqResponsesProvider(base):  # type: ignore[valid-type, misc]
        """`groq:responses:<model>` (src/providers/groq/responses.ts)."""

        def __init__(self, model_name: str, provider_options: dict[str, Any]) -> None:
            super().__init__(model_name, get_groq_provider_options(provider_options))

        def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
            api_key_envar = self.config.get("apiKeyEnvar") or "GROQ_API_KEY"
            return (
                self.config.get("apiKey")
                or envars.get_env_string(api_key_envar)
                or (self.env or {}).get(api_key_envar)
            )

        def is_reasoning_model(self, model_name: str | None = None) -> bool:
            return is_groq_reasoning_model(self.model_name) or super().is_reasoning_model()

        def is_reasoning_capability_model(self, model_name: str) -> bool:
            return is_groq_reasoning_model(model_name) or super().is_reasoning_capability_model(
                model_name
            )

        def supports_temperature(self, model_name: str | None = None) -> bool:
            if groq_supports_temperature(self.model_name):
                return True
            return super().supports_temperature()

        def supports_temperature_for_capability_model(self, model_name: str) -> bool:
            return (
                True
                if groq_supports_temperature(model_name)
                else super().supports_temperature_for_capability_model(model_name)
            )

        async def get_openai_body(self, *args: Any, **kwargs: Any) -> Any:
            result = await super().get_openai_body(*args, **kwargs)
            assert_groq_responses_service_tier(result["body"].get("service_tier", _MISSING))
            return result

        def default_id(self) -> str:
            return f"groq:responses:{self.model_name}"

        def __str__(self) -> str:
            return f"[Groq Responses Provider {self.model_name}]"

        def to_json(self) -> dict[str, Any]:
            return serialize_provider(self, "groq:responses", lambda: self.config.get("apiKey"))

    return GroqResponsesProvider(model_name, options)


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `groq:` factory in src/providers/registry.ts."""
    # groq:responses:<model> targets the Responses API.
    if provider_path.startswith("groq:responses:"):
        model_name = provider_path[len("groq:responses:") :]
        if not model_name:
            raise ValueError(
                f'Invalid groq:responses provider path: "{provider_path}". '
                "Use format groq:responses:<model> (e.g., groq:responses:openai/gpt-oss-120b)"
            )
        return _create_groq_responses_provider(model_name, options)
    model_name = provider_path[len("groq:") :]
    if not model_name:
        raise ValueError(
            f'Invalid groq provider path: "{provider_path}". '
            "Use format groq:<model> (e.g., groq:openai/gpt-oss-120b)"
        )
    return GroqProvider(model_name, options)
