"""Moonshot AI (Kimi) chat provider. Ports src/providers/moonshot.ts and the `moonshot:`
factory in src/providers/registry.ts.
"""

from __future__ import annotations

import math
import re
from typing import Any

from promptphoo.providers.compatible.common import serialize_provider
from promptphoo.providers.credentials import resolve_configured_api_key
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.shared import clamp_cached_tokens
from promptphoo.templating import render_vars_in_object

MOONSHOT_API_BASE_URL = "https://api.moonshot.ai/v1"
MOONSHOT_API_KEY_ENVAR = "MOONSHOT_API_KEY"
DEFAULT_MOONSHOT_MODEL = "kimi-k3"


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def pins_sampling_params(model_name: str) -> bool:
    """Kimi thinking models pin temperature, top_p, n and the penalties to fixed values."""
    return re.match(r"^kimi-", model_name, re.IGNORECASE) is not None


def calculate_moonshot_cost(
    config: dict[str, Any],
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    cached_tokens: Any = None,
) -> float | None:
    """Cost from user-supplied rates only; Moonshot has no built-in price table."""
    if not _is_finite(prompt_tokens) or not _is_finite(completion_tokens):
        return None
    input_cost = _first(config.get("inputCost"), config.get("cost"))
    output_cost = _first(config.get("outputCost"), config.get("cost"))
    if input_cost is None and output_cost is None:
        return None
    billable_cached = clamp_cached_tokens(cached_tokens, prompt_tokens)
    uncached_prompt_tokens = prompt_tokens - billable_cached
    cache_read_cost = _first(config.get("cacheReadCost"), input_cost)
    return (
        (input_cost or 0) * uncached_prompt_tokens
        + (cache_read_cost or 0) * billable_cached
        + (output_cost or 0) * completion_tokens
    )


class MoonshotProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        moonshot_config = provider_options.get("config") or {}
        resolved = {
            **moonshot_config,
            "apiKeyEnvar": _first(moonshot_config.get("apiKeyEnvar"), MOONSHOT_API_KEY_ENVAR),
            # Users may point at the China-mainland endpoint (https://api.moonshot.cn/v1).
            "apiBaseUrl": _first(moonshot_config.get("apiBaseUrl"), MOONSHOT_API_BASE_URL),
        }
        super().__init__(model_name, {**provider_options, "config": resolved})

    # Moonshot config/envar only; never OPENAI_API_KEY.
    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        return resolve_configured_api_key(self)

    def get_api_url(self, config: dict[str, Any] | None = None) -> str:
        return _first(self.config.get("apiBaseUrl"), MOONSHOT_API_BASE_URL)

    def get_organization(self, config: dict[str, Any] | None = None) -> str | None:
        return None

    def get_missing_api_key_error_message(self, config: dict[str, Any] | None = None) -> str:
        envar = _first(self.config.get("apiKeyEnvar"), MOONSHOT_API_KEY_ENVAR)
        return (
            f"Moonshot API key is not set. Set the {envar} "
            "environment variable or add `apiKey` to the provider config."
        )

    def default_id(self) -> str:
        return f"moonshot:{self.model_name}"

    def __str__(self) -> str:
        return f"[Moonshot Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "moonshot")

    async def get_openai_body(self, prompt, context=None, call_api_options=None):
        result = await super().get_openai_body(prompt, context, call_api_options)
        body, config = result["body"], result["config"]
        model_name = body["model"] if isinstance(body.get("model"), str) else self.model_name

        # reasoning_effort is K3-only.
        if config.get("reasoning_effort") is not None:
            if not re.match(r"^kimi-k3", model_name, re.IGNORECASE):
                raise ValueError(
                    f"Moonshot model {model_name} does not support reasoning_effort (kimi-k3 "
                    "family only). Remove it, use `passthrough: { thinking: ... }` on K2.x, or "
                    "force-send it via `passthrough: { reasoning_effort: ... }`."
                )
            body["reasoning_effort"] = render_vars_in_object(
                config["reasoning_effort"], (context or {}).get("vars")
            )

        # max_completion_tokens is canonical; prompt-level config beats provider-level.
        prompt_config = ((context or {}).get("prompt") or {}).get("config") or {}
        prompt_passthrough = prompt_config.get("passthrough") or {}
        passthrough = config.get("passthrough") or {}
        max_tokens = _first(
            prompt_passthrough.get("max_completion_tokens"),
            prompt_passthrough.get("max_tokens"),
            prompt_config.get("max_completion_tokens"),
            prompt_config.get("max_tokens"),
            passthrough.get("max_completion_tokens"),
            passthrough.get("max_tokens"),
            config.get("max_completion_tokens"),
            config.get("max_tokens"),
        )
        if max_tokens is not None or pins_sampling_params(model_name):
            body.pop("max_tokens", None)
            if max_tokens is None:
                body.pop("max_completion_tokens", None)
            else:
                body["max_completion_tokens"] = max_tokens

        if not pins_sampling_params(model_name):
            return result
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if config.get(key) is None and passthrough.get(key) is None:
                body.pop(key, None)
        return result

    def calculate_response_cost(
        self, data: Any, config: dict[str, Any], cached: bool
    ) -> float | None:
        if cached:
            return None
        usage = (data or {}).get("usage") if isinstance(data, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        cached_tokens = _first(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            usage.get("cached_tokens"),
        )
        return calculate_moonshot_cost(
            config,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            cached_tokens if _is_finite(cached_tokens) else 0,
        )


def create_moonshot_provider(provider_path: str, options: dict[str, Any] | None = None) -> Any:
    # `moonshot:<model>` and `moonshot:chat:<model>`.
    rest = provider_path.split(":")[1:]
    if rest and rest[0] == "chat":
        rest.pop(0)
    model_name = ":".join(rest) or DEFAULT_MOONSHOT_MODEL
    return MoonshotProvider(model_name, options or {})


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `moonshot:` factory in src/providers/registry.ts (`env: providerOptions.env ??
    context.env`; both are the merged env here)."""
    return create_moonshot_provider(provider_path, options)
