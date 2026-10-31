"""xAI (Grok) chat provider. Ports src/providers/xai/chat.ts and the `xai:` factory in
src/providers/registry.ts.

`xai:responses:`, `xai:image:`, `xai:video:` and `xai:voice:` need the Responses API, image,
video and voice providers, which promptphoo does not port; they fail with a clear error.
"""

from __future__ import annotations

import logging
import math
from typing import Any
from urllib.parse import urlsplit

from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.compatible.common import (
    configured_provider_options,
    not_supported,
    serialize_provider,
)
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.shared import (
    clamp_cached_tokens,
    get_openai_chat_output_limit_from_env,
    resolve_direct_test_variable,
)
from promptphoo.templating import render_vars_in_object

logger = logging.getLogger(__name__)

_MISSING = object()


def assert_xai_service_tier(service_tier: Any) -> None:
    if service_tier is not None and service_tier not in ("default", "priority"):
        raise ValueError('Invalid xAI service_tier. Use "default" or "priority".')


def _per_m(value: float) -> float:
    return value / 1e6


# Pricing is sourced from https://docs.x.ai/developers/pricing. Response billing uses the same
# $1e-10 scale in `usage.cost_in_usd_ticks`.
_GROK_43_AND_420_LONG_CONTEXT_COST = {
    "threshold": 200_000,
    "input": 2.5 / 1e6,
    "output": 5 / 1e6,
    "cache_read": 0.4 / 1e6,
}


def _models_with_cost(models: list[dict[str, Any]], cost: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for model in models:
        model_cost = dict(cost)
        if cost.get("longContext"):
            model_cost["longContext"] = dict(cost["longContext"])
        entry: dict[str, Any] = {"id": model["id"], "cost": model_cost}
        if model.get("aliases") is not None:
            entry["aliases"] = model["aliases"]
        out.append(entry)
    return out


XAI_CHAT_MODELS: list[dict[str, Any]] = [
    *_models_with_cost(
        [{"id": "grok-4.7"}, {"id": "grok-4.6"}],
        {
            "input": 2.0 / 1e6,
            "output": 6.0 / 1e6,
            "cache_read": 0.5 / 1e6,
            "longContext": {
                "threshold": 200_000,
                "input": 4.0 / 1e6,
                "output": 12.0 / 1e6,
                "cache_read": 1.0 / 1e6,
            },
        },
    ),
    {
        "id": "grok-4.5",
        "cost": {
            "input": 2.0 / 1e6,
            "output": 6.0 / 1e6,
            "cache_read": 0.3 / 1e6,
            "longContext": {
                "threshold": 200_000,
                "input": 4.0 / 1e6,
                "output": 12.0 / 1e6,
                "cache_read": 0.6 / 1e6,
            },
        },
        "aliases": ["grok-4.5-latest", "grok-build-latest"],
    },
    *_models_with_cost(
        [
            {
                "id": "grok-4.20-0309-reasoning",
                "aliases": [
                    "grok-4.20", "grok-4.20-reasoning", "grok-4.20-reasoning-latest",
                    "grok-4.20-0309", "grok-4.20-beta", "grok-4.20-beta-0309",
                    "grok-4.20-beta-0309-reasoning", "grok-4.20-beta-latest",
                    "grok-4.20-beta-latest-reasoning", "grok-4.20-beta-reasoning",
                    "grok-4.20-experimental-beta-0304",
                    "grok-4.20-experimental-beta-0304-reasoning",
                    "grok-4.20-experimental-beta-latest",
                    "grok-4.20-experimental-beta-reasoning-latest", "grok-4.20-reasoning-gv2",
                ],
            },
            {
                "id": "grok-4.20-0309-non-reasoning",
                "aliases": [
                    "grok-4.20-non-reasoning", "grok-4.20-non-reasoning-latest",
                    "grok-4.20-beta-non-reasoning", "grok-4.20-beta-latest-non-reasoning",
                    "grok-4.20-beta-0309-non-reasoning",
                    "grok-4.20-experimental-beta-0304-non-reasoning",
                    "grok-4.20-experimental-beta-non-reasoning-latest",
                    "grok-4.20-non-reasoning-gv2",
                ],
            },
            {
                "id": "grok-4.20-multi-agent-0309",
                "aliases": [
                    "grok-4.20-multi-agent", "grok-4.20-multi-agent-latest",
                    "grok-4.20-multi-agent-beta-latest", "grok-4.20-multi-agent-beta-0309",
                    "grok-4.20-multi-agent-experimental-beta-0304",
                    "grok-4.20-multi-agent-experimental-beta-latest",
                ],
            },
            {"id": "grok-4.3", "aliases": ["grok-4.3-latest", "grok-latest"]},
        ],
        {
            "input": 1.25 / 1e6,
            "output": 2.5 / 1e6,
            "cache_read": 0.2 / 1e6,
            "longContext": _GROK_43_AND_420_LONG_CONTEXT_COST,
        },
    ),
    *_models_with_cost(
        [
            {
                "id": "grok-4-1-fast-reasoning",
                "aliases": [
                    "grok-4-1-fast", "grok-4-1-fast-latest", "grok-4-1-fast-reasoning-latest",
                ],
            },
            {
                "id": "grok-4-1-fast-non-reasoning",
                "aliases": ["grok-4-1-fast-non-reasoning-latest"],
            },
        ],
        {"input": 0.2 / 1e6, "output": 0.5 / 1e6, "cache_read": 0.05 / 1e6},
    ),
    {
        "id": "grok-build-0.1",
        "cost": {
            "input": 1.0 / 1e6,
            "output": 2.0 / 1e6,
            "cache_read": 0.2 / 1e6,
            "longContext": {
                "threshold": 200_000,
                "input": 2.0 / 1e6,
                "output": 4.0 / 1e6,
                "cache_read": 0.4 / 1e6,
            },
        },
        "aliases": ["grok-code-fast-1", "grok-code-fast", "grok-code-fast-1-0825"],
    },
    *_models_with_cost(
        [
            {
                "id": "grok-4-fast-reasoning",
                "aliases": ["grok-4-fast", "grok-4-fast-latest", "grok-4-fast-reasoning-latest"],
            },
            {
                "id": "grok-4-fast-non-reasoning",
                "aliases": ["grok-4-fast-non-reasoning-latest"],
            },
        ],
        {"input": 0.2 / 1e6, "output": 0.5 / 1e6, "cache_read": 0.05 / 1e6},
    ),
    *_models_with_cost(
        [
            {"id": "grok-4-0709", "aliases": ["grok-4", "grok-4-latest"]},
            {"id": "grok-3-beta", "aliases": ["grok-3", "grok-3-latest"]},
            {"id": "grok-3-fast-beta", "aliases": ["grok-3-fast", "grok-3-fast-latest"]},
        ],
        {"input": 3.0 / 1e6, "output": 15.0 / 1e6, "cache_read": 0.75 / 1e6},
    ),
    *_models_with_cost(
        [
            {"id": "grok-3-mini-beta", "aliases": ["grok-3-mini", "grok-3-mini-latest"]},
            {
                "id": "grok-3-mini-fast-beta",
                "aliases": ["grok-3-mini-fast", "grok-3-mini-fast-latest"],
            },
        ],
        {"input": 0.3 / 1e6, "output": 0.5 / 1e6, "cache_read": 0.075 / 1e6},
    ),
    *_models_with_cost(
        [
            {"id": "grok-2-1212", "aliases": ["grok-2", "grok-2-latest"]},
            {"id": "grok-2-vision-1212", "aliases": ["grok-2-vision", "grok-2-vision-latest"]},
        ],
        {"input": 2.0 / 1e6, "output": 10.0 / 1e6},
    ),
    *_models_with_cost(
        [{"id": "grok-beta"}, {"id": "grok-vision-beta"}],
        {"input": 5.0 / 1e6, "output": 15.0 / 1e6},
    ),
]  # fmt: skip

# Legacy chat slugs xAI redirects to grok-4.3 and bills at grok-4.3 pricing.
_GROK_43_REDIRECTED_CHAT_MODELS = frozenset([
    "grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning", "grok-4-1-fast",
    "grok-4-1-fast-latest", "grok-4-1-fast-reasoning-latest",
    "grok-4-1-fast-non-reasoning-latest",
    "grok-4-fast-reasoning", "grok-4-fast-non-reasoning", "grok-4-fast", "grok-4-fast-latest",
    "grok-4-fast-reasoning-latest", "grok-4-fast-non-reasoning-latest",
    "grok-4-0709", "grok-4", "grok-4-latest",
    "grok-3", "grok-3-latest", "grok-3-beta", "grok-3-fast", "grok-3-fast-latest",
    "grok-3-fast-beta",
])  # fmt: skip

GROK_3_MINI_MODELS = [
    "grok-3-mini-beta", "grok-3-mini", "grok-3-mini-latest", "grok-3-mini-fast-beta",
    "grok-3-mini-fast", "grok-3-mini-fast-latest",
]  # fmt: skip

# Models that support reasoning_effort on the chat-completions-compatible API.
GROK_REASONING_EFFORT_MODELS = [
    "grok-4.7", "grok-4.6", "grok-4.5", "grok-4.5-latest", "grok-build-latest", "grok-4.3",
    "grok-4.3-latest", "grok-latest", "grok-3-mini-beta", "grok-3-mini", "grok-3-mini-latest",
    "grok-3-mini-fast-beta", "grok-3-mini-fast", "grok-3-mini-fast-latest",
]  # fmt: skip

# Grok 4.5 and newer models cannot disable reasoning.
GROK_45_MODELS = frozenset(
    ["grok-4.7", "grok-4.6", "grok-4.5", "grok-4.5-latest", "grok-build-latest"]
)


class XAIRequestConfigError(Exception):
    pass


def validate_xai_reasoning_effort(model_name: str, effort: Any, parameter: str) -> None:
    supports_none = model_name in ("grok-4.3", "grok-4.3-latest", "grok-latest")
    if (model_name not in GROK_45_MODELS and not supports_none) or effort is None:
        return
    supports_xhigh = model_name in ("grok-4.7", "grok-4.6")
    if supports_xhigh:
        allowed = ["low", "medium", "high", "xhigh"]
        choices = '"low", "medium", "high", or "xhigh"'
    elif supports_none:
        allowed = ["none", "low", "medium", "high"]
        choices = '"none", "low", "medium", or "high"'
    else:
        allowed = ["low", "medium", "high"]
        choices = '"low", "medium", or "high"'
    if not isinstance(effort, str) or effort not in allowed:
        guidance = (
            f"Use {choices}, or omit {parameter}."
            if supports_none
            else f'Use {choices}, or omit {parameter} to use the default "high".'
        )
        raise XAIRequestConfigError(
            f"xAI model {model_name} does not support {parameter} with the supplied value. "
            f"{guidance}"
        )


def resolve_grok47_reasoning_effort(value: Any, vars: dict[str, Any] | None = None) -> str | None:
    if value is None:
        return None
    value = resolve_direct_test_variable(value, vars)
    if isinstance(value, str) and value in ("none", "low", "medium", "high", "xhigh"):
        return value
    raise XAIRequestConfigError(
        "xAI Grok 4.7 reasoning effort must be a supported value or a simple test variable"
    )


# All reasoning models, including older families that reason without a tunable effort knob.
GROK_REASONING_MODELS = [
    "grok-4.7", "grok-4.6", "grok-4.5", "grok-4.5-latest", "grok-build-latest",
    "grok-4.20-0309-reasoning", "grok-4.20-reasoning", "grok-4.20", "grok-4.20-reasoning-latest",
    "grok-4.20-0309", "grok-4.20-beta", "grok-4.20-beta-0309", "grok-4.20-beta-0309-reasoning",
    "grok-4.20-beta-latest", "grok-4.20-beta-latest-reasoning", "grok-4.20-beta-reasoning",
    "grok-4.20-experimental-beta-0304", "grok-4.20-experimental-beta-0304-reasoning",
    "grok-4.20-experimental-beta-latest", "grok-4.20-experimental-beta-reasoning-latest",
    "grok-4.20-reasoning-gv2", "grok-4.20-multi-agent-0309", "grok-4.20-multi-agent",
    "grok-4.20-multi-agent-latest", "grok-4.20-multi-agent-beta-latest",
    "grok-4.20-multi-agent-beta-0309", "grok-4.20-multi-agent-experimental-beta-0304",
    "grok-4.20-multi-agent-experimental-beta-latest",
    "grok-4.3", "grok-4.3-latest", "grok-latest",
    "grok-4-1-fast-reasoning", "grok-4-1-fast", "grok-4-1-fast-latest",
    "grok-4-1-fast-reasoning-latest", "grok-build-0.1", "grok-code-fast-1", "grok-code-fast",
    "grok-code-fast-1-0825",
    "grok-4-fast-reasoning", "grok-4-fast", "grok-4-fast-latest", "grok-4-fast-reasoning-latest",
    "grok-4-0709", "grok-4", "grok-4-latest",
    "grok-3-mini-beta", "grok-3-mini", "grok-3-mini-latest", "grok-3-mini-fast-beta",
    "grok-3-mini-fast", "grok-3-mini-fast-latest",
]  # fmt: skip

_GROK_BUILD_CHAT_MODELS = frozenset(
    ["grok-build-0.1", "grok-code-fast-1", "grok-code-fast", "grok-code-fast-1-0825"]
)

# Grok-4+ models that reject presence_penalty, frequency_penalty and stop.
GROK_4_MODELS = [
    "grok-4.7", "grok-4.6", "grok-4.5", "grok-4.5-latest", "grok-build-latest",
    "grok-4.20-0309-reasoning", "grok-4.20-reasoning", "grok-4.20", "grok-4.20-reasoning-latest",
    "grok-4.20-0309", "grok-4.20-beta", "grok-4.20-beta-0309", "grok-4.20-beta-0309-reasoning",
    "grok-4.20-beta-latest", "grok-4.20-beta-latest-reasoning", "grok-4.20-beta-reasoning",
    "grok-4.20-experimental-beta-0304", "grok-4.20-experimental-beta-0304-reasoning",
    "grok-4.20-experimental-beta-latest", "grok-4.20-experimental-beta-reasoning-latest",
    "grok-4.20-reasoning-gv2", "grok-4.20-0309-non-reasoning", "grok-4.20-non-reasoning",
    "grok-4.20-non-reasoning-latest", "grok-4.20-beta-non-reasoning",
    "grok-4.20-beta-latest-non-reasoning", "grok-4.20-beta-0309-non-reasoning",
    "grok-4.20-experimental-beta-0304-non-reasoning",
    "grok-4.20-experimental-beta-non-reasoning-latest", "grok-4.20-non-reasoning-gv2",
    "grok-4.20-multi-agent-0309", "grok-4.20-multi-agent", "grok-4.20-multi-agent-latest",
    "grok-4.20-multi-agent-beta-latest", "grok-4.20-multi-agent-beta-0309",
    "grok-4.20-multi-agent-experimental-beta-0304",
    "grok-4.20-multi-agent-experimental-beta-latest",
    "grok-4.3", "grok-4.3-latest", "grok-latest",
    "grok-4-1-fast-reasoning", "grok-4-1-fast", "grok-4-1-fast-latest",
    "grok-4-1-fast-reasoning-latest", "grok-4-1-fast-non-reasoning",
    "grok-4-1-fast-non-reasoning-latest",
    "grok-build-0.1", "grok-code-fast-1", "grok-code-fast", "grok-code-fast-1-0825",
    "grok-4-fast-reasoning", "grok-4-fast", "grok-4-fast-latest", "grok-4-fast-reasoning-latest",
    "grok-4-fast-non-reasoning", "grok-4-fast-non-reasoning-latest",
    "grok-4-0709", "grok-4", "grok-4-latest",
]  # fmt: skip


def _url_origin(url: str) -> str | None:
    """WHATWG `new URL(url).origin`, or None when the URL does not parse."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    scheme = parts.scheme.lower()
    default_port = {"http": 80, "https": 443}.get(scheme)
    host = parts.hostname.lower()
    return f"{scheme}://{host}" + (f":{port}" if port and port != default_port else "")


def calculate_xai_cost(
    model_name: str,
    config: dict[str, Any],
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    reasoning_tokens: Any = None,
    cached_tokens: Any = None,
    options: dict[str, Any] | None = None,
) -> float | None:
    """`options`: `apiUrl`, `reasoningBilledSeparately` (chat completions report reasoning on
    top of completion tokens) and `serviceTier`."""
    options = options or {}
    config = config or {}
    completion = completion_tokens if completion_tokens is not None else 0
    reasoning = reasoning_tokens if reasoning_tokens is not None else 0
    if options.get("reasoningBilledSeparately"):
        billable_output = completion + reasoning
    else:
        billable_output = completion if completion > 0 else reasoning
    if (
        prompt_tokens is None
        or billable_output < 0
        or (prompt_tokens <= 0 and billable_output == 0)
    ):
        return None

    if model_name in _GROK_43_REDIRECTED_CHAT_MODELS:
        model = next((m for m in XAI_CHAT_MODELS if m["id"] == "grok-4.3"), None)
    else:
        model = next(
            (m for m in XAI_CHAT_MODELS
             if m["id"] == model_name or model_name in (m.get("aliases") or [])),
            None,
        )  # fmt: skip
    input_cost_override = _first(config.get("inputCost"), config.get("cost"))
    model_cost = (model or {}).get("cost")
    long_context = (model_cost or {}).get("longContext")
    if long_context and prompt_tokens >= long_context["threshold"]:
        model_cost = long_context
    api_url = options.get("apiUrl")
    region_multiplier = (
        1.1
        if (model or {}).get("id") in ("grok-4.7", "grok-4.6")
        and api_url
        and _url_origin(api_url) == "https://us.api.x.ai"
        else 1
    )
    catalog_multiplier = region_multiplier * (2 if options.get("serviceTier") == "priority" else 1)
    input_cost = _first(
        input_cost_override, model_cost and model_cost["input"] * catalog_multiplier
    )
    output_cost = _first(
        config.get("outputCost"),
        config.get("cost"),
        model_cost and model_cost["output"] * catalog_multiplier,
    )
    cache_read_cost = _first(
        config.get("cacheReadCost"),
        input_cost_override,
        input_cost
        if (model_cost or {}).get("cache_read") is None
        else model_cost["cache_read"] * catalog_multiplier,  # type: ignore[index]
    )

    billable_cached = clamp_cached_tokens(cached_tokens, prompt_tokens)
    uncached_prompt = prompt_tokens - billable_cached
    if (
        (uncached_prompt > 0 and input_cost is None)
        or (billable_cached > 0 and cache_read_cost is None)
        or (billable_output > 0 and output_cost is None)
    ):
        return None
    input_total = (input_cost or 0) * uncached_prompt + (cache_read_cost or 0) * billable_cached
    output_total = (output_cost or 0) * billable_output
    logger.debug(
        "XAI cost calculation for %s: promptTokens=%s, completionTokens=%s, "
        "reasoningTokens=%s, inputCost=%s, outputCost=%s",
        model_name,
        prompt_tokens,
        completion_tokens,
        reasoning_tokens or "N/A",
        input_total,
        output_total,
    )
    return input_total + output_total


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def get_xai_cost_in_usd(usage: Any) -> float | None:
    ticks = usage.get("cost_in_usd_ticks") if isinstance(usage, dict) else None
    if not isinstance(ticks, int | float) or isinstance(ticks, bool):
        return None
    if not math.isfinite(ticks) or ticks < 0:
        return None
    return ticks / 1e10


def has_xai_cost_overrides(config: dict[str, Any] | None) -> bool:
    config = config or {}
    return any(
        config.get(key) is not None for key in ("cost", "inputCost", "outputCost", "cacheReadCost")
    )


def get_xai_request_model(model_name: str, config: dict[str, Any] | None = None) -> str:
    passthrough = (config or {}).get("passthrough")
    model = passthrough.get("model") if isinstance(passthrough, dict) else None
    return model if isinstance(model, str) else model_name


def get_xai_request_option(key: str | list[str] | tuple[str, ...], *scopes: Any) -> Any:
    """The first scope (test options, prompt config, provider config) that sets `key`;
    `passthrough` beats the top level within a scope. Returns `_MISSING` when no scope sets it
    (JS undefined); a stored None is JS null."""
    keys = [key] if isinstance(key, str) else list(key)
    passthrough_replaced = False
    for index, scope in enumerate(scopes):
        config = scope if isinstance(scope, dict) else None
        raw = None
        if not passthrough_replaced and config is not None:
            candidate = config.get("passthrough")
            raw = candidate if isinstance(candidate, dict) else None
        if (
            index == 0
            and raw
            and any(name in raw for name in keys)
            and any(config is not None and name in config for name in keys)
        ):
            raise XAIRequestConfigError(
                f"xAI received {'/'.join(keys)} in both test options and test passthrough; "
                "use one location"
            )
        if config is not None and "passthrough" in config:
            passthrough_replaced = True
        for source in (raw, config):
            if source is None:
                continue
            available = [name for name in keys if name in source]
            if (
                len(available) == 2
                and all(source[name] is not None for name in available)
                and source[available[0]] != source[available[1]]
            ):
                raise XAIRequestConfigError(
                    "xAI Grok 4.7 received conflicting max_tokens and max_completion_tokens; "
                    "use one spelling"
                )
            for name in available:
                if len(keys) == 1 or source[name] is not None:
                    return source[name]
    return _MISSING


def _defined(value: Any) -> Any:
    return None if value is _MISSING else value


_XAI_TIP = (
    "Tip: Ensure your XAI_API_KEY environment variable is set correctly. You can get an API key "
    "from https://x.ai/"
)


class XAIProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        outer = provider_options.get("config") or {}
        xai_config = outer.get("config") if isinstance(outer.get("config"), dict) else None
        api_base_url = (xai_config or {}).get("apiBaseUrl")
        if api_base_url is None:
            region = (xai_config or {}).get("region")
            api_base_url = f"https://{region}.api.x.ai/v1" if region else "https://api.x.ai/v1"
        super().__init__(
            model_name,
            {
                **provider_options,
                "config": {
                    **outer,
                    **(xai_config or {}),
                    "apiKeyEnvar": "XAI_API_KEY",
                    "apiBaseUrl": api_base_url,
                },
            },
        )
        self.original_config = xai_config

    @property
    def api_key(self) -> Any:
        return (self.config or {}).get("apiKey")

    def is_reasoning_model(self, model_name: str | None = None) -> bool:
        return self.model_name in GROK_REASONING_MODELS

    def supports_reasoning_effort(self) -> bool:
        # Redirected legacy aliases still reject reasoning_effort; target grok-4.3 to tune it.
        return self.model_name in GROK_REASONING_EFFORT_MODELS

    def supports_temperature(self, model_name: str | None = None) -> bool:
        return True

    async def get_openai_body(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        call_api_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = context or {}
        prompt_config = (ctx.get("prompt") or {}).get("config")
        config = {**self.config, **(prompt_config or {})}
        model = get_xai_request_model(self.model_name, config)
        uses_grok47 = model == "grok-4.7"
        passthrough = config.get("passthrough")
        uses_passthrough_reasoning_model = (
            isinstance(passthrough, dict)
            and isinstance(passthrough.get("model"), str)
            and model in GROK_REASONING_MODELS
        )
        test_options = (ctx.get("test") or {}).get("options")
        vars = ctx.get("vars")
        effort: str | None = None
        if uses_grok47:
            effort = resolve_grok47_reasoning_effort(
                _defined(
                    get_xai_request_option(
                        "reasoning_effort", test_options, prompt_config, self.config
                    )
                ),
                vars,
            )
            validate_xai_reasoning_effort(model, effort, "reasoning_effort")
        result = await super().get_openai_body(prompt, context, call_api_options)
        if not result or not result.get("body"):
            return result
        body = result["body"]

        assert_xai_service_tier(body.get("service_tier"))

        if uses_grok47:
            if effort is None:
                body.pop("reasoning_effort", None)
            else:
                body["reasoning_effort"] = effort
        elif uses_passthrough_reasoning_model and model in GROK_REASONING_EFFORT_MODELS:
            configured = _defined(
                get_xai_request_option("reasoning_effort", test_options, prompt_config, self.config)
            )
            resolved = None if configured is None else render_vars_in_object(configured, vars)
            if resolved is None:
                body.pop("reasoning_effort", None)
            else:
                body["reasoning_effort"] = resolved
        if uses_grok47 or model in _GROK_BUILD_CHAT_MODELS or uses_passthrough_reasoning_model:
            token_limit = get_xai_request_option(
                ["max_completion_tokens", "max_tokens"], test_options, prompt_config, self.config
            )
            # JS `??`: a stored null also falls back to the env limit.
            if token_limit is _MISSING or token_limit is None:
                token_limit = get_openai_chat_output_limit_from_env()
            body.pop("max_tokens", None)
            body.pop("max_completion_tokens", None)
            if token_limit is not None:
                key = "max_tokens" if model in _GROK_BUILD_CHAT_MODELS else "max_completion_tokens"
                body[key] = token_limit

        # Filter out unsupported sampling controls for Grok-4-family models.
        if model in GROK_4_MODELS:
            body.pop("presence_penalty", None)
            body.pop("frequency_penalty", None)
            body.pop("stop", None)

        validate_xai_reasoning_effort(model, body.get("reasoning_effort"), "reasoning_effort")

        if model not in GROK_REASONING_EFFORT_MODELS and body.get("reasoning_effort"):
            del body["reasoning_effort"]

        # Live Search parameters.
        search_params = (self.original_config or {}).get("search_parameters")
        if search_params:
            body["search_parameters"] = render_vars_in_object(search_params, vars)
        return result

    def default_id(self) -> str:
        return f"xai:{self.model_name}"

    def __str__(self) -> str:
        return f"[xAI Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "xai", lambda: self.api_key)

    def calculate_response_cost(
        self, data: Any, config: dict[str, Any], cached: bool
    ) -> float | None:
        if cached:
            return 0
        data = data if isinstance(data, dict) else {}
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        reported = None if has_xai_cost_overrides(config) else get_xai_cost_in_usd(usage)
        if reported is not None:
            return reported
        usage = usage or {}
        completion_details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        return calculate_xai_cost(
            get_xai_request_model(self.model_name, config),
            config,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            completion_details.get("reasoning_tokens"),
            prompt_details.get("cached_tokens"),
            {
                "apiUrl": self.get_api_url(),
                "reasoningBilledSeparately": True,
                "serviceTier": "priority" if data.get("service_tier") == "priority" else None,
            },
        )

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        try:
            response = await super().call_api(prompt, context, options)
            error = response.get("error") if response else None
            if not response or error:
                if error and (
                    "502 Bad Gateway" in error
                    or "invalid API key" in error
                    or "authentication error" in error
                ):
                    return {**response, "error": f"x.ai API error: {error}\n\n{_XAI_TIP}"}
                return response
            return response
        except XAIRequestConfigError as err:
            return {"error": f"xAI request error: {err}"}
        except Exception as err:
            message = str(err)
            if "Error parsing response" in message and "<html" in message:
                return {
                    "error": "x.ai API error: Server returned an HTML error page instead of JSON. "
                    "This often indicates an invalid API key or server issues.\n\n" + _XAI_TIP
                }
            if "502" in message or "Bad Gateway" in message:
                return {
                    "error": "x.ai API error: 502 Bad Gateway - This often indicates an invalid "
                    "API key.\n\n" + _XAI_TIP
                }
            return {
                "error": f"x.ai API error: {message}\n\n"
                "If this persists, verify your API key at https://x.ai/"
            }


def create_xai_provider(provider_path: str, options: dict[str, Any] | None = None) -> XAIProvider:
    model_name = ":".join(provider_path.split(":")[1:])
    if not model_name:
        raise ValueError("Model name is required")
    return XAIProvider(model_name, options or {})


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `xai:` factory in src/providers/registry.ts."""
    splits = provider_path.split(":")
    model_type = splits[1] if len(splits) > 1 else None
    if model_type in ("embedding", "embeddings"):
        raise ValueError(
            "xAI does not currently expose a public embeddings API; "
            f'"{provider_path}" cannot be resolved. '
            "Use openai:embedding:* or another embedding provider instead."
        )
    if model_type == "image":
        raise not_supported(provider_path, "The xAI image provider")
    if model_type == "video":
        raise not_supported(provider_path, "The xAI video provider")
    if model_type == "responses":
        raise not_supported(
            provider_path,
            "The xAI Responses provider",
            "It needs the OpenAI Responses API port; use xai:<model> for chat completions.",
        )
    if model_type == "voice":
        raise not_supported(provider_path, "The xAI voice provider")
    return create_xai_provider(provider_path, configured_provider_options(options))
