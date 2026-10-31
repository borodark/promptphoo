"""Perplexity chat provider. Ports src/providers/perplexity.ts and the `perplexity:` factory in
src/providers/registry.ts."""

from __future__ import annotations

import math
import re
from typing import Any

from promptphoo.providers.compatible.common import configured_provider_options
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider


def calculate_perplexity_cost(
    model_name: str,
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    _usage_tier: str = "medium",
) -> float | None:
    """Legacy input/output token estimate only (deprecated upstream: use the API's
    `usage.cost.total_cost`)."""
    input_tokens = prompt_tokens or 0
    output_tokens = completion_tokens or 0
    prices = {
        "sonar-pro": (3, 15),
        "sonar-reasoning-pro": (2, 8),
        "sonar-reasoning": (1, 5),
        "sonar-deep-research": (2, 8),
        "r1-1776": (2, 8),
        "sonar": (1, 1),
    }
    price = prices.get(model_name.lower())
    if price is None:
        return None
    return (input_tokens / 1_000_000) * price[0] + (output_tokens / 1_000_000) * price[1]


PERPLEXITY_PASSTHROUGH_FIELDS = (
    "search_domain_filter",
    "search_recency_filter",
    "return_related_questions",
    "return_images",
    "search_after_date_filter",
    "search_before_date_filter",
    "web_search_options",
)

_HTTP_URL = re.compile(r"^https?://", re.IGNORECASE)


def _normalize_perplexity_citations(citations: list[Any]) -> list[Any]:
    return [
        {"url": c, "content": c} if isinstance(c, str) and _HTTP_URL.match(c) else c
        for c in citations
    ]


_UNSET = object()


class PerplexityProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        outer = provider_options.get("config") or {}
        # Handle the case when config is nested inside config.
        actual_config = outer.get("config") or outer or {}
        normalized = {
            **provider_options,
            "config": {
                **actual_config,
                "apiBaseUrl": "https://api.perplexity.ai",
                "apiKeyEnvar": "PERPLEXITY_API_KEY",
            },
        }
        super().__init__(model_name, normalized)
        self.model_name = model_name
        self.config = normalized["config"]

    async def get_openai_body(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        call_api_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = await super().get_openai_body(prompt, context, call_api_options)
        resolved_config = result["config"]
        resolved_passthrough = resolved_config.get("passthrough")
        prompt_config = ((context or {}).get("prompt") or {}).get("config")
        prompt_config = prompt_config if isinstance(prompt_config, dict) else None
        prompt_passthrough = (prompt_config or {}).get("passthrough")
        prompt_passthrough = prompt_passthrough if isinstance(prompt_passthrough, dict) else None

        for field in PERPLEXITY_PASSTHROUGH_FIELDS:
            if prompt_passthrough is not None and field in prompt_passthrough:
                result["body"][field] = prompt_passthrough[field]
                continue
            # Null clears an inherited option; only a missing key falls through.
            value: Any = _UNSET
            for scope in (prompt_config, resolved_passthrough, resolved_config):
                if isinstance(scope, dict) and field in scope:
                    value = scope[field]
                    break
            if value is not _UNSET:
                result["body"][field] = value
        return result

    def calculate_response_cost(
        self, data: Any, config: dict[str, Any], cached: bool
    ) -> float | None:
        usage = data.get("usage") if isinstance(data, dict) else None
        cost = usage.get("cost") if isinstance(usage, dict) else None
        total = cost.get("total_cost") if isinstance(cost, dict) else None
        if (
            isinstance(total, int | float)
            and not isinstance(total, bool)
            and math.isfinite(total)
            and total >= 0
        ):
            return total
        # Token counts omit request, search, citation, and reasoning charges.
        return None

    def get_provider_response_metadata(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        perplexity = {
            field: data[field]
            for field in ("citations", "search_results", "images", "related_questions")
            if isinstance(data.get(field), list)
        }
        out: dict[str, Any] = {}
        if isinstance(data.get("citations"), list):
            out["citations"] = _normalize_perplexity_citations(data["citations"])
        if perplexity:
            out["perplexity"] = perplexity
        return out

    def default_id(self) -> str:
        return self.model_name

    def __str__(self) -> str:
        return f"[Perplexity Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        config = {k: v for k, v in self.config.items() if k != "apiKey"}
        return {"provider": "perplexity", "model": self.model_name, "config": config}


def create_perplexity_provider(
    provider_path: str, options: dict[str, Any] | None = None
) -> PerplexityProvider:
    model_name = ":".join(provider_path.split(":")[1:]) or "sonar"
    return PerplexityProvider(model_name, options or {})


def create_provider(provider_path: str, options: dict[str, Any]) -> PerplexityProvider:
    """`configuredProviderFactory('perplexity:', ...)` in src/providers/registry.ts."""
    return create_perplexity_provider(provider_path, configured_provider_options(options))
