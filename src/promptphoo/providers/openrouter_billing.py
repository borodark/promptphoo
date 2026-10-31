"""OpenRouter usage accounting. Ports src/providers/openrouterBilling.ts."""

from __future__ import annotations

import math
from typing import Any, Literal

from promptphoo.providers.openai.urls import url_hostname


def is_openrouter_endpoint(api_url: str) -> bool:
    return url_hostname(api_url) == "openrouter.ai"


def _is_non_negative_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _coalesce(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def calculate_openrouter_response_cost(
    data: Any, config: dict[str, Any], api: Literal["chat", "responses"] = "chat"
) -> float | None:
    usage = data.get("usage") if isinstance(data, dict) else None
    usage = usage if isinstance(usage, dict) else None
    # Cache replay preserves logical cost; the evaluator tracks incurred spending separately.
    if any(config.get(k) is not None for k in ("cost", "inputCost", "outputCost")):
        # Explicit user rates override billing; incomplete rates must not fall back.
        input_cost = _coalesce(config.get("inputCost"), config.get("cost"))
        output_cost = _coalesce(config.get("outputCost"), config.get("cost"))
        u = usage or {}
        if api == "responses":
            prompt_tokens = _coalesce(u.get("input_tokens"), u.get("prompt_tokens"))
            completion_tokens = _coalesce(u.get("output_tokens"), u.get("completion_tokens"))
        else:
            prompt_tokens = u.get("prompt_tokens")
            completion_tokens = u.get("completion_tokens")
        if not all(
            _is_non_negative_finite_number(v)
            for v in (input_cost, output_cost, prompt_tokens, completion_tokens)
        ):
            return None
        cost = prompt_tokens * input_cost + completion_tokens * output_cost
        return cost if math.isfinite(cost) else None

    # BYOK is billed separately upstream, so the gateway charge cannot represent the total.
    if usage is not None and usage.get("is_byok") is True:
        return None
    # https://openrouter.ai/docs/cookbook/administration/usage-accounting
    cost = usage.get("cost") if usage is not None else None
    return cost if _is_non_negative_finite_number(cost) else None


def get_openrouter_billing_metadata(data: Any) -> dict[str, Any] | None:
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return None
    details = usage.get("cost_details")
    details = details if isinstance(details, dict) else {}
    amounts = {
        "accountCharge": usage.get("cost"),
        "reportedUpstreamInferenceCost": details.get("upstream_inference_cost"),
        "reportedUpstreamPromptCost": details.get("upstream_inference_prompt_cost"),
        "reportedUpstreamCompletionCost": details.get("upstream_inference_completions_cost"),
        "reportedServerToolCost": details.get("server_tool_cost"),
    }
    billing: dict[str, Any] = {
        name: amount for name, amount in amounts.items() if _is_non_negative_finite_number(amount)
    }
    if isinstance(usage.get("is_byok"), bool):
        billing["isByok"] = usage["is_byok"]
    return {"openrouter": billing} if billing else None
