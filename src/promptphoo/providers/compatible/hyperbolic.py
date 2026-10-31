"""Hyperbolic chat provider. Ports src/providers/hyperbolic/chat.ts and the `hyperbolic:`
factory in src/providers/registry.ts.

`hyperbolic:image:` and `hyperbolic:audio:` use Hyperbolic's own image and audio APIs and are
not supported by promptphoo.
"""

from __future__ import annotations

import logging
from typing import Any

from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.compatible.common import not_supported, serialize_provider
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider

logger = logging.getLogger(__name__)


def _models_with_cost(models: list[tuple[str, list[str]]], cost: dict[str, float]) -> list[dict]:
    return [{"id": i, "cost": dict(cost), "aliases": a} for i, a in models]


def _model(model_id: str, inp: float, out: float, aliases: list[str]) -> dict[str, Any]:
    return {"id": model_id, "cost": {"input": inp / 1e6, "output": out / 1e6}, "aliases": aliases}


HYPERBOLIC_CHAT_MODELS: list[dict[str, Any]] = [
    # DeepSeek Models
    *_models_with_cost(
        [("deepseek-ai/DeepSeek-R1", ["DeepSeek-R1"]),
         ("deepseek-ai/DeepSeek-R1-Zero", ["DeepSeek-R1-Zero"])],
        {"input": 0.5 / 1e6, "output": 2.18 / 1e6},
    ),
    *_models_with_cost(
        [("deepseek-ai/DeepSeek-V3", ["DeepSeek-V3"]),
         ("deepseek-ai/DeepSeek-V3-0324", ["DeepSeek-V3-0324"])],
        {"input": 0.27 / 1e6, "output": 1.1 / 1e6},
    ),
    _model("deepseek/DeepSeek-V2.5", 0.14, 0.28, ["DeepSeek-V2.5"]),
    # Qwen Models
    _model("qwen/Qwen3-235B-A22B", 2.0, 8.0, ["Qwen3-235B-A22B"]),
    *_models_with_cost(
        [("qwen/QwQ-32B", ["QwQ-32B"]),
         ("qwen/QwQ-32B-Preview", ["QwQ-32B-Preview", "Qwen/QwQ-32B-Preview"])],
        {"input": 0.15 / 1e6, "output": 0.6 / 1e6},
    ),
    _model("qwen/Qwen2.5-72B-Instruct", 0.35, 0.4, ["Qwen2.5-72B-Instruct", "Qwen2.5-72B"]),
    _model("qwen/Qwen2.5-Coder-32B", 0.18, 0.18, ["Qwen2.5-Coder-32B"]),
    # Llama Models
    _model("meta-llama/Llama-3.3-70B-Instruct", 0.35, 0.4,
           ["Llama-3.3-70B-Instruct", "Llama-3.3-70B"]),
    _model("meta-llama/Llama-3.2-3B", 0.02, 0.02, ["Llama-3.2-3B"]),
    *_models_with_cost(
        [("meta-llama/Llama-3.1-405B", ["Llama-3.1-405B"]),
         ("meta-llama/Llama-3.1-405B-BASE", ["Llama-3.1-405B-BASE"])],
        {"input": 3.0 / 1e6, "output": 3.0 / 1e6},
    ),
    _model("meta-llama/Llama-3.1-70B", 0.35, 0.4,
           ["Llama-3.1-70B", "Meta-Llama-3.1-70B-Instruct"]),
    _model("meta-llama/Llama-3.1-8B", 0.04, 0.04, ["Llama-3.1-8B"]),
    *_models_with_cost(
        [("meta-llama/Llama-3-70B", ["Llama-3-70B"]),
         # Hermes Models
         ("hermes/Hermes-3-70B", ["Hermes-3-70B"])],
        {"input": 0.35 / 1e6, "output": 0.4 / 1e6},
    ),
    # Vision-Language Models
    _model("qwen/Qwen2.5-VL-7B-Instruct", 0.2, 0.2,
           ["Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]),
    _model("qwen/Qwen2.5-VL-72B-Instruct", 0.6, 0.6,
           ["Qwen2.5-VL-72B-Instruct", "qwen/Qwen2.5-VL-72B-Instruct"]),
    _model("mistralai/Pixtral-12B", 0.15, 0.15, ["Pixtral-12B"]),
]  # fmt: skip

HYPERBOLIC_REASONING_MODELS = [
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-R1-Zero",
    "qwen/QwQ-32B",
    "qwen/QwQ-32B-Preview",
]


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def calculate_hyperbolic_cost(
    model_name: str,
    config: dict[str, Any],
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    reasoning_tokens: Any = None,
) -> float | None:
    if not prompt_tokens or not completion_tokens:
        return None
    model = next(
        (
            m
            for m in HYPERBOLIC_CHAT_MODELS
            if m["id"] == model_name or model_name in (m.get("aliases") or [])
        ),
        None,
    )
    if not model or not model.get("cost"):
        return None
    input_cost = _first(config.get("inputCost"), config.get("cost"), model["cost"]["input"])
    output_cost = _first(config.get("outputCost"), config.get("cost"), model["cost"]["output"])
    input_total = input_cost * prompt_tokens
    output_total = output_cost * completion_tokens
    logger.debug(
        "Hyperbolic cost calculation for %s: promptTokens=%s, completionTokens=%s, "
        "reasoningTokens=%s, inputCost=%s, outputCost=%s",
        model_name,
        prompt_tokens,
        completion_tokens,
        reasoning_tokens or "N/A",
        input_total,
        output_total,
    )
    return input_total + output_total


class HyperbolicProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        super().__init__(
            model_name,
            {
                **provider_options,
                "config": {
                    **(provider_options.get("config") or {}),
                    "apiKeyEnvar": "HYPERBOLIC_API_KEY",
                    "apiBaseUrl": "https://api.hyperbolic.xyz/v1",
                },
            },
        )
        self.original_config = (provider_options.get("config") or {}).get("config")

    def is_reasoning_model(self, model_name: str | None = None) -> bool:
        return self.model_name in HYPERBOLIC_REASONING_MODELS

    def supports_temperature(self, model_name: str | None = None) -> bool:
        # Reasoning models may have limited temperature support.
        return True

    def default_id(self) -> str:
        return f"hyperbolic:{self.model_name}"

    def __str__(self) -> str:
        return f"[Hyperbolic Provider {self.model_name}]"

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "hyperbolic", lambda: self.config.get("apiKey"))

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        response = await super().call_api(prompt, context, options)
        if not response or (
            response.get("error")
            and (
                (response.get("metadata") or {}).get("errorOrigin") != "tool"
                or not response.get("tokenUsage")
            )
        ):
            return response
        try:
            token_usage = response.get("tokenUsage")
            if token_usage and not response.get("cached"):
                reasoning_tokens = (token_usage.get("completionDetails") or {}).get(
                    "reasoning"
                ) or 0
                cost = calculate_hyperbolic_cost(
                    self.model_name,
                    self.config or {},
                    token_usage.get("prompt"),
                    token_usage.get("completion"),
                    reasoning_tokens,
                )
                # JS assigns undefined, which drops the key on serialization.
                if cost is None:
                    response.pop("cost", None)
                else:
                    response["cost"] = cost
        except Exception as err:
            logger.error("Error processing Hyperbolic response: %s", err)
        return response


def create_hyperbolic_provider(provider_path: str, options: dict[str, Any] | None = None) -> Any:
    model_name = ":".join(provider_path.split(":")[1:])
    if not model_name:
        raise ValueError("Invariant failed: Model name is required")
    return HyperbolicProvider(model_name, options or {})


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `hyperbolic:` factory in src/providers/registry.ts."""
    splits = provider_path.split(":")
    model_type = splits[1] if len(splits) > 1 else None
    if model_type == "image":
        raise not_supported(
            provider_path,
            "The Hyperbolic image provider",
            "Use hyperbolic:<model> for chat completions.",
        )
    if model_type == "audio":
        raise not_supported(
            provider_path,
            "The Hyperbolic audio provider",
            "Use hyperbolic:<model> for chat completions.",
        )
    return create_hyperbolic_provider(provider_path, options)
