"""NVIDIA NIM chat provider. Ports src/providers/nvidia/chat.ts and the `nvidia:` factory in
src/providers/registry.ts.
"""

from __future__ import annotations

from typing import Any

from promptphoo import envars
from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.compatible.common import serialize_provider
from promptphoo.providers.credentials import resolve_provider_env
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider

NVIDIA_NIM_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_RESERVED_PROVIDER_SUBTYPES = frozenset(
    ["chat", "completion", "embedding", "embeddings", "image", "moderation", "realtime",
     "responses"]
)  # fmt: skip


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def calculate_nvidia_cost(
    config: dict[str, Any],
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    cached: bool = False,
) -> float | None:
    input_cost = _first(config.get("inputCost"), config.get("cost"))
    output_cost = _first(config.get("outputCost"), config.get("cost"))
    if (
        input_cost is None
        or output_cost is None
        or prompt_tokens is None
        or completion_tokens is None
    ):
        return None
    return 0 if cached else input_cost * prompt_tokens + output_cost * completion_tokens


class NvidiaProvider(OpenAiChatCompletionProvider):
    def __init__(self, model_name: str, provider_options: dict[str, Any] | None = None) -> None:
        provider_options = provider_options or {}
        config = provider_options.get("config") or {}
        explicit_base_url = config.get("apiBaseUrl")
        env_base_url = (
            resolve_provider_env(provider_options.get("env"), ["NVIDIA_API_BASE_URL"]) or {}
        ).get("value")
        super().__init__(
            model_name,
            {
                **provider_options,
                "config": {
                    **config,
                    "apiBaseUrl": explicit_base_url or env_base_url or NVIDIA_NIM_API_BASE_URL,
                    "apiKeyEnvar": config.get("apiKeyEnvar") or "NVIDIA_API_KEY",
                },
            },
        )

    def default_id(self) -> str:
        return f"nvidia:{self.model_name}"

    def __str__(self) -> str:
        return f"[NVIDIA NIM Provider {self.model_name}]"

    # Don't fall through to OPENAI_API_KEY.
    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        envar = self.config.get("apiKeyEnvar") or "NVIDIA_API_KEY"
        return (
            self.config.get("apiKey") or (self.env or {}).get(envar) or envars.get_env_string(envar)
        )

    def get_organization(self, config: dict[str, Any] | None = None) -> str | None:
        return None

    # Never consult OPENAI_API_HOST / OPENAI_API_BASE_URL / OPENAI_BASE_URL.
    def get_api_url(self, config: dict[str, Any] | None = None) -> str:
        return self.config.get("apiBaseUrl") or NVIDIA_NIM_API_BASE_URL

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        response = await super().call_api(prompt, context, options)
        if response.get("error") and (
            (response.get("metadata") or {}).get("errorOrigin") != "tool"
            or not response.get("tokenUsage")
        ):
            return response
        config = {**self.config, **(((context or {}).get("prompt") or {}).get("config") or {})}
        token_usage = response.get("tokenUsage") or {}
        cost = calculate_nvidia_cost(
            config,
            token_usage.get("prompt"),
            token_usage.get("completion"),
            bool(response.get("cached")),
        )
        # JS assigns undefined, which drops the key on serialization.
        if cost is None:
            response.pop("cost", None)
        else:
            response["cost"] = cost
        return response

    def to_json(self) -> dict[str, Any]:
        return serialize_provider(self, "nvidia")


def create_nvidia_provider(provider_path: str, options: dict[str, Any] | None = None) -> Any:
    options = options or {}
    splits = provider_path.split(":")
    subtype = splits[1] if len(splits) > 1 else None
    if subtype in NVIDIA_RESERVED_PROVIDER_SUBTYPES:
        raise ValueError(
            f'Unsupported NVIDIA NIM provider subtype "{subtype}" in "{provider_path}". '
            'Use "nvidia:<model>" for chat-completion models.'
        )
    model_name = ":".join(splits[1:])
    if not model_name:
        raise ValueError(
            f'Invalid NVIDIA NIM provider path: "{provider_path}" — expected "nvidia:<model>"'
        )
    provider_options: dict[str, Any] = dict(options["config"]) if options.get("config") else {}
    if options.get("env") and not provider_options.get("env"):
        provider_options["env"] = options["env"]
    if options.get("id") and not provider_options.get("id"):
        provider_options["id"] = options["id"]
    return NvidiaProvider(model_name, provider_options)


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `nvidia:` factory in src/providers/registry.ts."""
    return create_nvidia_provider(provider_path, {"config": options, "env": options.get("env")})
