"""Helpers shared by the OpenAI-compatible provider families. Ports
src/providers/serialization.ts, src/providers/creator.ts and src/providers/openai/localOptions.ts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.completion import OpenAiCompletionProvider
from promptphoo.providers.openai.embedding import OpenAiEmbeddingProvider


def serialize_provider(
    provider: Any, name: str, get_api_key: Callable[[], Any] | None = None
) -> dict[str, Any]:
    """`{provider, model, config}` with `apiKey` dropped when one is set (serializeProvider)."""
    config = dict(provider.config or {})
    api_key = get_api_key() if get_api_key else config.get("apiKey")
    if api_key:
        config.pop("apiKey", None)
    return {"provider": name, "model": provider.model_name, "config": config}


def resolve_provider_creator_input(options: dict[str, Any] | None) -> dict[str, Any]:
    """Canonical registry input; the legacy `{config: ProviderOptions, id, env}` shape is
    accepted too (resolveProviderCreatorInput)."""
    options = options or {}
    if options.get("providerOptions") is not None:
        return options["providerOptions"]
    legacy = options.get("config") or {}
    out: dict[str, Any] = {**legacy}
    provider_id = options.get("id") or legacy.get("id")
    if provider_id is not None:
        out["id"] = provider_id
    else:
        out.pop("id", None)
    if options.get("env") or legacy.get("env"):
        out["env"] = {
            **(options.get("env") or {}),
            **{k: v for k, v in (legacy.get("env") or {}).items() if v is not None},
        }
    else:
        out["env"] = None
    return out


# Config keys promptfoo or its transport consumes, rather than model parameters.
LOCAL_OPTION_NAMES = frozenset([
    "apiKey", "apiKeyEnvar", "apiKeyRequired", "useDefaultApiKey", "apiHost", "apiBaseUrl",
    "organization", "headers", "maxRetries", "cost", "inputCost", "outputCost", "audioCost",
    "audioInputCost", "audioOutputCost", "passthrough", "mcp", "functionToolCallbacks",
    "showThinking", "omitDefaults", "basePath", "linkedTargetId",
])  # fmt: skip


def split_local_options(
    config: dict[str, Any] | None = None, extra_local_keys: Iterable[str] = ()
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns `(local_options, model_parameters)` (splitLocalOptions)."""
    extra = set(extra_local_keys)
    local: dict[str, Any] = {}
    params: dict[str, Any] = {}
    for key, value in (config or {}).items():
        (local if key in LOCAL_OPTION_NAMES or key in extra else params)[key] = value
    return local, params


# A route segment naming the OpenAI provider type; anything else is part of the model name.
PROVIDERS_BY_TYPE: dict[str, type] = {
    "chat": OpenAiChatCompletionProvider,
    "completion": OpenAiCompletionProvider,
    "embedding": OpenAiEmbeddingProvider,
    "embeddings": OpenAiEmbeddingProvider,
}


def configured_provider_options(options: dict[str, Any]) -> dict[str, Any]:
    """The `{config: providerOptions, env}` shape `configuredProviderFactory` passes
    (registry.ts). `options["env"]` is already the merged env, which is Node's `context.env`."""
    return {"config": options, "env": options.get("env")}


def not_supported(provider_path: str, kind: str, hint: str = "") -> ValueError:
    """The error for provider routes that need an SDK or API promptphoo does not port."""
    message = f"{kind} is not supported by promptphoo ({provider_path})."
    return ValueError(f"{message} {hint}".rstrip())
