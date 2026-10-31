"""Novita AI providers (chat, completion, embedding). Ports src/providers/novita.ts and the
`novita:` entry of src/providers/families/compatible.ts.
"""

from __future__ import annotations

import re
from typing import Any

from promptphoo import envars
from promptphoo.providers.compatible.common import resolve_provider_creator_input
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.completion import OpenAiCompletionProvider
from promptphoo.providers.openai.embedding import OpenAiEmbeddingProvider

NOVITA_API_BASE_URL = "https://api.novita.ai/openai/v1"
NOVITA_API_KEY_ENV_VAR = "NOVITA_API_KEY"
NOVITA_SUBTYPES = frozenset(["chat", "completion", "embedding", "embeddings"])


def get_novita_api_key(config: dict[str, Any], env: Any) -> str | None:
    if config.get("apiKey") is not None:
        return config["apiKey"]
    api_key_envar = config.get("apiKeyEnvar")
    api_key_envar = NOVITA_API_KEY_ENV_VAR if api_key_envar is None else api_key_envar
    value = envars.get_provider_env_string(env, api_key_envar)
    return value if value is not None else envars.get_env_string(api_key_envar)


def get_novita_config(
    config: dict[str, Any] | None, include_text_defaults: bool = True
) -> dict[str, Any]:
    config = config or {}
    out = {
        **config,
        "apiBaseUrl": config["apiBaseUrl"]
        if config.get("apiBaseUrl") is not None
        else NOVITA_API_BASE_URL,
        "apiKeyEnvar": config["apiKeyEnvar"]
        if config.get("apiKeyEnvar") is not None
        else NOVITA_API_KEY_ENV_VAR,
    }
    if include_text_defaults and config.get("temperature") is None:
        out["temperature"] = 1
    return out


def get_novita_missing_api_key_error_message(config: dict[str, Any]) -> str:
    envar = config.get("apiKeyEnvar")
    envar = NOVITA_API_KEY_ENV_VAR if envar is None else envar
    return (
        f"Novita API key is not set. Set the {envar} environment variable or add `apiKey` to "
        "the provider config."
    )


class _NovitaMixin:
    config: dict[str, Any]
    env: Any
    model_name: str
    NOVITA_TYPE = "chat"

    def default_id(self) -> str:
        return f"novita:{self.NOVITA_TYPE}:{self.model_name}"

    def get_api_key(self, config: dict[str, Any] | None = None) -> str | None:
        return get_novita_api_key(self.config, self.env)

    def get_organization(self, config: dict[str, Any] | None = None) -> str | None:
        return None

    def get_api_url(self, config: dict[str, Any] | None = None) -> str:
        url = self.config.get("apiBaseUrl")
        return NOVITA_API_BASE_URL if url is None else url

    def get_missing_api_key_error_message(self, config: dict[str, Any] | None = None) -> str:
        return get_novita_missing_api_key_error_message(self.config)


class NovitaChatCompletionProvider(_NovitaMixin, OpenAiChatCompletionProvider):
    NOVITA_TYPE = "chat"

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        super().__init__(
            model_name, {**options, "config": get_novita_config(options.get("config"))}
        )


class NovitaCompletionProvider(_NovitaMixin, OpenAiCompletionProvider):
    NOVITA_TYPE = "completion"

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        super().__init__(
            model_name, {**options, "config": get_novita_config(options.get("config"))}
        )


class NovitaEmbeddingProvider(_NovitaMixin, OpenAiEmbeddingProvider):
    NOVITA_TYPE = "embedding"

    def __init__(self, model_name: str, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        super().__init__(
            model_name, {**options, "config": get_novita_config(options.get("config"), False)}
        )


_SUBTYPE_LIKE = re.compile(r"^[a-z][a-z0-9_-]*$", re.I)


def create_novita_provider(provider_path: str, options: dict[str, Any] | None = None) -> Any:
    options = options or {}
    provider_options = resolve_provider_creator_input(options)
    splits = provider_path.split(":")
    model_type = splits[1] if len(splits) > 1 else ""
    is_typed = model_type in NOVITA_SUBTYPES
    # A model id always contains `/` or `.`; a bare word followed by more segments is a sub-type.
    if not is_typed and len(splits) >= 3 and _SUBTYPE_LIKE.match(model_type):
        raise ValueError(
            f'Unknown Novita provider sub-type "{model_type}". Supported sub-types are: chat, '
            "completion, embedding, embeddings. Use `novita:<model>` for chat, or "
            "`novita:<sub-type>:<model>` for an explicit sub-type."
        )
    model_name = ":".join(splits[2:] if is_typed else splits[1:])
    if not model_name.strip():
        raise ValueError(
            "Novita model name is required. Use `novita:<model-name>` or "
            "`novita:chat:<model-name>`."
        )
    if options.get("providerOptions") is not None:
        env = provider_options.get("env")
    else:
        env = options.get("env")
        if env is None:
            env = (options.get("config") or {}).get("env")
    novita_options = {
        **provider_options,
        "env": env,
        "config": provider_options.get("config") or {},
    }
    if model_type == "chat":
        return NovitaChatCompletionProvider(model_name, novita_options)
    if model_type == "completion":
        return NovitaCompletionProvider(model_name, novita_options)
    if model_type in ("embedding", "embeddings"):
        return NovitaEmbeddingProvider(model_name, novita_options)
    return NovitaChatCompletionProvider(model_name, novita_options)


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `novita:` family factory (families/compatible.ts)."""
    return create_novita_provider(provider_path, {"providerOptions": options})
