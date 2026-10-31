"""Docker Model Runner providers. Ports src/providers/docker.ts and the `docker:` entry of
src/providers/registry.ts.
"""

from __future__ import annotations

import logging
from typing import Any

from promptphoo import envars, fetch
from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.completion import OpenAiCompletionProvider
from promptphoo.providers.openai.embedding import OpenAiEmbeddingProvider

logger = logging.getLogger(__name__)


async def fetch_local_models(api_base_url: str) -> list[dict[str, Any]]:
    try:
        result = await fetch.fetch_with_cache(f"{api_base_url}/models", None, None, "json", True, 0)
        data = result.data
        models = data.get("data") if isinstance(data, dict) else None
        return models if models is not None else []
    except Exception as error:
        raise RuntimeError(
            "Failed to connect to Docker Model Runner. Is it enabled? Are the API endpoints "
            "enabled? For details, see https://docs.docker.com/ai/model-runner. \n"
            f"{error}"
        ) from error


async def has_local_model(model_id: str, api_base_url: str) -> bool:
    local_models = await fetch_local_models(api_base_url)
    target = (model_id or "").lower()
    return any(
        isinstance(model, dict)
        and isinstance(model.get("id"), str)
        and model["id"].lower() == target
        for model in local_models
    )


def parse_provider_path(provider_path: str) -> dict[str, str]:
    splits = provider_path.split(":")
    model_type = splits[1] if len(splits) > 1 else ""
    if model_type in ("chat", "completion", "embeddings"):
        return {"type": model_type, "model": ":".join(splits[2:])}
    if model_type == "embedding":
        # Either docker:embedding:model or docker:embeddings:model.
        return {"type": "embeddings", "model": ":".join(splits[2:])}
    return {"type": "chat", "model": ":".join(splits[1:])}


def _first(*values: Any) -> Any:
    return next((v for v in values if v is not None), None)


def create_docker_provider(provider_path: str, options: dict[str, Any] | None = None) -> Any:
    options = options or {}
    env = options.get("env")
    api_url = _first(
        envars.get_provider_env_string(env, "DOCKER_MODEL_RUNNER_BASE_URL"),
        envars.get_env_string("DOCKER_MODEL_RUNNER_BASE_URL"),
        "http://localhost:12434",
    )
    api_base_url = api_url + "/engines/v1"
    api_key = _first(
        envars.get_provider_env_string(env, "DOCKER_MODEL_RUNNER_API_KEY"),
        envars.get_env_string("DOCKER_MODEL_RUNNER_API_KEY"),
        "dmr",
    )
    openai_options = {
        **options,
        "config": {**(options.get("config") or {}), "apiBaseUrl": api_base_url, "apiKey": api_key},
    }
    parsed = parse_provider_path(provider_path)
    if parsed["type"] == "completion":
        return DMRCompletionProvider(parsed["model"], openai_options)
    if parsed["type"] == "embeddings":
        return DMREmbeddingProvider(parsed["model"], openai_options)
    return DMRChatCompletionProvider(parsed["model"], openai_options)


async def _warn_if_missing(provider: Any) -> None:
    if not await has_local_model(provider.model_name, provider.get_api_url()):
        logger.warning(
            "Model '%s' not found. Run 'docker model pull %s'.",
            provider.model_name,
            provider.model_name,
        )


class DMRChatCompletionProvider(OpenAiChatCompletionProvider):
    def get_gen_ai_system(self) -> str:
        return "docker"

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        await _warn_if_missing(self)
        return await super().call_api(prompt, context, options)


class DMRCompletionProvider(OpenAiCompletionProvider):
    def get_gen_ai_system(self) -> str:
        return "docker"

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        await _warn_if_missing(self)
        return await super().call_api(prompt, context, options)


class DMREmbeddingProvider(OpenAiEmbeddingProvider):
    async def call_embedding_api(self, text, context=None, options=None) -> dict[str, Any]:
        await _warn_if_missing(self)
        return await super().call_embedding_api(text, context, options)


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `docker:` registry entry: `{...providerOptions, env: context.env}`."""
    return create_docker_provider(provider_path, {**options, "env": options.get("env")})
