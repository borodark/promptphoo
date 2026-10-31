"""Default grading providers, chosen from the available credentials. Ports
src/providers/defaults.ts.

Selection order (first match wins): Azure OpenAI, Anthropic, Google AI Studio, Google Vertex
(ADC), Mistral, xAI, Codex, then OpenAI. An Azure Content Safety endpoint replaces the
moderation provider, and `set_default_completion_providers` / `set_default_embedding_providers`
override the result.

Families that promptphoo has not ported yet (Mistral, xAI, Voyage, Azure Content Safety) are
imported lazily; when the module is missing the branch is skipped, as if its credentials were
absent. Codex defaults are out of scope (see D33).
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from promptphoo import envars
from promptphoo.providers.base import ApiProvider
from promptphoo.providers.credentials import resolve_provider_api_key

logger = logging.getLogger(__name__)

EnvOverrides = dict[str, Any]

COMPLETION_PROVIDERS = [
    "gradingJsonProvider",
    "gradingProvider",
    "llmRubricProvider",
    "suggestionsProvider",
    "synthesizeProvider",
]

EMBEDDING_PROVIDERS = ["embeddingProvider"]

_GOOGLE_API_KEY_ENVARS = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "PALM_API_KEY"]

# Process-wide overrides, as upstream's module-level variables.
_default_completion_provider: ApiProvider | None = None
_default_embedding_provider: ApiProvider | None = None


def _optional(module_name: str, attr: str) -> Any:
    """`module.attr`, or None when that provider family has not been ported yet."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name and module_name.startswith(e.name):
            return None
        raise
    return getattr(module, attr, None)


# TODO(mistral): the Mistral provider is not ported (the OpenAI-compatible table skips it).
def _get_mistral_providers() -> Any:
    return _optional("promptphoo.providers.mistral.defaults", "get_mistral_providers")


def _get_mistral_embedding_provider_class() -> Any:
    return _optional("promptphoo.providers.mistral", "MistralEmbeddingProvider")


# TODO(xai): xAI defaults need the xAI chat and Responses providers (OpenAI-compatible branch).
def _get_xai_providers() -> Any:
    return _optional("promptphoo.providers.xai.defaults", "get_xai_providers")


# TODO(voyage): the Voyage embedding provider is not ported.
def _get_voyage_embedding_provider_class() -> Any:
    return _optional("promptphoo.providers.voyage", "VoyageEmbeddingProvider")


# TODO(azure-moderation): Azure Content Safety is not ported (see D24).
def _get_azure_moderation_provider_class() -> Any:
    return _optional("promptphoo.providers.azure.moderation", "AzureModerationProvider")


async def has_google_default_credentials() -> bool:
    from promptphoo.providers.google.auth import has_google_default_credentials as probe

    return await probe()


def has_codex_default_credentials(env: EnvOverrides | None = None) -> bool:
    """Codex SDK defaults are out of scope, so ChatGPT/Codex credentials never select them."""
    return False


def _env_string(env: EnvOverrides | None, name: str) -> Any:
    """`env?.NAME ?? getEnvString('NAME')`: an explicit empty value masks the ambient one."""
    value = env.get(name) if env else None
    return value if value is not None else envars.get_env_string(name)


async def _get_embedding_provider_for_azure_defaults(env: EnvOverrides | None) -> ApiProvider:
    if _default_embedding_provider:
        return _default_embedding_provider

    embedding_deployment_name = _env_string(env, "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    if embedding_deployment_name:
        from promptphoo.providers.azure.embedding import AzureEmbeddingProvider

        return AzureEmbeddingProvider(embedding_deployment_name, {"env": env})

    # A chat deployment is not an embedding deployment. Prefer configured embedding API
    # credentials before probing ADC, without changing Azure's chat selection.
    if resolve_provider_api_key(None, env, _GOOGLE_API_KEY_ENVARS):
        from promptphoo.providers.google.ai_studio import AIStudioEmbeddingProvider

        return AIStudioEmbeddingProvider("gemini-embedding-001", {"env": env})
    if resolve_provider_api_key(None, env, ["MISTRAL_API_KEY"]):
        mistral_embedding = _get_mistral_embedding_provider_class()
        if mistral_embedding is not None:
            return mistral_embedding({"env": env})
    if resolve_provider_api_key(None, env, ["VOYAGE_API_KEY"]):
        voyage_embedding = _get_voyage_embedding_provider_class()
        if voyage_embedding is not None:
            return voyage_embedding("voyage-3.5", {}, env)
    if await has_google_default_credentials():
        from promptphoo.providers.google.vertex import get_google_vertex_embedding_provider

        return get_google_vertex_embedding_provider(env)
    from promptphoo.providers.openai.defaults import get_openai_providers

    return get_openai_providers(env)["embeddingProvider"]


async def _get_default_provider_preferences(env: EnvOverrides | None) -> dict[str, bool]:
    has_anthropic_credentials = bool(resolve_provider_api_key(None, env, ["ANTHROPIC_API_KEY"]))
    has_openai_credentials = bool(resolve_provider_api_key(None, env, ["OPENAI_API_KEY"]))
    has_google_ai_studio_credentials = bool(
        resolve_provider_api_key(None, env, _GOOGLE_API_KEY_ENVARS)
    )
    has_azure_api_key = resolve_provider_api_key(
        None, env, ["AZURE_API_KEY", "AZURE_OPENAI_API_KEY"]
    )
    has_azure_client_creds = (
        _env_string(env, "AZURE_CLIENT_ID")
        and _env_string(env, "AZURE_CLIENT_SECRET")
        and _env_string(env, "AZURE_TENANT_ID")
    )
    has_mistral_credentials = bool(resolve_provider_api_key(None, env, ["MISTRAL_API_KEY"]))
    has_xai_credentials = bool(resolve_provider_api_key(None, env, ["XAI_API_KEY"]))

    prefer_azure = bool(
        not has_openai_credentials
        and (has_azure_api_key or has_azure_client_creds)
        and _env_string(env, "AZURE_DEPLOYMENT_NAME")
        and _env_string(env, "AZURE_OPENAI_DEPLOYMENT_NAME")
    )
    prefer_anthropic = not has_openai_credentials and has_anthropic_credentials
    should_use_fallback_defaults = (
        not prefer_azure
        and not has_openai_credentials
        and not has_anthropic_credentials
        and not has_google_ai_studio_credentials
    )
    use_google_vertex_defaults = (
        await has_google_default_credentials() if should_use_fallback_defaults else False
    )
    use_non_google_fallback_defaults = (
        should_use_fallback_defaults and not use_google_vertex_defaults
    )
    has_codex_credentials = (
        use_non_google_fallback_defaults
        and not has_mistral_credentials
        and not has_xai_credentials
        and has_codex_default_credentials(env)
    )
    return {
        "preferAnthropic": prefer_anthropic,
        "preferAzure": prefer_azure,
        "useCodexDefaults": has_codex_credentials,
        "useGoogleAiStudioDefaults": not has_openai_credentials
        and not has_anthropic_credentials
        and has_google_ai_studio_credentials,
        "useGoogleVertexDefaults": use_google_vertex_defaults,
        "useMistralDefaults": use_non_google_fallback_defaults and has_mistral_credentials,
        "useXAIDefaults": use_non_google_fallback_defaults
        and not has_mistral_credentials
        and has_xai_credentials,
    }


async def set_default_completion_providers(provider: ApiProvider | None) -> None:
    """Override every completion-type default provider (COMPLETION_PROVIDERS)."""
    global _default_completion_provider
    _default_completion_provider = provider


async def set_default_embedding_providers(provider: ApiProvider | None) -> None:
    global _default_embedding_provider
    _default_embedding_provider = provider


async def get_default_providers(env: EnvOverrides | None = None) -> dict[str, Any]:
    prefs = await _get_default_provider_preferences(env)

    from promptphoo.providers.openai.defaults import get_openai_providers

    openai_providers = get_openai_providers(env)
    mistral_providers = _get_mistral_providers() if prefs["useMistralDefaults"] else None
    xai_providers = _get_xai_providers() if prefs["useXAIDefaults"] else None

    providers: dict[str, Any]
    if prefs["preferAzure"]:
        logger.debug("Using Azure OpenAI default providers")
        deployment_name = _env_string(env, "AZURE_OPENAI_DEPLOYMENT_NAME")
        if not deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME must be set when using Azure OpenAI")
        from promptphoo.providers.azure.chat import AzureChatCompletionProvider

        azure_provider = AzureChatCompletionProvider(deployment_name, {"env": env})
        providers = {
            "embeddingProvider": await _get_embedding_provider_for_azure_defaults(env),
            "gradingJsonProvider": azure_provider,
            "gradingProvider": azure_provider,
            "moderationProvider": openai_providers["moderationProvider"],
            "suggestionsProvider": azure_provider,
            "synthesizeProvider": azure_provider,
            # Azure doesn't have web search by default
        }
    elif prefs["preferAnthropic"]:
        logger.debug("Using Anthropic default providers")
        from promptphoo.providers.anthropic.defaults import get_anthropic_providers

        anthropic_providers = get_anthropic_providers(env)
        providers = {
            "embeddingProvider": openai_providers["embeddingProvider"],
            "gradingJsonProvider": anthropic_providers["gradingJsonProvider"],
            "gradingProvider": anthropic_providers["gradingProvider"],
            "llmRubricProvider": anthropic_providers["llmRubricProvider"],
            "moderationProvider": openai_providers["moderationProvider"],
            "suggestionsProvider": anthropic_providers["suggestionsProvider"],
            "synthesizeProvider": anthropic_providers["synthesizeProvider"],
            "webSearchProvider": anthropic_providers["webSearchProvider"],
        }
    elif prefs["useGoogleAiStudioDefaults"]:
        logger.debug("Using Google AI Studio default providers")
        from promptphoo.providers.google.ai_studio import get_google_ai_studio_providers
        from promptphoo.providers.google.vertex import get_google_vertex_embedding_provider

        providers = {
            # AI Studio supports embeddings via google:embedding:*, but Vertex is the richer
            # default.
            "embeddingProvider": get_google_vertex_embedding_provider(env),
            "moderationProvider": openai_providers["moderationProvider"],
            **get_google_ai_studio_providers(env),
        }
    elif prefs["useGoogleVertexDefaults"]:
        logger.debug("Using Google Vertex default providers")
        from promptphoo.providers.google.vertex import get_google_vertex_providers

        providers = {
            "moderationProvider": openai_providers["moderationProvider"],
            **get_google_vertex_providers(env),
        }
    elif mistral_providers is not None:
        logger.debug("Using Mistral default providers")
        bundle = mistral_providers(env)
        providers = {
            "embeddingProvider": bundle["embeddingProvider"],
            "gradingJsonProvider": bundle["gradingJsonProvider"],
            "gradingProvider": bundle["gradingProvider"],
            "moderationProvider": openai_providers["moderationProvider"],
            "suggestionsProvider": bundle["suggestionsProvider"],
            "synthesizeProvider": bundle["synthesizeProvider"],
            # Mistral doesn't have web search
        }
    elif xai_providers is not None:
        logger.debug("Using xAI default providers")
        providers = {
            # xAI exposes neither an embeddings nor a moderation API.
            "embeddingProvider": openai_providers["embeddingProvider"],
            "moderationProvider": openai_providers["moderationProvider"],
            **xai_providers(env),
        }
    # Codex SDK defaults (prefs["useCodexDefaults"]) are out of scope and never selected.
    else:
        logger.debug("Using OpenAI default providers")
        providers = {
            "embeddingProvider": openai_providers["embeddingProvider"],
            "gradingJsonProvider": openai_providers["gradingJsonProvider"],
            "gradingProvider": openai_providers["gradingProvider"],
            "moderationProvider": openai_providers["moderationProvider"],
            "suggestionsProvider": openai_providers["suggestionsProvider"],
            "synthesizeProvider": openai_providers["gradingJsonProvider"],
            "webSearchProvider": openai_providers["webSearchProvider"],
        }

    # If an Azure Content Safety endpoint is available, use it for moderation
    if _env_string(env, "AZURE_CONTENT_SAFETY_ENDPOINT"):
        azure_moderation = _get_azure_moderation_provider_class()
        if azure_moderation is not None:
            providers["moderationProvider"] = azure_moderation("text-content-safety", {"env": env})
        else:
            logger.debug(
                "AZURE_CONTENT_SAFETY_ENDPOINT is set, but Azure Content Safety moderation is "
                "not supported by promptphoo; using the OpenAI moderation provider"
            )

    if _default_completion_provider:
        logger.debug(
            "Overriding default completion provider: %s", _default_completion_provider.id()
        )
        for key in COMPLETION_PROVIDERS:
            providers[key] = _default_completion_provider

    if _default_embedding_provider:
        for key in EMBEDDING_PROVIDERS:
            providers[key] = _default_embedding_provider
    return providers
