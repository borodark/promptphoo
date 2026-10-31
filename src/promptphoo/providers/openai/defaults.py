"""Default OpenAI grading providers. Ports src/providers/openai/defaults.ts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.embedding import OpenAiEmbeddingProvider
from promptphoo.providers.openai.moderation import OpenAiModerationProvider
from promptphoo.providers.openai.responses import OpenAiResponsesProvider

DEFAULT_OPENAI_MODEL = "gpt-6-sol"


def _create_providers(env: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "embeddingProvider": OpenAiEmbeddingProvider("text-embedding-3-large", {"env": env}),
        "gradingProvider": OpenAiChatCompletionProvider(DEFAULT_OPENAI_MODEL, {"env": env}),
        "gradingJsonProvider": OpenAiChatCompletionProvider(
            DEFAULT_OPENAI_MODEL,
            {"env": env, "config": {"response_format": {"type": "json_object"}}},
        ),
        "suggestionsProvider": OpenAiChatCompletionProvider(DEFAULT_OPENAI_MODEL, {"env": env}),
        "moderationProvider": OpenAiModerationProvider("omni-moderation-latest", {"env": env}),
        "webSearchProvider": OpenAiResponsesProvider(
            DEFAULT_OPENAI_MODEL,
            {"env": env, "config": {"tools": [{"type": "web_search_preview"}]}},
        ),
    }


_defaults = _create_providers()
DefaultEmbeddingProvider = _defaults["embeddingProvider"]
DefaultGradingProvider = _defaults["gradingProvider"]
DefaultGradingJsonProvider = _defaults["gradingJsonProvider"]
DefaultSuggestionsProvider = _defaults["suggestionsProvider"]
DefaultModerationProvider = _defaults["moderationProvider"]
DefaultWebSearchProvider = _defaults["webSearchProvider"]


def get_openai_providers(env: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Bind explicit helper overrides to the providers used after selection."""
    return _create_providers(env) if env else dict(_defaults)
