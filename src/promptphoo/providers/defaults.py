# SHIM: replaced by the grading-infrastructure branch at merge
"""Default grading providers. Ports src/providers/defaults.ts (OpenAI branch only) and
src/providers/openai/defaults.ts."""

from __future__ import annotations

from typing import Any

DEFAULT_OPENAI_MODEL = "gpt-6-sol"


def get_openai_providers(env: dict[str, Any] | None = None) -> dict[str, Any]:
    from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
    from promptphoo.providers.openai.embedding import OpenAiEmbeddingProvider
    from promptphoo.providers.openai.moderation import OpenAiModerationProvider
    from promptphoo.providers.openai.responses import OpenAiResponsesProvider

    opts: dict[str, Any] = {"env": env} if env else {}
    return {
        "embeddingProvider": OpenAiEmbeddingProvider("text-embedding-3-large", {**opts}),
        "gradingProvider": OpenAiChatCompletionProvider(DEFAULT_OPENAI_MODEL, {**opts}),
        "gradingJsonProvider": OpenAiChatCompletionProvider(
            DEFAULT_OPENAI_MODEL, {**opts, "config": {"response_format": {"type": "json_object"}}}
        ),
        "suggestionsProvider": OpenAiChatCompletionProvider(DEFAULT_OPENAI_MODEL, {**opts}),
        "moderationProvider": OpenAiModerationProvider("omni-moderation-latest", {**opts}),
        "webSearchProvider": OpenAiResponsesProvider(
            DEFAULT_OPENAI_MODEL, {**opts, "config": {"tools": [{"type": "web_search_preview"}]}}
        ),
    }


async def get_default_providers(env: dict[str, Any] | None = None) -> dict[str, Any]:
    p = get_openai_providers(env)
    return {
        "embeddingProvider": p["embeddingProvider"],
        "gradingJsonProvider": p["gradingJsonProvider"],
        "gradingProvider": p["gradingProvider"],
        "moderationProvider": p["moderationProvider"],
        "suggestionsProvider": p["suggestionsProvider"],
        "synthesizeProvider": p["gradingJsonProvider"],
        "webSearchProvider": p["webSearchProvider"],
    }
