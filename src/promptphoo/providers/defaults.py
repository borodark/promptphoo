# SHIM: replaced by the grading-infrastructure branch at merge
"""Default grading providers. Ports the OpenAI branch of src/providers/defaults.ts."""

from __future__ import annotations

from typing import Any

DEFAULT_OPENAI_MODEL = "gpt-6-sol"


async def get_default_providers(env: dict[str, Any] | None = None) -> dict[str, Any]:
    from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
    from promptphoo.providers.openai.embedding import OpenAiEmbeddingProvider
    from promptphoo.providers.openai.moderation import OpenAiModerationProvider
    from promptphoo.providers.openai.responses import OpenAiResponsesProvider

    opts: dict[str, Any] = {"env": env} if env else {}
    grading_json = OpenAiChatCompletionProvider(
        DEFAULT_OPENAI_MODEL, {**opts, "config": {"response_format": {"type": "json_object"}}}
    )
    return {
        "embeddingProvider": OpenAiEmbeddingProvider("text-embedding-3-large", dict(opts)),
        "gradingJsonProvider": grading_json,
        "gradingProvider": OpenAiChatCompletionProvider(DEFAULT_OPENAI_MODEL, dict(opts)),
        "moderationProvider": OpenAiModerationProvider("omni-moderation-latest", dict(opts)),
        "suggestionsProvider": OpenAiChatCompletionProvider(DEFAULT_OPENAI_MODEL, dict(opts)),
        "synthesizeProvider": grading_json,
        "webSearchProvider": OpenAiResponsesProvider(
            DEFAULT_OPENAI_MODEL, {**opts, "config": {"tools": [{"type": "web_search_preview"}]}}
        ),
    }
