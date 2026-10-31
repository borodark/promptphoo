# SHIM: replaced by the grading-infrastructure branch at merge
"""Default grading providers. Ports src/providers/defaults.ts (OpenAI defaults only)."""

from __future__ import annotations

from typing import Any

DEFAULT_OPENAI_MODEL = "gpt-6-sol"


async def get_default_providers(env: dict[str, Any] | None = None) -> dict[str, Any]:
    from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider

    return {
        "gradingProvider": OpenAiChatCompletionProvider(DEFAULT_OPENAI_MODEL, {"env": env}),
        "gradingJsonProvider": OpenAiChatCompletionProvider(
            DEFAULT_OPENAI_MODEL,
            {"env": env, "config": {"response_format": {"type": "json_object"}}},
        ),
        "suggestionsProvider": OpenAiChatCompletionProvider(DEFAULT_OPENAI_MODEL, {"env": env}),
    }
