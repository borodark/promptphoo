"""GitHub Models provider, retired upstream. Ports src/providers/github/index.ts and
src/providers/github/defaults.ts (the module is named ghmodels to avoid the vendor name)."""

from __future__ import annotations

from typing import Any

from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider

GITHUB_MODELS_RETIREMENT_MESSAGE = (
    "GitHub Models was retired on July 30, 2026, including its inference API. "
    "Configure another provider with its own credentials. GitHub Copilot is a separate service. "
    "See https://docs.github.com/en/github-models"
)


def create_github_provider(provider_path: str, options: dict[str, Any], context: Any = None):
    raise ValueError(GITHUB_MODELS_RETIREMENT_MESSAGE)


class RetiredGitHubProvider(OpenAiChatCompletionProvider):
    """Kept for consumers of the old default providers; never calls the retired API."""

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        return {"error": GITHUB_MODELS_RETIREMENT_MESSAGE}


GITHUB_CONFIG = {"apiBaseUrl": "https://models.github.ai/inference", "apiKeyEnvar": "GITHUB_TOKEN"}


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """The `github:` registry entry."""
    return create_github_provider(provider_path, options)
