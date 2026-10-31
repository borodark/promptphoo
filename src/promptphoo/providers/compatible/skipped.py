"""Provider prefixes from the OpenAI-compatible survey that promptphoo does not port (see D31).

They either need a separate SDK (`vercel:` runs on the Vercel AI SDK) or implement their own
non-OpenAI request path upstream (`mistral:`, `ai21:`). Each fails with a clear error instead of
"Could not identify provider".
"""

from __future__ import annotations

from typing import Any

from promptphoo.providers.compatible.common import not_supported

_SKIPPED: dict[str, tuple[str, str]] = {
    "vercel": (
        "The Vercel AI Gateway provider, which runs on the Vercel AI SDK,",
        "Use openai:chat:<model> with apiBaseUrl https://ai-gateway.vercel.sh/v1 and "
        "apiKeyEnvar AI_GATEWAY_API_KEY instead.",
    ),
    "mistral": (
        "The Mistral provider",
        "Use openai:chat:<model> with apiBaseUrl https://api.mistral.ai/v1 and apiKeyEnvar "
        "MISTRAL_API_KEY instead.",
    ),
    "ai21": (
        "The AI21 provider",
        "Use openai:chat:<model> with apiBaseUrl https://api.ai21.com/studio/v1 and apiKeyEnvar "
        "AI21_API_KEY instead.",
    ),
}

SKIPPED_PREFIXES = tuple(f"{prefix}:" for prefix in _SKIPPED)


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    kind, hint = _SKIPPED[provider_path.split(":", 1)[0]]
    raise not_supported(provider_path, kind, hint)
