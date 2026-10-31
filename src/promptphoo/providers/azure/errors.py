"""Error-classification helpers for the Azure providers. Ports src/providers/azure/errors.ts."""

from __future__ import annotations

from promptphoo.providers.base import ProviderResponse

CONTENT_FILTER_FALLBACK_MESSAGE = (
    "The generated content was filtered due to triggering Azure OpenAI Service's content "
    "filtering system."
)


def is_content_filter_error(error_message: str) -> bool:
    lower = error_message.lower()
    return any(
        needle in lower
        for needle in (
            "content_filter",
            "content filter",
            "filtered due to",
            "content filtering",
            "inappropriate content",
            "safety guidelines",
            "guardrail",
        )
    )


def is_rate_limit_error(error_message: str) -> bool:
    lower = error_message.lower()
    return (
        "rate limit" in lower
        or "quota exceeded" in lower
        or "too many requests" in lower
        or "429" in error_message
    )


def is_service_error(error_message: str) -> bool:
    return any(
        needle in error_message
        for needle in (
            "Service unavailable",
            "Bad gateway",
            "Gateway timeout",
            "Server is busy",
            "Sorry, something went wrong",
        )
    )


def format_content_filter_response(error_message: str) -> ProviderResponse:
    """Input-side filtering wins when the message names the prompt or input."""
    lower = error_message.lower()
    flagged_input = "prompt" in lower or "input" in lower
    return {
        "output": CONTENT_FILTER_FALLBACK_MESSAGE,
        "guardrails": {
            "flagged": True,
            "flaggedInput": flagged_input,
            "flaggedOutput": not flagged_input,
        },
    }
