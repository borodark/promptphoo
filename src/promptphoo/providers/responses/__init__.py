"""Shared Responses API code (OpenAI, Azure, xAI). Ports src/providers/responses/index.ts."""

from promptphoo.providers.responses.input import normalize_responses_input
from promptphoo.providers.responses.processor import ResponsesProcessor, get_responses_token_usage
from promptphoo.providers.responses.stream import get_responses_output_text

__all__ = [
    "ResponsesProcessor",
    "get_responses_output_text",
    "get_responses_token_usage",
    "normalize_responses_input",
]
