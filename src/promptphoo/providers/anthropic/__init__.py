"""Anthropic providers: `anthropic:messages:<model>`, `anthropic:completion:<model>` and the
`anthropic:<model>` shorthand. Ports the `anthropic:` factory of src/providers/registry.ts."""

from __future__ import annotations

from typing import Any

from promptphoo.providers.anthropic.completion import AnthropicCompletionProvider
from promptphoo.providers.anthropic.messages import AnthropicMessagesProvider
from promptphoo.providers.anthropic.util import (
    ANTHROPIC_SHORTHAND_MODEL_IDS,
    looks_like_claude_model_id,
)
from promptphoo.providers.shared import model_name_from_provider_path

__all__ = ["AnthropicCompletionProvider", "AnthropicMessagesProvider", "create_provider"]


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    if provider_path.startswith(("anthropic:claude-agent-sdk", "anthropic:claude-code")):
        raise ValueError(
            f"The Claude Agent SDK provider is not supported by promptphoo ({provider_path})."
        )
    splits = provider_path.split(":")
    model_type = splits[1] if len(splits) > 1 else ""
    model_name = model_name_from_provider_path(provider_path, 2)

    if model_type == "messages":
        return AnthropicMessagesProvider(model_name, options)
    if model_type == "completion":
        return AnthropicCompletionProvider(model_name, options)
    if model_type in AnthropicCompletionProvider.ANTHROPIC_COMPLETION_MODELS:
        return AnthropicCompletionProvider(model_name_from_provider_path(provider_path, 1), options)
    # A bare model name routes to the Messages API when it is catalogued or Claude-shaped.
    if model_type in ANTHROPIC_SHORTHAND_MODEL_IDS or looks_like_claude_model_id(model_type):
        return AnthropicMessagesProvider(model_name_from_provider_path(provider_path, 1), options)
    raise ValueError(
        f"Unknown Anthropic model type or model name: {model_type}. Use one of the following "
        "formats:\n"
        "- anthropic:messages:<model name> - For Messages API\n"
        "- anthropic:completion:<model name> - For Completion API\n"
        '- anthropic:<model name> - Shorthand for Messages API, for a model id starting with "claude-"'  # noqa: E501
    )
