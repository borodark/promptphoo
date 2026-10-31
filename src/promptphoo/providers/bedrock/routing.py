"""Pure Bedrock routing helpers. Ports src/providers/bedrock/routing.ts."""

from __future__ import annotations

import re
from typing import Literal

BedrockApiMode = Literal["invoke", "converse", "responses", "chat", "messages"]


def is_rejected_prefixed_mythos_id(model_name: str) -> bool:
    return re.match(r"^[^.]+\.(anthropic\.claude-mythos-(?:5|preview))$", model_name) is not None


# Grok 4.6 profiles are served natively; all inference profiles are invalid Mantle IDs.
NATIVE_GROK_PROFILE_MODELS = {"us.xai.grok-4.6", "global.xai.grok-4.6"}


def is_rejected_prefixed_grok_id(model_name: str, explicit_mantle_request: bool) -> bool:
    return ".xai." in model_name and (
        explicit_mantle_request or model_name not in NATIVE_GROK_PROFILE_MODELS
    )


RUNTIME_MESSAGES_MODELS = {
    "us.anthropic.claude-fable-5-1",
    "global.anthropic.claude-fable-5-1",
    "us.anthropic.claude-mythos-5-1",
    "global.anthropic.claude-mythos-5-1",
}


def is_bedrock_runtime_messages_model(model_name: str) -> bool:
    return model_name in RUNTIME_MESSAGES_MODELS


def is_bedrock_anthropic_messages_model(model_name: str) -> bool:
    return model_name in (
        "anthropic.claude-fable-5",
        "anthropic.claude-mythos-5",
        "anthropic.claude-fable-5-1",
        "anthropic.claude-mythos-preview",
        "anthropic.claude-opus-4-7",
        "anthropic.claude-opus-4-8",
        "anthropic.claude-opus-5",
        "anthropic.claude-sonnet-5",
    ) or is_bedrock_runtime_messages_model(model_name)


def is_bedrock_openai_responses_model(model_name: str) -> bool:
    return model_name.startswith("openai.") and "gpt-oss" not in model_name


def is_bedrock_gpt_oss_responses_model(model_name: str) -> bool:
    return re.match(r"^openai\.gpt-oss-(?:20b|120b)$", model_name) is not None


def is_bedrock_grok_model(model_name: str) -> bool:
    return model_name.startswith("xai.grok-")


def is_bedrock_mantle_responses_model(model_name: str) -> bool:
    return is_bedrock_openai_responses_model(model_name) or is_bedrock_grok_model(model_name)


def requires_bedrock_anthropic_messages_model(model_name: str) -> bool:
    return model_name in ("anthropic.claude-mythos-5", "anthropic.claude-mythos-preview")


_EXPLICIT_MODES: dict[str, BedrockApiMode] = {
    "completion": "invoke",
    "converse": "converse",
    "responses": "responses",
    "mantle": "chat",
    "messages": "messages",
}


def get_bedrock_text_route(id: str) -> dict[str, str] | None:
    """Resolve text API aliases ({apiMode, modelId}), not model availability."""
    if not id.startswith("bedrock:"):
        return None
    rest = id[len("bedrock:") :]
    subtype, *parts = rest.split(":")
    explicit_mode = _EXPLICIT_MODES.get(subtype)
    model_id = ":".join(parts) if explicit_mode else rest
    if explicit_mode in ("responses", "chat", "messages"):
        return {"apiMode": explicit_mode, "modelId": model_id}
    # Only bare IDs and legacy completion/converse aliases get automatic Responses routing.
    if (explicit_mode or not parts) and is_bedrock_mantle_responses_model(model_id):
        return {"apiMode": "responses", "modelId": model_id}
    if not explicit_mode and requires_bedrock_anthropic_messages_model(model_id):
        return {"apiMode": "messages", "modelId": model_id}
    if explicit_mode:
        return {"apiMode": explicit_mode, "modelId": model_id}
    # Specialized providers (kb, agents, embeddings, ...) are not text models.
    if parts and "." not in subtype and subtype != "arn":
        return None
    return {"apiMode": "invoke", "modelId": model_id}
