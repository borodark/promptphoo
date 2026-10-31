"""OpenAI provider family. `create_provider` ports the `openai:` factory in
src/providers/registry.ts.

Supported: chat, completion, embedding(s), moderation and bare `openai:<model>` ids that route
to them. Responses arrives in a later wave; assistants, realtime, image, tts, transcription,
video, live, decisions, agents and codex are not supported by promptphoo.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from promptphoo.providers.openai.chat import OpenAiChatCompletionProvider
from promptphoo.providers.openai.completion import OpenAiCompletionProvider
from promptphoo.providers.openai.embedding import OpenAiEmbeddingProvider
from promptphoo.providers.openai.generic import (
    OpenAiGenericProvider,
    has_header_override,
    resolve_openai_api_url,
)
from promptphoo.providers.openai.moderation import OpenAiModerationProvider
from promptphoo.providers.openai.tables import OPENAI_RESPONSES_MODEL_NAMES
from promptphoo.providers.openai.util import (
    OPENAI_DAYBREAK_ALIASES,
    OPENAI_REALTIME_MODELS,
    OPENAI_TTS_MODELS,
    assert_openai_api_model,
    get_retired_openai_model_route,
    is_openai_first_party_api_url,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OpenAiChatCompletionProvider",
    "OpenAiCompletionProvider",
    "OpenAiEmbeddingProvider",
    "OpenAiGenericProvider",
    "OpenAiModerationProvider",
    "create_provider",
    "has_header_override",
    "resolve_openai_api_url",
]

OPENAI_TTS_MODEL_NAMES = [m["id"] for m in OPENAI_TTS_MODELS]
OPENAI_REALTIME_MODEL_NAMES = [m["id"] for m in OPENAI_REALTIME_MODELS]

_CONFIG_MODEL_OVERRIDE_ROUTES = frozenset(["image", "video"])
_PASSTHROUGH_MODEL_ROUTES = frozenset(
    ["chat", "completion", "embedding", "embeddings", "responses", "speech", "tts"]
)
_FIXED_MODEL_ROUTES = frozenset(["moderation", "realtime", "transcription"])
_CODEX_TYPES = ("codex-app-server", "codex-desktop", "codex-sdk", "codex")


def _unsupported(provider_path: str, kind: str) -> ValueError:
    return ValueError(
        f"OpenAI {kind} providers are not supported by promptphoo ({provider_path}). "
        "Use openai:chat:<model>, openai:completion:<model>, openai:embedding:<model> or "
        "openai:moderation:<model>."
    )


def _responses_not_implemented(provider_path: str) -> NotImplementedError:
    return NotImplementedError(
        f"The OpenAI Responses API provider ({provider_path}) is not ported yet; it arrives in "
        "Phase 3 wave 2. Use openai:chat:<model> meanwhile."
    )


def _get_configured_openai_model(options: dict[str, Any]) -> str | None:
    model = (options.get("config") or {}).get("model")
    return model if isinstance(model, str) and model.strip() else None


def should_default_to_openai_responses(model_name: str) -> bool:
    # Some compatible gateways spell the legacy GPT-3.5 family as GPT-35.
    if re.match(r"^gpt-35(?:-|$)", model_name):
        return False
    version = re.match(r"^gpt-(\d+)(?:\.(\d+))?(?:-|$)", model_name)
    if not version:
        return False
    major, minor = int(version.group(1)), int(version.group(2) or 0)
    return major > 5 or (major == 5 and minor >= 6)


def get_effective_openai_api_model(
    model_type: str, model_name: str, configured_model: str | None, passthrough_model: Any
) -> str:
    if model_type in _CONFIG_MODEL_OVERRIDE_ROUTES:
        return configured_model or model_name or model_type
    retired_route = get_retired_openai_model_route(model_type)
    explicit_uses_configured = (
        model_type in _PASSTHROUGH_MODEL_ROUTES or model_type in _FIXED_MODEL_ROUTES
    )
    provider_overrides_model = (
        model_type in ("speech", "tts")
        or model_type in OPENAI_TTS_MODEL_NAMES
        or retired_route == "tts"
    )
    selected = (
        configured_model or model_name or model_type
        if provider_overrides_model
        else model_name or (configured_model if explicit_uses_configured else None) or model_type
    )
    if model_type in _FIXED_MODEL_ROUTES or model_type == "gpt-transcribe":
        return selected
    chat_names = OpenAiChatCompletionProvider.OPENAI_CHAT_MODEL_NAMES
    completion_names = OpenAiCompletionProvider.OPENAI_COMPLETION_MODEL_NAMES
    bare_uses_passthrough = not model_name and (
        should_default_to_openai_responses(model_type)
        or model_type in chat_names
        or model_type in completion_names
        or model_type in OPENAI_RESPONSES_MODEL_NAMES
        or model_type in OPENAI_TTS_MODEL_NAMES
        or retired_route in ("chat", "responses", "tts")
        or (
            not retired_route
            and model_type not in completion_names
            and model_type not in OPENAI_TTS_MODEL_NAMES
            and model_type not in OPENAI_REALTIME_MODEL_NAMES
        )
    )
    route_uses_passthrough = model_type in _PASSTHROUGH_MODEL_ROUTES or bare_uses_passthrough
    return (
        passthrough_model
        if route_uses_passthrough and isinstance(passthrough_model, str)
        else selected
    )


def create_provider(provider_path: str, options: dict[str, Any]) -> Any:
    """Build the provider for an `openai:<type>:<model>` (or bare `openai:<model>`) id."""
    options = dict(options or {})
    config = options.get("config") or {}
    env = options.get("env")
    splits = provider_path.split(":")
    model_type = splits[1] if len(splits) > 1 else ""
    model_name = ":".join(splits[2:])
    if model_type == "chatkit":
        raise ValueError(
            "The openai:chatkit provider has been removed. Export your Agent Builder workflow "
            "and evaluate it with the OpenAI Agents SDK, a custom provider, or an HTTP endpoint. "
            "Migration guide: https://www.promptfoo.dev/docs/providers/openai-chatkit/"
        )
    configured_model = _get_configured_openai_model(options)
    assistant_model = None
    if model_type == "assistant" and isinstance(config.get("modelName"), str):
        assistant_model = config["modelName"].strip() or None
    passthrough = config.get("passthrough")
    passthrough_model = passthrough.get("model") if isinstance(passthrough, dict) else None
    effective_api_model = get_effective_openai_api_model(
        model_type, model_name, configured_model, passthrough_model
    )
    allow_transcription = model_type in ("gpt-transcribe", "transcription")

    if model_type == "agents-api" or model_type in _CODEX_TYPES:
        raise _unsupported(provider_path, "Agents/Codex")
    requested_api_model = model_name or configured_model or model_type
    if any(
        isinstance(m, str) and (m == "gpt-live-transcribe" or m.startswith("gpt-live-transcribe-"))
        for m in (model_type, requested_api_model, passthrough_model)
    ):
        raise ValueError(
            "gpt-live-transcribe requires Realtime transcription sessions, which are not yet "
            "supported by promptfoo."
        )
    is_live = model_type == "live" or bool(
        re.match(r"^gpt-live-1(?:-\d{4}-\d{2}-\d{2})?$", model_type)
    )
    if not is_live and model_type != "agents" and (model_type != "assistant" or assistant_model):
        api_url = resolve_openai_api_url(config, env)
        assert_openai_api_model(
            assistant_model or effective_api_model, api_url, allow_transcription
        )

    if model_type == "chat":
        return OpenAiChatCompletionProvider(model_name or configured_model or "gpt-6-sol", options)
    if model_type in ("embedding", "embeddings"):
        return OpenAiEmbeddingProvider(
            model_name or configured_model or "text-embedding-3-large", options
        )
    if model_type == "completion":
        return OpenAiCompletionProvider(
            model_name or configured_model or "gpt-3.5-turbo-instruct", options
        )
    if model_type == "moderation":
        return OpenAiModerationProvider(
            model_name or configured_model or "omni-moderation-latest", options
        )
    if model_type == "realtime":
        raise _unsupported(provider_path, "Realtime")
    if model_type == "responses":
        raise _responses_not_implemented(provider_path)
    if model_type == "decisions":
        raise _unsupported(provider_path, "Decisions")
    if model_type == "transcription":
        raise _unsupported(provider_path, "transcription")
    if model_type in ("tts", "speech"):
        raise _unsupported(provider_path, "text-to-speech")
    if is_live:
        raise _unsupported(provider_path, "GPT-Live")
    if should_default_to_openai_responses(model_type) or (
        model_type in OPENAI_DAYBREAK_ALIASES
        and is_openai_first_party_api_url(resolve_openai_api_url(config, env))
    ):
        raise _responses_not_implemented(provider_path)
    if model_type in OpenAiChatCompletionProvider.OPENAI_CHAT_MODEL_NAMES:
        return OpenAiChatCompletionProvider(model_type, options)
    if model_type in OpenAiCompletionProvider.OPENAI_COMPLETION_MODEL_NAMES:
        return OpenAiCompletionProvider(model_type, options)
    if model_type in OPENAI_TTS_MODEL_NAMES:
        raise _unsupported(provider_path, "text-to-speech")
    if model_type == "gpt-transcribe":
        raise _unsupported(provider_path, "transcription")
    if model_type in OPENAI_REALTIME_MODEL_NAMES:
        raise _unsupported(provider_path, "Realtime")
    if model_type in OPENAI_RESPONSES_MODEL_NAMES:
        raise _responses_not_implemented(provider_path)
    # Retired IDs reach here only for custom OpenAI-compatible gateways that still serve them.
    route = get_retired_openai_model_route(model_type)
    if route == "chat":
        return OpenAiChatCompletionProvider(model_type, options)
    if route == "moderation":
        return OpenAiModerationProvider(model_type, options)
    if route == "responses":
        raise _responses_not_implemented(provider_path)
    if route in ("tts", "realtime"):
        raise _unsupported(provider_path, "text-to-speech" if route == "tts" else "Realtime")
    if model_type in ("agents", "assistant", "image", "video"):
        kind = {"agents": "Agents", "assistant": "Assistants", "image": "image",
                "video": "video"}[model_type]  # fmt: skip
        raise _unsupported(provider_path, kind)
    logger.warning(
        "Unknown OpenAI model type: %s. Treating it as a chat model. Use one of the following "
        "providers: openai:chat:<model name>, openai:completion:<model name>, "
        "openai:embeddings:<model name>, openai:image:<model name>, openai:video:<model name>, "
        "openai:tts:<model name>, openai:transcription:<model name>, "
        "openai:realtime:<model name>, openai:live:<model name>, openai:agents:<agent name>, "
        "openai:codex-sdk",
        model_type,
    )
    return OpenAiChatCompletionProvider(model_type, options)
