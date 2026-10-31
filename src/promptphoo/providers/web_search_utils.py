"""Web-search capability detection. Ports src/providers/webSearchUtils.ts."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_SEARCH_CHAT_MODEL = re.compile(
    r"(?:^|[/:])(?:gpt-5-search-api|gpt-4o(?:-mini)?-search-preview)(?:-|$)"
)
_ANTHROPIC_WEB_SEARCH = re.compile(r"^web_search_[0-9]{8}$")


def _config(provider: Any) -> dict[str, Any]:
    config = getattr(provider, "config", None)
    return config if isinstance(config, dict) else {}


def _has_tool(provider: Any, predicate: Any) -> bool:
    tools = _config(provider).get("tools")
    return isinstance(tools, list) and any(isinstance(t, dict) and predicate(t) for t in tools)


def _get_provider_id(provider: Any) -> str | None:
    id_fn = getattr(provider, "id", None)
    if not callable(id_fn):
        return None
    try:
        value = id_fn()
        return value if isinstance(value, str) else None
    except Exception as err:
        logger.debug("Failed to read provider id: %s", err)
        return None


def _class_name(provider: Any) -> str:
    return type(provider).__name__


def _is_openai_responses(provider: Any, id: str) -> bool:
    return "openai:responses" in id or _class_name(provider) == "OpenAiResponsesProvider"


def _is_xai_responses(provider: Any, id: str) -> bool:
    return "xai:responses" in id or _class_name(provider) == "XAIResponsesProvider"


def has_web_search_capability(provider: Any) -> bool:
    if not provider:
        return False
    id = _get_provider_id(provider)
    if not id:
        return False
    config = _config(provider)

    # Perplexity has built-in web search.
    if "perplexity" in id:
        return True
    if ("google" in id or "gemini" in id or "vertex" in id) and _has_tool(
        provider, lambda t: "googleSearch" in t
    ):
        return True
    search_parameters = config.get("search_parameters")
    if "xai" in id and (
        (isinstance(search_parameters, dict) and search_parameters.get("mode") == "on")
        or (
            _is_xai_responses(provider, id)
            and _has_tool(provider, lambda t: t.get("type") == "web_search")
        )
    ):
        return True
    if _is_openai_responses(provider, id) and _has_tool(
        provider, lambda t: t.get("type") in ("web_search", "web_search_preview")
    ):
        return True

    # Chat Completions search models always retrieve from the web before responding.
    is_openai_chat = not _is_openai_responses(provider, id) and (
        id.startswith("openai:chat:") or _class_name(provider) == "OpenAiChatCompletionProvider"
    )
    passthrough = config.get("passthrough")
    passthrough_model = passthrough.get("model") if isinstance(passthrough, dict) else None
    model_name = getattr(provider, "model_name", None)
    if isinstance(passthrough_model, str):
        chat_model_id = passthrough_model
    elif isinstance(model_name, str):
        chat_model_id = model_name
    else:
        chat_model_id = id
    if is_openai_chat and _SEARCH_CHAT_MODEL.search(chat_model_id):
        return True

    # Codex SDK supports web search when explicitly enabled.
    if id.startswith("openai:codex") and (
        config.get("web_search_mode") in ("live", "cached")
        or config.get("web_search_enabled") is True
    ):
        return True

    # Anthropic with any version of its dated `web_search_YYYYMMDD` server tool.
    return "anthropic" in id and _has_tool(
        provider,
        lambda t: isinstance(t.get("type"), str) and bool(_ANTHROPIC_WEB_SEARCH.match(t["type"])),
    )


async def load_web_search_provider(prefer_anthropic: bool = False) -> Any:
    """Load the first provider with web search, trying several families in order."""
    from promptphoo.providers import load_api_provider

    async def load(name: str, provider_id: str, options: dict[str, Any] | None = None) -> Any:
        try:
            return await load_api_provider(provider_id, options=options)
        except Exception as err:
            logger.debug("Failed to load %s provider: %s", name, err)
            return None

    anthropic = (
        "Anthropic web search",
        "anthropic:messages:claude-opus-5-5",
        {
            "config": {
                "tools": [{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}]
            }
        },
    )
    openai = (
        "OpenAI web search",
        "openai:responses:gpt-6-sol",
        {"config": {"tools": [{"type": "web_search_preview"}]}},
    )
    candidates = [
        anthropic if prefer_anthropic else openai,
        openai if prefer_anthropic else anthropic,
        ("Perplexity", "perplexity:sonar-pro", None),
        (
            "Google web search",
            "google:gemini-3.1-pro-preview",
            {"config": {"tools": [{"googleSearch": {}}]}},
        ),
        (
            "Vertex web search",
            "vertex:gemini-3.1-pro-preview",
            {"config": {"tools": [{"googleSearch": {}}]}},
        ),
        (
            "xAI web search",
            "xai:responses:grok-4.3",
            {"config": {"tools": [{"type": "web_search"}]}},
        ),
    ]
    for name, provider_id, options in candidates:
        provider = await load(name, provider_id, options)
        if provider and has_web_search_capability(provider):
            logger.info(
                "Using %s as web search provider", _get_provider_id(provider) or "loaded provider"
            )
            return provider
        if provider:
            logger.debug(
                "Loaded provider %s does not support web search",
                _get_provider_id(provider) or "unknown",
            )
    return None
