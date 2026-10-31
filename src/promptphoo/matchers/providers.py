# SHIM: replaced by the grading-infrastructure branch at merge
"""Grading provider selection and calls. Ports src/matchers/providers.ts (minimal subset)."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from promptphoo import cli_state
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.base import ApiProvider

logger = logging.getLogger(__name__)


def _is_provider(value: Any) -> bool:
    return isinstance(value, ApiProvider) or callable(getattr(value, "id", None))


def _is_simulated_user_provider_config(provider: Any) -> bool:
    if isinstance(provider, str):
        return provider == "promptfoo:simulated-user"
    if _is_provider(provider):
        return provider.id() == "promptfoo:simulated-user"
    if not isinstance(provider, dict):
        return False
    if isinstance(provider.get("id"), str):
        return provider["id"] == "promptfoo:simulated-user"
    return any(_is_simulated_user_provider_config(v) for v in provider.values())


async def _load(provider_id: str, **kw: Any) -> ApiProvider:
    from promptphoo.providers import load_api_provider

    return await load_api_provider(provider_id, base_path=cli_state.base_path(), **kw)


async def get_grading_provider(type: str, provider: Any, default_provider: Any) -> Any:  # noqa: A002
    if isinstance(provider, str):
        return await _load(provider)
    if provider is not None and _is_provider(provider):
        return provider
    if isinstance(provider, list):
        raise ValueError(
            "Provider must be an object or string, but received an array.\n\nCheck that the "
            f"provider {json_stringify(provider[0], 2)} is not nested in an array."
        )
    if isinstance(provider, dict):
        type_value = provider.get(type)
        if type_value:
            if isinstance(type_value, str):
                return await _load(type_value, env=cli_state.env())
            if isinstance(type_value, dict) and isinstance(type_value.get("id"), str):
                return await _load(type_value["id"], options=type_value, env=cli_state.env())
            return await get_grading_provider(type, type_value, default_provider)
        if provider.get("id"):
            return await _load(provider["id"], options=provider)
        raise ValueError(
            f"Invalid provider definition for output type '{type}': {json_stringify(provider, 2)}"
        )
    default_test = (cli_state.config() or {}).get("defaultTest")
    dt = default_test if isinstance(default_test, dict) else None
    dt_options = (dt or {}).get("options") or {}
    dt_provider = dt_options.get("provider")
    candidates = [
        (dt or {}).get("provider") or None,
        (dt_provider.get("text") if isinstance(dt_provider, dict) else None) or None,
        dt_provider or None,
    ]
    for cfg in candidates:
        if not cfg:
            continue
        if _is_simulated_user_provider_config(cfg):
            logger.debug(
                "[Grading] Skipping promptfoo:simulated-user as an implicit grader fallback"
            )
            continue
        return await get_grading_provider(type, cfg, default_provider)
    return default_provider


async def get_and_check_provider(
    type: str,  # noqa: A002
    provider: Any,
    default_provider: Any,
    check_name: str,
) -> Any:
    matched = await get_grading_provider(type, provider, default_provider)
    if not matched:
        if default_provider:
            logger.warning("[Grading] Falling back to default provider")
            return default_provider
        raise ValueError(f"No provider of type {type} found for '{check_name}'")

    valid = True
    if type == "embedding":
        valid = hasattr(matched, "call_embedding_api") or hasattr(matched, "call_similarity_api")
    elif type == "classification":
        valid = hasattr(matched, "call_classification_api")
    elif type == "moderation":
        valid = hasattr(matched, "call_moderation_api")
    if not valid:
        if provider:
            raise ValueError(
                f"Provider {matched.id()} is not a valid {type} provider for '{check_name}'"
            )
        if default_provider:
            logger.warning("[Grading] Falling back to default provider after type check failed")
            return default_provider
        raise ValueError(
            f"Provider {matched.id()} is not a valid {type} provider for '{check_name}'"
        )
    return matched


async def call_grading_provider(
    provider: Any,
    label: str,
    invoke: Callable[[dict[str, Any] | None], Awaitable[dict[str, Any]]],
    *,
    call_context: dict[str, Any] | None = None,
    operation_name: str | None = None,
) -> dict[str, Any]:
    return await invoke(call_context)


async def call_provider_with_context(
    provider: Any,
    prompt: str,
    label: str,
    vars: dict[str, Any],  # noqa: A002
    context: dict[str, Any] | None = None,
    prompt_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    call_api_context = {
        **(context or {}),
        "isGrading": True,
        "prompt": {
            "raw": prompt,
            "label": label,
            **({"config": prompt_config} if prompt_config else {}),
        },
        "vars": vars,
    }

    async def invoke(traced: dict[str, Any] | None) -> dict[str, Any]:
        return await provider.call_api(prompt, traced)

    return await call_grading_provider(provider, label, invoke, call_context=call_api_context)
