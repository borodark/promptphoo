# SHIM: replaced by the grading-infrastructure branch at merge
"""Grading-provider resolution and calls. Ports src/matchers/providers.ts (minimal subset)."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from promptphoo import cli_state
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.base import ApiProvider

logger = logging.getLogger(__name__)

_TYPE_METHODS = {
    "embedding": ("call_embedding_api", "call_similarity_api"),
    "classification": ("call_classification_api",),
    "moderation": ("call_moderation_api",),
}


def _has(provider: Any, name: str) -> bool:
    return callable(getattr(provider, name, None))


def _is_provider(value: Any) -> bool:
    return isinstance(value, ApiProvider) or callable(getattr(value, "id", None))


def get_grading_provider_call_options() -> dict[str, Any] | None:
    from promptphoo.scheduler.provider_call_execution_context import (
        get_provider_call_execution_context,
    )

    ctx = get_provider_call_execution_context() or {}
    signal = ctx.get("abortSignal")
    return {"abortSignal": signal} if signal else None


async def call_embedding_provider(
    provider: Any, input: str, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    options = get_grading_provider_call_options()
    if (context or options) and getattr(provider, "supports_embedding_cancellation", False):
        return await provider.call_embedding_api(input, context, options)
    return await provider.call_embedding_api(input)


async def call_grading_provider(
    provider: Any,
    label: str,
    invoke: Callable[[dict[str, Any] | None], Awaitable[Any]],
    *,
    call_context: dict[str, Any] | None = None,
    operation_name: str | None = None,
) -> Any:
    from promptphoo.scheduler.provider_call_execution_context import (
        get_provider_call_execution_context,
        run_provider_call_with_abort,
    )

    ctx = get_provider_call_execution_context() or {}
    signal = ctx.get("abortSignal")
    if signal is not None:
        signal.throw_if_aborted()
    return await run_provider_call_with_abort(lambda: invoke(call_context), signal)


async def call_provider_with_context(
    provider: Any,
    prompt: str,
    label: str,
    vars: dict[str, Any],
    context: dict[str, Any] | None = None,
    prompt_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_info: dict[str, Any] = {"raw": prompt, "label": label}
    if prompt_config:
        prompt_info["config"] = prompt_config
    call_api_context = {**(context or {}), "isGrading": True, "prompt": prompt_info, "vars": vars}
    options = get_grading_provider_call_options()

    async def invoke(traced_context: dict[str, Any] | None) -> Any:
        if options:
            return await provider.call_api(prompt, traced_context, options)
        return await provider.call_api(prompt, traced_context)

    return await call_grading_provider(provider, label, invoke, call_context=call_api_context)


async def _load(provider_id: str, **kw: Any) -> ApiProvider:
    from promptphoo.providers import load_api_provider

    return await load_api_provider(provider_id, base_path=cli_state.base_path(), **kw)


async def get_grading_provider(type: str, provider: Any, default_provider: Any) -> Any:
    if isinstance(provider, str):
        return await _load(provider)
    if provider is not None and not isinstance(provider, dict | list) and _is_provider(provider):
        return provider
    if isinstance(provider, list):
        first = json_stringify(provider[0] if provider else None, 2)
        raise ValueError(
            "Provider must be an object or string, but received an array.\n\n"
            f"Check that the provider {first} is not nested in an array."
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
    config = cli_state.config() or {}
    default_test = config.get("defaultTest")
    default_test = default_test if isinstance(default_test, dict) else {}
    options = default_test.get("options") or {}
    opt_provider = options.get("provider")
    candidates = [
        default_test.get("provider") or None,
        (opt_provider.get("text") if isinstance(opt_provider, dict) else None) or None,
        opt_provider or None,
    ]
    for candidate in candidates:
        if not candidate or _is_simulated_user(candidate):
            continue
        return await get_grading_provider(type, candidate, default_provider)
    return default_provider


def _is_simulated_user(provider: Any) -> bool:
    if isinstance(provider, str):
        return provider == "promptfoo:simulated-user"
    if _is_provider(provider) and not isinstance(provider, dict):
        return provider.id() == "promptfoo:simulated-user"
    if not isinstance(provider, dict):
        return False
    if isinstance(provider.get("id"), str):
        return provider["id"] == "promptfoo:simulated-user"
    return any(_is_simulated_user(v) for v in provider.values())


async def get_and_check_provider(
    type: str,
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

    methods = _TYPE_METHODS.get(type)
    if methods and not any(_has(matched, m) for m in methods):
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
