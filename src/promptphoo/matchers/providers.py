# SHIM: replaced by the grading-infrastructure branch at merge
"""Grading-provider resolution and calls. Ports src/matchers/providers.ts (the subset
comparison.py needs: no tracing, provider-call queue or remote grading)."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from promptphoo import cli_state
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.base import ApiProvider, is_api_provider

logger = logging.getLogger(__name__)


def _is_simulated_user_provider_config(provider: Any) -> bool:
    if isinstance(provider, str):
        return provider == "promptfoo:simulated-user"
    if isinstance(provider, ApiProvider):
        return provider.id() == "promptfoo:simulated-user"
    if not provider or not isinstance(provider, dict):
        return False
    if isinstance(provider.get("id"), str):
        return provider["id"] == "promptfoo:simulated-user"
    return any(_is_simulated_user_provider_config(v) for v in provider.values())


async def get_grading_provider(
    type: str,  # noqa: A002 - Node's parameter name
    provider: Any,
    default_provider: ApiProvider | None,
) -> ApiProvider | None:
    from promptphoo.providers import load_api_provider

    if isinstance(provider, str):
        return await load_api_provider(provider, base_path=cli_state.base_path())
    if is_api_provider(provider):
        return provider
    if isinstance(provider, list):
        raise ValueError(
            "Provider must be an object or string, but received an array.\n\n"
            f"Check that the provider {json_stringify(provider[0], 2)} is not nested in an array."
        )
    if isinstance(provider, dict):
        type_value = provider.get(type)
        if type_value:
            if isinstance(type_value, str):
                return await load_api_provider(
                    type_value, base_path=cli_state.base_path(), env=cli_state.env()
                )
            if isinstance(type_value, dict) and isinstance(type_value.get("id"), str):
                return await load_api_provider(
                    type_value["id"],
                    options=type_value,
                    base_path=cli_state.base_path(),
                    env=cli_state.env(),
                )
            return await get_grading_provider(type, type_value, default_provider)
        if provider.get("id"):
            return await load_api_provider(
                provider["id"], options=provider, base_path=cli_state.base_path()
            )
        raise ValueError(
            f"Invalid provider definition for output type '{type}': {json_stringify(provider, 2)}"
        )
    # No provider specified - check defaultTest providers as fallback
    default_test = (cli_state.config() or {}).get("defaultTest")
    default_obj = default_test if isinstance(default_test, dict) else None
    options_provider = ((default_obj or {}).get("options") or {}).get("provider")
    fallback = [
        (default_obj or {}).get("provider") or None,
        (options_provider.get("text") if isinstance(options_provider, dict) else None) or None,
        options_provider or None,
    ]
    for candidate in fallback:
        if not candidate:
            continue
        if _is_simulated_user_provider_config(candidate):
            logger.debug(
                "[Grading] Skipping promptfoo:simulated-user as an implicit grader fallback"
            )
            continue
        return await get_grading_provider(type, candidate, default_provider)
    return default_provider


async def get_and_check_provider(
    type: str,  # noqa: A002 - Node's parameter name
    provider: Any,
    default_provider: ApiProvider | None,
    check_name: str,
) -> ApiProvider:
    matched = await get_grading_provider(type, provider, default_provider)
    if not matched:
        if default_provider:
            logger.warning("[Grading] Falling back to default provider (%s, %s)", check_name, type)
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
            return default_provider
        raise ValueError(
            f"Provider {matched.id()} is not a valid {type} provider for '{check_name}'"
        )
    return matched


async def call_grading_provider(
    provider: ApiProvider,
    label: str,
    invoke: Callable[[dict[str, Any] | None], Awaitable[dict[str, Any]]],
    call_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from promptphoo.scheduler.provider_call_execution_context import (
        get_provider_call_execution_context,
    )

    execution = get_provider_call_execution_context() or {}
    registry = execution.get("rateLimitRegistry")
    if registry is not None:
        from promptphoo.scheduler import create_provider_rate_limit_options

        return await registry.execute(
            provider,
            lambda _headers=None: invoke(call_context),
            create_provider_rate_limit_options(execution.get("abortSignal")),
        )
    return await invoke(call_context)


async def call_provider_with_context(
    provider: ApiProvider,
    prompt: str,
    label: str,
    vars: dict[str, Any],  # noqa: A002 - Node's parameter name
    context: dict[str, Any] | None = None,
    prompt_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    call_api_context = {
        **(context or {}),
        "isGrading": True,
        "prompt": {"raw": prompt, "label": label, **({"config": prompt_config}
                                                       if prompt_config else {})},
        "vars": vars,
    }  # fmt: skip
    return await call_grading_provider(
        provider,
        label,
        lambda ctx: provider.call_api(prompt, ctx),
        call_api_context,
    )
