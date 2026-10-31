"""Grading provider resolution and grader calls. Ports src/matchers/providers.ts.

Grader calls go through the evaluator's scheduler when one is active: the provider call queue
(grouped execution), the rate-limit registry (unless the provider is already rate-limit
wrapped), cancellation and tracing, all read from the runtime-only execution context.

Upstream also wraps each call in `providerRegistry.withProvider` (resource lifetime tracking
for providers that open long-lived resources); promptphoo has no provider registry, so that
layer is omitted. Remote (cloud) grading is out of scope (see D32).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from promptphoo import cli_state
from promptphoo.jsvalues import is_truthy, json_stringify
from promptphoo.providers.base import ApiProvider, is_api_provider
from promptphoo.scheduler.provider_call_execution_context import (
    get_provider_call_execution_context,
    get_provider_call_tracing_context,
    run_provider_call_with_abort,
)
from promptphoo.scheduler.provider_wrapper import (
    create_provider_rate_limit_options,
    is_rate_limit_wrapped,
)

logger = logging.getLogger(__name__)

ProviderType = Literal["text", "embedding", "classification", "moderation"]

SIMULATED_USER_PROVIDER_ID = "promptfoo:simulated-user"


def get_remote_grading_context() -> dict[str, Any]:
    """Cloud target ids feed remote grading only, which promptphoo does not do."""
    return {}


def should_use_remote_grading(options: Any = None) -> bool:
    """Remote (promptfoo cloud) grading is out of scope: graders always run locally."""
    return False


def get_grading_provider_call_options() -> dict[str, Any] | None:
    execution_context = get_provider_call_execution_context()
    abort_signal = execution_context.get("abortSignal") if execution_context else None
    return {"abortSignal": abort_signal} if abort_signal else None


async def call_embedding_provider(
    provider: Any, input: str, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    options = get_grading_provider_call_options()
    if (context or options) and getattr(provider, "supports_embedding_cancellation", False):
        result = await provider.call_embedding_api(input, context, options)
    else:
        result = await provider.call_embedding_api(input)
    # A provider that cannot observe cancellation may report it as an error response.
    if isinstance(result, dict) and result.get("error") and options:
        options["abortSignal"].throw_if_aborted()
    return result


async def call_grading_provider(
    provider: Any,
    label: str,
    invoke: Callable[..., Awaitable[Any]],
    *,
    call_context: dict[str, Any] | None = None,
    operation_name: str | None = None,
) -> Any:
    """Apply tracing, rate limits and grouped scheduling to every grading-provider modality.

    `invoke(context)` or `invoke(context, on_response_headers)` performs the provider call.
    """
    execution_context = get_provider_call_execution_context() or {}
    tracing_context = get_provider_call_tracing_context()
    abort_signal = execution_context.get("abortSignal")

    async def call_provider(on_response_headers: Any = None) -> Any:
        async def invoke_provider(context: dict[str, Any] | None) -> Any:
            if on_response_headers:
                return await invoke(context, on_response_headers)
            return await invoke(context)

        with_provider_span = tracing_context.get("withProviderSpan") if tracing_context else None
        if with_provider_span:
            span_options: dict[str, Any] = {
                "provider": provider,
                "callContext": call_context,
                "role": "grader",
                "promptLabel": label,
            }
            if operation_name is not None:
                span_options["operationName"] = operation_name
            return await with_provider_span(span_options, invoke_provider)
        return await invoke_provider(call_context)

    async def execute_call() -> Any:
        # Never start a grader after cancellation; queued graders check once they reach the front.
        if abort_signal is not None:
            abort_signal.throw_if_aborted()
        registry = execution_context.get("rateLimitRegistry")
        if registry and not is_rate_limit_wrapped(provider):
            return await registry.execute(
                provider, call_provider, create_provider_rate_limit_options(abort_signal)
            )
        return await call_provider()

    queue = execution_context.get("providerCallQueue")

    async def run() -> Any:
        if queue:
            return await queue.enqueue(provider.id(), execute_call, abort_signal)
        return await execute_call()

    return await run_provider_call_with_abort(run, abort_signal)


async def call_provider_with_context(
    provider: Any,
    prompt: str,
    label: str,
    vars: dict[str, Any],  # noqa: A002 - Node's parameter name
    context: dict[str, Any] | None = None,
    prompt_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve evaluator context while adding this grading call's prompt metadata and
    cancellation."""
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
    context_options = get_grading_provider_call_options()

    async def invoke(traced_context: Any, on_response_headers: Any = None) -> Any:
        call_api_options = (
            {**(context_options or {}), "onResponseHeaders": on_response_headers}
            if on_response_headers
            else context_options
        )
        if call_api_options:
            return await provider.call_api(prompt, traced_context, call_api_options)
        return await provider.call_api(prompt, traced_context)

    return await call_grading_provider(provider, label, invoke, call_context=call_api_context)


async def _load_api_provider(provider_path: str, **kwargs: Any) -> ApiProvider:
    # Resolved at call time so tests can substitute promptphoo.providers.load_api_provider.
    from promptphoo import providers

    return await providers.load_api_provider(provider_path, **kwargs)


async def _load_from_provider_options(provider: dict[str, Any]) -> ApiProvider:
    if not isinstance(provider, dict):
        raise ValueError(
            "Invariant failed: Provider must be an object, but received a "
            f"{type(provider).__name__}: {provider}"
        )
    if not provider.get("id"):
        raise ValueError("Invariant failed: Provider supplied to assertion must have an id")
    return await _load_api_provider(
        provider["id"], options=provider, base_path=cli_state.base_path()
    )


def _is_simulated_user_provider_config(provider: Any) -> bool:
    if isinstance(provider, str):
        return provider == SIMULATED_USER_PROVIDER_ID
    if is_api_provider(provider):
        return provider.id() == SIMULATED_USER_PROVIDER_ID
    if not isinstance(provider, dict) or not provider:
        return False
    provider_id = provider.get("id")
    if isinstance(provider_id, str):
        return provider_id == SIMULATED_USER_PROVIDER_ID
    return any(_is_simulated_user_provider_config(value) for value in provider.values())


async def get_grading_provider(
    type: str,  # noqa: A002 - Node's parameter name
    provider: Any,
    default_provider: ApiProvider | None,
) -> ApiProvider | None:
    final_provider: ApiProvider | None
    if isinstance(provider, str):
        # Defined as a string
        final_provider = await _load_api_provider(provider, base_path=cli_state.base_path())
    elif is_api_provider(provider):
        # Defined as an ApiProvider interface
        final_provider = provider
    elif isinstance(provider, dict | list):
        type_value = provider.get(type) if isinstance(provider, dict) else None
        if is_truthy(type_value):
            # Apply evaluation overrides only when the selected typed grader is loaded.
            # Capturing them in the test config would retain credentials across later runs.
            if isinstance(type_value, str):
                final_provider = await _load_api_provider(
                    type_value, base_path=cli_state.base_path(), env=cli_state.env()
                )
            elif isinstance(type_value, dict) and isinstance(type_value.get("id"), str):
                final_provider = await _load_api_provider(
                    type_value["id"],
                    options=type_value,
                    base_path=cli_state.base_path(),
                    env=cli_state.env(),
                )
            else:
                final_provider = await get_grading_provider(type, type_value, default_provider)
        elif isinstance(provider, dict) and is_truthy(provider.get("id")):
            # Defined as ProviderOptions
            final_provider = await _load_from_provider_options(provider)
        elif isinstance(provider, list):
            first = json_stringify(provider[0], indent=2) if provider else "undefined"
            raise ValueError(
                "Provider must be an object or string, but received an array.\n\n"
                f"Check that the provider {first} is not nested in an array."
            )
        else:
            raise ValueError(
                f"Invalid provider definition for output type '{type}': "
                f"{json_stringify(provider, indent=2)}"
            )
    else:
        # No provider specified - check defaultTest providers as fallback
        default_test = (cli_state.config() or {}).get("defaultTest")
        default_test_obj = default_test if isinstance(default_test, dict) else None
        options = (default_test_obj or {}).get("options") or {}
        options_provider = options.get("provider")
        fallback_providers = [
            (default_test_obj or {}).get("provider"),
            options_provider.get("text") if isinstance(options_provider, dict) else None,
            options_provider,
        ]
        cfg = None
        for candidate in fallback_providers:
            if not is_truthy(candidate):
                continue
            if _is_simulated_user_provider_config(candidate):
                logger.debug(
                    "[Grading] Skipping promptfoo:simulated-user as an implicit grader fallback"
                )
                continue
            cfg = candidate
            break

        if cfg is not None:
            # Recursively resolve all provider shapes (string, object, etc.)
            final_provider = await get_grading_provider(type, cfg, default_provider)
            if final_provider:
                logger.debug(
                    "[Grading] Using provider from defaultTest fallback: %s", final_provider.id()
                )
        else:
            final_provider = default_provider
    return final_provider


def _has_method(provider: Any, name: str) -> bool:
    return callable(getattr(provider, name, None))


async def get_and_check_provider(
    type: str,  # noqa: A002 - Node's parameter name
    provider: Any,
    default_provider: ApiProvider | None,
    check_name: str,
) -> ApiProvider:
    matched_provider = await get_grading_provider(type, provider, default_provider)
    if not matched_provider:
        if default_provider:
            logger.warning(
                "[Grading] Falling back to default provider (checkName=%s, type=%s)",
                check_name,
                type,
            )
            return default_provider
        raise ValueError(f"No provider of type {type} found for '{check_name}'")

    is_valid_provider_type = True
    if type == "embedding":
        is_valid_provider_type = _has_method(matched_provider, "call_embedding_api") or (
            _has_method(matched_provider, "call_similarity_api")
        )
    elif type == "classification":
        is_valid_provider_type = _has_method(matched_provider, "call_classification_api")
    elif type == "moderation":
        is_valid_provider_type = _has_method(matched_provider, "call_moderation_api")

    if not is_valid_provider_type:
        # An explicitly configured provider of the wrong type is an error, rather than silently
        # falling back to a different model.
        if is_truthy(provider):
            raise ValueError(
                f"Provider {matched_provider.id()} is not a valid {type} provider for "
                f"'{check_name}'"
            )
        if default_provider:
            logger.warning(
                "[Grading] Falling back to default provider after type check failed "
                "(checkName=%s, providerId=%s, type=%s)",
                check_name,
                matched_provider.id(),
                type,
            )
            return default_provider
        raise ValueError(
            f"Provider {matched_provider.id()} is not a valid {type} provider for '{check_name}'"
        )

    return matched_provider
