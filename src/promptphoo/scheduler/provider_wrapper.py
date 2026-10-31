"""Provider wrapper that adds rate limiting to any ApiProvider.

Ports src/scheduler/providerWrapper.ts. Use it to wrap providers before passing them to code
paths that bypass the main evaluator (graders, assertion helpers).
"""

from __future__ import annotations

import math
import re
import sys
from typing import TYPE_CHECKING, Any

from promptphoo.fetch_context import compose_response_headers_observers
from promptphoo.providers.base import ApiProvider
from promptphoo.scheduler.header_parser import _parse_int, parse_retry_after
from promptphoo.scheduler.types import (
    RateLimitExecuteOptions,
    get_provider_response_headers,
    is_provider_response_rate_limited,
)

if TYPE_CHECKING:
    from promptphoo.scheduler.cancellation import AbortSignal
    from promptphoo.scheduler.rate_limit_registry import RateLimitRegistry

ProviderResponse = dict[str, Any]

TOKEN_USAGE_FIELDS = ("prompt", "completion", "cached", "total")
COMPLETION_DETAIL_FIELDS = (
    "reasoning",
    "acceptedPrediction",
    "rejectedPrediction",
    "cacheReadInputTokens",
    "cacheCreationInputTokens",
)

MAX_REASONABLE_TOKEN_COUNT = 1_000_000_000
_MAX_SAFE_INTEGER = 2**53 - 1


def is_safe_token_count(value: Any) -> bool:
    """Ports isSafeTokenCount from src/util/numeric.ts."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and float(value).is_integer()
        and abs(value) <= _MAX_SAFE_INTEGER
        and 0 <= value <= MAX_REASONABLE_TOKEN_COUNT
    )


def is_safe_cost(value: Any) -> bool:
    """Ports isSafeCost from src/util/numeric.ts."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _add_safe_token_metric(current: Any, value: Any) -> int | None:
    if not is_safe_token_count(value):
        return None
    total = (current or 0) + value
    return int(total) if abs(total) <= _MAX_SAFE_INTEGER else None


def _accumulate_retry_token_usage(target: dict[str, Any], response: ProviderResponse) -> None:
    usage = response.get("tokenUsage")
    usage = usage if isinstance(usage, dict) else {}
    for field in TOKEN_USAGE_FIELDS:
        total = _add_safe_token_metric(target.get(field), usage.get(field))
        if total is not None:
            target[field] = total

    details = usage.get("completionDetails")
    details = details if isinstance(details, dict) else {}
    for field in COMPLETION_DETAIL_FIELDS:
        value = details.get(field)
        total = _add_safe_token_metric((target.get("completionDetails") or {}).get(field), value)
        if total is not None:
            target.setdefault("completionDetails", {})
            target["completionDetails"][field] = total

    if response.get("cached"):
        requests: Any = 0
    elif is_safe_token_count(usage.get("numRequests")):
        requests = usage["numRequests"]
    else:
        requests = 1
    total_requests = _add_safe_token_metric(target.get("numRequests"), requests)
    if total_requests is not None:
        target["numRequests"] = total_requests


def is_rate_limit_wrapped(provider: Any) -> bool:
    """Whether a provider is already wrapped with rate limiting."""
    return getattr(provider, "_rate_limit_wrapped", False) is True


_RETRY_AFTER_MESSAGE_RE = re.compile(r"\bretry after ([0-9]+)\b", re.IGNORECASE | re.ASCII)


def _get_retry_after(result: ProviderResponse | None, error: BaseException | None) -> float | None:
    metadata = result.get("metadata") if isinstance(result, dict) else None
    if isinstance(metadata, dict) and metadata.get("rateLimitRetryable") is False:
        return None
    raw_headers = get_provider_response_headers(result)
    if raw_headers:
        # Normalize header keys to lowercase for consistent access
        headers = {str(key).lower(): value for key, value in raw_headers.items()}
        # Check retry-after-ms first (milliseconds)
        if headers.get("retry-after-ms"):
            ms = _parse_int(headers["retry-after-ms"])
            if ms is not None and ms >= 0:
                return ms
        # Check retry-after (robust parsing for seconds/HTTP-date)
        if headers.get("retry-after"):
            parsed = parse_retry_after(headers["retry-after"])
            if parsed is not None:
                return parsed
    # Try to extract from error message (some providers include it)
    if error is not None:
        message = getattr(error, "message", None)
        message = message if isinstance(message, str) else str(error)
        match = _RETRY_AFTER_MESSAGE_RE.search(message)
        if match:
            retry_after_ms = int(match.group(1)) * 1000
            # Node's Number.parseInt overflows to Infinity, which is rejected as non-finite.
            return retry_after_ms if retry_after_ms <= sys.float_info.max else None
    return None


def _finalize_result(result: ProviderResponse, retry_results: Any) -> ProviderResponse:
    if len(retry_results) == 0:
        return result
    token_usage: dict[str, Any] = {}
    combined_cost: float = 0
    has_cost = False
    for response in [*retry_results, result]:
        _accumulate_retry_token_usage(token_usage, response)
        if is_safe_cost(response.get("cost")):
            combined_cost += response["cost"]
            has_cost = True
    result_without_cost = {k: v for k, v in result.items() if k != "cost"}
    return {
        **result_without_cost,
        "tokenUsage": token_usage,
        **({"cost": combined_cost} if has_cost and math.isfinite(combined_cost) else {}),
    }


def create_provider_rate_limit_options(
    abort_signal: AbortSignal | None = None,
) -> RateLimitExecuteOptions[ProviderResponse]:
    """Rate limit detection options for a ProviderResponse.

    Shared between the provider wrapper and the evaluator for consistency.
    """
    options: RateLimitExecuteOptions[ProviderResponse] = {}
    if abort_signal:
        options["abortSignal"] = abort_signal

    def on_rate_limit_exhausted(result: ProviderResponse, error: BaseException) -> Any:
        # Provider errors are values carrying output, usage and HTTP metadata. Keep that
        # evidence when the scheduler has no retries left.
        if isinstance(result, dict) and result.get("error"):
            return result
        message = getattr(error, "message", None)
        return {**(result or {}), "error": message if isinstance(message, str) else str(error)}

    def get_headers(result: ProviderResponse | None) -> dict[str, str] | None:
        # Non-retryable rate limits must not feed the shared rate-limit state either: a billing
        # 429 that also carries `x-ratelimit-remaining-*: 0` and a reset timestamp would
        # otherwise park every queued and subsequent call until that reset instead of letting
        # them fail fast.
        metadata = result.get("metadata") if isinstance(result, dict) else None
        if isinstance(metadata, dict) and (
            metadata.get("rateLimitRetryable") is False or metadata.get("rateLimitKind") == "quota"
        ):
            return None
        return get_provider_response_headers(result)

    def is_retryable_result(result: ProviderResponse) -> bool:
        metadata = result.get("metadata") if isinstance(result, dict) else None
        return (
            isinstance(metadata, dict)
            and metadata.get("retryableErrorKind") == "transient_availability"
        )

    options.update({
        "onRateLimitExhausted": on_rate_limit_exhausted,
        "getHeaders": get_headers,
        "isRateLimited": is_provider_response_rate_limited,
        "getRetryAfter": _get_retry_after,
        "isRetryableResult": is_retryable_result,
        "finalizeResult": _finalize_result,
    })  # fmt: skip
    return options


class _RateLimitWrappedProvider(ApiProvider):
    """The wrapped provider: a copy of the original's own attributes (like Node's object
    spread) whose `call_api` goes through the registry. Other methods delegate."""

    _rate_limit_wrapped = True

    def __init__(self, provider: Any, registry: RateLimitRegistry) -> None:
        self.__dict__.update(provider.__dict__)
        self.handles_own_delay = getattr(provider, "handles_own_delay", False)
        self.handles_own_retries = getattr(provider, "handles_own_retries", False)
        self._wrapped_provider = provider
        self._registry = registry

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes missing from the copy (e.g. call_embedding_api).
        if name.startswith("__") or name == "_wrapped_provider":
            raise AttributeError(name)
        return getattr(self._wrapped_provider, name)

    def id(self) -> str:
        # Explicitly delegate id()
        return self._wrapped_provider.id()

    def default_id(self) -> str:
        return self._wrapped_provider.id()

    def to_json(self) -> dict[str, Any]:
        return self._wrapped_provider.to_json()

    async def cleanup(self) -> None:
        await self._wrapped_provider.cleanup()

    async def call_api(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ProviderResponse:
        provider = self._wrapped_provider

        async def call(on_response_headers: Any = None) -> ProviderResponse:
            if on_response_headers:
                call_options = {
                    **(options or {}),
                    "onResponseHeaders": compose_response_headers_observers(
                        on_response_headers, (options or {}).get("onResponseHeaders")
                    ),
                }
            else:
                call_options = options
            return await provider.call_api(prompt, context, call_options)

        return await self._registry.execute(
            provider,
            call,
            create_provider_rate_limit_options((options or {}).get("abortSignal")),
        )


def wrap_provider_with_rate_limiting(provider: Any, registry: RateLimitRegistry) -> Any:
    """Wrap a provider so every call_api goes through the registry (rate limits, retries and
    adaptive concurrency). Already-wrapped providers are returned unchanged."""
    # Don't double-wrap
    if is_rate_limit_wrapped(provider):
        return provider
    return _RateLimitWrappedProvider(provider, registry)


def wrap_providers_with_rate_limiting(providers: list[Any], registry: RateLimitRegistry) -> list:
    return [wrap_provider_with_rate_limiting(provider, registry) for provider in providers]
