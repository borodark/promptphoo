"""Shared scheduler types and ProviderResponse rate-limit detection.

Ports src/scheduler/types.ts.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

from promptphoo.fetch_context import (
    is_http_rate_limit_error,
    is_response_headers_observer_error_response,
    is_transient_connection_error,
)
from promptphoo.jsvalues import is_truthy

if TYPE_CHECKING:
    from promptphoo.scheduler.cancellation import AbortSignal

T = TypeVar("T")

__all__ = [
    "RateLimitExecuteOptions",
    "get_provider_response_headers",
    "is_provider_response_rate_limited",
    "is_transient_connection_error",
]


class RateLimitExecuteOptions(TypedDict, Generic[T], total=False):
    """Options for rate-limited execution (RateLimitRegistry.execute and provider wrappers)."""

    # Cancel queue and retry waits for this caller.
    abortSignal: AbortSignal | None
    # Extract rate limit headers from the result
    getHeaders: Callable[[T | None], dict[str, str] | None]
    # Detect if the result indicates a rate limit
    isRateLimited: Callable[[T | None, BaseException | None], bool]
    # Extract retry-after delay from result or error
    getRetryAfter: Callable[[T | None, BaseException | None], float | None]
    # Detect retryable result-level transient failures that are not rate limits.
    isRetryableResult: Callable[[T], bool]
    # Merge bounded prior retry results into the terminal result.
    finalizeResult: Callable[[T, Sequence[T]], T]
    # Preserve a structured failure result when retries are exhausted. Defaults to raising.
    onRateLimitExhausted: Callable[[T, BaseException], T]
    # Returning False vetoes another attempt, e.g. a nested call that already retried.
    canRetry: Callable[[], bool]


# Word-bounded: a bare "429" substring also matches token counts and request IDs, e.g.
# "prompt is too long: 204291 tokens".
HTTP_429_RE = re.compile(r"\b429\b", re.ASCII)


def _message(error: BaseException | None) -> str:
    if error is None:
        return ""
    message = getattr(error, "message", None)
    return message if isinstance(message, str) else str(error)


def _get(obj: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def is_provider_response_rate_limited(
    result: dict[str, Any] | None, error: BaseException | None
) -> bool:
    """Default rate limit detection for a ProviderResponse.

    True when a result/error indicates a *retryable* rate limit. Hard quota failures (billing
    exhaustion, daily limits) match the same `429` / `too many requests` substrings but are not
    retryable: retrying them amplifies load against an exhausted account.
    """
    if is_response_headers_observer_error_response(result):
        return False
    metadata = _get(result, "metadata")
    # Tool diagnostics may mention their own quota without describing the model request.
    response_error = None if _get(metadata, "errorOrigin") == "tool" else _get(result, "error")
    # Structured signal: never retry a hard quota.
    if _get(metadata, "rateLimitRetryable") is False or _get(metadata, "rateLimitKind") == "quota":
        return False
    if _get(metadata, "rateLimitKind") == "rate_limit":
        return True
    if is_http_rate_limit_error(error) and getattr(error, "kind", None) == "quota":
        return False
    # String fallback for providers that fold the structured error into
    # `error: formatRateLimitErrorMessage(...)`; "Quota exceeded:" is part of that formatter's
    # contract and may be wrapped in a prefix, so this is a substring match.
    response_error_text = response_error if isinstance(response_error, str) else None
    if response_error_text is not None and "Quota exceeded:" in response_error_text:
        return False
    error_message = _message(error) if error is not None else None
    if error_message is not None and "Quota exceeded:" in error_message:
        return False

    return bool(
        # Check HTTP status code (most reliable)
        _get(metadata, "http", "status") == 429
        # Check error field in response
        or HTTP_429_RE.search(response_error_text or "")
        or (response_error_text is not None and "rate limit" in response_error_text.lower())
        # Check thrown error message
        or HTTP_429_RE.search(error_message or "")
        or (error_message is not None and "rate limit" in error_message.lower())
        or (error_message is not None and "too many requests" in error_message.lower())
    )


def get_provider_response_headers(result: dict[str, Any] | None) -> dict[str, str] | None:
    """Rate limit headers from a ProviderResponse: metadata.http.headers or metadata.headers."""
    headers = _get(result, "metadata", "http", "headers")
    if not is_truthy(headers):
        headers = _get(result, "metadata", "headers")
    return headers if is_truthy(headers) else None
