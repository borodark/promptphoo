"""Retry policy. Ports src/scheduler/retryPolicy.ts."""

from __future__ import annotations

from typing import TypedDict

from promptphoo.fetch_context import is_http_rate_limit_error, is_transient_connection_error
from promptphoo.scheduler import clock


class RetryPolicy(TypedDict):
    maxRetries: int
    baseDelayMs: float
    maxDelayMs: float
    jitterFactor: float  # 0-1, adds randomness to prevent thundering herd


DEFAULT_RETRY_POLICY: RetryPolicy = {
    "maxRetries": 3,
    "baseDelayMs": 1000,
    "maxDelayMs": 60000,
    "jitterFactor": 0.2,
}


def get_retry_delay(
    attempt: int, policy: RetryPolicy, server_retry_after_ms: float | None = None
) -> float:
    """Delay for a retry attempt. Prefers a server-specified Retry-After when available."""
    # If server specified retry-after, use it (with jitter).
    # Note: server_retry_after_ms = 0 means "retry immediately"
    if server_retry_after_ms is not None and server_retry_after_ms >= 0:
        if server_retry_after_ms == 0:
            return 0  # Immediate retry
        jitter = server_retry_after_ms * policy["jitterFactor"] * clock.math_random()
        return min(server_retry_after_ms + jitter, policy["maxDelayMs"])

    # Exponential backoff with jitter
    exponential_delay = policy["baseDelayMs"] * 2**attempt
    jitter = exponential_delay * policy["jitterFactor"] * clock.math_random()
    return min(exponential_delay + jitter, policy["maxDelayMs"])


def _message(error: BaseException) -> str:
    message = getattr(error, "message", None)
    return message if isinstance(message, str) else str(error)


def should_retry(
    attempt: int,
    error: BaseException | None,
    is_rate_limited: bool,
    policy: RetryPolicy,
) -> bool:
    """Determine whether to retry a failed request."""
    if attempt >= policy["maxRetries"]:
        return False

    # Hard quotas (insufficient_quota, billing_hard_limit_reached, ...) won't resolve on retry;
    # retrying just amplifies load against an exhausted account. Check the structured signal
    # before the rate-limit retry path, since is_rate_limited may come from a substring match.
    if is_http_rate_limit_error(error) and getattr(error, "kind", None) == "quota":
        return False

    # Always retry rate limits (up to max)
    if is_rate_limited:
        return True

    # Retry transient errors: TLS/socket-level failures via is_transient_connection_error, and
    # higher-level transient failures (DNS, HTTP status codes, timeouts) via the patterns below.
    if error:
        message = _message(error).lower()
        return (
            is_transient_connection_error(error)
            or "timeout" in message
            or message.startswith("websocket request timed out after ")
            or "econnrefused" in message
            or "network" in message
            or "503" in message
            or "502" in message
            or "504" in message
        )

    return False
