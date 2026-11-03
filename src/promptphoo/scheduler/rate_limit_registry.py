"""Per-eval registry of provider rate-limit state. Ports src/scheduler/rateLimitRegistry.ts."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any, Required, TypedDict, TypeVar

from promptphoo import envars
from promptphoo.fetch_context import ResponseHeadersObserver, with_fetch_retry_context
from promptphoo.jsvalues import json_stringify
from promptphoo.scheduler import clock
from promptphoo.scheduler.events import EventEmitter
from promptphoo.scheduler.provider_call_execution_context import (
    get_provider_call_execution_context,
    run_provider_call_with_abort,
    with_provider_call_execution_context,
)
from promptphoo.scheduler.provider_rate_limit_state import (
    ProviderExecuteOptions,
    ProviderMetrics,
    ProviderRateLimitState,
    RateLimitExhaustedError,
)
from promptphoo.scheduler.rate_limit_key import get_rate_limit_key
from promptphoo.scheduler.types import RateLimitExecuteOptions

logger = logging.getLogger(__name__)

T = TypeVar("T")

CallFn = Callable[[ResponseHeadersObserver | None], Awaitable[T]]


class RateLimitRegistryOptions(TypedDict, total=False):
    maxConcurrency: Required[int]
    minConcurrency: int
    queueTimeoutMs: float  # Timeout for queued requests (default: 5 minutes)


_FORWARDED_EVENTS = (
    "ratelimit:hit",
    "ratelimit:warning",
    "ratelimit:learned",
    "concurrency:increased",
    "concurrency:decreased",
    "request:retrying",
)


def _to_base36(fraction: float) -> str:
    """`Math.random().toString(36).slice(2)`: the base-36 digits after the point."""
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = []
    for _ in range(11):
        fraction *= 36
        digit = int(fraction)
        out.append(digits[digit])
        fraction -= digit
        if fraction <= 0:
            break
    return "".join(out)


class RateLimitRegistry(EventEmitter):
    """Per-eval registry that manages rate limit state for all providers.

    Not a singleton: create one per evaluation context.
    """

    def __init__(self, options: RateLimitRegistryOptions) -> None:
        super().__init__()
        self._states: dict[str, ProviderRateLimitState] = {}
        self._max_concurrency = options["maxConcurrency"]
        min_concurrency = options.get("minConcurrency")
        self._min_concurrency = (
            min_concurrency
            if min_concurrency is not None
            else envars.get_env_int("PROMPTFOO_MIN_CONCURRENCY", 1)
        )
        # Queue timeout: 0 means disabled, default is 5 minutes
        queue_timeout = options.get("queueTimeoutMs")
        self._queue_timeout_ms = (
            queue_timeout
            if queue_timeout is not None
            else envars.get_env_int("PROMPTFOO_SCHEDULER_QUEUE_TIMEOUT_MS", 5 * 60 * 1000)
        )
        self._enabled = not envars.get_env_bool("PROMPTFOO_DISABLE_ADAPTIVE_SCHEDULER", False)

    async def execute(
        self,
        provider: Any,
        call_fn: CallFn[T],
        options: RateLimitExecuteOptions[T] | None = None,
    ) -> T:
        """Execute a provider call with rate limiting and retries.

        No idempotency tracking: each call is independent. The single integration point (the
        evaluator) ensures no double-wrapping.
        """
        options = options or {}
        abort_signal = options.get("abortSignal")
        provider_max_retries = _get_provider_max_retries(provider)

        # Even when the scheduler is disabled, propagate the retry context so fetch-level
        # retries pick up the provider's `maxRetries` as their default and `maxRetries: 0`
        # disables transient retries.
        if not self._enabled:
            return await with_fetch_retry_context(
                provider_max_retries,
                lambda: run_provider_call_with_abort(lambda: call_fn(None), abort_signal),
            )

        rate_limit_key = get_rate_limit_key(provider)
        parent_context = get_provider_call_execution_context()
        if (
            parent_context
            and parent_context.get("rateLimitProvider")
            and get_rate_limit_key(parent_context.get("rateLimitProvider")) != rate_limit_key
        ):
            on_nested = parent_context.get("onNestedScheduledCall")
            if on_nested:
                on_nested()
        nested_call_started = False
        if rate_limit_key not in self._states:
            # A provider's own cap (config.maxConcurrency) bounds its pool below the run's
            # concurrency, so a shared-quota grader is not driven as hard as the targets (D80).
            provider_cap = _get_provider_max_concurrency(provider)
            max_concurrency = (
                min(self._max_concurrency, provider_cap)
                if provider_cap is not None
                else self._max_concurrency
            )
            state = ProviderRateLimitState({
                "rateLimitKey": rate_limit_key,
                "maxConcurrency": max_concurrency,
                "minConcurrency": min(self._min_concurrency, max_concurrency),
                "queueTimeoutMs": self._queue_timeout_ms,
            })  # fmt: skip
            if provider_cap is not None:
                logger.debug(
                    "[RateLimit] %s capped at %d concurrent call(s) by config.maxConcurrency",
                    sanitize_provider_id_for_log(provider.id()), max_concurrency,
                )  # fmt: skip

            # Forward events
            for event in _FORWARDED_EVENTS:
                state.on(event, lambda data, event=event: self.emit(event, data))

            self._states[rate_limit_key] = state

        state = self._states[rate_limit_key]

        # Generate unique request ID for metrics/logging
        request_id = f"{rate_limit_key}-{int(clock.date_now())}-{_to_base36(clock.math_random())}"

        self.emit("request:started", {"rateLimitKey": rate_limit_key, "requestId": request_id,
                                      "queueDepth": state.get_queue_depth()})  # fmt: skip

        def mark_nested() -> None:
            nonlocal nested_call_started
            nested_call_started = True

        async def scheduled_call(on_response_headers: ResponseHeadersObserver | None) -> T:
            execution_context = get_provider_call_execution_context()
            # Update an existing evaluator scope only while this acquired call owns its slot.
            # Direct registry users do not acquire evaluator orchestration.
            if execution_context is not None:
                return await with_provider_call_execution_context(
                    {
                        **execution_context,
                        "rateLimitRegistry": self,
                        "rateLimitProvider": provider,
                        "onNestedScheduledCall": mark_nested,
                    },
                    lambda: call_fn(on_response_headers),
                )
            return await call_fn(on_response_headers)

        get_headers = options.get("getHeaders")
        execute_options: ProviderExecuteOptions[T] = {
            "abortSignal": abort_signal,
            # A nested provider already learned its quota and exhausted its own retry budget.
            # Replaying the parent repeats all earlier child work.
            "getHeaders": lambda result: (
                None if nested_call_started or not get_headers else get_headers(result)
            ),
            "canRetry": lambda: not nested_call_started,
            "maxRetriesOverride": 0
            if getattr(provider, "handles_own_retries", False)
            else provider_max_retries,
        }
        for key in ("isRateLimited", "isRetryableResult", "finalizeResult",
                    "onRateLimitExhausted", "getRetryAfter"):  # fmt: skip
            if options.get(key) is not None:
                execute_options[key] = options[key]  # type: ignore[literal-required]

        async def run() -> T:
            return await state.execute_with_retry(request_id, scheduled_call, execute_options)

        try:
            result = await with_fetch_retry_context(
                provider_max_retries, lambda: run_provider_call_with_abort(run, abort_signal)
            )
            self.emit("request:completed", {"rateLimitKey": rate_limit_key,
                                            "requestId": request_id})  # fmt: skip
            return result
        except Exception as error:
            self.emit("request:failed", {"rateLimitKey": rate_limit_key, "requestId": request_id,
                                         "error": _js_error_string(error)})  # fmt: skip
            on_exhausted = options.get("onRateLimitExhausted")
            if isinstance(error, RateLimitExhaustedError) and on_exhausted:
                return on_exhausted(error.result, error)
            raise

    def get_metrics(self) -> dict[str, ProviderMetrics]:
        """Metrics for all tracked providers."""
        return {key: state.get_metrics() for key, state in self._states.items()}

    def dispose(self) -> None:
        """Cleanup all resources."""
        for state in self._states.values():
            state.dispose()
        self._states.clear()
        self.remove_all_listeners()


def _js_error_string(error: BaseException) -> str:
    from promptphoo.scheduler.provider_rate_limit_state import _js_error_string as impl

    return impl(error)


_DIGITS_RE = re.compile(r"[0-9]+")
_MAX_SAFE_INTEGER = 2**53 - 1


def _get_provider_max_retries(provider: Any) -> int | None:
    """The provider's configured maxRetries, tolerating numeric strings and rejecting invalid
    values.

    None means "fall back to defaults", distinct from 0, which disables retries. Invalid
    user-supplied values (negatives, floats, non-numeric strings) are logged so config typos
    are not silently ignored.
    """
    config = getattr(provider, "config", None)
    raw = config.get("maxRetries") if isinstance(config, dict) else None

    if raw is None:
        return None

    if (
        isinstance(raw, (int, float))
        and not isinstance(raw, bool)
        and float(raw).is_integer()
        and raw >= 0
    ):
        return int(raw)
    if isinstance(raw, str):
        trimmed = raw.strip()
        if _DIGITS_RE.fullmatch(trimmed):
            # Reject digit strings past Number.MAX_SAFE_INTEGER, which would never terminate
            # the retry loops in Node.
            parsed = int(trimmed)
            if parsed <= _MAX_SAFE_INTEGER:
                return parsed

    logger.warning(
        "%s %s",
        "[RateLimit] Ignoring invalid provider.config.maxRetries; expected a non-negative integer.",
        json_stringify(
            {"maxRetries": raw, "providerId": sanitize_provider_id_for_log(provider.id())}
        ),
    )
    return None


def _get_provider_max_concurrency(provider: Any) -> int | None:
    """The provider's configured maxConcurrency: a positive integer, else None (no cap). Invalid
    values are logged like maxRetries typos."""
    config = getattr(provider, "config", None)
    raw = config.get("maxConcurrency") if isinstance(config, dict) else None
    if raw is None:
        return None
    value: int | None = None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and float(raw).is_integer():
        value = int(raw)
    elif isinstance(raw, str) and _DIGITS_RE.fullmatch(raw.strip()):
        value = int(raw.strip())
    if value is not None and value >= 1:
        return value
    logger.warning(
        "%s %s",
        "[RateLimit] Ignoring invalid provider.config.maxConcurrency; expected a positive integer.",
        json_stringify(
            {"maxConcurrency": raw, "providerId": sanitize_provider_id_for_log(provider.id())}
        ),
    )
    return None


_DESCRIPTION_RE = re.compile(r"(\S+)(\s+\(.+\))")


def sanitize_provider_id_for_log(provider_id: str) -> str:
    """Ports sanitizeProviderIdForLog from src/util/provider.ts."""
    match = _DESCRIPTION_RE.fullmatch(provider_id)
    if match:
        return f"{sanitize_provider_id_for_log(match.group(1))}{match.group(2)}"
    if "://" in provider_id or provider_id.startswith("/"):
        from promptphoo.sanitizer import sanitize_url

        return sanitize_url(provider_id)
    return provider_id
