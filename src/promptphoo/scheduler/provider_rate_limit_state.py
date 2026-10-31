"""Rate limit state and retry logic for one rate-limit key.

Ports src/scheduler/providerRateLimitState.ts.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from typing import Any, Required, TypedDict, TypeVar

from promptphoo.fetch_context import (
    ResponseHeadersObserver,
    create_response_headers_observer,
    is_response_headers_observer_error,
    is_response_headers_observer_error_response,
)
from promptphoo.scheduler import clock
from promptphoo.scheduler.adaptive_concurrency import (
    WARNING_THRESHOLD,
    AdaptiveConcurrency,
    ConcurrencyChangeResult,
)
from promptphoo.scheduler.cancellation import sleep_with_abort, throw_if_aborted
from promptphoo.scheduler.events import EventEmitter
from promptphoo.scheduler.header_parser import parse_rate_limit_headers
from promptphoo.scheduler.retry_policy import (
    DEFAULT_RETRY_POLICY,
    RetryPolicy,
    get_retry_delay,
    should_retry,
)
from promptphoo.scheduler.slot_queue import SlotQueue
from promptphoo.scheduler.types import RateLimitExecuteOptions

T = TypeVar("T")


class RateLimitExhaustedError(Exception):
    name = "RateLimitExhaustedError"

    def __init__(self, message: str, result: Any) -> None:
        super().__init__(message)
        self.message = message
        self.result = result


class _ResultFinalizationError(Exception):
    """Wraps a finalizeResult failure so the retry classifier skips it: a finalizer failure is
    not provider work and must never trigger another call."""

    name = "ResultFinalizationError"

    def __init__(self, original: BaseException) -> None:
        super().__init__("finalizeResult failed")
        self.original = original


class ProviderStateOptions(TypedDict, total=False):
    rateLimitKey: Required[str]
    maxConcurrency: Required[int]
    minConcurrency: Required[int]
    queueTimeoutMs: float
    retryPolicy: RetryPolicy


class ProviderMetrics(TypedDict):
    rateLimitKey: str
    activeRequests: int
    maxConcurrency: int
    queueDepth: int
    totalRequests: int
    completedRequests: int
    failedRequests: int
    rateLimitHits: int
    retriedRequests: int
    avgLatencyMs: float
    p50LatencyMs: float
    p99LatencyMs: float


class ProviderExecuteOptions(RateLimitExecuteOptions[T], total=False):
    # Per-call override for `maxRetries` only. Preserves the state's other policy fields
    # (backoff, jitter) so provider config cannot silently reset them.
    maxRetriesOverride: int | None


def _error_name(error: BaseException) -> str:
    name = getattr(error, "name", None)
    return name if isinstance(name, str) else type(error).__name__


def _has_error_response(result: Any) -> bool:
    return (
        isinstance(result, dict)
        and isinstance(result.get("error"), str)
        and len(result["error"]) > 0
    )


def _has_refusal_response(result: Any) -> bool:
    return isinstance(result, dict) and result.get("isRefusal") is True


class _AttemptSlot:
    """Per-attempt slot ownership (Node's per-iteration `let` bindings)."""

    def __init__(self) -> None:
        self.owns = True
        self.observed_headers: dict[str, str] | None = None


class ProviderRateLimitState(EventEmitter):
    """Manages rate limit state and retry logic for a single rate limit key."""

    def __init__(self, options: ProviderStateOptions) -> None:
        super().__init__()
        self.rate_limit_key: str = options["rateLimitKey"]
        self._retry_policy: RetryPolicy = options.get("retryPolicy") or DEFAULT_RETRY_POLICY

        self._adaptive_concurrency = AdaptiveConcurrency(
            options["maxConcurrency"], options["minConcurrency"]
        )

        self._slot_queue = SlotQueue({
            "maxConcurrency": options["maxConcurrency"],
            "minConcurrency": options["minConcurrency"],
            **({"queueTimeoutMs": options["queueTimeoutMs"]}  # type: ignore[typeddict-item]
               if options.get("queueTimeoutMs") is not None else {}),
            "onSlotAcquired": lambda queue_depth: self.emit(
                "slot:acquired", {"rateLimitKey": self.rate_limit_key, "queueDepth": queue_depth}
            ),
            "onSlotReleased": lambda queue_depth: self.emit(
                "slot:released", {"rateLimitKey": self.rate_limit_key, "queueDepth": queue_depth}
            ),
        })  # fmt: skip

        # Metrics
        self._total_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._rate_limit_hits = 0
        self._retried_requests = 0
        # Keep the latest 100 latency measurements for bounded metric sorting and storage.
        self._latencies: list[float] = []

        # Track whether ratelimit:learned was emitted for this provider
        self._has_learned_limits = False

    def _push_latency(self, value: float) -> None:
        self._latencies.append(value)
        if len(self._latencies) > 100:
            self._latencies.pop(0)

    async def execute_with_retry(
        self,
        request_id: str,
        call_fn: Callable[[ResponseHeadersObserver | None], Awaitable[T]],
        options: ProviderExecuteOptions[T],
    ) -> T:
        """Execute a call with rate limiting and retry logic."""
        self._total_requests += 1
        attempt = 0
        retry_results: list[T] = []
        override = options.get("maxRetriesOverride")
        retry_policy: RetryPolicy = (
            self._retry_policy
            if override is None
            else {**self._retry_policy, "maxRetries": override}
        )
        abort_signal = options.get("abortSignal")
        can_retry = options.get("canRetry")
        get_headers = options.get("getHeaders")
        is_rate_limited_fn = options.get("isRateLimited")
        is_retryable_result_fn = options.get("isRetryableResult")
        get_retry_after = options.get("getRetryAfter")
        finalize_result = options.get("finalizeResult")
        on_rate_limit_exhausted = options.get("onRateLimitExhausted")

        try:
            while True:
                try:
                    await self._slot_queue.acquire(f"{request_id}-{attempt}", abort_signal)
                except Exception as acquire_error:
                    self.emit(
                        "queue:cancelled"
                        if abort_signal is not None and abort_signal.aborted
                        else "queue:timeout",
                        {"rateLimitKey": self.rate_limit_key, "requestId": request_id,
                         "error": _js_error_string(acquire_error)},
                    )  # fmt: skip
                    raise

                # A result can release its slot before retry backoff. Only this owner may
                # release it, including aborts between the grant and call_fn invocation.
                slot = _AttemptSlot()

                def observe(
                    args: tuple[dict[str, str], Any],
                    already_observed: bool,
                    slot: _AttemptSlot = slot,
                ) -> bool:
                    headers, backoff = args
                    if slot.owns:
                        if backoff:
                            if not already_observed:
                                # The lower target fetch selected this deadline before its
                                # wait. Replaying relative headers when a consumer joins would
                                # extend it.
                                self._update_from_headers(headers, False, backoff["resetAt"])
                                self._handle_rate_limit(None, backoff["resetAt"])
                        else:
                            slot.observed_headers = headers
                            if not already_observed:
                                self._update_from_headers(headers, False)
                        return True
                    return already_observed

                on_response_headers = create_response_headers_observer(self, observe)

                def release_slot(slot: _AttemptSlot = slot) -> None:
                    if slot.owns:
                        slot.owns = False
                        self._slot_queue.release()

                start_time = clock.date_now()
                retry_error: BaseException | None = None
                retry_result: T | None = None
                is_rate_limited = False
                is_retryable_result = False
                retry_after_ms: float | None = None

                try:
                    throw_if_aborted(abort_signal)
                    result = await call_fn(on_response_headers)
                    retry_result = result
                    has_error_response = _has_error_response(result)
                    has_refusal_response = _has_refusal_response(result)
                    headers = get_headers(result) if get_headers else None
                    is_observer_error = is_response_headers_observer_error_response(result)
                    is_rate_limited = (
                        (can_retry is None or can_retry() is not False)
                        and not is_observer_error
                        and bool(is_rate_limited_fn(result, None) if is_rate_limited_fn else False)
                    )
                    is_retryable_result = (
                        (can_retry is None or can_retry() is not False)
                        and not is_observer_error
                        and bool(
                            is_retryable_result_fn(result) if is_retryable_result_fn else False
                        )
                    )
                    retry_after_ms = get_retry_after(result, None) if get_retry_after else None

                    # Observer diagnostics may carry another service's headers. Keep the actual
                    # wire quota already learned by on_response_headers instead.
                    if (
                        not is_observer_error
                        and headers
                        and (headers is not slot.observed_headers or is_rate_limited)
                    ):
                        self._update_from_headers(headers, is_rate_limited)
                    if is_rate_limited:
                        self._handle_rate_limit(retry_after_ms)
                    # Completed calls retain their result and quota even if cancellation
                    # arrived during completion. Cancellation still prevents another attempt.
                    self._push_latency(clock.date_now() - start_time)
                    release_slot()

                    # Keep an independent failure's diagnostic and metadata intact, but never
                    # retry it once the caller has cancelled.
                    aborted = abort_signal is not None and abort_signal.aborted
                    if aborted and has_error_response:
                        self._failed_requests += 1
                        return result

                    if (not is_rate_limited and not is_retryable_result) or (
                        aborted and has_refusal_response
                    ):
                        try:
                            finalized_result = (
                                finalize_result(result, retry_results) if finalize_result else None
                            )
                            if finalized_result is None:
                                finalized_result = result
                        except Exception as finalizer_error:
                            raise _ResultFinalizationError(finalizer_error) from finalizer_error
                        self._apply_concurrency_change(self._adaptive_concurrency.record_success())
                        self._completed_requests += 1
                        return finalized_result
                except Exception as error:
                    if slot.owns:
                        self._push_latency(clock.date_now() - start_time)

                    if is_response_headers_observer_error(on_response_headers, error):
                        raise

                    if isinstance(error, _ResultFinalizationError):
                        raise error.original from None

                    # Cancellation is final, even for a custom reason or a message that
                    # resembles a retryable error. Preserve unrelated provider errors.
                    if (
                        abort_signal is not None
                        and abort_signal.aborted
                        and error is abort_signal.reason
                    ) or _error_name(error) in ("AbortError", "AbortException"):
                        raise

                    retry_error = error
                    is_rate_limited = (can_retry is None or can_retry() is not False) and bool(
                        is_rate_limited_fn(None, retry_error)
                        if is_rate_limited_fn
                        else self._is_rate_limit_error(retry_error)
                    )
                    retry_after_ms = get_retry_after(None, retry_error) if get_retry_after else None
                    if is_rate_limited:
                        self._handle_rate_limit(retry_after_ms)
                    if abort_signal is not None and abort_signal.aborted:
                        raise
                finally:
                    release_slot()

                if (can_retry is not None and can_retry() is False) or not should_retry(
                    attempt, retry_error, is_rate_limited or is_retryable_result, retry_policy
                ):
                    if retry_error is not None:
                        raise retry_error
                    if is_retryable_result:
                        # Transient-availability exhaustion returns the terminal result with
                        # bounded retry usage merged, not the rate-limit sentinel.
                        finalized = (
                            finalize_result(retry_result, retry_results)  # type: ignore[arg-type]
                            if finalize_result
                            else None
                        )
                        self._failed_requests += 1
                        return finalized if finalized is not None else retry_result  # type: ignore[return-value]
                    if not on_rate_limit_exhausted and finalize_result:
                        # A bare finalizeResult keeps the older structured-return behavior; the
                        # registry's onRateLimitExhausted contract raises to convert.
                        finalized = finalize_result(retry_result, retry_results)  # type: ignore[arg-type]
                        self._failed_requests += 1
                        return finalized
                    raise RateLimitExhaustedError(
                        f"Rate limit exceeded for {self.rate_limit_key} after {attempt + 1} "
                        "attempts",
                        retry_result,
                    )

                # Only transient-availability retries feed the terminal merge; a rate-limited
                # retry must surface the last raw result as-is.
                if is_retryable_result and retry_result is not None:
                    retry_results.append(retry_result)

                attempt += 1
                delay = self._record_retry(
                    attempt,
                    retry_policy,
                    retry_after_ms,
                    "ratelimit" if is_rate_limited else "error",
                )

                # Both result and exception retries wait after their slot is released.
                await sleep_with_abort(delay, abort_signal)
        except BaseException:
            self._failed_requests += 1
            raise

    def _record_retry(
        self,
        attempt: int,
        retry_policy: RetryPolicy,
        retry_after_ms: float | None,
        reason: str,
    ) -> float:
        self._retried_requests += 1
        delay_ms = get_retry_delay(attempt, retry_policy, retry_after_ms)
        self.emit("request:retrying", {"rateLimitKey": self.rate_limit_key, "attempt": attempt,
                                       "delayMs": delay_ms, "reason": reason})  # fmt: skip
        return delay_ms

    def _update_from_headers(
        self,
        headers: dict[str, str],
        is_rate_limited: bool,
        selected_reset_at: float | None = None,
    ) -> None:
        """Update state from response headers.

        When `is_rate_limited` is False, retry-after headers are ignored so successful responses
        from providers/proxies that include them do not block the queue.
        """
        parsed = parse_rate_limit_headers(headers)
        if selected_reset_at is not None:
            existing_reset_at = self._slot_queue.get_reset_at()
            # A selected backoff must not shorten quota learned by a concurrent call. Fresh
            # successful responses still replace quota through the normal path.
            parsed["resetAt"] = (
                max(existing_reset_at, selected_reset_at)
                if existing_reset_at is not None and existing_reset_at > clock.date_now()
                else selected_reset_at
            )

        # Emit ratelimit:learned only once per provider, when limit headers first appear
        if not self._has_learned_limits and ("limitRequests" in parsed or "limitTokens" in parsed):
            self._has_learned_limits = True
            self.emit("ratelimit:learned", {
                "rateLimitKey": self.rate_limit_key,
                "requestLimit": parsed.get("limitRequests"),
                "tokenLimit": parsed.get("limitTokens"),
            })  # fmt: skip

        # Update slot queue with new state (remaining counts, limits, reset times)
        self._slot_queue.update_rate_limit_state(parsed)

        # Only enforce retry-after when the response is actually rate-limited.
        if is_rate_limited and "retryAfterMs" in parsed:
            self._slot_queue.mark_rate_limited(parsed["retryAfterMs"])

        # Check for proactive throttling
        ratios = self._slot_queue.get_remaining_ratio()
        request_ratio, token_ratio = ratios["requests"], ratios["tokens"]
        min_ratio = min(
            request_ratio if request_ratio is not None else 1,
            token_ratio if token_ratio is not None else 1,
        )

        if min_ratio < WARNING_THRESHOLD:
            self.emit("ratelimit:warning", {"rateLimitKey": self.rate_limit_key,
                                            "requestRatio": request_ratio,
                                            "tokenRatio": token_ratio})  # fmt: skip
            # Proactive concurrency reduction
            self._apply_concurrency_change(
                self._adaptive_concurrency.record_approaching_limit(min_ratio)
            )

    def _handle_rate_limit(
        self, retry_after_ms: float | None = None, selected_reset_at: float | None = None
    ) -> None:
        """Handle a rate limit hit. SlotQueue preserves any existing resetAt from headers."""
        self._rate_limit_hits += 1

        # Proactive queue processing may already have cleared an elapsed deadline. Keep the
        # selected timestamp explicit so it cannot become unknown quota.
        self._slot_queue.mark_rate_limited(retry_after_ms, selected_reset_at)

        change = self._adaptive_concurrency.record_rate_limit()
        self._apply_concurrency_change(change)

        self.emit("ratelimit:hit", {
            "rateLimitKey": self.rate_limit_key,
            "retryAfterMs": retry_after_ms,
            "resetAt": self._slot_queue.get_reset_at(),
            "concurrencyChange": change,
        })  # fmt: skip

    def _apply_concurrency_change(self, change: ConcurrencyChangeResult) -> None:
        if change["changed"]:
            self._slot_queue.set_max_concurrency(change["current"])
            self.emit(
                "concurrency:increased"
                if change["reason"] == "recovery"
                else "concurrency:decreased",
                {"rateLimitKey": self.rate_limit_key, **change},
            )

    def _is_rate_limit_error(self, error: BaseException) -> bool:
        message = _message(error).lower()
        return "429" in message or "rate limit" in message or "too many requests" in message

    def get_queue_depth(self) -> int:
        """Current queue depth without sorting latencies (cheap; for frequent checks)."""
        return self._slot_queue.get_queue_depth()

    def get_metrics(self) -> ProviderMetrics:
        ordered = sorted(self._latencies)
        n = len(ordered)
        return {
            "rateLimitKey": self.rate_limit_key,
            "activeRequests": self._slot_queue.get_active_count(),
            "maxConcurrency": self._slot_queue.get_max_concurrency(),
            "queueDepth": self._slot_queue.get_queue_depth(),
            "totalRequests": self._total_requests,
            "completedRequests": self._completed_requests,
            "failedRequests": self._failed_requests,
            "rateLimitHits": self._rate_limit_hits,
            "retriedRequests": self._retried_requests,
            "avgLatencyMs": sum(ordered) / n if n > 0 else 0,
            # Percentiles: for n elements, pX is at index floor((n-1) * X/100)
            "p50LatencyMs": ordered[math.floor((n - 1) * 0.5)] if n > 0 else 0,
            "p99LatencyMs": ordered[math.floor((n - 1) * 0.99)] if n > 0 else 0,
        }

    def dispose(self) -> None:
        self._slot_queue.dispose()
        self.remove_all_listeners()


def _message(error: BaseException) -> str:
    message = getattr(error, "message", None)
    return message if isinstance(message, str) else str(error)


def _js_error_string(error: BaseException) -> str:
    """`String(error)` for an Error: "<name>: <message>" (or just the name)."""
    name = _error_name(error)
    if name == "Exception":
        name = "Error"
    message = _message(error)
    return f"{name}: {message}" if message else name
