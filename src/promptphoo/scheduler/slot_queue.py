"""Concurrency slots with a FIFO queue. Ports src/scheduler/slotQueue.ts.

Timers are asyncio `call_later` handles. As in Node, slot allocation is synchronous
(`_process_queue` never awaits), so there is no race between the capacity check and the
increment. A waiter whose task is cancelled leaves the queue (or hands back a slot it was
granted but never observed), so cancellation cannot leak slots.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Required, TypedDict

from promptphoo.scheduler import clock

if TYPE_CHECKING:
    from promptphoo.scheduler.cancellation import AbortSignal
    from promptphoo.scheduler.header_parser import ParsedRateLimitHeaders


@dataclass(eq=False)
class _QueuedRequest:
    id: str
    resolve: Callable[[], None]
    reject: Callable[[BaseException], None]


class SlotQueueOptions(TypedDict, total=False):
    maxConcurrency: Required[int]
    minConcurrency: Required[int]
    queueTimeoutMs: float  # Optional timeout for queued requests
    onSlotAcquired: Callable[[int], None]
    onSlotReleased: Callable[[int], None]


DEFAULT_QUEUE_TIMEOUT_MS = 5 * 60 * 1000  # 5 minutes


def _call_later(delay_ms: float, callback: Callable[[], None]) -> asyncio.TimerHandle | None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    return loop.call_later(max(0.0, delay_ms) / 1000, callback)


class SlotQueue:
    def __init__(self, options: SlotQueueOptions) -> None:
        self._active_count = 0
        self._max_concurrency = options["maxConcurrency"]
        self._min_concurrency = options["minConcurrency"]
        self._waiting: list[_QueuedRequest] = []
        self._reset_timer: asyncio.TimerHandle | None = None
        timeout = options.get("queueTimeoutMs")
        self._queue_timeout_ms = DEFAULT_QUEUE_TIMEOUT_MS if timeout is None else timeout

        # Rate limit state
        self._reset_at: float | None = None
        self._remaining_requests: float | None = None
        self._remaining_tokens: float | None = None
        self._request_limit: float | None = None
        self._token_limit: float | None = None

        self._on_slot_acquired = options.get("onSlotAcquired")
        self._on_slot_released = options.get("onSlotReleased")

    async def acquire(self, request_id: str, abort_signal: AbortSignal | None = None) -> None:
        """Acquire a slot. All requests go through the queue to prevent race conditions.

        Returns when a slot is available and quota is not exhausted. An aborted signal stops the
        request from waiting, but a slot that is free immediately is still granted.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        timeout_handle: asyncio.TimerHandle | None = None

        def cleanup() -> None:
            if timeout_handle is not None:
                timeout_handle.cancel()
            if abort_signal is not None:
                abort_signal.remove_listener(on_abort)

        def remove_and_reject(error: BaseException) -> None:
            if request in self._waiting:
                self._waiting.remove(request)
                request.reject(error)
                self._process_queue()

        def on_abort() -> None:
            assert abort_signal is not None
            remove_and_reject(abort_signal.reason or Exception("aborted"))

        def resolve() -> None:
            cleanup()
            self._active_count += 1
            if self._on_slot_acquired:
                self._on_slot_acquired(len(self._waiting))
            if not future.done():
                future.set_result(None)

        def reject(error: BaseException) -> None:
            cleanup()
            if not future.done():
                future.set_exception(error)

        request = _QueuedRequest(id=request_id, resolve=resolve, reject=reject)

        if self._queue_timeout_ms > 0:
            timeout_handle = loop.call_later(
                self._queue_timeout_ms / 1000,
                lambda: remove_and_reject(
                    Exception(
                        f"Request {request_id} timed out after "
                        f"{_ms(self._queue_timeout_ms)}ms in queue"
                    )
                ),
            )
        self._waiting.append(request)
        if abort_signal is not None:
            abort_signal.add_listener(on_abort)

        # Immediately try to process queue (synchronous, no race)
        self._process_queue()
        if abort_signal is not None and abort_signal.aborted:
            on_abort()

        try:
            await future
        except asyncio.CancelledError:
            if request in self._waiting:
                self._waiting.remove(request)
                cleanup()
                self._process_queue()
            elif future.done() and not future.cancelled() and future.exception() is None:
                # Granted while the caller was being cancelled: hand the slot back.
                self.release()
            raise

    def release(self) -> None:
        """Release a slot and process the next queued request."""
        if self._active_count <= 0:
            # Prevent negative active count from unpaired release() calls
            return
        self._active_count -= 1
        if self._on_slot_released:
            self._on_slot_released(len(self._waiting))
        self._process_queue()

    def update_rate_limit_state(self, parsed: ParsedRateLimitHeaders) -> None:
        """Update rate limit state from parsed headers."""
        if "remainingRequests" in parsed:
            self._remaining_requests = parsed["remainingRequests"]
        if "limitRequests" in parsed:
            self._request_limit = parsed["limitRequests"]
        if "remainingTokens" in parsed:
            self._remaining_tokens = parsed["remainingTokens"]
        if "limitTokens" in parsed:
            self._token_limit = parsed["limitTokens"]
        if "resetAt" in parsed:
            self._reset_at = parsed["resetAt"]
            self._schedule_reset_processing()

    def mark_rate_limited(
        self, retry_after_ms: float | None = None, selected_reset_at: float | None = None
    ) -> None:
        """Mark that a rate limit was hit.

        Only updates resetAt if there is no later reset time already. A selected absolute
        deadline takes precedence over a relative retry delay.
        """
        self._remaining_requests = 0
        self._remaining_tokens = 0

        new_reset_at = selected_reset_at
        if new_reset_at is None and retry_after_ms is not None and retry_after_ms >= 0:
            new_reset_at = clock.date_now() + retry_after_ms
        if new_reset_at is not None:
            # Preserve explicit elapsed/zero deadlines and any later known reset.
            existing = self._reset_at if self._reset_at is not None else new_reset_at
            self._reset_at = max(existing, new_reset_at)
        elif self._reset_at is None:
            # No retryAfter provided and no existing reset: use a conservative default
            self._reset_at = clock.date_now() + 60000
        # With an existing resetAt and no retryAfterMs, keep the existing one

        self._schedule_reset_processing()

    def set_max_concurrency(self, value: int) -> None:
        """Adjust max concurrency (called by the adaptive algorithm)."""
        self._max_concurrency = max(self._min_concurrency, value)
        # If we now have capacity, process queue
        self._process_queue()

    def get_max_concurrency(self) -> int:
        return self._max_concurrency

    def get_active_count(self) -> int:
        return self._active_count

    def get_queue_depth(self) -> int:
        return len(self._waiting)

    def get_reset_at(self) -> float | None:
        return self._reset_at

    def _is_quota_exhausted(self) -> bool:
        """Whether quota is exhausted (should wait for reset). Checks request AND token quotas.

        Intentional side effect: clears stale quota state once the reset time has passed, so
        outdated rate limit info never blocks indefinitely.
        """
        now = clock.date_now()

        # JS truthiness: a resetAt of 0 counts as unset.
        if self._reset_at and now >= self._reset_at:
            self._remaining_requests = None
            self._remaining_tokens = None
            self._reset_at = None
            return False

        def quota_exhausted(remaining: float | None) -> bool:
            return (
                remaining is not None
                and remaining <= 0
                and bool(self._reset_at)
                and now < (self._reset_at or 0)
            )

        return quota_exhausted(self._remaining_requests) or quota_exhausted(self._remaining_tokens)

    def _clear_reset_timer(self) -> None:
        if self._reset_timer is not None:
            self._reset_timer.cancel()
        self._reset_timer = None

    def _schedule_reset_processing(self) -> None:
        """Schedule queue processing for when the rate limit window resets."""
        self._clear_reset_timer()

        if self._reset_at and self._waiting:
            delay = max(0.0, self._reset_at - clock.date_now())

            def on_reset() -> None:
                self._reset_timer = None
                # Clear exhausted state
                self._remaining_requests = None
                self._remaining_tokens = None
                self._reset_at = None
                self._process_queue()

            self._reset_timer = _call_later(delay, on_reset)

    def _process_queue(self) -> None:
        """Grant queued requests up to available capacity. Synchronous: no awaits."""
        while (
            self._waiting
            and self._active_count < self._max_concurrency
            and not self._is_quota_exhausted()
        ):
            request = self._waiting.pop(0)
            request.resolve()

        # If the queue still has items and quota is exhausted, ensure a reset is scheduled
        if self._waiting and self._is_quota_exhausted():
            self._schedule_reset_processing()
        elif self._reset_timer is not None:
            self._clear_reset_timer()

    def get_remaining_ratio(self) -> dict[str, float | None]:
        """Ratio of remaining/limit for requests and tokens (None when unknown)."""
        request_ratio: float | None = None
        token_ratio: float | None = None

        if (
            self._remaining_requests is not None
            and self._request_limit is not None
            and self._request_limit > 0
        ):
            request_ratio = self._remaining_requests / self._request_limit

        if (
            self._remaining_tokens is not None
            and self._token_limit is not None
            and self._token_limit > 0
        ):
            token_ratio = self._remaining_tokens / self._token_limit

        return {"requests": request_ratio, "tokens": token_ratio}

    def dispose(self) -> None:
        """Cleanup resources, rejecting pending acquire() calls with 'Queue disposed'."""
        self._clear_reset_timer()
        waiting, self._waiting = self._waiting, []
        for request in waiting:
            request.reject(Exception("Queue disposed"))


def _ms(value: float) -> str:
    from promptphoo.jsvalues import number_to_string

    return number_to_string(value)
