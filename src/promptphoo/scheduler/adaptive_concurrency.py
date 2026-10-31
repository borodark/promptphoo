"""Adaptive concurrency (AIMD-style). Ports src/scheduler/adaptiveConcurrency.ts."""

from __future__ import annotations

import math
from typing import Literal, TypedDict

DEFAULT_MIN_CONCURRENCY = 1
BACKOFF_FACTOR = 0.5  # Halve on rate limit
RECOVERY_FACTOR = 1.5  # +50% on recovery
RECOVERY_THRESHOLD = 5  # Successes before recovery
WARNING_THRESHOLD = 0.1  # 10% remaining triggers proactive reduction

ChangeReason = Literal["recovery", "ratelimit", "proactive"]


class ConcurrencyChangeResult(TypedDict):
    changed: bool
    previous: int
    current: int
    reason: ChangeReason


def _concurrency_change(
    previous: int, current: int, reason: ChangeReason, changed: bool | None = None
) -> ConcurrencyChangeResult:
    return {
        "changed": previous != current if changed is None else changed,
        "previous": previous,
        "current": current,
        "reason": reason,
    }


class AdaptiveConcurrency:
    """Manages adaptive concurrency based on rate limit feedback.

    Recovery path with constants (initial=10, min=1):
    1 -> ceil(1.5) = 2, 2 -> 3, 3 -> 5, 5 -> 8, 8 -> 10 (capped at initial), 5 successes each.
    Total: 25 requests to fully recover from min=1 to initial=10.
    """

    def __init__(self, initial: int, min_concurrency: int = DEFAULT_MIN_CONCURRENCY) -> None:
        self._initial = initial
        self._current = initial
        # Clamp min to be at least 1 and at most initial
        self._min = min(initial, max(1, min_concurrency))
        self._consecutive_successes = 0

    def record_success(self) -> ConcurrencyChangeResult:
        """Called on a successful request. May increase concurrency after sustained success."""
        self._consecutive_successes += 1

        if self._consecutive_successes >= RECOVERY_THRESHOLD and self._current < self._initial:
            previous = self._current
            self._current = min(self._initial, math.ceil(self._current * RECOVERY_FACTOR))
            self._consecutive_successes = 0
            return _concurrency_change(previous, self._current, "recovery")

        return _concurrency_change(self._current, self._current, "recovery", False)

    def record_rate_limit(self) -> ConcurrencyChangeResult:
        """Called on a rate limit (429). Reduces concurrency immediately."""
        self._consecutive_successes = 0
        previous = self._current
        self._current = max(self._min, math.floor(self._current * BACKOFF_FACTOR))
        return _concurrency_change(previous, self._current, "ratelimit")

    def record_approaching_limit(self, ratio: float) -> ConcurrencyChangeResult:
        """Called when approaching the rate limit; reduces concurrency by the remaining ratio.

        At 10% remaining keep 60% of current, at 5% keep 40%, at 1% keep ~24%:
        reductionFactor = 0.2 + (ratio / WARNING_THRESHOLD) * 0.4
        """
        # Clamp ratio to [0, 1] to handle edge cases
        clamped_ratio = max(0.0, min(1.0, ratio))

        if clamped_ratio >= WARNING_THRESHOLD or self._current <= self._min:
            return _concurrency_change(self._current, self._current, "proactive", False)

        previous = self._current
        reduction_factor = 0.2 + (clamped_ratio / WARNING_THRESHOLD) * 0.4
        self._current = max(self._min, math.floor(self._current * reduction_factor))
        return _concurrency_change(previous, self._current, "proactive")

    def get_current(self) -> int:
        return self._current

    def get_min(self) -> int:
        return self._min

    def get_initial(self) -> int:
        return self._initial
