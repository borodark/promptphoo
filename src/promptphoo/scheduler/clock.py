"""`Date.now()` and `Math.random()` indirection for the scheduler.

The scheduler compares absolute Unix-millisecond deadlines (from rate-limit headers) with the
current time, like Node's `Date.now()`. Tests replace `date_now` (and drive asyncio timers with
a virtual-time loop) instead of sleeping.
"""

from __future__ import annotations

import random as _random
import time


def date_now() -> float:
    return time.time() * 1000


def math_random() -> float:
    return _random.random()
