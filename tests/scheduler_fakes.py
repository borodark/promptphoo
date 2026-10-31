"""Virtual time for scheduler tests (the Python counterpart of vitest's fake timers).

`FakeTimeLoop` is an asyncio loop whose clock only moves when every task is blocked on a
timer: it then jumps straight to the next deadline instead of sleeping. `clock.date_now` is
patched to follow the same virtual clock, so retry backoff and rate-limit resets of minutes
run instantly and deterministically.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from promptphoo.scheduler import clock

START_MS = 1_786_132_800_000.0  # 2026-08-07T20:00:00Z


class FakeTimeLoop(asyncio.SelectorEventLoop):
    def __init__(self, start_ms: float = START_MS) -> None:
        super().__init__()
        self._virtual = 0.0
        self.start_ms = start_ms
        selector: Any = self._selector  # type: ignore[attr-defined]
        real_select = selector.select

        def select(timeout: float | None = None) -> Any:
            if timeout is None:
                raise RuntimeError("deadlock: no runnable task and no pending timer")
            if timeout > 0:
                self._virtual += timeout
            return real_select(0)

        selector.select = select

    def time(self) -> float:
        return self._virtual

    def now_ms(self) -> float:
        return self.start_ms + self._virtual * 1000


class FakeTime:
    def __init__(self, monkeypatch: Any) -> None:
        self._monkeypatch = monkeypatch
        self.loop: FakeTimeLoop | None = None
        self.start_ms = START_MS

    def now(self) -> float:
        return self.loop.now_ms() if self.loop else self.start_ms

    def run(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        self.loop = FakeTimeLoop(self.start_ms)
        self._monkeypatch.setattr(clock, "date_now", self.now)
        try:
            return self.loop.run_until_complete(fn())
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.close()


async def advance(ms: float) -> None:
    """Let virtual time pass (runs every timer due within `ms`)."""
    await asyncio.sleep(ms / 1000)


async def flush() -> None:
    """Run every ready callback without moving the clock."""
    for _ in range(10):
        await asyncio.sleep(0)
