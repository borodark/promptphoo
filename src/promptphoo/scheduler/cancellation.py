"""Cancellation-aware waits. Ports src/scheduler/cancellation.ts.

Node passes an `AbortSignal`; the Python port keeps that contract with a small
`AbortController` / `AbortSignal` pair so callers can cancel queue and retry waits for one
call. Ordinary asyncio task cancellation also works everywhere: every wait here is an
`await` that propagates `CancelledError` and cleans up its timer and listener.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable


class AbortError(Exception):
    """The default abort reason (Node's DOMException named "AbortError")."""

    name = "AbortError"

    def __init__(self, message: str = "This operation was aborted") -> None:
        super().__init__(message)
        self.message = message


class AbortSignal:
    def __init__(self) -> None:
        self.aborted = False
        self.reason: BaseException | None = None
        self._listeners: list[Callable[[], None]] = []

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def throw_if_aborted(self) -> None:
        if self.aborted:
            raise self.reason or AbortError()

    def _abort(self, reason: BaseException | None) -> None:
        if self.aborted:
            return
        self.aborted = True
        self.reason = reason if reason is not None else AbortError()
        listeners, self._listeners = self._listeners, []
        for listener in listeners:
            listener()

    @staticmethod
    def abort(reason: BaseException | None = None) -> AbortSignal:
        signal = AbortSignal()
        signal._abort(reason)
        return signal


class AbortController:
    def __init__(self) -> None:
        self.signal = AbortSignal()

    def abort(self, reason: BaseException | None = None) -> None:
        self.signal._abort(reason)


def throw_if_aborted(signal: AbortSignal | None = None) -> None:
    if signal is not None and signal.aborted:
        raise signal.reason or AbortError()


async def sleep_with_abort(ms: float, signal: AbortSignal | None = None) -> None:
    throw_if_aborted(signal)
    delay = max(0.0, ms) / 1000
    if signal is None:
        await asyncio.sleep(delay)
        return
    loop = asyncio.get_running_loop()
    future: asyncio.Future[None] = loop.create_future()

    def on_abort() -> None:
        if not future.done():
            future.set_exception(signal.reason or AbortError())

    def on_timer() -> None:
        if not future.done():
            future.set_result(None)

    timer = loop.call_later(delay, on_timer)
    signal.add_listener(on_abort)
    try:
        await future
    finally:
        timer.cancel()
        signal.remove_listener(on_abort)
