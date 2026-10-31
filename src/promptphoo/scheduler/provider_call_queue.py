"""Provider-grouped call queue for deferred grading. Ports src/scheduler/providerCallQueue.ts.

Node binds each call to its AsyncResource so it runs in the enqueuer's async context; the
Python port captures the enqueuer's `contextvars` context and runs the call inside it.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from promptphoo.scheduler.cancellation import throw_if_aborted

if TYPE_CHECKING:
    from promptphoo.scheduler.cancellation import AbortSignal

T = TypeVar("T")


@dataclass(eq=False)
class QueuedProviderCall:
    call: Callable[[], Awaitable[Any]]
    providerId: str
    reject: Callable[[BaseException], None]
    resolve: Callable[[Any], None]


class ProviderCallQueue(Protocol):
    async def enqueue(
        self,
        provider_id: str,
        call: Callable[[], Awaitable[T]],
        abort_signal: AbortSignal | None = None,
    ) -> T: ...


async def _as_coroutine(call: Callable[[], Awaitable[T]]) -> T:
    return await call()


class ProviderGroupedCallQueue:
    def __init__(self) -> None:
        self._jobs: list[QueuedProviderCall] = []
        self._waiters: list[Callable[[], None]] = []

    async def enqueue(
        self,
        provider_id: str,
        call: Callable[[], Awaitable[T]],
        abort_signal: AbortSignal | None = None,
    ) -> T:
        context = contextvars.copy_context()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        throw_if_aborted(abort_signal)

        def on_abort() -> None:
            if job in self._jobs:
                self._jobs.remove(job)
            if not future.done():
                assert abort_signal is not None
                future.set_exception(abort_signal.reason or Exception("aborted"))

        async def bound_call() -> T:
            if abort_signal is not None:
                abort_signal.remove_listener(on_abort)
            throw_if_aborted(abort_signal)
            return await loop.create_task(_as_coroutine(call), context=context)

        def resolve(value: Any) -> None:
            if not future.done():
                future.set_result(value)

        def reject(error: BaseException) -> None:
            if not future.done():
                future.set_exception(error)

        job = QueuedProviderCall(call=bound_call, providerId=provider_id, reject=reject,
                                 resolve=resolve)  # fmt: skip
        self._jobs.append(job)
        if abort_signal is not None:
            abort_signal.add_listener(on_abort)
        self._notify_waiters()
        try:
            return await future
        except asyncio.CancelledError:
            if job in self._jobs:
                self._jobs.remove(job)
            raise

    def has_jobs(self) -> bool:
        return len(self._jobs) > 0

    def take_next_group(self, preferred_provider_id: str | None = None) -> list[QueuedProviderCall]:
        if not self._jobs:
            return []

        provider_id = (
            preferred_provider_id
            if preferred_provider_id
            and any(job.providerId == preferred_provider_id for job in self._jobs)
            else self._jobs[0].providerId
        )
        group = [job for job in self._jobs if job.providerId == provider_id]
        self._jobs = [job for job in self._jobs if job.providerId != provider_id]
        return group

    async def wait_for_job(self) -> None:
        if self.has_jobs():
            return
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        def wake() -> None:
            if not future.done():
                future.set_result(None)

        self._waiters.append(wake)
        await future

    async def run(self, job: QueuedProviderCall) -> None:
        try:
            job.resolve(await job.call())
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            job.reject(error)

    def _notify_waiters(self) -> None:
        waiters, self._waiters = self._waiters, []
        for waiter in waiters:
            waiter()
