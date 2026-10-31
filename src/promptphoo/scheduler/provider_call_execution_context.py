"""Runtime-only scheduler context for provider calls made below the evaluator.

Ports src/scheduler/providerCallExecutionContext.ts. Node uses AsyncLocalStorage; here each
store is a ContextVar, so tasks spawned inside a scope inherit it.

This keeps scheduler internals out of the call-API context dict, which providers can inspect,
while still letting matcher helpers reuse the evaluator's cancellation and rate-limit
orchestration.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

if TYPE_CHECKING:
    from promptphoo.scheduler.cancellation import AbortSignal

T = TypeVar("T")


class ProviderCallExecutionContext(TypedDict, total=False):
    abortSignal: AbortSignal | None
    providerCallQueue: Any
    rateLimitRegistry: Any
    # Provider whose evaluator-owned slot is active for this call.
    rateLimitProvider: Any
    # A child scheduler owns retries; the parent must not replay the whole operation.
    onNestedScheduledCall: Callable[[], None]
    # Preserve completed target work if a CLI pause interrupts its enclosing strategy.
    onTargetResponse: Callable[[str, dict[str, Any]], None]


class ProviderCallTracingContext(TypedDict, total=False):
    """Runtime-only instrumentation hooks injected by the evaluator for one traced execution."""

    getActiveTraceparent: Callable[[], str | None]
    testIndex: int
    withGraderSpan: Callable[..., Awaitable[Any]]
    withProviderSpan: Callable[..., Awaitable[Any]]


_provider_call_execution_context: ContextVar[ProviderCallExecutionContext | None] = ContextVar(
    "provider_call_execution_context", default=None
)
_provider_call_tracing_context: ContextVar[ProviderCallTracingContext | None] = ContextVar(
    "provider_call_tracing_context", default=None
)


def get_provider_call_execution_context() -> ProviderCallExecutionContext | None:
    return _provider_call_execution_context.get()


async def with_provider_call_execution_context(
    context: ProviderCallExecutionContext, fn: Callable[[], Awaitable[T]]
) -> T:
    token = _provider_call_execution_context.set(context)
    try:
        return await fn()
    finally:
        _provider_call_execution_context.reset(token)


# Abandoned calls keep running after their caller stops waiting (as in Node); hold a reference
# so the task is not garbage-collected mid-flight.
_background_calls: set[asyncio.Future[Any]] = set()


async def run_provider_call_with_abort(
    call: Callable[[], Awaitable[T]], signal: AbortSignal | None = None
) -> T:
    """Run `call`, but stop waiting for it (raising `signal.reason`) once `signal` aborts.

    As in Node, the abandoned call is not cancelled. A failure already unwinding in the same
    loop turn wins over the abort. Cancelling the awaiting task cancels the call too.
    """
    if signal is None:
        return await call()
    signal.throw_if_aborted()
    loop = asyncio.get_running_loop()
    outer: asyncio.Future[T] = loop.create_future()
    pending_abort: asyncio.Handle | None = None

    def finish(complete: Callable[[], None]) -> None:
        if outer.done():
            return
        signal.remove_listener(on_abort)
        if pending_abort is not None:
            pending_abort.cancel()
        complete()

    def on_abort() -> None:
        nonlocal pending_abort

        def reject() -> None:
            reason = signal.reason
            assert reason is not None
            outer.set_exception(reason)

        # Preserve a provider failure already unwinding this turn, then stop waiting.
        pending_abort = loop.call_soon(lambda: finish(reject))

    signal.add_listener(on_abort)
    inner = asyncio.ensure_future(call())
    _background_calls.add(inner)

    def on_done(task: asyncio.Future[T]) -> None:
        _background_calls.discard(task)
        if task.cancelled():
            finish(lambda: _cancel(outer))
            return
        error = task.exception()
        if error is not None:
            finish(lambda: outer.set_exception(error))
        else:
            finish(lambda: outer.set_result(task.result()))

    inner.add_done_callback(on_done)
    try:
        return await outer
    except asyncio.CancelledError:
        inner.cancel()
        raise


def _cancel(future: asyncio.Future[Any]) -> None:
    future.cancel()


def get_provider_call_tracing_context() -> ProviderCallTracingContext | None:
    return _provider_call_tracing_context.get()


async def with_provider_call_tracing_context(
    tracing_context: ProviderCallTracingContext, fn: Callable[[], Awaitable[T]]
) -> T:
    token = _provider_call_tracing_context.set(tracing_context)
    try:
        return await fn()
    finally:
        _provider_call_tracing_context.reset(token)
