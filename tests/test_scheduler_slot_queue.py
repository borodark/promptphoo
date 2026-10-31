"""Ports test/scheduler/slotQueue.test.ts (virtual time instead of vitest fake timers)."""

from __future__ import annotations

import asyncio

import pytest
from scheduler_fakes import FakeTime, advance, flush

from promptphoo.scheduler import clock
from promptphoo.scheduler.cancellation import AbortController, AbortSignal
from promptphoo.scheduler.slot_queue import SlotQueue


@pytest.fixture
def fake(monkeypatch):
    return FakeTime(monkeypatch)


async def start(coro):
    """Begin an acquire (Node's un-awaited promise) and let it reach the queue."""
    task = asyncio.ensure_future(coro)
    await flush()
    return task


def test_initial_state(fake):
    queue = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
    assert queue.get_max_concurrency() == 5
    assert queue.get_active_count() == 0
    assert queue.get_queue_depth() == 0
    assert queue.get_reset_at() is None


def test_default_queue_timeout(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 0, "minConcurrency": 0})
        task = await start(queue.acquire("test-1"))
        await advance(4 * 60 * 1000)
        assert queue.get_queue_depth() == 1
        await advance(60 * 1000 + 1)
        with pytest.raises(Exception, match="timed out after 300000ms in queue"):
            await task

    fake.run(body)


def test_custom_queue_timeout(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 1, "minConcurrency": 1, "queueTimeoutMs": 1000})
        await queue.acquire("test-1")
        assert queue.get_active_count() == 1
        task = await start(queue.acquire("test-2"))
        assert queue.get_queue_depth() == 1
        await advance(1001)
        with pytest.raises(Exception, match="Request test-2 timed out after 1000ms in queue"):
            await task
        assert queue.get_queue_depth() == 0

    fake.run(body)


def test_basic_allocation_and_callbacks(fake):
    async def body():
        acquired, released = [], []
        queue = SlotQueue({"maxConcurrency": 3, "minConcurrency": 1,
                           "onSlotAcquired": acquired.append,
                           "onSlotReleased": released.append})  # fmt: skip
        assert await queue.acquire("test-1") is None
        assert queue.get_active_count() == 1
        queue.release()
        assert queue.get_active_count() == 0
        for i in range(3):
            await queue.acquire(f"t{i}")
        assert queue.get_active_count() == 3
        waiting = await start(queue.acquire("t3"))
        assert (queue.get_active_count(), queue.get_queue_depth()) == (3, 1)
        queue.release()
        await waiting
        assert acquired == [0, 0, 0, 0, 0]
        assert released == [0, 1]

    fake.run(body)


def test_fifo_and_no_over_allocation(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 2, "minConcurrency": 1})
        order: list[int] = []
        peak = 0

        async def worker(i: int) -> None:
            nonlocal peak
            await queue.acquire(f"r{i}")
            order.append(i)
            peak = max(peak, queue.get_active_count())
            await advance(10)
            queue.release()

        await asyncio.gather(*(worker(i) for i in range(10)))
        assert order == list(range(10))
        assert peak == 2
        assert queue.get_active_count() == 0

    fake.run(body)


def test_update_rate_limit_state_and_ratios(fake):
    queue = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
    assert queue.get_remaining_ratio() == {"requests": None, "tokens": None}
    queue.update_rate_limit_state({"remainingRequests": 25, "limitRequests": 100})
    assert queue.get_remaining_ratio() == {"requests": 0.25, "tokens": None}
    queue.update_rate_limit_state({"remainingTokens": 500, "limitTokens": 1000})
    assert queue.get_remaining_ratio() == {"requests": 0.25, "tokens": 0.5}
    queue.update_rate_limit_state({"resetAt": 12345})
    assert queue.get_reset_at() == 12345
    zero = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
    zero.update_rate_limit_state({"remainingRequests": 0, "limitRequests": 0})
    assert zero.get_remaining_ratio()["requests"] is None
    zero.update_rate_limit_state({"limitRequests": 10})
    assert zero.get_remaining_ratio()["requests"] == 0


def test_mark_rate_limited_deadlines(fake):
    async def body():
        now = clock.date_now()
        queue = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
        queue.mark_rate_limited(60000)
        assert queue.get_reset_at() == now + 60000
        default = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
        default.mark_rate_limited()
        assert default.get_reset_at() == now + 60000

        later = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
        later.update_rate_limit_state({"resetAt": now + 120000})
        later.mark_rate_limited(None, now + 5000)
        assert later.get_reset_at() == now + 120000

        longer = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
        longer.update_rate_limit_state({"resetAt": now + 30000})
        longer.mark_rate_limited(60000)
        assert longer.get_reset_at() == now + 60000

        keep = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
        keep.update_rate_limit_state({"resetAt": now + 30000})
        keep.mark_rate_limited()
        assert keep.get_reset_at() == now + 30000
        keep.mark_rate_limited(0)
        assert keep.get_reset_at() == now + 30000
        keep.mark_rate_limited(10000)
        assert keep.get_reset_at() == now + 30000

    fake.run(body)


@pytest.mark.parametrize("offset", [-1000, 0, 5000])
def test_selected_absolute_deadline(fake, offset):
    async def body():
        queue = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
        selected = clock.date_now() + offset
        await advance(25)
        queue.mark_rate_limited(None, selected)
        assert queue.get_reset_at() == selected
        task = await start(queue.acquire("selected-deadline"))
        if offset > 0:
            assert not task.done()
            await advance(selected - clock.date_now())
        await task
        assert queue.get_queue_depth() == 0
        queue.release()
        assert queue.get_active_count() == 0

    fake.run(body)


def test_absolute_zero_is_immediate(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
        queue.mark_rate_limited(None, 0)
        assert queue.get_reset_at() == 0
        await queue.acquire("elapsed-zero")
        assert queue.get_queue_depth() == 0

    fake.run(body)


def test_blocks_until_reset(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
        queue.update_rate_limit_state({"remainingRequests": 10, "limitRequests": 100})
        await queue.acquire("test-1")
        queue.mark_rate_limited(5000)
        task = await start(queue.acquire("test-2"))
        assert (queue.get_queue_depth(), queue.get_active_count()) == (1, 1)
        await advance(5001)
        assert task.done()
        assert (queue.get_active_count(), queue.get_queue_depth()) == (2, 0)

    fake.run(body)


@pytest.mark.parametrize("field", ["remainingRequests", "remainingTokens"])
def test_quota_exhaustion_blocks(fake, field):
    async def body():
        now = clock.date_now()
        queue = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
        queue.update_rate_limit_state({field: 0, "resetAt": now + 5000})  # type: ignore[misc]
        task = await start(queue.acquire("test-1"))
        assert queue.get_queue_depth() == 1
        await advance(5001)
        assert task.done()
        assert (queue.get_active_count(), queue.get_queue_depth()) == (1, 0)
        assert queue.get_reset_at() is None

        open_queue = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
        open_queue.update_rate_limit_state({field: 5, "resetAt": clock.date_now() + 5000})  # type: ignore[misc]
        await open_queue.acquire("ok")
        assert open_queue.get_active_count() == 1

    fake.run(body)


def test_acquire_after_reset_without_timer(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
        queue.update_rate_limit_state({"remainingRequests": 0,
                                       "resetAt": clock.date_now() + 3000})  # fmt: skip
        await advance(3001)
        await queue.acquire("test-1")
        assert queue.get_active_count() == 1

    fake.run(body)


def test_reset_timer_processes_queue_and_reschedules(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
        queue.update_rate_limit_state({"remainingRequests": 0,
                                       "resetAt": clock.date_now() + 10000})  # fmt: skip
        task = await start(queue.acquire("test-1"))
        await advance(1000)
        queue.update_rate_limit_state({"resetAt": clock.date_now() + 2000})
        await advance(1999)
        assert queue.get_queue_depth() == 1
        await advance(1)
        await flush()
        assert task.done()
        assert queue.get_active_count() == 1

    fake.run(body)


def test_aborted_signal_takes_free_slot_but_does_not_wait(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 1, "minConcurrency": 1})
        signal = AbortSignal.abort(Exception("eval paused"))
        assert await queue.acquire("free", signal) is None
        with pytest.raises(Exception, match="eval paused"):
            await queue.acquire("waiting", signal)
        assert (queue.get_active_count(), queue.get_queue_depth()) == (1, 0)

    fake.run(body)


def test_abort_while_waiting(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 1, "minConcurrency": 1})
        await queue.acquire("held")
        controller = AbortController()
        reason = Exception("stop")
        task = await start(queue.acquire("waiting", controller.signal))
        controller.abort(reason)
        with pytest.raises(Exception) as exc:
            await task
        assert exc.value is reason
        assert queue.get_queue_depth() == 0

    fake.run(body)


def test_task_cancellation_does_not_leak_slots(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 1, "minConcurrency": 1})
        await queue.acquire("held")
        task = await start(queue.acquire("waiting"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert queue.get_queue_depth() == 0

        # Granted, then cancelled before the waiter resumed: the slot is handed back.
        task = await start(queue.acquire("granted"))
        queue.release()  # grants "granted" synchronously
        assert queue.get_active_count() == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert queue.get_active_count() == 0

    fake.run(body)


def test_timeout_zero_disables(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 1, "minConcurrency": 1, "queueTimeoutMs": 0})
        await queue.acquire("test-1")
        await start(queue.acquire("test-2"))
        await advance(1_000_000)
        assert queue.get_queue_depth() == 1
        queue.dispose()

    fake.run(body)


def test_timeout_cleared_when_processed(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 1, "minConcurrency": 1, "queueTimeoutMs": 10000})
        await queue.acquire("test-1")
        task = await start(queue.acquire("test-2"))
        queue.release()
        assert await task is None
        await advance(20000)
        assert queue.get_active_count() == 1

    fake.run(body)


def test_dispose(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 1, "minConcurrency": 1, "queueTimeoutMs": 5000})
        await queue.acquire("test-1")
        second = await start(queue.acquire("test-2"))
        third = await start(queue.acquire("test-3"))
        assert queue.get_queue_depth() == 2
        queue.dispose()
        await advance(10000)
        for task in (second, third):
            with pytest.raises(Exception, match="Queue disposed"):
                await task
        assert queue.get_queue_depth() == 0

        reset = SlotQueue({"maxConcurrency": 1, "minConcurrency": 1})
        reset.update_rate_limit_state({"remainingRequests": 0,
                                       "resetAt": clock.date_now() + 60000})  # fmt: skip
        pending = await start(reset.acquire("test-1"))
        reset.dispose()
        with pytest.raises(Exception, match="Queue disposed"):
            await pending
        await advance(100000)

    fake.run(body)


def test_set_max_concurrency(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 5, "minConcurrency": 2})
        queue.set_max_concurrency(10)
        assert queue.get_max_concurrency() == 10
        queue.set_max_concurrency(1)
        assert queue.get_max_concurrency() == 2

        queue = SlotQueue({"maxConcurrency": 5, "minConcurrency": 2})
        for i in range(5):
            await queue.acquire(f"t{i}")
        await start(queue.acquire("t5"))
        await start(queue.acquire("t6"))
        assert (queue.get_active_count(), queue.get_queue_depth()) == (5, 2)
        queue.set_max_concurrency(7)
        assert (queue.get_active_count(), queue.get_queue_depth()) == (7, 0)

    fake.run(body)


def test_unbalanced_release(fake):
    async def body():
        queue = SlotQueue({"maxConcurrency": 5, "minConcurrency": 1})
        for _ in range(3):
            queue.release()
        assert queue.get_active_count() == 0
        await queue.acquire("test-1")
        queue.release()
        queue.release()
        assert queue.get_active_count() == 0

    fake.run(body)
