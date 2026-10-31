"""Ports test/scheduler/{providerRateLimitState,rateLimitRegistry,providerWrapper}.test.ts and
the provider-agnostic parts of the rateLimitRegistry behaviour/cancellation specs."""

from __future__ import annotations

import asyncio
import logging

import pytest
from scheduler_fakes import FakeTime, advance, flush

from promptphoo.fetch_context import get_fetch_retry_context_max_retries
from promptphoo.providers.base import ApiProvider
from promptphoo.scheduler import clock
from promptphoo.scheduler.cancellation import AbortController
from promptphoo.scheduler.provider_call_execution_context import (
    get_provider_call_execution_context,
    with_provider_call_execution_context,
)
from promptphoo.scheduler.provider_rate_limit_state import (
    ProviderRateLimitState,
    RateLimitExhaustedError,
)
from promptphoo.scheduler.provider_wrapper import (
    create_provider_rate_limit_options,
    is_rate_limit_wrapped,
    wrap_provider_with_rate_limiting,
    wrap_providers_with_rate_limiting,
)
from promptphoo.scheduler.rate_limit_key import get_rate_limit_key
from promptphoo.scheduler.rate_limit_registry import RateLimitRegistry

FAST_RETRY_POLICY = {"maxRetries": 3, "baseDelayMs": 1, "maxDelayMs": 10, "jitterFactor": 0}
NO_RETRY_POLICY = {"maxRetries": 0, "baseDelayMs": 1, "maxDelayMs": 1, "jitterFactor": 0}


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.delenv("PROMPTFOO_DISABLE_ADAPTIVE_SCHEDULER", raising=False)
    return FakeTime(monkeypatch)


def make_state(policy=None, key="test-provider", max_concurrency=5):
    return ProviderRateLimitState({"rateLimitKey": key, "maxConcurrency": max_concurrency,
                                   "minConcurrency": 1,
                                   "retryPolicy": policy or FAST_RETRY_POLICY})  # fmt: skip


async def value(v):
    return v


# --- ProviderRateLimitState ---------------------------------------------------------------


def test_success_path_and_metrics(fake):
    async def body():
        state = make_state()
        assert await state.execute_with_retry("req-1", lambda _h: value("success"), {}) == (
            "success"
        )
        await state.execute_with_retry("req-2", lambda _h: value("success"), {})
        metrics = state.get_metrics()
        assert metrics["rateLimitKey"] == "test-provider"
        assert metrics["maxConcurrency"] == 5
        assert (metrics["totalRequests"], metrics["completedRequests"],
                metrics["failedRequests"]) == (2, 2, 0)  # fmt: skip

    fake.run(body)


def test_completes_after_cancellation(fake):
    async def body():
        state = make_state()
        controller = AbortController()

        async def call(_h):
            controller.abort(Exception("eval paused"))
            return "completed"

        assert await state.execute_with_retry("x", call, {"abortSignal": controller.signal}) == (
            "completed"
        )
        m = state.get_metrics()
        assert (m["activeRequests"], m["failedRequests"]) == (0, 0)

    fake.run(body)


@pytest.mark.parametrize("kind", ["response", "thrown error"])
def test_cancel_retry_sleep(fake, kind):
    async def body():
        state = make_state()
        controller = AbortController()
        reason = Exception("cancel scheduled retry")
        retrying = asyncio.Event()
        state.on("request:retrying", lambda _d: retrying.set())
        calls = 0

        async def call(_h):
            nonlocal calls
            calls += 1
            if kind == "thrown error":
                raise Exception("429 rate limit")
            return "rate limited"

        task = asyncio.ensure_future(state.execute_with_retry("cancel-retry", call, {
            "abortSignal": controller.signal,
            "isRateLimited": lambda r, e: r == "rate limited" or "429" in str(e or ""),
            "getRetryAfter": lambda r, e: 60_000,
        }))  # fmt: skip
        await retrying.wait()
        controller.abort(reason)
        with pytest.raises(Exception) as exc:
            await task
        assert exc.value is reason
        await advance(60_000)
        assert calls == 1
        m = state.get_metrics()
        assert (m["activeRequests"], m["queueDepth"], m["failedRequests"]) == (0, 0, 1)

    fake.run(body)


def test_error_propagation(fake):
    async def body():
        state = make_state()

        async def boom(_h):
            raise Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            await state.execute_with_retry("req-1", boom, {})
        m = state.get_metrics()
        assert (m["failedRequests"], m["completedRequests"]) == (1, 0)

    fake.run(body)


def test_finalizer_failure_releases_slot_once(fake):
    async def body():
        state = make_state()
        gate: asyncio.Future[str] = asyncio.get_running_loop().create_future()

        async def pending_call(_h):
            return await gate

        pending = asyncio.ensure_future(state.execute_with_retry("pending", pending_call, {}))
        await flush()
        calls = 0

        async def call(_h):
            nonlocal calls
            calls += 1
            return {"output": "result"}

        def finalize(_r, _rs):
            raise Exception("HTTP 503 in finalizer")

        with pytest.raises(Exception, match="HTTP 503 in finalizer"):
            await state.execute_with_retry("fin", call, {"finalizeResult": finalize})
        assert calls == 1
        m = state.get_metrics()
        assert (m["activeRequests"], m["totalRequests"], m["completedRequests"],
                m["failedRequests"]) == (1, 2, 0, 1)  # fmt: skip
        gate.set_result("complete")
        assert await pending == "complete"
        m = state.get_metrics()
        assert (m["activeRequests"], m["completedRequests"]) == (0, 1)

    fake.run(body)


def test_cancellation_during_error_backoff_is_counted(fake):
    async def body():
        state = make_state()
        controller = AbortController()

        async def call(_h):
            raise Exception("rate limit")

        task = asyncio.ensure_future(
            state.execute_with_retry("req-1", call, {"abortSignal": controller.signal})
        )
        await flush()
        controller.abort()
        with pytest.raises(Exception) as exc:
            await task
        assert getattr(exc.value, "name", None) == "AbortError"
        m = state.get_metrics()
        assert (m["totalRequests"], m["failedRequests"]) == (1, 1)

    fake.run(body)


def test_rate_limit_detection_and_exhaustion(fake):
    async def body():
        state = make_state(NO_RETRY_POLICY)
        hits = []
        state.on("ratelimit:hit", hits.append)
        with pytest.raises(RateLimitExhaustedError) as exc:
            await state.execute_with_retry(
                "req-1",
                lambda _h: value({"status": 429}),
                {"isRateLimited": lambda r, e: r["status"] == 429},
            )
        # fmt: skip
        assert str(exc.value) == "Rate limit exceeded for test-provider after 1 attempts"
        assert exc.value.result == {"status": 429}
        assert len(hits) == 1

        structured = await state.execute_with_retry(
            "req-2",
            lambda _h: value({"error": "rate limited", "status": 429}),
            {"isRateLimited": lambda r, e: (r or {}).get("status") == 429,
             "finalizeResult": lambda r, rs: r},
        )  # fmt: skip
        assert structured == {"error": "rate limited", "status": 429}

    fake.run(body)


@pytest.mark.parametrize("message", ["Rate limit exceeded", "Error 429", "Too Many Requests"])
def test_rate_limit_error_by_message(fake, message):
    async def body():
        state = make_state(NO_RETRY_POLICY)
        hits = []
        state.on("ratelimit:hit", hits.append)

        async def call(_h):
            raise Exception(message)

        with pytest.raises(Exception, match=message):
            await state.execute_with_retry("req-1", call, {})
        assert len(hits) == 1

    fake.run(body)


def test_transient_result_retry_and_finalize(fake):
    async def body():
        state = make_state()
        calls = 0

        async def call(_h):
            nonlocal calls
            calls += 1
            return {"status": 503} if calls == 1 else {"status": 200}

        result = await state.execute_with_retry("t", call, {
            "isRetryableResult": lambda r: r["status"] == 503,
            "finalizeResult": lambda r, rs: {**r, "priorStatuses": [x["status"] for x in rs]},
        })  # fmt: skip
        assert calls == 2
        assert result == {"status": 200, "priorStatuses": [503]}

    fake.run(body)


@pytest.mark.parametrize(("succeed_on", "retried"), [(2, 1), (3, 2)])
def test_retry_until_success(fake, succeed_on, retried):
    async def body():
        state = make_state()
        events = []
        state.on("request:retrying", events.append)
        attempt = 0

        async def call(_h):
            nonlocal attempt
            attempt += 1
            if attempt < succeed_on:
                raise Exception("Rate limit")
            return "success"

        assert await state.execute_with_retry("r", call, {"getRetryAfter": lambda r, e: 0}) == (
            "success"
        )
        assert attempt == succeed_on
        assert len(events) == retried
        assert (events[0]["attempt"], events[0]["reason"], events[0]["delayMs"]) == (
            1, "ratelimit", 0)  # fmt: skip
        assert state.get_metrics()["retriedRequests"] == retried

    fake.run(body)


def test_exponential_backoff_with_default_policy(fake):
    async def body():
        state = ProviderRateLimitState(
            {"rateLimitKey": "k", "maxConcurrency": 5, "minConcurrency": 1}
        )
        delays = []
        state.on("request:retrying", lambda d: delays.append(d["delayMs"]))
        start = clock.date_now()

        async def call(_h):
            raise Exception("HTTP 503: Service Unavailable")

        with pytest.raises(Exception, match="503"):
            await state.execute_with_retry("r", call, {})
        # 3 retries, 1s/2s/4s plus up to 20% jitter, all on the virtual clock
        assert len(delays) == 3
        for delay, base in zip(delays, (2000, 4000, 8000), strict=True):
            assert base <= delay <= base * 1.2
        assert clock.date_now() - start == pytest.approx(sum(delays))

    fake.run(body)


def test_headers_learned_once_and_retry_after_only_when_limited(fake):
    async def body():
        state = make_state()
        learned = []
        state.on("ratelimit:learned", learned.append)
        for i in range(2):
            await state.execute_with_retry(f"r{i}", lambda _h: value("success"), {
                "getHeaders": lambda r: {"x-ratelimit-limit-requests": "100"}})  # fmt: skip
        assert learned == [{"rateLimitKey": "test-provider", "requestLimit": 100,
                            "tokenLimit": None}]  # fmt: skip

        # retry-after on a successful response must not block the queue
        await state.execute_with_retry("ok", lambda _h: value("success"), {
            "getHeaders": lambda r: {"retry-after-ms": "5000",
                                     "x-ratelimit-remaining-requests": "50",
                                     "x-ratelimit-limit-requests": "100"},
            "isRateLimited": lambda r, e: False,
        })  # fmt: skip
        started = clock.date_now()
        assert await state.execute_with_retry("ok2", lambda _h: value("second"), {}) == "second"
        assert clock.date_now() == started
        assert state.get_metrics()["rateLimitHits"] == 0

        limited = make_state(NO_RETRY_POLICY)
        hits = []
        limited.on("ratelimit:hit", hits.append)
        with pytest.raises(RateLimitExhaustedError):
            await limited.execute_with_retry("rl", lambda _h: value("limited"), {
                "getHeaders": lambda r: {"retry-after-ms": "5000"},
                "isRateLimited": lambda r, e: True,
            })  # fmt: skip
        assert len(hits) == 1
        assert hits[0]["retryAfterMs"] is None
        assert hits[0]["resetAt"] == clock.date_now() + 5000

    fake.run(body)


def test_adaptive_concurrency_changes(fake):
    async def body():
        state = make_state(NO_RETRY_POLICY, max_concurrency=10)
        decreased, increased = [], []
        state.on("concurrency:decreased", decreased.append)
        state.on("concurrency:increased", increased.append)
        with pytest.raises(RateLimitExhaustedError):
            await state.execute_with_retry("rl", lambda _h: value("x"), {
                "isRateLimited": lambda r, e: True, "getRetryAfter": lambda r, e: 0})  # fmt: skip
        assert decreased[0]["previous"] == 10 and decreased[0]["current"] == 5
        assert state.get_metrics()["maxConcurrency"] == 5
        for i in range(5):
            await state.execute_with_retry(f"ok{i}", lambda _h: value("ok"), {})
        assert increased[0]["current"] == 8
        assert state.get_metrics()["maxConcurrency"] == 8

    fake.run(body)


def test_proactive_throttling(fake):
    async def body():
        state = make_state(max_concurrency=10)
        warnings, decreased = [], []
        state.on("ratelimit:warning", warnings.append)
        state.on("concurrency:decreased", decreased.append)
        await state.execute_with_retry("r", lambda _h: value("ok"), {
            "getHeaders": lambda r: {"x-ratelimit-remaining-requests": "5",
                                     "x-ratelimit-limit-requests": "100"}})  # fmt: skip
        assert warnings == [{"rateLimitKey": "test-provider", "requestRatio": 0.05,
                             "tokenRatio": None}]  # fmt: skip
        assert decreased[0]["reason"] == "proactive" and decreased[0]["current"] == 4

    fake.run(body)


def test_quota_exhausted_queue_waits_for_reset(fake):
    async def body():
        state = make_state()
        start = clock.date_now()
        await state.execute_with_retry("r", lambda _h: value("ok"), {
            "getHeaders": lambda r: {"x-ratelimit-remaining-requests": "0",
                                     "x-ratelimit-limit-requests": "100",
                                     "x-ratelimit-reset-requests": "20s"}})  # fmt: skip
        assert await state.execute_with_retry("r2", lambda _h: value("later"), {}) == "later"
        assert clock.date_now() - start == pytest.approx(20000)

    fake.run(body)


def test_response_headers_observer_updates_quota(fake):
    async def body():
        state = make_state()
        start = clock.date_now()

        async def call(on_headers):
            on_headers({"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "7s"})
            return "ok"

        await state.execute_with_retry("r", call, {})
        await state.execute_with_retry("r2", lambda _h: value("next"), {})
        assert clock.date_now() - start == pytest.approx(7000)

    fake.run(body)


def test_metrics_and_slot_events(fake):
    async def body():
        state = make_state()
        acquired, released = [], []
        state.on("slot:acquired", acquired.append)
        state.on("slot:released", released.append)

        async def slow(_h):
            await advance(100)
            return "ok"

        await state.execute_with_retry("r", slow, {})
        m = state.get_metrics()
        assert m["avgLatencyMs"] == pytest.approx(100)
        assert m["p50LatencyMs"] == pytest.approx(100)
        assert acquired == [{"rateLimitKey": "test-provider", "queueDepth": 0}]
        assert released == [{"rateLimitKey": "test-provider", "queueDepth": 0}]
        assert state.get_queue_depth() == 0
        state.dispose()
        assert state.listener_count("slot:acquired") == 0

    fake.run(body)


# --- RateLimitRegistry --------------------------------------------------------------------


class _Provider(ApiProvider):
    def __init__(self, provider_id="test-provider", config=None, responses=None):
        super().__init__({"id": provider_id, "config": config})
        self.responses = list(responses or [])
        self.calls: list[tuple] = []

    async def call_api(self, prompt, context=None, options=None):
        self.calls.append((prompt, context, options))
        item = self.responses.pop(0) if self.responses else {"output": prompt}
        if isinstance(item, BaseException):
            raise item
        return item


def test_registry_options_and_env(fake, monkeypatch):
    registry = RateLimitRegistry({"maxConcurrency": 10})
    assert registry._min_concurrency == 1
    assert registry._queue_timeout_ms == 300000
    assert registry._enabled
    monkeypatch.setenv("PROMPTFOO_MIN_CONCURRENCY", "3")
    monkeypatch.setenv("PROMPTFOO_SCHEDULER_QUEUE_TIMEOUT_MS", "0")
    monkeypatch.setenv("PROMPTFOO_DISABLE_ADAPTIVE_SCHEDULER", "true")
    registry = RateLimitRegistry({"maxConcurrency": 10})
    assert (registry._min_concurrency, registry._queue_timeout_ms) == (3, 0)
    assert not registry._enabled
    assert RateLimitRegistry({"maxConcurrency": 10, "minConcurrency": 2})._min_concurrency == 2


def test_registry_disabled_bypasses_state(fake, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_DISABLE_ADAPTIVE_SCHEDULER", "true")

    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 10})
        seen = []

        async def call(on_headers):
            seen.append((on_headers, get_fetch_retry_context_max_retries()))
            return "direct"

        provider = _Provider(config={"maxRetries": 2})
        assert await registry.execute(provider, call) == "direct"
        assert seen == [(None, 2)]
        assert registry.get_metrics() == {}

        async def boom(_h):
            raise Exception("fail")

        with pytest.raises(Exception, match="fail"):
            await registry.execute(provider, boom)

    fake.run(body)


@pytest.mark.parametrize(("raw", "expected"), [(0, 0), (5, 5), ("2", 2), (3.0, 3)])
def test_registry_forwards_max_retries(fake, raw, expected):
    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 10})
        calls = 0

        async def call(_h):
            nonlocal calls
            calls += 1
            assert get_fetch_retry_context_max_retries() == expected
            raise Exception("HTTP 503")

        with pytest.raises(Exception, match="503"):
            await registry.execute(_Provider(config={"maxRetries": raw}), call)
        assert calls == expected + 1

    fake.run(body)


@pytest.mark.parametrize("raw", [-1, 2.5, "2.5", "1" + "0" * 400, str(2**53 - 1) + "0", True])
def test_registry_warns_on_invalid_max_retries(fake, raw, caplog):
    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 10})
        with caplog.at_level(logging.WARNING):
            assert await registry.execute(_Provider(config={"maxRetries": raw}),
                                          lambda _h: value("ok")) == "ok"  # fmt: skip
        assert "[RateLimit] Ignoring invalid provider.config.maxRetries; expected a " \
               "non-negative integer." in caplog.text  # fmt: skip
        assert '"providerId":"test-provider"' in caplog.text

    fake.run(body)


def test_registry_sanitizes_url_ids_in_warning(fake, caplog):
    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 10})
        provider = _Provider("https://example.com/api?api_key=secret&model=test",
                             {"maxRetries": -1})  # fmt: skip
        with caplog.at_level(logging.WARNING):
            await registry.execute(provider, lambda _h: value("ok"))
        assert "api_key=%5BREDACTED%5D" in caplog.text
        assert "secret" not in caplog.text

    fake.run(body)


def test_registry_no_warning_without_max_retries(fake, caplog):
    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 10})
        with caplog.at_level(logging.WARNING):
            await registry.execute(_Provider(), lambda _h: value("ok"))
        assert caplog.text == ""

    fake.run(body)


def test_registry_handles_own_retries(fake):
    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 10})
        provider = _Provider(config={"maxRetries": 5})
        provider.handles_own_retries = True
        calls = 0

        async def call(_h):
            nonlocal calls
            calls += 1
            # fetch-level retries still see the configured budget
            assert get_fetch_retry_context_max_retries() == 5
            raise Exception("HTTP 503")

        with pytest.raises(Exception, match="503"):
            await registry.execute(provider, call)
        assert calls == 1

    fake.run(body)


def test_registry_states_events_and_metrics(fake):
    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 10})
        events: list[tuple[str, dict]] = []
        for name in ("request:started", "request:completed", "request:failed"):
            registry.on(name, lambda d, name=name: events.append((name, d)))
        a = _Provider("openai:gpt-4", {"apiKey": "sk-key-aaaa"})
        b = _Provider("openai:gpt-4", {"apiKey": "sk-key-bbbb"})
        assert registry.get_metrics() == {}
        await registry.execute(a, lambda _h: value("a"))
        await registry.execute(a, lambda _h: value("a2"))
        await registry.execute(b, lambda _h: value("b"))
        metrics = registry.get_metrics()
        assert set(metrics) == {get_rate_limit_key(a), get_rate_limit_key(b)}
        assert metrics[get_rate_limit_key(a)]["totalRequests"] == 2

        async def boom(_h):
            raise Exception("provider failed")

        with pytest.raises(Exception, match="provider failed"):
            await registry.execute(b, boom)
        names = [n for n, _ in events]
        assert names == ["request:started", "request:completed"] * 3 + [
            "request:started", "request:failed"]  # fmt: skip
        assert events[-1][1]["error"] == "Error: provider failed"
        ids = {d["requestId"] for n, d in events if n == "request:started"}
        assert len(ids) == 4
        assert all(i.startswith("openai:gpt-4[") for i in ids)

        registry.dispose()
        assert registry.get_metrics() == {}
        assert registry.listener_count("request:started") == 0

    fake.run(body)


def test_registry_retries_rate_limited_responses(fake):
    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 4})
        limited = {
            "error": "API error: 429 Too Many Requests",
            "metadata": {"http": {"status": 429, "headers": {"retry-after": "2"}}},
        }
        provider = _Provider(responses=[limited, limited, {"output": "ok"}])
        retrying = []
        registry.on("request:retrying", retrying.append)
        start = clock.date_now()
        result = await registry.execute(provider, lambda _h: provider.call_api("p"),
                                        create_provider_rate_limit_options())  # fmt: skip
        assert result == {"output": "ok"}
        assert len(provider.calls) == 3
        assert [r["reason"] for r in retrying] == ["ratelimit", "ratelimit"]
        assert clock.date_now() - start >= 4000

    fake.run(body)


def test_registry_preserves_exhausted_provider_error(fake):
    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 4})
        limited = {"output": "partial", "metadata": {"http": {"status": 429}}}
        provider = _Provider(config={"maxRetries": 1}, responses=[limited, limited])
        result = await registry.execute(provider, lambda _h: provider.call_api("p"),
                                        create_provider_rate_limit_options())  # fmt: skip
        assert result == {
            **limited,
            "error": "Rate limit exceeded for test-provider after 2 attempts",
        }
        # fmt: skip
        with_error = {"error": "rate limit", "metadata": {"http": {"status": 429}}}
        provider = _Provider(config={"maxRetries": 0}, responses=[with_error])
        result = await registry.execute(provider, lambda _h: provider.call_api("p"),
                                        create_provider_rate_limit_options())  # fmt: skip
        assert result is with_error

    fake.run(body)


def test_registry_does_not_retry_quota(fake):
    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 4})
        quota = {
            "error": "Quota exceeded: HTTP 429 (code: insufficient_quota)",
            "metadata": {
                "rateLimitKind": "quota",
                "http": {
                    "status": 429,
                    "headers": {
                        "x-ratelimit-remaining-requests": "0",
                        "x-ratelimit-reset-requests": "3600s",
                    },
                },
            },
        }
        # fmt: skip
        provider = _Provider(responses=[quota, {"output": "next"}])
        start = clock.date_now()
        assert (
            await registry.execute(
                provider, lambda _h: provider.call_api("p"), create_provider_rate_limit_options()
            )
            is quota
        )
        assert await registry.execute(
            provider, lambda _h: provider.call_api("p"), create_provider_rate_limit_options()
        ) == {"output": "next"}
        # fmt: skip
        assert clock.date_now() == start

    fake.run(body)


def test_registry_nested_call_vetoes_parent_retry(fake):
    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 4})
        parent = _Provider("parent")
        child = _Provider("child", responses=[{"output": "child"}])
        parent_calls = 0

        async def parent_call(_h):
            nonlocal parent_calls
            parent_calls += 1
            await registry.execute(child, lambda _h2: child.call_api("c"))
            return {"error": "429 from parent after child"}

        result = await with_provider_call_execution_context(
            {"rateLimitRegistry": registry},
            lambda: registry.execute(parent, parent_call, create_provider_rate_limit_options()),
        )
        assert parent_calls == 1
        assert result == {"error": "429 from parent after child"}
        assert get_provider_call_execution_context() is None

    fake.run(body)


def test_registry_abort_stops_waiting(fake):
    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 1})
        controller = AbortController()
        gate = asyncio.get_running_loop().create_future()

        async def hold(_h):
            return await gate

        holder = asyncio.ensure_future(registry.execute(_Provider(), hold))
        await flush()
        waiter = asyncio.ensure_future(
            registry.execute(
                _Provider(), lambda _h: value("never"), {"abortSignal": controller.signal}
            )
        )
        # fmt: skip
        await flush()
        reason = Exception("eval paused")
        controller.abort(reason)
        with pytest.raises(Exception) as exc:
            await waiter
        assert exc.value is reason
        gate.set_result("held")
        assert await holder == "held"
        assert registry.get_metrics()["test-provider"]["queueDepth"] == 0

    fake.run(body)


def test_registry_adaptive_concurrency_limits_parallel_calls(fake):
    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 4})
        active = peak_after_limit = 0
        limited_once = False

        async def call(_h):
            nonlocal active, peak_after_limit, limited_once
            active += 1
            try:
                await advance(10)
                if not limited_once:
                    limited_once = True
                    return {"error": "429", "metadata": {"http": {"status": 429}}}
                peak_after_limit = max(peak_after_limit, active)
                return {"output": "ok"}
            finally:
                active -= 1

        provider = _Provider()
        await asyncio.gather(*(registry.execute(provider, call,
                                                create_provider_rate_limit_options())
                               for _ in range(12)))  # fmt: skip
        metrics = registry.get_metrics()["test-provider"]
        assert metrics["rateLimitHits"] == 1
        assert metrics["maxConcurrency"] <= 4
        assert peak_after_limit <= 4

    fake.run(body)


# --- providerWrapper ----------------------------------------------------------------------


def test_wrap_provider(fake):
    async def body():
        registry = RateLimitRegistry({"maxConcurrency": 4})
        provider = _Provider("wrapped", {"temperature": 0})
        provider.label = "Wrapped"
        wrapped = wrap_provider_with_rate_limiting(provider, registry)
        assert is_rate_limit_wrapped(wrapped) and not is_rate_limit_wrapped(provider)
        assert wrap_provider_with_rate_limiting(wrapped, registry) is wrapped
        assert wrapped.id() == "wrapped"
        assert (wrapped.label, wrapped.config) == ("Wrapped", {"temperature": 0})
        assert await wrapped.call_api("hello") == {"output": "hello"}
        _prompt, _context, options = provider.calls[0]
        assert callable(options["onResponseHeaders"])
        assert registry.get_metrics()["wrapped"]["completedRequests"] == 1
        both = wrap_providers_with_rate_limiting([provider, _Provider("b")], registry)
        assert all(is_rate_limit_wrapped(p) for p in both)

    fake.run(body)


def test_rate_limit_option_callbacks():
    options = create_provider_rate_limit_options()
    assert "abortSignal" not in options
    assert options["isRetryableResult"]({"error": "t", "metadata": {
        "retryableErrorKind": "transient_availability"}})  # fmt: skip
    assert not options["isRetryableResult"]({"error": "t", "metadata": {"http": {"status": 503}}})

    finalize = options["finalizeResult"]
    assert finalize({"output": "x"}, []) == {"output": "x"}
    retry = {"error": "temporary failure", "cost": 0.007, "tokenUsage": {
        "total": 5, "prompt": 3, "completion": 2, "numRequests": 1,
        "completionDetails": {"reasoning": 1}}}  # fmt: skip
    terminal = {"output": "Recovered", "cost": 0.013, "tokenUsage": {
        "total": 10, "prompt": 7, "completion": 3, "numRequests": 1,
        "completionDetails": {"reasoning": 2}}}  # fmt: skip
    assert finalize(terminal, [retry]) == {
        "output": "Recovered",
        "tokenUsage": {"total": 15, "prompt": 10, "completion": 5, "numRequests": 2,
                       "completionDetails": {"reasoning": 3}},
        "cost": 0.02,
    }  # fmt: skip
    unsafe = {
        "error": "t",
        "cost": float("inf"),
        "tokenUsage": {
            "total": float("inf"),
            "prompt": -1,
            "completion": 2,
            "numRequests": float("nan"),
        },
    }
    # fmt: skip
    assert finalize({"output": "Recovered"}, [unsafe]) == {
        "output": "Recovered", "tokenUsage": {"completion": 2, "numRequests": 2}}  # fmt: skip
    assert finalize({"output": "Recovered", "cost": float("inf")},
                    [{"error": "t", "cost": 0.01}]) == {
        "output": "Recovered", "tokenUsage": {"numRequests": 2}, "cost": 0.01}  # fmt: skip

    is_limited = options["isRateLimited"]
    assert is_limited({"output": "t", "metadata": {"http": {"status": 429, "headers": {}}}}, None)
    assert is_limited({"error": "SDK throttle", "metadata": {"rateLimitKind": "rate_limit"}},
                      None)  # fmt: skip
    assert is_limited(None, Exception("Error 429: rate limit exceeded"))

    get_headers = options["getHeaders"]
    assert get_headers({"metadata": {"http": {"status": 200, "headers": {"retry-after": "60"}}}}) \
        == {"retry-after": "60"}  # fmt: skip
    headers = {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "3600s"}
    quota = {
        "error": "Quota exceeded: HTTP 429 Too Many Requests (code: credit_balance_exhausted)",
        "metadata": {"rateLimitKind": "quota", "http": {"status": 429, "headers": headers}},
    }
    # fmt: skip
    assert get_headers(quota) is None
    assert not is_limited(quota, None)
    throttle = {**quota, "metadata": {**quota["metadata"], "rateLimitKind": "rate_limit"}}
    assert get_headers(throttle) == headers

    retry_after = options["getRetryAfter"]
    assert retry_after({"metadata": {"http": {"headers": {"Retry-After": "30"}}}}, None) == 30000
    assert (
        retry_after(
            {"metadata": {"http": {"headers": {"retry-after-ms": "9" * 400, "retry-after": "30"}}}},
            None,
        )
        == 30000
    )
    # fmt: skip
    assert retry_after({"metadata": {"http": {"headers": {"retry-after-ms": "1500"}}}},
                       None) == 1500  # fmt: skip
    assert retry_after(None, Exception("Please retry after 7 seconds")) == 7000
    assert retry_after(None, Exception(f"retry after {'9' * 400}")) is None
    assert (
        retry_after(
            {"metadata": {"rateLimitRetryable": False, "headers": {"retry-after": "5"}}}, None
        )
        is None
    )
    # fmt: skip

    exhausted = options["onRateLimitExhausted"]
    error = RateLimitExhaustedError("Rate limit exceeded for k after 4 attempts", None)
    with_error = {"error": "429"}
    assert exhausted(with_error, error) is with_error
    assert exhausted({"output": "x"}, error) == {
        "output": "x", "error": "Rate limit exceeded for k after 4 attempts"}  # fmt: skip

    controller = AbortController()
    assert create_provider_rate_limit_options(controller.signal)["abortSignal"] is (
        controller.signal
    )
