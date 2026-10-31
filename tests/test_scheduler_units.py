"""Ports test/scheduler/{adaptiveConcurrency,retryPolicy,rateLimitKey,headerParser,types,
cancellation}.test.ts."""

from __future__ import annotations

import asyncio
import email.utils
import re

import pytest
from scheduler_fakes import START_MS, FakeTime, advance

from promptphoo.fetch_context import is_transient_connection_error
from promptphoo.providers.base import ApiProvider
from promptphoo.scheduler import clock
from promptphoo.scheduler.adaptive_concurrency import WARNING_THRESHOLD, AdaptiveConcurrency
from promptphoo.scheduler.cancellation import AbortController, sleep_with_abort
from promptphoo.scheduler.header_parser import parse_rate_limit_headers, parse_retry_after
from promptphoo.scheduler.rate_limit_key import get_rate_limit_key
from promptphoo.scheduler.retry_policy import DEFAULT_RETRY_POLICY, get_retry_delay, should_retry
from promptphoo.scheduler.types import (
    get_provider_response_headers,
    is_provider_response_rate_limited,
)


class HttpRateLimitError(Exception):
    """Stand-in for the fetch layer's HttpRateLimitError (matched by name)."""

    name = "HttpRateLimitError"

    def __init__(self, kind: str) -> None:
        super().__init__(f"HTTP 429 ({kind})")
        self.kind = kind


@pytest.fixture
def now(monkeypatch):
    monkeypatch.setattr(clock, "date_now", lambda: START_MS)
    return START_MS


# --- adaptiveConcurrency ------------------------------------------------------------------


class TestAdaptiveConcurrency:
    def test_constructor(self):
        ac = AdaptiveConcurrency(10)
        assert (ac.get_current(), ac.get_initial(), ac.get_min()) == (10, 10, 1)
        assert AdaptiveConcurrency(10, 3).get_min() == 3
        assert AdaptiveConcurrency(10, 0).get_min() == 1
        clamped = AdaptiveConcurrency(5, 10)
        assert clamped.get_min() == 5
        assert clamped.record_rate_limit()["current"] == 5

    def test_no_change_before_threshold(self):
        ac = AdaptiveConcurrency(10)
        ac.record_rate_limit()
        for _ in range(4):
            assert ac.record_success()["changed"] is False
        assert ac.get_current() == 5

    def test_no_increase_at_initial(self):
        ac = AdaptiveConcurrency(10)
        for _ in range(10):
            assert ac.record_success() == {
                "changed": False,
                "previous": 10,
                "current": 10,
                "reason": "recovery",
            }

    def test_recovery_path(self):
        ac = AdaptiveConcurrency(10)
        for _ in range(3):
            ac.record_rate_limit()
        assert ac.get_current() == 1
        path = []
        for _ in range(25):
            change = ac.record_success()
            if change["changed"]:
                path.append(change["current"])
        assert path == [2, 3, 5, 8, 10]
        for _ in range(10):
            ac.record_success()
        assert ac.get_current() == 10

    def test_rate_limit_halves_and_respects_min(self):
        assert AdaptiveConcurrency(10).record_rate_limit() == {
            "changed": True,
            "previous": 10,
            "current": 5,
            "reason": "ratelimit",
        }
        assert AdaptiveConcurrency(7).record_rate_limit()["current"] == 3
        ac = AdaptiveConcurrency(10, 3)
        ac.record_rate_limit()
        ac.record_rate_limit()
        assert ac.get_current() == 3
        result = ac.record_rate_limit()
        assert (result["changed"], result["current"]) == (False, 3)

    def test_rate_limit_resets_success_counter(self):
        ac = AdaptiveConcurrency(10)
        ac.record_rate_limit()
        for _ in range(4):
            ac.record_success()
        ac.record_rate_limit()
        assert ac.get_current() == 2
        for _ in range(4):
            assert ac.record_success()["changed"] is False
        assert ac.record_success()["current"] == 3

    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [(0.15, 100), (WARNING_THRESHOLD, 100), (0.099, 59), (0.05, 40), (0.01, 24),
         (0.0, 20), (0.075, 50), (-0.5, 20)],
    )  # fmt: skip
    def test_approaching_limit(self, ratio, expected):
        ac = AdaptiveConcurrency(100)
        result = ac.record_approaching_limit(ratio)
        assert result["current"] == expected
        assert result["reason"] == "proactive"
        assert result["changed"] is (expected != 100)

    def test_approaching_limit_respects_min(self):
        ac = AdaptiveConcurrency(10, 5)
        assert ac.record_approaching_limit(0.01)["current"] == 5
        assert ac.record_approaching_limit(0.01)["changed"] is False


# --- retryPolicy --------------------------------------------------------------------------


class TestRetryPolicy:
    policy = {**DEFAULT_RETRY_POLICY, "jitterFactor": 0}

    def test_get_retry_delay(self):
        assert get_retry_delay(1, self.policy, 0) == 0
        assert get_retry_delay(1, self.policy, 5000) == 5000
        assert get_retry_delay(1, self.policy, 120000) == self.policy["maxDelayMs"]
        assert [get_retry_delay(a, self.policy) for a in (1, 2, 3)] == [2000, 4000, 8000]
        assert get_retry_delay(10, self.policy) == self.policy["maxDelayMs"]

    def test_jitter(self, monkeypatch):
        monkeypatch.setattr(clock, "math_random", lambda: 0.5)
        assert get_retry_delay(0, DEFAULT_RETRY_POLICY) == 1100
        assert get_retry_delay(0, DEFAULT_RETRY_POLICY, 1000) == 1100

    @pytest.mark.parametrize(
        ("message", "rate_limited", "expected"),
        [
            (None, True, True),
            ("HTTP 429: Too Many Requests", True, True),
            ("HTTP 503: Service Unavailable", False, True),
            ("Some random error", False, False),
            ("", False, False),
            ("Request timeout after 30000ms", False, True),
            ("WebSocket request timed out after 30000ms", False, True),
            ("codex app-server turn timed out after 30000ms", False, False),
            ("ECONNRESET", False, True),
            ("bad record mac", False, True),
            ("write EPROTO 00000000:error:0A000126", False, True),
            ("self signed certificate", False, False),
        ],
    )
    def test_should_retry(self, message, rate_limited, expected):
        error = Exception(message) if message is not None else None
        assert should_retry(0, error, rate_limited, DEFAULT_RETRY_POLICY) is expected

    def test_should_not_retry_after_max(self):
        assert not should_retry(3, None, True, DEFAULT_RETRY_POLICY)

    def test_quota_errors_are_not_retried(self):
        assert not should_retry(0, HttpRateLimitError("quota"), True, DEFAULT_RETRY_POLICY)
        assert should_retry(0, HttpRateLimitError("rate_limit"), True, DEFAULT_RETRY_POLICY)


# --- rateLimitKey -------------------------------------------------------------------------


class _Provider(ApiProvider):
    def __init__(self, provider_id: str, config=None):
        super().__init__({"id": provider_id, "config": config})


class TestRateLimitKey:
    def test_basic(self):
        assert get_rate_limit_key(_Provider("openai:gpt-4")) == "openai:gpt-4"
        assert get_rate_limit_key(_Provider("openai:gpt-4", {"temperature": 0.7})) == (
            "openai:gpt-4"
        )
        assert get_rate_limit_key(_Provider("openai:gpt-4", {"apiKey": ""})) == "openai:gpt-4"

    def test_hashed_config(self):
        key = get_rate_limit_key(_Provider("openai:gpt-4", {"apiKey": "sk-test-key-12345"}))
        assert re.fullmatch(r"openai:gpt-4\[[0-9a-f]{12}\]", key)
        assert "12345" not in key
        # sha256("apiKeyTail:2345")[:12]
        import hashlib

        assert key == f"openai:gpt-4[{hashlib.sha256(b'apiKeyTail:2345').hexdigest()[:12]}]"

    def test_distinct_and_stable(self):
        def key(**config):
            return get_rate_limit_key(_Provider("openai:gpt-4", config))

        assert key(apiKey="sk-key-1") != key(apiKey="sk-key-2")
        assert key(apiKey="sk-key-1") == key(apiKey="sk-key-1")
        assert key(apiBaseUrl="https://a") != key(apiBaseUrl="https://b")
        assert key(region="us-east-1") != key(region="eu-west-1")
        assert key(organization="org-1") != key(organization="org-2")
        assert key(apiKey="sk-abcdef", region="r", organization="o") == key(
            organization="o", region="r", apiKey="sk-abcdef"
        )

    def test_special_characters(self):
        key = get_rate_limit_key(_Provider("custom:model/v1.2-beta", {"apiKey": "test-key"}))
        assert re.fullmatch(r"custom:model/v1\.2-beta\[.{12}\]", key)


# --- headerParser -------------------------------------------------------------------------


class TestParseRateLimitHeaders:
    def test_openai(self, now):
        assert parse_rate_limit_headers(
            {"x-ratelimit-remaining-requests": "100", "x-ratelimit-limit-requests": "500",
             "x-ratelimit-remaining-tokens": "50000", "x-ratelimit-limit-tokens": "90000"}
        ) == {"remainingRequests": 100, "remainingTokens": 50000, "limitRequests": 500,
              "limitTokens": 90000}  # fmt: skip
        assert parse_rate_limit_headers(
            {"x-ratelimit-remaining": "250", "x-ratelimit-limit": "1000"}
        ) == {"remainingRequests": 250, "limitRequests": 1000}

    def test_reset_formats(self, now):
        def reset(name, value):
            return parse_rate_limit_headers({name: value}).get("resetAt")

        future_s = int(now // 1000) + 60
        assert reset("x-ratelimit-reset", str(future_s)) == future_s * 1000
        assert reset("x-ratelimit-reset", str(int(now + 60000))) == now + 60000
        assert reset("x-ratelimit-reset-requests", "30s") == now + 30000
        assert reset("anthropic-ratelimit-requests-reset", "1m30s") == now + 90000
        assert reset("ratelimit-reset", "120") == now + 120000
        for value, ms in [("500ms", 500), ("45s", 45000), ("1h30m", 5400000),
                          ("2h15m30s", 8130000), ("5m", 300000), ("1h", 3600000)]:  # fmt: skip
            assert reset("x-ratelimit-reset", value) == now + ms

    def test_rfc3339_resets(self, now):
        assert (
            parse_rate_limit_headers(
                {"anthropic-ratelimit-requests-reset": "2026-08-07T21:00:00Z"}
            )["resetAt"]
            == now + 3_600_000
        )
        assert (
            parse_rate_limit_headers({"anthropic-ratelimit-tokens-reset": "2026-08-07T21:00:00Z"})[
                "resetAt"
            ]
            == now + 3_600_000
        )
        assert parse_rate_limit_headers({
            "anthropic-ratelimit-requests-reset": "2026-08-07T20:30:00Z",
            "anthropic-ratelimit-tokens-reset": "2026-08-07T21:00:00Z",
        })["resetAt"] == now + 1_800_000  # fmt: skip
        assert (
            parse_rate_limit_headers(
                {"anthropic-ratelimit-requests-reset": "2026-08-07T23:00:00+02:00"}
            )["resetAt"]
            == now + 3_600_000
        )
        assert "resetAt" not in parse_rate_limit_headers(
            {"anthropic-ratelimit-requests-reset": "2030-01-01T00:00:00Z"}
        )

    def test_case_insensitive_and_priority(self, now):
        assert parse_rate_limit_headers(
            {
                "X-RateLimit-Remaining-Requests": "100",
                "X-RateLimit-Limit-Requests": "500",
                "Anthropic-RateLimit-Tokens-Remaining": "50000",
            }
        ) == {"remainingRequests": 100, "remainingTokens": 50000, "limitRequests": 500}
        assert parse_rate_limit_headers({"RETRY-AFTER": "60"}) == {
            "retryAfterMs": 60000,
            "resetAt": now + 60000,
        }
        assert (
            parse_rate_limit_headers(
                {"x-ratelimit-remaining-requests": "100", "x-ratelimit-remaining": "200"}
            )["remainingRequests"]
            == 100
        )
        assert (
            parse_rate_limit_headers(
                {"x-ratelimit-reset-requests": "30s", "x-ratelimit-reset": "60s"}
            )["resetAt"]
            == now + 30000
        )
        # fmt: skip

    def test_retry_after(self, now):
        assert parse_rate_limit_headers({"retry-after-ms": "5000"}) == {
            "retryAfterMs": 5000, "resetAt": now + 5000}  # fmt: skip
        assert parse_rate_limit_headers({"retry-after": "30"}) == {
            "retryAfterMs": 30000, "resetAt": now + 30000}  # fmt: skip
        both = parse_rate_limit_headers({"x-ratelimit-reset": "2m", "retry-after": "30"})
        assert both == {"resetAt": now + 120000, "retryAfterMs": 30000}
        assert (
            parse_rate_limit_headers({"retry-after-ms": "5000", "retry-after": "30"})[
                "retryAfterMs"
            ]
            == 5000
        )
        assert parse_rate_limit_headers({"retry-after-ms": "Infinity", "retry-after": "30"}) == {
            "retryAfterMs": 30000, "resetAt": now + 30000}  # fmt: skip

    def test_edge_cases(self, now):
        assert parse_rate_limit_headers({}) == {}
        assert (
            parse_rate_limit_headers(
                {"x-ratelimit-remaining-requests": "invalid", "x-ratelimit-limit-tokens": "nan"}
            )
            == {}
        )
        overflow = "9" * 400
        assert parse_rate_limit_headers({"x-ratelimit-remaining-requests": overflow,
                                         "x-ratelimit-reset": "Infinity",
                                         "retry-after-ms": overflow}) == {}  # fmt: skip
        assert parse_rate_limit_headers({"x-ratelimit-reset": overflow + "h"}) == {}
        assert parse_rate_limit_headers({"x-ratelimit-reset": "-1"}) == {}
        assert parse_rate_limit_headers({"x-ratelimit-reset": "2026"}) == {
            "resetAt": now + 2026 * 1000}  # fmt: skip
        assert parse_rate_limit_headers({"x-ratelimit-reset": "1.5"}) == {"resetAt": now + 1500}
        assert (
            parse_rate_limit_headers(
                {"x-ratelimit-remaining-requests": "-5", "x-ratelimit-limit-requests": "-100"}
            )
            == {}
        )
        assert parse_rate_limit_headers(
            {"x-ratelimit-remaining-requests": "0", "x-ratelimit-limit-requests": "0"}
        ) == {"remainingRequests": 0, "limitRequests": 0}
        assert parse_rate_limit_headers({"retry-after-ms": "invalid"}) == {}
        assert parse_rate_limit_headers({"retry-after-ms": "0"})["retryAfterMs"] == 0
        assert parse_rate_limit_headers({"retry-after-ms": "-5"}) == {}
        assert parse_rate_limit_headers({"x-ratelimit-remaining-requests": ""}) == {}
        spaced = parse_rate_limit_headers(
            {"x-ratelimit-remaining-requests": "  100  ", "retry-after": " 30 "}
        )
        assert (spaced["remainingRequests"], spaced["retryAfterMs"]) == (100, 30000)


class TestParseRetryAfter:
    def test_integer_seconds(self):
        assert parse_retry_after("60") == 60000
        assert parse_retry_after("0") == 0
        assert parse_retry_after("  30  ") == 30000
        assert parse_retry_after("30.5") is None
        assert parse_retry_after("30 seconds") is None

    def test_http_dates(self, now):
        def http_date(offset_ms):
            return email.utils.formatdate((now + offset_ms) / 1000, usegmt=True)

        assert 0 < parse_retry_after(http_date(60000)) <= 60000
        iso = "2026-08-07T20:02:00.000Z"
        assert parse_retry_after(iso) == 120000
        assert parse_retry_after(http_date(-60000)) == 0
        assert parse_retry_after(http_date(400 * 24 * 60 * 60 * 1000)) is None
        assert parse_retry_after(http_date(-400 * 24 * 60 * 60 * 1000)) is None

    @pytest.mark.parametrize("value", ["invalid", "Infinity", "9" * 400, "", "-30", "30s"])
    def test_invalid(self, value, now):
        assert parse_retry_after(value) is None


# --- types --------------------------------------------------------------------------------


class TestIsProviderResponseRateLimited:
    def test_http_status(self):
        assert is_provider_response_rate_limited({"metadata": {"http": {"status": 429}}}, None)
        assert not is_provider_response_rate_limited({"metadata": {"http": {"status": 500}}}, None)

    def test_error_field(self):
        assert is_provider_response_rate_limited({"error": "HTTP 429 Too Many"}, None)
        assert is_provider_response_rate_limited({"error": "Rate Limit exceeded"}, None)
        assert not is_provider_response_rate_limited({"error": None}, None)
        assert not is_provider_response_rate_limited(
            {"error": "prompt is too long: 204291 tokens"}, None
        )

    def test_tool_diagnostics(self):
        tool = {"error": "tool hit rate limit", "metadata": {"errorOrigin": "tool"}}
        assert not is_provider_response_rate_limited(tool, None)
        assert is_provider_response_rate_limited(
            {**tool, "metadata": {"errorOrigin": "tool", "http": {"status": 429}}}, None
        )
        assert is_provider_response_rate_limited(tool, Exception("Rate limit reached"))

    def test_thrown_errors(self):
        assert is_provider_response_rate_limited(None, Exception("Error 429"))
        assert is_provider_response_rate_limited(None, Exception("RATE LIMIT"))
        assert is_provider_response_rate_limited(None, Exception("Too Many Requests"))
        assert not is_provider_response_rate_limited(None, Exception(""))
        assert not is_provider_response_rate_limited(None, None)

    def test_structured_signals(self):
        assert not is_provider_response_rate_limited(
            {"error": "429", "metadata": {"rateLimitKind": "quota"}}, None
        )
        assert not is_provider_response_rate_limited(
            {"error": "429", "metadata": {"rateLimitRetryable": False}}, None
        )
        assert is_provider_response_rate_limited({"metadata": {"rateLimitKind": "rate_limit"}},
                                                 None)  # fmt: skip
        assert not is_provider_response_rate_limited(None, HttpRateLimitError("quota"))
        assert not is_provider_response_rate_limited(
            {"error": "API call error: HttpRateLimitError: Quota exceeded: 429"}, None
        )
        assert not is_provider_response_rate_limited(None, Exception("Quota exceeded: 429"))

    def test_edge_cases(self):
        assert not is_provider_response_rate_limited({"error": "status 4290"}, None)
        assert not is_provider_response_rate_limited({}, None)
        assert not is_provider_response_rate_limited({"metadata": {"http": None}}, None)


def test_get_provider_response_headers():
    http = {"x-ratelimit-remaining-requests": "10"}
    assert get_provider_response_headers({"metadata": {"http": {"headers": http}}}) is http
    flat = {"retry-after": "5"}
    assert get_provider_response_headers({"metadata": {"headers": flat}}) is flat
    assert (
        get_provider_response_headers({"metadata": {"http": {"headers": http}, "headers": flat}})
        is http
    )
    assert get_provider_response_headers({"output": "x"}) is None
    assert get_provider_response_headers(None) is None
    assert get_provider_response_headers({"metadata": {}}) is None
    # JS truthiness: an empty headers object is still returned
    assert get_provider_response_headers({"metadata": {"headers": {}}}) == {}


class _CodeError(Exception):
    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (Exception("bad record mac"), True),
        (Exception("write EPROTO"), True),
        (Exception("read ECONNRESET"), True),
        (Exception("socket hang up"), True),
        (None, False),
        (Exception("something else"), False),
        (Exception("429 rate limit"), False),
        (Exception("EPROTO wrong version number"), False),
        (Exception("EPROTO self signed certificate"), False),
        (Exception("EPROTO unable to verify"), False),
        (Exception("EPROTO unknown ca"), False),
        (Exception("EPROTO cert has expired"), False),
        (_CodeError("x", "ECONNRESET"), True),
        (_CodeError("x", "EPIPE"), True),
        (_CodeError("x", "ECONNREFUSED"), False),
        (ConnectionResetError("reset"), True),
    ],
)
def test_is_transient_connection_error(error, expected):
    assert is_transient_connection_error(error) is expected


# --- cancellation -------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", ["aborted", "elapsed"])
def test_sleep_with_abort_cleanup(monkeypatch, outcome):
    fake = FakeTime(monkeypatch)

    async def body():
        controller = AbortController()
        reason = Exception("caller stopped")
        task = asyncio.ensure_future(sleep_with_abort(60_000, controller.signal))
        await asyncio.sleep(0)
        assert len(controller.signal._listeners) == 1
        loop = asyncio.get_running_loop()
        assert len(loop._scheduled) == 1  # type: ignore[attr-defined]
        if outcome == "aborted":
            controller.abort(reason)
            with pytest.raises(Exception) as exc:
                await task
            assert exc.value is reason
        else:
            await advance(60_000)
            await task
        assert controller.signal._listeners == []
        assert all(h.cancelled() for h in loop._scheduled)  # type: ignore[attr-defined]

    fake.run(body)


def test_sleep_with_abort_rejects_when_already_aborted():
    controller = AbortController()
    controller.abort()

    async def body():
        await sleep_with_abort(10, controller.signal)

    with pytest.raises(Exception, match="aborted"):
        asyncio.run(body())


def test_sleep_with_abort_respects_task_cancellation(monkeypatch):
    fake = FakeTime(monkeypatch)

    async def body():
        controller = AbortController()
        task = asyncio.ensure_future(sleep_with_abort(60_000, controller.signal))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert controller.signal._listeners == []

    fake.run(body)


def test_scheduler_retry_context_reaches_fetch():
    from promptphoo import fetch, fetch_context

    with fetch_context.fetch_retry_context(0):
        assert fetch.get_fetch_retry_context_max_retries() == 0
    assert fetch.get_fetch_retry_context_max_retries() is None
