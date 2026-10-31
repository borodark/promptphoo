"""Tests for promptphoo.cache (fetchWithCache and helpers). Ports cases from upstream
test/cache.test.ts."""

from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest
import respx

from promptphoo import cache, fetch
from promptphoo.cache import (
    claim_cache_key_once,
    clear_cache,
    fetch_with_cache,
    get_cache_clear_generation,
    get_headers_for_cache_key,
    with_cache_enabled,
    with_cache_namespace,
)
from promptphoo.fetch import FetchResponse, JsError

URL = "https://api.example.com/data"
RESPONSE = {"data": "test data"}
HEADERS = {"content-type": "application/json", "x-session-id": "45"}


@pytest.fixture(autouse=True)
async def _memory_cache(monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CACHE_TYPE", "memory")
    monkeypatch.setattr(cache, "_instance", None)
    cache.enable_cache()
    await clear_cache()
    yield
    cache.enable_cache()
    cache._inflight_fetch_responses.clear()


def ok(body, status=200, status_text="OK", headers=None) -> FetchResponse:
    text = body if isinstance(body, str) else json.dumps(body)
    return FetchResponse(
        status, status_text, dict(HEADERS if headers is None else headers), text.encode()
    )


class FakeRetries:
    """Stand-in for fetch_with_retries, like upstream's mockFetchWithRetries."""

    def __init__(self, *results, delay: float = 0.0):
        self.results = list(results)
        self.calls: list[tuple] = []
        self.delay = delay

    async def __call__(self, url, options=None, timeout=None, max_retries=None, **kwargs):
        self.calls.append((url, options, timeout, max_retries))
        if self.delay:
            await asyncio.sleep(self.delay)
        item = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def mock_fetch(monkeypatch):
    def install(*results, delay: float = 0.0) -> FakeRetries:
        fake = FakeRetries(*results, delay=delay)
        monkeypatch.setattr(fetch, "fetch_with_retries", fake)
        return fake

    return install


async def test_fetch_and_cache_successful_requests(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    result = await fetch_with_cache(URL, {}, 1000)
    assert len(fake.calls) == 1
    assert (result.cached, result.data, result.status, result.status_text) == (
        False,
        RESPONSE,
        200,
        "OK",
    )
    assert result.headers == HEADERS
    assert callable(result.delete_from_cache)
    assert isinstance(result.latency_ms, int)

    cached = await fetch_with_cache(URL, {}, 1000)
    assert len(fake.calls) == 1
    assert (cached.cached, cached.data, cached.status, cached.headers) == (
        True,
        RESPONSE,
        200,
        HEADERS,
    )
    assert cached.latency_ms == result.latency_ms
    assert callable(cached.delete_from_cache)


async def test_forwards_timeout_and_max_retries(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    await fetch_with_cache(URL, {"method": "POST", "body": "{}"}, 1234, "json", False, 2)
    assert fake.calls[0][2:] == (1234, 2)


async def test_update_cache_replaces_entry(mock_fetch):
    fake = mock_fetch(ok({"status": "queued", "output": []}))
    result = await fetch_with_cache(URL, {}, 1000)
    assert result.update_cache is not None
    await result.update_cache(
        {"status": "completed", "output": ["done"]}, 200, "OK", {"x-completed": "true"}
    )
    cached = await fetch_with_cache(URL, {}, 1000)
    assert len(fake.calls) == 1
    assert cached.cached is True
    assert cached.data == {"status": "completed", "output": ["done"]}
    assert cached.headers == {"x-completed": "true"}


async def test_delete_from_cache_evicts(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    result = await fetch_with_cache(URL, {}, 1000)
    assert result.delete_from_cache is not None
    await result.delete_from_cache()
    await fetch_with_cache(URL, {}, 1000)
    assert len(fake.calls) == 2


async def test_concurrent_cache_miss_is_coalesced(mock_fetch):
    fake = mock_fetch(ok(RESPONSE), delay=0.01)
    result1, result2 = await asyncio.gather(
        fetch_with_cache(URL, {}, 1000), fetch_with_cache(URL, {}, 1000)
    )
    assert (result1.cached, result1.coalesced, result1.data) == (False, None, RESPONSE)
    assert (result2.cached, result2.coalesced, result2.data) == (False, True, RESPONSE)
    assert len(fake.calls) == 1
    cached = await fetch_with_cache(URL, {}, 1000)
    assert cached.cached is True
    assert len(fake.calls) == 1


async def test_different_timeouts_do_not_share_transport(mock_fetch):
    fake = mock_fetch(ok(RESPONSE), delay=0.01)
    await asyncio.gather(fetch_with_cache(URL, {}, 1000), fetch_with_cache(URL, {}, 2000))
    assert len(fake.calls) == 2


async def test_inflight_dedupe_isolated_by_namespace(mock_fetch):
    fake = mock_fetch(ok({"data": "repeat 0"}), ok({"data": "repeat 1"}), delay=0.01)

    async def in_ns(ns):
        with with_cache_namespace(ns):
            return await fetch_with_cache(URL, {}, 1000)

    r0, r1 = await asyncio.gather(in_ns("repeat:0"), in_ns("repeat:1"))
    assert len(fake.calls) == 2
    assert r0.data == {"data": "repeat 0"}
    assert r1.data == {"data": "repeat 1"}
    c0, c1 = await in_ns("repeat:0"), await in_ns("repeat:1")
    assert len(fake.calls) == 2
    assert (c0.cached, c0.data, c1.data) == (True, {"data": "repeat 0"}, {"data": "repeat 1"})


async def test_failed_requests_are_not_cached(mock_fetch):
    fake = mock_fetch(ok({"error": "Bad Request"}, 400, "Bad Request"))
    for _ in range(2):
        result = await fetch_with_cache(URL, {}, 1000)
        assert (result.status, result.status_text, result.data) == (
            400,
            "Bad Request",
            {"error": "Bad Request"},
        )
        assert result.cached is False
    assert len(fake.calls) == 2


async def test_failed_responses_reach_all_concurrent_callers_uncached(mock_fetch):
    fake = mock_fetch(ok({"error": "Bad Request"}, 400, "Bad Request"), delay=0.01)
    r1, r2 = await asyncio.gather(fetch_with_cache(URL, {}, 1000), fetch_with_cache(URL, {}, 1000))
    assert (r1.status, r1.coalesced, r2.status, r2.coalesced) == (400, None, 400, True)
    assert len(fake.calls) == 1
    await fetch_with_cache(URL, {}, 1000)
    assert len(fake.calls) == 2


async def test_successful_error_payloads_are_not_cached(mock_fetch):
    fake = mock_fetch(ok({"error": "Rate limit exceeded"}))
    for _ in range(2):
        result = await fetch_with_cache(URL, {}, 1000)
        assert (result.status, result.cached, result.data) == (
            200,
            False,
            {"error": "Rate limit exceeded"},
        )
    assert len(fake.calls) == 2


async def test_falsy_error_key_is_still_cacheable(mock_fetch):
    fake = mock_fetch(ok({"error": None, "result": 1}))
    await fetch_with_cache(URL, {}, 1000)
    assert (await fetch_with_cache(URL, {}, 1000)).cached is True
    assert len(fake.calls) == 1


async def test_non_json_response_raises_parse_error(mock_fetch):
    mock_fetch(ok("not json"))
    with pytest.raises(ValueError, match="Error parsing response from") as info:
        await fetch_with_cache(URL, {}, 1000, "json")
    assert "HTTP 200 OK. Received text: not json" in str(info.value)


async def test_non_json_error_response_also_raises(mock_fetch):
    mock_fetch(ok("<html>bad gateway</html>", 502, "Bad Gateway"))
    with pytest.raises(ValueError, match="HTTP 502 Bad Gateway"):
        await fetch_with_cache(URL, {}, 1000)


async def test_parse_errors_sanitize_credentials(mock_fetch):
    mock_fetch(ok("not json"))
    with pytest.raises(ValueError) as info:
        await fetch_with_cache(
            "https://user:secret-pass@api.example.com/x?api_key=sk-abc123", {}, 1000
        )
    message = str(info.value)
    assert "secret-pass" not in message
    assert "sk-abc123" not in message


async def test_empty_text_error_response(mock_fetch):
    mock_fetch(ok("", 400, "Bad Request"))
    result = await fetch_with_cache(URL, {}, 1000, "text")
    assert result.data == "Empty Response: 400: Bad Request"


async def test_text_format(mock_fetch):
    mock_fetch(ok("plain"))
    result = await fetch_with_cache(URL, {}, 1000, "text")
    assert result.data == "plain"


async def test_network_errors_propagate_and_allow_retry(mock_fetch):
    fake = mock_fetch(JsError("Network error"), delay=0.01)
    results = await asyncio.gather(
        fetch_with_cache(URL, {}, 1000), fetch_with_cache(URL, {}, 1000), return_exceptions=True
    )
    assert all(isinstance(r, JsError) for r in results)
    assert len(fake.calls) == 1
    fake.results = [ok(RESPONSE)]
    result = await fetch_with_cache(URL, {}, 1000)
    assert result.data == RESPONSE
    assert len(fake.calls) == 2


async def test_cache_key_includes_request_options(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    await fetch_with_cache(URL, {"method": "POST", "body": '{"a":1}'}, 1000)
    await fetch_with_cache(URL, {"method": "POST", "body": '{"a":2}'}, 1000)
    await fetch_with_cache(URL, {"method": "POST", "body": '{"a":1}'}, 1000)
    assert len(fake.calls) == 2


async def test_trace_headers_do_not_affect_cache_key(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    await fetch_with_cache(URL, {"headers": {"traceparent": "00-a-b-01"}}, 1000)
    result = await fetch_with_cache(URL, {"headers": {"traceparent": "00-c-d-01"}}, 1000)
    assert result.cached is True
    assert fake.calls[0][1]["headers"] == {"traceparent": "00-a-b-01"}


async def test_cache_isolated_by_headers_without_storing_secrets(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    await fetch_with_cache(URL, {"headers": {"Authorization": "Bearer sk-secret-one-123456"}}, 1000)
    await fetch_with_cache(URL, {"headers": {"Authorization": "Bearer sk-secret-two-123456"}}, 1000)
    assert len(fake.calls) == 2
    stored = json.dumps(cache._get_instance()._data)
    assert "sk-secret" not in stored

    identity = get_headers_for_cache_key(URL, {"headers": {"Authorization": "Bearer abc"}})
    names = [name for name, _ in identity]
    assert names == sorted(names)
    auth = dict((n, v) for n, v in identity)["authorization"]
    assert set(auth) == {"__promptfooSecretFingerprint"}


async def test_secret_body_and_query_values_are_fingerprinted():
    key1 = cache._get_fetch_cache_key(
        URL + "?api_key=sk-one", {"method": "POST", "body": '{"password":"p1"}'}, "POST", "json"
    )
    key2 = cache._get_fetch_cache_key(
        URL + "?api_key=sk-one", {"method": "POST", "body": '{"password":"p1"}'}, "POST", "json"
    )
    key3 = cache._get_fetch_cache_key(
        URL + "?api_key=sk-two", {"method": "POST", "body": '{"password":"p1"}'}, "POST", "json"
    )
    assert key1 == key2 != key3
    assert key1 is not None and key1.startswith("fetch:v3:")
    identity = cache._get_url_for_fetch_cache_key(URL + "?api_key=sk-one&page=2")
    assert identity["href"] == URL
    assert identity["searchParams"][1] == ["page", "2"]
    body = cache._get_body_string_for_fetch_cache_key('{"password":"p1","n":1}')
    assert body["encoding"] == "json"
    assert set(body["value"]["password"]) == {"__promptfooSecretFingerprint"}


async def test_method_casing_is_normalized(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    await fetch_with_cache(URL, {"method": "post", "body": "x"}, 1000)
    result = await fetch_with_cache(URL, {"method": "POST", "body": "x"}, 1000)
    assert result.cached is True
    assert len(fake.calls) == 1


async def test_option_order_is_canonical(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    await fetch_with_cache(URL, {"compress": False, "redirect": "follow"}, 1000)
    result = await fetch_with_cache(URL, {"redirect": "follow", "compress": False}, 1000)
    assert result.cached is True
    assert len(fake.calls) == 1


async def test_response_format_isolates_cache(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    await fetch_with_cache(URL, {}, 1000, "json")
    text = await fetch_with_cache(URL, {}, 1000, "text")
    assert text.cached is False
    assert text.data == json.dumps(RESPONSE)
    assert len(fake.calls) == 2


async def test_opaque_transport_options_are_not_cached(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    async with httpx.AsyncClient() as client:
        for _ in range(2):
            result = await fetch_with_cache(URL, {"dispatcher": client}, 1000)
            assert result.cached is False
    assert len(fake.calls) == 2


async def test_opaque_bodies_are_not_cached(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    for _ in range(2):
        await fetch_with_cache(URL, {"method": "POST", "body": object()}, 1000)
    assert len(fake.calls) == 2


async def test_bytes_bodies_are_cached_by_hash(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    await fetch_with_cache(URL, {"method": "POST", "body": b"\x00\x01"}, 1000)
    assert (await fetch_with_cache(URL, {"method": "POST", "body": b"\x00\x01"}, 1000)).cached
    assert not (await fetch_with_cache(URL, {"method": "POST", "body": b"\x00\x02"}, 1000)).cached
    assert len(fake.calls) == 2


async def test_bust_refetches_without_evicting(mock_fetch):
    fake = mock_fetch(ok({"v": 1}), ok({"v": 2}))
    await fetch_with_cache(URL, {}, 1000)
    busted = await fetch_with_cache(URL, {}, 1000, "json", True)
    assert (busted.cached, busted.data) == (False, {"v": 2})
    assert busted.update_cache is None
    cached = await fetch_with_cache(URL, {}, 1000)
    assert (cached.cached, cached.data) == (True, {"v": 1})
    assert len(fake.calls) == 2


async def test_object_cache_options_bust(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    await fetch_with_cache(URL, {}, 1000, "json", {"bust": False})
    assert (await fetch_with_cache(URL, {}, 1000, "json", {"bust": False})).cached
    assert not (await fetch_with_cache(URL, {}, 1000, "json", {"bust": True})).cached
    assert len(fake.calls) == 2


async def test_cache_disabled_always_fetches(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    cache.disable_cache()
    for _ in range(2):
        result = await fetch_with_cache(URL, {}, 1000)
        assert result.cached is False
        assert result.delete_from_cache is not None
        await result.delete_from_cache()
    assert len(fake.calls) == 2


async def test_with_cache_enabled_scopes_override(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    with with_cache_enabled(False):
        await fetch_with_cache(URL, {}, 1000)
        await fetch_with_cache(URL, {}, 1000)
    assert len(fake.calls) == 2
    cache.disable_cache()
    with with_cache_enabled(True):
        await fetch_with_cache(URL, {}, 1000)
        assert (await fetch_with_cache(URL, {}, 1000)).cached
    assert len(fake.calls) == 3


async def test_request_time_auth_requires_cache_policy(mock_fetch):
    mock_fetch(ok(RESPONSE))
    with pytest.raises(ValueError, match="Request-time authentication requires cache bypass"):
        await fetch_with_cache(URL, {"getAuthHeaders": lambda: {}}, 1000)
    assert (
        await fetch_with_cache(URL, {"getAuthHeaders": lambda: {}}, 1000, "json", True)
    ).status == 200


async def test_request_time_auth_with_principal_keys(mock_fetch):
    fake = mock_fetch(ok({"p": "a"}), ok({"p": "b"}))
    opts = {"getAuthHeaders": lambda: {}}
    a = await fetch_with_cache(URL, opts, 1000, "json", {"cacheKey": "principal-a"})
    b = await fetch_with_cache(URL, opts, 1000, "json", {"cacheKey": "principal-b"})
    a2 = await fetch_with_cache(URL, opts, 1000, "json", {"cacheKey": "principal-a"})
    assert (a.data, b.data, a2.data, a2.cached) == ({"p": "a"}, {"p": "b"}, {"p": "a"}, True)
    assert len(fake.calls) == 2


async def test_sanitize_response_requires_key_and_sanitizes_before_storage(mock_fetch):
    mock_fetch(ok({"token": "raw-secret", "ok": True}, headers={"x-secret": "s", "a": "b"}))

    def sanitize(resp):
        data = {k: v for k, v in resp["data"].items() if k != "token"}
        headers = {k: v for k, v in resp["headers"].items() if k != "x-secret"}
        return {"data": data, "statusText": resp["statusText"], "headers": headers}

    with pytest.raises(ValueError, match="Response sanitization requires cache bypass"):
        await fetch_with_cache(URL, {}, 1000, "json", {"sanitizeResponse": sanitize})

    options = {"sanitizeResponse": sanitize, "cacheKey": "sanitized:v1"}
    result = await fetch_with_cache(URL, {}, 1000, "json", options)
    assert result.data == {"ok": True}
    assert result.headers == {"a": "b"}
    assert "raw-secret" not in json.dumps(cache._get_instance()._data)
    cached = await fetch_with_cache(URL, {}, 1000, "json", options)
    assert (cached.cached, cached.data) == (True, {"ok": True})


async def test_sanitized_parse_errors_omit_body(mock_fetch):
    mock_fetch(ok("secret body text"))
    with pytest.raises(ValueError) as info:
        await fetch_with_cache(
            URL, {}, 1000, "json", {"bust": True, "sanitizeResponse": lambda r: r}
        )
    assert str(info.value) == f"Error parsing response from {URL}: Invalid JSON. HTTP 200."


async def test_explicit_cache_key_reused_across_credential_rotations(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    await fetch_with_cache(
        URL, {"headers": {"Authorization": "Bearer one"}}, 1000, "json", {"cacheKey": "safe"}
    )
    result = await fetch_with_cache(
        URL, {"headers": {"Authorization": "Bearer two"}}, 1000, "json", {"cacheKey": "safe"}
    )
    assert result.cached is True
    assert len(fake.calls) == 1
    assert "fetch:v3:safe" in cache._get_instance().stored_keys()


async def test_explicit_cache_key_scoped_to_namespace(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    with with_cache_namespace("ns1"):
        await fetch_with_cache(URL, {}, 1000, "json", {"cacheKey": "k"})
    with with_cache_namespace("ns2"):
        assert not (await fetch_with_cache(URL, {}, 1000, "json", {"cacheKey": "k"})).cached
    assert len(fake.calls) == 2
    assert set(cache._get_instance().stored_keys()) == {"ns1:fetch:v3:k", "ns2:fetch:v3:k"}


async def test_repeat_index_zero_matches_no_repeat_index(mock_fetch):
    fake = mock_fetch(ok(RESPONSE))
    await fetch_with_cache(URL, {}, 1000, "json", {"repeatIndex": 0})
    assert (await fetch_with_cache(URL, {}, 1000, "json", False)).cached
    assert len(fake.calls) == 1


async def test_repeat_indices_get_separate_entries(mock_fetch):
    mock_fetch(ok({"r": 1}), ok({"r": 2}))
    r1 = await fetch_with_cache(URL, {}, 1000, "json", {"repeatIndex": 1})
    r2 = await fetch_with_cache(URL, {}, 1000, "json", {"repeatIndex": 2})
    assert (r1.data, r2.data) == ({"r": 1}, {"r": 2})
    again = await fetch_with_cache(URL, {}, 1000, "json", {"repeatIndex": 1})
    assert (again.cached, again.data) == (True, {"r": 1})
    keys = cache._get_instance().stored_keys()
    assert sorted(k.rsplit(":", 1)[-1] for k in keys) == ["repeat1", "repeat2"]


async def test_explicit_cache_key_gets_repeat_suffix(mock_fetch):
    mock_fetch(ok(RESPONSE))
    await fetch_with_cache(URL, {}, 1000, "json", {"cacheKey": "k", "repeatIndex": 3})
    assert cache._get_instance().stored_keys() == ["fetch:v3:k:repeat3"]


async def test_no_repeat_suffix_when_namespace_isolates_repeat(mock_fetch):
    mock_fetch(ok(RESPONSE))
    with with_cache_namespace("repeat:2"):
        await fetch_with_cache(URL, {}, 1000, "json", {"cacheKey": "k", "repeatIndex": 2})
    assert cache._get_instance().stored_keys() == ["repeat:2:fetch:v3:k"]


async def test_on_response_prepared_called_for_fresh_responses(mock_fetch):
    mock_fetch(ok(RESPONSE))
    seen = []
    await fetch_with_cache(URL, {}, 1000, on_response_prepared=seen.append)
    await fetch_with_cache(URL, {}, 1000, on_response_prepared=seen.append)
    assert [r.cached for r in seen] == [False]


async def test_rate_limit_observer_failure_fails_only_that_caller(monkeypatch):
    async def fake(
        url, options=None, timeout=None, max_retries=None, *, on_rate_limit_backoff=None
    ):
        assert on_rate_limit_backoff is not None
        on_rate_limit_backoff({"headers": {}, "status": 429, "resetAt": 0})
        await asyncio.sleep(0.01)
        return ok(RESPONSE)

    monkeypatch.setattr(fetch, "fetch_with_retries", fake)

    def boom(observation):
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        await fetch_with_cache(URL, {}, 1000, on_rate_limit_backoff=boom)
    await asyncio.sleep(0.02)
    # The shared transport still completed and published its response.
    assert (await fetch_with_cache(URL, {}, 1000)).cached


@respx.mock
async def test_end_to_end_with_http_transport(monkeypatch):
    async def no_sleep(ms):
        return None

    monkeypatch.setattr(fetch, "sleep", no_sleep)
    route = respx.post("https://api.example.com/v1").mock(
        side_effect=[
            httpx.Response(
                429, json={"error": {"code": "rate_limit_exceeded"}}, headers={"retry-after": "0"}
            ),
            httpx.Response(200, json={"answer": 42}, headers={"x-request-id": "r1"}),
        ]
    )
    options = {"method": "POST", "headers": {"content-type": "application/json"}, "body": "{}"}
    result = await fetch_with_cache("https://api.example.com/v1", options, 5000)
    assert (result.status, result.data, result.headers["x-request-id"]) == (
        200,
        {"answer": 42},
        "r1",
    )
    again = await fetch_with_cache("https://api.example.com/v1", options, 5000)
    assert again.cached is True
    assert route.call_count == 2


# ---- cache utilities ---------------------------------------------------------------------------


async def test_claim_cache_key_once_per_namespace():
    assert claim_cache_key_once("billing:1") is True
    assert claim_cache_key_once("billing:1") is False
    with with_cache_namespace("other"):
        assert claim_cache_key_once("billing:1") is True
    await clear_cache()
    assert claim_cache_key_once("billing:1") is True


async def test_disk_claims_persist_and_clear(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTFOO_CACHE_TYPE", "disk")
    monkeypatch.setenv("PROMPTFOO_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(cache, "_instance", None)
    assert claim_cache_key_once("k") is True
    assert len(os.listdir(tmp_path / "claims")) == 1
    # Another process (simulated by forgetting the in-memory claims) sees the file.
    cache._claimed_cache_keys.clear()
    assert claim_cache_key_once("k") is False
    await clear_cache()
    assert not (tmp_path / "claims").exists()
    assert claim_cache_key_once("k") is True


async def test_disk_cache_persists_fetch_responses(monkeypatch, tmp_path, mock_fetch):
    monkeypatch.setenv("PROMPTFOO_CACHE_TYPE", "disk")
    monkeypatch.setenv("PROMPTFOO_CACHE_PATH", str(tmp_path))
    monkeypatch.setattr(cache, "_instance", None)
    fake = mock_fetch(ok(RESPONSE))
    await fetch_with_cache(URL, {}, 1000)
    assert (tmp_path / cache.CACHE_FILE_NAME).exists()
    monkeypatch.setattr(cache, "_instance", None)
    assert (await fetch_with_cache(URL, {}, 1000)).cached
    assert len(fake.calls) == 1


async def test_cache_clear_generation():
    start = get_cache_clear_generation()
    await clear_cache()
    assert get_cache_clear_generation() == start + 1
    with with_cache_namespace("ns"):
        await cache.get_cache().clear()
    assert get_cache_clear_generation() == start + 2


async def test_namespaced_clear_only_removes_namespace_keys():
    await cache._get_instance().set("plain", "1")
    with with_cache_namespace("ns"):
        await cache.get_cache().set("a", "2")
        await cache.get_cache().clear()
        assert await cache.get_cache().get("a") is None
    assert await cache._get_instance().get("plain") == "1"


def test_cache_enabled_state():
    cache.enable_cache()
    assert cache.is_cache_enabled()
    cache.disable_cache()
    assert not cache.is_cache_enabled()
    cache.enable_cache()


def test_abort_signal_scoped_key():
    class Signal:
        pass

    s1, s2 = Signal(), Signal()
    assert cache.get_abort_signal_scoped_key("k") == '["k",null]'
    first = cache.get_abort_signal_scoped_key("k", s1)
    assert first == cache.get_abort_signal_scoped_key("k", s1)
    assert first != cache.get_abort_signal_scoped_key("k", s2)
