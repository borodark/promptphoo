"""Tests for promptphoo.fetch. Ports cases from upstream test/fetch.test.ts,
test/rateLimit.test.ts, test/fetch.compression.test.ts and test/util/fetch/errors.test.ts."""

from __future__ import annotations

import asyncio
import base64
import gzip
import logging
import math

import httpx
import pytest
import respx

from promptphoo import cli_state, fetch
from promptphoo.fetch import (
    FetchResponse,
    HttpRateLimitError,
    JsError,
    RequestAuthenticationError,
    compute_rate_limit_wait_ms,
    fetch_with_proxy,
    fetch_with_retries,
    fetch_with_timeout,
    get_proxy_for_url,
    handle_rate_limit,
    header_parser,
    is_rate_limited,
    is_transient_error,
    rate_limit_timing_from_headers,
    read_bounded_text,
    with_fetch_retry_context,
)
from promptphoo.fetch import errors as fetch_errors

NOW_MS = 1_704_067_200_000  # 2024-01-01T00:00:00Z

_PROXY_VARS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "NPM_CONFIG_HTTP_PROXY",
    "NPM_CONFIG_HTTPS_PROXY",
    "NPM_CONFIG_PROXY",
    "NPM_CONFIG_NO_PROXY",
]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    for name in _PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    for name in (
        "PROMPTFOO_RETRY_5XX",
        "PROMPTFOO_REQUEST_BACKOFF_MS",
        "PROMPTFOO_INSECURE_SSL",
        "PROMPTFOO_CA_CERT_PATH",
        "PROMPTFOO_FETCH_CONNECTIONS",
        "REQUEST_TIMEOUT_MS",
    ):
        monkeypatch.delenv(name, raising=False)
    fetch.clear_agent_cache()
    yield
    fetch.clear_agent_cache()


@pytest.fixture
def sleeps(monkeypatch):
    calls: list[float] = []

    async def fake_sleep(ms: float) -> None:
        calls.append(ms)

    monkeypatch.setattr(fetch, "sleep", fake_sleep)
    return calls


@pytest.fixture
def frozen_now(monkeypatch):
    monkeypatch.setattr(header_parser, "date_now", lambda: NOW_MS)
    return NOW_MS


def make_response(
    status: int = 200,
    status_text: str = "OK",
    headers: dict[str, str] | None = None,
    body: bytes | str = b"",
) -> FetchResponse:
    content = body.encode() if isinstance(body, str) else body
    return FetchResponse(
        status, status_text, {k.lower(): v for k, v in (headers or {}).items()}, content
    )


def scripted(monkeypatch, *results):
    """Replace the transport with a script of responses / exceptions. Returns the call log."""
    calls: list[tuple[str, dict]] = []
    items = list(results)

    async def fake(url, options):
        calls.append((url, options))
        item = items.pop(0) if len(items) > 1 else items[0]
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(fetch, "monkey_patch_fetch", fake)
    return calls


def reason(text: str) -> dict:
    return {"reason_phrase": text.encode()}


# ---- fetchWithProxy ----------------------------------------------------------------------------


@respx.mock
async def test_adds_version_header_to_all_requests():
    route = respx.get("https://example.com/").respond(200, text="ok")
    response = await fetch_with_proxy("https://example.com/", {"headers": {"a": "b"}})
    assert response.ok and response.text() == "ok"
    request = route.calls.last.request
    assert request.headers["x-promptfoo-version"] == fetch.VERSION
    assert request.headers["a"] == "b"


@respx.mock
async def test_basic_auth_from_url_credentials():
    route = respx.get("https://example.com/api").respond(200)
    await fetch_with_proxy("https://user:pass@example.com/api", {"headers": {"x-other": "1"}})
    request = route.calls.last.request
    assert request.headers["authorization"] == "Basic " + base64.b64encode(b"user:pass").decode()
    assert request.headers["x-other"] == "1"
    assert "user" not in str(request.url)


@respx.mock
async def test_only_username_in_url():
    route = respx.get("https://example.com/").respond(200)
    await fetch_with_proxy("https://user@example.com/")
    assert route.calls.last.request.headers["authorization"] == (
        "Basic " + base64.b64encode(b"user:").decode()
    )


@respx.mock
async def test_decodes_percent_encoded_url_credentials():
    route = respx.get("https://example.com/").respond(200)
    await fetch_with_proxy("https://us%3Aer:p%40ss@example.com/")
    assert route.calls.last.request.headers["authorization"] == (
        "Basic " + base64.b64encode(b"us:er:p@ss").decode()
    )


@respx.mock
async def test_keeps_malformed_percent_escapes_in_credentials():
    route = respx.get("https://example.com/").respond(200)
    await fetch_with_proxy("https://user:p%ZZss@example.com/")
    assert route.calls.last.request.headers["authorization"] == (
        "Basic " + base64.b64encode(b"user:p%ZZss").decode()
    )


@respx.mock
async def test_existing_lowercase_authorization_wins_over_url_credentials(caplog):
    route = respx.get("https://example.com/").respond(200)
    with caplog.at_level(logging.WARNING):
        await fetch_with_proxy(
            "https://user:pass@example.com/", {"headers": {"authorization": "Bearer t"}}
        )
    request = route.calls.last.request
    assert request.headers.get_list("authorization") == ["Bearer t"]
    assert "user" not in str(request.url)
    assert "URL credentials will be ignored" in caplog.text


@respx.mock
async def test_no_auth_header_without_url_credentials():
    route = respx.get("https://example.com/").respond(200)
    await fetch_with_proxy("https://example.com/")
    assert "authorization" not in route.calls.last.request.headers


async def test_invalid_url_raises():
    with pytest.raises(JsError, match="Invalid URL"):
        await fetch_with_proxy("")


@respx.mock
async def test_compresses_string_bodies_when_requested():
    route = respx.post("https://example.com/").respond(200)
    await fetch_with_proxy(
        "https://example.com/", {"method": "POST", "body": "hello world", "compress": True}
    )
    request = route.calls.last.request
    assert request.headers["content-encoding"] == "gzip"
    assert gzip.decompress(request.content) == b"hello world"


@respx.mock
async def test_strips_decompression_headers():
    payload = gzip.compress(b'{"a":1}')
    respx.get("https://example.com/").respond(
        200, content=payload, headers={"content-encoding": "gzip", "x-keep": "1"}
    )
    response = await fetch_with_proxy("https://example.com/")
    assert response.json() == {"a": 1}
    assert "content-encoding" not in response.headers
    assert "content-length" not in response.headers
    assert response.headers["x-keep"] == "1"


@respx.mock
async def test_uses_caller_provided_dispatcher():
    route = respx.get("https://example.com/").respond(200)
    async with httpx.AsyncClient() as client:
        await fetch_with_proxy("https://example.com/", {"dispatcher": client})
    assert route.called
    assert fetch._clients == {}


# ---- request-time authentication --------------------------------------------------------------


@respx.mock
async def test_refreshes_credentials_for_direct_transient_retries(sleeps):
    route = respx.get("https://example.com/")
    route.side_effect = [
        httpx.Response(503, extensions=reason("Service Unavailable")),
        httpx.Response(200),
    ]
    tokens = iter(["first", "second"])

    async def get_auth_headers():
        return {"Authorization": f"Bearer {next(tokens)}"}

    response = await fetch_with_proxy("https://example.com/", {"getAuthHeaders": get_auth_headers})
    assert response.status == 200
    sent = [call.request.headers["authorization"] for call in route.calls]
    assert sent == ["Bearer first", "Bearer second"]


@respx.mock
async def test_explicit_headers_override_auth_headers_case_insensitively():
    route = respx.get("https://example.com/").respond(200)
    await fetch_with_proxy(
        "https://example.com/",
        {
            "headers": {"AUTHORIZATION": "Bearer explicit"},
            "getAuthHeaders": lambda: {"Authorization": "Bearer dyn"},
        },
    )
    assert route.calls.last.request.headers.get_list("authorization") == ["Bearer explicit"]


@respx.mock
async def test_credential_resolution_failure_is_not_retried(sleeps):
    route = respx.get("https://example.com/").respond(200)

    def failing():
        raise ValueError("no token")

    with pytest.raises(RequestAuthenticationError, match="no token"):
        await fetch_with_retries("https://example.com/", {"getAuthHeaders": failing}, 1000, 3)
    assert not route.called
    assert sleeps == []


# ---- transient retries in fetchWithProxy --------------------------------------------------------


@respx.mock
async def test_retries_on_503_service_unavailable(sleeps):
    route = respx.get("https://example.com/")
    route.side_effect = [
        httpx.Response(503, extensions=reason("Service Unavailable")),
        httpx.Response(200),
    ]
    response = await fetch_with_proxy("https://example.com/")
    assert response.status == 200
    assert route.call_count == 2
    assert sleeps == [1000]


@respx.mock
async def test_does_not_retry_502_without_matching_status_text(sleeps):
    route = respx.get("https://example.com/").mock(
        return_value=httpx.Response(502, extensions=reason("Authentication failed"))
    )
    response = await fetch_with_proxy("https://example.com/")
    assert response.status == 502
    assert route.call_count == 1
    assert sleeps == []


@respx.mock
async def test_transient_retries_up_to_three_times_with_exponential_backoff(sleeps):
    route = respx.get("https://example.com/").mock(
        return_value=httpx.Response(503, extensions=reason("Service Unavailable"))
    )
    response = await fetch_with_proxy("https://example.com/")
    assert response.status == 503
    assert route.call_count == 4
    assert sleeps == [1000, 2000, 4000]


@respx.mock
async def test_no_transient_retry_when_disabled(sleeps):
    route = respx.get("https://example.com/").mock(
        return_value=httpx.Response(503, extensions=reason("Service Unavailable"))
    )
    await fetch_with_proxy("https://example.com/", {"disableTransientRetries": True})
    assert route.call_count == 1


@respx.mock
async def test_retry_context_zero_disables_transient_retries(sleeps):
    route = respx.get("https://example.com/").mock(
        return_value=httpx.Response(503, extensions=reason("Service Unavailable"))
    )
    with with_fetch_retry_context(0):
        await fetch_with_proxy("https://example.com/")
    assert route.call_count == 1


@respx.mock
async def test_explicit_disable_false_overrides_retry_context_zero(sleeps):
    route = respx.get("https://example.com/")
    route.side_effect = [
        httpx.Response(503, extensions=reason("Service Unavailable")),
        httpx.Response(200),
    ]
    with with_fetch_retry_context(0):
        response = await fetch_with_proxy(
            "https://example.com/", {"disableTransientRetries": False}
        )
    assert response.status == 200
    assert route.call_count == 2


@respx.mock
async def test_retries_cloudflare_524(sleeps):
    route = respx.get("https://example.com/")
    route.side_effect = [
        httpx.Response(524, extensions=reason("A Timeout Occurred")),
        httpx.Response(200),
    ]
    assert (await fetch_with_proxy("https://example.com/")).status == 200
    assert route.call_count == 2


@respx.mock
async def test_fetch_with_retries_disables_transient_retries_in_proxy(sleeps):
    route = respx.get("https://example.com/").mock(
        return_value=httpx.Response(503, extensions=reason("Service Unavailable"))
    )
    response = await fetch_with_retries("https://example.com/", {}, 1000, 2)
    assert response.status == 503
    assert route.call_count == 1


# ---- proxy-from-env ----------------------------------------------------------------------------


def test_proxy_env_precedence(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "http://all:1")
    assert get_proxy_for_url("https://example.com") == "http://all:1"
    monkeypatch.setenv("NPM_CONFIG_PROXY", "http://npm:1")
    assert get_proxy_for_url("https://example.com") == "http://npm:1"
    monkeypatch.setenv("HTTPS_PROXY", "http://https:1")
    assert get_proxy_for_url("https://example.com") == "http://https:1"
    assert get_proxy_for_url("http://example.com") == "http://npm:1"
    monkeypatch.setenv("npm_config_https_proxy", "http://npm-https:1")
    assert get_proxy_for_url("https://example.com") == "http://npm-https:1"


def test_proxy_without_scheme_gets_protocol_prefix(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "proxy.local:8080")
    assert get_proxy_for_url("https://example.com") == "https://proxy.local:8080"


@pytest.mark.parametrize(
    ("no_proxy", "url", "proxied"),
    [
        ("localhost", "http://localhost:3000/x", False),
        ("127.0.0.1", "http://127.0.0.1/x", False),
        ("localhost,127.0.0.1,.internal.com", "https://api.internal.com/", False),
        ("localhost,127.0.0.1,.internal.com", "https://internal.com/", True),
        ("localhost,127.0.0.1", "https://api.example.com/", True),
        ("*.example.com", "https://api.example.com/", False),
        ("*", "https://anything.test/", False),
        ("example.com:8443", "https://example.com:8443/", False),
        ("example.com:8443", "https://example.com/", True),
        ("EXAMPLE.COM", "https://Example.com/", False),
        ("localhost", "https://user:pass@localhost/", False),
    ],
)
def test_no_proxy(monkeypatch, no_proxy, url, proxied):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy:8080")
    monkeypatch.setenv("NO_PROXY", no_proxy)
    assert bool(get_proxy_for_url(url)) is proxied


@pytest.mark.parametrize("url", ["example.com/path", "not a url", "http://[bad"])
def test_bad_urls_have_no_proxy(monkeypatch, url):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    assert get_proxy_for_url(url) == ""


@respx.mock
async def test_requests_use_a_proxy_client_when_configured(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:8080")
    respx.get("https://example.com/").respond(200)
    await fetch_with_proxy("https://example.com/")
    assert any(key[0] == "http://proxy.local:8080" for key in fetch._clients)


@respx.mock
async def test_clients_are_reused_per_concurrency(monkeypatch):
    respx.get("https://example.com/").respond(200)
    await asyncio.gather(*(fetch_with_proxy("https://example.com/") for _ in range(3)))
    assert len(fetch._clients) == 1
    with cli_state.with_max_concurrency(8):
        await fetch_with_proxy("https://example.com/")
    assert sorted(key[2] for key in fetch._clients) == [4, 8]


# ---- TLS and pool options -----------------------------------------------------------------------


def test_tls_verifies_by_default():
    # Upstream skips verification by default; promptphoo verifies (D61).
    assert fetch.get_tls_options().reject_unauthorized is True
    assert fetch._verify_for(fetch.get_tls_options()) is True


def test_insecure_ssl_true_disables_verification(monkeypatch):
    monkeypatch.setenv("PROMPTFOO_INSECURE_SSL", "true")
    tls = fetch.get_tls_options()
    assert tls.reject_unauthorized is False
    assert fetch._verify_for(tls) is False


def test_insecure_ssl_false_enables_verification(monkeypatch):
    monkeypatch.setenv("PROMPTFOO_INSECURE_SSL", "false")
    tls = fetch.get_tls_options()
    assert tls.reject_unauthorized is True
    assert fetch._verify_for(tls) is True


def test_ca_cert_path_relative_to_base_path(monkeypatch, tmp_path):
    (tmp_path / "ca.pem").write_text("CERT")
    monkeypatch.setenv("PROMPTFOO_CA_CERT_PATH", "ca.pem")
    with cli_state.with_base_path(str(tmp_path)):
        assert fetch.get_tls_options().ca == "CERT"


def test_missing_ca_cert_warns(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("PROMPTFOO_CA_CERT_PATH", str(tmp_path / "missing.pem"))
    with caplog.at_level(logging.WARNING):
        assert fetch.get_tls_options().ca is None
    assert "Failed to read CA certificate" in caplog.text


@pytest.mark.parametrize(("value", "expected"), [(" 12 ", 12), ("0", 4), ("abc", 4), ("1.5", 4)])
def test_connection_pool_size(monkeypatch, value, expected):
    monkeypatch.setenv("PROMPTFOO_FETCH_CONNECTIONS", value)
    assert fetch.get_connection_pool_size() == expected


def test_connection_pool_size_follows_max_concurrency():
    with cli_state.with_max_concurrency(9):
        assert fetch.get_connection_pool_size() == 9


async def test_large_pools_are_split_into_small_clients():
    # httpcore's pool cost is quadratic in its size, so -j 256 is served by 16 clients of 16
    # keep-alive connections each, used in turn: the same 256 connections in total.
    with cli_state.with_max_concurrency(256):
        clients = [fetch.get_client() for _ in range(32)]
    ((group, _, _),) = fetch._clients.values()
    assert len(group) == 16
    assert {c._transport._pool._max_keepalive_connections for c in group} == {16}
    assert clients[:16] == list(group) and clients[16:] == list(group)
    contexts = {id(c._transport._pool._ssl_context) for c in group}
    assert len(contexts) == 1  # one CA bundle load for the whole group
    await fetch.close_client()
    assert all(c.is_closed for c in group)


async def test_small_pools_keep_one_client():
    with cli_state.with_max_concurrency(16):
        assert fetch.get_client() is fetch.get_client()
    ((group, _, _),) = fetch._clients.values()
    assert len(group) == 1
    await fetch.close_client()


def test_missing_sniffio_is_recorded_once():
    # httpcore imports sniffio on every lock operation; without the sys.modules entry each
    # failed import rescanned sys.path.
    import importlib.util
    import sys

    if importlib.util.find_spec("sniffio") is None and "sniffio" not in sys.modules:
        pytest.fail("fetch did not record the missing sniffio module")
    if sys.modules.get("sniffio", ...) is None:
        with pytest.raises(ImportError):
            import sniffio  # noqa: F401


def test_request_timeout_ms(monkeypatch):
    assert fetch.get_request_timeout_ms() == 300_000
    monkeypatch.setenv("REQUEST_TIMEOUT_MS", "1234")
    assert fetch.get_request_timeout_ms() == 1234


# ---- fetchWithTimeout --------------------------------------------------------------------------


@respx.mock
async def test_fetch_with_timeout_resolves():
    respx.get("https://example.com/").respond(200, json={"ok": True})
    response = await fetch_with_timeout("https://example.com/", {}, 5000)
    assert response.json() == {"ok": True}


async def test_fetch_with_timeout_rejects_on_timeout(monkeypatch):
    async def slow(url, options):
        await asyncio.sleep(10)

    monkeypatch.setattr(fetch, "monkey_patch_fetch", slow)
    with pytest.raises(JsError, match="^Request timed out after 20 ms$"):
        await fetch_with_timeout("https://example.com/", {}, "20")


# ---- isRateLimited / isTransientError ----------------------------------------------------------


def test_is_rate_limited():
    assert is_rate_limited(make_response(headers={"X-RateLimit-Remaining": "0"}))
    assert is_rate_limited(make_response(429, "Too Many Requests"))
    assert is_rate_limited(make_response(headers={"x-ratelimit-remaining-requests": "0"}))
    assert is_rate_limited(make_response(headers={"x-ratelimit-remaining-tokens": "0"}))
    assert not is_rate_limited(make_response(headers={"X-RateLimit-Remaining": "10"}))


def test_is_rate_limited_requires_status():
    with pytest.raises(JsError, match="Response status is missing"):
        is_rate_limited(make_response(0))


@pytest.mark.parametrize(
    ("status", "text", "expected"),
    [
        (502, "Bad Gateway", True),
        (503, "Service Unavailable", True),
        (504, "Gateway Timeout", True),
        (502, "BAD GATEWAY", True),
        (503, "service unavailable", True),
        (502, "Proxy Error", False),
        (500, "Internal Server Error", False),
        (524, "A Timeout Occurred", True),
        (524, "timeout", True),
        (524, "A TIMEOUT OCCURRED", True),
        (524, "Origin Error", False),
        (503, "", False),
    ],
)
def test_is_transient_error(status, text, expected):
    assert is_transient_error(make_response(status, text)) is expected


# ---- computeRateLimitWaitMs / handleRateLimit --------------------------------------------------


def test_compute_wait_default(frozen_now):
    assert compute_rate_limit_wait_ms(make_response()) == 60_000


def test_compute_wait_retry_after_seconds(frozen_now):
    assert compute_rate_limit_wait_ms(make_response(headers={"Retry-After": "7"})) == 7_000


@pytest.mark.parametrize(
    ("value", "expected"), [("25", 25), ("0", 0), ("invalid", 7000), ("-1", 7000)]
)
def test_compute_wait_retry_after_ms(frozen_now, value, expected):
    response = make_response(headers={"retry-after-ms": value, "Retry-After": "7"})
    assert compute_rate_limit_wait_ms(response) == expected


def test_compute_wait_prefers_openai_reset(frozen_now):
    response = make_response(headers={"x-ratelimit-reset-requests": "3s", "Retry-After": "60"})
    assert compute_rate_limit_wait_ms(response) == 3_000


def test_compute_wait_non_finite_reset_falls_back(frozen_now):
    response = make_response(headers={"X-RateLimit-Reset": "Infinity", "Retry-After": "7"})
    assert compute_rate_limit_wait_ms(response) == 7_000


def test_compute_wait_standard_reset_header(frozen_now):
    reset_seconds = (NOW_MS + 5000) // 1000
    response = make_response(headers={"X-RateLimit-Reset": str(reset_seconds)})
    assert compute_rate_limit_wait_ms(response) == 6000


@pytest.mark.parametrize(
    ("headers", "base"),
    [
        ({"x-ratelimit-reset-requests": "5"}, "base 5000ms"),
        ({"x-ratelimit-reset-requests": "500ms"}, "base 500ms"),
        ({"x-ratelimit-reset-requests": "1m30s"}, "base 90000ms"),
        ({"Retry-After": "5"}, "base 5000ms"),
        ({"Retry-After": "Mon, 01 Jan 2024 00:00:05 GMT"}, "base 5000ms"),
        ({"Retry-After": "invalid"}, "base 60000ms"),
        ({}, "base 60000ms"),
    ],
)
async def test_handle_rate_limit_logs_base_wait(
    frozen_now, sleeps, monkeypatch, caplog, headers, base
):
    monkeypatch.setattr(fetch.random, "random", lambda: 0)
    with caplog.at_level(logging.DEBUG, logger="promptphoo.fetch"):
        await handle_rate_limit(make_response(429, headers=headers))
    assert f"{base} + 0ms jitter" in caplog.text
    assert len(sleeps) == 1


async def test_handle_rate_limit_adds_bounded_jitter(frozen_now, sleeps, monkeypatch, caplog):
    values = iter([0.5, 0.1, 0.9])
    monkeypatch.setattr(fetch.random, "random", lambda: next(values))
    with caplog.at_level(logging.DEBUG, logger="promptphoo.fetch"):
        for _ in range(3):
            await handle_rate_limit(make_response(headers={"Retry-After": "5"}))
    assert "base 5000ms + 500ms jitter" in caplog.text
    assert sleeps == [5500, 5100, 5900]


async def test_handle_rate_limit_reports_observation(frozen_now, sleeps):
    observations = []
    await handle_rate_limit(make_response(429, headers={"Retry-After": "2"}), observations.append)
    assert observations == [
        {"headers": {"retry-after": "2"}, "status": 429, "resetAt": NOW_MS + 2000}
    ]


def test_rate_limit_timing_from_headers(frozen_now):
    assert rate_limit_timing_from_headers({"Retry-After": "3"}) == {
        "retryAfterMs": 3000,
        "resetAt": NOW_MS + 3000,
    }
    assert rate_limit_timing_from_headers({}) == {}


def test_read_bounded_text():
    response = make_response(body="héllo world")
    assert read_bounded_text(response, 100) == "héllo world"
    assert read_bounded_text(response, 1) == "h"


# ---- fetchWithRetries --------------------------------------------------------------------------


async def test_rate_limit_test_should_fetch_data(monkeypatch, sleeps):
    scripted(monkeypatch, make_response(body='{"a":1}'))
    response = await fetch_with_retries("https://example.com", {}, 1000)
    assert response.json() == {"a": 1}


async def test_exactly_one_attempt_when_retries_is_zero(monkeypatch, sleeps):
    calls = scripted(monkeypatch, make_response())
    await fetch_with_retries("https://example.com", {}, 1000, 0)
    assert len(calls) == 1
    assert sleeps == []


async def test_negative_retries_treated_as_zero(monkeypatch, sleeps):
    calls = scripted(monkeypatch, make_response())
    await fetch_with_retries("https://example.com", {}, 1000, -1)
    assert len(calls) == 1


async def test_honors_retry_context_when_argument_omitted(monkeypatch, sleeps):
    calls = scripted(monkeypatch, JsError("Network error"))
    with (
        with_fetch_retry_context(0),
        pytest.raises(JsError, match=r"^Request failed after 0 retries: Error: Network error$"),
    ):
        await fetch_with_retries("https://example.com", {}, 1000)
    assert len(calls) == 1
    assert sleeps == []


async def test_explicit_max_retries_beats_retry_context(monkeypatch, sleeps):
    calls = scripted(monkeypatch, JsError("Network error"))
    with (
        with_fetch_retry_context(0),
        pytest.raises(JsError, match="Request failed after 2 retries: Error: Network error"),
    ):
        await fetch_with_retries("https://example.com", {}, 1000, 2)
    assert len(calls) == 3
    assert len(sleeps) == 2


async def test_inner_retry_context_shadows_outer(monkeypatch, sleeps):
    calls = scripted(monkeypatch, JsError("Network error"))
    with (
        with_fetch_retry_context(0),
        with_fetch_retry_context(2),
        pytest.raises(JsError, match="after 2 retries"),
    ):
        await fetch_with_retries("https://example.com", {}, 1000)
    assert len(calls) == 3


async def test_default_max_retries_is_four(monkeypatch, sleeps):
    calls = scripted(monkeypatch, JsError("Network error"))
    with pytest.raises(JsError, match="Request failed after 4 retries: Error: Network error"):
        await fetch_with_retries("https://example.com", {}, 1000)
    assert len(calls) == 5
    assert len(sleeps) == 4


async def test_does_not_sleep_after_final_attempt(monkeypatch, sleeps):
    calls = scripted(monkeypatch, JsError("Network error"))
    with pytest.raises(JsError, match="after 1 retries"):
        await fetch_with_retries("https://example.com", {}, 1000, 1)
    assert len(calls) == 2
    assert len(sleeps) == 1


async def test_exponential_backoff_uses_request_backoff_env(monkeypatch, sleeps):
    monkeypatch.setenv("PROMPTFOO_REQUEST_BACKOFF_MS", "100")
    monkeypatch.setattr(fetch.random, "random", lambda: 0.5)
    scripted(monkeypatch, JsError("Network error"))
    with pytest.raises(JsError):
        await fetch_with_retries("https://example.com", {}, 1000, 3)
    assert sleeps == [600, 1200, 2400]


async def test_detailed_error_information(monkeypatch, sleeps):
    scripted(monkeypatch, JsError("Network error", code="ECONNREFUSED", cause="Connection refused"))
    with pytest.raises(JsError) as info:
        await fetch_with_retries("https://example.com", {}, 1000, 1)
    assert str(info.value) == (
        "Request failed after 1 retries: Error: Network error (Cause: Connection refused) "
        "(Code: ECONNREFUSED)"
    )


@respx.mock
async def test_transport_exceptions_are_retried(sleeps):
    route = respx.get("https://example.com/")
    route.side_effect = [httpx.ConnectError("connection refused"), httpx.Response(200)]
    response = await fetch_with_retries("https://example.com/", {}, 1000, 2)
    assert response.status == 200
    assert len(sleeps) == 1


async def test_redacts_credentials_in_retry_failure(monkeypatch, sleeps, caplog):
    url = "https://webhook-user:webhook-password@n8n.example.com/webhook/agent?token=webhook-secret"
    scripted(monkeypatch, JsError(f"Network error for {url}"))
    with caplog.at_level(logging.DEBUG, logger="promptphoo.fetch"), pytest.raises(JsError) as info:
        await fetch_with_retries(url, {}, 1000, 0)
    for output in (str(info.value), caplog.records[-1].getMessage()):
        assert "n8n.example.com" in output
        assert "webhook-user" not in output
        assert "webhook-password" not in output
        assert "webhook-secret" not in output


async def test_retries_5xx_when_enabled(monkeypatch, sleeps):
    monkeypatch.setenv("PROMPTFOO_RETRY_5XX", "true")
    calls = scripted(monkeypatch, make_response(502, "Bad Gateway"), make_response())
    response = await fetch_with_retries("https://example.com", {}, 1000, 2)
    assert response.status == 200
    assert len(calls) == 2
    assert len(sleeps) == 1


async def test_5xx_returned_when_retry_5xx_disabled(monkeypatch, sleeps):
    calls = scripted(monkeypatch, make_response(500, "Internal Server Error"))
    response = await fetch_with_retries("https://example.com", {}, 1000, 2)
    assert response.status == 500
    assert len(calls) == 1


async def test_5xx_retry_exhaustion_message(monkeypatch, sleeps):
    monkeypatch.setenv("PROMPTFOO_RETRY_5XX", "true")
    scripted(monkeypatch, make_response(500, "Internal Server Error"))
    with pytest.raises(
        JsError,
        match=r"^Request failed after 1 retries: Error: Internal Server Error: 500 Internal Server "
        r"Error$",
    ):
        await fetch_with_retries("https://example.com", {}, 1000, 1)


async def test_rate_limit_with_retry_after_then_success(monkeypatch, sleeps, caplog):
    calls = scripted(
        monkeypatch, make_response(429, "Too Many Requests", {"Retry-After": "1"}), make_response()
    )
    with caplog.at_level(logging.DEBUG, logger="promptphoo.fetch"):
        response = await fetch_with_retries("https://example.com", {}, 1000, 2)
    assert response.status == 200
    assert len(calls) == 2
    assert len(sleeps) == 1
    assert sleeps[0] >= 1000
    assert "Rate limited on URL" in caplog.text
    assert "attempt 1/3" in caplog.text


async def test_rate_limit_using_x_limit_headers(monkeypatch, sleeps, frozen_now):
    reset = str((NOW_MS + 4000) // 1000)
    scripted(
        monkeypatch,
        make_response(headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": reset}),
        make_response(body="ok"),
    )
    monkeypatch.setattr(fetch.random, "random", lambda: 0)
    response = await fetch_with_retries("https://example.com", {}, 1000, 2)
    assert response.text() == "ok"
    assert sleeps == [5000]


async def test_rate_limit_default_wait(monkeypatch, sleeps, frozen_now):
    monkeypatch.setattr(fetch.random, "random", lambda: 0)
    scripted(monkeypatch, make_response(429, "Too Many Requests"), make_response())
    await fetch_with_retries("https://example.com", {}, 1000, 2)
    assert sleeps == [60_000]


async def test_rate_limit_exhaustion_raises_structured_error(monkeypatch, sleeps):
    scripted(monkeypatch, make_response(429, "Too Many Requests"))
    with pytest.raises(HttpRateLimitError, match=r"Rate limit exceeded.*429.*Too Many Requests"):
        await fetch_with_retries("https://example.com", {}, 1000, 2)


async def test_fail_fast_when_max_retries_zero_without_sleeping(monkeypatch, sleeps):
    calls = scripted(monkeypatch, make_response(429, "Too Many Requests", {"Retry-After": "60"}))
    with pytest.raises(HttpRateLimitError):
        await fetch_with_retries("https://example.com", {}, 1000, 0)
    assert len(calls) == 1
    assert sleeps == []


async def test_soft_rate_limit_exhaustion_is_plain_error(monkeypatch, sleeps):
    scripted(monkeypatch, make_response(200, "OK", {"x-ratelimit-remaining-requests": "0"}))
    # The plain error is caught by the retry loop like any transport failure, as upstream.
    with pytest.raises(
        JsError,
        match=r"^Request failed after 0 retries: Error: Rate limited: 200 OK after 1 attempts$",
    ) as info:
        await fetch_with_retries("https://example.com", {}, 1000, 0)
    assert not isinstance(info.value, HttpRateLimitError)


async def test_rate_limit_logs_redact_query_secrets(monkeypatch, sleeps, caplog):
    scripted(monkeypatch, make_response(429, "Too Many Requests"))
    with (
        caplog.at_level(logging.DEBUG, logger="promptphoo.fetch"),
        pytest.raises(HttpRateLimitError),
    ):
        await fetch_with_retries(
            "https://n8n.example.com/webhook/agent?token=webhook-secret", {}, 1000, 0
        )
    assert "n8n.example.com" in caplog.text
    assert "webhook-secret" not in caplog.text


async def test_abort_errors_are_not_retried(monkeypatch, sleeps):
    calls = scripted(monkeypatch, JsError("The operation was aborted", name="AbortError"))
    with pytest.raises(JsError, match="aborted"):
        await fetch_with_retries("https://example.com", {}, 1000, 3)
    assert len(calls) == 1
    assert sleeps == []


async def test_cancellation_propagates(monkeypatch):
    async def slow(url, options):
        await asyncio.sleep(10)

    monkeypatch.setattr(fetch, "monkey_patch_fetch", slow)
    task = asyncio.ensure_future(fetch_with_retries("https://example.com", {}, 60_000, 3))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _json_429(body, headers=None, status_text="Too Many Requests"):
    import json

    return make_response(429, status_text, headers, json.dumps(body))


async def test_rate_limit_error_from_body_code(monkeypatch, sleeps):
    scripted(
        monkeypatch,
        _json_429(
            {"error": {"code": "rate_limit_exceeded", "message": "slow"}}, {"Retry-After": "7"}
        ),
    )
    with pytest.raises(HttpRateLimitError) as info:
        await fetch_with_retries("https://example.com", {}, 1000, 1)
    err = info.value
    assert (err.kind, err.status, err.code, err.retry_after_ms) == (
        "rate_limit",
        429,
        "rate_limit_exceeded",
        7000,
    )


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"code": "insufficient_quota", "type": "insufficient_quota"}},
        {"error": {"code": "credit_balance_exhausted", "type": "insufficient_quota"}},
        {"error": {"code": "billing_hard_limit_reached"}},
        {"error": {"message": "x", "metadata": {"provider_code": "CREDIT_BALANCE_EXHAUSTED"}}},
        {"error": {"code": "weird_code", "type": "insufficient_quota"}},
    ],
)
async def test_quota_fails_fast(monkeypatch, sleeps, body):
    calls = scripted(monkeypatch, _json_429(body))
    with pytest.raises(HttpRateLimitError) as info:
        await fetch_with_retries("https://example.com", {}, 1000, 3)
    assert info.value.kind == "quota"
    assert str(info.value).startswith("Quota exceeded: HTTP 429 Too Many Requests")
    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"code": "rate_limit_exceeded", "type": "quota_exceeded"}},
        {"error": {"code": "insufficient_quota", "message": "deployment saturated"}},
    ],
)
async def test_short_retry_after_downgrades_quota(monkeypatch, sleeps, body):
    calls = scripted(monkeypatch, _json_429(body, {"Retry-After": "1"}))
    with pytest.raises(HttpRateLimitError) as info:
        await fetch_with_retries("https://example.com", {}, 1000, 1)
    assert info.value.kind == "rate_limit"
    assert len(calls) == 2
    assert len(sleeps) == 1


@pytest.mark.parametrize("wait_ms", [0, 25])
async def test_retry_after_ms_backoff(monkeypatch, sleeps, wait_ms):
    monkeypatch.setattr(fetch.random, "random", lambda: 0)
    success = make_response()
    scripted(
        monkeypatch,
        _json_429({"error": {"code": "insufficient_quota"}}, {"retry-after-ms": str(wait_ms)}),
        success,
    )
    response = await fetch_with_retries("https://example.com", {}, 1000, 1)
    assert response is success
    assert sleeps == [wait_ms]


async def test_structured_error_preserves_headers_and_reset(monkeypatch, sleeps, frozen_now):
    scripted(
        monkeypatch,
        _json_429(
            {},
            {
                "Retry-After": "12",
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-reset-requests": "12s",
            },
        ),
    )
    with pytest.raises(HttpRateLimitError) as info:
        await fetch_with_retries("https://example.com", {}, 1000, 0)
    err = info.value
    assert err.retry_after_ms == 12_000
    assert err.reset_at == NOW_MS + 12_000
    assert err.headers is not None and err.headers["retry-after"] == "12"


async def test_plain_text_429_body_preserved(monkeypatch, sleeps):
    scripted(
        monkeypatch, make_response(429, "Too Many Requests", body="plain text rate limit notice")
    )
    with pytest.raises(HttpRateLimitError) as info:
        await fetch_with_retries("https://example.com", {}, 1000, 0)
    assert info.value.kind == "rate_limit"
    assert info.value.code is None
    assert info.value.body == "plain text rate limit notice"


async def test_empty_429_body(monkeypatch, sleeps):
    scripted(monkeypatch, make_response(429, "Too Many Requests"))
    with pytest.raises(HttpRateLimitError) as info:
        await fetch_with_retries("https://example.com", {}, 1000, 0)
    assert info.value.body is None


async def test_anthropic_error_type_classification(monkeypatch, sleeps):
    scripted(monkeypatch, _json_429({"type": "error", "error": {"type": "rate_limit_error"}}))
    with pytest.raises(HttpRateLimitError) as info:
        await fetch_with_retries("https://example.com", {}, 1000, 0)
    assert info.value.code == "rate_limit_error"
    assert info.value.kind == "rate_limit"


async def test_peek_bounds_429_body(monkeypatch, sleeps):
    big = b'{"error":{"code":"insufficient_quota"}}' + b" " * (fetch.RATE_LIMIT_BODY_PEEK_BYTES * 2)
    scripted(monkeypatch, make_response(429, "Too Many Requests", body=big))
    with pytest.raises(HttpRateLimitError) as info:
        await fetch_with_retries("https://example.com", {}, 1000, 0)
    # The JSON prefix is valid, but the trailing whitespace beyond the cap is not read.
    assert info.value.kind == "quota"


async def test_rate_limit_observer_receives_backoff(monkeypatch, sleeps, frozen_now):
    scripted(
        monkeypatch, make_response(429, "Too Many Requests", {"Retry-After": "3"}), make_response()
    )
    observations = []
    await fetch_with_retries(
        "https://example.com", {}, 1000, 1, on_rate_limit_backoff=observations.append
    )
    assert [o["resetAt"] for o in observations] == [NOW_MS + 3000]


# ---- errors.ts ---------------------------------------------------------------------------------


def test_is_abort_error():
    assert fetch_errors.is_abort_error(JsError("x", name="AbortError"))
    assert fetch_errors.is_abort_error(JsError("x", name="AbortException"))
    assert fetch_errors.is_abort_error(asyncio.CancelledError())
    assert not fetch_errors.is_abort_error(JsError("x"))
    assert not fetch_errors.is_abort_error("AbortError")


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, True), (403, True), (404, True), (501, True), (500, False), (200, False), (429, False)],
)
def test_is_non_transient_http_status(status, expected):
    assert fetch_errors.is_non_transient_http_status(status) is expected


def test_find_target_error_status():
    def result(status):
        return {"response": {"metadata": {"http": {"status": status}}}}

    assert fetch_errors.find_target_error_status([]) is None
    assert fetch_errors.find_target_error_status([{"response": {}}]) is None
    assert fetch_errors.find_target_error_status([result(429), result(500), result(200)]) is None
    assert fetch_errors.find_target_error_status([result(200), result(403), result(404)]) == 403


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (None, False),
        (JsError("x", code="ECONNRESET"), True),
        (JsError("x", code="EPIPE"), True),
        (JsError("read ECONNRESET"), True),
        (JsError("Socket Hang Up"), True),
        (JsError("Bad Record MAC"), True),
        (JsError("write EPROTO"), True),
        (JsError("EPROTO wrong version number"), False),
        (JsError("EPROTO unable to verify the first certificate"), False),
        (JsError("EPROTO unknown ca"), False),
        (JsError("EPROTO certificate verify failed"), False),
        (JsError("self signed certificate"), False),
        (ConnectionResetError("reset"), True),
    ],
)
def test_is_transient_connection_error(error, expected):
    assert fetch_errors.is_transient_connection_error(error) is expected


def test_extract_rate_limit_error_code():
    extract = fetch_errors.extract_rate_limit_error_code
    assert extract({"error": {"code": "insufficient_quota"}}) == "insufficient_quota"
    assert extract({"error": {"type": "rate_limit_error"}}) == "rate_limit_error"
    assert extract({"error": {"code": "a", "type": "b"}}) == "a"
    assert (
        extract({"error": {"code": "x", "metadata": {"provider_code": "Billing_Not_Active"}}})
        == "billing_not_active"
    )
    assert extract({"code": "top"}) == "top"
    assert extract({"code": "root", "error": {"type": "nested"}}) == "root"
    assert extract({"code": "transport", "error": {"code": "body"}}) == "body"
    assert extract({"type": "t"}) == "t"
    assert extract("str") is None
    assert extract(None) is None
    assert extract({"message": "x"}) is None
    assert extract({"error": {"code": ""}}) is None


def test_extract_rate_limit_error_type():
    extract = fetch_errors.extract_rate_limit_error_type
    assert extract({"error": {"type": "insufficient_quota"}}) == "insufficient_quota"
    assert extract({"type": "root"}) == "root"
    assert extract({"type": "root", "error": {"code": "c"}}) == "root"
    assert extract({"error": {"code": "c"}}) is None


@pytest.mark.parametrize(
    ("kwargs", "kind"),
    [
        ({"code": "insufficient_quota"}, "quota"),
        ({"code": "credit_balance_exhausted"}, "quota"),
        ({"code": "rate_limit_exceeded"}, "rate_limit"),
        ({"code": None}, "rate_limit"),
        ({"code": "credit_balance_exhausted", "type": "insufficient_quota"}, "quota"),
        ({"code": "mystery", "type": "insufficient_quota"}, "quota"),
        ({"code": "rate_limit_exceeded", "type": "insufficient_quota"}, "rate_limit"),
        ({"code": "rate_limit_exceeded", "type": "credit_balance_exhausted"}, "quota"),
        ({"type": "credit_balance_exhausted", "retry_after_ms": 1000}, "quota"),
        ({"code": "x", "type": "y"}, "rate_limit"),
        ({"type": "insufficient_quota", "retry_after_ms": 1000}, "rate_limit"),
        ({"type": "insufficient_quota", "reset_at": NOW_MS + 60_000}, "rate_limit"),
        ({"type": "insufficient_quota", "reset_at": NOW_MS + 2 * 3600_000}, "quota"),
        ({"code": "billing_hard_limit_reached", "retry_after_ms": 1000}, "quota"),
    ],
)
def test_http_rate_limit_error_classification(frozen_now, kwargs, kind):
    assert HttpRateLimitError(status=429, **kwargs).kind == kind


def test_http_rate_limit_error_fields_and_message():
    headers = {"retry-after": "5"}
    err = HttpRateLimitError(
        status=429,
        status_text="",
        retry_after_ms=5000,
        reset_at=-1,
        code="rate_limit_exceeded",
        headers=headers,
        body={"a": 1},
    )
    headers["retry-after"] = "mutated"
    assert err.status_text == "Too Many Requests"
    assert err.reset_at is None
    assert err.headers == {"retry-after": "5"}
    assert err.body == {"a": 1}
    assert str(err) == "Rate limit exceeded: HTTP 429 Too Many Requests (code: rate_limit_exceeded)"
    assert err.name == "HttpRateLimitError"
    assert HttpRateLimitError(status=429, retry_after_ms=math.inf).retry_after_ms is None


def test_format_rate_limit_error_message(frozen_now):
    fmt = fetch_errors.format_rate_limit_error_message
    rl = HttpRateLimitError(status=429, code="rate_limit_exceeded", retry_after_ms=7400)
    assert fmt(rl) == (
        "Rate limit exceeded: HTTP 429 Too Many Requests (code: rate_limit_exceeded) "
        "[retry after 7s]"
    )
    reset = HttpRateLimitError(status=429, reset_at=NOW_MS + 12_500)
    assert fmt(reset, "deployment x") == (
        "Rate limit exceeded: HTTP 429 Too Many Requests deployment x [resets in 13s]"
    )
    quota = HttpRateLimitError(status=429, code="insufficient_quota")
    assert fmt(quota) == (
        "Quota exceeded: HTTP 429 Too Many Requests (code: insufficient_quota). Retries will "
        "not help — check your billing or daily quota."
    )
    assert fetch_errors.format_rate_limit_detail(quota) == ""


# ---- scheduler/headerParser.ts -----------------------------------------------------------------


def test_parse_rate_limit_headers(frozen_now):
    parsed = header_parser.parse_rate_limit_headers(
        {
            "x-ratelimit-remaining-requests": "10",
            "anthropic-ratelimit-tokens-remaining": "500",
            "RateLimit-Limit": "100",
            "anthropic-ratelimit-tokens-reset": "2024-01-01T00:00:30Z",
            "retry-after": "4",
        }
    )
    assert parsed == {
        "remainingRequests": 10,
        "remainingTokens": 500,
        "limitRequests": 100,
        "resetAt": NOW_MS + 30_000,
        "retryAfterMs": 4000,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1s", NOW_MS + 1000),
        ("100ms", NOW_MS + 100),
        ("1h30s", NOW_MS + 3_630_000),
        ("2h15m30s", NOW_MS + 8_130_000),
        ("1.5", NOW_MS + 1500),
        ("1704067260", 1_704_067_260_000),
        ("1704067260000", 1_704_067_260_000),
        ("2026-08-07T21:00:00Z", None),
        ("garbage", None),
    ],
)
def test_reset_header_formats(frozen_now, value, expected):
    parsed = header_parser.parse_rate_limit_headers({"x-ratelimit-reset": value})
    assert parsed.get("resetAt") == expected


def test_parse_retry_after():
    assert header_parser.parse_retry_after("0") == 0
    assert header_parser.parse_retry_after(" 3 ") == 3000
    assert header_parser.parse_retry_after("3.5") is None
    assert header_parser.parse_retry_after("soon") is None
