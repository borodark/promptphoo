"""HTTP provider. Ports the deterministic cases of test/providers/http/*.test.ts,
test/providers/httpTransforms.test.ts and test/providers/http-multipart.test.ts."""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import logging
from typing import Any

import httpx
import pytest
import respx

from promptphoo import cache, cli_state
from promptphoo import fetch as fetch_module
from promptphoo.fetch import FetchWithCacheResult
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.http import (
    HttpProvider,
    create_provider,
    create_session_parser,
    create_transform_request,
    create_transform_response,
    create_validate_status,
    determine_request_body,
    diagnose_private_key_material,
    escape_json_variables,
    estimate_token_count,
    extract_body_from_raw_request,
    generate_signature,
    parse_raw_request,
    preprocess_signature_auth_config,
    process_json_body,
    process_text_body,
    render_raw_request_with_nunjucks,
    url_encode_raw_request_path,
)
from promptphoo.providers.http import signature as signature_module

MOCK_URL = "http://example.com/api"


class FakeFetch:
    """Stands in for fetch_with_cache, recording (url, options, timeout, format, bust, retries)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.responses: list[Any] = []

    def add(
        self,
        data: Any = "",
        status: int = 200,
        status_text: str = "OK",
        headers: dict[str, str] | None = None,
        cached: bool = False,
    ) -> None:
        self.responses.append(
            FetchWithCacheResult(
                data=data,
                cached=cached,
                status=status,
                status_text=status_text,
                headers=headers or {},
            )
        )

    def add_error(self, error: Exception) -> None:
        self.responses.append(error)

    async def __call__(
        self, url, options=None, timeout=None, format="json", bust=False, retries=None
    ):
        self.calls.append((url, options, timeout, format, bust, retries))
        if not self.responses:
            raise AssertionError("fetch_with_cache called without a queued response")
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response

    @property
    def last(self) -> tuple[Any, ...]:
        return self.calls[-1]


@pytest.fixture
def fake_fetch(monkeypatch):
    fake = FakeFetch()
    monkeypatch.setattr(fetch_module, "fetch_with_cache", fake)
    return fake


@pytest.fixture(autouse=True)
def _isolation():
    with cache.with_cache_enabled(False), cli_state.with_config({}):
        yield


def _ctx(**vars: Any) -> dict[str, Any]:
    return {"vars": vars, "prompt": {"raw": "p", "label": "p"}}


# --- basic calls (core.test.ts) --------------------------------------------------------------


async def test_calls_the_api_and_returns_the_response(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {"key": "{{ prompt }}"},
                "transformResponse": lambda data: data["result"],
            }
        },
    )
    fake_fetch.add(json.dumps({"result": "response text"}))
    result = await provider.call_api("test prompt")
    assert result["output"] == "response text"
    url, options, timeout, fmt, bust, retries = fake_fetch.last
    assert url == MOCK_URL
    assert options == {
        "method": "POST",
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"key": "test prompt"}, separators=(",", ":")),
    }
    assert isinstance(timeout, int)
    assert (fmt, bust, retries) == ("text", False, None)


async def test_api_call_errors_propagate(fake_fetch):
    provider = HttpProvider(MOCK_URL, {"config": {"method": "POST", "body": {"key": "v"}}})
    fake_fetch.add_error(RuntimeError("Network error"))
    with pytest.raises(RuntimeError, match="Network error"):
        await provider.call_api("test prompt")


async def test_preserves_original_token_usage_when_transform_returns_text(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "POST",
                "body": {"prompt": "{{ prompt }}"},
                "transformResponse": lambda data: data["output"],
            }
        },
    )
    usage = {"prompt": 12, "completion": 7, "total": 19, "completionDetails": {"reasoning": 3}}
    fake_fetch.add(json.dumps({"output": "response text", "tokenUsage": usage}))
    result = await provider.call_api("test prompt")
    assert result["output"] == "response text"
    assert result["tokenUsage"] == usage


async def test_custom_method_headers_and_query_params(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "PATCH",
                "headers": {"Authorization": "Bearer token"},
                "body": {"key": "{{ prompt }}"},
                "queryParams": {"foo": "bar"},
            }
        },
    )
    fake_fetch.add("custom response")
    await provider.call_api("test prompt")
    url, options, *_ = fake_fetch.last
    assert url == f"{MOCK_URL}?foo=bar"
    assert options["method"] == "PATCH"
    assert options["headers"] == {
        "content-type": "application/json",
        "authorization": "Bearer token",
    }


async def test_substitutes_url_path_variables(fake_fetch):
    provider = HttpProvider(
        "http://example.com/users/{{userId}}/profile",
        {"config": {"method": "GET", "headers": {"Content-Type": "application/json"}}},
    )
    fake_fetch.add(json.dumps({"user": "data"}))
    await provider.call_api("test prompt", _ctx(userId="12345"))
    url, options, *_ = fake_fetch.last
    assert url == "http://example.com/users/12345/profile"
    assert options == {"method": "GET", "headers": {"content-type": "application/json"}}


@pytest.mark.parametrize(
    ("parser", "expected"),
    [
        (lambda data: data["custom"], "parsed"),
        ("json.result", "parsed"),
        ("text", json.dumps({"result": "parsed", "custom": "parsed"})),
    ],
)
async def test_response_transform_types(fake_fetch, parser, expected):
    provider = HttpProvider(
        MOCK_URL, {"config": {"body": {"key": "{{ prompt }}"}, "transformResponse": parser}}
    )
    fake_fetch.add(json.dumps({"result": "parsed", "custom": "parsed"}))
    result = await provider.call_api("test prompt")
    assert result["output"] == expected


async def test_renders_nunjucks_in_headers(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "POST",
                "headers": {"X-Custom-Header": "{{ prompt | upper }}"},
                "body": {"key": "{{ prompt }}"},
            }
        },
    )
    fake_fetch.add("custom response")
    await provider.call_api("test prompt")
    assert fake_fetch.last[1] == {
        "method": "POST",
        "headers": {"content-type": "application/json", "x-custom-header": "TEST PROMPT"},
        "body": '{"key":"test prompt"}',
    }


async def test_exposes_evaluation_id_in_vars(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json",
                    "X-Evaluation-Id": "{{evaluationId}}",
                },
                "body": {"query": "{{prompt}}", "evaluation_id": "{{evaluationId}}"},
            }
        },
    )
    fake_fetch.add("ok")
    await provider.call_api("test prompt", {"vars": {}, "evaluationId": "eval-123"})
    options = fake_fetch.last[1]
    assert options["headers"]["x-evaluation-id"] == "eval-123"
    assert options["body"] == '{"query":"test prompt","evaluation_id":"eval-123"}'


def test_invalid_config_type():
    with pytest.raises(ValueError, match="expected object, received string"):
        HttpProvider(MOCK_URL, {"config": "this isnt json"})


def test_negative_max_retries_is_rejected():
    with pytest.raises(ValueError, match="Too small: expected number to be >=0"):
        HttpProvider(MOCK_URL, {"config": {"body": "x", "maxRetries": -1}})


def test_requires_body_or_get():
    with pytest.raises(
        AssertionError,
        match=r"Expected HTTP provider http://test.com to have a config containing \{body\}",
    ):
        HttpProvider("http://test.com", {"config": {"method": "POST"}})


def test_id_and_string_representation():
    provider = HttpProvider(MOCK_URL, {"config": {"body": "yo mama"}})
    assert provider.id() == MOCK_URL
    assert str(provider) == f"[HTTP Provider {MOCK_URL}]"


def test_config_url_overrides_provider_path():
    provider = HttpProvider("https", {"config": {"url": "https://x.test/a", "method": "GET"}})
    assert provider.id() == "https://x.test/a"


def test_to_json_matches_node_shape():
    provider = HttpProvider(
        "https://example.test/chat",
        {
            "label": "lbl",
            "config": {
                "method": "POST",
                "unknownKey": 1,
                "tokenEstimation": {},
                "session": {"url": "https://x/s", "responseParser": "data.body.id"},
                "headers": {"A": "b"},
                "body": {"q": "{{prompt}}"},
                "auth": {"type": "bearer", "token": "t", "extra": 1},
                "tls": {"caPath": "x"},
            },
        },
    )
    # JSON.stringify(provider) from promptfoo 0.125.1 (loadApiProviders).
    expected = (
        '{"url":"https://example.test/chat","config":{"body":{"q":"{{prompt}}"},"headers":{"A":"b"},'
        '"method":"POST","session":{"url":"https://x/s","method":"POST","responseParser":'
        '"data.body.id"},"tokenEstimation":{"enabled":false,"multiplier":1.3},"auth":{"token":"t",'
        '"type":"bearer"},"tls":{"caPath":"x","rejectUnauthorized":true}},"transformResponse":{},'
        '"sessionParser":{},"transformRequest":{},"validateStatus":{},"signatureCache":{"ttl":0,'
        '"ttlResolution":1,"ttlAutopurge":false,"updateAgeOnGet":false,"updateAgeOnHas":false,'
        '"allowStale":false,"noDisposeOnSet":false,"noUpdateTTL":false,"maxEntrySize":0,'
        '"noDeleteOnFetchRejection":false,"noDeleteOnStaleGet":false,"allowStaleOnFetchAbort":'
        'false,"allowStaleOnFetchRejection":false,"ignoreFetchAbort":false,"backgroundFetchSize":1},'
        '"authTokenCache":{},"tokenRefreshLocks":{},"fetchedSessions":{},'
        '"sessionEndpointParser":{},"label":"lbl"}'
    )
    assert json_stringify(provider) == expected


def test_create_provider_factory_and_registry():
    from promptphoo.providers.registry import find_factory

    factory = find_factory("https://example.com/x")
    assert factory is not None
    provider = factory("https://example.com/x", {"config": {"method": "GET"}})
    assert isinstance(provider, HttpProvider)
    assert isinstance(create_provider("http://a.test", {"config": {"method": "GET"}}), HttpProvider)


async def test_get_with_query_params(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {"config": {"method": "GET", "queryParams": {"q": "{{ prompt }}", "foo": "bar"}}},
    )
    fake_fetch.add("response data")
    await provider.call_api("test prompt")
    url, options, *_ = fake_fetch.last
    assert url == f"{MOCK_URL}?q=test+prompt&foo=bar"
    assert "body" not in options


async def test_query_params_appended_to_existing_query(fake_fetch):
    provider = HttpProvider(
        "http://example.com/api?existing=param",
        {"config": {"method": "GET", "queryParams": {"additional": "value"}}},
    )
    fake_fetch.add("ok")
    await provider.call_api("test")
    assert fake_fetch.last[0] == "http://example.com/api?existing=param&additional=value"


async def test_query_params_fallback_for_malformed_url(fake_fetch):
    provider = HttpProvider(
        "not-a-valid-url",
        {"config": {"method": "GET", "queryParams": {"param": "value"}}},
    )
    fake_fetch.add("ok")
    await provider.call_api("test")
    assert fake_fetch.last[0] == "not-a-valid-url?param=value"


async def test_respects_max_retries(fake_fetch):
    provider = HttpProvider(MOCK_URL, {"config": {"body": {"a": "b"}, "maxRetries": 2}})
    fake_fetch.add("ok")
    await provider.call_api("x")
    assert fake_fetch.last[5] == 2


async def test_debug_context_busts_cache(fake_fetch):
    provider = HttpProvider(MOCK_URL, {"config": {"method": "POST", "body": {"a": "b"}}})
    fake_fetch.add("ok")
    await provider.call_api("x", {"vars": {}, "debug": True})
    assert fake_fetch.last[4] is True
    await provider.call_api("x", {"vars": {}, "bustCache": False, "debug": True})
    assert fake_fetch.last[4] is False


async def test_traceparent_headers(fake_fetch):
    provider = HttpProvider(MOCK_URL, {"config": {"method": "POST", "body": {"a": "b"}}})
    fake_fetch.add("ok")
    await provider.call_api("x", {"vars": {}, "traceparent": "00-abc-def-01", "tracestate": "k=v"})
    headers = fake_fetch.last[1]["headers"]
    assert headers["traceparent"] == "00-abc-def-01"
    assert headers["tracestate"] == "k=v"


# --- content type / headers ------------------------------------------------------------------


def test_default_headers():
    provider = HttpProvider(MOCK_URL, {"config": {"method": "GET"}})
    assert provider.get_default_headers({}) == {}
    provider = HttpProvider(MOCK_URL, {"config": {"method": "POST", "body": {"a": 1}}})
    assert provider.get_default_headers({"a": 1}) == {"content-type": "application/json"}
    assert provider.get_default_headers("a=1") == {
        "content-type": "application/x-www-form-urlencoded"
    }


def test_validate_content_type_and_body():
    provider = HttpProvider(MOCK_URL, {"config": {"method": "POST", "body": {"a": 1}}})
    provider.validate_content_type_and_body({"content-type": "application/json"}, {"a": 1})
    with pytest.raises(ValueError, match="Content-Type is not application/json"):
        provider.validate_content_type_and_body({"content-type": "text/plain"}, {"a": 1})


async def test_get_headers_combines_and_renders(monkeypatch):
    monkeypatch.setenv("SECRET_HEADER", "from-env")
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "POST",
                "body": {"a": 1},
                "headers": {"X-Custom": "{{ prompt }}", "X-Env": "{{ env.SECRET_HEADER }}"},
            }
        },
    )
    headers = await provider.get_headers({"content-type": "application/json"}, {"prompt": "hi"})
    assert headers == {
        "content-type": "application/json",
        "x-custom": "hi",
        "x-env": "from-env",
    }


async def test_object_body_with_non_json_content_type_errors(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {"config": {"method": "POST", "headers": {"Content-Type": "text/plain"}, "body": {"a": 1}}},
    )
    with pytest.raises(ValueError, match="Content-Type is not application/json"):
        await provider.call_api("x")


async def test_string_body_without_content_type_is_form_encoded(fake_fetch):
    provider = HttpProvider(MOCK_URL, {"config": {"method": "POST", "body": "Hello {{ prompt }}"}})
    fake_fetch.add("ok")
    await provider.call_api("world")
    options = fake_fetch.last[1]
    assert options["headers"] == {"content-type": "application/x-www-form-urlencoded"}
    assert options["body"] == "Hello world"


async def test_stringified_json_body_with_json_content_type(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": '{"key": "{{ prompt }}"}',
            }
        },
    )
    fake_fetch.add("ok")
    await provider.call_api("test prompt")
    assert fake_fetch.last[1]["body"] == '{"key":"test prompt"}'


async def test_nested_object_vars_render_with_dump(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {"names": "{{ names | dump }}", "simple": "{{ simple }}"},
            }
        },
    )
    fake_fetch.add("ok")
    names = [{"firstName": "Jane", "lastName": "Smith"}, {"firstName": "John", "lastName": "Doe"}]
    await provider.call_api("x", _ctx(names=names, simple="test-value"))
    assert json.loads(fake_fetch.last[1]["body"]) == {"names": names, "simple": "test-value"}


# --- processJsonBody / processTextBody / determineRequestBody -------------------------------


def test_process_json_body_simple_and_nested():
    assert process_json_body({"key": "value", "prompt": "{{ prompt }}"}, {"prompt": "t"}) == {
        "key": "value",
        "prompt": "t",
    }
    body = {
        "key": "{{var1}}",
        "nested": {"key2": "{{var2}}", "items": ["{{var3}}", {"n": "{{var4}}"}]},
    }
    assert process_json_body(body, {"var1": "1", "var2": "2", "var3": "3", "var4": "4"}) == {
        "key": "1",
        "nested": {"key2": "2", "items": ["3", {"n": "4"}]},
    }


def test_process_json_body_parses_json_strings():
    body = {"key": "{{var1}}", "jsonString": '{"parsed":{{var2}}}'}
    assert process_json_body(body, {"var1": "value1", "var2": 123}) == {
        "key": "value1",
        "jsonString": {"parsed": 123},
    }


def test_process_json_body_keeps_primitive_strings_in_objects():
    body = {"num": "{{ n }}", "flag": "{{ b }}", "list": "{{ l }}"}
    assert process_json_body(body, {"n": "42", "b": "true", "l": "[1,2]"}) == {
        "num": "42",
        "flag": "true",
        "list": [1, 2],
    }


def test_process_json_body_raw_json_strings():
    body = '{\n  "input": "Text with control char: \u0001",\n  "role": "user"\n}'
    assert process_json_body(body, {"prompt": "t"}) == body
    bad = '{\n  "input": "{{prompt}}",\n  "role": "user",\n}'
    assert process_json_body(bad, {"prompt": "test prompt"}) == (
        '{\n  "input": "test prompt",\n  "role": "user",\n}'
    )
    assert process_json_body('{"input": "{{prompt}}", "role": "user"}', {"prompt": "p"}) == {
        "input": "p",
        "role": "user",
    }
    assert process_json_body('"hello"', {}) == "hello"
    assert process_json_body("42", {"x": 1}) == 42
    assert process_json_body("true", {"x": 1}) is True


def test_process_json_body_auto_escapes_template_values():
    body = '{\n  "message": "{{prompt}}"\n}'
    prompt = "Multi-line prompt\nwith actual newlines\nand more text"
    assert process_json_body(body, {"prompt": prompt}) == {"message": prompt}
    body = '{\n  "message": "{{prompt}}",\n  "role": "user"\n}'
    prompt = 'Text with "quotes" and \ttabs and \nmore stuff'
    assert process_json_body(body, {"prompt": prompt}) == {"message": prompt, "role": "user"}
    body = '{\n  "message": "{{prompt}}"\n  missing_comma: true\n}'
    assert process_json_body(body, {"prompt": "Some text with\nnewlines"}) == (
        '{\n  "message": "Some text with\nnewlines"\n  missing_comma: true\n}'
    )


def test_escape_json_variables():
    assert escape_json_variables({"a": "x\ny", "b": 'q"', "c": "t\tz", "d": "\\", "e": 1}) == {
        "a": "x\\ny",
        "b": 'q\\"',
        "c": "t\\tz",
        "d": "\\\\",
        "e": 1,
    }
    assert escape_json_variables({"u": "\u0001", "empty": ""}) == {"u": "\\u0001", "empty": ""}
    rendered = render_raw_request_with_nunjucks(
        '{"text": "{{ prompt }}"}', escape_json_variables({"prompt": 'a\n"b"'})
    )
    assert json.loads(rendered) == {"text": 'a\n"b"'}


def test_process_text_body():
    assert process_text_body("Hello {{ name }}", {"name": "World"}) == "Hello World"
    assert process_text_body(None, {}) is None
    assert process_text_body("Hello {{ name", {"name": "x"}) == "Hello {{ name"
    with pytest.raises(AssertionError, match="Expected body to be a string"):
        process_text_body({"a": 1}, {})


def test_determine_request_body():
    assert determine_request_body(True, {"promptField": "v"}, {"configField": "c"}, {}) == {
        "configField": "c",
        "promptField": "v",
    }
    assert determine_request_body(True, "tp", {"message": "{{ prompt }}"}, {}) == {"message": "tp"}
    assert determine_request_body(False, "tp", "Message: {{ prompt }}", {}) == "Message: tp"
    assert determine_request_body(True, {"message": "m"}, None, {}) == {"message": "m"}
    assert determine_request_body(True, "tp", ["static", "{{ prompt }}"], {}) == ["static", "tp"]
    assert determine_request_body(True, "tp", '{"message":"{{prompt}}"}', {}) == {"message": "tp"}
    assert determine_request_body(True, "tp", '["{{prompt}}", "static"]', {}) == ["tp", "static"]
    assert determine_request_body(True, "tp", "Message: {{prompt}}", {}) == "Message: tp"
    assert determine_request_body(False, "tp", '{"message":"{{prompt}}"}', {}) == (
        '{"message":"tp"}'
    )
    assert determine_request_body(True, {"promptField": "p"}, '{"configField":"value"}', {}) == {
        "configField": "value",
        "promptField": "p",
    }


# --- raw requests --------------------------------------------------------------------------


async def test_raw_get_request(fake_fetch):
    raw = "GET /api/data HTTP/1.1\nHost: example.com\nUser-Agent: TestAgent/1.0"
    provider = HttpProvider("http", {"config": {"request": raw}})
    fake_fetch.add(json.dumps({"result": "success"}))
    result = await provider.call_api("test prompt")
    url, options, *_ = fake_fetch.last
    assert url == "http://example.com/api/data"
    assert options == {
        "method": "GET",
        "headers": {"host": "example.com", "user-agent": "TestAgent/1.0"},
    }
    assert result["output"] == {"result": "success"}


async def test_raw_post_request_with_body(fake_fetch):
    raw = (
        "POST /api/submit HTTP/1.1\nHost: example.com\nContent-Type: application/json\n\n"
        '{"data": "{{prompt}}"}'
    )
    provider = HttpProvider("https", {"config": {"request": raw}})
    fake_fetch.add(json.dumps({"result": "received"}))
    await provider.call_api("test data")
    url, options, *rest = fake_fetch.last
    assert url == "https://example.com/api/submit"
    assert options == {
        "method": "POST",
        "headers": {"host": "example.com", "content-type": "application/json"},
        "body": '{"data": "test data"}',
    }
    assert rest[1:] == ["text", False, None]


async def test_raw_request_path_variables(fake_fetch):
    raw = "GET /api/users/{{userId}}/profile HTTP/1.1\nHost: example.com\nAccept: application/json"
    provider = HttpProvider("https", {"config": {"request": raw}})
    fake_fetch.add("{}")
    await provider.call_api("x", _ctx(userId="12345"))
    assert fake_fetch.last[0] == "https://example.com/api/users/12345/profile"


async def test_raw_request_from_file(fake_fetch, tmp_path):
    (tmp_path / "request.txt").write_text("GET /api/data HTTP/1.1\nHost: example.com\n")
    with cli_state.with_base_path(str(tmp_path)):
        provider = HttpProvider("https", {"config": {"request": "file://request.txt"}})
    fake_fetch.add("{}")
    await provider.call_api("x")
    assert fake_fetch.last[0] == "https://example.com/api/data"


async def test_invalid_raw_request(fake_fetch):
    provider = HttpProvider("http", {"config": {"request": "yo mama"}})
    with pytest.raises(ValueError, match="not valid"):
        await provider.call_api("x")


async def test_raw_request_removes_content_length(fake_fetch):
    raw = (
        "POST /api/submit HTTP/1.1\nHost: example.com\nContent-Type: application/json\n"
        'Content-Length: 1000\n\n{"data": "{{prompt}}"}'
    )
    provider = HttpProvider("https", {"config": {"request": raw}})
    fake_fetch.add("{}")
    await provider.call_api("test data")
    assert "content-length" not in fake_fetch.last[1]["headers"]


@pytest.mark.parametrize(("use_https", "scheme"), [(True, "https"), (False, "http")])
async def test_raw_request_use_https(fake_fetch, use_https, scheme):
    raw = "GET /api/data HTTP/1.1\nHost: example.com"
    provider = HttpProvider("http", {"config": {"request": raw, "useHttps": use_https}})
    fake_fetch.add("{}")
    await provider.call_api("x")
    assert fake_fetch.last[0] == f"{scheme}://example.com/api/data"


async def test_raw_get_with_query_params(fake_fetch):
    raw = "GET /api/data?q={{prompt}}&limit=10 HTTP/1.1\nHost: example.com"
    provider = HttpProvider("https", {"config": {"request": raw}})
    fake_fetch.add("{}")
    await provider.call_api("hello world")
    assert fake_fetch.last[0] == "https://example.com/api/data?q=hello%20world&limit=10"


async def test_raw_multipart_request(fake_fetch):
    raw = (
        "POST /upload HTTP/1.1\nHost: example.com\n"
        "Content-Type: multipart/form-data; boundary=----Boundary\n\n"
        '------Boundary\nContent-Disposition: form-data; name="text"\n\n{{prompt}}\n'
        "------Boundary--"
    )
    provider = HttpProvider("https", {"config": {"request": raw}})
    fake_fetch.add("{}")
    await provider.call_api("hello")
    body = fake_fetch.last[1]["body"]
    assert body == (
        '------Boundary\r\nContent-Disposition: form-data; name="text"\r\n\r\nhello\r\n'
        "------Boundary--"
    )


async def test_raw_form_urlencoded_request(fake_fetch):
    raw = (
        "POST /submit HTTP/1.1\nHost: example.com\n"
        "Content-Type: application/x-www-form-urlencoded\n\nfield1={{prompt}}&field2=value2"
    )
    provider = HttpProvider("https", {"config": {"request": raw}})
    fake_fetch.add("{}")
    await provider.call_api("test")
    assert fake_fetch.last[1]["body"] == "field1=test&field2=value2"


async def test_raw_request_keeps_literal_nunjucks_in_cookie(fake_fetch):
    raw = (
        "POST /api HTTP/1.1\nHost: example.com\nCookie: data={%22a%22:1}\n"
        'Content-Type: application/json\n\n{"q": "{{ prompt }}"}'
    )
    provider = HttpProvider("https", {"config": {"request": raw}})
    fake_fetch.add("{}")
    await provider.call_api("hi")
    options = fake_fetch.last[1]
    assert options["headers"]["cookie"] == "data={%22a%22:1}"
    assert options["body"] == '{"q": "hi"}'


async def test_raw_request_dotted_vars_and_crlf(fake_fetch):
    raw = "GET /u/{{ user.id }} HTTP/1.1\r\nHost: example.com\nX-Name: {{ user.name }}\r\n"
    provider = HttpProvider("https", {"config": {"request": raw}})
    fake_fetch.add("{}")
    await provider.call_api("x", _ctx(user={"id": 7, "name": "Ann"}))
    url, options, *_ = fake_fetch.last
    assert url == "https://example.com/u/7"
    assert options["headers"]["x-name"] == "Ann"


def test_url_encode_raw_request_path():
    assert url_encode_raw_request_path("GET /api/data HTTP/1.1") == "GET /api/data HTTP/1.1"
    assert url_encode_raw_request_path("GET /api/data?key=value HTTP/1.1") == (
        "GET /api/data?key=value HTTP/1.1"
    )
    assert url_encode_raw_request_path("GET /api/data?query=hello world HTTP/1.1") == (
        "GET /api/data?query=hello%20world HTTP/1.1"
    )
    assert url_encode_raw_request_path("GET /api/data?query=already%20encoded HTTP/1.1") == (
        "GET /api/data?query=already%20encoded HTTP/1.1"
    )
    assert url_encode_raw_request_path("GET /path with spaces/resource HTTP/1.1") == (
        "GET /path%20with%20spaces/resource HTTP/1.1"
    )
    assert (
        url_encode_raw_request_path("GET /path/with [brackets]?param=value&special=a+b+c HTTP/1.1")
        == "GET /path/with%20[brackets]?param=value&special=a+b+c HTTP/1.1"
    )
    raw = (
        "GET /summarized?topic=hello world&start=01/01/2025 HTTP/2\nHost: foo.bar.com\n"
        "Accept: application/json"
    )
    assert url_encode_raw_request_path(raw) == raw.replace("hello world", "hello%20world")


@pytest.mark.parametrize("eol", ["\n", "\r\n"])
def test_url_encode_keeps_dollar_patterns(eol):
    rest = f"{eol}X-Literal: $$ $& $` $'{eol}{eol}body $$ $& $` $'"
    raw = "POST /api/data?query=turn $$ into $& or $` or $' HTTP/1.1" + rest
    assert url_encode_raw_request_path(raw) == (
        "POST /api/data?query=turn%20$$%20into%20$&%20or%20$`%20or%20$%27 HTTP/1.1" + rest
    )


@pytest.mark.parametrize(
    "raw",
    ["GET HTTP/1.1", "This is not a valid HTTP request line", "GET /api/data?query=test"],
)
def test_url_encode_rejects_malformed_lines(raw):
    with pytest.raises(ValueError, match="not valid"):
        url_encode_raw_request_path(raw)


def test_url_encode_does_not_leak_secrets(caplog):
    caplog.set_level(logging.DEBUG)
    raw = "GET /v1/chat?api_key=sk-realsecret1234567890&note=foo FTP/1.1"
    with pytest.raises(ValueError, match="protocol is not valid") as info:
        url_encode_raw_request_path(raw)
    assert "sk-realsecret1234567890" not in str(info.value)
    assert "sk-realsecret1234567890" not in caplog.text
    caplog.clear()
    url_encode_raw_request_path("GET /api/data?api_key=secret%zz&query=hello world HTTP/1.1")
    assert "api_key=%5BREDACTED%5D" in caplog.text
    assert "secret%zz" not in caplog.text


def test_parse_raw_request_errors():
    with pytest.raises(ValueError, match="HTTP request method FOO is not valid"):
        parse_raw_request("FOO /x HTTP/1.1\nHost: a")
    with pytest.raises(
        ValueError, match="Error parsing raw HTTP request: HttpZError: Incorrect header row"
    ):
        parse_raw_request("GET /x HTTP/1.1\nHost: a\nnot a header")


def test_parse_raw_request_shape():
    parsed = parse_raw_request("POST /x?a=1 HTTP/2\nHost: h\nContent-Type: text/plain\n\n  hi  ")
    assert parsed == {
        "method": "POST",
        "url": "/x?a=1",
        "httpVersion": "HTTP/2",
        "headers": {"host": "h", "content-type": "text/plain"},
        "body": {"contentType": "text/plain", "text": "hi"},
    }


def test_extract_body_from_raw_request():
    assert extract_body_from_raw_request('POST /a HTTP/1.1\nHost: x\n\n{"a": 1}') == '{"a": 1}'
    assert extract_body_from_raw_request("GET /a HTTP/1.1\nHost: x") is None
    assert extract_body_from_raw_request("POST /a HTTP/1.1\nHost: x\n\n   \n") is None
    assert extract_body_from_raw_request("POST /a HTTP/1.1\nHost: x\n\nl1\n\nl2") == "l1\r\n\r\nl2"
    assert extract_body_from_raw_request("POST /a HTTP/1.1\r\nHost: x\n\n  b  \n") == "b"


# --- transforms --------------------------------------------------------------------------


async def test_create_transform_response_variants():
    default = create_transform_response(None)
    assert await default({"key": "value"}, "") == {"output": {"key": "value"}}
    assert await default(None, "raw text") == {"output": "raw text"}

    fn = create_transform_response(lambda data: {"output": data["x"], "tokenUsage": {"total": 1}})
    assert await fn({"x": "y"}, "") == {"output": "y", "tokenUsage": {"total": 1}}
    assert await create_transform_response(lambda d: "success")({}, "") == {"output": "success"}
    assert await create_transform_response("json.result")({"result": "parsed"}, "") == {
        "output": "parsed"
    }
    assert await create_transform_response("42")({}, "") == {"output": 42}
    ctx_fn = create_transform_response(lambda d, t, c: {"output": d["v"], "status": c["status"]})
    assert await ctx_fn({"v": "value"}, "", {"status": 200}) == {"output": "value", "status": 200}


async def test_create_transform_response_errors():
    with pytest.raises(ValueError, match="should be pre-loaded"):
        create_transform_response("file://custom-parser.js")
    with pytest.raises(ValueError, match="Unsupported response transform type: number"):
        create_transform_response(123)

    def boom(data):
        raise ValueError("Parser error")

    with pytest.raises(ValueError, match="Parser error"):
        await create_transform_response(boom)({"data": "test"}, "test")
    with pytest.raises(ValueError, match="Failed to transform response"):
        await create_transform_response("json.a.b(")({}, "")


@pytest.mark.parametrize(
    "parser",
    [
        "(json, text) => json.output",
        "function(json, text) { return json.output; }",
        "(json) => {\n  return json.output;\n}",
    ],
)
async def test_js_function_transforms_are_rejected(parser):
    with pytest.raises(ValueError, match="not supported by promptphoo") as info:
        await create_transform_response(parser)({"output": "x"}, "")
    assert str(info.value).startswith("Failed to transform response: Error: ")


async def test_create_transform_request_variants():
    assert await create_transform_request(None)("test prompt", {}) == "test prompt"
    assert await create_transform_request("({ text: prompt.toUpperCase() })")("hello", {}) == {
        "text": "HELLO"
    }
    assert await create_transform_request("prompt.toUpperCase()")("hello", {}) == "HELLO"
    assert await create_transform_request("({ text: prompt, extra: vars.extra })")(
        "hello", {"extra": "data"}
    ) == {"text": "hello", "extra": "data"}
    assert await create_transform_request(lambda p: {"transformed": p.upper()})("hello", {}) == {
        "transformed": "HELLO"
    }

    async def async_fn(prompt, vars, context):
        return {"transformed": prompt.upper(), "ctx": context}

    assert await create_transform_request(async_fn)("hello", {}, {"c": 1}) == {
        "transformed": "HELLO",
        "ctx": {"c": 1},
    }


async def test_create_transform_request_errors():
    def boom(prompt):
        raise ValueError("Transform function error")

    with pytest.raises(ValueError, match="Error in request transform function: Transform function"):
        await create_transform_request(boom)("x", {})
    with pytest.raises(ValueError, match="should be pre-loaded"):
        create_transform_request("file://error-transform.js")
    with pytest.raises(ValueError, match="Failed to transform request:"):
        await create_transform_request("prompt.foo(")("x", {})
    with pytest.raises(ValueError, match="Failed to transform request:"):
        await create_transform_request("(p) => p")("x", {})
    with pytest.raises(ValueError, match="Unsupported request transform type: number"):
        create_transform_request(123)


async def test_string_request_transform_in_provider(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {"static": "x"},
                "transformRequest": "({ text: prompt })",
            }
        },
    )
    fake_fetch.add("ok")
    await provider.call_api("hello")
    assert json.loads(fake_fetch.last[1]["body"]) == {"static": "x", "text": "hello"}


async def test_python_file_transforms(fake_fetch, tmp_path):
    (tmp_path / "hooks.py").write_text(
        "def transform_response(json, text, context):\n"
        "    return {'output': json['answer'], 'metadata': {'status': context['response']['status']}}\n"  # noqa: E501
        "def shout(prompt, vars, context):\n"
        "    return prompt.upper() + vars['suffix']\n"
        "def session_parser(data):\n"
        "    return data['headers'].get('x-sid')\n"
        "def only_2xx(status):\n"
        "    return 200 <= status < 300\n"
    )
    with cli_state.with_base_path(str(tmp_path)):
        provider = HttpProvider(
            MOCK_URL,
            {
                "config": {
                    "method": "POST",
                    "body": "{{ prompt }}",
                    "transformResponse": "file://hooks.py",
                    "transformRequest": "file://hooks.py:shout",
                    "sessionParser": "file://hooks.py",
                    "validateStatus": "file://hooks.py:only_2xx",
                }
            },
        )
        fake_fetch.add(json.dumps({"answer": 42}), headers={"x-sid": "s-1"})
        result = await provider.call_api("hi", _ctx(suffix="!"))
        assert fake_fetch.last[1]["body"] == "HI!"
        assert result["output"] == 42
        assert result["metadata"] == {"status": 200}
        assert result["sessionId"] == "s-1"

        fake_fetch.responses = []
        fake_fetch.add("nope", status=503, status_text="Service Unavailable")
        with pytest.raises(ValueError, match="HTTP call failed with status 503"):
            await provider.call_api("hi", _ctx(suffix="!"))


async def test_malformed_python_file_transform(fake_fetch, tmp_path):
    (tmp_path / "hooks.py").write_text("x = 1\n")
    with cli_state.with_base_path(str(tmp_path)):
        provider = HttpProvider(
            MOCK_URL,
            {"config": {"method": "POST", "body": "x", "transformResponse": "file://hooks.py:x"}},
        )
        fake_fetch.add("ok")
        with pytest.raises(ValueError, match="Transform module malformed: hooks.py"):
            await provider.call_api("x")


async def test_javascript_file_transform_is_rejected(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {"config": {"method": "POST", "body": "x", "transformResponse": "file://parse.js"}},
    )
    fake_fetch.add("ok")
    with pytest.raises(ValueError, match="JavaScript transform files are not supported"):
        await provider.call_api("x")


async def test_deprecated_response_parser(fake_fetch):
    provider = HttpProvider(
        MOCK_URL, {"config": {"body": {"a": 1}, "responseParser": "json.legacy"}}
    )
    fake_fetch.add(json.dumps({"legacy": "old", "new": "new"}))
    assert (await provider.call_api("x"))["output"] == "old"
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "body": {"a": 1},
                "responseParser": "json.legacy",
                "transformResponse": "json.new",
            }
        },
    )
    assert (await provider.call_api("x"))["output"] == "new"


async def test_transform_response_error_object_is_preserved(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {"config": {"body": {"a": 1}, "transformResponse": "({ error: json.message })"}},
    )
    fake_fetch.add(json.dumps({"message": "denied"}))
    result = await provider.call_api("x")
    assert result["error"] == "denied"
    assert "output" not in result


async def test_string_transform_error_message(fake_fetch):
    provider = HttpProvider(
        MOCK_URL, {"config": {"body": {"a": 1}, "transformResponse": "json.items.map(1)"}}
    )
    fake_fetch.add(json.dumps({"items": 3}))
    with pytest.raises(
        ValueError, match=r"^Failed to transform response: Error: Transform failed:"
    ):
        await provider.call_api("x")


async def test_dict_keys_win_over_dict_methods(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {"config": {"body": {"a": 1}, "transformResponse": "json.items[0] + json.keys + json.get"}},
    )
    fake_fetch.add(json.dumps({"items": [1], "keys": 2, "get": 3}))
    assert (await provider.call_api("x"))["output"] == 6


# --- validateStatus ------------------------------------------------------------------------


async def test_validate_status_variants():
    assert await create_validate_status(None)(500) is True
    assert await create_validate_status("status >= 200 && status < 300")(204) is True
    assert await create_validate_status("status >= 200 && status < 300")(404) is False
    assert await create_validate_status("200 <= status < 400")(302) is True
    assert await create_validate_status("status in [200, 404]")(404) is True
    assert await create_validate_status(lambda s: s < 500)(499) is True
    in_list = create_validate_status([200, "400-404"])
    assert [await in_list(s) for s in (200, 201, 401, 405)] == [True, False, True, False]
    ranges = create_validate_status("200-299, 404")
    assert [await ranges(s) for s in (250, 404, 500)] == [True, True, False]


def test_validate_status_errors():
    with pytest.raises(ValueError, match="Invalid status validator expression"):
        create_validate_status("status >===> 500")
    with pytest.raises(ValueError, match="Invalid status validator expression: JavaScript"):
        create_validate_status("(status) => status < 500")
    with pytest.raises(ValueError, match="Status validator malformed: missing.py"):
        create_validate_status("file://missing.py")
    with pytest.raises(ValueError, match="Unsupported status validator type: number"):
        create_validate_status(123)


async def test_validate_status_failure_message(fake_fetch):
    provider = HttpProvider(
        MOCK_URL, {"config": {"body": {"a": 1}, "validateStatus": "status < 500"}}
    )
    fake_fetch.add("Error occurred", status=500, status_text="Internal Server Error")
    with pytest.raises(
        ValueError, match="HTTP call failed with status 500 Internal Server Error: Error occurred"
    ):
        await provider.call_api("x")


async def test_raw_request_validate_status_failure(fake_fetch):
    provider = HttpProvider(
        "http",
        {
            "config": {
                "request": "GET /api HTTP/1.1\nHost: example.com",
                "validateStatus": "status === 200",
            }
        },
    )
    fake_fetch.add("Not found", status=404, status_text="Not Found")
    with pytest.raises(ValueError, match="HTTP call failed with status 404 Not Found: Not found"):
        await provider.call_api("x")


async def test_accepts_all_status_codes_by_default(fake_fetch):
    provider = HttpProvider(MOCK_URL, {"config": {"body": {"a": 1}}})
    fake_fetch.add('{"error":"x"}', status=500, status_text="Internal Server Error")
    result = await provider.call_api("x")
    assert result["output"] == {"error": "x"}
    assert result["metadata"]["http"]["status"] == 500


# --- sessions ------------------------------------------------------------------------------


async def test_session_parser_variants():
    assert await create_session_parser(None)({}) == ""
    assert await create_session_parser(lambda d: d["headers"]["sid"])({"headers": {"sid": 1}}) == 1
    header = create_session_parser("data.headers['x-session-id']")
    assert await header({"headers": {"x-session-id": "abc"}}) == "abc"
    body = create_session_parser("data.body.session.id")
    assert await body({"body": {"session": {"id": "nested"}}}) == "nested"
    with pytest.raises(ValueError, match="Unsupported response transform type: number"):
        create_session_parser(123)


async def test_session_id_from_headers(fake_fetch):
    provider = HttpProvider(
        MOCK_URL, {"config": {"body": {"a": 1}, "sessionParser": "data.headers['x-session-id']"}}
    )
    fake_fetch.add("{}", headers={"x-session-id": "test-session"})
    assert (await provider.call_api("x"))["sessionId"] == "test-session"


async def test_session_parser_errors_are_raised(fake_fetch):
    def bad(data):
        raise ValueError("Session parsing failed")

    provider = HttpProvider(MOCK_URL, {"config": {"body": {"a": 1}, "sessionParser": bad}})
    fake_fetch.add("{}")
    with pytest.raises(ValueError, match="Session parsing failed"):
        await provider.call_api("x")


def _session_config(**session: Any) -> dict[str, Any]:
    return {
        "config": {
            "method": "POST",
            "body": {"session": "{{ sessionId }}", "q": "{{ prompt }}"},
            "session": {
                "url": "http://example.com/session",
                "responseParser": "data.body.id",
                **session,
            },
        }
    }


async def test_session_endpoint_first_call_and_reuse(fake_fetch):
    provider = HttpProvider(MOCK_URL, _session_config())
    fake_fetch.add(json.dumps({"id": "sess-1"}))
    fake_fetch.add(json.dumps({"ok": True}))
    fake_fetch.add(json.dumps({"ok": True}))
    first = await provider.call_api("hi", {"vars": {}})
    session_url, session_options, _, fmt, bust, _ = fake_fetch.calls[0]
    assert session_url == "http://example.com/session"
    assert session_options == {"method": "POST", "headers": {}}
    assert (fmt, bust) == ("text", True)
    assert json.loads(fake_fetch.calls[1][1]["body"]) == {"session": "sess-1", "q": "hi"}
    assert first["sessionId"] == "sess-1"

    # Hydra pattern: the strategy passes back a session we created, so it is reused.
    await provider.call_api("again", {"vars": {"sessionId": "sess-1"}})
    assert len(fake_fetch.calls) == 3
    assert json.loads(fake_fetch.calls[2][1]["body"])["session"] == "sess-1"


async def test_session_endpoint_fresh_for_unknown_session(fake_fetch):
    provider = HttpProvider(MOCK_URL, _session_config())
    fake_fetch.add(json.dumps({"id": "server-session"}))
    fake_fetch.add("{}")
    result = await provider.call_api("hi", {"vars": {"sessionId": "client-uuid"}})
    assert fake_fetch.calls[0][0] == "http://example.com/session"
    assert result["sessionId"] == "server-session"


async def test_session_endpoint_errors(fake_fetch):
    provider = HttpProvider(MOCK_URL, _session_config())
    fake_fetch.add("Unauthorized", status=401, status_text="Unauthorized")
    with pytest.raises(
        ValueError,
        match="Session endpoint request failed with status 401 Unauthorized: Unauthorized",
    ):
        await provider.call_api("hi", {"vars": {}})
    provider = HttpProvider(MOCK_URL, _session_config())
    fake_fetch.responses = []
    fake_fetch.add(json.dumps({"other": "x"}))
    with pytest.raises(
        ValueError, match='Session endpoint did not return a session ID. Response: {"other":"x"}'
    ):
        await provider.call_api("hi", {"vars": {}})


async def test_session_endpoint_get_and_templating(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        _session_config(
            url="http://example.com/session/{{ user }}",
            method="GET",
            headers={"X-Key": "{{ key }}"},
            body={"ignored": True},
            responseParser="data.headers['x-session']",
        ),
    )
    fake_fetch.add("", headers={"x-session": "hdr-session"})
    fake_fetch.add("{}")
    await provider.call_api("hi", {"vars": {"user": "u1", "key": "k1"}})
    url, options, *_ = fake_fetch.calls[0]
    assert url == "http://example.com/session/u1"
    assert options == {"method": "GET", "headers": {"x-key": "k1"}}


async def test_session_endpoint_object_body(fake_fetch):
    provider = HttpProvider(MOCK_URL, _session_config(body={"user": "{{ user }}"}))
    fake_fetch.add(json.dumps({"id": "s"}))
    fake_fetch.add("{}")
    await provider.call_api("hi", {"vars": {"user": "bob"}})
    options = fake_fetch.calls[0][1]
    assert options["body"] == '{"user":"bob"}'
    assert options["headers"] == {"content-type": "application/json"}


# --- token estimation ----------------------------------------------------------------------


def test_estimate_token_count():
    assert estimate_token_count("Hello world this is a test", 1.0) == 6
    assert estimate_token_count("Hello world", 1.5) == 3
    assert estimate_token_count("", 1.3) == 0
    assert estimate_token_count("hello    world   test", 1.0) == 3
    assert estimate_token_count("one two three") == 4


async def test_token_estimation(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "body": {"a": "{{ prompt }}"},
                "tokenEstimation": {"enabled": True, "multiplier": 2},
            }
        },
    )
    fake_fetch.add(json.dumps({"x": 1}))
    result = await provider.call_api("one two")
    # The completion falls back to the raw text when the output is an object.
    assert result["tokenUsage"] == {"prompt": 4, "completion": 4, "total": 8, "numRequests": 1}

    disabled = HttpProvider(MOCK_URL, {"config": {"body": {"a": 1}}})
    result = await disabled.call_api("one two")
    assert result["tokenUsage"]["numRequests"] == 1
    assert result["tokenUsage"]["total"] == 0


async def test_token_estimation_does_not_override_transform_usage(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "body": {"a": 1},
                "tokenEstimation": {"enabled": True},
                "transformResponse": "({ output: json.t, tokenUsage: { total: 99 } })",
            }
        },
    )
    fake_fetch.add(json.dumps({"t": "a b c"}))
    assert (await provider.call_api("x y"))["tokenUsage"] == {"total": 99}


async def test_token_estimation_enabled_in_redteam_mode(fake_fetch):
    with cli_state.with_config({"redteam": {}}):
        provider = HttpProvider(MOCK_URL, {"config": {"body": {"a": 1}}})
    assert provider.config["tokenEstimation"] == {"enabled": True, "multiplier": 1.3}


async def test_token_estimation_raw_mode(fake_fetch):
    provider = HttpProvider(
        "http",
        {
            "config": {
                "request": "POST /a HTTP/1.1\nHost: x.test\n\n{{prompt}}",
                "tokenEstimation": {"enabled": True, "multiplier": 1},
                "transformResponse": "text",
            }
        },
    )
    fake_fetch.add("three word answer")
    result = await provider.call_api("two words")
    assert result["tokenUsage"] == {"prompt": 2, "completion": 3, "total": 5, "numRequests": 1}


# --- tools ---------------------------------------------------------------------------------


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]  # fmt: skip  # noqa: E501


async def test_tools_template_variables(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "POST",
                "body": {"tools": "{{ tools }}", "tool_choice": "{{ tool_choice }}"},
                "tools": TOOLS,
                "tool_choice": "required",
            }
        },
    )
    fake_fetch.add("{}")
    await provider.call_api("x")
    assert json.loads(fake_fetch.last[1]["body"]) == {"tools": TOOLS, "tool_choice": "required"}


async def test_tools_transformed_to_anthropic(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "POST",
                "body": {"tools": "{{ tools }}", "tool_choice": "{{ tool_choice }}"},
                "tools": TOOLS,
                "tool_choice": "required",
                "transformToolsFormat": "anthropic",
            }
        },
    )
    fake_fetch.add("{}")
    await provider.call_api("x")
    assert json.loads(fake_fetch.last[1]["body"]) == {
        "tools": [
            {
                "name": "get_weather",
                "description": "Weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        "tool_choice": {"type": "any"},
    }


async def test_prompt_config_tools_take_precedence(fake_fetch, caplog):
    provider = HttpProvider(
        MOCK_URL,
        {"config": {"method": "POST", "body": {"t": "{{ tools }}"}, "tools": [{"type": "x"}]}},
    )
    fake_fetch.add("{}")
    await provider.call_api("x", {"vars": {}, "prompt": {"raw": "x", "config": {"tools": TOOLS}}})
    assert json.loads(fake_fetch.last[1]["body"]) == {"t": TOOLS}

    warn = HttpProvider(
        MOCK_URL,
        {"config": {"method": "POST", "body": {"a": 1}, "tool_choice": "auto", "tools": []}},
    )
    await warn.call_api("x")
    assert "tool_choice is set but tools is empty or undefined" in caplog.text


# --- auth ----------------------------------------------------------------------------------


async def test_bearer_auth(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "POST",
                "body": {"a": 1},
                "auth": {"type": "bearer", "token": "{{ t }}"},
            }
        },
    )
    fake_fetch.add("{}")
    await provider.call_api("x", _ctx(t="my-token"))
    assert fake_fetch.last[1]["headers"]["authorization"] == "Bearer my-token"


async def test_bearer_auth_raw_mode(fake_fetch):
    provider = HttpProvider(
        "http",
        {
            "config": {
                "request": "GET /a HTTP/1.1\nHost: example.com",
                "auth": {"type": "bearer", "token": "tok"},
            }
        },
    )
    fake_fetch.add("{}")
    await provider.call_api("x")
    assert fake_fetch.last[1]["headers"]["authorization"] == "Bearer tok"


async def test_basic_auth(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {"config": {"method": "GET", "auth": {"type": "basic", "username": "u", "password": "p"}}},
    )
    fake_fetch.add("{}")
    await provider.call_api("x")
    assert (
        fake_fetch.last[1]["headers"]["authorization"]
        == "Basic " + base64.b64encode(b"u:p").decode()
    )


async def test_api_key_header_and_query(fake_fetch):
    header = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "POST",
                "body": {"a": 1},
                "auth": {
                    "type": "api_key",
                    "keyName": "X-API-Key",
                    "value": "k",
                    "placement": "header",
                },
            }
        },
    )
    fake_fetch.add("{}")
    await header.call_api("x")
    assert fake_fetch.last[1]["headers"]["x-api-key"] == "k"

    query = HttpProvider(
        "http://example.com/api?existing=1#frag",
        {
            "config": {
                "method": "GET",
                "queryParams": {"q": "1"},
                "auth": {
                    "type": "api_key",
                    "keyName": "api_key",
                    "value": "a b&c",
                    "placement": "query",
                },
            }
        },
    )
    await query.call_api("x")
    assert fake_fetch.last[0] == "http://example.com/api?existing=1&q=1&api_key=a+b%26c#frag"


async def test_api_key_query_raw_mode(fake_fetch):
    provider = HttpProvider(
        "http",
        {
            "config": {
                "request": "GET /api?x=1 HTTP/1.1\nHost: example.com",
                "auth": {"type": "api_key", "keyName": "key", "value": "v", "placement": "query"},
            }
        },
    )
    fake_fetch.add("{}")
    await provider.call_api("x")
    assert fake_fetch.last[0] == "http://example.com/api?x=1&key=v"


def _oauth_config(**auth: Any) -> dict[str, Any]:
    return {
        "config": {
            "method": "POST",
            "headers": {"Authorization": "Bearer {{ token }}"},
            "body": {"q": "{{ prompt }}"},
            "auth": {
                "type": "oauth",
                "grantType": "client_credentials",
                "tokenUrl": "https://auth.test/token",
                "clientId": "cid",
                "clientSecret": "secret",
                "scopes": ["read", "write"],
                **auth,
            },
        }
    }


async def test_oauth_client_credentials_with_caching():
    with respx.mock(assert_all_called=True) as mock:
        token_route = mock.post("https://auth.test/token").mock(
            return_value=httpx.Response(200, json={"access_token": "tok-1", "expires_in": 3600})
        )
        api_route = mock.post(MOCK_URL).mock(return_value=httpx.Response(200, json={"ok": 1}))
        provider = HttpProvider(MOCK_URL, _oauth_config())
        results = await asyncio.gather(*(provider.call_api(f"p{i}") for i in range(3)))
        assert all(r["output"] == {"ok": 1} for r in results)
        assert token_route.call_count == 1
        sent = token_route.calls[0].request
        assert sent.headers["content-type"] == "application/x-www-form-urlencoded"
        assert sent.content.decode() == (
            "grant_type=client_credentials&client_id=cid&client_secret=secret&scope=read+write"
        )
        assert api_route.call_count == 3
        assert api_route.calls[0].request.headers["authorization"] == "Bearer tok-1"


async def test_oauth_password_grant_and_refresh(monkeypatch):
    now = [1_000_000]
    monkeypatch.setattr(signature_module, "now_ms", lambda: now[0])
    with respx.mock() as mock:
        tokens = (f"tok-{i}" for i in itertools.count())
        token_route = mock.post("https://auth.test/token").mock(
            side_effect=lambda req: httpx.Response(
                200, json={"access_token": next(tokens), "expires_in": 120}
            )
        )
        mock.post(MOCK_URL).mock(return_value=httpx.Response(200, text="ok"))
        provider = HttpProvider(
            MOCK_URL, _oauth_config(grantType="password", username="{{ user }}", password="pw")
        )
        await provider.call_api("x", _ctx(user="alice"))
        body = token_route.calls[0].request.content.decode()
        assert "grant_type=password" in body and "username=alice&password=pw" in body
        now[0] += 30_000  # still outside the 60s refresh buffer
        await provider.call_api("x", _ctx(user="alice"))
        assert token_route.call_count == 1
        now[0] += 40_000  # inside the refresh buffer: refresh
        await provider.call_api("x", _ctx(user="alice"))
        assert token_route.call_count == 2
        # Different rendered credentials never share a token.
        await provider.call_api("x", _ctx(user="bob"))
        assert token_route.call_count == 3


async def test_oauth_failure_message():
    with respx.mock() as mock:
        mock.post("https://auth.test/token").mock(
            return_value=httpx.Response(401, text="bad client")
        )
        provider = HttpProvider(MOCK_URL, _oauth_config())
        with pytest.raises(
            ValueError,
            match="Failed to refresh OAuth token: Error: OAuth token request failed with status",
        ):
            await provider.call_api("x")


async def test_oauth_missing_access_token():
    with respx.mock() as mock:
        mock.post("https://auth.test/token").mock(return_value=httpx.Response(200, json={}))
        provider = HttpProvider(MOCK_URL, _oauth_config())
        with pytest.raises(ValueError, match="OAuth token response missing access_token"):
            await provider.call_api("x")


async def test_oauth_retries_after_failed_in_flight_refresh():
    with respx.mock() as mock:
        responses = iter(
            [httpx.Response(500, text="down"), httpx.Response(200, json={"access_token": "t2"})]
        )
        token_route = mock.post("https://auth.test/token").mock(
            side_effect=lambda r: next(responses)
        )
        mock.post(MOCK_URL).mock(return_value=httpx.Response(200, text="ok"))
        provider = HttpProvider(MOCK_URL, _oauth_config())
        first, second = await asyncio.gather(
            provider.call_api("a"), provider.call_api("b"), return_exceptions=True
        )
        assert isinstance(first, ValueError)
        assert isinstance(second, dict)
        assert token_route.call_count == 2


async def test_file_auth_with_python(fake_fetch, tmp_path, monkeypatch):
    monkeypatch.setattr(signature_module, "now_ms", lambda: 1_000_000)
    (tmp_path / "auth.py").write_text(
        "calls = []\n"
        "def get_auth(context):\n"
        "    calls.append(context['vars'].get('user'))\n"
        "    return {'token': 'file-' + str(len(calls)), 'expiration': 1_000_000 + 3_600_000}\n"
        "def never(context):\n"
        "    return {'token': 'forever'}\n"
        "def empty(context):\n"
        "    return {'token': ''}\n"
    )
    with cli_state.with_base_path(str(tmp_path)):
        provider = HttpProvider(
            MOCK_URL,
            {
                "config": {
                    "method": "POST",
                    "headers": {"Authorization": "Bearer {{ token }}"},
                    "queryParams": {"exp": "{{ expiration }}"},
                    "body": {"t": "{{ token }}"},
                    "auth": {"type": "file", "path": "file://auth.py"},
                }
            },
        )
        fake_fetch.add("{}")
        await provider.call_api("x", _ctx(user="a"))
        url, options, *_ = fake_fetch.last
        assert options["headers"]["authorization"] == "Bearer file-1"
        assert url == f"{MOCK_URL}?exp=4600000"
        assert options["body"] == '{"t":"file-1"}'
        await provider.call_api("x", _ctx(user="a"))
        assert fake_fetch.last[1]["headers"]["authorization"] == "Bearer file-1"
        await provider.call_api("x", _ctx(user="b"))
        assert fake_fetch.last[1]["headers"]["authorization"] == "Bearer file-2"

        never = HttpProvider(
            MOCK_URL,
            {
                "config": {
                    "method": "GET",
                    "headers": {"A": "{{ token }}"},
                    "auth": {"type": "file", "path": "file://auth.py:never"},
                }
            },
        )
        await never.call_api("x")
        assert fake_fetch.last[1]["headers"]["a"] == "forever"

        empty = HttpProvider(
            MOCK_URL,
            {"config": {"method": "GET", "auth": {"type": "file", "path": "file://auth.py:empty"}}},
        )
        with pytest.raises(
            ValueError, match=r"(?s)Failed to refresh file auth token: Error: .*too_small"
        ):
            await empty.call_api("x")

        missing = HttpProvider(
            MOCK_URL,
            {
                "config": {
                    "method": "GET",
                    "auth": {"type": "file", "path": "file://auth.py:missing_auth"},
                }
            },
        )
        with pytest.raises(ValueError, match="Function 'missing_auth' not found"):
            await missing.call_api("x")


async def test_file_auth_javascript_is_rejected(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {"config": {"method": "GET", "auth": {"type": "file", "path": "./auth/get-token.js"}}},
    )
    assert provider.config["auth"] == {"type": "file", "path": "./auth/get-token.js"}
    with pytest.raises(ValueError, match="JavaScript auth files are not supported"):
        await provider.call_api("x")


async def test_token_overwrite_warning(fake_fetch, caplog):
    # The token request and the API call both go through the fake fetch_with_cache.
    provider = HttpProvider(MOCK_URL, _oauth_config())
    fake_fetch.add('{"access_token":"t"}')
    fake_fetch.add("{}")
    await provider.call_api("x", _ctx(token="user-token"))
    assert fake_fetch.calls[0][0] == "https://auth.test/token"
    assert "`token` is already defined in vars and will be overwritten" in caplog.text
    assert fake_fetch.last[1]["headers"]["authorization"] == "Bearer t"


# --- signature auth ------------------------------------------------------------------------


def _rsa_pem() -> tuple[str, Any]:
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return pem, key.public_key()


def _verify(public_key: Any, signature: str, data: str) -> None:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    public_key.verify(
        base64.b64decode(signature), data.encode(), padding.PKCS1v15(), hashes.SHA256()
    )


async def test_signature_auth_pem(fake_fetch, monkeypatch, tmp_path):
    pem, public_key = _rsa_pem()
    (tmp_path / "key.pem").write_text(pem)
    now = [1_700_000_000_000]
    monkeypatch.setattr(signature_module, "now_ms", lambda: now[0])
    with cli_state.with_base_path(str(tmp_path)):
        provider = HttpProvider(
            MOCK_URL,
            {
                "config": {
                    "method": "POST",
                    "headers": {
                        "X-Signature": "{{ signature }}",
                        "X-Ts": "{{ signatureTimestamp }}",
                    },
                    "body": {"a": 1},
                    "signatureAuth": {
                        "privateKeyPath": "key.pem",
                        "signatureDataTemplate": "ts:{{signatureTimestamp}}\\nend",
                    },
                }
            },
        )
        assert provider.config["signatureAuth"]["type"] == "pem"
        fake_fetch.add("{}")
        await provider.call_api("x")
        headers = fake_fetch.last[1]["headers"]
        assert headers["x-ts"] == str(now[0])
        _verify(public_key, headers["x-signature"], f"ts:{now[0]}\nend")

        # Reused within the validity window, regenerated after it.
        now[0] += 1000
        await provider.call_api("x")
        assert fake_fetch.last[1]["headers"]["x-ts"] == str(now[0] - 1000)
        now[0] += 300_000
        await provider.call_api("x")
        assert fake_fetch.last[1]["headers"]["x-ts"] == str(now[0])


async def test_signature_auth_inline_key_and_errors(fake_fetch):
    pem, public_key = _rsa_pem()
    signature = await generate_signature(
        {
            "type": "pem",
            "privateKey": pem,
            "signatureDataTemplate": "{{signatureTimestamp}}",
            "signatureAlgorithm": "SHA256",
        },
        12345,
    )
    _verify(public_key, signature, "12345")

    with pytest.raises(ValueError, match="cannot be used: it is a public key, not a private key"):
        await generate_signature({"type": "pem", "privateKey": "-----BEGIN PUBLIC KEY-----\nx"}, 1)
    with pytest.raises(
        ValueError, match="^Failed to generate signature: PEM private key is required"
    ):
        await generate_signature({"type": "pem"}, 1)
    with pytest.raises(ValueError, match="JKS keystore password is required"):
        await generate_signature({"type": "jks", "keystorePath": "x.jks"}, 1)
    with pytest.raises(ValueError, match="JKS keystores are not supported by promptphoo"):
        await generate_signature(
            {"type": "jks", "keystorePath": "x.jks", "keystorePassword": "p"}, 1
        )
    with pytest.raises(ValueError, match="PFX certificate password is required"):
        await generate_signature({"type": "pfx", "pfxPath": "x.pfx"}, 1)
    with pytest.raises(ValueError, match="Unsupported signature auth type: weird"):
        await generate_signature({"type": "weird"}, 1)


async def test_signature_auth_pfx(tmp_path):
    import datetime

    pytest.importorskip("cryptography")

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import pkcs12
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(datetime.datetime(2020, 1, 1))
        .not_valid_after(datetime.datetime(2040, 1, 1))
        .sign(key, hashes.SHA256())
    )
    pfx = pkcs12.serialize_key_and_certificates(
        b"t", key, cert, None, serialization.BestAvailableEncryption(b"pw")
    )
    (tmp_path / "cert.pfx").write_bytes(pfx)
    with cli_state.with_base_path(str(tmp_path)):
        signature = await generate_signature(
            {
                "type": "pfx",
                "pfxPath": "cert.pfx",
                "pfxPassword": "pw",
                "signatureDataTemplate": "d",
            },
            1,
        )
        _verify(key.public_key(), signature, "d")
        content = base64.b64encode(pfx).decode()
        signature = await generate_signature(
            {
                "type": "pfx",
                "pfxContent": content,
                "pfxPassword": "pw",
                "signatureDataTemplate": "d",
            },
            1,
        )
        _verify(key.public_key(), signature, "d")
        with pytest.raises(ValueError, match="PFX file not found: .*missing.pfx"):
            await generate_signature(
                {"type": "pfx", "pfxPath": "missing.pfx", "pfxPassword": "pw"}, 1
            )


async def test_signature_failure_is_not_cached(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {"config": {"method": "GET", "signatureAuth": {"type": "pem", "privateKey": "not a key"}}},
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="it is not PEM-encoded"):
            await provider.call_api("x")
    assert provider._signature_cache == {}


def test_preprocess_signature_auth_config():
    assert preprocess_signature_auth_config(None) is None
    assert preprocess_signature_auth_config(
        {
            "certificateContent": "Y29udGVudA==",
            "certificatePassword": "pw",
            "certificateFilename": "c.pfx",
        }
    ) == {
        "type": "pfx",
        "pfxContent": "Y29udGVudA==",
        "pfxPassword": "pw",
        "certificateFilename": "c.pfx",
    }
    assert preprocess_signature_auth_config(
        {"certificateContent": base64.b64encode(b"KEY").decode(), "type": "pem"}
    ) == {"type": "pem", "privateKey": "KEY"}
    assert preprocess_signature_auth_config({"keystorePath": "a.jks"}) == {
        "keystorePath": "a.jks",
        "type": "jks",
    }
    with pytest.raises(ValueError, match="Cannot determine certificate type from filename: c.txt"):
        preprocess_signature_auth_config(
            {"certificateContent": "x", "certificateFilename": "c.txt"}
        )
    with pytest.raises(ValueError, match="Unknown certificate type: zip"):
        preprocess_signature_auth_config({"type": "zip", "certificateContent": "x"})


def test_diagnose_private_key_material():
    assert diagnose_private_key_material("") == "it is empty"
    assert diagnose_private_key_material(None) == "it is empty"
    assert diagnose_private_key_material("-----BEGIN CERTIFICATE-----") == (
        "it is a certificate, not a private key"
    )
    assert "passphrase-protected" in diagnose_private_key_material(
        "-----BEGIN ENCRYPTED PRIVATE KEY-----"
    )
    assert "truncated" in diagnose_private_key_material("-----BEGIN PRIVATE KEY-----\nabc")
    assert (
        diagnose_private_key_material("-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----")
        is None
    )


# --- TLS -----------------------------------------------------------------------------------


def test_tls_requires_cert_and_key_together():
    with pytest.raises(ValueError, match="Both certificate and key must be provided"):
        HttpProvider(MOCK_URL, {"config": {"method": "GET", "tls": {"certPath": "c.pem"}}})
    HttpProvider(MOCK_URL, {"config": {"method": "GET", "tls": {"pfxPath": "c.pfx"}}})


async def test_tls_agent_is_passed_as_dispatcher(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "GET",
                "tls": {"rejectUnauthorized": False, "minVersion": "TLSv1.2", "ciphers": "HIGH"},
            }
        },
    )
    fake_fetch.add("{}")
    await provider.call_api("x")
    dispatcher = fake_fetch.last[1]["dispatcher"]
    assert isinstance(dispatcher, httpx.AsyncClient)
    assert provider._https_agent is dispatcher
    await provider.call_api("x")
    assert fake_fetch.last[1]["dispatcher"] is dispatcher
    await provider.cleanup()


def test_tls_ssl_context_options(tmp_path):
    import ssl

    from promptphoo.providers.http.tls import create_ssl_context

    ctx = create_ssl_context(
        {"rejectUnauthorized": False, "minVersion": "TLSv1.2", "maxVersion": "TLSv1.3"}
    )
    assert ctx.verify_mode == ssl.CERT_NONE
    assert ctx.check_hostname is False
    assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2
    with pytest.raises(ValueError, match="Failed to load JKS certificate"):
        create_ssl_context({"jksPath": "a.jks"})
    with pytest.raises(FileNotFoundError), cli_state.with_base_path(str(tmp_path)):
        create_ssl_context({"caPath": "missing-ca.pem"})


async def test_tls_through_fetch_with_respx():
    with respx.mock() as mock:
        route = mock.get("https://secure.test/x").mock(return_value=httpx.Response(200, text="ok"))
        provider = HttpProvider(
            "https://secure.test/x",
            {"config": {"method": "GET", "tls": {"rejectUnauthorized": False}}},
        )
        result = await provider.call_api("x")
        assert result["output"] == "ok"
        assert route.called
        await provider.cleanup()


# --- multipart -----------------------------------------------------------------------------


def test_multipart_config_validation():
    parts = {"parts": [{"kind": "field", "name": "a", "value": "b"}]}
    with pytest.raises(ValueError, match="cannot include both body and multipart"):
        HttpProvider(MOCK_URL, {"config": {"body": {"a": 1}, "multipart": parts}})
    with pytest.raises(ValueError, match="cannot include both request and multipart"):
        HttpProvider(MOCK_URL, {"config": {"request": "GET / HTTP/1.1", "multipart": parts}})


async def test_multipart_request(fake_fetch, tmp_path):
    (tmp_path / "doc.txt").write_text("file contents")
    with cli_state.with_base_path(str(tmp_path)):
        provider = HttpProvider(
            MOCK_URL,
            {
                "config": {
                    "headers": {"Content-Type": "application/json", "X-A": "b"},
                    "multipart": {
                        "parts": [
                            {"kind": "field", "name": "question", "value": "{{ prompt }}"},
                            {
                                "kind": "file",
                                "name": "upload",
                                "source": {"type": "path", "path": "doc.txt"},
                            },
                            {
                                "kind": "file",
                                "name": "generated",
                                "filename": "{{ name }}.pdf",
                                "source": {
                                    "type": "generated",
                                    "format": "pdf",
                                    "text": "Hi {{ prompt }}",
                                },
                            },
                        ]
                    },
                }
            },
        )
        fake_fetch.add("{}")
        result = await provider.call_api("what?", {"vars": {"name": "report"}, "debug": True})
    url, options, _, _, bust, _ = fake_fetch.last
    assert options["method"] == "POST"
    assert bust is True
    content_type = options["headers"]["content-type"]
    assert content_type.startswith("multipart/form-data; boundary=----formdata-undici-0")
    assert options["headers"]["x-a"] == "b"
    boundary = content_type.split("boundary=")[1]
    body = options["body"]
    assert body.startswith(
        f'--{boundary}\r\nContent-Disposition: form-data; name="question"\r\n\r\nwhat?\r\n'.encode()
    )
    assert (
        b'Content-Disposition: form-data; name="upload"; filename="doc.txt"\r\n'
        b"Content-Type: text/plain\r\n\r\nfile contents\r\n"
    ) in body
    assert b'filename="report.pdf"\r\nContent-Type: application/pdf\r\n\r\n%PDF-1.4' in body
    assert body.endswith(f"--{boundary}--\r\n".encode())
    assert result["metadata"]["multipart"]["fields"] == [{"field": "question", "value": "what?"}]
    assert result["metadata"]["multipart"]["files"][0] == {
        "field": "upload",
        "filename": "doc.txt",
        "contentType": "text/plain",
        "sizeBytes": 13,
        "source": "path",
    }


async def test_multipart_get_is_rejected(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "GET",
                "multipart": {"parts": [{"kind": "field", "name": "a", "value": "b"}]},
            }
        },
    )
    with pytest.raises(ValueError, match="HTTP provider GET requests cannot use multipart"):
        await provider.call_api("x")


# --- metadata sanitization -----------------------------------------------------------------


async def test_debug_metadata_redacts_auth_headers(fake_fetch):
    provider = HttpProvider(
        MOCK_URL,
        {
            "config": {
                "method": "POST",
                "headers": {"Authorization": "Bearer secret-token", "X-API-Key": "k", "X-Ok": "v"},
                "body": {"password": "hunter2", "q": "{{ prompt }}"},
            }
        },
    )
    fake_fetch.add("{}", headers={"set-cookie": "session=abc", "x-plain": "1"})
    result = await provider.call_api("x", {"vars": {}, "debug": True})
    http_meta = result["metadata"]["http"]
    assert http_meta["requestHeaders"]["authorization"] == "[REDACTED]"
    assert http_meta["requestHeaders"]["x-api-key"] == "[REDACTED]"
    assert http_meta["requestHeaders"]["x-ok"] == "v"
    assert result["metadata"]["finalRequestBody"] == {"password": "[REDACTED]", "q": "x"}
    # The request itself is not redacted.
    assert fake_fetch.last[1]["headers"]["authorization"] == "Bearer secret-token"


async def test_transformed_raw_request_is_redacted(fake_fetch):
    provider = HttpProvider(
        "http",
        {
            "config": {
                "request": "POST /a HTTP/1.1\nHost: example.com\nContent-Type: text/plain\n\n{{prompt}}",  # noqa: E501
                "transformRequest": "'GET /x?api_key=sk-abcdefghijklmnopqrstuvwxyz HTTP/1.1\\nAuthorization: Bearer abc'",  # noqa: E501
            }
        },
    )
    fake_fetch.add("ok")
    result = await provider.call_api("x", {"vars": {}, "debug": True})
    transformed = result["metadata"]["transformedRequest"]
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in transformed
    assert "Bearer abc" not in transformed


# --- end-to-end eval -----------------------------------------------------------------------


async def test_end_to_end_eval_against_respx(tmp_path, monkeypatch):
    from promptphoo import db, paths
    from promptphoo.evaluator.run import evaluate
    from promptphoo.models.eval import Eval
    from promptphoo.providers import load_api_providers

    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path))
    paths.set_config_directory_path(None)
    db.close_db()
    db.run_db_migrations()
    try:
        with respx.mock(assert_all_called=True) as mock:

            def reply(request: httpx.Request) -> httpx.Response:
                question = json.loads(request.content)["messages"][0]["content"]
                return httpx.Response(200, json={"answer": f"echo: {question}"})

            route = mock.post("https://example.test/chat").mock(side_effect=reply)
            providers = await load_api_providers(
                [
                    {
                        "id": "https://example.test/chat",
                        "config": {
                            "method": "POST",
                            "headers": {"Content-Type": "application/json"},
                            "body": {"messages": [{"role": "user", "content": "{{prompt}}"}]},
                            "transformResponse": "json.answer",
                        },
                    }
                ]
            )
            suite = {
                "providers": providers,
                "prompts": [{"raw": "Hi {{name}}", "label": "greet"}],
                "tests": [
                    {
                        "vars": {"name": "Ann"},
                        "assert": [{"type": "equals", "value": "echo: Hi Ann"}],
                    },
                    {"vars": {"name": "Bo"}, "assert": [{"type": "contains", "value": "nope"}]},
                ],
            }
            record = Eval({"tests": suite["tests"]}, author=None)
            await evaluate(suite, record, {"cache": False})
            assert route.call_count == 2
        results = sorted(
            record.results, key=lambda r: r["testIdx"] if isinstance(r, dict) else r.test_idx
        )
        outputs = [
            (r["response"]["output"], r["success"])
            if isinstance(r, dict)
            else (r.response["output"], r.success)
            for r in results
        ]
        assert outputs == [("echo: Hi Ann", True), ("echo: Hi Bo", False)]
    finally:
        db.close_db()
