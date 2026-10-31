"""Tests for promptphoo.sanitizer and promptphoo.models.result_sanitize.

Ported from test/util/sanitizer.test.ts and the sanitizeProvider / toSerializableProviderRef /
sanitizeResultForJsonlArtifact parts of test/models/evalResult.test.ts. The golden fixture
(tests/fixtures/sanitizer_golden.json, from tools/sanitizer_golden.py) holds real Node output of
upstream sanitizer.ts for every string literal in the upstream test file plus fuzzed URLs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from promptphoo import sanitizer as s
from promptphoo.models import result_sanitize as rs
from promptphoo.providers.base import ApiProvider
from promptphoo.sanitizer import (
    REDACTED,
    is_credential_header,
    is_secret_env_var_name,
    looks_like_secret,
    preserve_tracing_credential_references,
    safe_json_stringify,
    sanitize_config_for_output,
    sanitize_object,
    sanitize_runtime_options,
    sanitize_tracing_config_for_persistence,
    sanitize_url,
    sanitize_url_for_logging,
    strip_provider_prompt_selectors,
)

GOLDEN = Path(__file__).parent / "fixtures" / "sanitizer_golden.json"

# ---------------------------------------------------------------------------------------------
# Golden parity with Node
# ---------------------------------------------------------------------------------------------


def _golden_object(obj: Any, opts: dict[str, Any] | None = None) -> Any:
    opts = opts or {}
    md = opts.get("maxDepth", 4)
    return sanitize_object(
        obj,
        sanitize_urls=opts.get("sanitizeUrls", False),
        max_depth=math.inf if md == "inf" else md,
    )


_GOLDEN_FUNCS = {
    "sanitizeUrl": sanitize_url,
    "sanitizeUrlForLogging": sanitize_url_for_logging,
    "sanitizeUrlEncodedString": s.sanitize_url_encoded_string,
    "looksLikeSecret": looks_like_secret,
    "isSecretEnvVarName": is_secret_env_var_name,
    "isCredentialHeader": is_credential_header,
    "sanitizeObject": _golden_object,
}


def test_matches_node_golden_output() -> None:
    rows = json.loads(GOLDEN.read_text())
    mismatches = []
    for fn, args, expected in rows:
        try:
            actual: dict[str, Any] = {"ok": _GOLDEN_FUNCS[fn](*args)}
        except Exception as e:  # pragma: no cover - reported below
            actual = {"err": repr(e)}
        if json.dumps(actual) != json.dumps(expected):
            mismatches.append((fn, args, expected, actual))
    assert not mismatches[:5], f"{len(mismatches)} of {len(rows)} differ"
    assert len(rows) > 5000


# ---------------------------------------------------------------------------------------------
# sanitizeRuntimeOptions / safeJsonStringify
# ---------------------------------------------------------------------------------------------


def test_sanitize_runtime_options() -> None:
    assert sanitize_runtime_options(
        {
            "abortSignal": object(),
            "progressCallback": lambda *a: None,
            "providerFilter": "selected-target",
        }
    ) == {"providerFilter": "selected-target"}
    assert sanitize_runtime_options(None) is None


def test_safe_json_stringify_drops_circular_references() -> None:
    obj: dict[str, Any] = {"a": 1, "f": lambda: 1, "items": [1.0, float("nan")]}
    obj["self"] = obj
    shared = {"x": 1}
    obj["s1"] = shared
    obj["s2"] = shared
    assert safe_json_stringify(obj) == '{"a":1,"items":[1,null],"s1":{"x":1},"s2":{"x":1}}'
    assert safe_json_stringify({"a": [1]}, True) == '{\n  "a": [\n    1\n  ]\n}'
    assert safe_json_stringify(lambda: 1) is None


# ---------------------------------------------------------------------------------------------
# isCredentialHeader / looksLikeSecret / isSecretEnvVarName
# ---------------------------------------------------------------------------------------------


def _header_case_variants(name: str) -> set[str]:
    import re

    def flip(m: re.Match[str]) -> str:
        return m.group(0).lower() if m.start() % 2 == 0 else m.group(0).upper()

    mixed = re.sub(r"[a-z]", flip, name, flags=re.I)
    return {name, name.lower(), name.upper(), mixed, re.sub(r"key\Z", "Key", mixed, flags=re.I)}


@pytest.mark.parametrize(
    "name",
    [
        "Ocp-Apim-Subscription-Key",
        "X-Subscription-Key",
        "subscription_key",
        "subscriptionKey",
        "X-Functions-Key",
        "X-Arbitrary-Vendor-Key",
        "X-Session",
        "X-Session-Id",
        "vendorSessionId",
        "vendor_session_id",
        "X-Session-Access",
        "X-Gateway-Authentication",
        "X-Gateway-Token",
        "X-Gateway-Cookie",
        "GatewayTokenV2",
        "GatewaySecret",
        "GatewayPassword",
        "GatewayCredentials",
        "GatewayApiKeyV2",
        "X-Goog-Iap-Jwt-Assertion",
        "_oauth2_proxy",
    ],
)
def test_is_credential_header_recognizes_credential_names(name: str) -> None:
    for variant in _header_case_variants(name):
        assert is_credential_header(variant, "short"), variant
        assert is_credential_header(variant, "9be880e3-e5dc-4be7-8739-a4b587fdfb13"), variant


@pytest.mark.parametrize(
    "name",
    [
        "X-Subscription-Id",
        "X-Subscription-Tier",
        "X-Correlation-Id",
        "Idempotency-Key",
        "Cache-Key",
        "X-Routing-Key",
        "X-Partition-Key",
        "X-Public-Key",
        "Sec-WebSocket-Key",
        "X-Session-Timeout",
        "X-Session-Type",
        "X-Session-Id-Mode",
        "vendorSessionTimeout",
        "X-Access-Region",
        "X-Session-Access-Mode",
    ],
)
def test_is_credential_header_preserves_ordinary_metadata(name: str) -> None:
    for variant in _header_case_variants(name):
        assert not is_credential_header(variant, "us"), variant
        assert not is_credential_header(variant, "0123456789abcdef0123456789abcdef"), variant


@pytest.mark.parametrize("name", ["Idempotency-Key", "Cache-Key", "X-Public-Key"])
def test_credential_values_under_public_key_role_headers(name: str) -> None:
    assert is_credential_header(name, "Bearer short-credential")
    assert is_credential_header(name, "sk-abcdefghijklmnopqrstuvw")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("A" * 63, False),
        ("A" * 64, True),
        ("aZ09+/=_-" * 8, True),
        ("A" * 64 + " ", False),
        ("A" * 64 + "\n", False),
        ("A" * 64 + " ", False),
        ("A" * 63 + "é", False),
        ("A" * 64 + "\0", False),
        ("sk-abcdefghijklmnopqrstuv", True),
        ("Bearer abcdefghijklmnopqrstuvwxyz", True),
        ("AKIAABCDEFGHIJKLMNOP", True),
    ],
)
def test_looks_like_secret(value: str, expected: bool) -> None:
    assert looks_like_secret(value) is expected


@pytest.mark.parametrize(
    "name",
    [
        "GITHUB_TOKEN",
        "STRIPE_SECRET_KEY",
        "PGPASSWORD",
        "DB_PASSWD",
        "OPENAI_API_KEY",
        "github_token",
        "_GITHUB_TOKEN",
        "AWS_SECRETKEY",
        "GCP_PRIVATEKEY",
        "APP_ENCRYPTIONKEY",
        "DBPWD",
        "DATABASEDSN",
        "MLFLOW_BASIC_AUTH",
        "NPM_CONFIG__AUTH",
    ],
)
def test_secret_env_var_names(name: str) -> None:
    assert is_secret_env_var_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "MAX_TOKENS",
        "RETRY_KEYS",
        "LOG_LEVEL",
        "SERVICE_URL",
        "PORTKEY_API_BASE_URL",
        "MONKEY",
        "WATSONX_AI_AUTH_TYPE",
        "OAUTH_SCOPE",
        "maxTokens",
        "tokenCount",
        "keyName",
        "cacheKey",
    ],
)
def test_non_secret_env_var_names(name: str) -> None:
    assert not is_secret_env_var_name(name)


# ---------------------------------------------------------------------------------------------
# sanitizeObject
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "url",
        "apiBaseUrl",
        "gatewayUrl",
        "baseUrl",
        "endpoint",
        "apiHost",
        "HTTPS_PROXY",
        "MONGODB_URI",
    ],
)
def test_env_maps_redact_url_credentials(key: str) -> None:
    for value in [
        "https://user:short-password@host/v1?token=short-token",
        "/api?token=short-token",
    ]:
        inp = {key: value}
        out = json.dumps(
            [
                sanitize_object(inp, sanitize_urls=True),
                sanitize_object({"env": inp}, sanitize_urls=True),
            ]
        )
        assert "short-password" not in out
        assert "short-token" not in out
        assert inp[key] == value


@pytest.mark.parametrize(
    "value",
    [
        "http://localhost:1975",
        "HTTPS://GW.Example.COM/v1",
        "https://gw.example/v1?design=blue&assignment=a1",
        "127.0.0.1:11434",
        "my-resource.openai.azure.com/openai",
        "{{ env.GATEWAY_URL }}",
    ],
)
def test_preserves_non_secret_url_fields(value: str) -> None:
    inp = {"apiBaseUrl": value, "env": {"OLLAMA_BASE_URL": value}}
    assert sanitize_object(inp) == inp


@pytest.mark.parametrize(
    "data", [{"api_key": "short-secret"}, {"baseUrl": "https://gw.example/v1", "api_key": "x"}]
)
def test_redacts_json_shaped_url_values(data: dict[str, str]) -> None:
    value = json.dumps(data, separators=(",", ":"))
    result = sanitize_object({"url": value, "apiBaseUrl": value, "env": {"GATEWAY_URL": value}})
    for sanitized in (result["url"], result["apiBaseUrl"], result["env"]["GATEWAY_URL"]):
        assert json.loads(sanitized) == {**data, "api_key": REDACTED}


def test_redacts_gateway_urls_without_changing_runtime_config() -> None:
    url = "https://gateway-user:gateway-password@gateway.example/v1?token=short-secret"
    config = {
        "env": {"ENVOY_API_BASE_URL": url, "SECRET_URL": url},
        "providers": [{"config": {"apiBaseUrl": url}}],
    }
    result = sanitize_object(config, max_depth=math.inf, sanitize_urls=True)
    assert (
        result["env"]["ENVOY_API_BASE_URL"]
        == "https://***:***@gateway.example/v1?token=%5BREDACTED%5D"
    )
    assert result["env"]["SECRET_URL"] == REDACTED
    assert result["providers"][0]["config"]["apiBaseUrl"] == result["env"]["ENVOY_API_BASE_URL"]
    assert config["env"]["ENVOY_API_BASE_URL"] == url


def test_redacts_credential_named_env_variables() -> None:
    result = sanitize_object(
        {
            "env": {
                "API_KEY": "sk-value",
                "github_token": "ghp_lowercase_value",
                "PORTKEY_API_BASE_URL": "https://api.portkey.ai/v1",
                "GITHUB_TOKEN": "ghp_value",
                "MY_SERVICE_SECRET": "plainvalue123",
                "PGPASSWORD": "hunter2",
                "LOG_LEVEL": "debug",
                "MAX_TOKENS": "4096",
            }
        }
    )
    assert result["env"] == {
        "API_KEY": REDACTED,
        "github_token": REDACTED,
        "PORTKEY_API_BASE_URL": "https://api.portkey.ai/v1",
        "GITHUB_TOKEN": REDACTED,
        "MY_SERVICE_SECRET": REDACTED,
        "PGPASSWORD": REDACTED,
        "LOG_LEVEL": "debug",
        "MAX_TOKENS": "4096",
    }


def test_env_map_nested_in_provider_config() -> None:
    result = sanitize_object(
        {"config": {"mcp": {"servers": [{"command": "node", "env": {"GITHUB_TOKEN": "x"}}]}}},
        max_depth=math.inf,
    )
    assert result["config"]["mcp"]["servers"][0]["env"]["GITHUB_TOKEN"] == REDACTED


def test_does_not_widen_redaction_outside_env_maps() -> None:
    inp = {"maxTokens": 4096, "tokenCount": 12, "keyName": "X-Api-Key", "MAX_TOKENS": "2048"}
    assert sanitize_object(inp) == inp


def test_primitives() -> None:
    assert sanitize_object(None) is None
    assert sanitize_object(42) == 42
    assert sanitize_object(True) is True
    assert sanitize_object("plain") == "plain"
    assert sanitize_object("sk-abcdefghijklmnopqrstuvwxyz") == REDACTED
    assert sanitize_object('{"password":"x","a":1}') == '{"password":"[REDACTED]","a":1}'


def test_functions_are_omitted() -> None:
    def named() -> str:
        return "x"

    assert sanitize_object({"func": named, "level1": {"func": lambda: 1, "v": 1}}) == {
        "level1": {"v": 1}
    }


def test_depth_limit() -> None:
    deep = {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}
    assert sanitize_object(deep) == {"a": {"b": {"c": {"d": {"e": "[...]"}}}}}
    assert sanitize_object(deep, max_depth=math.inf) == deep


def test_class_instances_and_errors() -> None:
    class TestClass:
        def __init__(self) -> None:
            self.data = "test"

        def method(self) -> str:
            return "method"

    result = sanitize_object({"obj": TestClass(), "error": ValueError("boom"), "set": {1, 2}})
    assert result == {
        "obj": {"data": "test"},
        "error": {"name": "ValueError", "message": "boom"},
        "set": {},
    }


def test_circular_references() -> None:
    obj: dict[str, Any] = {"name": "test", "password": "secret"}
    obj["self"] = obj
    result = sanitize_object(obj)
    assert result == {"name": "test", "password": REDACTED, "self": "[Circular]"}


def test_edge_cases() -> None:
    assert sanitize_object({}) == {}
    assert sanitize_object({"password": "", "token": None, "apiKey": 0}) == {
        "password": REDACTED,
        "token": REDACTED,
        "apiKey": REDACTED,
    }
    assert sanitize_object({"password": False, "token": True}) == {
        "password": REDACTED,
        "token": REDACTED,
    }
    inp = {"password": "secret", "data": "public"}
    result = sanitize_object(inp)
    assert inp["password"] == "secret" and result is not inp
    # V8 enumerates integer-like keys first.
    assert list(sanitize_object({"b": 1, "1": "x", "0": "y"})) == ["0", "1", "b"]
    assert sanitize_object({"api-key": "secret", "key.with.dots": "v"}) == {
        "api-key": REDACTED,
        "key.with.dots": "v",
    }


def test_errors_are_redacted_or_raised() -> None:
    class Broken:
        def to_json(self) -> Any:
            raise RuntimeError("Getter failed")

    # fast-safe-stringify turns a throwing toJSON into its canonical failure string.
    assert sanitize_object({"x": Broken()}) == s.UNABLE_TO_SERIALIZE

    class Exploding(dict):
        def items(self):  # type: ignore[override]
            raise RuntimeError("Parse error")

    assert sanitize_object(Exploding(a=1)) == REDACTED
    with pytest.raises(RuntimeError, match="Parse error"):
        sanitize_object(Exploding(a=1), throw_on_error=True)


@pytest.mark.parametrize(
    "api_host",
    [
        "gateway.example",
        "gateway.example:8443",
        "gateway.example/path",
        "gateway.example/path/",
        "https://api.azure.com",
        "https://api.azure.com/",
        "http://gateway.example:8080",
        "https://gateway.example/path",
    ],
)
def test_preserves_api_host_formatting(api_host: str) -> None:
    assert sanitize_object({"apiHost": api_host}) == {"apiHost": api_host}


@pytest.mark.parametrize("key", ["apiBaseUrl", "server_url", "apiHost"])
def test_redacts_credential_paths(key: str) -> None:
    prefix = "" if key == "apiHost" else "https://"
    for endpoint in (
        f"{prefix}gateway.example/auth-supersecretvalue123",
        f"{prefix}gateway.example/auth/opaquegateway7294/v1",
    ):
        out = json.dumps(sanitize_object({key: endpoint}))
        assert "supersecretvalue123" not in out and "opaquegateway7294" not in out


def test_url_keyed_fields() -> None:
    for url in ("example.com", "/relative/path", "my-tokenizer-model"):
        assert sanitize_object({"vars": {"url": url}}) == {"vars": {"url": url}}
    assert sanitize_object({"url": "https://user:hunter2@host/api"}) == {
        "url": "https://***:***@host/api"
    }
    assert sanitize_object({"url": "ht!tp://x?token=sk-1234567890abcdefghij"}) == {"url": REDACTED}


def test_url_map_keys_that_collide_after_redaction() -> None:
    redacted = "https://example.com/?token=%5BREDACTED%5D"
    values = {
        "https://example.com/?token=first-private-value": "first",
        "https://example.com/?token=second-private-value": "second",
        redacted: "original",
        redacted + "#1": "original-fragment",
    }
    output = sanitize_object(values, sanitize_urls=True)
    assert sorted(output.values()) == sorted(values.values())
    assert output[redacted] == "original"
    assert output[redacted + "#1"] == "original-fragment"
    assert "private-value" not in json.dumps(output)


def test_limits_general_url_redaction_to_url_fields() -> None:
    url = "https://cdn.example/doc?X-Amz-Signature=short-secret&q=hello world"
    vars_ = {"image": url, "noteUrl": "see https://shop.example/?token=abc for details"}
    assert sanitize_object(vars_) == vars_
    assert sanitize_object(url) == url
    assert "short-secret" not in sanitize_object({"url": url})["url"]
    assert "short-secret" not in json.dumps(
        sanitize_config_for_output({"tests": [{"vars": vars_}]})
    )


def test_legacy_aliases() -> None:
    assert s.sanitize_headers is sanitize_object
    assert s.sanitize_query_params is sanitize_object
    assert s.sanitize_body is sanitize_object


# ---------------------------------------------------------------------------------------------
# sanitizeUrl
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("", ""),
        ("   ", "   "),
        ("not-a-valid-url", "not-a-valid-url"),
        ("not-a-valid-url?api_key=secret123", REDACTED),
        ("ht!tp://x?private_token=abc123", REDACTED),
        ("http://[bad]/?github%5Ftoken=abc123", REDACTED),
        ("ht!tp://invalid", "ht!tp://invalid"),
        ("https://user:pass@example.com/api", "https://***:***@example.com/api"),
        (
            "https://example.com/api?data=ok;api_key=sk-1234567890abcdefghij",
            "https://example.com/api?data=%5BREDACTED%5D",
        ),
        (
            "https://example.com/callback#access_token=sk-1234567890abcdefghij&state=ok",
            "https://example.com/callback#access_token=%5BREDACTED%5D&state=ok",
        ),
        (
            "https://example.com/api?data=public#section",
            "https://example.com/api?data=public#section",
        ),
        ("/api/openai/completion", "/api/openai/completion"),
        (
            "/api/endpoint?api_key=secret123&data=public",
            "/api/endpoint?api_key=%5BREDACTED%5D&data=public",
        ),
        (
            "/api/endpoint?cursor=" + "a" * 64 + "&data=public",
            "/api/endpoint?cursor=%5BREDACTED%5D&data=public",
        ),
        ("/api/endpoint?token=secret#section", "/api/endpoint?token=%5BREDACTED%5D#section"),
        ("/", "/"),
        ("//example.com/api?api_key=secret123", REDACTED),
        # Unchanged URLs keep their original spelling.
        ("HTTPS://Example.COM:443/a/../b", "HTTPS://Example.COM:443/a/../b"),
        (
            "HTTPS://Example.COM:443/a/../b?token=1",
            "https://example.com/b?token=%5BREDACTED%5D",
        ),
        (
            "ws://127.0.0.1/sessions/{{ sessionId }}?token=runtime-secret",
            "ws://127.0.0.1/sessions/{{ sessionId }}?token=%5BREDACTED%5D",
        ),
    ],
)
def test_sanitize_url(url: str, expected: str) -> None:
    assert sanitize_url(url) == expected


def test_sanitize_url_non_strings() -> None:
    assert sanitize_url(None) is None  # type: ignore[arg-type]
    assert sanitize_url(123) == 123  # type: ignore[arg-type]


def test_sanitize_url_for_logging_malformed_escape() -> None:
    assert "opaque" not in sanitize_url_for_logging("https://gateway.example/auth/opaque%ZZ")


# ---------------------------------------------------------------------------------------------
# stripProviderPromptSelectors / sanitizeConfigForOutput
# ---------------------------------------------------------------------------------------------


class LiveProvider(ApiProvider):
    def __init__(self, pid: str = "echo", **attrs: Any) -> None:
        super().__init__({"id": pid, "config": attrs.pop("config", {})})
        for k, v in attrs.items():
            setattr(self, k, v)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id(), "config": self.config}
        if self.prompts is not None:
            out["prompts"] = self.prompts
        return out


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {
            "label": "ordinary map label",
            "config": {"prompts": ["ordinary map configuration"]},
            "prompts": ["ordinary map setting"],
        },
    ],
)
def test_strip_provider_prompt_selectors_on_type_maps(metadata: dict[str, Any]) -> None:
    provider = {
        "id": "echo",
        "prompts": ["ordinary selector"],
        "config": {"prompts": ["ordinary provider payload"]},
    }
    application = {"prompts": ["ordinary application payload"]}
    m = {
        "text": provider,
        "embedding": "echo",
        "classification": provider,
        "moderation": provider,
        "application": application,
        **metadata,
    }
    projected = strip_provider_prompt_selectors(m)
    expected_provider = {"id": "echo", "config": provider["config"]}
    assert projected == {
        **m,
        "text": expected_provider,
        "classification": expected_provider,
        "moderation": expected_provider,
    }
    assert projected["application"] is application
    assert m["text"] is provider
    assert provider["prompts"] == ["ordinary selector"]


@pytest.mark.parametrize(
    "prompts",
    [
        "private literal",
        ["private literal"],
        [{"raw": "private literal", "label": "public label"}],
        {"private literal": "public label"},
    ],
)
def test_config_output_omits_prompts_when_stripping(prompts: Any) -> None:
    config = {"prompts": prompts}
    assert "prompts" not in sanitize_config_for_output(config, should_strip_prompt_text=True)
    assert sanitize_config_for_output(config)["prompts"] == prompts


@pytest.mark.parametrize(
    "providers",
    [
        {"id": "echo", "prompts": ["ordinary selector"]},
        [{"id": "echo", "prompts": ["ordinary selector"]}],
        [{"echo": {"prompts": ["ordinary selector"]}}],
        [LiveProvider(prompts=["ordinary selector"])],
    ],
)
def test_config_output_omits_provider_selectors(providers: Any) -> None:
    provider_prompt_map = {"echo": ["ordinary selector"]}
    inp = {
        "providers": providers,
        "providerPromptMap": provider_prompt_map,
        "metadata": {"prompts": ["ordinary metadata"]},
    }
    stripped = sanitize_config_for_output(inp, should_strip_prompt_text=True)
    assert "ordinary selector" not in json.dumps(stripped)
    assert stripped["metadata"] == inp["metadata"]
    assert "providerPromptMap" not in stripped
    assert sanitize_config_for_output(inp)["providerPromptMap"] == provider_prompt_map
    assert "ordinary selector" in json.dumps(sanitize_config_for_output(inp))


@pytest.mark.parametrize("strip", [True, False])
def test_config_output_projects_live_grading_maps(strip: bool) -> None:
    provider = LiveProvider(
        prompts=["ordinary live selector"], config={"prompts": ["ordinary provider payload"]}
    )
    m = {
        "text": provider,
        "embedding": provider,
        "classification": provider,
        "moderation": provider,
        "label": "ordinary map label",
        "config": {"prompts": ["ordinary map configuration"]},
        "prompts": ["ordinary map setting"],
        "application": {"prompts": ["ordinary application payload"]},
    }
    test = {
        "options": {"provider": m},
        "assert": [{"type": "assert-set", "assert": [{"type": "equals", "provider": m}]}],
    }
    config = {
        "tests": [test],
        "defaultTest": test,
        "scenarios": [{"config": [test], "tests": [test]}],
    }
    projected_provider: dict[str, Any] = {"id": "echo", "config": provider.config}
    if not strip:
        projected_provider["prompts"] = provider.prompts
    projected_map = {
        **m,
        "text": projected_provider,
        "embedding": projected_provider,
        "classification": projected_provider,
        "moderation": projected_provider,
    }
    projected_test = {
        "options": {"provider": projected_map},
        "assert": [
            {"type": "assert-set", "assert": [{"type": "equals", "provider": projected_map}]}
        ],
    }
    assert sanitize_config_for_output(config, should_strip_prompt_text=strip) == {
        "tests": [projected_test],
        "defaultTest": projected_test,
        "scenarios": [{"config": [projected_test], "tests": [projected_test]}],
    }
    assert m["text"] is provider


def test_config_output_preserves_base_path_and_provider_origins() -> None:
    base_path = "/home/" + "nested/" * 15 + "project"
    assert sanitize_config_for_output({"basePath": base_path})["basePath"] == base_path
    for strip_metadata in (False, True):
        provider_base_path = base_path + "/tests"
        test = {
            "provider": "python:provider.py",
            "metadata": {
                "note": "private-note",
                "__promptfoo": {"providerBasePath": provider_base_path},
            },
        }
        config = {
            "tests": [test],
            "defaultTest": test,
            "scenarios": [{"config": [test], "tests": [test]}],
        }
        output = sanitize_config_for_output(config, should_strip_metadata=strip_metadata)
        md: dict[str, Any] = {} if strip_metadata else {"note": "private-note"}
        expected = {
            **test,
            "metadata": {**md, "__promptfoo": {"providerBasePath": provider_base_path}},
        }
        assert output["tests"] == [expected]
        assert output["defaultTest"] == expected
        assert output["scenarios"] == [{"config": [expected], "tests": [expected]}]
    assert sanitize_object(base_path + "/tests") == REDACTED


def test_config_output_keeps_remote_safety() -> None:
    secret = "sk-" + "a" * 32
    remote_path = "/home/" + "nested/" * 15 + "remote"
    config = {
        "tests": [
            {"metadata": {"apiKey": secret, "__promptfoo": {"providerBasePath": secret}}},
            {"metadata": {"__promptfoo": {"remote": True, "providerBasePath": remote_path}}},
        ]
    }
    output = json.dumps(sanitize_config_for_output(config))
    assert secret not in output and remote_path not in output
    stripped = sanitize_config_for_output(config, should_strip_metadata=True)
    assert stripped["tests"][1] == {"metadata": {"__promptfoo": {"remote": True}}}


def test_config_output_strips_saved_test_data() -> None:
    test = {
        "vars": {"input": "private test vars"},
        "metadata": {"note": "private metadata", "__promptfoo": {"remote": True}},
        "providerOutput": "private recorded output",
        "assert": [{"type": "equals", "value": "answer"}],
    }
    config = {
        "tests": [test],
        "defaultTest": test,
        "scenarios": [{"config": [test], "tests": [test]}],
    }
    output = sanitize_config_for_output(
        config,
        should_strip_test_vars=True,
        should_strip_metadata=True,
        should_strip_response_output=True,
    )
    assert "private" not in json.dumps(output)
    assert output["tests"] == [
        {"metadata": {"__promptfoo": {"remote": True}}, "assert": test["assert"]}
    ]
    assert config["tests"][0]["vars"]["input"] == "private test vars"


@pytest.mark.parametrize("scheme", ["https", "http", "ws", "wss"])
def test_config_output_redacts_provider_id_urls(scheme: str) -> None:
    url = f"{scheme}://gateway.example/v1?tenantClientSecret=short-value"
    safe_url = "HTTPS://Safe.Example/v1?oauth=true&useSession=false&sameSiteCookie=lax"
    redacted = f"{scheme}://gateway.example/v1?tenantClientSecret=%5BREDACTED%5D"
    output = sanitize_config_for_output(
        {"providers": [url, {"id": url}, {url: {}}, {"id": safe_url}]}
    )
    assert output["providers"] == [redacted, {"id": redacted}, {redacted: {}}, {"id": safe_url}]


def test_config_output_retains_safe_tracing_env_aliases() -> None:
    config = {
        "env": {"LANGFUSE_SECRET_KEY": "{{ env.ACTUAL_SECRET }}", "ACTUAL_SECRET": "private-value"},
        "tracing": {
            "enabled": True,
            "provider": {
                "id": "langfuse",
                "endpoint": "https://cloud.langfuse.com",
                "auth": {"username": "public-key", "password": "{{ env.LANGFUSE_SECRET_KEY }}"},
            },
        },
    }
    output = sanitize_config_for_output(config)
    assert output["env"] == {"LANGFUSE_SECRET_KEY": "{{ env.ACTUAL_SECRET }}"}
    assert output["tracing"]["provider"]["auth"] == {
        "username": "public-key",
        "password": "{{ env.LANGFUSE_SECRET_KEY }}",
    }
    assert "private-value" not in json.dumps(output)
    assert config["env"]["ACTUAL_SECRET"] == "private-value"


# ---------------------------------------------------------------------------------------------
# sanitizeTracingConfigForPersistence
# ---------------------------------------------------------------------------------------------


def test_tracing_preserves_langfuse_references() -> None:
    source_provider = {
        "id": "langfuse",
        "endpoint": "https://cloud.langfuse.com",
        "auth": {
            "username": "{{ env.LANGFUSE_PUBLIC_KEY }}",
            "password": "{{ env.LANGFUSE_SECRET_KEY }}",
        },
    }
    source = {"tracing": {"enabled": True, "provider": source_provider}}
    rendered = {
        "tracing": {
            "enabled": True,
            "provider": {
                **source_provider,
                "auth": {"username": "pk-public", "password": "sk-private"},
            },
        }
    }
    preserve_tracing_credential_references(source, rendered)
    persisted = sanitize_tracing_config_for_persistence(rendered)
    assert persisted["tracing"]["provider"]["auth"] == {
        "username": "pk-public",
        "password": "{{ env.LANGFUSE_SECRET_KEY }}",
    }
    assert "sk-private" not in json.dumps(persisted)


def test_tracing_removes_transitive_credentials() -> None:
    provider = {
        "id": "tempo",
        "endpoint": "https://tempo.example.com",
        "auth": {"token": "{{ env.TEMPO_READER }}"},
    }
    source = {
        "env": {
            "TEMPO_SOURCE_SECRET": "private-tempo-secret",
            "TEMPO_INTERMEDIATE": "{{ env.TEMPO_SOURCE_SECRET }}",
            "TEMPO_READER": "{{ env.TEMPO_INTERMEDIATE }}",
            "REGION": "us-west-2",
        },
        "tracing": {"enabled": True, "provider": provider},
    }
    rendered = {
        "env": {
            "TEMPO_SOURCE_SECRET": "private-tempo-secret",
            "TEMPO_INTERMEDIATE": "private-tempo-secret",
            "TEMPO_READER": "private-tempo-secret",
            "REGION": "us-west-2",
        },
        "tracing": {
            "enabled": True,
            "provider": {**provider, "auth": {"token": "private-tempo-secret"}},
        },
    }
    preserve_tracing_credential_references(source, rendered)
    persisted = sanitize_tracing_config_for_persistence(rendered)
    assert persisted["tracing"]["provider"]["auth"]["token"] == "{{ env.TEMPO_READER }}"
    assert persisted["env"] == {
        "TEMPO_INTERMEDIATE": "{{ env.TEMPO_SOURCE_SECRET }}",
        "TEMPO_READER": "{{ env.TEMPO_INTERMEDIATE }}",
        "REGION": "us-west-2",
    }
    assert "private-tempo-secret" not in json.dumps(persisted)


def test_tracing_cyclic_env_references() -> None:
    config = {
        "env": {"TEMPO_READER": "{{ env.TEMPO_SOURCE }}", "TEMPO_SOURCE": "{{ env.TEMPO_READER }}"},
        "tracing": {
            "enabled": True,
            "provider": {
                "id": "tempo",
                "endpoint": "https://tempo.example.com",
                "auth": {"token": "{{ env.TEMPO_READER }}"},
            },
        },
    }
    assert sanitize_tracing_config_for_persistence(config)["env"] == config["env"]


def test_tracing_endpoint_credentials_and_headers() -> None:
    config = {
        "tracing": {
            "provider": {
                "id": "tempo",
                "endpoint": "https://user:pw@tempo.example.com/api?x=1#frag",
                "headers": {
                    "X-Tenant-Id": "acme",
                    "Authorization": "Bearer abc",
                    "X-Scope-OrgID": "{{ env.ORG }}",
                },
            }
        }
    }
    provider = sanitize_tracing_config_for_persistence(config)["tracing"]["provider"]
    assert provider["endpoint"] == "https://tempo.example.com/api"
    assert provider["headers"] == {"X-Tenant-Id": "acme", "X-Scope-OrgID": "{{ env.ORG }}"}
    assert sanitize_tracing_config_for_persistence({"a": 1}) == {"a": 1}


# ---------------------------------------------------------------------------------------------
# models/evalResult helpers
# ---------------------------------------------------------------------------------------------


class MockProvider(ApiProvider):
    def default_id(self) -> str:
        return "mock"


def test_sanitize_provider_shapes() -> None:
    api = MockProvider({"id": "test-provider", "label": "Test Provider", "config": {"apiKey": "k"}})
    api.prompts = ["ordinary runtime selector"]
    expected = {"id": "test-provider", "label": "Test Provider", "config": {"apiKey": REDACTED}}
    assert rs.sanitize_provider(api) == expected
    assert api.prompts == ["ordinary runtime selector"]
    options = {
        "id": "test-provider",
        "label": "Test Provider",
        "prompts": ["x"],
        "config": {"apiKey": "k"},
    }
    assert rs.sanitize_provider(options) == expected
    assert rs.sanitize_provider({**options, "id": lambda: "test-provider"}) == expected
    assert rs.sanitize_provider("openai:gpt-4o") == "openai:gpt-4o"


def test_sanitize_provider_redacts_templated_websocket_url() -> None:
    url = "ws://127.0.0.1/sessions/{{ sessionId }}?token=runtime-secret"
    provider = MockProvider({"id": url, "config": {"url": url, "messageTemplate": "{{ prompt }}"}})
    assert rs.sanitize_provider(provider) == {
        "id": url,  # upstream's WebSocketProvider sanitizes its own id; ours is verbatim
        "config": {
            "url": "ws://127.0.0.1/sessions/{{ sessionId }}?token=%5BREDACTED%5D",
            "messageTemplate": "{{ prompt }}",
        },
    }


def test_to_serializable_provider_ref_omits_runtime_state() -> None:
    provider = MockProvider({"id": "echo"})
    provider.transform = lambda output: "runtime transform"
    provider.inputs = {"query": "A short question"}
    provider.delay = 0
    provider.runtime_only = "ordinary working state"  # type: ignore[attr-defined]
    saved = rs.to_serializable_provider_ref(provider)
    assert saved == {
        "id": "echo",
        "config": {},
        "inputs": {"query": "A short question"},
        "delay": 0,
    }
    provider.inputs["query"] = "Runtime-only description"
    assert saved["inputs"]["query"] == "A short question"


def test_sanitize_for_db() -> None:
    obj: dict[str, Any] = {"a": 1, "fn": print}
    obj["self"] = obj
    assert rs.sanitize_for_db(obj) == {"a": 1}
    assert rs.sanitize_for_db(None) is None
    assert rs.sanitize_for_db([1, lambda: 1]) == [1, None]
    assert rs.sanitize_for_db_with_secrets({"provider": {"config": {"apiKey": "k"}}}) == {
        "provider": {"config": {"apiKey": REDACTED}}
    }


def test_redact_sensitive_result_fields_for_db() -> None:
    response = {
        "output": {"password": "model output should stay intact"},
        "metadata": {
            "http": {
                "status": 200,
                "headers": {
                    "content-type": "application/json",
                    "openai-project": "proj",
                    "set-cookie": "session=secret",
                    "x-ratelimit-remaining-requests": "199",
                    "X-Request-Id": "req",
                },
                "requestHeaders": {
                    "authorization": "Bearer sk",
                    "api-key": "azure",
                    "X-API-Key": "custom",
                    "x-safe-debug": "keep-me",
                },
            },
            "headers": {"x-api-key": "legacy", "x-other": "ok"},
        },
    }
    grading = {
        "pass": True,
        "componentResults": [
            {"pass": True, "metadata": {"http": {"headers": {"set-cookie": "g", "ok": "1"}}}}
        ],
        "assertion": {"type": "llm-rubric", "provider": {"id": "x", "config": {"apiKey": "k"}}},
    }
    metadata = {
        "http": {"headers": {"openai-project": "meta"}},
        "headers": {"x-api-key": "legacy", "x-user": "u"},
    }
    out = rs.redact_sensitive_result_fields_for_db(
        response=response, grading_result=grading, metadata=metadata
    )
    http = out["response"]["metadata"]["http"]
    assert out["response"]["output"] == response["output"]
    assert http["headers"] == {
        "content-type": "application/json",
        "openai-project": REDACTED,
        "set-cookie": REDACTED,
        "x-ratelimit-remaining-requests": REDACTED,
        "X-Request-Id": REDACTED,
    }
    assert http["requestHeaders"] == {
        "authorization": REDACTED,
        "api-key": REDACTED,
        "X-API-Key": REDACTED,
        "x-safe-debug": "keep-me",
    }
    assert out["response"]["metadata"]["headers"] == {"x-api-key": REDACTED, "x-other": "ok"}
    assert out["metadata"]["http"]["headers"] == {"openai-project": REDACTED}
    # Legacy result-level headers are redacted only where they echo the transport.
    assert out["metadata"]["headers"] == {"x-api-key": REDACTED, "x-user": "u"}
    assert out["gradingResult"]["componentResults"][0]["metadata"]["http"]["headers"] == {
        "set-cookie": REDACTED,
        "ok": "1",
    }
    assert out["gradingResult"]["assertion"]["provider"]["config"]["apiKey"] == REDACTED
    assert response["metadata"]["http"]["headers"]["set-cookie"] == "session=secret"
    # Unchanged inputs keep their identity.
    assert rs.sanitize_response_for_db({"output": "x"}) == {"output": "x"}
    assert rs.sanitize_metadata_for_db({"headers": {"cookie": "c"}}) == {"headers": {"cookie": "c"}}


def test_trace_linkage_round_trip(caplog: pytest.LogCaptureFixture) -> None:
    md = rs.persist_trace_metadata({"user": 1}, "trace-1", "eval-1")
    assert md == {
        "user": 1,
        "__promptfoo": {"traceLinkage": {"traceId": "trace-1", "evaluationId": "eval-1"}},
    }
    assert rs.surface_trace_metadata(md) == {
        "traceId": "trace-1",
        "evaluationId": "eval-1",
        "metadata": {"user": 1},
    }
    assert rs.persist_trace_metadata(md, None, None) == {"user": 1}
    assert rs.persist_trace_metadata(None, None, "eval-only") == {
        "__promptfoo": {"traceLinkage": {"evaluationId": "eval-only"}}
    }
    with caplog.at_level("WARNING"):
        out = rs.persist_trace_metadata({"__promptfoo": "bad"}, "t", None)
    assert out == {"__promptfoo": {"traceLinkage": {"traceId": "t"}}}
    assert "Overwriting non-object metadata.__promptfoo" in caplog.text
    malformed = {"__promptfoo": {"traceLinkage": {"traceId": 5}, "other": 1}}
    assert rs.surface_trace_metadata(malformed) == {"metadata": {"__promptfoo": {"other": 1}}}
    assert rs.surface_trace_metadata(None) == {"metadata": {}}


def test_repeat_linkage_round_trip() -> None:
    md = rs.persist_repeat_metadata({"a": 1}, 0, "group")
    assert md == {
        "a": 1,
        "__promptfoo": {"repeatLinkage": {"repeatIndex": 0, "repeatGroupId": "group"}},
    }
    assert rs.surface_repeat_metadata(md) == {
        "repeatIndex": 0,
        "repeatGroupId": "group",
        "metadata": {"a": 1},
    }
    assert rs.persist_repeat_metadata(md, None, "group") == {"a": 1}
    assert rs.strip_repeat_linkage_from_metadata({"a": 1}) == {"a": 1}
    bad = {"__promptfoo": {"repeatLinkage": {"repeatIndex": -1, "repeatGroupId": ""}}}
    assert rs.surface_repeat_metadata(bad) == {"metadata": {}}


def test_is_result_failure_reason() -> None:
    assert rs.is_result_failure_reason(2)
    assert not rs.is_result_failure_reason(3)
    assert not rs.is_result_failure_reason(True)


def test_get_strip_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMPTFOO_STRIP_METADATA", "true")
    flags = rs.get_strip_flags({"PROMPTFOO_STRIP_PROMPT_TEXT": True})
    assert flags == {
        "shouldStripPromptText": True,
        "shouldStripResponseOutput": False,
        "shouldStripTestVars": False,
        "shouldStripGradingResult": False,
        "shouldStripMetadata": True,
    }
    assert rs.get_strip_flags({"PROMPTFOO_STRIP_METADATA": "false"})["shouldStripMetadata"] is False


@pytest.mark.parametrize("mapped", [True, False])
def test_project_prompt_preserves_live_provider_serialization(mapped: bool) -> None:
    provider = LiveProvider(
        prompts=["ordinary selector"],
        config={"nested": {"one": {"two": {"three": {"value": "public setting"}}}}},
    )
    prompt = {
        "raw": "Hello",
        "label": "Hello",
        "template": "Hello",
        "config": {"provider": {"text": provider} if mapped else provider},
    }
    projected = rs.project_prompt(prompt, True)
    serialized = (
        projected["config"]["provider"]["text"] if mapped else projected["config"]["provider"]
    )
    assert serialized == {"id": "echo", "config": provider.config}
    assert projected["raw"] == "[prompt stripped]" and "template" not in projected
    assert rs.project_prompt(prompt, False) is prompt


@pytest.mark.parametrize("strip", [True, False])
@pytest.mark.parametrize("conversation_output", ["native-conversation-text", [{"text": "t"}]])
def test_jsonl_projects_native_conversation_output(strip: bool, conversation_output: Any) -> None:
    response = {"output": "display output", "conversationOutput": conversation_output}
    inp = {"promptIdx": 0, "testIdx": 0, "response": response, "success": True}
    flags = rs.get_strip_flags({"PROMPTFOO_STRIP_RESPONSE_OUTPUT": str(strip).lower()})
    projected = rs.sanitize_result_for_jsonl_artifact(inp, flags)
    if strip:
        assert projected["response"] == {"output": "[output stripped]"}
    else:
        assert projected["response"] == response
    assert inp["response"] is response


@pytest.mark.parametrize("strip", [True, False])
@pytest.mark.parametrize("mapped", [True, False])
def test_jsonl_projects_live_grading_references(strip: bool, mapped: bool) -> None:
    provider = {
        "id": "echo",
        "prompts": ["ordinary nested selector"],
        "config": {"prompts": ["ordinary provider configuration"]},
    }
    live_id = "https://grader.example/run?mode=ordinary"
    live = LiveProvider(
        live_id,
        prompts=list(provider["prompts"]),
        config=provider["config"],
        transform="output",
        delay=0,
        inputs={"query": "A short question"},
        runtime_only="ordinary working state",
    )
    provider_map = {
        "text": live,
        "embedding": live,
        "classification": live,
        "moderation": live,
        "label": "ordinary map label",
        "config": {"prompts": ["ordinary map configuration"]},
        "prompts": ["ordinary application setting"],
        "application": {"text": {"prompts": ["ordinary application payload"]}},
    }
    grading_provider = provider_map if mapped else live
    assertion = {"type": "equals", "value": "ok", "provider": grading_provider}
    test_case = {
        "provider": live,
        "options": {"provider": grading_provider},
        "assert": [{"type": "assert-set", "assert": [assertion]}],
        "metadata": {"prompts": ["ordinary metadata"]},
    }
    inp = {
        "promptIdx": 0,
        "testIdx": 0,
        "provider": provider,
        "testCase": test_case,
        "prompt": {
            "raw": "Hello",
            "label": "Greeting",
            "config": {
                "provider": grading_provider,
                "options": {"provider": {"prompts": ["ordinary application options"]}},
                "assert": [{"provider": {"prompts": ["ordinary application assertions"]}}],
            },
        },
        "gradingResult": {
            "pass": True,
            "score": 1,
            "reason": "ok",
            "assertion": assertion,
            "componentResults": [
                {"pass": True, "score": 1, "reason": "ok", "assertion": assertion}
            ],
            "metadata": {"prompts": ["ordinary grading metadata"]},
        },
    }
    flags = rs.get_strip_flags({"PROMPTFOO_STRIP_PROMPT_TEXT": str(strip).lower()})
    projected = rs.sanitize_result_for_jsonl_artifact(inp, flags)

    text = json.dumps(projected)
    assert ("ordinary nested selector" in text) is (not strip)
    assert projected["testCase"]["metadata"] == test_case["metadata"]
    expected_provider: dict[str, Any] = {
        "id": live_id,
        "config": provider["config"],
        "transform": "output",
        "delay": 0,
        "inputs": {"query": "A short question"},
    }
    if not strip:
        expected_provider["prompts"] = provider["prompts"]
    expected_map = {
        **provider_map,
        "text": expected_provider,
        "embedding": expected_provider,
        "classification": expected_provider,
        "moderation": expected_provider,
    }
    expected_ref = expected_map if mapped else expected_provider
    assert projected["testCase"]["provider"] == expected_provider
    assert projected["testCase"]["options"]["provider"] == expected_ref
    assert projected["prompt"]["config"]["provider"] == expected_ref
    assert projected["gradingResult"]["assertion"]["provider"] == expected_ref
    assert (
        projected["gradingResult"]["componentResults"][0]["assertion"]["provider"] == expected_ref
    )
    assert "ordinary working state" not in text
    assert projected["prompt"]["config"]["options"] == inp["prompt"]["config"]["options"]
    assert projected["gradingResult"]["metadata"] == {"prompts": ["ordinary grading metadata"]}
    assert test_case["options"]["provider"] is grading_provider
    assert provider_map["text"] is live


def test_jsonl_redacts_grader_settings_and_strips() -> None:
    credential = "fixture-grader-credential"
    assertion = {"type": "equals", "value": '{ "token": "test input",  "number": 1 }'}
    grading = {
        "pass": True,
        "assertion": assertion,
        "componentResults": [
            {
                "pass": True,
                "assertion": {
                    **assertion,
                    "provider": {
                        "id": f"https://grader.example/check?api_key={credential}",
                        "config": {"apiKey": credential},
                    },
                    "config": {"headers": {"Authorization": credential}},
                },
            }
        ],
    }
    inp = {
        "testCase": {"vars": {"a": "b"}},
        "vars": {"a": "b"},
        "gradingResult": grading,
        "metadata": {"m": 1},
    }
    artifact = rs.sanitize_result_for_jsonl_artifact(inp, rs.get_strip_flags({}))
    assert artifact["gradingResult"]["assertion"]["value"] == assertion["value"]
    assert credential not in json.dumps(artifact["gradingResult"])
    assert grading["componentResults"][0]["assertion"]["provider"]["config"]["apiKey"] == credential
    stripped = rs.sanitize_result_for_jsonl_artifact(
        inp,
        {
            "shouldStripTestVars": True,
            "shouldStripGradingResult": True,
            "shouldStripMetadata": True,
        },
    )
    assert stripped["vars"] == {}
    assert "vars" not in stripped["testCase"]
    assert stripped["gradingResult"] is None
    assert stripped["metadata"] == {}
    assert "response" not in stripped and "namedScores" not in stripped


def test_project_output_metadata_strips_media() -> None:
    response_md = {"audio": {"data": "x"}, "image": "data:image/png;base64,AAAA", "n": 1}
    md = {**response_md, "user": "keep"}
    projected = rs.project_output_metadata(md, True, response_md)
    assert projected == {"image": "[output stripped]", "n": 1, "user": "keep"}
    blob = {"uri": "promptfoo://blob/" + "a" * 64}
    assert rs.project_output_metadata({"b": blob}, True, {"b": blob}) == {"b": "[output stripped]"}
