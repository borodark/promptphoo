"""Ports test/tracing/providers/{index,tempo,langfuse,braintrust}.test.ts.

Upstream mocks fetchWithProxy; here requests go through promptphoo's fetch stack and httpx
is mocked with respx, so the assertions read the recorded httpx requests.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from promptphoo.tracing.providers import (
    TraceProviderError,
    create_trace_provider,
    is_external_trace_provider,
)
from promptphoo.tracing.providers.braintrust import BraintrustProvider
from promptphoo.tracing.providers.fetch import (
    has_trace_credential_path,
    validate_trace_provider_endpoint,
)
from promptphoo.tracing.providers.langfuse import LangfuseProvider
from promptphoo.tracing.providers.tempo import TempoProvider
from promptphoo.tracing.span_filter import is_relevant_span
from promptphoo.tracing.tool_attributes import get_tool_name_from_attributes

TRACE_ID = "0123456789abcdef0123456789abcdef"
PROJECT_ID = "12345678-1234-4123-8123-123456789abc"
MAX_BYTES = 10 * 1024 * 1024


def json_response(data: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, content=json.dumps(data).encode(), headers={"content-type": "application/json"}
    )


def contains(actual: Any, expected: Any) -> bool:
    """vitest toMatchObject: recursive subset match for dicts."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and contains(actual[k], v) for k, v in expected.items()
        )
    return actual == expected


@pytest.fixture
def mock_router():
    with respx.mock(assert_all_called=False) as router:
        yield router


def query(request: httpx.Request) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(str(request.url)).query).items()}


# --- index.test.ts --------------------------------------------------------------------------


def test_creates_braintrust_provider():
    provider = create_trace_provider(
        {
            "id": "braintrust",
            "endpoint": "https://api.braintrust.dev",
            "projectId": PROJECT_ID,
            "auth": {"token": "test-token"},
        }
    )
    assert isinstance(provider, BraintrustProvider)
    assert provider.id == "braintrust"


def test_creates_langfuse_provider():
    provider = create_trace_provider(
        {
            "id": "langfuse",
            "endpoint": "https://cloud.langfuse.com",
            "auth": {"username": "public-key", "password": "secret-key"},
        }
    )
    assert isinstance(provider, LangfuseProvider)
    assert provider.id == "langfuse"


def test_creates_tempo_provider():
    provider = create_trace_provider({"id": "tempo", "endpoint": "http://tempo:3200"})
    assert isinstance(provider, TempoProvider)
    assert provider.id == "tempo"


def test_create_throws_for_local():
    with pytest.raises(ValueError, match="Local provider id should use TraceStore directly"):
        create_trace_provider({"id": "local"})


def test_create_throws_for_unknown():
    with pytest.raises(ValueError, match="Unknown trace provider id: unknown"):
        create_trace_provider({"id": "unknown", "endpoint": "http://unknown:3200"})


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, False),
        ({"id": "local", "endpoint": "http://localhost:3200"}, False),
        ({"id": "tempo"}, False),
        ({"id": "tempo", "endpoint": "http://tempo:3200"}, True),
        ({"id": "jaeger", "endpoint": "http://jaeger:16686"}, True),
    ],
)
def test_is_external_trace_provider(config, expected):
    assert is_external_trace_provider(config) is expected


# --- fetch.ts / contracts -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pathname", "expected"),
    [
        ("/tempo", False),
        ("/team-west/", False),
        ("/token-privateTenantCredential123", True),
        ("/" + "a" * 32, True),
        ("/12345678-1234-4123-8123-123456789abc", True),
        ("/%E0%A4%A", True),  # malformed escape
    ],
)
def test_has_trace_credential_path(pathname, expected):
    assert has_trace_credential_path(pathname) is expected


def test_validate_endpoint_messages():
    with pytest.raises(ValueError, match="^Tempo provider endpoint must be a valid HTTP"):
        validate_trace_provider_endpoint("not-a-url", "Tempo")
    with pytest.raises(ValueError, match="without credentials, query parameters, or fragments"):
        validate_trace_provider_endpoint("https://example.com?x=1", "Tempo")
    validate_trace_provider_endpoint("https://example.com/base/", "Tempo")


# --- tempo.test.ts --------------------------------------------------------------------------

TEMPO_RESPONSE: dict[str, Any] = {
    "batches": [
        {
            "resource": {
                "attributes": [{"key": "service.name", "value": {"stringValue": "target-service"}}]
            },
            "scopeSpans": [
                {
                    "scope": {"name": "instrumentation"},
                    "spans": [
                        {
                            "traceId": TRACE_ID,
                            "spanId": "0123456789abcdef",
                            "name": "target.call",
                            "kind": 3,
                            "startTimeUnixNano": "1704067200000000000",
                            "endTimeUnixNano": "1704067201000000000",
                            "attributes": [
                                {"key": "gen_ai.usage.total_tokens", "value": {"intValue": "42"}},
                                {
                                    "key": "nested",
                                    "value": {
                                        "kvlistValue": {
                                            "values": [
                                                {"key": "enabled", "value": {"boolValue": True}}
                                            ]
                                        }
                                    },
                                },
                            ],
                            "status": {"code": "STATUS_CODE_OK"},
                        },
                        {
                            "traceId": TRACE_ID,
                            "spanId": "1123456789abcdef",
                            "parentSpanId": "0123456789abcdef",
                            "name": "internal.setup",
                            "kind": "SPAN_KIND_INTERNAL",
                            "startTimeUnixNano": "1704067200100000000",
                        },
                    ],
                }
            ],
        }
    ]
}
TEMPO_URL = f"http://tempo:3200/api/traces/{TRACE_ID}"


@pytest.fixture
def tempo(mock_router):
    return mock_router.get(TEMPO_URL).mock(return_value=json_response(TEMPO_RESPONSE))


@pytest.mark.parametrize(
    "config",
    [
        {"id": "tempo"},
        {"id": "tempo", "endpoint": "not-a-url"},
        {"id": "tempo", "endpoint": "file:///tmp/traces"},
        {"id": "tempo", "endpoint": "https://user:secret@example.com"},
        {"id": "tempo", "endpoint": "https://example.com/tempo?token=secret"},
        {"id": "tempo", "endpoint": "https://example.com/tempo#section"},
        {"id": "tempo", "endpoint": "https://example.com", "timeout": -1},
    ],
)
def test_tempo_rejects_invalid_configuration(config):
    with pytest.raises(ValueError):
        TempoProvider(config)


async def test_tempo_fetches_and_normalizes_spans(tempo):
    provider = TempoProvider({"id": "tempo", "endpoint": "http://tempo:3200/"})
    result = await provider.fetch_trace(TRACE_ID)
    assert result is not None
    spans = result["spans"]
    assert len(spans) == 2
    assert contains(
        spans[0],
        {
            "spanId": "0123456789abcdef",
            "name": "target.call",
            "startTime": 1704067200000,
            "endTime": 1704067201000,
            "statusCode": 1,
            "attributes": {
                "service.name": "target-service",
                "otel.scope.name": "instrumentation",
                "otel.span.kind": "client",
                "otel.span.kind_code": 3,
                "gen_ai.usage.total_tokens": 42,
                "nested": {"enabled": True},
            },
        },
    )
    assert contains(
        spans[1],
        {
            "spanId": "1123456789abcdef",
            "parentSpanId": "0123456789abcdef",
            "attributes": {"otel.span.kind": "internal"},
        },
    )
    assert "endTime" not in spans[1] and "statusCode" not in spans[1]
    assert tempo.call_count == 1
    request = tempo.calls[0].request
    assert str(request.url) == TEMPO_URL
    assert request.method == "GET"
    assert isinstance(result["fetchedAt"], int)


async def test_tempo_accepts_base64_span_ids(mock_router):
    encoded = copy.deepcopy(TEMPO_RESPONSE)
    for span in encoded["batches"][0]["scopeSpans"][0]["spans"]:
        span["spanId"] = base64.b64encode(bytes.fromhex(span["spanId"])).decode()
        if span.get("parentSpanId"):
            span["parentSpanId"] = base64.b64encode(bytes.fromhex(span["parentSpanId"])).decode()
    mock_router.get(TEMPO_URL).mock(return_value=json_response(encoded))
    result = await TempoProvider({"id": "tempo", "endpoint": "http://tempo:3200"}).fetch_trace(
        TRACE_ID
    )
    assert result is not None
    assert [s["spanId"] for s in result["spans"]] == ["0123456789abcdef", "1123456789abcdef"]


@pytest.mark.parametrize("trace_id", ["../../admin", "abc123", "0" * 32])
async def test_tempo_rejects_invalid_trace_ids(trace_id, tempo):
    provider = TempoProvider({"id": "tempo", "endpoint": "http://tempo:3200"})
    with pytest.raises(TraceProviderError):
        await provider.fetch_trace(trace_id)
    assert tempo.call_count == 0


async def test_tempo_drops_malformed_spans(mock_router, caplog):
    mixed = copy.deepcopy(TEMPO_RESPONSE)
    spans = mixed["batches"][0]["scopeSpans"][0]["spans"]
    spans[0:0] = [
        {**spans[0], "spanId": "!!!", "name": "malformed.span"},
        {**spans[0], "spanId": "2123456789abcdef", "traceId": "f" * 32},
        {**spans[1], "spanId": "3123456789abcdef", "parentSpanId": "!!!"},
    ]
    mock_router.get(TEMPO_URL).mock(return_value=json_response(mixed))
    with caplog.at_level(logging.WARNING):
        result = await TempoProvider({"id": "tempo", "endpoint": "http://tempo:3200"}).fetch_trace(
            TRACE_ID
        )
    assert result is not None
    assert [s["name"] for s in result["spans"]] == ["target.call", "internal.setup"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["[TempoProvider] Skipped 3 malformed spans"]


async def test_tempo_skips_malformed_batches_and_scopes(mock_router, caplog):
    valid = copy.deepcopy(TEMPO_RESPONSE["batches"][0])
    payload = {
        "batches": [
            None,
            {"scopeSpans": {}},
            {**valid, "scopeSpans": [None, {"spans": {}}, *valid["scopeSpans"]]},
        ]
    }
    mock_router.get(TEMPO_URL).mock(return_value=json_response(payload))
    with caplog.at_level(logging.WARNING):
        result = await TempoProvider({"id": "tempo", "endpoint": "http://tempo:3200"}).fetch_trace(
            TRACE_ID
        )
    assert result is not None
    assert [s["name"] for s in result["spans"]] == ["target.call", "internal.setup"]
    assert result["spans"][0]["attributes"]["service.name"] == "target-service"
    assert "[TempoProvider] Skipped 4 malformed spans" in caplog.text


class _Signal:
    def __init__(self, aborted: bool = False) -> None:
        self.aborted = aborted
        self.reason = None


async def test_tempo_forwards_bearer_auth_and_tenant_headers(tempo):
    provider = TempoProvider(
        {
            "id": "tempo",
            "endpoint": "http://tempo:3200",
            "auth": {"token": "secret-token"},
            "headers": {"X-Scope-OrgID": "tenant-a"},
            "timeout": 250,
        }
    )
    await provider.fetch_trace(TRACE_ID, {"abortSignal": _Signal()})
    headers = tempo.calls[0].request.headers
    assert headers["Authorization"] == "Bearer secret-token"
    assert headers["X-Scope-OrgID"] == "tenant-a"
    assert headers["Accept"] == "application/json"


async def test_tempo_aborted_signal_skips_request(tempo):
    provider = TempoProvider({"id": "tempo", "endpoint": "http://tempo:3200"})
    with pytest.raises(Exception, match="aborted"):
        await provider.fetch_trace(TRACE_ID, {"abortSignal": _Signal(aborted=True)})
    assert tempo.call_count == 0


async def test_tempo_basic_auth(tempo):
    provider = TempoProvider(
        {
            "id": "tempo",
            "endpoint": "http://tempo:3200",
            "auth": {"username": "user", "password": "pass"},
        }
    )
    await provider.fetch_trace(TRACE_ID)
    assert tempo.calls[0].request.headers["Authorization"] == "Basic dXNlcjpwYXNz"


async def test_tempo_classifies_http_responses(mock_router):
    provider = TempoProvider({"id": "tempo", "endpoint": "http://tempo:3200"})
    route = mock_router.get(TEMPO_URL)
    route.mock(return_value=json_response({}, 404))
    assert await provider.fetch_trace(TRACE_ID) is None

    route.mock(return_value=json_response({}, 401))
    with pytest.raises(TraceProviderError) as e401:
        await provider.fetch_trace(TRACE_ID)
    assert (e401.value.status_code, e401.value.retryable) == (401, False)

    route.mock(return_value=json_response({}, 503))
    with pytest.raises(TraceProviderError) as e503:
        await provider.fetch_trace(TRACE_ID)
    assert (e503.value.status_code, e503.value.retryable) == (503, True)


async def test_tempo_rejects_invalid_or_oversized_responses(mock_router):
    provider = TempoProvider({"id": "tempo", "endpoint": "http://tempo:3200"})
    route = mock_router.get(TEMPO_URL)
    route.mock(return_value=json_response({"unexpected": []}))
    with pytest.raises(TraceProviderError, match="invalid trace response"):
        await provider.fetch_trace(TRACE_ID)

    route.mock(
        return_value=httpx.Response(200, content=b"{}", headers={"content-length": "10485761"})
    )
    with pytest.raises(TraceProviderError, match="maximum response size"):
        await provider.fetch_trace(TRACE_ID)


async def test_tempo_rejects_oversized_bodies(mock_router):
    mock_router.get(TEMPO_URL).mock(
        return_value=httpx.Response(200, content=b" " * (MAX_BYTES + 1))
    )
    provider = TempoProvider({"id": "tempo", "endpoint": "http://tempo:3200"})
    with pytest.raises(TraceProviderError, match="maximum response size"):
        await provider.fetch_trace(TRACE_ID)


async def test_tempo_rejects_redirects(mock_router):
    mock_router.get(TEMPO_URL).mock(
        return_value=httpx.Response(302, headers={"location": "http://elsewhere/x"})
    )
    mock_router.get("http://elsewhere/x").mock(return_value=json_response(TEMPO_RESPONSE))
    provider = TempoProvider({"id": "tempo", "endpoint": "http://tempo:3200"})
    with pytest.raises(Exception, match="fetch failed"):
        await provider.fetch_trace(TRACE_ID)


# --- langfuse.test.ts -----------------------------------------------------------------------

LANGFUSE_CONFIG: dict[str, Any] = {
    "id": "langfuse",
    "endpoint": "https://cloud.langfuse.com",
    "auth": {"username": "pk-public", "password": "sk-secret"},
}
LANGFUSE_URL = "https://cloud.langfuse.com/api/public/v2/observations"
LANGFUSE_AUTH = "Basic " + base64.b64encode(b"pk-public:sk-secret").decode()

OBSERVATIONS: list[dict[str, Any]] = [
    {
        "id": "root-span",
        "traceId": TRACE_ID,
        "parentObservationId": None,
        "name": "customer.request",
        "type": "SPAN",
        "startTime": "2024-01-01T00:00:00.000Z",
        "endTime": "2024-01-01T00:00:01.000Z",
        "level": "DEFAULT",
        "metadata": {"service.name": "customer-agent", "tenant": "team-west"},
        "input": '{"prompt":"hello"}',
        "output": '{"text":"world"}',
    },
    {
        "id": "generation-span",
        "traceId": TRACE_ID,
        "parentObservationId": "root-span",
        "name": "chat gpt-4o",
        "type": "GENERATION",
        "startTime": "2024-01-01T00:00:00.100Z",
        "endTime": "2024-01-01T00:00:00.500Z",
        "level": "ERROR",
        "statusMessage": "model unavailable",
        "providedModelName": "gpt-4o",
        "modelParameters": {"temperature": 0.2},
        "usageDetails": {"input": 98, "output": 68, "total": 166},
        "inputUsage": 98,
        "outputUsage": 68,
        "totalUsage": 166,
        "inputCost": 0.00049,
        "outputCost": 0.00204,
        "totalCost": 0.00253,
    },
]


@pytest.fixture
def langfuse(mock_router):
    return mock_router.get(url__startswith=LANGFUSE_URL).mock(
        return_value=json_response({"data": OBSERVATIONS, "meta": {}})
    )


async def fetch_langfuse(options: dict[str, Any] | None = None, **config: Any):
    return await LangfuseProvider({**LANGFUSE_CONFIG, **config}).fetch_trace(TRACE_ID, options)


@pytest.mark.parametrize(
    "value",
    [
        {"id": "langfuse"},
        {**LANGFUSE_CONFIG, "endpoint": "not-a-url"},
        {**LANGFUSE_CONFIG, "endpoint": "file:///tmp/traces"},
        {**LANGFUSE_CONFIG, "endpoint": "https://user:secret@langfuse.example.com"},
        {**LANGFUSE_CONFIG, "endpoint": "https://langfuse.example.com?token=secret"},
        {**LANGFUSE_CONFIG, "endpoint": "https://langfuse.example.com#secret"},
        {
            **LANGFUSE_CONFIG,
            "endpoint": "https://langfuse.example.com/token-privateTenantCredential123",
        },
        {**LANGFUSE_CONFIG, "auth": {}},
        {**LANGFUSE_CONFIG, "auth": {"username": "public-key"}},
        {**LANGFUSE_CONFIG, "auth": {"password": "secret-key"}},
        {**LANGFUSE_CONFIG, "auth": {**LANGFUSE_CONFIG["auth"], "token": "bearer-token"}},
        {**LANGFUSE_CONFIG, "timeout": -1},
    ],
)
def test_langfuse_rejects_unsafe_configuration(value):
    with pytest.raises(ValueError):
        LangfuseProvider(value)


@pytest.mark.parametrize("trace_id", ["../../admin", "abc123", "0" * 32])
async def test_langfuse_rejects_invalid_trace_ids(trace_id, langfuse):
    with pytest.raises(TraceProviderError):
        await LangfuseProvider(LANGFUSE_CONFIG).fetch_trace(trace_id)
    assert langfuse.call_count == 0


async def test_langfuse_retrieves_observations(langfuse):
    result = await fetch_langfuse()
    assert result is not None
    spans = result["spans"]
    assert len(spans) == 2
    assert contains(
        spans[0],
        {
            "spanId": "root-span",
            "name": "customer.request",
            "startTime": 1704067200000,
            "endTime": 1704067201000,
            "statusCode": 1,
            "attributes": {
                "service.name": "customer-agent",
                "tenant": "team-west",
                "langfuse.observation.type": "SPAN",
                "langfuse.input": {"prompt": "hello"},
                "langfuse.output": {"text": "world"},
            },
        },
    )
    assert "parentSpanId" not in spans[0]
    assert contains(
        spans[1],
        {
            "parentSpanId": "root-span",
            "name": "chat gpt-4o",
            "statusCode": 2,
            "statusMessage": "model unavailable",
            "attributes": {
                "langfuse.observation.type": "GENERATION",
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.request.temperature": 0.2,
                "langfuse.model.parameters": {"temperature": 0.2},
                "langfuse.usage": {"input": 98, "output": 68, "total": 166},
                "gen_ai.usage.input_tokens": 98,
                "gen_ai.usage.output_tokens": 68,
                "langfuse.usage.total_tokens": 166,
                "promptfoo.usage.total_tokens": 166,
                "langfuse.cost.input": 0.00049,
                "langfuse.cost.output": 0.00204,
                "langfuse.cost.total": 0.00253,
            },
        },
    )
    request = langfuse.calls[0].request
    assert urlsplit(str(request.url)).path == "/api/public/v2/observations"
    params = query(request)
    assert params["traceId"] == TRACE_ID
    assert params["fields"] == "core,basic,io,metadata,model,usage"
    assert request.method == "GET"
    assert request.headers["Authorization"] == LANGFUSE_AUTH


async def test_langfuse_reverse_proxy_paths_and_protected_headers(mock_router):
    route = mock_router.get(url__startswith="https://langfuse.example.com/").mock(
        return_value=json_response({"data": OBSERVATIONS, "meta": {}})
    )
    await LangfuseProvider(
        {
            **LANGFUSE_CONFIG,
            "endpoint": "https://langfuse.example.com/team-west/",
            "headers": {
                "authorization": "Bearer unexpected",
                "ACCEPT": "text/plain",
                "X-Tenant": "west",
            },
        }
    ).fetch_trace(TRACE_ID)
    request = route.calls[0].request
    assert urlsplit(str(request.url)).path == "/team-west/api/public/v2/observations"
    assert request.headers["X-Tenant"] == "west"
    assert request.headers.get_list("Accept") == ["application/json"]
    assert request.headers.get_list("Authorization") == [LANGFUSE_AUTH]


async def test_langfuse_normalizes_tool_observations(langfuse):
    langfuse.mock(
        return_value=json_response(
            {
                "data": [
                    {
                        "id": "tool-span",
                        "traceId": TRACE_ID,
                        "name": "search",
                        "type": "TOOL",
                        "startTime": "2024-01-01T00:00:00.000Z",
                        "input": '{"query":"customer orders"}',
                        "output": '{"count":3}',
                    }
                ]
            }
        )
    )
    result = await fetch_langfuse()
    assert result is not None
    span = result["spans"][0]
    assert contains(
        span["attributes"],
        {
            "langfuse.observation.type": "TOOL",
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": "search",
            "gen_ai.tool.call.arguments": {"query": "customer orders"},
            "gen_ai.tool.call.result": {"count": 3},
            "tool.name": "search",
            "tool.arguments": '{"query":"customer orders"}',
        },
    )
    assert get_tool_name_from_attributes(span["attributes"]) == "search"
    assert is_relevant_span(span) is True


@pytest.mark.parametrize(
    ("obs_type", "operation"),
    [
        ("GENERATION", "chat"),
        ("EMBEDDING", "embeddings"),
        ("TOOL", "execute_tool"),
        ("AGENT", "invoke_agent"),
        ("CHAIN", "invoke_workflow"),
        ("RETRIEVER", "retrieval"),
    ],
)
async def test_langfuse_maps_operations(obs_type, operation, langfuse):
    langfuse.mock(
        return_value=json_response(
            {
                "data": [
                    {
                        **OBSERVATIONS[0],
                        "id": f"{obs_type}-span",
                        "name": f"{obs_type} action",
                        "type": obs_type,
                    }
                ]
            }
        )
    )
    result = await fetch_langfuse()
    assert result is not None
    span = result["spans"][0]
    assert contains(
        span["attributes"],
        {"langfuse.observation.type": obs_type, "gen_ai.operation.name": operation},
    )
    assert is_relevant_span(span) is True


@pytest.mark.parametrize(
    ("name", "operation"),
    [
        ("text_completion gpt-4o", "text_completion"),
        ("text-completion gpt-4o", "text_completion"),
        ("generate_content gemini", "generate_content"),
    ],
)
async def test_langfuse_infers_generation_operation(name, operation, langfuse):
    langfuse.mock(return_value=json_response({"data": [{**OBSERVATIONS[1], "name": name}]}))
    result = await fetch_langfuse()
    assert result is not None
    assert result["spans"][0]["attributes"]["gen_ai.operation.name"] == operation


@pytest.mark.parametrize(
    ("obs_type", "attribute"),
    [
        ("AGENT", "gen_ai.agent.name"),
        ("CHAIN", "gen_ai.workflow.name"),
        ("EVALUATOR", "gen_ai.evaluation.name"),
        ("GUARDRAIL", "guardrail.name"),
    ],
)
async def test_langfuse_normalizes_observation_names(obs_type, attribute, langfuse):
    langfuse.mock(
        return_value=json_response(
            {
                "data": [
                    {
                        **OBSERVATIONS[0],
                        "id": f"{obs_type}-span",
                        "name": "customer-action",
                        "type": obs_type,
                    }
                ]
            }
        )
    )
    result = await fetch_langfuse()
    assert result is not None
    span = result["spans"][0]
    assert span["attributes"][attribute] == "customer-action"
    assert is_relevant_span(span) is True


async def test_langfuse_incomplete_generation_stays_relevant(langfuse):
    langfuse.mock(
        return_value=json_response(
            {
                "data": [
                    {
                        "id": "incomplete-generation",
                        "traceId": TRACE_ID,
                        "name": "customer-model",
                        "type": "GENERATION",
                        "startTime": "2024-01-01T00:00:00.000Z",
                        "input": '{"prompt":"hello"}',
                        "output": '{"text":"world"}',
                    }
                ]
            }
        )
    )
    result = await fetch_langfuse()
    assert result is not None
    span = result["spans"][0]
    assert contains(
        span["attributes"],
        {
            "gen_ai.operation.name": "chat",
            "langfuse.input": {"prompt": "hello"},
            "langfuse.output": {"text": "world"},
        },
    )
    assert is_relevant_span(span) is True


async def test_langfuse_restores_original_otel_attributes(langfuse):
    langfuse.mock(
        return_value=json_response(
            {
                "data": [
                    {
                        **OBSERVATIONS[1],
                        "metadata": {
                            "tenant": "team-west",
                            "attributes": {
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.provider.name": "gcp.gemini",
                                "gen_ai.request.model": "original-model",
                                "gen_ai.request.temperature": 0.7,
                                "http.request.method": "POST",
                            },
                            "resourceAttributes": {
                                "service.name": "customer-agent",
                                "deployment.environment.name": "production",
                            },
                        },
                    }
                ]
            }
        )
    )
    result = await fetch_langfuse()
    assert result is not None
    assert contains(
        result["spans"][0]["attributes"],
        {
            "tenant": "team-west",
            "service.name": "customer-agent",
            "deployment.environment.name": "production",
            "gen_ai.operation.name": "generate_content",
            "gen_ai.provider.name": "gcp.gemini",
            "gen_ai.request.model": "original-model",
            "gen_ai.request.temperature": 0.7,
            "http.request.method": "POST",
        },
    )


async def test_langfuse_normalizes_model_settings_and_detailed_usage(langfuse):
    observation = {
        k: v
        for k, v in OBSERVATIONS[1].items()
        if k not in ("inputUsage", "outputUsage", "totalUsage")
    }
    observation["modelParameters"] = {
        "temperature": 0.6,
        "maxTokens": 512,
        "top_p": 0.9,
        "presencePenalty": 0.1,
        "ignored": 42,
    }
    observation["usageDetails"] = {
        "input": 120,
        "output": 30,
        "total": 150,
        "reasoning_tokens": 12,
        "cache_read_input_tokens": 40,
        "cache_creation_input_tokens": 10,
    }
    langfuse.mock(return_value=json_response({"data": [observation]}))
    result = await fetch_langfuse()
    assert result is not None
    assert contains(
        result["spans"][0]["attributes"],
        {
            "gen_ai.request.temperature": 0.6,
            "gen_ai.request.max_tokens": 512,
            "gen_ai.request.top_p": 0.9,
            "gen_ai.request.presence_penalty": 0.1,
            "gen_ai.usage.input_tokens": 120,
            "gen_ai.usage.output_tokens": 30,
            "gen_ai.usage.reasoning.output_tokens": 12,
            "gen_ai.usage.cache_read.input_tokens": 40,
            "gen_ai.usage.cache_creation.input_tokens": 10,
            "promptfoo.usage.total_tokens": 150,
        },
    )


async def test_langfuse_omits_reserved_metadata_keys(langfuse):
    metadata = json.loads(
        '{"__proto__":{"polluted":true},"constructor":"unsafe","prototype":"unsafe",'
        '"tenant":"west"}'
    )
    langfuse.mock(return_value=json_response({"data": [{**OBSERVATIONS[0], "metadata": metadata}]}))
    result = await fetch_langfuse()
    assert result is not None
    attributes = result["spans"][0]["attributes"]
    assert attributes["tenant"] == "west"
    for key in ("__proto__", "constructor", "prototype"):
        assert key not in attributes


async def test_langfuse_omits_reserved_path_segments(langfuse):
    metadata = json.loads(
        '{"attributes":{"__proto__":{"polluted":true},"gen_ai.constructor.polluted":true,'
        '"gen_ai.operation.name":"text_completion"},"resourceAttributes":{"prototype":"unsafe",'
        '"service.name":"safe-service"}}'
    )
    langfuse.mock(return_value=json_response({"data": [{**OBSERVATIONS[1], "metadata": metadata}]}))
    result = await fetch_langfuse()
    assert result is not None
    attributes = result["spans"][0]["attributes"]
    assert contains(
        attributes,
        {"gen_ai.operation.name": "text_completion", "service.name": "safe-service"},
    )
    for key in ("__proto__", "gen_ai.constructor.polluted", "prototype"):
        assert key not in attributes


async def test_langfuse_normalizes_cost_details(langfuse):
    langfuse.mock(
        return_value=json_response(
            {
                "data": [
                    {
                        **OBSERVATIONS[1],
                        "costDetails": {"input": 0.001, "output": 0.002, "total": 0.003},
                    }
                ]
            }
        )
    )
    result = await fetch_langfuse()
    assert result is not None
    assert contains(
        result["spans"][0]["attributes"],
        {"langfuse.cost.input": 0.001, "langfuse.cost.output": 0.002, "langfuse.cost.total": 0.003},
    )


async def test_langfuse_page_pagination_and_dedup(langfuse):
    langfuse.side_effect = [
        json_response({"data": [OBSERVATIONS[0]], "meta": {"page": 1, "totalPages": 2}}),
        json_response({"data": OBSERVATIONS, "meta": {"page": 2, "totalPages": 2}}),
    ]
    result = await fetch_langfuse()
    assert result is not None
    assert len(result["spans"]) == 2
    assert langfuse.call_count == 2
    assert query(langfuse.calls[1].request)["page"] == "2"


async def test_langfuse_stable_page_size(langfuse):
    first_page = [{**OBSERVATIONS[0], "id": f"span-{i}"} for i in range(1000)]
    langfuse.side_effect = [
        json_response({"data": first_page, "meta": {"page": 1, "totalPages": 2}}),
        json_response(
            {
                "data": [{**OBSERVATIONS[0], "id": "span-1000"}],
                "meta": {"page": 2, "totalPages": 2},
            }
        ),
    ]
    result = await fetch_langfuse({"maxSpans": 1001})
    assert result is not None
    assert len(result["spans"]) == 1001
    assert langfuse.call_count == 2
    assert query(langfuse.calls[0].request)["limit"] == "1000"
    assert query(langfuse.calls[1].request)["limit"] == "1000"


async def test_langfuse_cursor_pagination(langfuse):
    langfuse.side_effect = [
        json_response({"data": [OBSERVATIONS[0]], "meta": {"cursor": "next-page"}}),
        json_response({"data": OBSERVATIONS, "meta": {}}),
    ]
    result = await fetch_langfuse()
    assert result is not None
    assert len(result["spans"]) == 2
    assert langfuse.call_count == 2
    assert query(langfuse.calls[1].request)["cursor"] == "next-page"


async def test_langfuse_rejects_repeated_cursor(langfuse):
    langfuse.mock(
        side_effect=lambda request: json_response(
            {"data": [OBSERVATIONS[0]], "meta": {"cursor": "again"}}
        )
    )
    with pytest.raises(TraceProviderError, match="repeated pagination cursor"):
        await fetch_langfuse()
    assert langfuse.call_count == 2


async def test_langfuse_null_when_not_indexed(langfuse):
    langfuse.mock(return_value=json_response({"data": [], "meta": {}}))
    assert await fetch_langfuse() is None


async def test_langfuse_zero_page_metadata(langfuse):
    langfuse.mock(return_value=json_response({"data": [], "meta": {"page": 1, "totalPages": 0}}))
    assert await fetch_langfuse() is None
    assert langfuse.call_count == 1


async def test_langfuse_null_on_404(langfuse):
    langfuse.mock(return_value=json_response({"error": "not found"}, 404))
    assert await fetch_langfuse() is None


@pytest.mark.parametrize("status", [408, 429, 500, 503])
async def test_langfuse_retryable_statuses(status, langfuse):
    langfuse.mock(return_value=json_response({"error": "busy"}, status))
    with pytest.raises(TraceProviderError) as exc:
        await fetch_langfuse()
    assert (exc.value.status_code, exc.value.retryable) == (status, True)


async def test_langfuse_invalid_credentials_not_retried(langfuse):
    langfuse.mock(return_value=json_response({"error": "unauthorized"}, 401))
    with pytest.raises(TraceProviderError) as exc:
        await fetch_langfuse()
    assert (exc.value.status_code, exc.value.retryable) == (401, False)


async def test_langfuse_filters_by_start_time(langfuse):
    result = await fetch_langfuse({"earliestStartTime": 1704067200050})
    assert result is not None
    assert [s["name"] for s in result["spans"]] == ["chat gpt-4o"]
    assert query(langfuse.calls[0].request)["fromStartTime"] == "2024-01-01T00:00:00.050Z"


async def test_langfuse_caps_span_count(langfuse):
    langfuse.mock(
        return_value=json_response({"data": OBSERVATIONS, "meta": {"cursor": "next-page"}})
    )
    result = await fetch_langfuse({"maxSpans": 1})
    assert result is not None
    assert len(result["spans"]) == 1
    assert langfuse.call_count == 1
    assert query(langfuse.calls[0].request)["limit"] == "1"


async def test_langfuse_never_sends_non_positive_limit(langfuse):
    await fetch_langfuse({"maxSpans": 0})
    assert query(langfuse.calls[0].request)["limit"] == "1"


async def test_langfuse_skips_malformed_observations(langfuse):
    base = OBSERVATIONS[0]
    langfuse.mock(
        return_value=json_response(
            {
                "data": [
                    None,
                    {**base, "id": ""},
                    {**base, "traceId": "fedcba9876543210fedcba9876543210"},
                    {**base, "startTime": "not-a-date"},
                    {**base, "endTime": "2023-12-31T23:59:59.000Z"},
                    {**base, "parentObservationId": base["id"]},
                    OBSERVATIONS[1],
                ]
            }
        )
    )
    result = await fetch_langfuse()
    assert result is not None
    assert [s["spanId"] for s in result["spans"]] == ["generation-span"]


@pytest.mark.parametrize(
    "payload",
    [
        {"result": "not observations"},
        {"data": OBSERVATIONS, "meta": {"cursor": 123}},
        {"data": OBSERVATIONS, "meta": {"page": 0, "totalPages": 2}},
        {"data": OBSERVATIONS, "meta": {"page": 2, "totalPages": 1}},
    ],
)
async def test_langfuse_rejects_malformed_payloads(payload, langfuse):
    langfuse.mock(return_value=json_response(payload))
    with pytest.raises(TraceProviderError):
        await fetch_langfuse()


async def test_langfuse_rejects_oversized_declared_length(langfuse):
    langfuse.mock(
        return_value=httpx.Response(
            200, content=b"{}", headers={"content-length": str(MAX_BYTES + 1)}
        )
    )
    with pytest.raises(TraceProviderError, match="maximum response size"):
        await fetch_langfuse()


async def test_langfuse_size_limit_spans_pages(langfuse):
    first = json_response({"data": [OBSERVATIONS[0]], "meta": {"page": 1, "totalPages": 2}})
    second = httpx.Response(200, content=b"{}", headers={"content-length": str(MAX_BYTES)})
    langfuse.side_effect = [first, second]
    with pytest.raises(TraceProviderError, match="maximum response size"):
        await fetch_langfuse()


async def test_langfuse_rejects_oversized_bodies(langfuse):
    langfuse.mock(return_value=httpx.Response(200, content=b" " * (MAX_BYTES + 1)))
    with pytest.raises(TraceProviderError, match="maximum response size"):
        await fetch_langfuse()


# --- braintrust.test.ts ---------------------------------------------------------------------

BRAINTRUST_CONFIG: dict[str, Any] = {
    "id": "braintrust",
    "endpoint": "https://api.braintrust.dev",
    "projectId": PROJECT_ID,
    "auth": {"token": "test-token"},
}
BTQL_URL = "https://api.braintrust.dev/btql"
ROWS: list[dict[str, Any]] = [
    {
        "id": "root-event",
        "span_id": "root-span",
        "root_span_id": "root-span",
        "created": "2024-01-01T00:00:00.000Z",
        "metadata": {"service.name": "customer-agent", "trace_id": TRACE_ID},
        "metrics": {"start": 1704067200, "end": 1704067201, "tokens": 42},
        "span_attributes": {"name": "customer.request", "type": "task"},
        "input": {"prompt": "hello"},
        "output": {"text": "world"},
    },
    {
        "id": "child-event",
        "span_id": "child-span",
        "root_span_id": "root-span",
        "span_parents": ["root-span"],
        "created": "2024-01-01T00:00:00.100Z",
        "metrics": {"start": 1704067200.1, "end": 1704067200.5},
        "span_attributes": {"name": "tool.search", "type": "tool"},
        "error": "tool unavailable",
    },
]


@pytest.fixture
def braintrust(mock_router):
    return mock_router.post(BTQL_URL).mock(return_value=json_response({"rows": ROWS}))


async def fetch_braintrust(options: dict[str, Any] | None = None, **config: Any):
    return await BraintrustProvider({**BRAINTRUST_CONFIG, **config}).fetch_trace(TRACE_ID, options)


@pytest.mark.parametrize(
    "value",
    [
        {"id": "braintrust"},
        {**BRAINTRUST_CONFIG, "endpoint": "file:///tmp/traces"},
        {**BRAINTRUST_CONFIG, "endpoint": "https://user:secret@braintrust.example.com"},
        {**BRAINTRUST_CONFIG, "endpoint": "https://api.braintrust.dev?tenant=west"},
        {**BRAINTRUST_CONFIG, "endpoint": "https://api.braintrust.dev#tenant-west"},
        {
            **BRAINTRUST_CONFIG,
            "endpoint": "https://api.braintrust.dev/token-privateTenantCredential123",
        },
        {**BRAINTRUST_CONFIG, "projectId": "invalid-project"},
        {**BRAINTRUST_CONFIG, "auth": {}},
        {**BRAINTRUST_CONFIG, "timeout": -1},
    ],
)
def test_braintrust_rejects_unsafe_configuration(value):
    with pytest.raises(ValueError):
        BraintrustProvider(value)


@pytest.mark.parametrize("trace_id", ["../../admin", "abc123", "0" * 32])
async def test_braintrust_rejects_invalid_trace_ids(trace_id, braintrust):
    with pytest.raises(TraceProviderError):
        await BraintrustProvider(BRAINTRUST_CONFIG).fetch_trace(trace_id)
    assert braintrust.call_count == 0


async def test_braintrust_queries_full_traces(braintrust):
    result = await fetch_braintrust()
    assert result is not None
    spans = result["spans"]
    assert len(spans) == 2
    assert contains(
        spans[0],
        {
            "spanId": "root-span",
            "name": "customer.request",
            "startTime": 1704067200000,
            "endTime": 1704067201000,
            "statusCode": 1,
            "attributes": {
                "service.name": "customer-agent",
                "braintrust.span.type": "task",
                "braintrust.metrics": {"start": 1704067200, "end": 1704067201, "tokens": 42},
                "braintrust.input": {"prompt": "hello"},
                "braintrust.output": {"text": "world"},
            },
        },
    )
    assert contains(
        spans[1],
        {
            "parentSpanId": "root-span",
            "name": "tool.search",
            "statusCode": 2,
            "statusMessage": "tool unavailable",
            "startTime": 1704067200100,
            "endTime": 1704067200500,
        },
    )
    request = braintrust.calls[0].request
    assert str(request.url) == BTQL_URL
    assert request.method == "POST"
    assert request.headers["Authorization"] == "Bearer test-token"
    body = json.loads(request.content)
    assert body["fmt"] == "json"
    assert f"project_logs('{PROJECT_ID}', shape => 'traces')" in body["query"]
    assert f"metadata.trace_id = '{TRACE_ID}'" in body["query"]
    assert "created >= now() - INTERVAL 1 DAY" in body["query"]
    assert body["query"].endswith("LIMIT 10000")


async def test_braintrust_links_nested_spans_to_immediate_parent(braintrust):
    braintrust.mock(
        return_value=json_response(
            {
                "rows": [
                    *ROWS,
                    {
                        "id": "grandchild-event",
                        "span_id": "grandchild-span",
                        "span_parents": ["root-span", "child-span", "grandchild-span"],
                        "created": "2024-01-01T00:00:00.200Z",
                        "span_attributes": {"name": "tool.follow_up", "type": "tool"},
                    },
                ]
            }
        )
    )
    result = await fetch_braintrust()
    assert result is not None
    assert contains(result["spans"][2], {"spanId": "grandchild-span", "parentSpanId": "child-span"})


async def test_braintrust_normalizes_tool_spans(braintrust):
    braintrust.mock(
        return_value=json_response(
            {
                "rows": [
                    {
                        "id": "search-event",
                        "span_id": "search-span",
                        "created": "2024-01-01T00:00:00.000Z",
                        "input": {"query": "customer orders"},
                        "span_attributes": {"name": "search", "type": "tool"},
                    }
                ]
            }
        )
    )
    result = await fetch_braintrust()
    assert result is not None
    span = result["spans"][0]
    assert contains(
        span["attributes"],
        {
            "braintrust.input": {"query": "customer orders"},
            "gen_ai.tool.name": "search",
            "tool.name": "search",
            "tool.arguments": '{"query":"customer orders"}',
        },
    )
    assert get_tool_name_from_attributes(span["attributes"]) == "search"
    assert is_relevant_span(span) is True


async def test_braintrust_protects_provider_headers(braintrust):
    await fetch_braintrust(
        headers={
            "authorization": "Bearer unexpected",
            "aCcEpT": "text/plain",
            "content-TYPE": "text/plain",
            "X-Tenant": "tenant-a",
        }
    )
    headers = braintrust.calls[0].request.headers
    assert headers.get_list("Accept") == ["application/json"]
    assert headers.get_list("Content-Type") == ["application/json"]
    assert headers.get_list("Authorization") == ["Bearer test-token"]
    assert headers["X-Tenant"] == "tenant-a"


async def test_braintrust_accepts_data_shaped_responses(braintrust):
    braintrust.mock(return_value=json_response({"data": [ROWS[0]]}))
    result = await fetch_braintrust()
    assert result is not None
    assert len(result["spans"]) == 1


async def test_braintrust_null_when_not_indexed(braintrust):
    braintrust.mock(return_value=json_response({"rows": []}))
    assert await fetch_braintrust() is None


@pytest.mark.parametrize("status", [408, 429, 500, 503])
async def test_braintrust_retryable_statuses(status, braintrust):
    braintrust.mock(return_value=json_response({"error": "busy"}, status))
    with pytest.raises(TraceProviderError) as exc:
        await fetch_braintrust()
    assert (exc.value.status_code, exc.value.retryable) == (status, True)


async def test_braintrust_missing_endpoint_is_configuration_error(braintrust):
    braintrust.mock(return_value=json_response({"error": "not found"}, 404))
    with pytest.raises(TraceProviderError) as exc:
        await fetch_braintrust()
    assert "check the endpoint and project configuration" in str(exc.value)
    assert (exc.value.status_code, exc.value.retryable) == (404, False)


async def test_braintrust_unauthorized_not_retryable(braintrust):
    braintrust.mock(return_value=json_response({"error": "unauthorized"}, 401))
    with pytest.raises(TraceProviderError) as exc:
        await fetch_braintrust()
    assert (exc.value.status_code, exc.value.retryable) == (401, False)


async def test_braintrust_rejects_oversized_declared_length(braintrust):
    braintrust.mock(
        return_value=httpx.Response(200, content=b"{}", headers={"content-length": "10485761"})
    )
    with pytest.raises(TraceProviderError, match="maximum response size"):
        await fetch_braintrust()


async def test_braintrust_rejects_oversized_bodies(braintrust):
    braintrust.mock(return_value=httpx.Response(200, content=b" " * (MAX_BYTES + 1)))
    with pytest.raises(TraceProviderError, match="maximum response size"):
        await fetch_braintrust()


async def test_braintrust_filters_start_time_and_caps_count(braintrust):
    provider = BraintrustProvider(BRAINTRUST_CONFIG)
    capped = await provider.fetch_trace(TRACE_ID, {"maxSpans": 1})
    assert capped is not None and len(capped["spans"]) == 1
    filtered = await provider.fetch_trace(TRACE_ID, {"earliestStartTime": 1704067200050})
    assert filtered is not None
    assert [s["name"] for s in filtered["spans"]] == ["tool.search"]


async def test_braintrust_rejects_malformed_payloads(braintrust):
    braintrust.mock(return_value=json_response({"result": "not span rows"}))
    with pytest.raises(TraceProviderError, match="invalid query response"):
        await fetch_braintrust()
