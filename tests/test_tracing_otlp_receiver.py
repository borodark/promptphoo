"""Ports test/tracing/otlpReceiver.test.ts. Requests go through `OTLPReceiver.handle` (the
express app; upstream uses supertest) against a recording trace store, and a few run over
real sockets on ephemeral localhost ports and the real SQLite store."""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import socket
from typing import Any

import pytest

from promptphoo import db
from promptphoo.jsvalues import json_stringify
from promptphoo.tracing import otlp_receiver as receiver_module
from promptphoo.tracing.otlp_receiver import OTLPReceiver, Request

OPTS = {"skipTraceCheck": False, "warnIfMissingTrace": False}


class RecordingStore:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.added: list[tuple[str, list[dict[str, Any]], Any]] = []
        self.metadata: list[Any] = []
        self.add_error: Exception | None = None

    async def create_trace(self, trace):
        self.created.append(trace)

    async def add_spans(self, trace_id, spans, options=None):
        if self.add_error is not None:
            error, self.add_error = self.add_error, None
            raise error
        # JSON.stringify drops JS-undefined keys, as toEqual ignores them upstream.
        self.added.append((trace_id, json.loads(json_stringify(spans)), options))
        return {"stored": True}

    async def get_trace_metadata(self, trace_id):
        return self.metadata.pop(0) if self.metadata else None


@pytest.fixture
def store():
    return RecordingStore()


def _receiver(store: RecordingStore, **options: Any) -> OTLPReceiver:
    receiver = OTLPReceiver(options)
    receiver.trace_store = store  # type: ignore[assignment]
    return receiver


async def _call(
    receiver: OTLPReceiver,
    method: str,
    path: str,
    body: Any = None,
    content_type: str | None = "application/json",
) -> tuple[int, Any]:
    raw: bytes | None = None
    headers: dict[str, str] = {}
    if body is not None:
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["content-length"] = str(len(raw))
        if content_type:
            headers["content-type"] = content_type
    response = await receiver.handle(Request(method, path, headers, raw))
    text = response.body.decode()
    try:
        return response.status, json.loads(text)
    except ValueError:
        return response.status, text


def _b64(hex_id: str) -> str:
    return base64.b64encode(bytes.fromhex(hex_id)).decode()


def _request(*spans: dict[str, Any], resource: list | None = None, scope=None) -> dict:
    resource_spans: dict[str, Any] = {"scopeSpans": [{"spans": list(spans)}]}
    if scope:
        resource_spans["scopeSpans"][0]["scope"] = scope
    if resource:
        resource_spans["resource"] = {"attributes": resource}
    return {"resourceSpans": [resource_spans]}


def _own(attributes: dict) -> dict:
    """The span's attributes without the receiver's `otel.span.kind*` additions."""
    return {k: v for k, v in attributes.items() if not k.startswith("otel.span.kind")}


def _sv(value: str) -> dict:
    return {"stringValue": value}


# --- health and service info ------------------------------------------------------------------


async def test_health(store):
    assert await _call(_receiver(store), "GET", "/health") == (200, {"status": "ok"})


async def test_service_info(store):
    assert await _call(_receiver(store), "GET", "/v1/traces") == (
        200,
        {"service": "promptfoo-otlp-receiver", "version": "1.0.0",
         "supported_formats": ["json", "protobuf"]},
    )  # fmt: skip
    _, info = await _call(_receiver(store, acceptFormats=["json"]), "GET", "/v1/traces")
    assert info["supported_formats"] == ["json"]


# --- trace ingestion -----------------------------------------------------------------------------


async def test_accepts_otlp_json(store):
    body = _request(
        {"traceId": _b64("12345678901234567890123456789012"), "spanId": _b64("1234567890123456"),
         "name": "test-span", "kind": 1, "startTimeUnixNano": "1700000000000000000",
         "endTimeUnixNano": "1700000001000000000",
         "attributes": [{"key": "test.attribute", "value": _sv("test-value")},
                        {"key": "test.count", "value": {"intValue": "42"}}],
         "status": {"code": 0, "message": "OK"}},
        resource=[{"key": "service.name", "value": _sv("test-service")},
                  {"key": "service.version", "value": _sv("1.0.0")}],
        scope={"name": "test-tracer", "version": "1.0.0"},
    )  # fmt: skip
    assert await _call(_receiver(store), "POST", "/v1/traces", body) == (
        200,
        {"partialSuccess": {}},
    )
    [(trace_id, [span], options)] = store.added
    assert trace_id == "12345678901234567890123456789012"
    assert options == OPTS
    assert span["spanId"] == "1234567890123456"
    assert span["name"] == "test-span"
    assert span["startTime"] == 1700000000000
    assert span["endTime"] == 1700000001000
    assert span["statusCode"] == 0 and span["statusMessage"] == "OK"
    assert span["attributes"] == {
        "service.name": "test-service", "service.version": "1.0.0",
        "test.attribute": "test-value", "test.count": 42,
        "otel.scope.name": "test-tracer", "otel.scope.version": "1.0.0",
        "otel.span.kind": "internal", "otel.span.kind_code": 1,
    }  # fmt: skip
    assert store.created == []  # no evaluation.id: spans only


async def test_accepts_json_with_charset(store):
    body = _request(
        {
            "traceId": _b64("1" * 32),
            "spanId": _b64("2" * 16),
            "name": "json-with-charset",
            "startTimeUnixNano": "1700000000000000000",
        }
    )
    status, _ = await _call(
        _receiver(store), "POST", "/v1/traces", body, "application/json; charset=utf-8"
    )
    assert status == 200
    assert store.added[0][1][0]["name"] == "json-with-charset"


async def test_multiple_spans_in_one_request(store):
    trace = _b64("a" * 32)
    body = _request(
        {"traceId": trace, "spanId": _b64("1" * 16), "name": "span-1",
         "startTimeUnixNano": "1000000000"},
        {"traceId": trace, "spanId": _b64("2" * 16), "parentSpanId": _b64("1" * 16),
         "name": "span-2", "startTimeUnixNano": "2000000000"},
    )  # fmt: skip
    await _call(_receiver(store), "POST", "/v1/traces", body)
    [(trace_id, spans, _)] = store.added
    assert trace_id == "a" * 32
    assert [(s["spanId"], s.get("parentSpanId"), s["name"]) for s in spans] == [
        ("1" * 16, None, "span-1"), ("2" * 16, "1" * 16, "span-2"),
    ]  # fmt: skip
    assert spans[0]["startTime"] == 1000


async def test_attribute_types(store):
    body = _request({"traceId": _b64("b" * 32), "spanId": _b64("3" * 16),
                     "name": "test-attributes", "startTimeUnixNano": "1000000000",
                     "attributes": [
                         {"key": "string.attr", "value": _sv("hello")},
                         {"key": "int.attr", "value": {"intValue": "123"}},
                         {"key": "double.attr", "value": {"doubleValue": 3.14}},
                         {"key": "bool.attr", "value": {"boolValue": True}},
                         {"key": "array.attr", "value": {"arrayValue": {
                             "values": [_sv("a"), _sv("b")]}}},
                     ]})  # fmt: skip
    await _call(_receiver(store), "POST", "/v1/traces", body)
    assert _own(store.added[0][1][0]["attributes"]) == {
        "string.attr": "hello", "int.attr": 123, "double.attr": 3.14, "bool.attr": True,
        "array.attr": ["a", "b"],
    }  # fmt: skip


async def test_rejects_unsupported_content_type(store):
    assert await _call(_receiver(store), "POST", "/v1/traces", b"invalid data", "text/plain") == (
        415,
        {"error": "Unsupported content type"},
    )


async def test_rejects_disabled_formats(store):
    protobuf_only = _receiver(store, acceptFormats=["protobuf"])
    unsupported = (415, {"error": "Unsupported content type"})
    assert await _call(protobuf_only, "POST", "/v1/traces", {"resourceSpans": []}) == unsupported
    # Malformed JSON is not even parsed when JSON is disabled.
    assert await _call(protobuf_only, "POST", "/v1/traces", b"{ invalid json") == unsupported
    json_only = _receiver(store, acceptFormats=["json"])
    payload = _protobuf_payload()
    assert (
        await _call(json_only, "POST", "/v1/traces", payload, "application/x-protobuf")
        == unsupported
    )
    assert store.added == []


async def test_rejects_invalid_protobuf(store):
    status, body = await _call(
        _receiver(store), "POST", "/v1/traces", b"dummy protobuf data", "application/x-protobuf"
    )
    assert status == 400
    assert "invalid protobuf" in body["error"].lower()


async def test_rejects_overflowing_protobuf_lengths_without_storing(store):
    status, body = await _call(
        _receiver(store), "POST", "/v1/traces", bytes([0x0A, 0x80, 0x80, 0x80, 0x80, 0x10]),
        "application/x-protobuf",
    )  # fmt: skip
    assert (status, body) == (400, {"error": "Invalid protobuf data: invalid length encoding"})
    assert store.created == [] and store.added == []


def _protobuf_payload() -> bytes:
    """protobufjs's encoding of upstream's 'should accept valid OTLP protobuf traces' request."""

    def field(number: int, wire: int, payload: bytes | int) -> bytes:
        def varint(n: int) -> bytes:
            out = bytearray()
            while True:
                out.append((n & 0x7F) | (0x80 if n > 0x7F else 0))
                n >>= 7
                if not n:
                    return bytes(out)

        tag = varint(number << 3 | wire)
        if wire == 0:
            assert isinstance(payload, int)
            return tag + varint(payload)
        if wire == 1:
            assert isinstance(payload, int)
            return tag + payload.to_bytes(8, "little")
        assert isinstance(payload, bytes)
        return tag + varint(len(payload)) + payload

    def kv(key: str, value: bytes) -> bytes:
        return field(1, 2, key.encode()) + field(2, 2, value)

    span = (
        field(1, 2, bytes.fromhex("deadbeef" * 4))
        + field(2, 2, bytes.fromhex("1234567890abcdef"))
        + field(5, 2, b"protobuf-test-span")
        + field(6, 0, 2)
        + field(7, 1, 1700000000000000000)
        + field(8, 1, 1700000001000000000)
        + field(9, 2, kv("http.method", field(1, 2, b"POST")))
        + field(9, 2, kv("http.status_code", field(3, 0, 200)))
        + field(15, 2, field(2, 2, b"Success") + field(3, 0, 1))
    )
    scope = field(1, 2, b"opentelemetry.instrumentation.test") + field(2, 2, b"1.0.0")
    resource = field(1, 2, kv("service.name", field(1, 2, b"test-python-service")))
    resource_spans = field(1, 2, resource) + field(2, 2, field(1, 2, scope) + field(2, 2, span))
    return field(1, 2, resource_spans)


async def test_accepts_otlp_protobuf(store):
    status, body = await _call(
        _receiver(store), "POST", "/v1/traces", _protobuf_payload(), "application/x-protobuf"
    )
    assert (status, body) == (200, {"partialSuccess": {}})
    [(trace_id, [span], options)] = store.added
    assert trace_id == "deadbeef" * 4 and options == OPTS
    assert span["spanId"] == "1234567890abcdef"
    assert span["name"] == "protobuf-test-span"
    assert span["startTime"] == 1700000000000 and span["endTime"] == 1700000001000
    assert span["statusCode"] == 1 and span["statusMessage"] == "Success"
    assert span["attributes"] == {
        "service.name": "test-python-service", "http.method": "POST", "http.status_code": 200,
        "otel.scope.name": "opentelemetry.instrumentation.test", "otel.scope.version": "1.0.0",
        "otel.span.kind": "server", "otel.span.kind_code": 2,
    }  # fmt: skip


async def test_malformed_json(store):
    assert await _call(_receiver(store), "POST", "/v1/traces", b"{ invalid json") == (
        400,
        {"error": "Invalid JSON"},
    )


async def test_trace_store_errors(store):
    store.add_error = RuntimeError("Database error")
    body = _request({"traceId": _b64("c" * 32), "spanId": _b64("4" * 16), "name": "error-span",
                     "startTimeUnixNano": "1000000000"})  # fmt: skip
    assert await _call(_receiver(store), "POST", "/v1/traces", body) == (
        500,
        {"error": "Internal server error"},
    )


async def test_gzip_and_deflate_bodies(store):
    import zlib

    body = json.dumps(_request({"traceId": "a" * 32, "spanId": "b" * 16, "name": "zipped",
                                "startTimeUnixNano": "1"})).encode()  # fmt: skip
    receiver = _receiver(store)
    for encoding, data in (("gzip", gzip.compress(body)), ("deflate", zlib.compress(body))):
        headers = {"content-type": "application/json", "content-encoding": encoding,
                   "content-length": str(len(data))}  # fmt: skip
        response = await receiver.handle(Request("POST", "/v1/traces", headers, data))
        assert response.status == 200
    assert [spans[0]["name"] for _, spans, _ in store.added] == ["zipped", "zipped"]


async def test_routing_is_case_insensitive_and_404s_like_express(store):
    receiver = _receiver(store)
    assert (await _call(receiver, "GET", "/HEALTH/"))[0] == 200
    status, page = await _call(receiver, "GET", "/nope?x=1")
    assert status == 404
    assert "<pre>Cannot GET /nope</pre>" in page


# --- ingest-time redaction ------------------------------------------------------------------------


async def test_redacts_matched_attribute_values_before_persisting(store):
    receiver = _receiver(store, redactAttributes=["password", "authorization", "tool.arguments"])
    body = _request({"traceId": "a" * 32, "spanId": "1234567890abcdef", "name": "tool.call",
                     "startTimeUnixNano": "1000000000", "endTimeUnixNano": "2000000000",
                     "attributes": [
                         {"key": "tool.name", "value": _sv("lookup_user")},
                         {"key": "tool.arguments", "value": _sv('{"ssn":"123-45-6789"}')},
                         {"key": "http.password", "value": _sv("pw123")},
                         {"key": "AUTHORIZATION", "value": _sv("Bearer xyz")},
                         {"key": "request.context", "value": {"kvlistValue": {"values": [
                             {"key": "user.password", "value": _sv("nested-pw")},
                             {"key": "safe", "value": _sv("visible")}]}}},
                         {"key": "tool.result", "value": _sv("ok")},
                     ]})  # fmt: skip
    await _call(receiver, "POST", "/v1/traces", body)
    [(trace_id, [span], options)] = store.added
    assert trace_id == "a" * 32 and options == OPTS
    assert _own(span["attributes"]) == {
        "tool.name": "lookup_user", "tool.arguments": "[REDACTED]",
        "http.password": "[REDACTED]", "AUTHORIZATION": "[REDACTED]",
        "request.context": {"user.password": "[REDACTED]", "safe": "visible"},
        "tool.result": "ok",
    }  # fmt: skip


async def test_redacts_name_and_status_message_echoing_a_redacted_value(store):
    receiver = _receiver(store, acceptFormats=["json"], redactAttributes=["authorization"])
    body = _request({"traceId": "e" * 32, "spanId": "1234567890abcdef",
                     "name": "Bearer secret-token", "startTimeUnixNano": "1000000000",
                     "endTimeUnixNano": "2000000000",
                     "status": {"code": 2, "message": "Bearer secret-token"},
                     "attributes": [{"key": "authorization", "value": _sv("Bearer secret-token")},
                                    {"key": "safe", "value": _sv("visible")}]})  # fmt: skip
    await _call(receiver, "POST", "/v1/traces", body)
    [span] = store.added[0][1]
    assert span["name"] == "[REDACTED]" and span["statusMessage"] == "[REDACTED]"
    assert _own(span["attributes"]) == {"authorization": "[REDACTED]", "safe": "visible"}


async def test_replaces_nested_values_of_a_matched_key_wholesale(store):
    receiver = _receiver(store, acceptFormats=["json"], redactAttributes=["secret"])
    body = _request({"traceId": "f" * 32, "spanId": "1234567890abcdef", "name": "tool.call",
                     "startTimeUnixNano": "1000000000",
                     "attributes": [
                         {"key": "secret.config", "value": {"kvlistValue": {"values": [
                             {"key": "k", "value": _sv("v")}]}}},
                         {"key": "secret.list", "value": {"arrayValue": {
                             "values": [_sv("one"), _sv("two")]}}},
                     ]})  # fmt: skip
    await _call(receiver, "POST", "/v1/traces", body)
    assert _own(store.added[0][1][0]["attributes"]) == {
        "secret.config": "[REDACTED]",
        "secret.list": "[REDACTED]",
    }


def _tool_span(trace_id: str, *attrs: tuple[str, str]) -> dict:
    return _request({"traceId": trace_id, "spanId": "1234567890abcdef", "name": "tool.call",
                     "startTimeUnixNano": "1000000000",
                     "attributes": [{"key": k, "value": _sv(v)} for k, v in attrs]})  # fmt: skip


async def test_trace_specific_redaction_replaces_the_receiver_default(store):
    receiver = _receiver(store, acceptFormats=["json"], redactAttributes=["authorization"])
    store.metadata.append({"otlpHttpRedactAttributes": ["password"]})
    await _call(receiver, "POST", "/v1/traces", _tool_span(
        "b" * 32, ("AUTHORIZATION", "Bearer abc"), ("http.password", "pw123")))  # fmt: skip
    assert _own(store.added[0][1][0]["attributes"]) == {
        "AUTHORIZATION": "Bearer abc",
        "http.password": "[REDACTED]",
    }


async def test_linked_trace_without_policy_skips_receiver_default(store):
    receiver = _receiver(store, acceptFormats=["json"], redactAttributes=["authorization"])
    store.metadata.append({})
    await _call(receiver, "POST", "/v1/traces",
                _tool_span("c" * 32, ("AUTHORIZATION", "Bearer visible")))  # fmt: skip
    assert _own(store.added[0][1][0]["attributes"]) == {"AUTHORIZATION": "Bearer visible"}


async def test_receiver_created_traces_persist_the_redaction_policy(store):
    receiver = _receiver(store, acceptFormats=["json"], redactAttributes=["authorization"])
    store.metadata.append({"otlpHttpRedactAttributes": ["authorization"]})
    await _call(receiver, "POST", "/v1/traces", _tool_span(
        "d" * 32, ("evaluation.id", "receiver-created-eval"),
        ("test.case.id", "receiver-created-test"), ("AUTHORIZATION", "Bearer hidden")))  # fmt: skip
    assert store.created == [{
        "traceId": "d" * 32, "evaluationId": "receiver-created-eval",
        "testCaseId": "receiver-created-test",
        "metadata": {"otlpHttpRedactAttributes": ["authorization"]},
    }]  # fmt: skip
    assert store.added[0][1][0]["attributes"]["AUTHORIZATION"] == "[REDACTED]"


async def test_receiver_created_traces_persist_command_tool_names(store):
    receiver = _receiver(store, acceptFormats=["json"], commandToolNames=["bash", "shell"])
    await _call(receiver, "POST", "/v1/traces", _tool_span(
        "e" * 32, ("evaluation.id", "receiver-created-eval"),
        ("test.case.id", "receiver-created-test"), ("tool.name", "bash"),
        ("tool.arguments", '{"cmd":"npm test"}')))  # fmt: skip
    assert store.created == [{
        "traceId": "e" * 32, "evaluationId": "receiver-created-eval",
        "testCaseId": "receiver-created-test", "metadata": {"commandToolNames": ["bash", "shell"]},
    }]  # fmt: skip


async def test_registered_evaluation_policy_on_a_shared_receiver(store):
    receiver = _receiver(store, acceptFormats=["json"])
    receiver.register_trace_policy({"evaluationId": "registered-eval",
                                    "commandToolNames": ["bash"],
                                    "redactAttributes": ["authorization"]})  # fmt: skip
    await _call(receiver, "POST", "/v1/traces", _tool_span(
        "f" * 32, ("evaluation.id", "registered-eval"), ("test.case.id", "registered-test"),
        ("AUTHORIZATION", "Bearer hidden")))  # fmt: skip
    assert store.created == [{
        "traceId": "f" * 32, "evaluationId": "registered-eval", "testCaseId": "registered-test",
        "metadata": {"commandToolNames": ["bash"], "otlpHttpRedactAttributes": ["authorization"]},
    }]  # fmt: skip
    assert store.added[0][1][0]["attributes"]["AUTHORIZATION"] == "[REDACTED]"


async def test_deregistered_policy_is_dropped(store):
    receiver = _receiver(store, acceptFormats=["json"])
    receiver.register_trace_policy({"evaluationId": "gc-eval", "redactAttributes": ["auth"]})
    receiver.deregister_trace_policy("gc-eval")
    await _call(receiver, "POST", "/v1/traces", _tool_span(
        "1" * 32, ("evaluation.id", "gc-eval"), ("AUTHORIZATION", "Bearer hidden")))  # fmt: skip
    assert store.added[0][1][0]["attributes"]["AUTHORIZATION"] == "Bearer hidden"


async def test_empty_registered_policy_wins_over_startup_defaults(store):
    receiver = _receiver(store, acceptFormats=["json"], commandToolNames=["bash"],
                         redactAttributes=["authorization"])  # fmt: skip
    receiver.register_trace_policy({"evaluationId": "no-redaction-eval"})
    await _call(receiver, "POST", "/v1/traces", _tool_span(
        "1" * 32, ("evaluation.id", "no-redaction-eval"), ("test.case.id", "no-redaction-test"),
        ("AUTHORIZATION", "Bearer hidden")))  # fmt: skip
    assert store.created == [{
        "traceId": "1" * 32, "evaluationId": "no-redaction-eval",
        "testCaseId": "no-redaction-test", "metadata": None,
    }]  # fmt: skip
    assert store.added[0][1][0]["attributes"]["AUTHORIZATION"] == "Bearer hidden"


# --- ids ----------------------------------------------------------------------------------------


async def test_base64_ids_become_hex(store):
    hex_id = "deadbeef" * 4
    await _call(_receiver(store), "POST", "/v1/traces", _request(
        {"traceId": _b64(hex_id), "spanId": _b64("1234567890abcdef"),
         "name": "test-conversion", "startTimeUnixNano": "1000000000"}))  # fmt: skip
    assert store.added[0][0] == hex_id
    assert store.added[0][1][0]["spanId"] == "1234567890abcdef"


async def test_unlinked_spans_create_no_trace_rows(store):
    await _call(_receiver(store), "POST", "/v1/traces", _request(
        {"traceId": _b64("a" * 32), "spanId": _b64("b" * 16),
         "name": "child-span-without-linkage", "startTimeUnixNano": "1000000000"}))  # fmt: skip
    assert store.created == []
    assert store.added[0][0] == "a" * 32 and store.added[0][2] == OPTS


# --- /v1/logs -------------------------------------------------------------------------------------

HEX_TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
HEX_PARENT = "00f067aa0ba902b7"


def _logs(*records: dict, resource: list | None = None, scope: dict | None = None) -> dict:
    return {"resourceLogs": [{
        "resource": {"attributes": resource if resource is not None else [
            {"key": "service.name", "value": _sv("claude-agent-sdk")}]},
        "scopeLogs": [{"scope": scope or {"name": "com.anthropic.claude_code", "version": "1.0.0"},
                       "logRecords": list(records)}],
    }]}  # fmt: skip


def _record(event: str | None = None, **extra: Any) -> dict:
    record: dict[str, Any] = {"timeUnixNano": "1700000000000000000", "traceId": HEX_TRACE,
                              "spanId": HEX_PARENT}  # fmt: skip
    if event:
        record["attributes"] = [{"key": "event.name", "value": _sv(event)}]
    record.update(extra)
    return record


async def _post_logs(store, body, receiver=None):
    status, _ = await _call(receiver or _receiver(store), "POST", "/v1/logs", body)
    assert status == 200
    return [span for _, spans, _ in store.added for span in spans]


async def test_log_becomes_a_child_span(store):
    [span] = await _post_logs(store, _logs(_record(
        severityNumber=9, severityText="INFO", body=_sv("Tool execution complete"),
        attributes=[{"key": "event.name", "value": _sv("claude_code.tool.execution")},
                    {"key": "tool_name", "value": _sv("Bash")},
                    {"key": "duration_ms", "value": {"intValue": "342"}}])))  # fmt: skip
    assert store.added[0][0] == HEX_TRACE
    assert span["name"] == "claude_code.tool.execution"
    assert span["parentSpanId"] == HEX_PARENT
    assert span["attributes"]["tool_name"] == "Bash"
    assert span["attributes"]["duration_ms"] == 342
    assert span["attributes"]["otel.log.body"] == "Tool execution complete"
    assert span["attributes"]["service.name"] == "claude-agent-sdk"
    assert (span["startTime"], span["endTime"]) == (1700000000000, 1700000000001)


async def test_logs_without_trace_ids_and_denylisted_logs_are_dropped(store):
    no_trace = {"timeUnixNano": "1700000000000000000",
                "attributes": [{"key": "event.name", "value": _sv("orphan")}]}  # fmt: skip
    assert await _post_logs(store, _logs(no_trace)) == []
    assert await _post_logs(store, _logs(_record("claude_code.tracing"))) == []
    assert await _post_logs(store, _logs(_record("x", traceId="not-a-real-trace-id"))) == []
    assert await _post_logs(store, {"resourceLogs": []}) == []


async def test_log_span_names(store):
    spans = await _post_logs(store, _logs(
        _record(body=_sv("agent.message")),
        _record(attributes=[{"key": "claude_code.event.name",
                             "value": _sv("claude_code.llm_request")}]),
        _record(body=_sv("A" * 200)),
    ))  # fmt: skip
    assert [s["name"] for s in spans] == ["agent.message", "claude_code.llm_request", "otel.log"]


async def test_log_body_copied_into_the_name_is_redacted(store):
    receiver = _receiver(store, redactAttributes=["otel.log.body"])
    [span] = await _post_logs(store, _logs(_record(body=_sv("secret agent message"))), receiver)
    assert span["name"] == "[REDACTED]"
    assert span["attributes"]["otel.log.body"] == "[REDACTED]"


async def test_logs_reject_non_json(store):
    assert await _call(_receiver(store), "POST", "/v1/logs", b"\x00", "application/x-protobuf") == (
        415,
        {"error": "Only application/json is supported for /v1/logs"},
    )


@pytest.mark.parametrize("span_id", ["0000000000000000", "AAAAAAAAAAA="])
async def test_zero_span_id_means_no_parent(store, span_id):
    [span] = await _post_logs(store, _logs(_record("claude_code.noise", spanId=span_id)))
    assert "parentSpanId" not in span


async def test_multiple_log_records(store):
    spans = await _post_logs(store, _logs(
        _record("claude_code.tool.execution"),
        _record("claude_code.llm_request", timeUnixNano="1700000000100000000"),
    ))  # fmt: skip
    assert [s["name"] for s in spans] == ["claude_code.tool.execution", "claude_code.llm_request"]


RESOURCE_CONTEXT = [
    {"key": "promptfoo.trace_id", "value": _sv(HEX_TRACE)},
    {"key": "promptfoo.parent_span_id", "value": _sv(HEX_PARENT)},
]


async def test_logs_fall_back_to_resource_trace_context(store):
    record = {"timeUnixNano": "1700000000000000000", "body": _sv("claude_code.tool_result"),
              "attributes": [{"key": "event.name", "value": _sv("tool_result")},
                             {"key": "tool_name", "value": _sv("Read")}]}  # fmt: skip
    [span] = await _post_logs(store, _logs(
        record, resource=[{"key": "service.name", "value": _sv("claude-agent-sdk")},
                          *RESOURCE_CONTEXT]))  # fmt: skip
    assert store.added[0][0] == HEX_TRACE
    assert span["parentSpanId"] == HEX_PARENT and span["name"] == "tool_result"


@pytest.mark.parametrize(
    ("trace_id", "span_id"),
    [("", ""), ("0" * 32, "0" * 16),
     (base64.b64encode(bytes(16)).decode(), base64.b64encode(bytes(8)).decode())],
    ids=["empty", "zero-hex", "zero-base64"],
)  # fmt: skip
async def test_codex_logs_fall_back_to_resource_context(store, trace_id, span_id):
    record = _record("codex.api_request", traceId=trace_id, spanId=span_id,
                     body=_sv("codex.api_request"))  # fmt: skip
    [span] = await _post_logs(
        store,
        _logs(
            record,
            scope={"name": "codex_otel"},
            resource=[{"key": "service.name", "value": _sv("codex-cli")}, *RESOURCE_CONTEXT],
        ),
    )
    assert store.added[0][0] == HEX_TRACE
    assert span["parentSpanId"] == HEX_PARENT and span["name"] == "codex.api_request"


async def test_error_severity_keeps_ok_status(store):
    [span] = await _post_logs(store, _logs(_record(
        "claude_code.error", severityNumber=17, severityText="ERROR",
        body=_sv("something failed"))))  # fmt: skip
    assert span["statusCode"] == 1 and span["statusMessage"] == "ERROR"
    assert span["attributes"]["otel.log.severity_number"] == 17


async def test_oversize_log_body_is_truncated(store):
    huge = "x" * 20_000
    [span] = await _post_logs(store, _logs(_record("claude_code.noise", body=_sv(huge))))
    stored = span["attributes"]["otel.log.body"]
    assert len(stored) < len(huge) and stored.endswith("... [truncated]")


async def test_malformed_record_is_skipped_but_good_ones_kept(store):
    spans = await _post_logs(store, _logs(
        _record(attributes=[{"key": "broken"}]),
        _record("claude_code.tool.execution", timeUnixNano="1700000000100000000"),
    ))  # fmt: skip
    assert len(store.added) == 1 and store.added[0][0] == HEX_TRACE
    assert [s["name"] for s in spans] == ["claude_code.tool.execution"]


# --- listening, singleton startup --------------------------------------------------------------


@pytest.fixture
def real_db(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("IS_TESTING", raising=False)
    db.close_db()
    db.run_db_migrations()
    db.get_db().execute(
        "INSERT INTO evals (id, created_at, config, results) VALUES ('eval-1', 0, '{}', '{}')"
    )
    yield
    db.close_db()


def _http(port: int, raw: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port)) as s:
        s.sendall(raw)
        data = b""
        while chunk := s.recv(65536):
            data += chunk
    return data


async def test_listen_rejects_a_port_in_use(store):
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = blocker.getsockname()[1]
    try:
        with pytest.raises(OSError, match=f"EADDRINUSE: address already in use 127.0.0.1:{port}"):
            await _receiver(store, acceptFormats=["json"]).listen(port, "127.0.0.1")
    finally:
        blocker.close()


async def test_singleton_recovers_after_a_failed_listen(real_db):
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    try:
        with pytest.raises(OSError, match="EADDRINUSE"):
            await receiver_module.start_otlp_receiver(blocker.getsockname()[1], "127.0.0.1",
                                                      ["json"])  # fmt: skip
        failed = receiver_module.get_otlp_receiver_instance()
        await receiver_module.start_otlp_receiver(0, "127.0.0.1", ["protobuf"])
        started = receiver_module.get_otlp_receiver_instance()
        assert started is not None and started is not failed
        assert started.accept_formats == ["protobuf"]
    finally:
        blocker.close()
        await receiver_module.stop_otlp_receiver()
    assert receiver_module.get_otlp_receiver_instance() is None


async def test_spans_posted_over_http_land_in_the_store(real_db):
    from promptphoo.tracing.store import get_trace_store

    await get_trace_store().create_trace({"traceId": "a" * 32, "evaluationId": "eval-1",
                                          "testCaseId": "0-0", "metadata": {}})  # fmt: skip
    await receiver_module.start_otlp_receiver(0, "127.0.0.1", None)
    receiver = receiver_module.get_otlp_receiver_instance()
    assert receiver is not None
    port = receiver.bound_port
    body = json.dumps(
        _request(
            {
                "traceId": "a" * 32,
                "spanId": "b" * 16,
                "name": "over-http",
                "startTimeUnixNano": "1700000000000000000",
                "attributes": [{"key": "k", "value": _sv("v")}],
            }
        )
    ).encode()
    raw = (b"POST /v1/traces HTTP/1.1\r\nHost: x\r\nConnection: close\r\n"
           b"Content-Type: application/json\r\nContent-Length: " + str(len(body)).encode()
           + b"\r\n\r\n" + body)  # fmt: skip
    try:
        response = await asyncio.to_thread(_http, port, raw)
    finally:
        await receiver_module.stop_otlp_receiver()
    head, _, payload = response.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 200 OK\r\nX-Powered-By: Express\r\n")
    assert payload == b'{"partialSuccess":{}}'
    [span] = await get_trace_store().get_spans("a" * 32)
    assert span == {"spanId": "b" * 16, "name": "over-http", "startTime": 1700000000000,
                    # No `kind` in the request: kind_code is undefined, so it is not stored.
                    "attributes": {"k": "v", "otel.span.kind": "unspecified"}}  # fmt: skip
