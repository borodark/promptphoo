"""A provider that exports spans to the eval's OTLP receiver, the way an instrumented app
does: one OTLP/JSON request and one OTLP/protobuf request, both parented to the test case's
traceparent. Standard library only, so Node's Python provider runner can load it too.

The receiver address comes from the provider config (`receiver`); span times are fixed so
the stored rows are the same from run to run.
"""

import base64
import json
import struct
import urllib.request


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(number, wire, payload):
    tag = _varint(number << 3 | wire)
    if wire == 0:
        return tag + _varint(payload)
    if wire == 1:
        return tag + struct.pack("<Q", payload)
    return tag + _varint(len(payload)) + payload


def _kv(key, any_value):
    return _field(1, 2, key.encode()) + _field(2, 2, any_value)


def _protobuf_request(trace_id, parent_span_id, test_idx):
    attrs = b"".join(
        _field(9, 2, kv)  # Span.attributes
        for kv in (
            _kv("gen_ai.request.model", _field(1, 2, b"fake-model")),
            _kv("gen_ai.usage.input_tokens", _field(3, 0, 12)),
            _kv("temperature", _field(4, 1, struct.unpack("<Q", struct.pack("<d", 0.25))[0])),
            _kv("streaming", _field(2, 0, 1)),
            _kv("api.secret_key", _field(1, 2, b"sk-should-be-redacted")),
        )
    )
    span = (
        _field(1, 2, bytes.fromhex(trace_id))
        + _field(2, 2, bytes.fromhex(f"{test_idx + 1:02x}" + "b0" * 7))
        + _field(4, 2, bytes.fromhex(parent_span_id))
        + _field(5, 2, b"llm.chat")
        + _field(6, 0, 3)
        + _field(7, 1, 1_700_000_000_100_000_000)
        + _field(8, 1, 1_700_000_000_350_000_000)
        + attrs
        + _field(15, 2, _field(3, 0, 1))
    )
    scope = _field(1, 2, b"fake.instrumentation") + _field(2, 2, b"0.1.0")
    resource = _field(1, 2, _kv("service.name", _field(1, 2, b"posting-provider")))
    scope_spans = _field(1, 2, scope) + _field(2, 2, span)
    return _field(1, 2, _field(1, 2, resource) + _field(2, 2, scope_spans))


def _json_request(trace_id, parent_span_id, test_idx):
    agent_id = f"{test_idx + 1:02x}" + "a0" * 7
    tool_id = f"{test_idx + 1:02x}" + "c0" * 7
    trace_b64 = base64.b64encode(bytes.fromhex(trace_id)).decode()
    return {
        "resourceSpans": [{
            "resource": {"attributes": [
                {"key": "service.name", "value": {"stringValue": "posting-provider"}},
            ]},
            "scopeSpans": [{
                "scope": {"name": "fake.agent", "version": "1.2.3"},
                "spans": [
                    {"traceId": trace_b64, "spanId": agent_id, "parentSpanId": parent_span_id,
                     "name": "agent.run", "kind": 1,
                     "startTimeUnixNano": "1700000000000000000",
                     "endTimeUnixNano": "1700000000500000000",
                     "attributes": [
                         {"key": "gen_ai.operation.name",
                          "value": {"stringValue": "invoke_agent"}},
                         {"key": "steps", "value": {"intValue": "3"}},
                         {"key": "tags", "value": {"arrayValue": {"values": [
                             {"stringValue": "a"}, {"intValue": 2}]}}},
                         {"key": "config", "value": {"kvlistValue": {"values": [
                             {"key": "depth", "value": {"doubleValue": 1.5}},
                             {"key": "client_secret", "value": {"stringValue": "hidden"}},
                         ]}}},
                     ],
                     "status": {"code": 1}},
                    {"traceId": trace_b64, "spanId": base64.b64encode(bytes.fromhex(tool_id))
                     .decode(), "parentSpanId": agent_id,
                     "name": "tool.search", "kind": 3,
                     "startTimeUnixNano": "1700000000400000000",
                     "endTimeUnixNano": "1700000000450000000",
                     "attributes": [{"key": "tool.name", "value": {"stringValue": "search"}}],
                     "status": {"code": 2, "message": "search failed"}},
                ],
            }],
        }],
    }  # fmt: skip


def _post(url, body, content_type):
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": content_type}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.read().decode()


def call_api(prompt, options, context):
    receiver = options["config"]["receiver"]
    _, trace_id, parent_span_id, _ = context["traceparent"].split("-")
    test_idx = int(context["vars"].get("idx", 0))
    results = [
        _post(f"{receiver}/v1/traces",
              json.dumps(_json_request(trace_id, parent_span_id, test_idx)).encode(),
              "application/json"),
        _post(f"{receiver}/v1/traces",
              _protobuf_request(trace_id, parent_span_id, test_idx), "application/x-protobuf"),
    ]  # fmt: skip
    return {"output": json.dumps(results)}
