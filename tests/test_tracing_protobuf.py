"""Ports test/tracing/protobuf.test.ts, and replays the protobufjs golden file
(tests/fixtures/tracing/protobuf_golden.jsonl, made by tools/otlp_protobuf_golden.py): valid
and mutated ExportTraceServiceRequests with what protobufjs decoded them to, or its error."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import pytest

from promptphoo.tracing.protobuf import (
    bytes_to_base64,
    bytes_to_hex,
    decode_export_trace_service_request,
)

GOLDEN = Path(__file__).parent / "fixtures" / "tracing" / "protobuf_golden.jsonl"


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _field(number: int, wire: int, payload: bytes | int) -> bytes:
    tag = _varint(number << 3 | wire)
    if wire == 0:
        assert isinstance(payload, int)
        return tag + _varint(payload)
    if wire == 1:
        assert isinstance(payload, int)
        return tag + payload.to_bytes(8, "little")
    assert isinstance(payload, bytes)
    return tag + _varint(len(payload)) + payload


def _str(number: int, value: str) -> bytes:
    return _field(number, 2, value.encode())


def _kv(key: str, any_value: bytes) -> bytes:
    return _str(1, key) + _field(2, 2, any_value)


TRACE_ID = bytes.fromhex("aabbccddeeff00112233445566778899")
SPAN_ID = bytes.fromhex("1122334455667788")


def _span(**extra: bytes) -> bytes:
    body = (
        _field(1, 2, TRACE_ID)
        + _field(2, 2, SPAN_ID)
        + _str(5, "test-span")
        + _field(6, 0, 1)
        + _field(7, 1, 1_000_000_000)
        + _field(8, 1, 2_000_000_000)
    )
    return body + b"".join(extra.values())


def _request(*spans: bytes, resource: bytes = b"") -> bytes:
    scope_spans = _field(1, 2, _str(1, "test-scope")) + b"".join(_field(2, 2, s) for s in spans)
    resource_spans = (_field(1, 2, resource) if resource else b"") + _field(2, 2, scope_spans)
    return _field(1, 2, resource_spans)


# --- bytesToHex ------------------------------------------------------------------------------


def test_bytes_to_hex():
    assert bytes_to_hex(bytes([0xDE, 0xAD, 0xBE, 0xEF]), 8) == "deadbeef"
    assert bytes_to_hex(bytes([0x00, 0x01]), 8) == "00000001"
    assert bytes_to_hex(None, 8) == "00000000"
    assert bytes_to_hex(b"", 8) == "00000000"
    trace = bytes([0x12, 0x34, 0x56, 0x78, 0x90] * 3 + [0x12])
    assert bytes_to_hex(trace, 32) == "12345678901234567890123456789012"


def test_bytes_to_base64():
    assert bytes_to_base64(SPAN_ID) == base64.b64encode(SPAN_ID).decode()


# --- decodeExportTraceServiceRequest -----------------------------------------------------------


def test_decode_a_simple_request():
    decoded = decode_export_trace_service_request(_request(_span()))
    [resource_spans] = decoded["resourceSpans"]
    [scope_spans] = resource_spans["scopeSpans"]
    assert scope_spans["scope"]["name"] == "test-scope"
    [span] = scope_spans["spans"]
    assert bytes_to_hex(span["traceId"], 32) == TRACE_ID.hex()
    assert bytes_to_hex(span["spanId"], 16) == SPAN_ID.hex()
    assert span["name"] == "test-span"
    assert span["kind"] == 1
    assert span["startTimeUnixNano"] == 1_000_000_000
    assert span["endTimeUnixNano"] == 2_000_000_000


def test_decode_multiple_spans():
    second = _span().replace(b"test-span", b"other-one")
    decoded = decode_export_trace_service_request(_request(_span(), second))
    spans = decoded["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert [s["name"] for s in spans] == ["test-span", "other-one"]


def test_decode_attribute_types():
    attrs = (
        _field(9, 2, _kv("string.attr", _str(1, "hello")))
        + _field(9, 2, _kv("bool.attr", _field(2, 0, 1)))
        + _field(9, 2, _kv("int.attr", _field(3, 0, 42)))
        + _field(9, 2, _kv("double.attr", _field(4, 1, int.from_bytes(
            __import__("struct").pack("<d", 3.14), "little"))))
        + _field(9, 2, _kv("array.attr", _field(5, 2, _field(1, 2, _str(1, "a")))))
    )  # fmt: skip
    [span] = decode_export_trace_service_request(_request(_span(a=attrs)))["resourceSpans"][0][
        "scopeSpans"
    ][0]["spans"]
    values = {a["key"]: a["value"] for a in span["attributes"]}
    assert values["string.attr"]["stringValue"] == "hello"
    assert values["bool.attr"]["boolValue"] is True
    assert values["int.attr"]["intValue"] == 42
    assert math.isclose(values["double.attr"]["doubleValue"], 3.14)
    assert values["array.attr"]["arrayValue"]["values"][0]["stringValue"] == "a"


def test_decode_resource_attributes():
    resource = _field(1, 2, _kv("service.name", _str(1, "test-service")))
    decoded = decode_export_trace_service_request(_request(_span(), resource=resource))
    [attr] = decoded["resourceSpans"][0]["resource"]["attributes"]
    assert attr["key"] == "service.name"
    assert attr["value"]["stringValue"] == "test-service"


def test_invalid_data_raises():
    with pytest.raises(ValueError, match="(?i)invalid protobuf"):
        decode_export_trace_service_request(b"not valid protobuf data")


def test_nested_length_crossing_the_parent_boundary():
    with pytest.raises(ValueError, match="(?i)invalid protobuf"):
        decode_export_trace_service_request(bytes([0x0A, 0x01, 0x12, 0x00]))


@pytest.mark.parametrize(
    "prefix",
    [[0x0A], [0x0A, 0x06, 0x1A], [0x0A, 0x0A, 0x12, 0x08, 0x12, 0x06, 0x0A], [0x7A]],
    ids=["resourceSpans", "schemaUrl", "traceId", "unknown"],
)
def test_overflowing_length_prefix(prefix):
    # 2^32 must not wrap to a zero-byte field or message.
    with pytest.raises(ValueError, match="^Invalid protobuf data: invalid length encoding$"):
        decode_export_trace_service_request(bytes([*prefix, 0x80, 0x80, 0x80, 0x80, 0x10]))


def test_empty_resource_spans():
    assert decode_export_trace_service_request(b"") == {"resourceSpans": []}


def test_bytearray_and_memoryview_input():
    payload = _request(_span())
    expected = decode_export_trace_service_request(payload)
    assert decode_export_trace_service_request(bytearray(payload)) == expected
    assert decode_export_trace_service_request(memoryview(payload)) == expected


# --- golden replay ------------------------------------------------------------------------------


def _to_plain(v):
    if isinstance(v, bytes):
        return {"$bytes": v.hex()}
    if isinstance(v, dict):
        return {k: _to_plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_to_plain(x) for x in v]
    if isinstance(v, float):
        if not math.isfinite(v):
            return {"$number": "Infinity" if v > 0 else ("-Infinity" if v < 0 else "NaN")}
        if v == 0 and math.copysign(1, v) < 0:
            return {"$number": "-0"}
    return v


def _golden_cases():
    with GOLDEN.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_golden_file_covers_valid_and_rejected_payloads():
    cases = _golden_cases()
    assert len(cases) >= 200
    assert sum("ok" in c["node"] for c in cases) >= 50
    assert sum("error" in c["node"] for c in cases) >= 50


def test_decoder_matches_protobufjs_golden():
    from promptphoo.jsvalues import json_stringify

    mismatches = []
    for case in _golden_cases():
        payload = base64.b64decode(case["payload"])
        try:
            got = {"ok": _to_plain(decode_export_trace_service_request(payload))}
        except ValueError as e:
            got = {"error": str(e)}
        # Round-trip through JSON.stringify so numbers print as Node prints them.
        if json.loads(json_stringify(got)) != case["node"]:
            mismatches.append(case["payload"])
    assert mismatches == []
