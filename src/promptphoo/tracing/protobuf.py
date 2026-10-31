"""OTLP protobuf decoding for `Content-Type: application/x-protobuf`. Ports src/tracing/protobuf.ts.

Upstream decodes with protobufjs 8 over the vendored `.proto` files in src/tracing/proto. Here a
small wire-format reader decodes just `ExportTraceServiceRequest` and the messages it contains,
reproducing what protobufjs does with them:

- the generated decoder's rules: a field read with the wrong wire type is skipped as unknown,
  unknown fields are skipped, a repeated singular message field merges into the earlier one,
  the last member of a oneof wins, and proto3 scalars equal to their default stay unset;
- protobufjs's `Reader` checks and error messages (`index out of range: 3 + 10 > 12`,
  `invalid wire type 6 at offset 9`, `max depth exceeded` past 100 nested messages, ...);
- `toObject({longs: Number, bytes: Uint8Array, defaults: true, arrays: true})`: every non-oneof
  field is present with its default (`""`, `b""`, `0`, `False`, `[]`, `None` for messages);
  64-bit integers become doubles (`float`) exactly as Long's `toNumber` rounds them.

Decoded objects use protobufjs's camelCase field names.
"""

from __future__ import annotations

import base64
import logging
import struct
from typing import Any

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 100  # protobufjs util.recursionLimit (protoc's default)


class ProtobufError(Exception):
    """A protobufjs decode error; the message is protobufjs's."""


# --- schema ----------------------------------------------------------------------------------

# kind: a scalar type name, or ("message", TypeName). Flags: "repeated", "oneof".
_SCALAR_WIRE = {"string": 2, "bytes": 2, "bool": 0, "int64": 0, "uint32": 0, "int32": 0,
                "fixed64": 1, "fixed32": 5, "double": 1}  # fmt: skip
_DEFAULTS: dict[str, Any] = {"string": "", "bytes": b"", "bool": False, "int64": 0.0,
                             "uint32": 0, "int32": 0, "fixed64": 0.0, "fixed32": 0,
                             "double": 0.0}  # fmt: skip

_Field = tuple[str, Any, str]  # (name, kind, flags)

_SCHEMA: dict[str, dict[int, _Field]] = {
    "ExportTraceServiceRequest": {1: ("resourceSpans", ("message", "ResourceSpans"), "repeated")},
    "ResourceSpans": {
        1: ("resource", ("message", "Resource"), ""),
        2: ("scopeSpans", ("message", "ScopeSpans"), "repeated"),
        3: ("schemaUrl", "string", ""),
    },
    "Resource": {
        1: ("attributes", ("message", "KeyValue"), "repeated"),
        2: ("droppedAttributesCount", "uint32", ""),
    },
    "ScopeSpans": {
        1: ("scope", ("message", "InstrumentationScope"), ""),
        2: ("spans", ("message", "Span"), "repeated"),
        3: ("schemaUrl", "string", ""),
    },
    "InstrumentationScope": {
        1: ("name", "string", ""),
        2: ("version", "string", ""),
        3: ("attributes", ("message", "KeyValue"), "repeated"),
        4: ("droppedAttributesCount", "uint32", ""),
    },
    "Span": {
        1: ("traceId", "bytes", ""),
        2: ("spanId", "bytes", ""),
        3: ("traceState", "string", ""),
        4: ("parentSpanId", "bytes", ""),
        16: ("flags", "fixed32", ""),
        5: ("name", "string", ""),
        6: ("kind", "int32", ""),  # SpanKind enum (open)
        7: ("startTimeUnixNano", "fixed64", ""),
        8: ("endTimeUnixNano", "fixed64", ""),
        9: ("attributes", ("message", "KeyValue"), "repeated"),
        10: ("droppedAttributesCount", "uint32", ""),
        11: ("events", ("message", "Event"), "repeated"),
        12: ("droppedEventsCount", "uint32", ""),
        13: ("links", ("message", "Link"), "repeated"),
        14: ("droppedLinksCount", "uint32", ""),
        15: ("status", ("message", "Status"), ""),
    },
    "Event": {
        1: ("timeUnixNano", "fixed64", ""),
        2: ("name", "string", ""),
        3: ("attributes", ("message", "KeyValue"), "repeated"),
        4: ("droppedAttributesCount", "uint32", ""),
    },
    "Link": {
        1: ("traceId", "bytes", ""),
        2: ("spanId", "bytes", ""),
        3: ("traceState", "string", ""),
        4: ("attributes", ("message", "KeyValue"), "repeated"),
        5: ("droppedAttributesCount", "uint32", ""),
        6: ("flags", "fixed32", ""),
    },
    "Status": {2: ("message", "string", ""), 3: ("code", "int32", "")},  # StatusCode enum
    "AnyValue": {
        1: ("stringValue", "string", "oneof"),
        2: ("boolValue", "bool", "oneof"),
        3: ("intValue", "int64", "oneof"),
        4: ("doubleValue", "double", "oneof"),
        5: ("arrayValue", ("message", "ArrayValue"), "oneof"),
        6: ("kvlistValue", ("message", "KeyValueList"), "oneof"),
        7: ("bytesValue", "bytes", "oneof"),
    },
    "ArrayValue": {1: ("values", ("message", "AnyValue"), "repeated")},
    "KeyValueList": {1: ("values", ("message", "KeyValue"), "repeated")},
    "KeyValue": {1: ("key", "string", ""), 2: ("value", ("message", "AnyValue"), "")},
}

_ONEOF_MEMBERS = {
    type_name: [name for name, _kind, flags in fields.values() if flags == "oneof"]
    for type_name, fields in _SCHEMA.items()
}


# --- reader ----------------------------------------------------------------------------------


class _Reader:
    """protobufjs Reader over a byte buffer; `len` is the current (nested) read limit."""

    __slots__ = ("buf", "len", "pos")

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0
        self.len = len(buf)

    def out_of_range(self, write_length: int = 0) -> ProtobufError:
        return ProtobufError(f"index out of range: {self.pos} + {write_length or 1} > {self.len}")

    def _varint32_near_end(self) -> int:
        value = 0
        for i in range(4):
            if self.pos >= self.len:
                raise self.out_of_range()
            b = self.buf[self.pos]
            self.pos += 1
            value = (value | (b & 127) << (i * 7)) & 0xFFFFFFFF
            if b < 128:
                return value
        raise self.out_of_range()

    def _varint32_head(self) -> tuple[int, int, bool]:
        """The first four bytes of a 32-bit varint: (value, pos, done)."""
        buf, pos = self.buf, self.pos
        value = 0
        for i in range(4):
            b = buf[pos]
            pos += 1
            value |= (b & 127) << (i * 7)
            if b < 128:
                return value, pos, True
        return value, pos, False

    def uint32(self) -> int:
        if self.len - self.pos < 5:
            if self.pos >= self.len:
                raise self.out_of_range()
            if self.buf[self.pos] >= 128:
                return self._varint32_near_end()
        value, pos, done = self._varint32_head()
        if done:
            self.pos = pos
            return value
        b = self.buf[pos]
        pos += 1
        value = (value | (b & 15) << 28) & 0xFFFFFFFF
        if b < 128:
            self.pos = pos
            return value
        for _ in range(5):
            if pos >= self.len:
                self.pos = pos
                raise self.out_of_range()
            b = self.buf[pos]
            pos += 1
            if b < 128:
                self.pos = pos
                return value
        self.pos = pos
        raise ProtobufError("invalid varint encoding")

    def tag(self) -> int:
        if self.len - self.pos < 5:
            if self.pos >= self.len:
                raise self.out_of_range()
            if self.buf[self.pos] >= 128:
                return self._varint32_near_end()
        value, pos, done = self._varint32_head()
        if done:
            self.pos = pos
            return value
        b = self.buf[pos]
        value = (value | (b & 15) << 28) & 0xFFFFFFFF
        if b < 128 and (b & 112) == 0:
            self.pos = pos + 1
            return value
        self.pos = pos + 1
        raise ProtobufError("invalid tag encoding")

    def size(self) -> int:
        if self.len - self.pos < 5:
            if self.pos >= self.len:
                raise self.out_of_range()
            if self.buf[self.pos] >= 128:
                return self._varint32_near_end()
        value, pos, done = self._varint32_head()
        if done:
            self.pos = pos
            return value
        self.pos = pos + 1
        b = self.buf[pos]
        if b > 7:
            raise ProtobufError("invalid length encoding")
        return (value | b << 28) & 0xFFFFFFFF

    def int32(self) -> int:
        value = self.uint32()
        return value - (1 << 32) if value & 0x80000000 else value

    def _long_varint(self) -> tuple[int, int]:
        lo = hi = 0
        buf = self.buf
        if self.len - self.pos > 4:  # fast route (lo)
            for i in range(4):
                b = buf[self.pos]
                self.pos += 1
                lo = (lo | (b & 127) << (i * 7)) & 0xFFFFFFFF
                if b < 128:
                    return lo, hi
            b = buf[self.pos]
            self.pos += 1
            lo = (lo | (b & 127) << 28) & 0xFFFFFFFF
            hi = (hi | (b & 127) >> 4) & 0xFFFFFFFF
            if b < 128:
                return lo, hi
        else:
            for i in range(4):
                if self.pos >= self.len:
                    raise self.out_of_range()
                b = buf[self.pos]
                self.pos += 1
                lo = (lo | (b & 127) << (i * 7)) & 0xFFFFFFFF
                if b < 128:
                    return lo, hi
            raise self.out_of_range()
        fast = self.len - self.pos > 4
        for i in range(5):  # 6th..10th
            if not fast and self.pos >= self.len:
                raise self.out_of_range()
            b = buf[self.pos]
            self.pos += 1
            hi = (hi | (b & 127) << (i * 7 + 3)) & 0xFFFFFFFF
            if b < 128:
                return lo, hi
        raise ProtobufError("invalid varint encoding")

    def int64(self) -> float:
        lo, hi = self._long_varint()
        value = lo + hi * 4294967296
        if hi >> 31:
            value -= 1 << 64
        return float(value)  # LongBits.toNumber: one correctly rounded double

    def bool(self) -> bool:
        value = False
        for _ in range(10):
            if self.pos >= self.len:
                raise self.out_of_range()
            b = self.buf[self.pos]
            self.pos += 1
            if b & 127:
                value = True
            if b < 128:
                return value
        raise ProtobufError("invalid varint encoding")

    def fixed32(self) -> int:
        if self.pos + 4 > self.len:
            raise self.out_of_range(4)
        (value,) = struct.unpack_from("<I", self.buf, self.pos)
        self.pos += 4
        return value

    def fixed64(self) -> float:
        if self.pos + 8 > self.len:
            raise self.out_of_range(8)
        (value,) = struct.unpack_from("<Q", self.buf, self.pos)
        self.pos += 8
        return float(value)

    def double(self) -> float:
        if self.pos + 8 > self.len:
            raise self.out_of_range(4)  # sic: protobufjs reports 4 here
        (value,) = struct.unpack_from("<d", self.buf, self.pos)
        self.pos += 8
        return value

    def bytes(self) -> bytes:
        length = self.size()
        start, end = self.pos, self.pos + length
        if end > self.len:
            raise self.out_of_range(length)
        self.pos = end
        return bytes(self.buf[start:end])

    def string(self) -> str:
        """stringVerify: proto3 strings are UTF-8 validated (TextDecoder fatal, ignoreBOM)."""
        raw = self.bytes()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ProtobufError("The encoded data was not valid for encoding utf-8") from None

    def skip(self, length: int | None = None) -> None:
        if length is not None:
            if self.pos + length > self.len:
                raise self.out_of_range(length)
            self.pos += length
            return
        while True:
            if self.pos >= self.len:
                raise self.out_of_range()
            b = self.buf[self.pos]
            self.pos += 1
            if not b & 128:
                return

    def skip_type(self, wire_type: int, depth: int = 0, field_number: int | None = None) -> None:
        if depth > RECURSION_LIMIT:
            raise ProtobufError("max depth exceeded")
        if field_number == 0:
            raise ProtobufError("illegal tag: field number 0")
        if wire_type == 0:
            self.skip()
        elif wire_type == 1:
            self.skip(8)
        elif wire_type == 2:
            self.skip(self.size())
        elif wire_type == 3:
            while True:
                tag = self.tag()
                nested_field = tag >> 3
                wire_type = tag & 7
                if not nested_field:
                    raise ProtobufError("illegal tag: field number 0")
                if wire_type == 4:
                    if field_number is not None and nested_field != field_number:
                        raise ProtobufError("invalid end group tag")
                    break
                self.skip_type(wire_type, depth + 1, nested_field)
        elif wire_type == 5:
            self.skip(4)
        else:
            raise ProtobufError(f"invalid wire type {wire_type} at offset {self.pos}")


# --- decoder ---------------------------------------------------------------------------------


def _read_scalar(r: _Reader, kind: str) -> Any:
    if kind == "string":
        return r.string()
    if kind == "bytes":
        return r.bytes()
    if kind == "bool":
        return r.bool()
    if kind == "int64":
        return r.int64()
    if kind == "uint32":
        return r.uint32()
    if kind == "int32":
        return r.int32()
    if kind == "fixed64":
        return r.fixed64()
    if kind == "fixed32":
        return r.fixed32()
    return r.double()


def _decode(
    type_name: str,
    r: _Reader,
    length: int | None = None,
    depth: int = 0,
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The protobufjs-generated `decode(r, l, z, q, g)` for one message type."""
    if depth > RECURSION_LIMIT:
        raise ProtobufError("max depth exceeded")
    outer_len = r.len
    if length is None:
        end = r.len
    else:
        end = r.pos + length
        if end > r.len:
            raise ProtobufError("index out of range")
        r.len = end
    message = target if target is not None else {}
    fields = _SCHEMA[type_name]
    while r.pos < end:
        tag = r.tag()
        wire_type = tag & 7
        field_number = tag >> 3
        field = fields.get(field_number)
        if field is not None:
            name, kind, flags = field
            if isinstance(kind, tuple):
                if wire_type == 2:
                    nested = kind[1]
                    if flags == "repeated":
                        message.setdefault(name, []).append(_decode(nested, r, r.size(), depth + 1))
                    else:
                        message[name] = _decode(nested, r, r.size(), depth + 1, message.get(name))
                    if flags == "oneof":
                        _set_oneof(type_name, message, name)
                    continue
            elif wire_type == _SCALAR_WIRE[kind]:
                value = _read_scalar(r, kind)
                if flags == "oneof":
                    message[name] = value
                    _set_oneof(type_name, message, name)
                elif value or (kind == "double" and _is_negative_zero(value)):
                    message[name] = value
                else:
                    message.pop(name, None)
                continue
        r.skip_type(wire_type, depth, field_number)
    if length is not None:
        if r.pos != end:
            raise ProtobufError("index out of range")
        r.len = outer_len
    return message


def _is_negative_zero(value: float) -> bool:
    return value == 0 and struct.pack("<d", value) != b"\x00" * 8


def _set_oneof(type_name: str, message: dict[str, Any], name: str) -> None:
    """protobufjs's oneOfSetter: setting one member deletes the others."""
    for member in _ONEOF_MEMBERS[type_name]:
        if member != name:
            message.pop(member, None)


def _to_object(type_name: str, message: dict[str, Any]) -> dict[str, Any]:
    """Type.toObject with `defaults: true, arrays: true`: defaults for every non-oneof field."""
    out: dict[str, Any] = {}
    for name, kind, flags in _SCHEMA[type_name].values():
        present = name in message
        if flags == "oneof" and not present:
            continue
        if isinstance(kind, tuple):
            if flags == "repeated":
                out[name] = [_to_object(kind[1], m) for m in message.get(name, [])]
            else:
                out[name] = _to_object(kind[1], message[name]) if present else None
        elif present:
            out[name] = message[name]
        else:
            out[name] = _DEFAULTS[kind]
    return out


def bytes_to_hex(data: bytes | None, expected_length: int) -> str:
    """bytesToHex: lowercase hex left-padded with zeros; all zeros for missing/empty bytes."""
    if not data:
        return "0" * expected_length
    return data.hex().rjust(expected_length, "0")


def bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def decode_export_trace_service_request(data: bytes | bytearray | memoryview) -> dict[str, Any]:
    """decodeExportTraceServiceRequest. Raises ValueError("Invalid protobuf data: ...")."""
    try:
        message = _decode("ExportTraceServiceRequest", _Reader(bytes(data)))
        decoded = _to_object("ExportTraceServiceRequest", message)
        logger.debug(
            "[Protobuf] Decoded ExportTraceServiceRequest with %d resource spans",
            len(decoded.get("resourceSpans") or []),
        )
        return decoded
    except ProtobufError as error:
        logger.error("[Protobuf] Failed to decode ExportTraceServiceRequest: Error: %s", error)
        raise ValueError(f"Invalid protobuf data: {error}") from error
