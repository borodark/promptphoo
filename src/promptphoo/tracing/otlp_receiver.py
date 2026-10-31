"""The OTLP HTTP trace receiver. Ports src/tracing/otlpReceiver.ts.

Upstream is an express 5 app. Here a small asyncio HTTP/1.1 server (stdlib only) replays the
same middleware pipeline, so clients see the same status codes and JSON bodies:

- `POST /v1/traces`: OTLP JSON (`application/json`) or protobuf (`application/x-protobuf`),
  each only when listed in `acceptFormats`; any other content type is 415
  `{"error":"Unsupported content type"}`. Success is 200 `{"partialSuccess":{}}`.
- `POST /v1/logs`: OTLP JSON logs only (415 otherwise); each log record becomes a 1 ms span.
- `GET /health` (`{"status":"ok"}`), `GET /v1/traces` (service info), `GET /debug/status`.
- body-parser behaviour: a 10 MB limit, `Content-Encoding` gzip/deflate/identity, strict JSON
  (an object or array), utf-* charsets, `{}` for an empty JSON body. Malformed JSON is 400
  `{"error":"Invalid JSON"}`, a bad protobuf payload is 400 with protobufjs's message, and every
  other parser or processing failure is 500 `{"error":"Internal server error"}`, because
  upstream's global error handler maps them all there.
- express routing: paths are case-insensitive, a trailing slash is allowed, `HEAD` runs the
  `GET` route without a body, and unknown routes get finalhandler's 404 page.

The JSON request handling evaluates upstream's property accesses with JS semantics (see
`_get`), so malformed payloads fail, or slip through, exactly where Node's do.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import math
import os
import re
import secrets
import time
import zlib
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import (
    UNDEFINED,
    is_truthy,
    js_string,
    json_stringify,
    node_base64_decode,
)
from promptphoo.tracing.protobuf import (
    bytes_to_base64,
    bytes_to_hex,
    decode_export_trace_service_request,
)
from promptphoo.tracing.sanitize_attributes import js_length, js_slice
from promptphoo.tracing.store import TraceStore, get_trace_store

logger = logging.getLogger(__name__)

PROMPTFOO_RESOURCE_ATTR_TRACE_ID = "promptfoo.trace_id"
PROMPTFOO_RESOURCE_ATTR_PARENT_SPAN_ID = "promptfoo.parent_span_id"

# Log event names we don't want cluttering traces (internal, not actionable).
LOG_EVENT_NAME_DENYLIST = frozenset({"claude_code.tracing"})
# Nominal span duration for a log-derived span, so it renders as a thin bar.
LOG_SPAN_DURATION_MS = 1
# Longest log body used as a span name when no event.name attribute is present.
MAX_BODY_AS_NAME_LENGTH = 128
# OTEL severityNumber >= 17 corresponds to ERROR and above.
SEVERITY_NUMBER_ERROR = 17
# Hard cap on the otel.log.body attribute.
MAX_LOG_BODY_ATTR_LENGTH = 8192

BODY_LIMIT_BYTES = 10 * 1024 * 1024  # express.json / express.raw `limit: '10mb'`

SPAN_KIND_MAP = {"0": "unspecified", "1": "internal", "2": "server", "3": "client",
                 "4": "producer", "5": "consumer"}  # fmt: skip
DEFAULT_ACCEPT_FORMATS = ("json", "protobuf")
OTLP_CONTENT_TYPES = {"json": "application/json", "protobuf": "application/x-protobuf"}

_BASE64_ZERO_SPAN_ID = re.compile(r"A{11,12}=?\Z")
_BASE64_ZERO_TRACE_ID = re.compile(r"A{22}(?:==)?\Z")
_ZEROS = re.compile(r"0+\Z")
_HEX_I = re.compile(r"[0-9a-fA-F]+\Z")
_HEX = re.compile(r"[0-9a-f]+\Z")

_PROCESS_START = time.monotonic()


# --- JS value semantics ------------------------------------------------------------------------


def _get(obj: Any, key: str) -> Any:
    """`obj.key` in JS: TypeError on null/undefined, `UNDEFINED` for an absent property."""
    if obj is None or obj is UNDEFINED:
        kind = "null" if obj is None else "undefined"
        raise TypeError(f"Cannot read properties of {kind} (reading '{key}')")
    if isinstance(obj, dict):
        return obj.get(key, UNDEFINED)
    if key == "length":
        if isinstance(obj, str):
            return js_length(obj)
        if isinstance(obj, list):
            return len(obj)
    return UNDEFINED


def _opt(obj: Any, key: str) -> Any:
    """`obj?.key`."""
    if obj is None or obj is UNDEFINED:
        return UNDEFINED
    return _get(obj, key)


def _nullish(value: Any) -> bool:
    return value is None or value is UNDEFINED


def _iter(value: Any) -> Iterable[Any]:
    """`for (const x of value)`: arrays and strings are iterable, everything else throws."""
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        return list(value)
    raise TypeError(f"{_js_type_name(value)} is not iterable")


def _js_type_name(value: Any) -> str:
    if value is UNDEFINED:
        return "undefined"
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    return js_string(value)


def _property_key(key: Any) -> str:
    return key if isinstance(key, str) else js_string(key)


_JS_WHITESPACE = " \t\n\v\f\r           " \
    "      　﻿"  # fmt: skip
_DECIMAL = re.compile(r"[+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)\Z")
_RADIX = re.compile(r"0([xXoObB])([0-9a-fA-F]+)\Z")


def js_number(value: Any) -> float:
    """`Number(value)`."""
    if value is UNDEFINED:
        return math.nan
    if value is None or value is False:
        return 0.0
    if value is True:
        return 1.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, list):
        return js_number(js_string(value))
    if not isinstance(value, str):
        return math.nan
    text = value.strip(_JS_WHITESPACE)
    if text == "":
        return 0.0
    if text in ("Infinity", "+Infinity"):
        return math.inf
    if text == "-Infinity":
        return -math.inf
    radix = _RADIX.match(text)
    if radix:
        base = {"x": 16, "o": 8, "b": 2}[radix.group(1).lower()]
        try:
            return float(int(radix.group(2), base))
        except ValueError:
            return math.nan
    if _DECIMAL.match(text):
        return float(text)
    return math.nan


def _js_div(value: float, divisor: float) -> float:
    return value / divisor if math.isfinite(value) else value


def _as_js_numbers(value: Any) -> Any:
    """JSON.parse yields doubles: integers beyond 2**53 lose precision as they do in Node."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) < 2**53 else float(value)
    if isinstance(value, dict):
        return {k: _as_js_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_js_numbers(v) for v in value]
    return value


# --- helpers -----------------------------------------------------------------------------------


def _truncate_log_body(value: Any) -> Any:
    if not isinstance(value, str) or js_length(value) <= MAX_LOG_BODY_ATTR_LENGTH:
        return value
    return f"{js_slice(value, MAX_LOG_BODY_ATTR_LENGTH - 15)}... [truncated]"


def _is_zero_span_id(value: Any) -> bool:
    text = js_string(value)
    return bool(_ZEROS.match(text) or _BASE64_ZERO_SPAN_ID.match(text))


def _is_zero_trace_id(value: Any) -> bool:
    text = js_string(value)
    return bool(_ZEROS.match(text) or _BASE64_ZERO_TRACE_ID.match(text))


def _random_span_id() -> str:
    return secrets.token_hex(8)


def _get_string_attribute(attributes: dict[str, Any], key: str) -> str | None:
    value = attributes.get(key)
    return value if isinstance(value, str) else None


def _resolve_log_span_name(attributes: dict[str, Any], body_value: Any) -> str:
    event_name = attributes.get("event.name", UNDEFINED)
    if _nullish(event_name):
        event_name = attributes.get("claude_code.event.name", UNDEFINED)
    if isinstance(event_name, str) and event_name:
        return event_name
    if (
        isinstance(body_value, str)
        and body_value
        and js_length(body_value) <= MAX_BODY_AS_NAME_LENGTH
    ):
        return body_value
    return "otel.log"


def _normalize_accept_formats(accept_formats: Iterable[str] | None) -> list[str]:
    formats = DEFAULT_ACCEPT_FORMATS if accept_formats is None else accept_formats
    normalized = list(dict.fromkeys(formats))
    return normalized if normalized else list(DEFAULT_ACCEPT_FORMATS)


def normalize_redact_attribute_patterns(redact_attributes: Any) -> list[str]:
    if _nullish(redact_attributes):
        return []
    return list(
        dict.fromkeys(
            p
            for p in (
                pattern.strip().lower() if isinstance(pattern, str) else ""
                for pattern in redact_attributes
            )
            if p
        )
    )


def _normalize_command_tool_names(command_tool_names: Any) -> list[str] | None:
    names = [] if _nullish(command_tool_names) else command_tool_names
    normalized = list(
        dict.fromkeys(
            n for n in (name.strip() if isinstance(name, str) else "" for name in names) if n
        )
    )
    return normalized or None


def _normalize_trace_policy(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "commandToolNames": _normalize_command_tool_names(policy.get("commandToolNames")),
        "redactAttributePatterns": normalize_redact_attribute_patterns(
            policy.get("redactAttributes")
        ),
    }


def _metadata_from_trace_policy(policy: dict[str, Any]) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    if policy.get("commandToolNames"):
        metadata["commandToolNames"] = policy["commandToolNames"]
    if policy["redactAttributePatterns"]:
        metadata["otlpHttpRedactAttributes"] = policy["redactAttributePatterns"]
    return metadata or None


def get_request_format(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    mime_type = content_type.split(";", 1)[0].strip().lower()
    if mime_type == OTLP_CONTENT_TYPES["json"]:
        return "json"
    if mime_type == OTLP_CONTENT_TYPES["protobuf"]:
        return "protobuf"
    return None


def _error_message(error: BaseException) -> str:
    return str(error)


# --- HTTP plumbing -----------------------------------------------------------------------------


class HttpError(Exception):
    """An error passed to express's `next(err)`; `syntax` marks body-parser's JSON errors."""

    def __init__(self, status: int, message: str, *, syntax: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.syntax = syntax


class Request:
    def __init__(self, method: str, target: str, headers: dict[str, str], body: bytes | None):
        self.method = method
        self.target = target
        self.path = urlsplit(target).path or "/"
        self.headers = headers  # lower-cased names; repeated headers joined with ", "
        self.raw_body = body
        self.body: Any = UNDEFINED


class Response:
    def __init__(self) -> None:
        self.status = 200
        self.headers: dict[str, str] = {}
        self.body = b""

    def json(self, status: int, payload: Any) -> None:
        self.status = status
        self.body = json_stringify(payload).encode("utf-8")
        self.headers["Content-Type"] = "application/json; charset=utf-8"
        self.headers["Content-Length"] = str(len(self.body))
        self.headers["ETag"] = _weak_etag(self.body)


def _weak_etag(body: bytes) -> str:
    """The `etag` package's weak ETag: `W/"<len hex>-<sha1 base64, 27 chars>"`."""
    digest = base64.b64encode(hashlib.sha1(body).digest()).decode("ascii")[:27]
    return f'W/"{len(body):x}-{digest}"'


_STATUS_TEXT = {200: "OK", 400: "Bad Request", 404: "Not Found", 413: "Payload Too Large",
                415: "Unsupported Media Type", 500: "Internal Server Error"}  # fmt: skip


def _not_found_page(method: str, target: str) -> bytes:
    """finalhandler's 404 document for `Cannot <METHOD> <path>`."""
    escaped = (
        target.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&#39;")
    )  # fmt: skip
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>Error</title>\n</head>\n<body>\n<pre>Cannot {method} {escaped}</pre>\n"
        "</body>\n</html>\n"
    ).encode()


def _route_matches(path: str, route: str) -> bool:
    """An express route: case-insensitive, optional trailing slash."""
    lowered = path.lower()
    return lowered in (route, route + "/")


def _mount_matches(path: str, mount: str) -> bool:
    """`app.use(mount, ...)`: the mount path or anything below it."""
    lowered = path.lower()
    return lowered == mount or lowered.startswith(mount + "/")


def _has_body(headers: dict[str, str]) -> bool:
    """type-is `hasBody`: a Transfer-Encoding header or a numeric Content-Length."""
    if "transfer-encoding" in headers:
        return True
    length = headers.get("content-length")
    if length is None:
        return False
    return not math.isnan(js_number(length))


def _content_charset(content_type: str | None) -> str | None:
    if not content_type:
        return None
    for param in content_type.split(";")[1:]:
        name, _, value = param.strip().partition("=")
        if name.strip().lower() == "charset":
            return value.strip().strip('"').lower()
    return None


def _read_raw_body(request: Request) -> bytes:
    """raw-body behind body-parser: inflate, then enforce the 10 MB limit."""
    raw = request.raw_body or b""
    encoding = (request.headers.get("content-encoding") or "identity").lower()
    if encoding == "identity":
        length = request.headers.get("content-length")
        if length is not None and js_number(length) > BODY_LIMIT_BYTES:
            raise HttpError(413, "request entity too large")
        if len(raw) > BODY_LIMIT_BYTES:
            raise HttpError(413, "request entity too large")
        return raw
    if encoding == "gzip":
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif encoding == "deflate":
        decompressor = zlib.decompressobj()
    else:
        # body-parser also inflates br (and zstd); Python's stdlib cannot (see D52).
        raise HttpError(415, f'unsupported content encoding "{encoding}"')
    try:
        data = decompressor.decompress(raw, BODY_LIMIT_BYTES + 1)
    except zlib.error as error:
        raise HttpError(400, str(error)) from error
    if len(data) > BODY_LIMIT_BYTES:
        raise HttpError(413, "request entity too large")
    return data


def _json_parser(request: Request) -> None:
    """express.json({limit: '10mb'}): strict mode, utf-* charsets, `{}` for an empty body."""
    if not _has_body(request.headers):
        return
    charset = _content_charset(request.headers.get("content-type")) or "utf-8"
    if not charset.startswith("utf-"):
        raise HttpError(415, f'unsupported charset "{charset.upper()}"')
    data = _read_raw_body(request)
    try:
        text = data.decode(charset, errors="replace")
    except LookupError as error:
        raise HttpError(415, f'unsupported charset "{charset.upper()}"') from error
    if text.startswith("﻿"):
        text = text[1:]
    if text == "":
        request.body = {}
        return
    first = text.lstrip(" \t\n\r")[:1]
    if first not in ("{", "["):
        raise HttpError(400, f"Unexpected token '{first}', not valid JSON", syntax=True)
    try:
        request.body = _as_js_numbers(js_json_parse(text))
    except ValueError as error:
        raise HttpError(400, str(error), syntax=True) from error


def _raw_parser(request: Request) -> None:
    """express.raw({limit: '10mb'}): the body as bytes."""
    if not _has_body(request.headers):
        return
    request.body = _read_raw_body(request)


# --- the receiver ------------------------------------------------------------------------------


class OTLPReceiver:
    def __init__(self, options: dict[str, Any] | None = None) -> None:
        options = options or {}
        self.accept_formats = _normalize_accept_formats(options.get("acceptFormats"))
        self.trace_store: TraceStore = get_trace_store()
        self.port: int | None = None
        self.server: asyncio.Server | None = None
        self.command_tool_names: list[str] | None = None
        self.redact_attribute_patterns: list[str] = []
        self.trace_policies_by_evaluation_id: dict[str, dict[str, Any]] = {}
        self._connections: set[_Connection] = set()
        self.set_command_tool_names(options.get("commandToolNames"))
        self.set_redact_attributes(options.get("redactAttributes"))
        self.register_trace_policy(options.get("tracePolicy"))
        logger.debug("[OtlpReceiver] Initializing OTLP receiver")

    # --- policy --------------------------------------------------------------------------

    def set_redact_attributes(self, redact_attributes: Any = None) -> None:
        self.redact_attribute_patterns = normalize_redact_attribute_patterns(redact_attributes)

    def set_command_tool_names(self, command_tool_names: Any = None) -> None:
        self.command_tool_names = _normalize_command_tool_names(command_tool_names)

    def set_accept_formats(self, accept_formats: Iterable[str] | None = None) -> None:
        self.accept_formats = _normalize_accept_formats(accept_formats)

    def register_trace_policy(self, policy: dict[str, Any] | None = None) -> None:
        if not policy or not policy.get("evaluationId"):
            return
        self.trace_policies_by_evaluation_id[policy["evaluationId"]] = _normalize_trace_policy(
            policy
        )

    def deregister_trace_policy(self, evaluation_id: str | None = None) -> None:
        if evaluation_id:
            self.trace_policies_by_evaluation_id.pop(evaluation_id, None)

    @staticmethod
    def _should_redact_attribute(key: str, patterns: list[str]) -> bool:
        if not patterns:
            return False
        lowered = key.lower()
        return any(pattern in lowered for pattern in patterns)

    def _redact_attribute_value(self, value: Any, patterns: list[str]) -> Any:
        if isinstance(value, list):
            return [self._redact_attribute_value(item, patterns) for item in value]
        if not isinstance(value, dict):
            return value
        return {
            key: "[REDACTED]"
            if self._should_redact_attribute(key, patterns)
            else self._redact_attribute_value(nested, patterns)
            for key, nested in value.items()
        }

    def redact_attributes(
        self, attributes: dict[str, Any] | None, patterns: list[str] | None = None
    ) -> dict[str, Any]:
        if patterns is None:
            patterns = self.redact_attribute_patterns
        if not attributes or not patterns:
            return attributes if attributes is not None else {}
        return {
            key: "[REDACTED]"
            if self._should_redact_attribute(key, patterns)
            else self._redact_attribute_value(value, patterns)
            for key, value in attributes.items()
        }

    def _redact_span(self, span: dict[str, Any], patterns: list[str]) -> dict[str, Any]:
        if not patterns:
            return span
        attributes = span.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}
        # A span name or status message that echoes a redacted attribute value is scrubbed too.
        redacted_source_values = {
            value
            for key, value in attributes.items()
            if isinstance(value, str) and self._should_redact_attribute(key, patterns)
        }

        def scrub_echo(value: Any) -> Any:
            if isinstance(value, str) and value in redacted_source_values:
                return "[REDACTED]"
            return value

        return {
            **span,
            "name": scrub_echo(span.get("name", UNDEFINED)),
            "statusMessage": scrub_echo(span.get("statusMessage", UNDEFINED)),
            "attributes": self.redact_attributes(attributes, patterns),
        }

    # --- request pipeline ----------------------------------------------------------------

    async def handle(self, request: Request) -> Response:
        """The express app: middleware, parsers, routes, then 404 or the global error handler."""
        response = Response()
        try:
            handled = await self._dispatch(request, response)
            if not handled:
                response.status = 404
                response.body = _not_found_page(request.method, request.path)
                response.headers["Content-Security-Policy"] = "default-src 'none'"
                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["Content-Type"] = "text/html; charset=utf-8"
        except HttpError as error:
            logger.error("[OtlpReceiver] Global error handler: %s", error)
            if error.syntax:
                response.json(400, {"error": "Invalid JSON"})
            else:
                response.json(500, {"error": "Internal server error"})
        except Exception as error:  # express's global error handler
            logger.error("[OtlpReceiver] Global error handler: %s", error)
            response.json(500, {"error": "Internal server error"})
        response.headers["X-Powered-By"] = "Express"
        return response

    async def _dispatch(self, request: Request, response: Response) -> bool:
        method, path = request.method, request.path
        content_type = request.headers.get("content-type")
        request_format = get_request_format(content_type)
        if _mount_matches(path, "/v1/traces"):
            # Reject disabled content types before any body parser runs.
            if method == "POST" and (
                not request_format or request_format not in self.accept_formats
            ):
                response.json(415, {"error": "Unsupported content type"})
                return True
            if "json" in self.accept_formats and request_format == "json":
                _json_parser(request)
            elif "protobuf" in self.accept_formats and request_format == "protobuf":
                _raw_parser(request)
        if _mount_matches(path, "/v1/logs"):
            if method == "POST" and request_format != "json":
                response.json(415, {"error": "Only application/json is supported for /v1/logs"})
                return True
            if request_format == "json" and request.body is UNDEFINED:
                _json_parser(request)
        logger.debug("[OtlpReceiver] Middleware configured for accepted OTLP formats")

        is_get = method in ("GET", "HEAD")
        if method == "POST" and _route_matches(path, "/v1/traces"):
            await self._post_traces(request, response)
            return True
        if method == "POST" and _route_matches(path, "/v1/logs"):
            await self._post_logs(request, response)
            return True
        if is_get and _route_matches(path, "/health"):
            logger.debug("[OtlpReceiver] Health check requested")
            response.json(200, {"status": "ok"})
            return True
        if is_get and _route_matches(path, "/v1/traces"):
            response.json(200, {
                "service": "promptfoo-otlp-receiver",
                "version": "1.0.0",
                "supported_formats": self.accept_formats,
            })  # fmt: skip
            return True
        if is_get and _route_matches(path, "/debug/status"):
            response.json(200, {
                "status": "running",
                "uptime": time.monotonic() - _PROCESS_START,
                "timestamp": _iso_now(),
                "port": self.port or 4318,
            })  # fmt: skip
            return True
        return False

    async def _post_traces(self, request: Request, response: Response) -> None:
        content_type = request.headers.get("content-type") or "unknown"
        body = request.body
        body_size = 0
        if is_truthy(body) and body is not UNDEFINED:
            body_size = len(body) if isinstance(body, bytes) else js_length(json_stringify(body))
        logger.debug(
            "[OtlpReceiver] Received trace request: %s with %d bytes",
            request.headers.get("content-type", "undefined"),
            body_size,
        )
        logger.debug("[OtlpReceiver] Starting to process traces")
        request_format = get_request_format(content_type)
        if not request_format or request_format not in self.accept_formats:
            response.json(415, {"error": "Unsupported content type"})
            return
        try:
            traces = await self._parse_incoming_request(request_format, body)
            logger.debug("[OtlpReceiver] Parsed %d traces from request", len(traces))
            await self._persist_traces(self._group_traces(traces))
            response.json(200, {"partialSuccess": {}})
            logger.debug("[OtlpReceiver] Successfully processed traces")
        except Exception as error:
            self._handle_processing_error(error, response)

    async def _post_logs(self, request: Request, response: Response) -> None:
        logger.debug("[OtlpReceiver] Received logs request")
        try:
            traces = self._parse_otlp_logs_json_request(request.body)
            logger.debug("[OtlpReceiver] Parsed %d logs into span records", len(traces))
            if traces:
                await self._persist_traces(self._group_traces(traces))
            response.json(200, {"partialSuccess": {}})
        except Exception as error:
            self._handle_processing_error(error, response)

    async def _parse_incoming_request(self, request_format: str, body: Any) -> list[dict]:
        if request_format == "json":
            logger.debug("[OtlpReceiver] Parsing OTLP JSON request")
            return self._parse_otlp_json_request(body)
        logger.debug("[OtlpReceiver] Parsing OTLP protobuf request")
        logger.debug(
            "[OtlpReceiver] Request body size: %d bytes",
            len(body) if isinstance(body, bytes) else 0,
        )
        return self._parse_otlp_protobuf_request(body)

    def _group_traces(self, traces: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for trace in traces:
            group = grouped.setdefault(trace["traceId"], {"spans": [], "info": {}})
            group["spans"].append(trace["span"])
            attributes = trace["span"].get("attributes")
            evaluation_id = _opt(attributes, "evaluation.id")
            test_case_id = _opt(attributes, "test.case.id")
            if is_truthy(evaluation_id):
                group["info"]["evaluationId"] = evaluation_id
            if is_truthy(test_case_id):
                group["info"]["testCaseId"] = test_case_id
        logger.debug("[OtlpReceiver] Grouped spans into %d traces", len(grouped))
        return grouped

    async def _persist_traces(self, traces: dict[str, dict[str, Any]]) -> None:
        await self._create_trace_records(traces)
        await self._store_spans(traces)

    async def _create_trace_records(self, traces: dict[str, dict[str, Any]]) -> None:
        for trace_id, group in traces.items():
            info = group["info"]
            if not info.get("evaluationId") or not info.get("testCaseId"):
                logger.debug(
                    "[OtlpReceiver] Skipping trace record creation for unlinked trace %s",
                    trace_id,
                )
                continue
            try:
                logger.debug("[OtlpReceiver] Creating trace record for %s", trace_id)
                await self.trace_store.create_trace(
                    {
                        "traceId": trace_id,
                        "evaluationId": info["evaluationId"],
                        "testCaseId": info["testCaseId"],
                        "metadata": self._get_trace_metadata(info),
                    }
                )
            except Exception as error:
                # Trace might already exist, which is fine
                logger.debug("[OtlpReceiver] Trace %s may already exist: %s", trace_id, error)

    def _get_registered_trace_policy(self, info: dict[str, Any] | None) -> dict | None:
        evaluation_id = (info or {}).get("evaluationId")
        if not evaluation_id or not isinstance(evaluation_id, str):
            return None
        return self.trace_policies_by_evaluation_id.get(evaluation_id)

    def _get_fallback_trace_policy(self) -> dict[str, Any]:
        return {
            "commandToolNames": self.command_tool_names,
            "redactAttributePatterns": self.redact_attribute_patterns,
        }

    def _get_trace_metadata(self, info: dict[str, Any] | None) -> dict[str, Any] | None:
        return _metadata_from_trace_policy(
            self._get_registered_trace_policy(info) or self._get_fallback_trace_policy()
        )

    async def _store_spans(self, traces: dict[str, dict[str, Any]]) -> None:
        for trace_id, group in traces.items():
            spans = group["spans"]
            logger.debug("[OtlpReceiver] Storing %d spans for trace %s", len(spans), trace_id)
            patterns = await self._get_redact_attribute_patterns(trace_id, group["info"])
            sanitized = [self._redact_span(span, patterns) for span in spans] if patterns else spans
            await self.trace_store.add_spans(
                trace_id, sanitized, {"skipTraceCheck": False, "warnIfMissingTrace": False}
            )

    async def _get_redact_attribute_patterns(
        self, trace_id: str, info: dict[str, Any] | None
    ) -> list[str]:
        metadata = await self.trace_store.get_trace_metadata(trace_id)
        if metadata is None:
            registered = self._get_registered_trace_policy(info)
            if registered is not None:
                return registered["redactAttributePatterns"]
            return self._get_fallback_trace_policy()["redactAttributePatterns"]
        patterns = metadata.get("otlpHttpRedactAttributes") if isinstance(metadata, dict) else None
        return normalize_redact_attribute_patterns(patterns) if isinstance(patterns, list) else []

    def _handle_processing_error(self, error: BaseException, response: Response) -> None:
        logger.error("[OtlpReceiver] Failed to process OTLP traces: %s", error)
        message = _error_message(error)
        if "invalid protobuf" in message.lower():
            response.json(400, {"error": message})
            return
        response.json(500, {"error": "Internal server error"})

    # --- JSON ----------------------------------------------------------------------------

    def _parse_otlp_json_request(self, body: Any) -> list[dict[str, Any]]:
        traces: list[dict[str, Any]] = []
        resource_spans = _get(body, "resourceSpans")
        for resource_span in _iter(resource_spans):
            resource_attributes = self._parse_attributes(
                _opt(_get(resource_span, "resource"), "attributes")
            )
            logger.debug("[OtlpReceiver] Parsed %d resource attributes", len(resource_attributes))
            for scope_span in _iter(_get(resource_span, "scopeSpans")):
                scope = _get(scope_span, "scope")
                for span in _iter(_get(scope_span, "spans")):
                    trace_id = self._convert_id(_get(span, "traceId"), 32)
                    span_id = self._convert_id(_get(span, "spanId"), 16)
                    raw_parent = _get(span, "parentSpanId")
                    parent_span_id = (
                        self._convert_id(raw_parent, 16) if is_truthy(raw_parent) else UNDEFINED
                    )
                    kind = _get(span, "kind")
                    attributes = {
                        **resource_attributes,
                        **self._parse_attributes(_get(span, "attributes")),
                        "otel.scope.name": _opt(scope, "name"),
                        "otel.scope.version": _opt(scope, "version"),
                        "otel.span.kind": SPAN_KIND_MAP.get(_property_key(kind), "unspecified"),
                        "otel.span.kind_code": kind,
                    }
                    end = _get(span, "endTimeUnixNano")
                    status = _get(span, "status")
                    traces.append({
                        "traceId": trace_id,
                        "span": {
                            "spanId": span_id,
                            "parentSpanId": parent_span_id,
                            "name": _get(span, "name"),
                            "startTime": _js_div(
                                js_number(_get(span, "startTimeUnixNano")), 1_000_000
                            ),
                            "endTime": _js_div(js_number(end), 1_000_000)
                            if is_truthy(end) else UNDEFINED,
                            "attributes": attributes,
                            "statusCode": _opt(status, "code"),
                            "statusMessage": _opt(status, "message"),
                        },
                    })  # fmt: skip
        return traces

    def _parse_attributes(self, attributes: Any) -> dict[str, Any]:
        if not is_truthy(attributes):
            return {}
        result: dict[str, Any] = {}
        for attr in _iter(attributes):
            value = self._parse_attribute_value(_get(attr, "value"))
            if value is not UNDEFINED:
                result[_property_key(_get(attr, "key"))] = value
        return result

    def _parse_attribute_value(self, value: Any) -> Any:
        string_value = _get(value, "stringValue")
        if string_value is not UNDEFINED:
            return string_value
        int_value = _get(value, "intValue")
        if int_value is not UNDEFINED:
            return js_number(int_value)
        double_value = _get(value, "doubleValue")
        if double_value is not UNDEFINED:
            return double_value
        bool_value = _get(value, "boolValue")
        if bool_value is not UNDEFINED:
            return bool_value
        array_values = _opt(_get(value, "arrayValue"), "values")
        if is_truthy(array_values):
            if not isinstance(array_values, list):
                raise TypeError("value.arrayValue.values.map is not a function")
            return [self._parse_attribute_value(v) for v in array_values]
        kv_values = _opt(_get(value, "kvlistValue"), "values")
        if is_truthy(kv_values):
            kv_map: dict[str, Any] = {}
            for kv in _iter(kv_values):
                kv_map[_property_key(_get(kv, "key"))] = self._parse_attribute_value(
                    _get(kv, "value")
                )
            return kv_map
        return UNDEFINED

    def _convert_id(self, value: Any, expected_hex_length: int) -> str:
        length = _get(value, "length")
        logger.debug(
            "[OtlpReceiver] Converting ID: %s (length: %s, expected hex length: %d)",
            js_string(value),
            js_string(length),
            expected_hex_length,
        )
        if isinstance(value, str) and length == expected_hex_length and _HEX_I.match(value):
            logger.debug("[OtlpReceiver] ID is already hex format")
            return value.lower()
        try:
            if not isinstance(value, str):
                raise TypeError(
                    "The first argument must be of type string or an instance of Buffer, "
                    "ArrayBuffer, or Array or an Array-like Object."
                )
            buffer = node_base64_decode(value)
            hex_value = buffer.hex()
            logger.debug(
                "[OtlpReceiver] Base64 decoded: %s -> %s (%d bytes)", value, hex_value, len(buffer)
            )
            utf8_string = buffer.decode("utf-8", errors="replace")
            if js_length(utf8_string) == expected_hex_length and _HEX_I.match(utf8_string):
                logger.debug("[OtlpReceiver] Detected hex string encoded as UTF-8: %s", utf8_string)
                return utf8_string.lower()
            if len(hex_value) == expected_hex_length:
                return hex_value
            logger.warning(
                "[OtlpReceiver] Unexpected ID format: %s -> %s (expected %d hex chars)",
                value,
                hex_value,
                expected_hex_length,
            )
            return value.lower()
        except TypeError as error:
            logger.error("[OtlpReceiver] Failed to convert ID: TypeError: %s", error)
            raise TypeError("id.toLowerCase is not a function") from error

    # --- logs ----------------------------------------------------------------------------

    def _parse_otlp_logs_json_request(self, body: Any) -> list[dict[str, Any]]:
        traces: list[dict[str, Any]] = []
        resource_logs = _opt(body, "resourceLogs")
        if _nullish(resource_logs):
            resource_logs = []
        logger.debug(
            "[OtlpReceiver] Parsing logs request with %s resource logs",
            js_string(_get(resource_logs, "length")),
        )
        for resource_log in _iter(resource_logs):
            resource_attributes = self._parse_attributes(
                _opt(_get(resource_log, "resource"), "attributes")
            )
            scope_logs = _get(resource_log, "scopeLogs")
            for scope_log in _iter([] if _nullish(scope_logs) else scope_logs):
                records = _get(scope_log, "logRecords")
                for log in _iter([] if _nullish(records) else records):
                    # One malformed record must not drop the whole batch.
                    try:
                        parsed = self._log_record_to_parsed_trace(
                            log, scope_log, resource_attributes
                        )
                        if parsed:
                            traces.append(parsed)
                    except Exception as error:
                        scope_name = _opt(_get(scope_log, "scope"), "name")
                        logger.warning(
                            "[OtlpReceiver] Skipping malformed log record in scope %s: %s",
                            "(unknown)" if _nullish(scope_name) else js_string(scope_name),
                            error,
                        )
        return traces

    def _log_record_to_parsed_trace(
        self, log: Any, scope_log: Any, resource_attributes: dict[str, Any]
    ) -> dict[str, Any] | None:
        inline_trace_id = _get(log, "traceId")
        has_valid_inline_trace_id = is_truthy(inline_trace_id) and not _is_zero_trace_id(
            inline_trace_id
        )
        raw_trace_id = (
            inline_trace_id
            if has_valid_inline_trace_id
            else _get_string_attribute(resource_attributes, PROMPTFOO_RESOURCE_ATTR_TRACE_ID)
        )
        if not is_truthy(raw_trace_id):
            logger.debug(
                "[OtlpReceiver] Dropping log: no traceId and no %s resource attribute",
                PROMPTFOO_RESOURCE_ATTR_TRACE_ID,
            )
            return None
        trace_id = self._convert_id(raw_trace_id, 32)
        if len(trace_id) != 32 or not _HEX.match(trace_id):
            logger.debug("[OtlpReceiver] Dropping log: invalid trace_id shape '%s'", raw_trace_id)
            return None

        scope = _get(scope_log, "scope")
        severity_number = _get(log, "severityNumber")
        severity_text = _get(log, "severityText")
        attributes: dict[str, Any] = {
            **resource_attributes,
            **self._parse_attributes(_get(log, "attributes")),
            "otel.scope.name": _opt(scope, "name"),
            "otel.scope.version": _opt(scope, "version"),
            "otel.log.severity_number": severity_number,
            "otel.log.severity_text": severity_text,
        }
        body = _get(log, "body")
        body_value = self._parse_attribute_value(body) if is_truthy(body) else UNDEFINED
        name = _resolve_log_span_name(attributes, body_value)
        if name in LOG_EVENT_NAME_DENYLIST:
            logger.debug("[OtlpReceiver] Dropping log: event '%s' is in the denylist", name)
            return None
        if body_value is not UNDEFINED:
            attributes["otel.log.body"] = _truncate_log_body(body_value)

        time_nano = _get(log, "timeUnixNano")
        if _nullish(time_nano):
            time_nano = _get(log, "observedTimeUnixNano")
        start_time = (
            _js_div(js_number(time_nano), 1_000_000)
            if is_truthy(time_nano)
            else float(int(time.time() * 1000))
        )
        end_time = start_time + LOG_SPAN_DURATION_MS

        inline_span_id = _get(log, "spanId")
        has_valid_inline_span_id = is_truthy(inline_span_id) and not _is_zero_span_id(
            inline_span_id
        )
        raw_parent_span_id = (
            inline_span_id
            if has_valid_inline_span_id
            else _get_string_attribute(resource_attributes, PROMPTFOO_RESOURCE_ATTR_PARENT_SPAN_ID)
        )
        parent_span_id = (
            self._convert_id(raw_parent_span_id, 16) if is_truthy(raw_parent_span_id) else UNDEFINED
        )
        severity_is_error = (
            isinstance(severity_number, int | float)
            and not isinstance(severity_number, bool)
            and severity_number >= SEVERITY_NUMBER_ERROR
        )
        return {
            "traceId": trace_id,
            "span": {
                "spanId": _random_span_id(),
                "parentSpanId": parent_span_id,
                "name": name,
                "startTime": start_time,
                "endTime": end_time,
                "attributes": attributes,
                # OTEL logs don't carry a span status; OK unless severity indicates error.
                "statusCode": 1,
                "statusMessage": severity_text if severity_is_error else UNDEFINED,
            },
        }

    # --- protobuf ------------------------------------------------------------------------

    def _parse_otlp_protobuf_request(self, body: Any) -> list[dict[str, Any]]:
        if not isinstance(body, bytes):
            raise ValueError("Invalid protobuf data: illegal buffer")
        decoded = decode_export_trace_service_request(body)
        logger.debug(
            "[OtlpReceiver] Parsing protobuf request with %d resource spans",
            len(decoded["resourceSpans"]),
        )
        traces: list[dict[str, Any]] = []
        for resource_span in decoded["resourceSpans"]:
            resource = resource_span["resource"]
            resource_attributes = self._parse_decoded_attributes(
                resource["attributes"] if resource else None
            )
            for scope_span in resource_span["scopeSpans"]:
                for span in scope_span["spans"]:
                    traces.append(
                        self._create_decoded_parsed_trace(resource_attributes, scope_span, span)
                    )
        return traces

    def _create_decoded_parsed_trace(
        self, resource_attributes: dict[str, Any], scope_span: dict[str, Any], span: dict
    ) -> dict[str, Any]:
        trace_id = bytes_to_hex(span["traceId"], 32)
        span_id = bytes_to_hex(span["spanId"], 16)
        parent_span_id = bytes_to_hex(span["parentSpanId"], 16) if span["parentSpanId"] else None
        logger.debug(
            "[OtlpReceiver] Processing protobuf span: %s (%s) in trace %s",
            span["name"],
            span_id,
            trace_id,
        )
        kind = span["kind"]
        scope = scope_span["scope"]
        status = span["status"]
        out: dict[str, Any] = {
            "spanId": span_id,
            "parentSpanId": parent_span_id if parent_span_id is not None else UNDEFINED,
            "name": span["name"],
            "startTime": span["startTimeUnixNano"] / 1_000_000,
            "endTime": span["endTimeUnixNano"] / 1_000_000,
            "attributes": {
                **resource_attributes,
                **self._parse_decoded_attributes(span["attributes"]),
                "otel.scope.name": scope["name"] if scope else UNDEFINED,
                "otel.scope.version": scope["version"] if scope else UNDEFINED,
                "otel.span.kind": SPAN_KIND_MAP.get(str(kind), "unspecified"),
                "otel.span.kind_code": kind,
            },
            "statusCode": status["code"] if status else UNDEFINED,
            "statusMessage": status["message"] if status else UNDEFINED,
        }
        return {"traceId": trace_id, "span": out}

    def _parse_decoded_attributes(self, attributes: list[dict[str, Any]] | None) -> dict:
        if not attributes:
            return {}
        result: dict[str, Any] = {}
        for attr in attributes:
            value = self._parse_decoded_attribute_value(attr["value"])
            if value is not UNDEFINED:
                result[attr["key"]] = value
        return result

    def _parse_decoded_attribute_value(self, value: dict[str, Any] | None) -> Any:
        if not value:
            return UNDEFINED
        if "stringValue" in value:
            return value["stringValue"]
        if "intValue" in value:
            return value["intValue"]
        if "doubleValue" in value:
            return value["doubleValue"]
        if "boolValue" in value:
            return value["boolValue"]
        if "bytesValue" in value:
            return bytes_to_base64(value["bytesValue"])
        if "arrayValue" in value:
            return [self._parse_decoded_attribute_value(v) for v in value["arrayValue"]["values"]]
        if "kvlistValue" in value:
            return {
                kv["key"]: self._parse_decoded_attribute_value(kv["value"])
                for kv in value["kvlistValue"]["values"]
            }
        return UNDEFINED

    # --- lifecycle -----------------------------------------------------------------------

    @property
    def bound_port(self) -> int | None:
        """The port actually bound (differs from `port` when listening on port 0)."""
        if self.server is None or not self.server.sockets:
            return None
        return self.server.sockets[0].getsockname()[1]

    async def listen(self, port: int = 4318, host: str = "127.0.0.1") -> None:
        self.port = port
        logger.debug("[OtlpReceiver] Starting receiver on %s:%s", host, port)
        try:
            self.server = await asyncio.start_server(self._on_connection, host, port)
        except OSError as error:
            message = _listen_error_message(error, host, port)
            logger.error("[OtlpReceiver] Failed to start: Error: %s", message)
            raise OSError(error.errno, message) from error
        logger.info("[OtlpReceiver] Listening on http://%s:%s", host, port)
        logger.debug("[OtlpReceiver] Receiver fully initialized and ready to accept traces")

    async def stop(self) -> None:
        logger.debug("[OtlpReceiver] Stopping receiver")
        server = self.server
        if server is None:
            logger.debug("[OtlpReceiver] No server to stop")
            return
        server.close()
        # http.Server#close: stop accepting, drop idle keep-alive sockets, let requests finish.
        for connection in list(self._connections):
            connection.close_when_idle()
        await server.wait_closed()
        logger.info("[OtlpReceiver] Server stopped")
        self.server = None

    async def _on_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        connection = _Connection(self, reader, writer)
        self._connections.add(connection)
        try:
            await connection.serve()
        finally:
            self._connections.discard(connection)


def _iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


_LISTEN_ERRORS = {
    "EADDRINUSE": "address already in use",
    "EACCES": "permission denied",
    "EADDRNOTAVAIL": "address not available",
}


def _listen_error_message(error: OSError, host: str, port: int) -> str:
    import errno as errno_module

    name = errno_module.errorcode.get(error.errno or 0, "")
    if name in _LISTEN_ERRORS:
        return f"listen {name}: {_LISTEN_ERRORS[name]} {host}:{port}"
    if isinstance(error, OSError) and error.errno is None:  # getaddrinfo failures
        return f"getaddrinfo ENOTFOUND {host}"
    return f"listen {name or 'UNKNOWN'}: {os.strerror(error.errno or 0).lower()} {host}:{port}"


# --- connections -------------------------------------------------------------------------------

_MAX_HEADER_BYTES = 16 * 1024  # Node's --max-http-header-size default


class _BadRequest(Exception):
    pass


class _Connection:
    """One HTTP/1.1 connection: keep-alive, Content-Length or chunked bodies, 100-continue."""

    def __init__(
        self, receiver: OTLPReceiver, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.receiver = receiver
        self.reader = reader
        self.writer = writer
        self.busy = False
        self.closing = False

    def close_when_idle(self) -> None:
        self.closing = True
        if not self.busy:
            self.writer.close()

    async def serve(self) -> None:
        try:
            while not self.closing:
                try:
                    request = await self._read_request()
                except _BadRequest:
                    self.writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
                    await self.writer.drain()
                    break
                if request is None:
                    break
                self.busy = True
                try:
                    response = await self.receiver.handle(request[0])
                    keep_alive = request[1] and not self.closing
                    await self._write_response(request[0], response, keep_alive)
                finally:
                    self.busy = False
                if not keep_alive:
                    break
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.CancelledError):
            pass
        finally:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except (ConnectionError, OSError, asyncio.CancelledError):
                pass

    async def _read_request(self) -> tuple[Request, bool] | None:
        try:
            head = await self.reader.readuntil(b"\r\n\r\n")
        except asyncio.IncompleteReadError as error:
            if error.partial.strip():
                raise _BadRequest from error
            return None
        except asyncio.LimitOverrunError as error:
            raise _BadRequest from error
        if len(head) > _MAX_HEADER_BYTES:
            raise _BadRequest
        lines = head.decode("latin-1").split("\r\n")
        try:
            method, target, version = lines[0].split(" ")
        except ValueError as error:
            raise _BadRequest from error
        if not version.startswith("HTTP/1."):
            raise _BadRequest
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line:
                continue
            name, sep, value = line.partition(":")
            if not sep or not name or name != name.strip():
                raise _BadRequest
            key = name.lower()
            value = value.strip()
            headers[key] = f"{headers[key]}, {value}" if key in headers else value
        connection = headers.get("connection", "").lower()
        keep_alive = (
            "close" not in connection if version == "HTTP/1.1" else "keep-alive" in connection
        )
        if headers.get("expect", "").lower() == "100-continue":
            self.writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            await self.writer.drain()
        body = await self._read_body(headers)
        return Request(method, target, headers, body), keep_alive

    async def _read_body(self, headers: dict[str, str]) -> bytes | None:
        if "chunked" in headers.get("transfer-encoding", "").lower():
            chunks = bytearray()
            while True:
                size_line = await self.reader.readuntil(b"\r\n")
                try:
                    size = int(size_line.split(b";", 1)[0].strip(), 16)
                except ValueError as error:
                    raise _BadRequest from error
                if size == 0:
                    while (await self.reader.readuntil(b"\r\n")) != b"\r\n":
                        pass  # trailers
                    return bytes(chunks)
                chunk = await self.reader.readexactly(size + 2)
                if len(chunks) <= BODY_LIMIT_BYTES * 2:  # keep enough to see the limit
                    chunks += chunk[:-2]
        length = headers.get("content-length")
        if length is None:
            return None
        try:
            size = int(length)
        except ValueError as error:
            raise _BadRequest from error
        if size < 0:
            raise _BadRequest
        if size > BODY_LIMIT_BYTES:
            # raw-body rejects on the declared length; drain the body without keeping it.
            remaining = size
            while remaining > 0:
                chunk = await self.reader.read(min(remaining, 1 << 16))
                if not chunk:
                    raise asyncio.IncompleteReadError(b"", remaining)
                remaining -= len(chunk)
            return b""
        return await self.reader.readexactly(size)

    async def _write_response(self, request: Request, response: Response, keep_alive: bool):
        reason = _STATUS_TEXT.get(response.status, "")
        headers = dict(response.headers)
        headers.setdefault("Content-Length", str(len(response.body)))
        headers["Date"] = _http_date()
        headers["Connection"] = "keep-alive" if keep_alive else "close"
        if keep_alive:
            headers["Keep-Alive"] = "timeout=5"
        # express's header order: X-Powered-By first.
        ordered = {"X-Powered-By": headers.pop("X-Powered-By", "Express"), **headers}
        head = f"HTTP/1.1 {response.status} {reason}\r\n" + "".join(
            f"{name}: {value}\r\n" for name, value in ordered.items()
        )
        self.writer.write(head.encode("latin-1") + b"\r\n")
        if request.method != "HEAD":
            self.writer.write(response.body)
        await self.writer.drain()


def _http_date() -> str:
    from email.utils import formatdate

    return formatdate(usegmt=True)


# --- singleton ---------------------------------------------------------------------------------

_otlp_receiver: OTLPReceiver | None = None


def get_otlp_receiver_instance() -> OTLPReceiver | None:
    """The running singleton, if any (for tests and the evaluator)."""
    return _otlp_receiver


def _get_otlp_receiver(options: dict[str, Any] | None = None) -> OTLPReceiver:
    global _otlp_receiver
    options = options or {}
    if _otlp_receiver is not None:
        _otlp_receiver.set_accept_formats(options.get("acceptFormats"))
        _otlp_receiver.set_command_tool_names(options.get("commandToolNames"))
        _otlp_receiver.set_redact_attributes(options.get("redactAttributes"))
        _otlp_receiver.register_trace_policy(options.get("tracePolicy"))
        return _otlp_receiver
    _otlp_receiver = OTLPReceiver(options)
    return _otlp_receiver


async def start_otlp_receiver(
    port: int | None = None,
    host: str | None = None,
    accept_formats: Iterable[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """startOTLPReceiver(port, host, acceptFormats, {commandToolNames, redactAttributes,
    tracePolicy})."""
    global _otlp_receiver
    logger.debug("[OtlpReceiver] Starting receiver through startOTLPReceiver function")
    receiver = _get_otlp_receiver({"acceptFormats": accept_formats, **(extra or {})})
    try:
        await receiver.listen(4318 if port is None else port, host or "127.0.0.1")
    except BaseException:
        # A receiver that never bound must not dictate options for a later retry.
        if _otlp_receiver is receiver:
            _otlp_receiver = None
        raise


def register_otlp_receiver_trace_policy(policy: dict[str, Any] | None = None) -> None:
    if _otlp_receiver is not None:
        _otlp_receiver.register_trace_policy(policy)


def deregister_otlp_receiver_trace_policy(evaluation_id: str | None = None) -> None:
    if _otlp_receiver is not None:
        _otlp_receiver.deregister_trace_policy(evaluation_id)


async def stop_otlp_receiver() -> None:
    global _otlp_receiver
    logger.debug("[OtlpReceiver] Stopping receiver through stopOTLPReceiver function")
    if _otlp_receiver is not None:
        await _otlp_receiver.stop()
        _otlp_receiver = None


__all__ = [
    "OTLPReceiver",
    "deregister_otlp_receiver_trace_policy",
    "get_otlp_receiver_instance",
    "get_request_format",
    "js_number",
    "register_otlp_receiver_trace_policy",
    "start_otlp_receiver",
    "stop_otlp_receiver",
]
