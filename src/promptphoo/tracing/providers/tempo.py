"""Ports src/tracing/providers/tempo.ts: Grafana Tempo's `/api/traces/<id>` (OTLP JSON)."""

from __future__ import annotations

import base64
import binascii
import logging
import math
import re
import time
from typing import Any

from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import is_truthy
from promptphoo.tracing.providers.fetch import (
    MAX_TRACE_RESPONSE_BYTES,
    fetch_trace_response,
    read_limited_response,
    release_response,
    validate_trace_provider_endpoint,
)
from promptphoo.tracing.providers.types import TraceProviderError

logger = logging.getLogger(__name__)

MAX_SPANS = 10_000
SPAN_KIND_NAMES = ["unspecified", "internal", "server", "client", "producer", "consumer"]
TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}\Z", re.IGNORECASE)
BASE64_TRACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9+/]{22}(?:==)?\Z")
SPAN_ID_PATTERN = re.compile(r"^[0-9a-f]{16}\Z", re.IGNORECASE)
BASE64_SPAN_ID_PATTERN = re.compile(r"^[A-Za-z0-9+/]{11}=?\Z")
MAX_SAFE_INTEGER = 2**53 - 1
_ALL_ZERO = re.compile(r"^0+\Z")


def is_positive_safe_integer(value: Any) -> bool:
    """`Number.isSafeInteger(value) && value > 0`."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return False
    return 0 < value <= MAX_SAFE_INTEGER


def is_trace_id(trace_id: Any) -> bool:
    return (
        isinstance(trace_id, str)
        and TRACE_ID_PATTERN.search(trace_id) is not None
        and not _ALL_ZERO.search(trace_id)
    )


def now_ms() -> int:
    return int(time.time() * 1000)


def _big_int(value: Any) -> int:
    """BigInt(value) for the string/number nanosecond timestamps Tempo returns."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("The number cannot be converted to a BigInt")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return 0
        return int(text, 0) if re.fullmatch(r"0[xXoObB][0-9a-fA-F]+", text) else int(text)
    raise ValueError(f"Cannot convert {value!r} to a BigInt")


def _nano_to_ms(value: Any) -> int:
    nanos = _big_int(value)
    milliseconds = nanos // 1_000_000 if nanos >= 0 else -(-nanos // 1_000_000)
    if milliseconds < 0 or milliseconds > MAX_SAFE_INTEGER:
        raise ValueError("Span timestamp is outside the supported range")
    return milliseconds


def _extract_attribute_value(value: Any) -> Any:
    from promptphoo.matchers.rubric import js_number

    if not isinstance(value, dict):
        raise TypeError("Attribute value must be an object")
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        number = js_number(value["intValue"])
        if (
            isinstance(number, int | float)
            and math.isfinite(number)
            and float(number).is_integer()
            and abs(number) <= MAX_SAFE_INTEGER
        ):
            return int(number)
        return value["intValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "bytesValue" in value:
        return value["bytesValue"]
    if is_truthy(value.get("arrayValue")):
        return [_extract_attribute_value(v) for v in value["arrayValue"].get("values") or []]
    if is_truthy(value.get("kvlistValue")):
        return _attributes_to_record(value["kvlistValue"].get("values"))
    return None


def _attributes_to_record(attributes: Any) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for entry in attributes or []:
        record[entry["key"]] = _extract_attribute_value(entry["value"])
    return record


def _decode_id(
    id_: Any, hex_pattern: re.Pattern[str], base64_pattern: re.Pattern[str], byte_length: int
) -> str | None:
    if not id_:
        return None
    if not isinstance(id_, str):
        raise TypeError("ID must be a string")
    if hex_pattern.search(id_):
        return None if _ALL_ZERO.search(id_) else id_.lower()
    if not base64_pattern.search(id_):
        return None
    try:
        decoded = base64.b64decode(id_ + "=" * (-len(id_) % 4))
    except binascii.Error:
        return None
    if len(decoded) != byte_length or base64.b64encode(decoded).decode("ascii").rstrip(
        "="
    ) != id_.rstrip("="):
        return None
    hex_id = decoded.hex()
    return None if _ALL_ZERO.search(hex_id) else hex_id


def _normalize_status_code(code: Any) -> int | float | None:
    from promptphoo.matchers.rubric import js_number

    if isinstance(code, int | float) and not isinstance(code, bool):
        return code
    if not code:
        return None
    numeric = js_number(code)
    if not (isinstance(numeric, float) and math.isnan(numeric)):
        return numeric
    if not isinstance(code, str):
        return None
    upper = code.upper()
    if upper in ("STATUS_CODE_OK", "OK"):
        return 1
    if upper in ("STATUS_CODE_ERROR", "ERROR"):
        return 2
    if upper in ("STATUS_CODE_UNSET", "UNSET"):
        return 0
    return None


def _transform_span(
    span: Any, trace_id: str, resource_attributes: dict[str, Any], scope_name: Any
) -> dict[str, Any] | None:
    if not isinstance(span, dict):
        raise TypeError("Span must be an object")
    if (
        _decode_id(span.get("traceId"), TRACE_ID_PATTERN, BASE64_TRACE_ID_PATTERN, 16)
        != trace_id.lower()
    ):
        raise ValueError("Span trace ID must match the requested trace")

    span_id = _decode_id(span.get("spanId"), SPAN_ID_PATTERN, BASE64_SPAN_ID_PATTERN, 8)
    if not span_id:
        raise ValueError("Span ID must be a valid nonzero eight-byte identifier")

    parent_span_id = _decode_id(
        span.get("parentSpanId"), SPAN_ID_PATTERN, BASE64_SPAN_ID_PATTERN, 8
    )
    if span.get("parentSpanId") and not parent_span_id:
        raise ValueError("Parent span ID must be a valid nonzero eight-byte identifier")

    name = span.get("name")
    if not isinstance(name, str) or len(name.strip()) == 0:
        raise ValueError("Span name must be a nonempty string")
    status = span.get("status")
    status = status if isinstance(status, dict) else {}
    if status.get("message") is not None and not isinstance(status["message"], str):
        raise ValueError("Span status message must be a string")

    start_time = _nano_to_ms(span.get("startTimeUnixNano"))
    end_time_nano = span.get("endTimeUnixNano")
    end_time = _nano_to_ms(end_time_nano) if end_time_nano else None
    if end_time_nano and _big_int(end_time_nano) < _big_int(span.get("startTimeUnixNano")):
        raise ValueError("Span end time must not precede its start time")

    attributes: dict[str, Any] = {
        **resource_attributes,
        **_attributes_to_record(span.get("attributes")),
    }
    if scope_name:
        attributes["otel.scope.name"] = scope_name
    kind = span.get("kind")
    if isinstance(kind, int | float) and not isinstance(kind, bool):
        index = int(kind) if float(kind).is_integer() else -1
        attributes["otel.span.kind"] = (
            SPAN_KIND_NAMES[index] if 0 <= index < len(SPAN_KIND_NAMES) else "unspecified"
        )
        attributes["otel.span.kind_code"] = kind
    if isinstance(kind, str):
        attributes["otel.span.kind"] = re.sub(r"^SPAN_KIND_", "", kind, flags=re.I).lower()

    out: dict[str, Any] = {"spanId": span_id}
    if parent_span_id is not None:
        out["parentSpanId"] = parent_span_id
    out["name"] = name
    out["startTime"] = start_time
    if end_time is not None:
        out["endTime"] = end_time
    out["attributes"] = attributes
    status_code = _normalize_status_code(status.get("code"))
    if status_code is not None:
        out["statusCode"] = status_code
    if status.get("message") is not None:
        out["statusMessage"] = status["message"]
    return out


class TempoProvider:
    id = "tempo"

    def __init__(self, config: dict[str, Any]) -> None:
        if not config.get("endpoint"):
            raise ValueError("Tempo provider requires endpoint configuration")
        validate_trace_provider_endpoint(config["endpoint"], "Tempo")
        if config.get("timeout") is not None and not is_positive_safe_integer(config["timeout"]):
            raise ValueError("Tempo provider timeout must be a positive integer")
        self.config = config
        self.base_url = re.sub(r"/\Z", "", config["endpoint"])

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = dict(self.config.get("headers") or {})
        auth = self.config.get("auth") or {}
        has_configured_authentication = bool(
            auth.get("token") or (auth.get("username") and auth.get("password"))
        )
        for header in list(headers):
            normalized = header.lower()
            if normalized == "accept" or (
                has_configured_authentication and normalized == "authorization"
            ):
                del headers[header]
        headers["Accept"] = "application/json"
        if auth.get("token"):
            headers["Authorization"] = f"Bearer {auth['token']}"
        elif auth.get("username") and auth.get("password"):
            credentials = base64.b64encode(
                f"{auth['username']}:{auth['password']}".encode()
            ).decode("ascii")
            headers["Authorization"] = f"Basic {credentials}"
        return headers

    def _transform_spans(self, data: dict[str, Any], trace_id: str) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        seen_span_ids: set[str] = set()
        malformed_spans = 0

        for batch in data.get("batches") or []:
            if not isinstance(batch, dict) or not isinstance(batch.get("scopeSpans"), list):
                malformed_spans += 1
                continue
            resource = batch.get("resource")
            resource_attributes = _attributes_to_record(
                resource.get("attributes") if isinstance(resource, dict) else None
            )
            for scope_span in batch["scopeSpans"]:
                if not isinstance(scope_span, dict) or not isinstance(
                    scope_span.get("spans"), list
                ):
                    malformed_spans += 1
                    continue
                scope = scope_span.get("scope")
                scope_name = scope.get("name") if isinstance(scope, dict) else None
                for span in scope_span["spans"]:
                    if len(spans) >= MAX_SPANS:
                        return spans
                    try:
                        normalized = _transform_span(
                            span, trace_id, resource_attributes, scope_name
                        )
                        if normalized and normalized["spanId"] not in seen_span_ids:
                            seen_span_ids.add(normalized["spanId"])
                            spans.append(normalized)
                    except Exception:
                        malformed_spans += 1

        if malformed_spans > 0:
            logger.warning("[TempoProvider] Skipped %d malformed spans", malformed_spans)
        return spans

    async def fetch_trace(
        self, trace_id: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        if not is_trace_id(trace_id):
            raise TraceProviderError("Trace ID must contain 32 hexadecimal characters")

        options = options or {}
        response = await fetch_trace_response(
            f"{self.base_url}/api/traces/{trace_id}",
            {"disableTransientRetries": True, "method": "GET", "headers": self._build_headers()},
            timeout_ms=self.config.get("timeout") or 10_000,
            abort_signal=options.get("abortSignal"),
        )

        if response.status == 404:
            await release_response(response, "Tempo")
            return None
        if not response.ok:
            await release_response(response, "Tempo")
            raise TraceProviderError(
                f"Tempo returned HTTP {response.status}",
                status_code=response.status,
                retryable=response.status in (408, 429) or response.status >= 500,
            )

        if content_length_exceeds(response.header("content-length"), MAX_TRACE_RESPONSE_BYTES):
            await release_response(response, "Tempo")
            raise TraceProviderError("Tempo trace exceeds the maximum response size")
        body = await read_limited_response(response, "Tempo")
        data = js_json_parse(body)
        if not isinstance(data, dict) or not isinstance(data.get("batches"), list):
            raise TraceProviderError("Tempo returned an invalid trace response")

        return {"spans": self._transform_spans(data, trace_id), "fetchedAt": now_ms()}


def content_length_exceeds(header: str | None, limit: int) -> bool:
    """`Number(response.headers.get('content-length')) > limit` (null and NaN are not)."""
    from promptphoo.matchers.rubric import js_number

    if header is None:
        return False
    number = js_number(header)
    return not (isinstance(number, float) and math.isnan(number)) and number > limit
