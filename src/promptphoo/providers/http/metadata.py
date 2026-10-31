"""Redaction of request bodies and raw requests before they reach logs or response metadata.
Ports the sanitize* / formatRawRequestForDebugMetadata helpers of src/providers/http.ts.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from promptphoo.jsvalues import js_string
from promptphoo.providers.http.raw import extract_body_from_raw_request, parse_raw_request
from promptphoo.sanitizer import (
    REDACTED,
    is_secret_field,
    looks_like_secret,
    safe_json_stringify,
    sanitize_object,
    sanitize_url,
    sanitize_url_encoded_string,
)

logger = logging.getLogger(__name__)


def sanitize_multipart_fields(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **f,
            "value": REDACTED
            if is_secret_field(f["field"]) or looks_like_secret(f["value"])
            else f["value"],
        }
        for f in fields
    ]


def sanitize_multipart_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "field": REDACTED if is_secret_field(f["field"]) else f["field"],
            "filename": REDACTED if looks_like_secret(f["filename"]) else f["filename"],
            "contentType": f["contentType"],
            "sizeBytes": f["sizeBytes"],
            "source": f["source"],
        }
        for f in files
    ]


def get_header_value(headers: dict[str, str] | None, header_name: str) -> str | None:
    if not headers:
        return None
    name = header_name.lower()
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def _get_multipart_boundary(content_type: str) -> str | None:
    m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type, re.I)
    if not m:
        return None
    return m.group(1) if m.group(1) is not None else m.group(2).strip()


def _sanitize_multipart_fallback_body(body: str) -> str:
    trimmed = body.strip()
    if not trimmed:
        return body
    sanitized = sanitize_url_encoded_string(trimmed)
    return body if sanitized == trimmed else body.replace(trimmed, sanitized, 1)


def _get_multipart_part_field_name(header_block: str) -> str | None:
    m = re.search(
        r'content-disposition:[^\r\n]*\bname=(?:"([^"]*)"|([^;\r\n]+))', header_block, re.I
    )
    if not m:
        return None
    name = m.group(1) if m.group(1) is not None else (m.group(2) or "").strip()
    return name or None


def _sanitize_multipart_part_body(part: str) -> str:
    separator = "\r\n\r\n" if "\r\n\r\n" in part else "\n\n"
    idx = part.find(separator)
    if idx == -1:
        return _sanitize_multipart_fallback_body(part)
    header_block = part[:idx]
    body_start = idx + len(separator)
    body = part[body_start:]
    field_name = _get_multipart_part_field_name(header_block)
    trailing = re.search(r"(\r?\n)\Z", body)
    trailing_newline = trailing.group(1) if trailing else ""
    body_without_terminator = body[: -len(trailing_newline)] if trailing_newline else body
    should_redact = bool(field_name and is_secret_field(field_name)) or looks_like_secret(
        body_without_terminator
    )
    if not should_redact:
        return part
    return f"{part[:body_start]}{REDACTED}{trailing_newline}"


def _sanitize_multipart_body(body: str, content_type: str) -> str:
    boundary = _get_multipart_boundary(content_type)
    if not boundary:
        return _sanitize_multipart_fallback_body(body)
    sep = f"--{boundary}"
    return sep.join(_sanitize_multipart_part_body(p) for p in body.split(sep))


def sanitize_request_body_for_metadata(body: Any, headers: dict[str, str] | None = None) -> Any:
    sanitized = sanitize_object(body, context="request body")
    if not isinstance(sanitized, str) or len(sanitized) == 0:
        return sanitized
    content_type = get_header_value(headers, "content-type")
    normalized = content_type.lower() if content_type else None
    if content_type and normalized and "multipart/form-data" in normalized:
        return _sanitize_multipart_body(sanitized, content_type)
    if normalized and "application/x-www-form-urlencoded" in normalized:
        return sanitize_url_encoded_string(sanitized)
    return sanitized


def _looks_like_raw_http_request(value: str) -> bool:
    first_line = re.split(r"\r?\n", value, maxsplit=1)[0].strip()
    return re.fullmatch(r"[A-Z]+\s+\S+\s+HTTP/\d(?:\.\d)?", first_line, re.I) is not None


def _decode_escaped_string_literal(value: str) -> str | None:
    if not re.search(r'[\\](?:r|n|t|"|\\)', value):
        return None
    try:
        decoded = json.loads(f'"{value}"')
    except ValueError:
        return None
    return decoded if isinstance(decoded, str) else None


def _sanitize_raw_request_string_for_metadata(value: str) -> str | None:
    if not _looks_like_raw_http_request(value):
        return None
    try:
        parsed = parse_raw_request(value)
        body = extract_body_from_raw_request(value)
        if body is None:
            body = (parsed.get("body") or {}).get("text")
        return format_raw_request_for_debug_metadata(parsed, body)
    except Exception:
        return None


def sanitize_transformed_request_for_metadata(
    value: Any, headers: dict[str, str] | None = None
) -> Any:
    sanitized = sanitize_request_body_for_metadata(value, headers)
    if not isinstance(sanitized, str):
        return sanitized
    raw = _sanitize_raw_request_string_for_metadata(sanitized)
    if raw:
        return raw
    decoded = _decode_escaped_string_literal(sanitized)
    if decoded:
        raw = _sanitize_raw_request_string_for_metadata(decoded)
        if raw:
            return raw

    lines = []
    for line in re.split(r"\r?\n", sanitized):
        request_line = re.fullmatch(r"([A-Z]+)\s+(\S+)\s+(HTTP/\d(?:\.\d)?)", line, re.I)
        if request_line:
            lines.append(
                f"{request_line.group(1)} {sanitize_url(request_line.group(2))} "
                f"{request_line.group(3)}"
            )
            continue
        header_line = re.fullmatch(r"([^:\s][^:]*):\s*(.*)", line, re.S)
        if header_line:
            name, header_value = header_line.group(1), header_line.group(2)
            if is_secret_field(name) or looks_like_secret(header_value):
                lines.append(f"{name}: {REDACTED}")
                continue
        sanitized_line = sanitize_request_body_for_metadata(line, headers)
        lines.append(
            sanitized_line
            if isinstance(sanitized_line, str)
            else (safe_json_stringify(sanitized_line) or js_string(sanitized_line))
        )
    return "\n".join(lines)


def log_transformed_prompt(
    transformed_prompt: Any, prompt: Any, headers: dict[str, str] | None = None
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "[HTTP Provider]: Transformed prompt: %s. Original prompt: %s",
        safe_json_stringify(sanitize_transformed_request_for_metadata(transformed_prompt, headers)),
        safe_json_stringify(sanitize_transformed_request_for_metadata(prompt, headers)),
    )


def format_raw_request_for_debug_metadata(
    parsed_request: dict[str, Any], body_content: str | None = None
) -> str:
    lines = [
        f"{parsed_request['method']} {sanitize_url(parsed_request['url'])} "
        f"{parsed_request['httpVersion']}"
    ]
    sanitized_headers = sanitize_object(parsed_request["headers"], context="request headers")
    for key, value in (sanitized_headers or {}).items():
        lines.append(f"{key}: {value}")
    sanitized_body = sanitize_transformed_request_for_metadata(
        body_content, parsed_request["headers"]
    )
    if sanitized_body is not None:
        lines.extend(["", js_string(sanitized_body)])
    return "\n".join(lines)
