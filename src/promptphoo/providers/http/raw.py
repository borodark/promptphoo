"""Raw HTTP request (`config.request`) handling. Ports the raw-request helpers of
src/providers/http.ts, plus the parts of the `http-z` 8.1.1 request parser they rely on.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from promptphoo.jsvalues import json_stringify
from promptphoo.sanitizer import sanitize_url
from promptphoo.templating import get_engine
from promptphoo.whatwg_url import URL, InvalidUrlError, form_urlencoded_parse

logger = logging.getLogger(__name__)

# require('http').METHODS in Node 22.
HTTP_METHODS = (
    "ACL", "BIND", "CHECKOUT", "CONNECT", "COPY", "DELETE", "GET", "HEAD", "LINK", "LOCK",
    "M-SEARCH", "MERGE", "MKACTIVITY", "MKCALENDAR", "MKCOL", "MOVE", "NOTIFY", "OPTIONS",
    "PATCH", "POST", "PROPFIND", "PROPPATCH", "PURGE", "PUT", "QUERY", "REBIND", "REPORT",
    "SEARCH", "SOURCE", "SUBSCRIBE", "TRACE", "UNBIND", "UNLINK", "UNLOCK", "UNSUBSCRIBE",
)  # fmt: skip


def escape_json_variables(vars: dict[str, Any]) -> dict[str, Any]:
    """Escape string values for substitution inside a JSON string literal."""
    return {k: json_stringify(v)[1:-1] if isinstance(v, str) else v for k, v in vars.items()}


def render_raw_request_with_nunjucks(template: str, vars: dict[str, Any]) -> str:
    """Render only the `{{...}}` tags of a raw request; any `{%...%}` text stays literal."""
    var_token = "__PF_VAR__"
    placeholders: list[str] = []

    def stash(m: re.Match[str]) -> str:
        placeholders.append(m.group(0))
        return f"{var_token}{len(placeholders) - 1}__"

    working = re.sub(r"\{\{[\s\S]*?\}\}", stash, template)
    working = "{% raw %}" + working + "{% endraw %}"
    working = re.sub(
        rf"{var_token}(\d+)__",
        lambda m: "{% endraw %}" + placeholders[int(m.group(1))] + "{% raw %}",
        working,
    )
    return get_engine().render_string(working, vars)


def _fail(message: str) -> None:
    logger.error(message)
    raise ValueError(message)


def url_encode_raw_request_path(raw_request: str) -> str:
    """Re-encode the request-target of the first line through WHATWG URL parsing."""
    first_line = raw_request.split("\n")[0]

    def safe_first_line() -> str:
        return sanitize_url(first_line)

    first_space = first_line.find(" ")
    method = first_line[:first_space]
    if not method or method not in HTTP_METHODS:
        _fail(
            f"[Http Provider] HTTP request method {method} is not valid. From: {safe_first_line()}"
        )
    last_space = first_line.rfind(" ")
    if last_space == -1:
        _fail(
            "[Http Provider] HTTP request URL is not valid. Protocol is missing. "
            f"From: {safe_first_line()}"
        )
    url = first_line[first_space + 1 : last_space] if last_space > first_space else ""
    if len(url) == 0:
        _fail(f"[Http Provider] HTTP request URL is not valid. From: {safe_first_line()}")

    protocol = first_line[last_space + 1 :] if last_space < len(first_line) else ""
    if not protocol.lower().startswith("http"):
        _fail(f"[Http Provider] HTTP request protocol is not valid. From: {safe_first_line()}")

    sanitized_url = sanitize_url(url)
    sanitized_first_line = f"{method} {sanitized_url}{f' {protocol}' if protocol else ''}"
    logger.debug(
        "[Http Provider] Encoding URL: %s from first line of raw request: %s",
        sanitized_url,
        sanitized_first_line,
    )
    try:
        parsed_url = URL(url, "http://placeholder-base.com")
    except InvalidUrlError as err:
        _fail(f"[Http Provider] Error parsing URL in HTTP request: TypeError: {err}")
        raise  # unreachable
    new_first_line = (
        f"{method} {parsed_url.pathname}{parsed_url.search}{' ' + protocol if protocol else ''}"
    )
    return raw_request.replace(first_line, new_first_line, 1)


def normalize_http_line_endings(value: str) -> str:
    """Convert every line ending to CRLF and trim surrounding whitespace."""
    normalized = value.replace("\r\n", "\n").strip()
    return normalized.replace("\n", "\r\n")


def get_raw_request_http_version(value: str) -> str:
    first_line = normalize_http_line_endings(value).split("\r\n", 1)[0].strip()
    match = re.search(r"\s+(HTTP/\d(?:\.\d)?)\Z", first_line, re.I)
    return match.group(1).upper() if match else "HTTP/1.1"


# --- http-z 8.1.1 (request parsing only) ------------------------------------------------------

EOL = "\r\n"
_REQUEST_START_ROW = re.compile(
    r"^(CONNECT|OPTIONS|TRACE|GET|HEAD|POST|PUT|PATCH|DELETE)\s\S*\s"
    r"(HTTP)/(1\.0|1\.1|2(\.0){0,1}|3(\.0){0,1})\Z"
)
_RESPONSE_START_ROW = re.compile(r"^(HTTP)/(1\.0|1\.1|2(\.0){0,1}|3(\.0){0,1})\s\d{3}\s[^\r\n]*\Z")
_QUOTED_HEADER_VALUE = re.compile(r'^"[\u0009 !#-~]+"\Z')
_BOUNDARY = re.compile(r"""(?<=boundary=)"?[A-Za-z0-9'()+_,.:=?-]+"?""")
_CONTENT_DISPOSITION = re.compile(
    r"^Content-Disposition:\s*(form-data|inline|attachment)"
    r'(?:\s*;\s*(name|filename)\s*=\s*(?:"([^"]+)"|([^;\s]+)))*\r\n',
    re.I,
)
_CONTENT_TYPE = re.compile(r"^Content-Type:[\S ]*\r\n", re.I)
_DISPOSITION_TYPE = re.compile(r"(?<=Content-Disposition:)\s*(form-data|inline|attachment)")
_CHUNK_ROW = re.compile(r"^[0-9a-fA-F]+\r\n")
_MULTIPART_TYPES = (
    "multipart/form-data",
    "multipart/alternative",
    "multipart/mixed",
    "multipart/related",
)


class HttpZError(Exception):
    def __str__(self) -> str:
        return f"HttpZError: {self.args[0]}"


def _split_by(value: str, delimiter: str) -> list[str]:
    if not value:
        return []
    idx = value.find(delimiter)
    if idx == -1:
        return []
    return [value[:idx].strip(), value[idx + len(delimiter) :].strip()]


def _prettify_header_name(name: str) -> str:
    return "-".join(p[:1].upper() + p[1:].lower() if p else "" for p in name.split("-"))


def _trim_chars(value: str, chars: str) -> str:
    m = re.match(rf"^([{chars}]*)(.*?)([{chars}]*)\Z", value, re.S)
    return m.group(2) if m else value


def _parse_form_data_param(group: str) -> dict[str, Any]:
    group = re.sub(r"^\r\n", "", group, count=1)
    group = re.sub(r"\r\n\Z", "", group, count=1)
    disposition = _CONTENT_DISPOSITION.match(group)
    if not disposition:
        raise HttpZError("Incorrect Content-Disposition")
    group = group.replace(disposition.group(0), "", 1)
    header = re.sub(r"\r\n\Z", "", disposition.group(0))
    content_type_match = _CONTENT_TYPE.match(group)
    content_type = None
    if content_type_match:
        group = group.replace(content_type_match.group(0), "", 1)
        content_type = re.sub(
            r"\r\n\Z", "", re.sub(r"^content-type: *", "", content_type_match.group(0).lower())
        )
    disposition_type = _DISPOSITION_TYPE.search(header)
    if not disposition_type:
        raise HttpZError("Incorrect Content-Disposition type")
    kind = disposition_type.group(0).strip().lower()
    param: dict[str, Any] = {}
    if kind == "form-data":
        name = re.search(r'(?<=name=)(?:"([^"]+)"|([^;\s]+))+', header, re.I)
        if not name:
            raise HttpZError("Incorrect Content-Disposition, expected param name")
        param["name"] = _trim_chars(name.group(0), '"')
    if not group.startswith(EOL):
        raise HttpZError("Incorrect form-data parameter")
    param["value"] = group[len(EOL) :]
    if kind != "form-data":
        param["type"] = kind
    if content_type is not None:
        param["contentType"] = content_type
    filename = re.search(r'(?<=filename=)(?:"([^"]+)"|([^;\s]+))+', header, re.I)
    if filename:
        param["fileName"] = _trim_chars(filename.group(0), '"')
    return param


def http_z_parse_request(raw_message: str) -> dict[str, Any]:
    """`httpZ.parse(raw)` for request messages: method, target, headers and body."""
    first_row = raw_message.split(EOL)[0]
    if not _REQUEST_START_ROW.match(first_row):
        if _RESPONSE_START_ROW.match(first_row):
            raise HttpZError("Expected a request message, got a response")
        raise HttpZError("rawMessage has incorrect format")

    parts = _split_by(raw_message, EOL + EOL)
    if not parts:
        raise HttpZError(
            "Incorrect message format, expected: start-line CRLF *(header-field CRLF) CRLF "
            "[message-body]"
        )
    head, body_rows = parts
    header_rows = head.split(EOL)
    start_row, header_rows = header_rows[0], header_rows[1:]
    cookies_row = next((r for r in header_rows if r.lower().startswith("cookie:")), None)

    if not _REQUEST_START_ROW.match(start_row):
        raise HttpZError("Incorrect startRow format, expected: Method request-target HTTP-Version")
    row_parts = start_row.split(" ")
    method = row_parts[0].upper()
    target = row_parts[1]
    try:
        URL(target, "http://superrandomhost28476561927456.com") if target else None
    except InvalidUrlError:
        raise HttpZError("Invalid target") from None

    headers: list[dict[str, str]] = []
    for row in header_rows:
        split = _split_by(row, ":")
        if not split or not split[0]:
            raise HttpZError("Incorrect header row format, expected: Name: Value")
        name, value = split
        if _QUOTED_HEADER_VALUE.match(value):
            value = _trim_chars(value, '"')
        headers.append({"name": _prettify_header_name(name), "value": value})

    if cookies_row is not None:
        split = _split_by(cookies_row, ":")
        if not split or not split[0]:
            raise HttpZError("Incorrect cookie row format, expected: Cookie: Name1=Value1;...")
        if split[1]:
            for pair in split[1].split(";"):
                cookie = _split_by(pair, "=")
                if not cookie or not cookie[0]:
                    raise HttpZError("Incorrect cookie pair format, expected: Name1=Value1;...")

    model: dict[str, Any] = {"method": method, "target": target, "headers": headers}
    if body_rows:
        body_rows = _process_transfer_encoding_chunked(headers, body_rows)
        body: dict[str, Any] = {}
        content_type = next(
            (h["value"] for h in headers if h["name"] == "Content-Type" and h["value"]), None
        )
        if content_type:
            body["contentType"] = content_type.lower().split(";")[0]
        if body.get("contentType") in _MULTIPART_TYPES:
            body["boundary"] = _get_boundary(content_type)
            groups = body_rows.split(f"--{body['boundary']}")
            body["params"] = [_parse_form_data_param(g) for g in groups[1:-1]]
        elif body.get("contentType") == "application/x-www-form-urlencoded":
            body["params"] = [
                {"name": n, "value": v} for n, v in form_urlencoded_parse(body_rows.lstrip("?"))
            ]
        else:
            body["text"] = body_rows
        model["body"] = body
    return model


def _process_transfer_encoding_chunked(headers: list[dict[str, str]], text: str) -> str:
    if not any(h["name"] == "Transfer-Encoding" and "chunked" in h["value"] for h in headers):
        return text
    original = text
    chunks: list[str] = []
    while True:
        row = _CHUNK_ROW.match(text)
        first_row = row.group(0) if row else ""
        try:
            length = int(first_row.strip(), 16) if first_row else 0
        except ValueError:
            length = 0
        if not length:
            raise HttpZError("Incorrect row, expected: NumberEOL")
        text = text[len(first_row) :]
        chunks.append(text[:length])
        text = text[length + len(EOL) :]
        if not text:
            break
    del original
    return "".join(chunks)


def _get_boundary(content_type: str | None) -> str:
    if not content_type:
        raise HttpZError(
            "Message with multipart/form-data body must have Content-Type header with boundary"
        )
    params = content_type.split(";")[1] if ";" in content_type else ""
    if not params:
        raise HttpZError(
            "Message with multipart/form-data body must have Content-Type header with boundary"
        )
    boundary = _BOUNDARY.search(params)
    if not boundary:
        raise HttpZError("Incorrect boundary, expected: boundary=value")
    return _trim_chars(boundary.group(0), '"')


# ----------------------------------------------------------------------------------------------


def parse_raw_request(value: str) -> dict[str, Any]:
    """Parse a raw HTTP request into {method, url, httpVersion, headers, body}."""
    adjusted = normalize_http_line_endings(value) + "\r\n\r\n"
    # If the injectVar is in a query param, the URL in the first line must be encoded.
    encoded = url_encode_raw_request_path(adjusted)
    try:
        model = http_z_parse_request(encoded)
    except Exception as err:
        raise ValueError(f"Error parsing raw HTTP request: {_js_error_string(err)}") from err
    headers: dict[str, str] = {}
    for header in model["headers"]:
        headers[header["name"].lower()] = header["value"]
    out: dict[str, Any] = {
        "method": model["method"],
        "url": model["target"],
        "httpVersion": get_raw_request_http_version(value),
        "headers": headers,
    }
    if "body" in model:
        out["body"] = model["body"]
    return out


def _js_error_string(err: BaseException) -> str:
    return str(err) if isinstance(err, HttpZError) else f"Error: {err}"


def extract_body_from_raw_request(raw_request: str) -> str | None:
    """The raw body text of a request (after the blank line), trimmed; None when empty."""
    adjusted = normalize_http_line_endings(raw_request)
    idx = adjusted.find("\r\n\r\n")
    if idx == -1:
        return None
    body = adjusted[idx + 4 :].strip()
    return body if body else None
