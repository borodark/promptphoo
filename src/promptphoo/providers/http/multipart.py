"""Structured multipart bodies (`config.multipart`). Ports src/providers/httpMultipart.ts and
src/providers/urlEncoding.ts.

Node hands a FormData to fetch, which picks the boundary and encodes the parts; here the body
is encoded the way undici does it and the Content-Type header carries the boundary.
"""

from __future__ import annotations

import base64
import os
import random
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

from promptphoo import cli_state
from promptphoo.jsvalues import js_string
from promptphoo.sanitizer import decode_uri_component
from promptphoo.templating import get_engine


def decode_url_component(value: str) -> str:
    """Decode a URL component while retaining malformed escapes as written."""
    try:
        return decode_uri_component(value)
    except ValueError:
        return value


@dataclass
class MultipartFormData:
    """An encoded multipart/form-data body (what undici produces for a FormData)."""

    boundary: str
    content: bytes

    @property
    def content_type(self) -> str:
        return f"multipart/form-data; boundary={self.boundary}"


def _render_template(value: str, vars: dict[str, Any]) -> str:
    return get_engine().render_string(value, vars)


def _get_mime_type_for_generated_format(fmt: str | None) -> str:
    if fmt == "png":
        return "image/png"
    if fmt in ("jpg", "jpeg"):
        return "image/jpeg"
    return "application/pdf"


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def create_basic_pdf(text: str) -> bytes:
    display_text = _escape_pdf_text(re.sub(r"\s+", " ", text).strip()[:800])
    content = f"BT\n/F1 16 Tf\n72 720 Td\n({display_text}) Tj\nET"
    objects = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << "
        "/F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        "4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {len(content.encode())} >>\nstream\n{content}\nendstream\nendobj\n",
    ]
    pdf = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf.encode()))
        pdf += obj
    xref_offset = len(pdf.encode())
    pdf += f"xref\n0 {len(objects) + 1}\n"
    pdf += "0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{str(offset).rjust(10, '0')} 00000 n \n"
    pdf += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
    pdf += f"startxref\n{xref_offset}\n%%EOF\n"
    return pdf.encode("utf-8")


_PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)
_JPEG_1X1 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP////////////////////////////////////////////////////////"
    "//////////////////////////////2wBDAf////////////////////////////////////////////////////"
    "//////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/"
    "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAq"
    "f/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/ASP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/ASP/"
    "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Al//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IV//2g"
    "AMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8QH//EABQRAQAAAAAAAAAAAAAAAAAAABD/"
    "2gAIAQIBAT8QH//EABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z"
)


def create_generated_image(fmt: str) -> bytes:
    return base64.b64decode(_PNG_1X1 if fmt == "png" else _JPEG_1X1)


def _create_generated_file(source: dict[str, Any], vars: dict[str, Any]) -> dict[str, Any]:
    fmt = source.get("format") or "pdf"
    content_type = _get_mime_type_for_generated_format(fmt)
    text = _render_template(
        source.get("text") or "Promptfoo generated document for multipart HTTP target testing.",
        vars,
    )
    if fmt == "pdf":
        return {"buffer": create_basic_pdf(text), "contentType": content_type, "extension": "pdf"}
    return {
        "buffer": create_generated_image(fmt),
        "contentType": content_type,
        "extension": "jpg" if fmt == "jpg" else fmt,
    }


def normalize_file_path(file_path: str) -> str:
    if not file_path.startswith("file://"):
        return file_path
    url = file_path
    if url.startswith("file://localhost/"):
        url = "file:///" + url[len("file://localhost/") :]
    win_drive = re.match(r"^file:///?([a-zA-Z]:[\\/].*)\Z", url, re.S)
    if win_drive:
        return os.path.normpath(decode_url_component(win_drive.group(1)))
    if url.startswith("file:///"):
        # fileURLToPath on POSIX: percent-decode the path; an encoded '/' is rejected.
        path = url[len("file://") :].split("?", 1)[0].split("#", 1)[0]
        if re.search(r"%2f", path, re.I):
            return decode_url_component(url[len("file://") :])
        return urllib.parse.unquote(path)
    return url[len("file://") :]


def resolve_path(file_path: str) -> str:
    without_scheme = normalize_file_path(file_path)
    if os.path.isabs(without_scheme):
        return without_scheme
    return os.path.abspath(os.path.join(cli_state.base_path() or os.getcwd(), without_scheme))


_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".txt": "text/plain",
    ".json": "application/json",
}


def _get_content_type_from_filename(filename: str) -> str:
    return _CONTENT_TYPES.get(os.path.splitext(filename)[1].lower(), "application/octet-stream")


def _load_file_part(source: dict[str, Any], vars: dict[str, Any]) -> dict[str, Any]:
    resolved = resolve_path(_render_template(source["path"], vars))
    with open(resolved, "rb") as f:
        buffer = f.read()
    return {
        "buffer": buffer,
        "filename": os.path.basename(resolved),
        "contentType": _get_content_type_from_filename(resolved),
    }


def _escape_form_name(value: str) -> str:
    return value.replace("\n", "%0A").replace("\r", "%0D").replace('"', "%22")


def _normalize_linefeeds(value: str) -> str:
    return re.sub(r"\r?\n|\r", "\r\n", value)


def encode_form_data(
    entries: list[tuple[str, Any]], boundary: str | None = None
) -> MultipartFormData:
    """Encode (name, str | (bytes, filename, content_type)) entries like undici's extractBody."""
    if boundary is None:
        boundary = f"----formdata-undici-0{str(random.randrange(10**11)).rjust(11, '0')}"
    prefix = f"--{boundary}\r\nContent-Disposition: form-data"
    chunks: list[bytes] = []
    for name, value in entries:
        escaped_name = _escape_form_name(_normalize_linefeeds(name))
        if isinstance(value, str):
            chunks.append(
                f'{prefix}; name="{escaped_name}"\r\n\r\n{_normalize_linefeeds(value)}\r\n'.encode()
            )
        else:
            data, filename, content_type = value
            head = f'{prefix}; name="{escaped_name}"'
            if filename:
                head += f'; filename="{_escape_form_name(filename)}"'
            head += f"\r\nContent-Type: {content_type or 'application/octet-stream'}\r\n\r\n"
            chunks.append(head.encode() + data + b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return MultipartFormData(boundary, b"".join(chunks))


def render_http_multipart_body(config: dict[str, Any], vars: dict[str, Any]) -> dict[str, Any]:
    """Render `config.multipart` into {body: MultipartFormData, fields, files}."""
    entries: list[tuple[str, Any]] = []
    fields: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for part in config.get("parts") or []:
        field = _render_template(part["name"], vars)
        if part.get("kind") == "field":
            value = _render_template(js_string(part.get("value")), vars)
            entries.append((field, value))
            fields.append({"field": field, "value": value})
            continue
        source = part.get("source") or {}
        if source.get("type") == "path":
            loaded = _load_file_part(source, vars)
            default_filename = loaded["filename"]
        else:
            loaded = _create_generated_file(source, vars)
            default_filename = f"promptfoo-document.{loaded['extension']}"
        filename_template = part.get("filenameTemplate") or part.get("filename") or default_filename
        filename = _render_template(filename_template, vars)
        content_type = part.get("contentType") or loaded["contentType"]
        entries.append((field, (loaded["buffer"], filename, content_type)))
        files.append(
            {
                "field": field,
                "filename": filename,
                "contentType": content_type,
                "sizeBytes": len(loaded["buffer"]),
                "source": source.get("type"),
            }
        )
    return {"body": encode_form_data(entries), "fields": fields, "files": files}
