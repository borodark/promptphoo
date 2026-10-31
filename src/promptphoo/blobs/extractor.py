"""Ports src/blobs/extractor.ts: moving media out of provider responses into the blob store.

`extract_and_store_binary_data` scans, in upstream's order: `audio.data`, `images[].data` data
URLs, a raw SVG `output` (stored as a preview, the text is kept), `turns[].audio.data`, a data
URL `output`, OpenAI-style `{"data":[{"b64_json":...}]}` JSON output, `metadata.audio.data`,
and data URLs anywhere under `metadata`. Payloads of 1KB to 50MB are stored (SVG previews at
any size) and replaced by a `blobRef` or a `promptfoo://blob/<sha256>` URI. Byte-identical
payloads share one store per response, and the blob URIs already present in the response are
recorded as references of the eval.

The raw SVG check stands in for fast-xml-parser's `XMLValidator` with a non-namespaced expat
well-formedness parse plus the ported fxp parser's root check. They agree on ordinary SVG; an
SVG using an HTML entity such as `&nbsp;` passes fxp's validator but not expat (see D48).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
from collections.abc import Callable
from typing import Any

from promptphoo.blobs.blob_refs import extract_blob_hashes_from_value
from promptphoo.blobs.constants import BLOB_MAX_SIZE, BLOB_MIN_SIZE, BLOB_SCHEME

logger = logging.getLogger("promptphoo")

_DATA_URL_RE = re.compile(r"^data:(audio|image)/[^;]+;base64,")
# JS `.` stops at the four line terminators, and `$` (no m flag) means end of input.
_BASE64_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,([^\n\r  ]+)\Z")
_AUDIO_MIME_RE = re.compile(r"^audio/[a-z0-9_+-]+\Z", re.I)
_AUDIO_FORMAT_RE = re.compile(r"^[a-z0-9_-]+\Z", re.I)
_AUDIO_MIME_MAP = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "m4a": "audio/mp4",
    "webm": "audio/webm",
}
# String.prototype.trim's whitespace set (ECMAScript WhiteSpace + LineTerminator).
_JS_WHITESPACE = " \t\n\v\f\r                 　﻿"


def buffer_from_base64(data: str) -> bytes:
    """Buffer.from(data, 'base64'): accepts url-safe letters, skips junk, stops at '='."""
    cleaned = data.replace("-", "+").replace("_", "/")
    cleaned = cleaned.split("=", 1)[0]
    cleaned = re.sub(r"[^A-Za-z0-9+/]", "", cleaned)
    if len(cleaned) % 4 == 1:
        cleaned = cleaned[:-1]
    try:
        return base64.b64decode(cleaned + "=" * (-len(cleaned) % 4))
    except binascii.Error:
        return b""


def _is_data_url(value: str) -> bool:
    return bool(_DATA_URL_RE.match(value))


def _extract_base64(value: str) -> tuple[bytes, str] | None:
    match = _BASE64_DATA_URL_RE.match(value)
    if not match:
        return None
    return buffer_from_base64(match.group(2)), match.group(1)


def _should_externalize(buffer: bytes, min_size_bytes: int | None = None) -> bool:
    min_size = BLOB_MIN_SIZE if min_size_bytes is None else min_size_bytes
    return min_size <= len(buffer) <= BLOB_MAX_SIZE


def _get_kind_from_mime_type(mime_type: str) -> str:
    return "audio" if mime_type.startswith("audio/") else "image"


def normalize_audio_mime_type(fmt: str | None) -> str:
    """Normalize an audio format to a MIME type ('wav' -> 'audio/wav'). Periods are never
    allowed, so 'wav.html' cannot become an HTML-sniffable MIME type."""
    if not fmt:
        return "audio/wav"
    trimmed = fmt.strip(_JS_WHITESPACE)
    if _AUDIO_MIME_RE.match(trimmed):
        return trimmed
    lower = trimmed.lower()
    if lower in _AUDIO_MIME_MAP:
        return _AUDIO_MIME_MAP[lower]
    if not _AUDIO_FORMAT_RE.match(lower):
        logger.warning("[BlobExtractor] Invalid audio format, using default (format=%r)", fmt)
        return "audio/wav"
    return f"audio/{lower}"


def _parse_binary(base64_or_data_url: str, default_mime_type: str) -> tuple[bytes, str] | None:
    if _is_data_url(base64_or_data_url):
        return _extract_base64(base64_or_data_url)
    return buffer_from_base64(base64_or_data_url), default_mime_type


def is_blob_storage_enabled() -> bool:
    """Single toggle: externalize by default; opt out with PROMPTFOO_INLINE_MEDIA=true."""
    from promptphoo.envars import get_env_bool

    return not get_env_bool("PROMPTFOO_INLINE_MEDIA", False)


def _maybe_store(
    base64_or_data_url: str,
    default_mime_type: str,
    context: dict[str, Any],
    location: str,
    kind: str,
    min_size_bytes: int | None = None,
) -> dict[str, Any] | None:
    from promptphoo.blobs import store_blob

    parsed = _parse_binary(base64_or_data_url, default_mime_type)
    if not parsed or not _should_externalize(parsed[0], min_size_bytes):
        return None
    if not is_blob_storage_enabled():
        return None
    buffer, mime_type = parsed
    # Blob extraction is local-only. Remote synchronization happens when an eval is shared.
    ref_context = {**context, "location": location, "kind": kind}
    return store_blob(buffer, mime_type or "application/octet-stream", ref_context).ref


StoreOnce = Callable[..., "dict[str, Any] | None"]


def _create_store_once(blob_context: dict[str, Any]) -> StoreOnce:
    """Per-response store-once: byte-identical payloads give the same BlobRef however they are
    encoded (raw base64 or a data URL) and wherever they appear, with one store_blob write."""
    cache: dict[str, dict[str, Any]] = {}

    def store_once(
        base64_or_data_url: str,
        default_mime_type: str,
        location: str,
        kind: str,
        min_size_bytes: int | None = None,
    ) -> dict[str, Any] | None:
        parsed = _parse_binary(base64_or_data_url, default_mime_type)
        if not parsed or not _should_externalize(parsed[0], min_size_bytes):
            return None
        cache_key = f"{kind}:{parsed[0].hex()}"
        if cache_key in cache:
            return cache[cache_key]
        stored = _maybe_store(
            base64_or_data_url, default_mime_type, blob_context, location, kind, min_size_bytes
        )
        if stored:
            cache[cache_key] = stored
        return stored

    return store_once


# --- raw SVG output previews ----------------------------------------------------------------


def _is_well_formed_xml(text: str) -> bool:
    """fast-xml-parser `XMLValidator.validate(text) === true`, approximated (module docstring)."""
    import xml.parsers.expat as expat

    parser = expat.ParserCreate()
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    try:
        parser.Parse(text, True)
    except (expat.ExpatError, ValueError, UnicodeError):
        return False
    return True


def _get_raw_svg_output_preview(output: str) -> tuple[str, str] | None:
    from promptphoo.assertions.markup import _fxp_parse

    trimmed = output.strip(_JS_WHITESPACE)
    if not trimmed.startswith("<"):
        return None
    if not _is_well_formed_xml(trimmed):
        return None
    try:
        parsed = _fxp_parse(trimmed)
    except Exception:
        return None
    if not isinstance(parsed, dict) or "svg" not in parsed:
        return None
    buffer = trimmed.encode("utf-8")
    data_url = f"data:image/svg+xml;base64,{base64.b64encode(buffer).decode('ascii')}"
    return data_url, f"{BLOB_SCHEME}{hashlib.sha256(buffer).hexdigest()}"


def _existing_blob_uris(metadata: Any) -> list[str]:
    blob_uris = metadata.get("blobUris") if isinstance(metadata, dict) else None
    if not isinstance(blob_uris, list):
        return []
    return [value for value in blob_uris if isinstance(value, str)]


def _append_metadata_blob_uri(metadata: Any, uri: str) -> dict[str, Any]:
    existing = _existing_blob_uris(metadata)
    base = metadata if isinstance(metadata, dict) else {}
    return {**base, "blobUris": list(dict.fromkeys([*existing, uri]))}


def _store_raw_svg_output_preview(
    output: Any, metadata: Any, store_once: StoreOnce, context: dict[str, Any] | None
) -> tuple[Any, bool]:
    if not isinstance(output, str):
        return metadata, False
    preview = _get_raw_svg_output_preview(output)
    if not preview:
        return metadata, False
    data_url, uri = preview
    if uri in _existing_blob_uris(metadata):
        return metadata, False
    stored = store_once(data_url, "image/svg+xml", "response.output", "image", 0)
    if not stored:
        return metadata, False
    logger.debug("[BlobExtractor] Stored raw SVG output blob %s %s", context or {}, stored["hash"])
    return _append_metadata_blob_uri(metadata, stored["uri"]), True


# --- metadata --------------------------------------------------------------------------------


def _externalize_data_urls(value: Any, store_once: StoreOnce, location: str) -> tuple[Any, bool]:
    if isinstance(value, str):
        if not _is_data_url(value):
            return value, False
        parsed = _extract_base64(value)
        if not parsed:
            return value, False
        # The raw data URL goes through store_once so it canonicalizes on the parsed bytes,
        # sharing the per-response cache with output / images[] / turns[] / top-level audio.
        stored = store_once(value, parsed[1], location, _get_kind_from_mime_type(parsed[1]))
        if not stored:
            return value, False
        return stored["uri"], True

    if isinstance(value, list):
        mutated = False
        next_values = []
        for idx, item in enumerate(value):
            next_value, child_mutated = _externalize_data_urls(
                item, store_once, f"{location}[{idx}]"
            )
            mutated = mutated or child_mutated
            next_values.append(next_value)
        return (next_values, True) if mutated else (value, False)

    if isinstance(value, dict):
        mutated = False
        next_object = dict(value)
        for key, child in value.items():
            next_value, child_mutated = _externalize_data_urls(
                child, store_once, f"{location}.{key}" if location else key
            )
            if child_mutated:
                next_object[key] = next_value
                mutated = True
        return (next_object, True) if mutated else (value, False)

    return value, False


def _externalize_metadata_audio(metadata: Any, store_once: StoreOnce) -> tuple[Any, bool]:
    if not isinstance(metadata, dict):
        return metadata, False
    audio = metadata.get("audio")
    if not isinstance(audio, dict) or not isinstance(audio.get("data"), str):
        return metadata, False
    fmt = audio.get("format")
    stored = store_once(
        audio["data"],
        normalize_audio_mime_type(fmt if isinstance(fmt, str) else None),
        "response.metadata.audio.data",
        "audio",
    )
    if not stored:
        return metadata, False
    return {**metadata, "audio": _replace_data_with_ref(audio, stored)}, True


def _replace_data_with_ref(record: dict[str, Any], stored: dict[str, Any]) -> dict[str, Any]:
    """`{...record, data: undefined, blobRef: stored}` as JSON sees it."""
    out = {k: v for k, v in record.items() if k not in ("data", "blobRef")}
    out["blobRef"] = stored
    return out


def _js_truthy_string(value: Any) -> bool:
    return isinstance(value, str) and value != ""


# --- the entry point ------------------------------------------------------------------------


def extract_and_store_binary_data(
    response: dict[str, Any] | None, context: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Best-effort extraction of binary data from a provider response. Returns the response
    itself when nothing changed, else a shallow copy with the media replaced."""
    if not response:
        return response

    from promptphoo.jsonutil import js_json_parse

    mutated = False
    next_: dict[str, Any] = dict(response)
    blob_context = context or {}
    store_once = _create_store_once(blob_context)

    # Audio at top level
    audio = response.get("audio")
    if isinstance(audio, dict) and _js_truthy_string(audio.get("data")):
        fmt = audio.get("format")
        stored = store_once(
            audio["data"],
            normalize_audio_mime_type(fmt if isinstance(fmt, str) else None),
            "response.audio.data",
            "audio",
        )
        if stored:
            next_["audio"] = _replace_data_with_ref(audio, stored)
            mutated = True
            logger.debug("[BlobExtractor] Stored audio blob %s", stored["hash"])

    # Images array
    images = response.get("images")
    if isinstance(images, list) and images:
        externalized = []
        for idx, img in enumerate(images):
            data = img.get("data") if isinstance(img, dict) else None
            if not (isinstance(data, str) and data and _is_data_url(data)):
                externalized.append(img)
                continue
            stored = store_once(
                data, img.get("mimeType") or "image/png", f"response.images[{idx}].data", "image"
            )
            if stored:
                mutated = True
                logger.debug("[BlobExtractor] Stored image blob %s", stored["hash"])
                externalized.append(_replace_data_with_ref(img, stored))
            else:
                externalized.append(img)
        next_["images"] = externalized

    # Raw SVG text is user-visible media even when it is smaller than the generic
    # externalization threshold. Store a preview blob and keep the text output.
    svg_metadata, svg_mutated = _store_raw_svg_output_preview(
        response.get("output"),
        next_.get("metadata") or response.get("metadata"),
        store_once,
        context,
    )
    if svg_mutated:
        next_["metadata"] = svg_metadata
        mutated = True

    # Turns audio (multi-turn)
    turns = response.get("turns")
    if isinstance(turns, list):
        updated_turns = []
        for idx, turn in enumerate(turns):
            turn_audio = turn.get("audio") if isinstance(turn, dict) else None
            if isinstance(turn_audio, dict) and _js_truthy_string(turn_audio.get("data")):
                fmt = turn_audio.get("format")
                stored = store_once(
                    turn_audio["data"],
                    normalize_audio_mime_type(fmt if isinstance(fmt, str) else None),
                    f"response.turns[{idx}].audio.data",
                    "audio",
                )
                if stored:
                    mutated = True
                    assert isinstance(turn, dict)
                    replaced = _replace_data_with_ref(turn_audio, stored)
                    updated_turns.append({**turn, "audio": replaced})
                    continue
            updated_turns.append(turn)
        next_["turns"] = updated_turns

    # Output data URL (images/audio) inside a string
    output = response.get("output")
    if isinstance(output, str) and _is_data_url(output):
        parsed = _extract_base64(output)
        if parsed and _should_externalize(parsed[0]):
            stored = store_once(
                output, parsed[1], "response.output", _get_kind_from_mime_type(parsed[1])
            )
            if stored:
                next_["output"] = stored["uri"]
                mutated = True
                logger.debug("[BlobExtractor] Stored output blob %s", stored["hash"])

    # OpenAI (and similar) image responses often arrive as JSON strings with b64_json fields.
    if (
        isinstance(output, str)
        and output.strip(_JS_WHITESPACE).startswith("{")
        and (
            (_js_truthy(response.get("isBase64")) and response.get("format") == "json")
            or '"b64_json"' in output
            or "b64_json" in output
        )
    ):
        try:
            payload = js_json_parse(output)
            items = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(items, list):
                json_mutated = False
                stored_uris: list[str] = []
                for item in items:
                    b64 = item.get("b64_json") if isinstance(item, dict) else None
                    if _js_truthy_string(b64):
                        assert isinstance(item, dict) and isinstance(b64, str)
                        stored = store_once(
                            b64, "image/png", "response.output.data[].b64_json", "image"
                        )
                        if stored:
                            item["b64_json"] = stored["uri"]
                            stored_uris.append(stored["uri"])
                            json_mutated = True
                            mutated = True
                            logger.debug(
                                "[BlobExtractor] Stored image blob from b64_json %s",
                                stored["hash"],
                            )
                if json_mutated:
                    # Prefer a simple blob ref output so graders/UI don't have to parse JSON
                    from promptphoo.jsvalues import json_stringify

                    if len(stored_uris) == 1:
                        next_["output"] = stored_uris[0]
                    elif len(stored_uris) > 1:
                        next_["output"] = json_stringify(stored_uris)
                    else:
                        next_["output"] = json_stringify(payload)
                    metadata = response.get("metadata")
                    next_["metadata"] = {
                        **(metadata if isinstance(metadata, dict) else {}),
                        "blobUris": stored_uris,
                        **({"originalFormat": response["format"]} if "format" in response else {}),
                    }
        except Exception as err:
            logger.debug(
                "[BlobExtractor] Failed to parse base64 JSON output (location=response.output): %s",
                err,
            )

    metadata = (
        next_.get("metadata") if _js_truthy(next_.get("metadata")) else response.get("metadata")
    )
    if _js_truthy(metadata):
        audio_value, audio_mutated = _externalize_metadata_audio(metadata, store_once)
        value, data_url_mutated = _externalize_data_urls(
            audio_value, store_once, "response.metadata"
        )
        if audio_mutated or data_url_mutated:
            next_["metadata"] = value
            mutated = True

    final_response = next_ if mutated else response
    if blob_context.get("evalId"):
        _record_existing_blob_references(final_response, blob_context, "response")
    return final_response


def _js_truthy(value: Any) -> bool:
    from promptphoo.jsvalues import is_truthy

    return is_truthy(value)


def _record_existing_blob_references(value: Any, context: dict[str, Any], location: str) -> None:
    from promptphoo.blobs import record_blob_reference

    hashes = list(dict.fromkeys(extract_blob_hashes_from_value(value)))
    if hashes:
        for hash_ in hashes:
            record_blob_reference(hash_, {**context, "location": location})
        return
    if isinstance(value, list):
        for idx, child in enumerate(value):
            _record_existing_blob_references(child, context, f"{location}[{idx}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _record_existing_blob_references(
                child, context, f"{location}.{key}" if location else key
            )
