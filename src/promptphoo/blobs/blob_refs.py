"""Ports src/blobs/blobRefs.ts: finding `promptfoo://blob/<sha256>` references in values."""

from __future__ import annotations

import re
from typing import Any

from promptphoo.blobs.constants import BLOB_SCHEME

BLOB_HASH_REGEX = re.compile(r"^[a-f0-9]{64}\Z", re.I)
BLOB_URI_REGEX = re.compile(r"promptfoo://blob/([a-f0-9]{64})", re.I)

# Shared bounds for scanning result values for blob URIs at share time.
BLOB_SCAN_MAX_DEPTH = 8
BLOB_SCAN_MAX_STRING_LENGTH = 100_000


def normalize_blob_hash(hash_: str) -> str:
    return hash_.lower()


def _should_scan_string(value: str, max_string_length: int | None) -> bool:
    if BLOB_SCHEME not in value:
        return False
    return (
        max_string_length is None
        or value.startswith(BLOB_SCHEME)
        or len(value) <= max_string_length
    )


def extract_blob_hashes_from_string(value: str, max_string_length: int | None = None) -> list[str]:
    if not _should_scan_string(value, max_string_length):
        return []
    return [normalize_blob_hash(m.group(1)) for m in BLOB_URI_REGEX.finditer(value)]


def extract_blob_hashes_from_value(value: Any, max_string_length: int | None = None) -> list[str]:
    if isinstance(value, str):
        return extract_blob_hashes_from_string(value, max_string_length)
    if not isinstance(value, dict):
        # Arrays are objects in JS too, but `candidate.hash`/`.uri` are undefined on them.
        return []
    hash_ = value.get("hash")
    if isinstance(hash_, str) and BLOB_HASH_REGEX.match(hash_):
        return [normalize_blob_hash(hash_)]
    uri = value.get("uri")
    if isinstance(uri, str):
        return extract_blob_hashes_from_string(uri, max_string_length)
    return []


def collect_blob_hashes(
    value: Any,
    *,
    max_depth: int | None = None,
    max_string_length: int | None = None,
) -> dict[str, None]:
    """collectBlobHashes. The result is a dict used as an ordered set, so that iterating it
    follows discovery order the way a JS `Set` does (export writes `blobAssets` in that order)."""
    found: dict[str, None] = {}
    _collect(value, max_depth, max_string_length, found, set(), 0)
    return found


def _collect(
    value: Any,
    max_depth: int | None,
    max_string_length: int | None,
    found: dict[str, None],
    visited: set[int],
    depth: int,
) -> None:
    if max_depth is not None and depth > max_depth:
        return
    for hash_ in extract_blob_hashes_from_value(value, max_string_length):
        found.setdefault(hash_, None)
    if not isinstance(value, dict | list):
        return
    if id(value) in visited:
        return
    visited.add(id(value))
    children = value if isinstance(value, list) else list(value.values())
    for child in children:
        _collect(child, max_depth, max_string_length, found, visited, depth + 1)
