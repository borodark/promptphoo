"""Ports src/blobs/types.ts (and the BlobRef contract in src/contracts/blobs.ts).

`BlobRef` and `BlobMetadata` stay plain dicts with Node's camelCase keys, in Node's key order:
`{uri, hash, mimeType, sizeBytes, provider}` and `{mimeType, sizeBytes, createdAt, provider,
key}`. They end up inside persisted provider responses and `.meta.json` sidecars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class StoredBlob:
    data: bytes
    metadata: dict[str, Any]


@dataclass
class BlobStoreResult:
    ref: dict[str, Any]
    deduplicated: bool


class BlobStorageProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def store(self, data: bytes, mime_type: str) -> BlobStoreResult: ...

    def get_by_hash(self, hash_: str) -> StoredBlob: ...

    def exists(self, hash_: str) -> bool: ...

    def delete_by_hash(self, hash_: str) -> None: ...

    def get_url(self, hash_: str, expires_in_seconds: int | None = None) -> str | None: ...
