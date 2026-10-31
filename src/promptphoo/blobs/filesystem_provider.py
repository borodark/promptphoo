"""Ports src/blobs/filesystemProvider.ts.

Blobs live at `<configDir>/blobs/<hash[0:2]>/<hash[2:4]>/<sha256>` with a pretty-printed
`<file>.meta.json` sidecar (`{mimeType, sizeBytes, createdAt, provider, key}`), the same files
Node writes, so either runtime reads the other's store. Writes are staged in a
`mkdtemp("<file>.")` directory and published with `os.replace`, data first and then metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from typing import Any

from promptphoo.blobs.blob_refs import BLOB_HASH_REGEX
from promptphoo.blobs.constants import BLOB_SCHEME, DEFAULT_FILESYSTEM_SUBDIR
from promptphoo.blobs.types import BlobStoreResult, StoredBlob
from promptphoo.jsvalues import json_stringify

logger = logging.getLogger("promptphoo")


def iso_now() -> str:
    """new Date().toISOString()."""
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _build_uri(hash_: str) -> str:
    return f"{BLOB_SCHEME}{hash_}"


def _publish_file(source: str, destination: str) -> None:
    retry = 0
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            # Windows can briefly lock a competing writer's destination. Keep replacement atomic.
            if sys.platform != "win32" or retry == 5:
                raise
            retry += 1
            time.sleep(0.05 * retry)


class FilesystemBlobStorageProvider:
    provider_id = "filesystem"

    def __init__(self, base_path: str | None = None) -> None:
        from promptphoo.paths import get_config_directory_path

        default_base = os.path.join(get_config_directory_path(True), DEFAULT_FILESYSTEM_SUBDIR)
        self.base_path = os.path.abspath(base_path or default_base)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)
            logger.debug("[BlobFS] Created blob directory %s", self.base_path)

    def _assert_valid_hash(self, hash_: str) -> None:
        if not isinstance(hash_, str) or not BLOB_HASH_REGEX.match(hash_):
            raise ValueError(f'[BlobFS] Invalid blob hash: "{hash_}"')

    def _resolve_path_in_base(self, unsafe_path: str) -> str:
        target = os.path.abspath(
            unsafe_path if os.path.isabs(unsafe_path) else os.path.join(self.base_path, unsafe_path)
        )
        safe_base = os.path.abspath(self.base_path) + os.sep
        if not target.startswith(safe_base):
            raise ValueError("[BlobFS] Path traversal attempt detected")
        return target

    def get_file_path(self, hash_: str) -> str:
        self._assert_valid_hash(hash_)
        return self._resolve_path_in_base(os.path.join(hash_[:2], hash_[2:4], hash_))

    @staticmethod
    def _metadata_path(file_path: str) -> str:
        return f"{file_path}.meta.json"

    def store(self, data: bytes, mime_type: str) -> BlobStoreResult:
        hash_ = hashlib.sha256(data).hexdigest()
        file_path = self.get_file_path(hash_)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Check if file already exists (deduplication)
        if os.path.exists(file_path):
            meta = self._read_metadata(file_path)
            mime = meta.get("mimeType") if isinstance(meta, dict) else None
            if isinstance(mime, str) and mime:
                assert isinstance(meta, dict)
                size = meta.get("sizeBytes")
                provider = meta.get("provider")
                ref = self._build_ref(
                    hash_,
                    mime,
                    len(data) if size is None else size,
                    self.provider_id if provider is None else provider,
                )
                return BlobStoreResult(ref=ref, deduplicated=True)
            # A concurrent delete may remove metadata before staged bytes are published.
            # Republish incomplete blobs instead of making missing metadata permanent.

        metadata = {
            "mimeType": mime_type,
            "sizeBytes": len(data),
            "createdAt": iso_now(),
            "provider": self.provider_id,
            "key": file_path,
        }
        # Stage complete bytes and metadata before publishing either file.
        directory, prefix = os.path.split(f"{file_path}.")
        staging_dir = tempfile.mkdtemp(prefix=prefix, dir=directory)
        try:
            staged_data = os.path.join(staging_dir, "data")
            staged_metadata = os.path.join(staging_dir, "metadata.json")
            with open(staged_data, "xb") as f:
                f.write(data)
            with open(staged_metadata, "x", encoding="utf-8", newline="") as f:
                f.write(json_stringify(metadata, 2))
            # Publish complete bytes first: a failed data rename must not change another
            # writer's MIME.
            _publish_file(staged_data, file_path)
            _publish_file(staged_metadata, self._metadata_path(file_path))
        finally:
            try:
                shutil.rmtree(staging_dir, ignore_errors=False)
            except FileNotFoundError:
                pass
            except OSError as error:
                # Never mask the original write error or remove another writer's files.
                logger.warning("[BlobFS] Failed to remove staging directory: %s", error)

        return BlobStoreResult(
            ref=self._build_ref(hash_, mime_type, len(data), self.provider_id),
            deduplicated=False,
        )

    def get_by_hash(self, hash_: str) -> StoredBlob:
        file_path = self.get_file_path(hash_)
        try:
            with open(file_path, "rb") as f:
                data = f.read()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Blob not found: {hash_}") from error
        metadata = self._read_metadata(file_path) or {
            "mimeType": "application/octet-stream",
            "sizeBytes": len(data),
            "createdAt": iso_now(),
            "provider": self.provider_id,
            "key": file_path,
        }
        return StoredBlob(data=data, metadata=metadata)

    def exists(self, hash_: str) -> bool:
        try:
            return os.path.exists(self.get_file_path(hash_))
        except (ValueError, OSError):
            return False

    def delete_by_hash(self, hash_: str) -> None:
        try:
            file_path = self.get_file_path(hash_)
        except ValueError:
            # Invalid hashes and path traversal attempts remain a no-op.
            return
        for target in (file_path, self._metadata_path(file_path)):
            try:
                os.unlink(target)
            except FileNotFoundError:
                pass

    def get_url(self, hash_: str, expires_in_seconds: int | None = None) -> str | None:
        # Local filesystem is proxied; return None to signal the proxy route.
        del hash_, expires_in_seconds
        return None

    @staticmethod
    def _build_ref(hash_: str, mime_type: str, size_bytes: Any, provider: Any) -> dict[str, Any]:
        return {
            "uri": _build_uri(hash_),
            "hash": hash_,
            "mimeType": mime_type,
            "sizeBytes": size_bytes,
            "provider": provider,
        }

    def _read_metadata(self, file_path: str) -> Any:
        meta_path = self._metadata_path(self._resolve_path_in_base(file_path))
        try:
            with open(meta_path, encoding="utf-8") as f:
                return json.loads(f.read())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            logger.warning("[BlobFS] Failed to read metadata: %s", error)
            return None
