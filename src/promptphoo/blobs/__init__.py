"""Blob storage. Ports src/blobs/index.ts.

Media bytes go to the provider store (the filesystem by default) and are tracked in the shared
DB's `blob_assets` (one row per hash) and `blob_references` (one row per eval use) tables,
exactly as Node writes them. The API is synchronous: the filesystem and sqlite3 both are, and
the evaluator calls it between awaits. The cloud-only `remoteUpload`/`shareUpload` modules are
not ported (see D48).
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from promptphoo.blobs.constants import BLOB_MAX_SIZE, BLOB_MIN_SIZE, BLOB_SCHEME
from promptphoo.blobs.types import BlobStorageProvider, BlobStoreResult, StoredBlob

__all__ = [
    "BLOB_MAX_SIZE",
    "BLOB_MIN_SIZE",
    "BLOB_SCHEME",
    "BlobStorageProvider",
    "BlobStoreResult",
    "StoredBlob",
    "get_blob_by_hash",
    "get_blob_storage_provider",
    "get_share_authorized_blob",
    "is_blob_allowed_for_share",
    "is_safe_inline_blob_mime_type",
    "record_blob_reference",
    "reset_blob_storage_provider",
    "set_blob_storage_provider",
    "store_blob",
]

logger = logging.getLogger("promptphoo")

# MIME types retained by portable imports and allowed for inline blob responses.
SAFE_INLINE_BLOB_MIME_TYPES = {
    "image/avif",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
    "video/mp4",
    "video/ogg",
    "video/webm",
}
_SAFE_INLINE_AUDIO_MIME_TYPE_REGEX = re.compile(r"^audio/[a-z0-9_+-]+\Z", re.I)


def is_safe_inline_blob_mime_type(mime_type: str) -> bool:
    return mime_type.lower() in SAFE_INLINE_BLOB_MIME_TYPES or bool(
        _SAFE_INLINE_AUDIO_MIME_TYPE_REGEX.match(mime_type)
    )


_default_provider: BlobStorageProvider | None = None


def _create_default_provider() -> BlobStorageProvider:
    # OSS: filesystem-only media storage. Callers can override with set_blob_storage_provider().
    from promptphoo.blobs.filesystem_provider import FilesystemBlobStorageProvider

    return FilesystemBlobStorageProvider()


def get_blob_storage_provider() -> BlobStorageProvider:
    global _default_provider
    if _default_provider is None:
        _default_provider = _create_default_provider()
        logger.debug("[BlobStorage] Initialized provider %s", _default_provider.provider_id)
    return _default_provider


def set_blob_storage_provider(provider: BlobStorageProvider) -> None:
    global _default_provider
    _default_provider = provider
    logger.debug("[BlobStorage] Provider set %s", provider.provider_id)


def reset_blob_storage_provider() -> None:
    global _default_provider
    _default_provider = None


def store_blob(
    data: bytes, mime_type: str, ref_context: dict[str, Any] | None = None
) -> BlobStoreResult:
    """storeBlob: write the bytes, then register the asset (and a reference when the context
    names an eval) in one transaction. A failed transaction keeps the stored bytes: another
    eval may already reference them."""
    from promptphoo import db

    provider = get_blob_storage_provider()
    result = provider.store(data, mime_type)
    ref = result.ref
    ctx = ref_context or {}

    conn = db.get_db()
    with db.transaction(conn):
        # A deduplicated file may be an orphan from a failed transaction. Adopt it with the
        # current caller's MIME type, which imports sanitize.
        conn.execute(
            "INSERT INTO blob_assets (hash, size_bytes, mime_type, provider, created_at) "
            "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT DO NOTHING",
            (
                ref["hash"],
                ref["sizeBytes"],
                mime_type if result.deduplicated else ref["mimeType"],
                ref["provider"],
            ),
        )
        if ctx.get("evalId"):
            _insert_reference(conn, ref["hash"], ctx, conflict_clause=" ON CONFLICT DO NOTHING")
        row = conn.execute(
            "SELECT mime_type FROM blob_assets WHERE hash = ?", (ref["hash"],)
        ).fetchone()
    return BlobStoreResult(ref={**ref, "mimeType": row[0]}, deduplicated=result.deduplicated)


def _insert_reference(conn: Any, hash_: str, ctx: dict[str, Any], conflict_clause: str) -> None:
    conn.execute(
        "INSERT INTO blob_references (id, blob_hash, eval_id, test_idx, prompt_idx, location, "
        f"kind, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP){conflict_clause}",
        (
            str(uuid.uuid4()),
            hash_,
            ctx["evalId"],
            ctx.get("testIdx"),
            ctx.get("promptIdx"),
            ctx.get("location"),
            ctx.get("kind"),
        ),
    )


def get_blob_by_hash(hash_: str) -> StoredBlob:
    from promptphoo import db

    provider = get_blob_storage_provider()
    blob = provider.get_by_hash(hash_)
    row = (
        db.get_db().execute("SELECT mime_type FROM blob_assets WHERE hash = ?", (hash_,)).fetchone()
    )
    # Registered metadata takes precedence over sidecars retained from failed stores.
    if row:
        return StoredBlob(data=blob.data, metadata={**blob.metadata, "mimeType": row[0]})
    return blob


def is_blob_allowed_for_share(hash_: str, eval_id: str) -> bool:
    """Result text may contain copied blob URIs, so only independently classified or imported
    references authorize reading local bytes for a share or a media export."""
    from promptphoo import db

    row = (
        db.get_db()
        .execute(
            "SELECT id FROM blob_references WHERE blob_hash = ? AND eval_id = ? "
            "AND (kind IS NOT NULL OR location = 'import') LIMIT 1",
            (hash_, eval_id),
        )
        .fetchone()
    )
    return bool(row)


def get_share_authorized_blob(hash_: str, local_eval_id: str) -> StoredBlob | None:
    """Read local blob bytes only when the eval has trusted provenance for the hash."""
    if not is_blob_allowed_for_share(hash_, local_eval_id):
        logger.warning(
            "[Share] Skipping blob reference that is not authorized for this eval "
            "(evalId=%s, hash=%s)",
            local_eval_id,
            hash_,
        )
        return None
    return get_blob_by_hash(hash_)


def record_blob_reference(hash_: str, ref_context: dict[str, Any]) -> None:
    """recordBlobReference: link an already registered blob to an eval, or strengthen the
    existing link (a kind, or `import` provenance). Never adopts unregistered bytes."""
    from promptphoo import db

    eval_id = ref_context.get("evalId")
    if not eval_id:
        return

    provider = get_blob_storage_provider()
    try:
        exists = provider.exists(hash_)
    except Exception:
        exists = False
    if not exists:
        logger.debug(
            "[BlobStorage] Attempted to record reference for missing blob "
            "(hash=%s, evalId=%s, location=%s)",
            hash_,
            eval_id,
            ref_context.get("location"),
        )
        return

    conn = db.get_db()
    # A failed store can retain bytes without committing their asset registration.
    # Referencing those bytes must not adopt them or create an invalid foreign key.
    if not conn.execute("SELECT hash FROM blob_assets WHERE hash = ?", (hash_,)).fetchone():
        return

    existing = conn.execute(
        "SELECT id, kind, location FROM blob_references WHERE blob_hash = ? AND eval_id = ? "
        "LIMIT 1",
        (hash_, eval_id),
    ).fetchone()
    if existing:
        ref_id, kind, location = existing
        stronger: dict[str, Any] = {}
        if ref_context.get("kind") and not kind:
            stronger["kind"] = ref_context["kind"]
        if ref_context.get("location") == "import" and location != "import":
            stronger["location"] = "import"
        if stronger:
            assignments = ", ".join(f"{column} = ?" for column in stronger)
            conn.execute(
                f"UPDATE blob_references SET {assignments} WHERE id = ?",
                (*stronger.values(), ref_id),
            )
        return

    _insert_reference(conn, hash_, ref_context, conflict_clause="")
