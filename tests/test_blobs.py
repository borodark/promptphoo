"""Blob storage. Ports test/blobs/{blobRefs,filesystemProvider,index,extractor}.test.ts, the
blob cases of test/util/jsonExport.test.ts, test/evaluator/{assertions,execution}.test.ts and
test/commands/import.test.ts, and the EvalResult persistence call sites."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pytest

from promptphoo import blobs, db, paths
from promptphoo.blobs import extractor
from promptphoo.blobs.blob_refs import (
    collect_blob_hashes,
    extract_blob_hashes_from_string,
    extract_blob_hashes_from_value,
    normalize_blob_hash,
)
from promptphoo.blobs.extractor import (
    extract_and_store_binary_data,
    is_blob_storage_enabled,
    normalize_audio_mime_type,
)
from promptphoo.blobs.filesystem_provider import FilesystemBlobStorageProvider
from promptphoo.blobs.types import BlobStoreResult, StoredBlob
from promptphoo.providers.base import ApiProvider

HASH_A = "a" * 64
HASH_B = "b" * 64


def uri(hash_: str) -> str:
    return f"promptfoo://blob/{hash_}"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(cfg))
    monkeypatch.delenv("IS_TESTING", raising=False)
    monkeypatch.delenv("PROMPTFOO_INLINE_MEDIA", raising=False)
    paths.set_config_directory_path(None)
    db.close_db()
    db.run_db_migrations()
    yield cfg
    db.close_db()


def _insert_eval(eval_id: str) -> None:
    db.get_db().execute(
        "INSERT INTO evals (id, created_at, config, results) VALUES (?, 0, '{}', '{}')", (eval_id,)
    )


def _rows(sql: str, *args: Any) -> list[tuple]:
    return db.get_db().execute(sql, args).fetchall()


# --- blobRefs ---------------------------------------------------------------------------------


def test_normalize_blob_hash():
    assert normalize_blob_hash("ABCDEF") == "abcdef"


def test_extract_blob_hashes_from_string():
    assert extract_blob_hashes_from_string("just some text") == []
    value = f"before {uri(HASH_A.upper())} middle {uri(HASH_B)} after"
    assert extract_blob_hashes_from_string(value) == [HASH_A, HASH_B]
    assert extract_blob_hashes_from_string(f"{'x' * 200} {uri(HASH_A)}", 100) == []
    assert extract_blob_hashes_from_string(f"{uri(HASH_A)}{'y' * 200}", 100) == [HASH_A]


def test_extract_blob_hashes_from_value():
    assert extract_blob_hashes_from_value(uri(HASH_A)) == [HASH_A]
    assert extract_blob_hashes_from_value({"hash": HASH_A.upper()}) == [HASH_A]
    assert extract_blob_hashes_from_value({"uri": uri(HASH_B)}) == [HASH_B]
    assert extract_blob_hashes_from_value(42) == []
    assert extract_blob_hashes_from_value(None) == []
    assert extract_blob_hashes_from_value({"hash": "not-a-hash"}) == []


def test_collect_blob_hashes():
    value = {
        "results": [{"response": {"output": uri(HASH_A)}}, {"response": {"output": uri(HASH_A)}}],
        "traces": [{"metadata": {"attachment": uri(HASH_B)}}],
    }
    assert list(collect_blob_hashes(value)) == [HASH_A, HASH_B]
    node: dict[str, Any] = {"output": uri(HASH_A)}
    node["self"] = node
    assert list(collect_blob_hashes(node)) == [HASH_A]
    nested = {"a": {"b": uri(HASH_A)}}
    assert list(collect_blob_hashes(nested, max_depth=1)) == []
    assert list(collect_blob_hashes(nested, max_depth=2)) == [HASH_A]
    deep: Any = uri(HASH_A)
    for _ in range(50_000):
        deep = {"child": deep}
    assert list(collect_blob_hashes(deep, max_depth=64)) == []
    buried = {"output": f"{'x' * 200} {uri(HASH_A)}"}
    prefixed = {"output": f"{uri(HASH_B)}{'y' * 200}"}
    assert list(collect_blob_hashes(buried, max_string_length=100)) == []
    assert list(collect_blob_hashes(prefixed, max_string_length=100)) == [HASH_B]


# --- FilesystemBlobStorageProvider -------------------------------------------------------------


def test_filesystem_store_and_retrieve(tmp_path):
    provider = FilesystemBlobStorageProvider(str(tmp_path))
    result = provider.store(b"blob-data", "application/octet-stream")
    assert provider.get_by_hash(result.ref["hash"]).data == b"blob-data"
    h = sha(b"blob-data")
    assert result.ref == {
        "uri": uri(h),
        "hash": h,
        "mimeType": "application/octet-stream",
        "sizeBytes": 9,
        "provider": "filesystem",
    }
    path = tmp_path / h[:2] / h[2:4] / h
    meta_text = (tmp_path / h[:2] / h[2:4] / f"{h}.meta.json").read_text()
    meta = json.loads(meta_text)
    assert list(meta) == ["mimeType", "sizeBytes", "createdAt", "provider", "key"]
    assert meta["key"] == str(path) and meta_text.startswith('{\n  "mimeType"')
    # No staging directories are left behind.
    assert sorted(p.name for p in path.parent.iterdir()) == [h, f"{h}.meta.json"]


def test_filesystem_deduplicates(tmp_path):
    provider = FilesystemBlobStorageProvider(str(tmp_path))
    first = provider.store(b"same", "application/octet-stream")
    second = provider.store(b"same", "image/png")
    assert first.ref["hash"] == second.ref["hash"]
    assert second.deduplicated is True
    assert second.ref["mimeType"] == "application/octet-stream"


def test_filesystem_republishes_when_metadata_missing(tmp_path):
    provider = FilesystemBlobStorageProvider(str(tmp_path))
    ref = provider.store(b"data", "text/plain").ref
    os.unlink(provider.get_file_path(ref["hash"]) + ".meta.json")
    again = provider.store(b"data", "image/png")
    assert again.deduplicated is False
    assert provider.get_by_hash(ref["hash"]).metadata["mimeType"] == "image/png"


def test_filesystem_rejects_invalid_hashes(tmp_path):
    provider = FilesystemBlobStorageProvider(str(tmp_path))
    assert provider.exists("../../etc/passwd") is False
    with pytest.raises(ValueError, match="(?i)invalid blob hash"):
        provider.get_by_hash("../../etc/passwd")
    with pytest.raises(FileNotFoundError, match=f"Blob not found: {HASH_A}"):
        provider.get_by_hash(HASH_A)


@pytest.mark.parametrize("target", ["data", "metadata"])
def test_filesystem_reports_real_deletion_error(tmp_path, target):
    provider = FilesystemBlobStorageProvider(str(tmp_path))
    ref = provider.store(b"data", "text/plain").ref
    file_path = Path(provider.get_file_path(ref["hash"]))
    meta_path = Path(f"{file_path}.meta.json")
    failing = file_path if target == "data" else meta_path
    failing.unlink()
    failing.mkdir()
    with pytest.raises(OSError):
        provider.delete_by_hash(ref["hash"])
    assert failing.is_dir()
    if target == "data":
        assert json.loads(meta_path.read_text())["mimeType"] == "text/plain"
    else:
        assert not file_path.exists()


@pytest.mark.parametrize("missing", ["none", "data", "metadata"])
def test_filesystem_deletes_remaining_files(tmp_path, missing):
    provider = FilesystemBlobStorageProvider(str(tmp_path))
    ref = provider.store(b"data", "text/plain").ref
    file_path = Path(provider.get_file_path(ref["hash"]))
    meta_path = Path(f"{file_path}.meta.json")
    if missing != "none":
        (file_path if missing == "data" else meta_path).unlink()
    provider.delete_by_hash(ref["hash"])
    assert not file_path.exists() and not meta_path.exists()
    provider.delete_by_hash(ref["hash"])


def test_filesystem_invalid_hash_delete_is_noop(tmp_path):
    provider = FilesystemBlobStorageProvider(str(tmp_path / "blobs"))
    protected = tmp_path / "protected"
    protected.write_text("retained")
    for h in ["../protected", str(protected), "", "not-a-hash", "a" * 63, "g" * 64]:
        provider.delete_by_hash(h)
    assert protected.read_text() == "retained"


def test_default_provider_uses_config_dir(config_dir):
    provider = blobs.get_blob_storage_provider()
    assert isinstance(provider, FilesystemBlobStorageProvider)
    assert provider.base_path == str(config_dir / "blobs")
    assert (config_dir / "blobs").is_dir()
    assert blobs.get_blob_storage_provider() is provider


# --- index: share authorization and provenance -------------------------------------------------


class StubProvider:
    provider_id = "test-stub"

    def __init__(self, data: bytes = b"trusted-bytes") -> None:
        self.data = data

    def store(self, data: bytes, mime_type: str) -> BlobStoreResult:
        raise NotImplementedError

    def get_by_hash(self, hash_: str) -> StoredBlob:
        meta = {"createdAt": "2026-06-08T00:00:00.000Z", "key": hash_, "mimeType": "image/png",
                "provider": "test-stub", "sizeBytes": len(self.data)}  # fmt: skip
        return StoredBlob(data=self.data, metadata=meta)

    def exists(self, hash_: str) -> bool:
        return True

    def delete_by_hash(self, hash_: str) -> None:
        return None

    def get_url(self, hash_: str, expires_in_seconds: int | None = None) -> str | None:
        return None


@pytest.fixture
def share_rows(config_dir):
    eval_id, other = f"eval-{uuid.uuid4()}", f"eval-{uuid.uuid4()}"
    trusted, unclassified, imported, other_hash = "a" * 64, "b" * 64, "c" * 64, "d" * 64
    _insert_eval(eval_id)
    _insert_eval(other)
    conn = db.get_db()
    for h in (trusted, unclassified, imported, other_hash):
        conn.execute(
            "INSERT INTO blob_assets (hash, size_bytes, mime_type, provider) VALUES (?, 1, "
            "'image/png', 'filesystem')",
            (h,),
        )
    for h, e, kind, location in [
        (trusted, eval_id, "image", "response.output"),
        (unclassified, eval_id, None, "response.output"),
        (imported, eval_id, None, "import"),
        (other_hash, other, "image", "response.output"),
    ]:
        conn.execute(
            "INSERT INTO blob_references (id, blob_hash, eval_id, kind, location) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), h, e, kind, location),
        )
    return eval_id, trusted, unclassified, imported, other_hash


def test_share_authorization(share_rows):
    eval_id, trusted, unclassified, imported, other_hash = share_rows
    assert blobs.is_blob_allowed_for_share(trusted, eval_id) is True
    assert blobs.is_blob_allowed_for_share(imported, eval_id) is True
    assert blobs.is_blob_allowed_for_share(unclassified, eval_id) is False
    assert blobs.is_blob_allowed_for_share(other_hash, eval_id) is False
    db.get_db().execute("DELETE FROM blob_references WHERE blob_hash = ?", (trusted,))
    assert blobs.is_blob_allowed_for_share(trusted, eval_id) is False


def test_share_authorized_blob_reads_only_trusted(share_rows):
    eval_id, trusted, unclassified, _imported, other_hash = share_rows
    blobs.set_blob_storage_provider(StubProvider())
    assert blobs.get_share_authorized_blob(unclassified, eval_id) is None
    assert blobs.get_share_authorized_blob(other_hash, eval_id) is None
    blob = blobs.get_share_authorized_blob(trusted, eval_id)
    assert blob is not None and blob.data == b"trusted-bytes"


@pytest.fixture
def provenance(config_dir):
    eval_id, h = f"eval-{uuid.uuid4()}", "e" * 64
    blobs.set_blob_storage_provider(StubProvider())
    _insert_eval(eval_id)
    db.get_db().execute(
        "INSERT INTO blob_assets (hash, size_bytes, mime_type, provider) VALUES (?, 1, "
        "'image/png', 'filesystem')",
        (h,),
    )

    def refs() -> list[tuple]:
        return _rows("SELECT kind, location FROM blob_references WHERE eval_id = ?", eval_id)

    return eval_id, h, refs


def test_record_reference_upgrades_kind(provenance):
    eval_id, h, refs = provenance
    blobs.record_blob_reference(h, {"evalId": eval_id, "location": "response.output"})
    assert blobs.is_blob_allowed_for_share(h, eval_id) is False
    blobs.record_blob_reference(
        h, {"evalId": eval_id, "kind": "image", "location": "response.images[0].data"}
    )
    assert blobs.is_blob_allowed_for_share(h, eval_id) is True
    assert refs() == [("image", "response.output")]


def test_record_reference_without_kind_does_not_authorize(provenance):
    eval_id, h, refs = provenance
    blobs.record_blob_reference(h, {"evalId": eval_id, "location": "response.output"})
    blobs.record_blob_reference(h, {"evalId": eval_id, "location": "response.metadata"})
    assert blobs.is_blob_allowed_for_share(h, eval_id) is False
    assert refs() == [(None, "response.output")]


def test_record_reference_never_downgrades(provenance):
    eval_id, h, refs = provenance
    blobs.record_blob_reference(h, {"evalId": eval_id, "kind": "image", "location": "x"})
    blobs.record_blob_reference(h, {"evalId": eval_id, "location": "response.metadata"})
    assert refs() == [("image", "x")]


def test_record_reference_upgrades_to_import(provenance):
    eval_id, h, refs = provenance
    blobs.record_blob_reference(h, {"evalId": eval_id, "location": "response.output"})
    blobs.record_blob_reference(h, {"evalId": eval_id, "location": "import"})
    assert blobs.is_blob_allowed_for_share(h, eval_id) is True
    assert refs() == [(None, "import")]


def test_record_reference_needs_eval_and_registered_asset(provenance):
    eval_id, _h, refs = provenance
    blobs.record_blob_reference("f" * 64, {"evalId": eval_id, "location": "import"})
    blobs.record_blob_reference("e" * 64, {"location": "import"})
    assert refs() == []


# --- index: storeBlob persistence -------------------------------------------------------------

DATA = b"shared blob rollback fixture"
H = sha(DATA)


@pytest.fixture
def store_env(config_dir):
    first, second, missing = (f"eval-{uuid.uuid4()}", f"eval-{uuid.uuid4()}",
                              f"missing-{uuid.uuid4()}")  # fmt: skip
    _insert_eval(first)
    _insert_eval(second)
    provider = FilesystemBlobStorageProvider(str(config_dir / "blobs"))
    blobs.set_blob_storage_provider(provider)
    file_path = config_dir / "blobs" / H[:2] / H[2:4] / H

    def files() -> tuple[bytes, str]:
        return file_path.read_bytes(), Path(f"{file_path}.meta.json").read_text()

    def rows() -> dict[str, list[tuple]]:
        return {
            "assets": _rows("SELECT hash, mime_type FROM blob_assets WHERE hash = ?", H),
            "references": _rows(
                "SELECT id, eval_id, location FROM blob_references WHERE blob_hash = ?", H
            ),
        }

    return first, second, missing, provider, files, rows


def test_store_failure_keeps_shared_files(store_env):
    first, _second, missing, provider, files, rows = store_env
    blobs.store_blob(DATA, "application/octet-stream", {"evalId": first, "location": "import"})
    before_files, before_rows = files(), rows()
    with pytest.raises(sqlite3.IntegrityError):
        blobs.store_blob(DATA, "application/octet-stream", {"evalId": missing})
    assert rows() == before_rows
    assert provider.exists(H) and files() == before_files
    blob = blobs.get_share_authorized_blob(H, first)
    assert blob is not None and blob.data == DATA
    assert blobs.is_blob_allowed_for_share(H, missing) is False


def test_store_failure_retains_unreferenced_files_for_later_adoption(store_env):
    first, _second, missing, provider, files, rows = store_env
    with pytest.raises(sqlite3.IntegrityError):
        blobs.store_blob(DATA, "application/octet-stream", {"evalId": missing})
    assert rows() == {"assets": [], "references": []}
    assert provider.exists(H)
    before = files()
    assert blobs.get_share_authorized_blob(H, first) is None
    blobs.record_blob_reference(H, {"evalId": first, "location": "import"})
    assert rows() == {"assets": [], "references": []}
    ctx = {"evalId": first, "location": "import"}
    adopted = blobs.store_blob(DATA, "application/octet-stream", ctx)
    assert adopted.deduplicated is True
    assert files() == before
    assert len(rows()["assets"]) == 1 and len(rows()["references"]) == 1


def test_store_cascade_keeps_other_eval_authorized(store_env):
    first, second, _missing, _provider, files, rows = store_env
    a = blobs.store_blob(DATA, "application/octet-stream", {"evalId": first, "location": "import"})
    b = blobs.store_blob(DATA, "application/octet-stream", {"evalId": second, "location": "import"})
    assert (a.deduplicated, b.deduplicated) == (False, True)
    assert len(rows()["assets"]) == 1 and len(rows()["references"]) == 2
    db.get_db().execute("DELETE FROM evals WHERE id = ?", (first,))
    assert len(rows()["references"]) == 1
    assert blobs.get_share_authorized_blob(H, first) is None
    blob = blobs.get_share_authorized_blob(H, second)
    assert blob is not None and blob.data == DATA


def test_store_keeps_registered_mime_type(store_env):
    first, second, _missing, _provider, files, rows = store_env
    blobs.store_blob(DATA, "image/png", {"evalId": first, "location": "import"})
    before = files()
    stored = blobs.store_blob(DATA, "audio/wav", {"evalId": second, "location": "import"})
    assert stored.deduplicated is True and stored.ref["mimeType"] == "image/png"
    assert rows()["assets"][0][1] == "image/png"
    blob = blobs.get_share_authorized_blob(H, second)
    assert blob is not None and blob.metadata["mimeType"] == "image/png"
    assert files() == before


def test_store_uses_provider_metadata_for_new_bytes(store_env, monkeypatch):
    first, _second, _missing, provider, _files, rows = store_env
    original = provider.store
    monkeypatch.setattr(provider, "store", lambda data, _mime: original(data, "image/png"))
    stored = blobs.store_blob(DATA, "application/octet-stream", {"evalId": first,
                                                                 "location": "import"})  # fmt: skip
    assert stored.deduplicated is False and stored.ref["mimeType"] == "image/png"
    assert rows()["assets"][0][1] == "image/png"


def test_store_asset_first_import_without_unclassified_access(store_env):
    first, second, missing, _provider, files, rows = store_env
    stored = blobs.store_blob(DATA, "application/octet-stream")
    assert stored.ref["hash"] == H
    assert len(rows()["assets"]) == 1 and rows()["references"] == []
    assert blobs.get_share_authorized_blob(H, first) is None
    with pytest.raises(sqlite3.IntegrityError):
        blobs.record_blob_reference(H, {"evalId": missing, "location": "import"})
    assert rows()["references"] == []
    blobs.record_blob_reference(H, {"evalId": first, "location": "response.output"})
    assert blobs.get_share_authorized_blob(H, first) is None
    blobs.record_blob_reference(H, {"evalId": first, "location": "import"})
    assert len(rows()["references"]) == 1
    blob = blobs.get_share_authorized_blob(H, first)
    assert blob is not None and blob.data == DATA
    assert blobs.get_share_authorized_blob(H, second) is None


def test_is_safe_inline_blob_mime_type():
    assert blobs.is_safe_inline_blob_mime_type("IMAGE/PNG")
    assert blobs.is_safe_inline_blob_mime_type("audio/x-wav")
    assert not blobs.is_safe_inline_blob_mime_type("image/svg+xml")
    assert not blobs.is_safe_inline_blob_mime_type("audio/wav.html")
    assert not blobs.is_safe_inline_blob_mime_type("text/html")


# --- extractor: normalizeAudioMimeType ---------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        (None, "audio/wav"), ("", "audio/wav"), ("   ", "audio/wav"),
        ("audio/mpeg", "audio/mpeg"), ("audio/x-custom", "audio/x-custom"),
        ("audio/vnd.company.format", "audio/wav"), ("audio/wav.html", "audio/wav"),
        ("mp3", "audio/mpeg"), ("m4a", "audio/mp4"), ("WAV", "audio/wav"), ("Mp3", "audio/mpeg"),
        ("  wav  ", "audio/wav"), ("\tmp3\n", "audio/mpeg"), ("opus", "audio/opus"),
        ("codec_v2", "audio/codec_v2"), ("x-wav-hd", "audio/x-wav-hd"),
        ("format-1.0", "audio/wav"), ("wav.html", "audio/wav"), ("../../etc/passwd", "audio/wav"),
        ("path/to/file", "audio/wav"), ("audio\nwav", "audio/wav"),
        ("wav\r\nContent-Type: text/html", "audio/wav"), ("audio;wav", "audio/wav"),
        ("audio:wav", "audio/wav"), ("audio wav", "audio/wav"), ("audio\\wav", "audio/wav"),
        ("audio/wav/extra", "audio/wav"),
    ],
)  # fmt: skip
def test_normalize_audio_mime_type(fmt, expected):
    assert normalize_audio_mime_type(fmt) == expected


def test_is_blob_storage_enabled(monkeypatch):
    monkeypatch.delenv("PROMPTFOO_INLINE_MEDIA", raising=False)
    assert is_blob_storage_enabled() is True
    monkeypatch.setenv("PROMPTFOO_INLINE_MEDIA", "true")
    assert is_blob_storage_enabled() is False
    monkeypatch.setenv("PROMPTFOO_INLINE_MEDIA", "false")
    assert is_blob_storage_enabled() is True


def test_buffer_from_base64_matches_node():
    # Expected values from Node 22's Buffer.from(s, 'base64').
    cases = {"SGVsbG8=": "48656c6c6f", "SGVsbG8": "48656c6c6f", "SGV sbG8": "48656c6c6f",
             "SGVs=bG8": "48656c", "S": "", "SG": "48", "-_-_": "fbffbf",
             "@@SGVs": "48656c", "SGVsbG8==QQ": "48656c6c6f"}  # fmt: skip
    for text, expected in cases.items():
        assert extractor.buffer_from_base64(text).hex() == expected


# --- extractor: extraction with a mocked store ------------------------------------------------


@pytest.fixture
def mock_store(monkeypatch):
    calls: list[tuple[bytes, str, dict]] = []
    recorded: list[tuple[str, dict]] = []

    def store_blob(data, mime_type, ctx=None):
        calls.append((data, mime_type, ctx or {}))
        return BlobStoreResult(
            ref={"uri": "promptfoo://blob/abc123def456", "hash": "abc123def456"},
            deduplicated=False,
        )

    monkeypatch.setattr(blobs, "store_blob", store_blob)
    monkeypatch.setattr(blobs, "record_blob_reference", lambda h, c: recorded.append((h, c)))
    monkeypatch.delenv("PROMPTFOO_INLINE_MEDIA", raising=False)
    return calls, recorded


LARGE = base64.b64encode(bytes(2000)).decode()
DATA_URI = f"data:image/png;base64,{LARGE}"
SVG = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="4"/></svg>'


def test_extract_handles_empty_responses(mock_store):
    assert extract_and_store_binary_data(None) is None
    response = {"output": "test"}
    assert extract_and_store_binary_data(response) is response


def test_extract_audio_and_turns(mock_store):
    calls, _ = mock_store
    result = extract_and_store_binary_data(
        {"output": "t", "audio": {"data": LARGE, "format": "wav", "transcript": "hi"}}
    )
    assert result is not None
    assert result["audio"] == {"format": "wav", "transcript": "hi",
                               "blobRef": {"uri": "promptfoo://blob/abc123def456",
                                           "hash": "abc123def456"}}  # fmt: skip
    small = {"output": "t", "turns": [{"audio": {"data": "SGVsbG8gV29ybGQ=", "format": "mp3"}}]}
    assert extract_and_store_binary_data(small) is small
    assert len(calls) == 1 and calls[0][1] == "audio/wav"
    assert calls[0][2] == {"location": "response.audio.data", "kind": "audio"}


def test_extract_output_data_url(mock_store):
    calls, _ = mock_store
    result = extract_and_store_binary_data({"output": DATA_URI})
    assert result == {"output": "promptfoo://blob/abc123def456"}
    assert len(calls) == 1 and calls[0][0] == bytes(2000)


def test_extract_images(mock_store):
    calls, _ = mock_store
    result = extract_and_store_binary_data(
        {"output": "text output", "images": [{"data": DATA_URI, "mimeType": "image/png"}]}
    )
    assert result is not None
    image = result["images"][0]
    assert "data" not in image and image["blobRef"]["uri"].startswith("promptfoo://blob/")


def test_extract_reuses_blob_for_output_and_images(mock_store):
    calls, _ = mock_store
    result = extract_and_store_binary_data(
        {"output": DATA_URI, "images": [{"data": DATA_URI, "mimeType": "image/png"}]}
    )
    assert result is not None
    assert result["output"] == result["images"][0]["blobRef"]["uri"]
    assert len(calls) == 1
    assert calls[0][1:] == ("image/png", {"location": "response.images[0].data", "kind": "image"})


def test_extract_reuses_blob_for_identical_images(mock_store):
    calls, _ = mock_store
    image = {"data": DATA_URI, "mimeType": "image/png"}
    result = extract_and_store_binary_data({"output": "o", "images": [image, dict(image)]})
    assert result is not None
    assert result["images"][0]["blobRef"] == result["images"][1]["blobRef"]
    assert len(calls) == 1


def test_extract_keeps_existing_blob_refs(mock_store):
    ref = {"uri": "promptfoo://blob/existing", "hash": "existing", "mimeType": "image/jpeg",
           "sizeBytes": 100, "provider": "local"}  # fmt: skip
    response = {"output": "text output", "images": [{"blobRef": ref}]}
    assert extract_and_store_binary_data(response) == response
    mixed = extract_and_store_binary_data(
        {"output": "o", "images": [{"data": DATA_URI, "mimeType": "image/png"}, {"blobRef": ref}]}
    )
    assert mixed is not None
    assert "data" not in mixed["images"][0] and mixed["images"][1]["blobRef"]["hash"] == "existing"


def test_extract_records_references_in_output_text(mock_store):
    _calls, recorded = mock_store
    output = f"First {uri(HASH_A)} repeated {uri(HASH_A)} second {uri(HASH_B)}"
    extract_and_store_binary_data(
        {"output": output}, {"evalId": "eval-mixed-output", "testIdx": 1, "promptIdx": 2}
    )
    ctx = {"evalId": "eval-mixed-output", "testIdx": 1, "promptIdx": 2,
           "location": "response.output"}  # fmt: skip
    assert recorded == [(HASH_A, ctx), (HASH_B, ctx)]


def test_extract_metadata_audio(mock_store):
    calls, _ = mock_store
    audio = {"data": LARGE, "format": "wav", "transcript": "hello"}
    result = extract_and_store_binary_data(
        {"output": "test", "audio": dict(audio), "metadata": {"audio": dict(audio)}}
    )
    assert result is not None
    assert result["metadata"]["audio"]["blobRef"] == result["audio"]["blobRef"]
    assert "data" not in result["metadata"]["audio"]
    assert len(calls) == 1 and calls[0][2]["location"] == "response.audio.data"
    calls.clear()
    only = extract_and_store_binary_data({"output": "t", "metadata": {"audio": dict(audio)}})
    assert only is not None and "blobRef" in only["metadata"]["audio"]
    assert calls[0][2] == {"location": "response.metadata.audio.data", "kind": "audio"}


def test_extract_turns_and_metadata_share_one_blob(mock_store):
    calls, _ = mock_store
    audio = {"data": LARGE, "format": "wav", "transcript": "hi"}
    result = extract_and_store_binary_data(
        {"output": "t", "turns": [{"role": "assistant", "audio": dict(audio)}],
         "metadata": {"audio": dict(audio)}}
    )  # fmt: skip
    assert result is not None
    assert result["metadata"]["audio"]["blobRef"] == result["turns"][0]["audio"]["blobRef"]
    assert len(calls) == 1 and calls[0][2]["location"] == "response.turns[0].audio.data"


def test_extract_metadata_data_url_shares_image_blob(mock_store):
    calls, _ = mock_store
    result = extract_and_store_binary_data(
        {"output": "text", "images": [{"data": DATA_URI, "mimeType": "image/png"}],
         "metadata": {"previewImage": DATA_URI}}
    )  # fmt: skip
    assert result is not None
    assert result["metadata"]["previewImage"] == result["images"][0]["blobRef"]["uri"]
    assert len(calls) == 1 and calls[0][2]["location"] == "response.images[0].data"


def test_extract_raw_svg_preview(mock_store):
    calls, _ = mock_store
    result = extract_and_store_binary_data({"output": SVG})
    assert result == {"output": SVG, "metadata": {"blobUris": ["promptfoo://blob/abc123def456"]}}
    assert calls == [(SVG.encode(), "image/svg+xml", {"location": "response.output",
                                                      "kind": "image"})]  # fmt: skip


def test_extract_raw_svg_already_indexed(mock_store):
    calls, _ = mock_store
    response = {"output": SVG, "metadata": {"blobUris": [uri(sha(SVG.encode()))]}}
    assert extract_and_store_binary_data(response) is response
    assert calls == []


@pytest.mark.parametrize(
    "output",
    ["<svg><circle/></svg>\n\n<svg><rect/></svg>", "<div><svg/></div>", "<svg>", "svg"],
)
def test_extract_ignores_non_svg_documents(mock_store, output):
    calls, _ = mock_store
    response = {"output": output}
    assert extract_and_store_binary_data(response) is response
    assert calls == []


def test_extract_svg_with_undeclared_prefix(mock_store):
    calls, _ = mock_store
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><use xlink:href="#a"/></svg>'
    extract_and_store_binary_data({"output": svg})
    assert len(calls) == 1


def test_extract_b64_json(mock_store):
    calls, _ = mock_store
    output = json.dumps({"data": [{"b64_json": LARGE}], "created": 1})
    result = extract_and_store_binary_data({"output": output, "format": "json", "isBase64": True})
    assert result == {
        "output": "promptfoo://blob/abc123def456",
        "format": "json",
        "isBase64": True,
        "metadata": {"blobUris": ["promptfoo://blob/abc123def456"], "originalFormat": "json"},
    }
    assert calls[0][1:] == ("image/png", {"location": "response.output.data[].b64_json",
                                          "kind": "image"})  # fmt: skip


def test_extract_b64_json_swallows_storage_errors(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("FOREIGN KEY constraint failed")

    monkeypatch.setattr(blobs, "store_blob", boom)
    monkeypatch.delenv("PROMPTFOO_INLINE_MEDIA", raising=False)
    response = {"output": json.dumps({"data": [{"b64_json": LARGE}]})}
    assert extract_and_store_binary_data(response) is response
    with pytest.raises(RuntimeError):
        extract_and_store_binary_data({"output": DATA_URI})


def test_extract_inline_media_skips_store(mock_store, monkeypatch):
    calls, recorded = mock_store
    monkeypatch.setenv("PROMPTFOO_INLINE_MEDIA", "true")
    response = {"output": DATA_URI, "metadata": {"x": uri(HASH_A)}}
    assert extract_and_store_binary_data(response, {"evalId": "e"}) is response
    assert calls == []
    assert recorded == [(HASH_A, {"evalId": "e", "location": "response.metadata.x"})]


def test_extract_data_url_with_newline_is_left_inline(mock_store):
    calls, _ = mock_store
    response = {"output": f"data:image/png;base64,{LARGE[:1000]}\n{LARGE[1000:]}"}
    assert extract_and_store_binary_data(response) is response
    assert calls == []


# --- evaluator and EvalResult call sites -------------------------------------------------------


class MediaProvider(ApiProvider):
    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__({"id": "media-provider"})
        self.response = response

    async def call_api(self, prompt, context=None, options=None) -> Any:
        return json.loads(json.dumps(self.response))


class Judge(ApiProvider):
    def __init__(self) -> None:
        super().__init__({"id": "judge"})
        self.prompts: list[str] = []

    def get_audio_input_format(self) -> str:
        return "openai"

    async def call_api(self, prompt, context=None, options=None) -> Any:
        self.prompts.append(prompt)
        return {"output": json.dumps({"pass": True, "score": 1, "reason": "ok"})}


def _run(provider: Any, tests: list[dict[str, Any]]):
    from promptphoo.evaluator.run import evaluate
    from promptphoo.models.eval import Eval

    suite = {"providers": [provider], "prompts": [{"raw": "Generate", "label": "p"}],
             "tests": tests}  # fmt: skip
    record = Eval.create({}, suite["prompts"])
    asyncio.run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    return record


def test_evaluator_keeps_raw_svg_for_judges_and_indexes_preview(config_dir):
    judge = Judge()
    record = _run(
        MediaProvider({"output": SVG}),
        [{"assert": [{"type": "llm-rubric", "value": "Output should be SVG", "provider": judge}]}],
    )
    assert len(judge.prompts) == 1
    judge_input = json.loads(judge.prompts[0])[-1]["content"]
    assert SVG in judge_input and "promptfoo://blob/" not in judge_input
    (row,) = record.load_results()
    assert row.response["output"] == SVG
    assert row.response["metadata"]["blobUris"] == [uri(sha(SVG.encode()))]
    assert (config_dir / "blobs" / sha(SVG.encode())[:2]).is_dir()


def test_evaluator_grades_inline_audio_and_persists_blob_ref(config_dir):
    judge = Judge()
    wav = base64.b64encode(b"\x01" * 2048).decode()
    record = _run(
        MediaProvider({"output": "Hello.", "audio": {"data": wav, "format": "wav"}}),
        [{"assert": [{"type": "llm-rubric", "value": "Calm.", "provider": judge}]}],
    )
    assert wav in judge.prompts[0]
    (row,) = record.load_results()
    assert row.success is True
    assert "data" not in row.response["audio"]
    ref = row.response["audio"]["blobRef"]
    assert ref["hash"] == sha(b"\x01" * 2048) and ref["mimeType"] == "audio/wav"
    refs = _rows("SELECT eval_id, test_idx, prompt_idx, location, kind FROM blob_references")
    assert refs == [(record.id, 0, 0, "response.audio.data", "audio")]


def test_eval_result_creation_externalizes_media(config_dir):
    from promptphoo.models.eval import Eval, EvalResult

    record = Eval.create({}, [{"raw": "p", "label": "p"}])
    png = b"\x89PNG" + bytes(2000)
    result = {
        "promptIdx": 0, "testIdx": 3, "testCase": {}, "prompt": {"raw": "p", "label": "p"},
        "success": True, "score": 1, "provider": {"id": "x"},
        "response": {"output": "data:image/png;base64," + base64.b64encode(png).decode()},
    }  # fmt: skip
    saved = EvalResult.create_from_evaluate_result(record.id, result)
    assert saved.response == {"output": uri(sha(png))}
    assert _rows("SELECT test_idx, prompt_idx, location, kind FROM blob_references") == [
        (3, 0, "response.output", "image")
    ]


# --- export --include-media ------------------------------------------------------------------


class _FakeEval:
    def __init__(self, results: list[dict], env: dict | None = None) -> None:
        self.id = "eval-fake"
        self.config = {"env": env} if env else {}
        self.created_at = 0
        self.author = None
        self.vars: list[str] = []
        self.runtime_options = None
        self._results = results

    def to_evaluate_summary(self) -> dict[str, Any]:
        return {"version": 3, "timestamp": "2025-01-01T00:00:00.000Z", "prompts": [],
                "results": self._results, "stats": {"successes": 1, "failures": 0}}  # fmt: skip


def _authorize(monkeypatch, data: bytes) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def fake(hash_, eval_id):
        calls.append((hash_, eval_id))
        return StoredBlob(data=data, metadata={"mimeType": "image/png"})

    monkeypatch.setattr(blobs, "get_share_authorized_blob", fake)
    return calls


def test_export_embeds_blobs_only_with_include_media(tmp_path, monkeypatch):
    from promptphoo.output.writer import write_output

    data = b"portable image"
    calls = _authorize(monkeypatch, data)
    record = _FakeEval([{"response": {"output": uri(HASH_A)}}])
    out = tmp_path / "out.json"
    write_output(str(out), record, None)
    assert "blobAssets" not in json.loads(out.read_text())
    write_output(str(out), record, None, include_media=True)
    assert calls == [(HASH_A, "eval-fake")]
    assert json.loads(out.read_text())["blobAssets"] == [
        {"hash": HASH_A, "mimeType": "image/png", "sizeBytes": len(data),
         "data": base64.b64encode(data).decode()}
    ]  # fmt: skip
    yaml_out = tmp_path / "out.yaml"
    write_output(str(yaml_out), record, None, include_media=True)
    assert "blobAssets:" in yaml_out.read_text()


def test_export_media_scan_includes_traces(monkeypatch):
    from promptphoo.output.writer import export_blob_assets

    _authorize(monkeypatch, b"trace image")
    traces = [{"traceId": "t", "metadata": {"attachment": uri(HASH_B)}, "spans": []}]
    assets = export_blob_assets("eval-fake", {"results": []}, False, traces)
    assert [a["hash"] for a in assets] == [HASH_B]


def test_export_skips_stripped_response_blob_uris(tmp_path, monkeypatch):
    from promptphoo.output.writer import create_output_data

    calls = _authorize(monkeypatch, b"x")
    monkeypatch.setenv("PROMPTFOO_STRIP_RESPONSE_OUTPUT", "true")
    record = _FakeEval(
        [{"response": {"output": "[output stripped]", "metadata": {"blobUris": [uri("c" * 64)]}}}]
    )
    data = create_output_data(record, None, include_media=True)
    assert "blobAssets" not in data and calls == []


def test_export_skips_missing_and_unauthorized_blobs(config_dir, caplog):
    from promptphoo.output.writer import export_blob_assets

    eval_id = f"eval-{uuid.uuid4()}"
    _insert_eval(eval_id)
    kept = blobs.store_blob(b"k" * 10, "image/png", {"evalId": eval_id, "kind": "image"}).ref
    gone = blobs.store_blob(b"y" * 10, "image/png", {"evalId": eval_id, "kind": "image"}).ref
    unclassified = blobs.store_blob(b"z" * 10, "image/png", {"evalId": eval_id}).ref
    text = " ".join([kept["uri"], gone["uri"], unclassified["uri"], uri("f" * 64)])
    provider = blobs.get_blob_storage_provider()
    assert isinstance(provider, FilesystemBlobStorageProvider)
    os.unlink(provider.get_file_path(gone["hash"]))
    assets = export_blob_assets(eval_id, {"results": [{"response": {"output": text}}]}, False)
    assert assets == [{"hash": kept["hash"], "mimeType": "image/png", "sizeBytes": 10,
                       "data": base64.b64encode(b"k" * 10).decode()}]  # fmt: skip
    assert any("Skipping missing blob in eval export" in r.getMessage() for r in caplog.records)
