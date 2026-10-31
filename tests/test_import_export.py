"""`export` / `import` commands, the OpenAI Evals importer and src/util/exportToFile.

Ports test/commands/export.test.ts, test/commands/import.test.ts and
test/util/exportToFile/*.test.ts. Fixtures in tests/fixtures/import_export/: sample-export.json and
the OpenAI JSONL files are upstream's; node-v4-export.json is `promptfoo export eval` output from
Node 0.125.1; legacy-v2-export.json and v4-with-traces.json are hand-built variants.
"""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import io
import json
import logging
import os
import sqlite3
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from promptphoo import db, paths
from promptphoo.cli import export as export_cli
from promptphoo.cli import import_ as import_cli
from promptphoo.importers.parse import parse_import_file
from promptphoo.models.eval import Eval
from promptphoo.output.export_to_file import (
    JsonlFileWriter,
    convert_eval_result_to_table_cell,
    convert_test_results_to_table_row,
)

FIXTURES = Path(__file__).parent / "fixtures" / "import_export"
SAMPLE = FIXTURES / "sample-export.json"
NODE_V4 = FIXTURES / "node-v4-export.json"
LEGACY = FIXTURES / "legacy-v2-export.json"
TRACED = FIXTURES / "v4-with-traces.json"


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(cfg))
    monkeypatch.delenv("IS_TESTING", raising=False)
    monkeypatch.delenv("PROMPTFOO_LOG_DIR", raising=False)
    paths.set_config_directory_path(None)
    db.close_db()
    db.run_db_migrations()
    yield cfg
    db.close_db()


@pytest.fixture
def caplog_info(caplog):
    caplog.set_level(logging.DEBUG, logger="promptphoo")
    return caplog


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(tmp_path: Path, name: str, data) -> str:
    p = tmp_path / name
    p.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return str(p)


def _results(eval_id: str) -> list[dict]:
    conn = db.get_db()
    cur = conn.cursor()
    cur.row_factory = lambda c, r: {col[0]: r[i] for i, col in enumerate(c.description)}
    return cur.execute(
        "SELECT * FROM eval_results WHERE eval_id = ? ORDER BY test_idx, prompt_idx", (eval_id,)
    ).fetchall()


def _rows(sql: str, args: tuple = ()) -> list[tuple]:
    cur = db.get_db().cursor()
    cur.row_factory = None
    return cur.execute(sql, args).fetchall()


def _count(table: str, where: str = "", args: tuple = ()) -> int:
    return _rows(f"SELECT count(*) FROM {table} {where}", args)[0][0]


def _errors(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _infos(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]


def _normalize_export(data: dict) -> dict:
    data = copy.deepcopy(data)
    data["metadata"].pop("exportedAt", None)
    data["metadata"].pop("nodeVersion", None)
    for result in data["results"].get("results") or []:
        result["id"] = "<uuid>"
    return data


# --- parse / OpenAI Evals ---------------------------------------------------------------------


@pytest.mark.parametrize("text", ["[]", "[null]", "null", "42", '{"other":true}'])
def test_parse_preserves_non_openai_json(text):
    assert parse_import_file(text) == {"source": "promptfoo-json", "evalData": json.loads(text)}


def test_parse_rejects_nan_like_json_parse():
    with pytest.raises(ValueError, match="is not valid JSON"):
        parse_import_file('{"a": NaN}')


def test_parse_invalid_json_message_matches_v8():
    with pytest.raises(ValueError) as err:
        parse_import_file("invalid json\n")
    assert str(err.value) == "Unexpected token 'i', \"invalid json\n\" is not valid JSON"


def test_parse_jsonl_schema_mismatch():
    with pytest.raises(ValueError, match="parsed as JSONL but line 1 is not a valid OpenAI"):
        parse_import_file('{"foo":1}\n{"bar":2}')


def test_parse_jsonl_strips_bom_like_js_trim():
    row = json.dumps(
        {"run_id": "r", "data_source_idx": 0, "item": {"input": "x"}, "grades": {"A": 1}}
    )
    parsed = parse_import_file("\ufeff" + row + "\n" + row)
    assert parsed["source"] == "openai-evals-jsonl"


def test_import_openai_evals_items(config_dir, caplog_info):
    assert import_cli.import_eval(str(FIXTURES / "openai-evals-items.jsonl")) == 0
    record = Eval.find_by_id("openai-evals-evalrun_import_fixture")
    assert record is not None
    assert record.config["description"] == "Imported from OpenAI Evals run evalrun_import_fixture"
    assert record.config["metadata"]["openaiEvalsImport"] == {
        "format": "dashboard-output-items-jsonl",
        "rowCount": 2,
        "runIds": ["evalrun_import_fixture"],
    }
    assert len(record.prompts) == 1
    metrics = record.prompts[0]["metrics"]
    assert (metrics["testPassCount"], metrics["testFailCount"]) == (1, 1)
    assert (metrics["assertPassCount"], metrics["assertFailCount"]) == (5, 1)
    assert "testCacheHitCount" not in metrics
    assert metrics["namedScoreWeights"] == {}

    rows = _results(record.id)
    assert len(rows) == 2
    first = json.loads(rows[0]["metadata"])["openai"]
    assert json.loads(rows[0]["named_scores"]) == {
        "Relevance": 6,
        "Directness": 3,
        "Tool called": 1,
    }
    grading = json.loads(rows[0]["grading_result"])
    assert grading["pass"] is True and len(grading["componentResults"]) == 3
    assert first["dataSourceIdx"] == 10
    assert first["graderSamples"]["Relevance"] == {
        "outputs": [{"role": "assistant", "content": '{"result":6}'}]
    }
    assert first["outputItem"]["run_id"] == "evalrun_import_fixture"
    assert rows[1]["success"] == 0
    assert rows[1]["failure_reason"] == 1
    assert "Relevance" in json.loads(rows[1]["grading_result"])["reason"]

    # run-based collision
    caplog_info.clear()
    assert import_cli.import_eval(str(FIXTURES / "openai-evals-items.jsonl")) == 1
    assert any(
        "Eval openai-evals-evalrun_import_fixture already exists" in m for m in _errors(caplog_info)
    )


def test_import_openai_multi_run(config_dir):
    assert import_cli.import_eval(str(FIXTURES / "openai-evals-multi-run-items.jsonl")) == 0
    eval_id = _rows("SELECT id FROM evals")[0][0]
    record = Eval.find_by_id(eval_id)
    assert record is not None
    assert [p["provider"] for p in record.prompts] == [
        "openai-evals:evalrun_error_fixture",
        "openai-evals:evalrun_pass_fixture",
    ]
    assert record.prompts[0]["metrics"]["testErrorCount"] == 1
    assert record.prompts[0]["metrics"]["testFailCount"] == 0
    usage = record.prompts[1]["metrics"]["tokenUsage"]
    assert (usage["total"], usage["prompt"], usage["completion"], usage["cached"]) == (10, 8, 2, 1)
    assert usage["numRequests"] == 1
    rows = _results(eval_id)
    assert [r["test_idx"] for r in rows] == [0, 0]
    assert len({r["prompt_id"] for r in rows}) == 2
    assert _count("evals_to_prompts", "WHERE eval_id = ?", (eval_id,)) == 2
    by_run = {json.loads(r["metadata"])["openai"]["runId"]: r for r in rows}
    err = by_run["evalrun_error_fixture"]
    assert err["failure_reason"] == 2
    assert err["error"] == "Response input item missing role"
    ok = json.loads(by_run["evalrun_pass_fixture"]["response"])
    assert ok["output"] == "Hardware" and ok["finishReason"] == "stop"
    assert ok["tokenUsage"] == {
        "total": 10, "prompt": 8, "completion": 2, "cached": 1, "numRequests": 1
    }  # fmt: skip


@pytest.mark.parametrize(
    ("items", "test_count", "test_indices"),
    [
        ([{"input": "First dataset item"}, {"input": "Different dataset item"}], 2, [0, 1]),
        (
            [{"a": 1, "b": 2, "nested": {"x": 10, "y": 20}},
             {"nested": {"y": 20, "x": 10}, "b": 2, "a": 1}],
            1,
            [0, 0],
        ),
    ],
)  # fmt: skip
def test_import_openai_alignment(config_dir, tmp_path, items, test_count, test_indices):
    rows = [
        {"run_id": f"run_{i}", "data_source_idx": 0, "item": item, "grades": {"G": 1},
         "passes": {"G": True}}
        for i, item in enumerate(items)
    ]  # fmt: skip
    path = _write(tmp_path, "rows.jsonl", "\n".join(json.dumps(r) for r in rows))
    assert import_cli.import_eval(path) == 0
    eval_id = _rows("SELECT id FROM evals")[0][0]
    record = Eval.find_by_id(eval_id)
    assert record is not None
    assert len(record.config["tests"]) == test_count
    assert sorted(r["test_idx"] for r in _results(eval_id)) == test_indices


@pytest.mark.parametrize("as_array", [False, True])
def test_import_openai_score_only_rows(config_dir, tmp_path, as_array):
    row = {
        "run_id": "evalrun_without_passes",
        "data_source_idx": 0,
        "item": {"input": "Grade this later"},
        "grades": {"Quality": 0.5},
        "eval_id": "eval_future_shape",
    }
    path = _write(tmp_path, "row.json", [row] if as_array else row)
    assert import_cli.import_eval(path) == 0
    record = Eval.find_by_id("openai-evals-evalrun_without_passes")
    assert record is not None
    (result,) = _results(record.id)
    grading = json.loads(result["grading_result"])
    assert result["success"] == 1 and grading["pass"] is True
    assert grading["componentResults"] == []
    assert "did not include grader pass results" in grading["reason"]
    metadata = json.loads(result["metadata"])["openai"]
    assert metadata["passes"] == {}
    assert metadata["outputItem"]["eval_id"] == "eval_future_shape"


def test_import_openai_mixed_jsonl_fails(config_dir, tmp_path, caplog_info):
    good = json.dumps(
        {"run_id": "r", "data_source_idx": 0, "item": {"input": "hi"}, "grades": {"X": 1}}
    )
    path = _write(tmp_path, "mixed.jsonl", f"{good}\nthis is not json at all\n{good}")
    assert import_cli.import_eval(path) == 1
    assert any(m.startswith("Failed to import eval") for m in _errors(caplog_info))
    assert _count("evals") == 0


def test_import_openai_sanitizes_run_id(config_dir, tmp_path):
    row = {
        "run_id": "evalrun/with spaces & slashes",
        "data_source_idx": 0,
        "item": {"input": "sanitize me"},
        "grades": {"Match": 1},
        "passes": {"Match": True},
    }
    assert import_cli.import_eval(_write(tmp_path, "r.jsonl", row)) == 0
    record = Eval.find_by_id("openai-evals-evalrun_with_spaces___slashes")
    assert record is not None
    assert record.config["metadata"]["openaiEvalsImport"]["runIds"] == [
        "evalrun/with spaces & slashes"
    ]


# --- import: errors ----------------------------------------------------------------------------


def test_import_missing_file(config_dir, caplog_info):
    assert import_cli.import_eval("non-existent-file.json") == 1
    assert _errors(caplog_info) == [
        "Failed to import eval: ENOENT: no such file or directory, open 'non-existent-file.json'"
    ]


def test_import_invalid_json(config_dir, tmp_path, caplog_info):
    assert import_cli.import_eval(_write(tmp_path, "bad.json", "invalid json")) == 1
    assert _errors(caplog_info) == [
        "Failed to import eval: Unexpected token 'i', \"invalid json\" is not valid JSON"
    ]


def test_import_rejects_unsupported_results(config_dir, tmp_path, caplog_info):
    data = {
        "id": "eval-unsupported-results",
        "config": {"description": "unsupported results shape"},
        "results": {"results": [{"success": True, "response": {"output": "not enough"}}]},
    }
    assert import_cli.import_eval(_write(tmp_path, "u.json", data)) == 1
    assert _errors(caplog_info) == ["Failed to import eval: Unsupported eval export results format"]
    assert Eval.find_by_id("eval-unsupported-results") is None


def test_import_with_corrupt_created_at(config_dir, tmp_path, caplog_info):
    data = _load(SAMPLE)
    data["metadata"]["evaluationCreatedAt"] = "this-is-not-a-date"
    assert import_cli.import_eval(_write(tmp_path, "d.json", data)) == 0
    assert any(
        'unparseable metadata.evaluationCreatedAt ("this-is-not-a-date")' in m
        for m in _warnings(caplog_info)
    )
    record = Eval.find_by_id(data["evalId"])
    assert record is not None
    # falls back to results.timestamp
    assert record.created_at == import_cli.parse_imported_date(data["results"]["timestamp"], "x")


# --- import: sample file -----------------------------------------------------------------------


def test_import_sample_preserves_id_prompts_and_rows(config_dir, caplog_info):
    sample = _load(SAMPLE)
    assert import_cli.import_eval(str(SAMPLE)) == 0
    assert _infos(caplog_info)[-1] == (
        f"Eval with ID {sample['evalId']} has been successfully imported."
    )
    record = Eval.find_by_id(sample["evalId"])
    assert record is not None
    assert len(record.prompts) == 2
    assert record.config["description"] == sample["config"]["description"]
    assert record.created_at == import_cli.parse_imported_date(
        sample["metadata"]["evaluationCreatedAt"], "x"
    )
    assert record.author == sample["metadata"]["author"]
    assert len(_results(record.id)) == len(sample["results"]["results"])
    # prompts, datasets and tags rows like Eval.create
    from promptphoo.evaluator.run import generate_id_from_prompt

    prompt_ids = {generate_id_from_prompt(p) for p in sample["results"]["prompts"]}
    assert _count("evals_to_prompts", "WHERE eval_id = ?", (record.id,)) == len(prompt_ids)
    assert _count("evals_to_datasets", "WHERE eval_id = ?", (record.id,)) == 1
    tags = sample["config"].get("tags") or {}
    assert _count("evals_to_tags", "WHERE eval_id = ?", (record.id,)) == len(tags)


def test_import_derives_vars_from_results(config_dir, tmp_path):
    sample = _load(SAMPLE)
    sample.pop("vars", None)
    assert import_cli.import_eval(_write(tmp_path, "s.json", sample)) == 0
    record = Eval.find_by_id(sample["evalId"])
    assert record is not None
    expected: dict[str, None] = {}
    for result in sample["results"]["results"]:
        expected.update(dict.fromkeys(result.get("vars") or {}))
    assert record.vars == list(expected)


def test_import_preserves_vars_runtime_options_and_durations(config_dir, tmp_path):
    sample = _load(SAMPLE)
    sample["vars"] = ["riddle", "unused_export_var", 7]
    sample["runtimeOptions"] = {"cache": False, "maxConcurrency": 2, "repeat": 3}
    sample["results"]["stats"].update(
        durationMs=4321, generationDurationMs=123, evaluationDurationMs=4198
    )
    assert import_cli.import_eval(_write(tmp_path, "s.json", sample)) == 0
    record = Eval.find_by_id(sample["evalId"])
    assert record is not None
    assert record.vars == ["riddle", "unused_export_var"]
    assert record.runtime_options == sample["runtimeOptions"]
    assert (record.duration_ms, record.generation_duration_ms, record.evaluation_duration_ms) == (
        4321,
        123,
        4198,
    )


def test_import_preserves_author_over_cloud_identity(config_dir, tmp_path):
    (config_dir / "promptfoo.yaml").write_text(
        "account:\n  email: someone-else@example.com\ncloud:\n  apiKey: k\n", encoding="utf-8"
    )
    sample = _load(SAMPLE)
    sample["metadata"]["author"] = "original@example.com"
    assert import_cli.import_eval(_write(tmp_path, "s.json", sample)) == 0
    record = Eval.find_by_id(sample["evalId"])
    assert record is not None and record.author == "original@example.com"


def test_import_legacy_table_backed_export(config_dir):
    assert import_cli.import_eval(str(LEGACY)) == 0
    record = Eval.find_by_id("eval-legacy-table-backed")
    assert record is not None
    assert record.author == "legacy-author"
    config_text = json.dumps(record.config)
    assert "imported-legacy-secret" not in config_text
    assert "imported-header-secret" not in config_text
    assert record.config["tracing"]["provider"]["headers"] == {"X-Scope-OrgID": "tenant-a"}
    assert _rows("SELECT is_redteam FROM evals")[0][0] == 1
    summary = record.to_evaluate_summary()
    assert summary["version"] == 2
    assert summary["results"][0]["vars"] == {"topic": "legacy"}
    assert summary["table"]["head"]["vars"] == ["topic"]
    assert _count("eval_results") == 0


# --- import: traces ----------------------------------------------------------------------------


def test_import_traces_and_clear_dangling_linkage(config_dir, caplog_info):
    data = _load(TRACED)
    assert import_cli.import_eval(str(TRACED)) == 0
    assert _warnings(caplog_info) == [
        "Skipping malformed trace span during import",
        "Skipping malformed trace during import",
    ]
    assert _rows("SELECT trace_id, evaluation_id, test_case_id FROM traces") == [
        ("a" * 32, data["evalId"], "tc-0")
    ]
    assert _rows("SELECT span_id FROM spans ORDER BY span_id") == [
        ("s1",),
        ("s2",),
    ]
    metas = [json.loads(r["metadata"]) for r in _results(data["evalId"])]
    linked = [m["__promptfoo"]["traceLinkage"] for m in metas if "__promptfoo" in m]
    assert linked == [{"traceId": "a" * 32, "evaluationId": data["evalId"]}]


def test_import_new_id_remaps_trace_ids(config_dir):
    assert import_cli.import_eval(str(TRACED)) == 0
    assert import_cli.import_eval(str(TRACED), new_id=True) == 0
    rows = _rows("SELECT trace_id, evaluation_id FROM traces")
    assert len(rows) == 2
    assert len({r[0] for r in rows}) == 2 and len({r[1] for r in rows}) == 2


# --- import: blob assets -----------------------------------------------------------------------


def _with_blob(data: dict, payload: bytes, *, hash_: str | None = None, size: int | None = None):
    digest = hash_ or hashlib.sha256(payload).hexdigest()
    data["results"]["results"][0]["response"] = {"output": f"promptfoo://blob/{digest}"}
    data["blobAssets"] = [
        {
            "hash": digest,
            "mimeType": "image/png",
            "sizeBytes": len(payload) if size is None else size,
            "data": base64.b64encode(payload).decode(),
        }
    ]
    return data


def test_import_stores_blob_assets_and_records_import_references(config_dir, tmp_path):
    from promptphoo.blobs import get_blob_by_hash

    payload = b"portable image from eval export"
    data = _with_blob(_load(SAMPLE), payload)
    assert import_cli.import_eval(_write(tmp_path, "b.json", data)) == 0
    digest = hashlib.sha256(payload).hexdigest()
    blob = get_blob_by_hash(digest)
    assert blob.data == payload and blob.metadata["mimeType"] == "image/png"
    assert (config_dir / "blobs" / digest[:2] / digest[2:4] / digest).read_bytes() == payload
    assert _rows("SELECT blob_hash, eval_id, location, kind FROM blob_references") == [
        (digest, data["evalId"], "import", None)
    ]


def test_import_blob_assets_referenced_only_by_traces(config_dir, tmp_path):
    payload = b"portable trace image from eval export"
    digest = hashlib.sha256(payload).hexdigest()
    data = _with_blob(_load(SAMPLE), payload)
    data["results"]["results"][0]["response"] = {"output": "no media"}
    data["traces"] = [
        {"traceId": "trace-media-import", "evaluationId": data["evalId"],
         "testCaseId": "trace-media-case", "metadata": {"attachment": f"promptfoo://blob/{digest}"},
         "spans": []}
    ]  # fmt: skip
    assert import_cli.import_eval(_write(tmp_path, "b.json", data)) == 0
    assert _rows("SELECT blob_hash, location FROM blob_references") == [(digest, "import")]


@pytest.mark.parametrize(
    ("declared", "expected"),
    [("text/html", "application/octet-stream"), ("image/svg+xml", "application/octet-stream"),
     (" IMAGE/PNG ", "image/png"), ("audio/wav", "audio/wav")],
)  # fmt: skip
def test_import_sanitizes_blob_mime_types(config_dir, tmp_path, declared, expected):
    data = _with_blob(_load(SAMPLE), b"<svg/> or <html>")
    data["blobAssets"][0]["mimeType"] = declared
    assert import_cli.import_eval(_write(tmp_path, "b.json", data)) == 0
    assert _rows("SELECT mime_type FROM blob_assets") == [(expected,)]


def test_import_reference_only_result_does_not_adopt_unregistered_file(config_dir, tmp_path):
    from promptphoo.blobs import get_share_authorized_blob, store_blob
    from promptphoo.output.writer import create_output_data

    payload = b"unregistered file retained after failed import-store transaction"
    digest = hashlib.sha256(payload).hexdigest()
    with pytest.raises(sqlite3.IntegrityError):
        store_blob(payload, "image/png", {"evalId": "missing-blob-import-eval"})
    blob_path = config_dir / "blobs" / digest[:2] / digest[2:4] / digest
    assert blob_path.read_bytes() == payload
    data = _load(SAMPLE)
    data["results"]["results"] = [data["results"]["results"][0]]
    data["results"]["results"][0]["response"] = {"output": f"promptfoo://blob/{digest}"}
    assert import_cli.import_eval(_write(tmp_path, "r.json", data)) == 0
    assert _count("blob_assets") == 0 and _count("blob_references") == 0
    assert get_share_authorized_blob(digest, data["evalId"]) is None
    record = Eval.find_by_id(data["evalId"])
    assert "blobAssets" not in create_output_data(record, None, include_media=True)


def test_import_skips_blob_scan_without_embedded_assets(config_dir, monkeypatch):
    def fail(*_a, **_k):
        raise AssertionError("collect_blob_hashes should not run")

    monkeypatch.setattr(import_cli, "collect_blob_hashes", fail)
    assert import_cli.import_eval(str(SAMPLE)) == 0


def test_import_force_keeps_existing_when_blob_store_fails(config_dir, tmp_path, monkeypatch,
                                                         caplog_info):  # fmt: skip
    sample = _load(SAMPLE)
    assert import_cli.import_eval(str(SAMPLE)) == 0
    replacement = _with_blob(_load(SAMPLE), b"valid replacement bytes")

    def boom(*_a, **_k):
        raise RuntimeError("embedded media storage failed")

    monkeypatch.setattr(import_cli, "store_blob", boom)
    assert import_cli.import_eval(_write(tmp_path, "r.json", replacement), force=True) == 1
    assert _errors(caplog_info) == ["Failed to import eval: embedded media storage failed"]
    assert len(_results(sample["evalId"])) == len(sample["results"]["results"])


def test_import_rejects_blob_hash_mismatch(config_dir, tmp_path, caplog_info):
    data = _with_blob(_load(SAMPLE), b"wrong bytes", hash_="f" * 64)
    assert import_cli.import_eval(_write(tmp_path, "b.json", data)) == 1
    assert _errors(caplog_info) == [
        f"Failed to import eval: Embedded blob hash mismatch for {'f' * 64}"
    ]


def test_import_rejects_blob_size_mismatch_and_limit(config_dir, tmp_path, caplog_info):
    data = _with_blob(_load(SAMPLE), b"abc", size=4)
    assert import_cli.import_eval(_write(tmp_path, "b.json", data)) == 1
    assert "size does not match its exported metadata" in _errors(caplog_info)[0]
    caplog_info.clear()
    data = _with_blob(_load(SAMPLE), b"abc", size=import_cli.BLOB_MAX_SIZE + 1)
    assert import_cli.import_eval(_write(tmp_path, "b2.json", data)) == 1
    assert "exceeds the 52428800 byte import limit" in _errors(caplog_info)[0]


def test_import_ignores_unreferenced_and_malformed_blobs(config_dir, tmp_path, caplog_info):
    data = _load(SAMPLE)
    data["blobAssets"] = [
        {"hash": "e" * 64, "mimeType": "image/png", "sizeBytes": 1, "data": "not-validated"},
        {"hash": "nope"},
    ]
    assert import_cli.import_eval(_write(tmp_path, "b.json", data)) == 0
    assert _warnings(caplog_info) == ["Skipping malformed embedded blob during import"]
    assert _count("blob_assets") == 0
    assert not (config_dir / "blobs").exists() or not any((config_dir / "blobs").rglob("*e*"))


def test_collect_blob_hashes_bounds():
    h = "A" * 64
    value = {"a": [{"hash": h}, f"x promptfoo://blob/{'b' * 64} y"], "deep": {"uri": "z"}}
    assert list(import_cli.collect_blob_hashes(value)) == ["a" * 64, "b" * 64]
    assert list(import_cli.collect_blob_hashes(value, max_depth=1)) == []
    long_text = "x" * 50 + f"promptfoo://blob/{'c' * 64}"
    assert list(import_cli.collect_blob_hashes(long_text, max_string_length=10)) == []


# --- import: collisions ------------------------------------------------------------------------


def test_import_rejects_existing_eval(config_dir, caplog_info):
    sample = _load(SAMPLE)
    assert import_cli.import_eval(str(SAMPLE)) == 0
    caplog_info.clear()
    assert import_cli.import_eval(str(SAMPLE)) == 1
    assert _errors(caplog_info) == [
        f"Eval {sample['evalId']} already exists. Use --new-id to import with a new ID, or "
        "--force to replace it."
    ]


def test_import_new_id(config_dir):
    sample = _load(SAMPLE)
    assert import_cli.import_eval(str(SAMPLE)) == 0
    assert import_cli.import_eval(str(SAMPLE), new_id=True) == 0
    ids = [r[0] for r in _rows("SELECT id FROM evals")]
    assert len(ids) == 2 and sample["evalId"] in ids
    other = next(i for i in ids if i != sample["evalId"])
    assert other.startswith("eval-") and other[9:] == sample["evalId"][9:]


def test_import_force_replaces(config_dir, caplog_info):
    sample = _load(SAMPLE)
    assert import_cli.import_eval(str(SAMPLE)) == 0
    assert import_cli.import_eval(str(SAMPLE), force=True) == 0
    assert f"Replacing existing eval {sample['evalId']}" in _infos(caplog_info)
    assert _count("evals") == 1
    assert len(_results(sample["evalId"])) == len(sample["results"]["results"])


def test_import_force_keeps_existing_when_preflight_fails(config_dir, tmp_path, caplog_info):
    sample = _load(SAMPLE)
    assert import_cli.import_eval(str(SAMPLE)) == 0
    bad = _with_blob(_load(SAMPLE), b"wrong replacement bytes", hash_="f" * 64)
    assert import_cli.import_eval(_write(tmp_path, "bad.json", bad), force=True) == 1
    assert any("Embedded blob hash mismatch" in m for m in _errors(caplog_info))
    assert len(_results(sample["evalId"])) == len(sample["results"]["results"])


def test_import_force_failure_after_delete_warns(config_dir, monkeypatch, caplog_info):
    sample = _load(SAMPLE)
    assert import_cli.import_eval(str(SAMPLE)) == 0
    from promptphoo.db import import_export as ie

    def boom(*_a, **_k):
        raise RuntimeError("result write failed")

    monkeypatch.setattr(ie, "create_many_from_evaluate_result", boom)
    assert import_cli.import_eval(str(SAMPLE), force=True) == 1
    assert _errors(caplog_info) == [
        "Failed to import eval: result write failed",
        f"The existing eval {sample['evalId']} was deleted for a --force replacement that did "
        "not complete. Re-import it from a backup.",
    ]
    assert _results(sample["evalId"]) == []


def test_import_force_when_absent(config_dir):
    assert import_cli.import_eval(str(SAMPLE), force=True) == 0
    assert _count("evals") == 1


def test_import_force_deletes_traces_of_replaced_eval(config_dir):
    assert import_cli.import_eval(str(TRACED)) == 0
    assert import_cli.import_eval(str(TRACED), force=True) == 0
    assert _count("traces") == 1 and _count("spans") == 2


# --- import: date parsing ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-01-02T03:04:05.000Z", 1704164645000),
        ("2024-01-02T03:04:05.123+02:00", 1704157445123),
        ("2024-01-02", 1704153600000),
        (1704164645000, 1704164645000),
        ("Tue, 02 Jan 2024 03:04:05 GMT", 1704164645000),
        ("not a date", None),
        ({"a": 1}, None),
        (float("nan"), None),
        (9e15, None),
    ],
)
def test_parse_imported_date(value, expected):
    assert import_cli.parse_imported_date(value, "f") == expected


# --- export eval -------------------------------------------------------------------------------


def test_export_latest_and_by_id(config_dir, tmp_path, caplog_info):
    assert import_cli.import_eval(str(NODE_V4)) == 0
    eval_id = _load(NODE_V4)["evalId"]
    out = tmp_path / "out.json"
    assert export_cli.export_eval("latest", str(out)) == 0
    assert _infos(caplog_info)[-1] == (
        f"Eval with ID latest has been successfully exported to {out}."
    )
    by_id = tmp_path / "by_id.json"
    assert export_cli.export_eval(eval_id, str(by_id), include_media=True) == 0
    assert _normalize_export(_load(out)) == _normalize_export(_load(by_id))


def test_export_matches_node_export_of_same_eval(config_dir, tmp_path):
    """Node export -> Python import -> Python export is the Node export, modulo the
    nondeterministic fields (exportedAt, nodeVersion, eval_results ids)."""
    assert import_cli.import_eval(str(NODE_V4)) == 0
    out = tmp_path / "out.json"
    assert export_cli.export_eval("latest", str(out)) == 0
    assert _normalize_export(_load(out)) == _normalize_export(_load(NODE_V4))
    text = out.read_text(encoding="utf-8")
    assert not text.endswith("\n") and text.startswith('{\n  "evalId"')


def test_export_round_trips_through_import(config_dir, tmp_path):
    assert import_cli.import_eval(str(SAMPLE)) == 0
    first = tmp_path / "first.json"
    assert export_cli.export_eval("latest", str(first)) == 0
    db.close_db()
    os.environ["PROMPTFOO_CONFIG_DIR"] = str(tmp_path / "cfg2")
    db.run_db_migrations()
    assert import_cli.import_eval(str(first)) == 0
    second = tmp_path / "second.json"
    assert export_cli.export_eval("latest", str(second)) == 0
    assert _normalize_export(_load(first)) == _normalize_export(_load(second))


def test_export_legacy_eval(config_dir, tmp_path):
    assert import_cli.import_eval(str(LEGACY)) == 0
    out = tmp_path / "legacy.json"
    assert export_cli.export_eval("eval-legacy-table-backed", str(out)) == 0
    data = _load(out)
    assert data["evalId"] == "eval-legacy-table-backed"
    assert data["results"]["version"] == 2
    assert data["results"]["table"]["head"]["vars"] == ["topic"]
    # an emptied tracing auth map stays an (empty) object, as in Node
    assert data["config"]["tracing"]["provider"]["auth"] == {}
    assert data["metadata"]["author"] == "legacy-author"
    # and re-imports as a legacy eval
    db.close_db()
    os.environ["PROMPTFOO_CONFIG_DIR"] = str(tmp_path / "cfg2")
    db.run_db_migrations()
    assert import_cli.import_eval(str(out)) == 0
    reloaded = Eval.find_by_id("eval-legacy-table-backed")
    assert reloaded is not None and reloaded.version() == 3


def test_export_to_stdout(config_dir, capsys):
    assert import_cli.import_eval(str(NODE_V4)) == 0
    from promptphoo.cli.main import setup_logging

    setup_logging()
    capsys.readouterr()
    assert export_cli.export_eval("latest") == 0
    out = capsys.readouterr().out
    assert _normalize_export(json.loads(out)) == _normalize_export(_load(NODE_V4))


def test_export_not_found(config_dir, caplog_info):
    assert export_cli.export_eval("nonexistent-id") == 1
    assert _errors(caplog_info) == ["No eval found with ID nonexistent-id"]


def test_export_error(config_dir, tmp_path, caplog_info):
    assert import_cli.import_eval(str(NODE_V4)) == 0
    assert export_cli.export_eval("latest", str(tmp_path / "out.unknown")) == 1
    assert _errors(caplog_info)[0].startswith(
        "Failed to export eval: Error: Invariant failed: Unsupported output file format unknown"
    )


@pytest.mark.parametrize("fmt", ["yaml", "csv", "html", "xml", "jsonl", "junit.xml", "txt"])
def test_export_other_formats(config_dir, tmp_path, fmt):
    assert import_cli.import_eval(str(NODE_V4)) == 0
    out = tmp_path / f"out.{fmt}"
    assert export_cli.export_eval("latest", str(out)) == 0
    assert out.stat().st_size > 0


def test_cli_registration(config_dir, tmp_path):
    from promptphoo.cli.main import build_cli

    cli = build_cli(["export"])
    runner = CliRunner()
    result = runner.invoke(cli, ["import", str(SAMPLE)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(cli, ["import", str(SAMPLE)])
    assert result.exit_code == 1
    out = tmp_path / "e.json"
    result = runner.invoke(cli, ["export", "eval", "latest", "-o", str(out)])
    assert result.exit_code == 0 and out.exists()
    result = runner.invoke(cli, ["export", "eval", "missing-id"])
    assert result.exit_code == 1
    assert "--include-media" in runner.invoke(cli, ["export", "eval", "--help"]).output
    help_text = runner.invoke(cli, ["import", "--help"]).output
    assert "--new-id" in help_text and "--force" in help_text


# --- export logs -------------------------------------------------------------------------------


def test_export_logs_missing_directory(config_dir, caplog_info):
    log_dir = os.path.join(str(config_dir), "logs")
    assert export_cli.export_logs() == 1
    assert _errors(caplog_info) == [
        'No log directory found. Logs are created when running commands like "promptfoo eval".'
        f"\nLog directory: {log_dir}"
    ]


def test_export_logs_no_files(config_dir, caplog_info):
    (config_dir / "logs").mkdir()
    (config_dir / "logs" / "other.txt").write_text("x")
    assert export_cli.export_logs() == 1
    assert _errors(caplog_info)[0].startswith("No log files found in the logs directory.")


@pytest.mark.parametrize("count", ["invalid", "0", "-2"])
def test_export_logs_bad_count(config_dir, caplog_info, count):
    (config_dir / "logs").mkdir()
    (config_dir / "logs" / "promptfoo-debug-2025-01-01.log").write_text("x")
    assert export_cli.export_logs(count=count) == 1
    assert _errors(caplog_info) == ["Count must be a positive number"]


def test_export_logs_archive(config_dir, tmp_path, caplog_info, monkeypatch):
    logs = tmp_path / "custom-logs"
    logs.mkdir()
    monkeypatch.setenv("PROMPTFOO_LOG_DIR", str(logs))
    old = logs / "promptfoo-debug-2025-01-01.log"
    new = logs / "promptfoo-error-2025-01-02.log"
    old.write_text("hello logs")
    new.write_text("x" * 600)
    os.utime(old, (1735689600, 1735689600))
    os.utime(new, (1735776000, 1735776000))
    out = tmp_path / "logs"
    assert export_cli.export_logs(output=str(out)) == 0
    archive = tmp_path / "logs.gz"
    assert _infos(caplog_info)[0] == "Collecting 2 log file(s)..."
    assert f"Log files have been collected in: {archive}" in _infos(caplog_info)
    raw = gzip.decompress(archive.read_bytes())
    assert len(raw) % 512 == 0
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        members = tar.getmembers()
        assert [m.name for m in members] == [new.name, old.name]  # newest first
        assert members[1].mtime == 1735689600 and members[1].mode == 0o644
        extracted = tar.extractfile(members[1])
        assert extracted is not None and extracted.read() == b"hello logs"
    caplog_info.clear()
    assert export_cli.export_logs(count="1abc", output=str(tmp_path / "one.gz")) == 0
    with tarfile.open(tmp_path / "one.gz") as tar:
        assert [m.name for m in tar.getmembers()] == [new.name]


def test_export_logs_default_name(config_dir, tmp_path, monkeypatch):
    (config_dir / "logs").mkdir()
    (config_dir / "logs" / "promptfoo-debug-2025-01-01.log").write_text("x")
    monkeypatch.chdir(tmp_path)
    assert export_cli.export_logs() == 0
    (name,) = [p.name for p in tmp_path.iterdir() if p.name.startswith("promptfoo-logs-")]
    import re

    assert re.fullmatch(r"promptfoo-logs-\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d-\d{3}Z\.gz", name)


def test_tar_header_matches_upstream_layout():
    header = export_cli.tar_header("a.log", 10, 1735689600)
    assert len(header) == 512
    assert header[124:136] == b"00000000012 "
    assert header[156] == 0x30
    stored = int(header[148:154], 8)
    assert stored == sum(header[:148]) + sum(header[156:]) + 8 * 32


# --- exportToFile --------------------------------------------------------------------------------


def _mock_result(**overrides):
    base = {
        "id": "test-id",
        "testIdx": 0,
        "promptIdx": 0,
        "description": "Test description",
        "testCase": {"vars": {}},
        "prompt": {"raw": "test prompt"},
        "response": {"output": "test output"},
        "provider": {"id": "test-provider", "label": "Test Provider"},
        "success": True,
        "score": 1,
        "latencyMs": 100,
        "namedScores": {},
        "cost": 0,
    }
    for key in ("testCase", "prompt", "response", "provider"):
        if key in overrides:
            overrides[key] = {**base[key], **overrides[key]}
    return {**base, **overrides}


def test_cell_success_and_failure():
    ok = convert_eval_result_to_table_cell(
        {"id": "t", "evalId": "e", "testCase": {"assert": [{"type": "contains"}]}, "success": True,
         "response": {"output": "test output"}, "prompt": {"raw": "p"},
         "provider": {"id": "prov"}, "failureReason": 0}
    )  # fmt: skip
    assert (ok["text"], ok["pass"], ok["provider"], ok["prompt"], ok["cost"]) == (
        "test output", True, "prov", "p", 0
    )  # fmt: skip
    failed = convert_eval_result_to_table_cell(
        {"id": "t", "testCase": {"assert": [{}]}, "success": False, "error": "test error",
         "response": {"output": "test output"}, "prompt": {"raw": "p"}, "provider": {"id": "p"}}
    )  # fmt: skip
    assert failed["text"] == "test output" and failed["pass"] is False


def test_cell_error_falsy_and_null_outputs():
    err = convert_eval_result_to_table_cell(
        {"id": "t", "testCase": {}, "error": "test error", "prompt": {"raw": "p"},
         "provider": {"id": "p"}}
    )  # fmt: skip
    assert err["text"] == "test error"
    for output, text in ((0, "0"), (False, "false"), ({"a": 1}, '{"a":1}')):
        cell = convert_eval_result_to_table_cell(_mock_result(response={"output": output}))
        assert cell["text"] == text
    null_err = convert_eval_result_to_table_cell(
        _mock_result(response={"output": None}, error="Provider returned null")
    )
    assert null_err["text"] == "Provider returned null"
    assert convert_eval_result_to_table_cell(_mock_result(response={"output": None}))["text"] == ""


def test_cell_from_eval_result_object(config_dir):
    assert import_cli.import_eval(str(NODE_V4)) == 0
    record = Eval.find_by_id(_load(NODE_V4)["evalId"])
    assert record is not None
    results = record.load_results()
    cell = convert_eval_result_to_table_cell(results[0])
    assert cell["evalId"] == record.id and cell["persisted"] is True
    assert cell["provider"] == "echo" and cell["text"] == results[0].response["output"]
    assert "description" not in cell


def test_row_basic_and_vars():
    row = convert_test_results_to_table_row(
        [_mock_result(description="test description",
                      testCase={"vars": {"var1": "value1", "var2": "value2", "o": {"c": "o"},
                                         "zero": 0, "f": False}})],
        ["var1", "var2", "missing", "o", "zero", "f"],
    )  # fmt: skip
    assert row["description"] == "test description"
    assert row["vars"] == ["value1", "value2", "", '{"c":"o"}', "0", "false"]
    assert row["outputs"][0]["text"] == "test output"
    assert row["testIdx"] == 0 and row["test"] == {"vars": row["test"]["vars"]}


def test_row_places_outputs_by_prompt_idx():
    row = convert_test_results_to_table_row(
        [_mock_result(promptIdx=1, response={"output": "second"}), _mock_result(promptIdx=0)], []
    )
    assert [o["text"] for o in row["outputs"]] == ["test output", "second"]


@pytest.mark.parametrize(
    ("vars_", "metadata", "expected"),
    [
        ({"sessionId": "from-vars"}, {"sessionId": "from-meta"}, "from-vars"),
        ({}, {"sessionId": "from-meta"}, "from-meta"),
        ({"sessionId": ""}, {"sessionId": "from-meta"}, "from-meta"),
        ({}, None, ""),
        ({}, {"sessionIds": ["a", "", None, "b"]}, "a\nb"),
        ({}, {"sessionIds": ["only"]}, "only"),
        ({}, {"sessionIds": ["x"], "sessionId": "y"}, "x"),
        ({}, {"sessionIds": [], "sessionId": "y"}, "y"),
        ({}, {"sessionIds": "nope", "sessionId": "y"}, "y"),
        ({"sessionId": "v"}, {"sessionIds": ["x"]}, "v"),
        ({}, {"sessionId": 123}, "123"),
        ({"sessionId": {"id": 1}}, None, '{"id":1}'),
    ],
)
def test_row_session_id(vars_, metadata, expected):
    result = _mock_result(testCase={"vars": vars_})
    if metadata is not None:
        result["metadata"] = metadata
    assert convert_test_results_to_table_row([result], ["sessionId"])["vars"] == [expected]


def test_jsonl_file_writer(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text("keep\n")
    untouched = JsonlFileWriter(str(path))
    untouched.close()
    assert path.read_text() == "keep\n"
    writer = JsonlFileWriter(str(path))
    writer.write({"a": 1.0})
    writer.write([1, "x"])
    writer.close()
    assert path.read_text() == '{"a":1}\n[1,"x"]\n'
    appender = JsonlFileWriter(str(path), append=True)
    appender.write(None)
    appender.close()
    assert path.read_text().endswith("null\n")
    bad = JsonlFileWriter(str(tmp_path / "missing" / "x.jsonl"))
    with pytest.raises(OSError, match="Failed to write JSONL output"):
        bad.write({})
