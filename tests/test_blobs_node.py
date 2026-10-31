"""Node <-> Python interop for blob storage. Skipped unless NODE_PROMPTFOO is set, e.g.
`NODE_PROMPTFOO="node ../promptfoo/dist/src/main.js"` (promptfoo 0.125.1).

`tests/fixtures/blobs/` is a deterministic `file://provider.py` that returns a fixed PNG (as a
data URL output, an `images[]` entry plus a metadata data URL, and OpenAI-style `b64_json`),
a WAV `audio.data`, a raw SVG output, and a data URL below the 1KB threshold. Both CLIs run
the same config in throwaway `PROMPTFOO_CONFIG_DIR`s, and the tests compare:

- the blob files and their `.meta.json` sidecars (minus `createdAt` and the absolute `key`),
- the `blob_assets` and `blob_references` rows (minus ids, eval ids and timestamps),
- the persisted `eval_results.response` JSON,
- `export --include-media` from either runtime over either runtime's DB, and the import of
  each export into the other runtime.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

NODE = os.environ.get("NODE_PROMPTFOO")
pytestmark = pytest.mark.skipif(not NODE, reason="NODE_PROMPTFOO is not set")

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "blobs"


EXTRA_ENV: dict[str, str] = {}


def _env(config_dir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROMPTFOO_")}
    env.update(EXTRA_ENV)
    env.update(
        PROMPTFOO_CONFIG_DIR=str(config_dir),
        PROMPTFOO_DISABLE_TELEMETRY="1",
        PROMPTFOO_DISABLE_UPDATE="1",
        PYTHONPATH=str(ROOT / "src"),
    )
    env.pop("IS_TESTING", None)
    return env


def node(config_dir: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    assert NODE
    return subprocess.run(
        [*shlex.split(NODE), *args], env=_env(config_dir), cwd=cwd, capture_output=True,
        text=True, timeout=300,
    )  # fmt: skip


def py(config_dir: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "promptphoo", *args], env=_env(config_dir), cwd=cwd,
        capture_output=True, text=True, timeout=300,
    )  # fmt: skip


RUNNERS = {"node": node, "py": py}


def _ok(proc: subprocess.CompletedProcess) -> subprocess.CompletedProcess:
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc


def _blob_files(config_dir: Path) -> dict[str, object]:
    base = config_dir / "blobs"
    out: dict[str, object] = {}
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        if rel.endswith(".meta.json"):
            meta = json.loads(path.read_text(encoding="utf-8"))
            assert meta.pop("key") == str(path)[: -len(".meta.json")]
            assert meta.pop("createdAt").endswith("Z")
            out[rel] = meta
        else:
            out[rel] = path.read_bytes()
    return out


def _blob_rows(config_dir: Path) -> dict[str, list[tuple]]:
    conn = sqlite3.connect(config_dir / "promptfoo.db")
    try:
        assets = conn.execute(
            "select hash, size_bytes, mime_type, provider from blob_assets order by hash"
        ).fetchall()
        refs = conn.execute(
            "select blob_hash, test_idx, prompt_idx, location, kind from blob_references "
            "order by blob_hash, test_idx, prompt_idx, location"
        ).fetchall()
        dangling = conn.execute(
            "select count(*) from blob_references r left join evals e on e.id = r.eval_id "
            "where e.id is null"
        ).fetchone()[0]
    finally:
        conn.close()
    assert dangling == 0
    return {"assets": assets, "references": refs}


def _responses(config_dir: Path) -> dict[tuple[int, int], object]:
    conn = sqlite3.connect(config_dir / "promptfoo.db")
    try:
        rows = conn.execute("select test_idx, prompt_idx, response from eval_results").fetchall()
    finally:
        conn.close()
    out = {}
    for test_idx, prompt_idx, response in rows:
        parsed = json.loads(response)
        parsed.pop("latencyMs", None)
        out[(test_idx, prompt_idx)] = parsed
    return out


def _run_eval(runner: str, base: Path, *extra: str) -> Path:
    work = base / f"{runner}_work"
    shutil.copytree(FIXTURE, work, ignore=shutil.ignore_patterns("__pycache__"))
    cfg = base / f"{runner}_db"
    args = ("eval", "-c", "promptfooconfig.yaml", "--no-cache", "--no-table", *extra)
    _ok(RUNNERS[runner](cfg, *args, cwd=work))
    return cfg


@pytest.fixture(scope="module")
def evals(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("blob_evals")
    return {runner: _run_eval(runner, base) for runner in RUNNERS}


def test_eval_writes_same_blob_files_as_node(evals):
    py_files, node_files = _blob_files(evals["py"]), _blob_files(evals["node"])
    assert py_files == node_files
    # PNG, WAV and the raw SVG preview; the sub-1KB data URL stays inline.
    assert sorted(p for p in py_files if not p.endswith(".meta.json")) == [
        "26/26/2626d48fa51a6675da8dad2cfa1230ca0dfc3496127c53492c81fd1877edcfbf",
        "78/83/7883b5acf7f684e4ebd67ce617763dca44ad7b69329ae2d57583f5bfefcde41a",
        "c9/3d/c93d3514feb5f23e7d77daed8f603228a9462e261f2a408d61e541efa42672a0",
    ]


def test_eval_writes_same_blob_rows_as_node(evals):
    py_rows, node_rows = _blob_rows(evals["py"]), _blob_rows(evals["node"])
    assert py_rows == node_rows
    assert len(py_rows["references"]) == 10


def test_eval_persists_same_responses_as_node(evals):
    assert _responses(evals["py"]) == _responses(evals["node"])


def _normalized_export(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["metadata"].pop("exportedAt", None)
    data["metadata"].pop("nodeVersion", None)
    for result in data["results"]["results"]:
        result.pop("latencyMs", None)
        (result.get("response") or {}).pop("latencyMs", None)
    return data


@pytest.mark.parametrize("writer", ["py", "node"])
def test_include_media_export_matches_across_runtimes(evals, tmp_path, writer):
    """Either exporter embeds the same `blobAssets` for an eval either runtime wrote."""
    cfg = evals[writer]
    a, b = tmp_path / "node.json", tmp_path / "py.json"
    _ok(node(cfg, "export", "eval", "latest", "--include-media", "-o", str(a)))
    _ok(py(cfg, "export", "eval", "latest", "--include-media", "-o", str(b)))
    node_export, py_export = _normalized_export(a), _normalized_export(b)
    assert py_export == node_export
    assets = py_export["blobAssets"]
    assert [(x["hash"][:8], x["mimeType"], x["sizeBytes"]) for x in assets] == [
        ("2626d48f", "image/png", 3172),
        ("7883b5ac", "audio/wav", 4044),
        ("c93d3514", "image/svg+xml", 61),
    ]


def test_stdout_include_media_export_matches_node(evals, tmp_path):
    cfg = evals["py"]
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(_ok(node(cfg, "export", "eval", "latest", "--include-media")).stdout)
    b.write_text(_ok(py(cfg, "export", "eval", "latest", "--include-media")).stdout)
    assert _normalized_export(a) == _normalized_export(b)


@pytest.mark.parametrize(("exporter", "importer"), [("py", "node"), ("node", "py"),
                                                    ("py", "py"), ("node", "node")])  # fmt: skip
def test_include_media_import_restores_blobs(evals, tmp_path, exporter, importer):
    """An --include-media export from one runtime imports into the other with the same blob
    files and rows as a same-runtime import, and re-exports the same media."""
    source = tmp_path / "source.json"
    _ok(RUNNERS[exporter](evals[exporter], "export", "eval", "latest", "--include-media", "-o",
                          str(source)))  # fmt: skip
    target = tmp_path / "target"
    _ok(RUNNERS[importer](target, "import", str(source)))
    reference = tmp_path / "reference"
    _ok(node(reference, "import", str(source)))
    assert _blob_files(target) == _blob_files(reference)
    assert _blob_rows(target) == _blob_rows(reference)
    rows = _blob_rows(target)
    assert {r[3] for r in rows["references"]} == {"import"}
    again = tmp_path / "again.json"
    _ok(py(target, "export", "eval", "latest", "--include-media", "-o", str(again)))
    exported = json.loads(source.read_text())["blobAssets"]
    # Imports keep only inert MIME types, so the SVG preview comes back as octet-stream.
    sanitized = [
        {**a, "mimeType": "application/octet-stream"} if a["mimeType"] == "image/svg+xml" else a
        for a in exported
    ]
    assert json.loads(again.read_text())["blobAssets"] == sanitized


def test_inline_media_env_keeps_media_inline_like_node(tmp_path, monkeypatch):
    monkeypatch.setitem(EXTRA_ENV, "PROMPTFOO_INLINE_MEDIA", "true")
    cfgs = {runner: _run_eval(runner, tmp_path) for runner in RUNNERS}
    for cfg in cfgs.values():
        assert not (cfg / "blobs").exists() or not any((cfg / "blobs").rglob("*"))
        assert _blob_rows(cfg) == {"assets": [], "references": []}
    py_responses, node_responses = _responses(cfgs["py"]), _responses(cfgs["node"])
    assert py_responses == node_responses
    assert py_responses[(0, 0)]["output"].startswith("data:image/png;base64,")
