"""Node <-> Python interop for `export` and `import`. Skipped unless NODE_PROMPTFOO is set, e.g.
`NODE_PROMPTFOO="node ../promptfoo/dist/src/main.js"` (promptfoo 0.125.1).

Each test runs both CLIs as subprocesses against throwaway `PROMPTFOO_CONFIG_DIR`s. The only
nondeterministic parts of an export of one DB eval are `metadata.exportedAt` (wall clock),
`metadata.nodeVersion` (`v22.x` vs `python-3.x`) and, once an export has been re-imported, each
result's `id` (import mints fresh eval_results uuids, in Node too). `_normalize` drops exactly
those; everything else must match byte for byte.
"""

from __future__ import annotations

import json
import os
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

NODE = os.environ.get("NODE_PROMPTFOO")
pytestmark = pytest.mark.skipif(not NODE, reason="NODE_PROMPTFOO is not set")

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "import_export"

CONFIG = """\
description: interop
prompts:
  - "Hello {{name}}"
  - "Bye {{name}}"
providers:
  - echo
tags:
  team: alpha
tests:
  - vars: {name: Ann}
    assert:
      - type: contains
        value: Ann
  - vars: {name: Bob}
    assert:
      - type: equals
        value: nope
"""


def _env(config_dir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROMPTFOO_")}
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


def _normalize(path: Path, *, result_ids: bool = True) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["metadata"].pop("exportedAt", None)
    data["metadata"].pop("nodeVersion", None)
    if result_ids:
        for result in data["results"].get("results") or []:
            result["id"] = "<uuid>"
    return json.dumps(data, indent=2, ensure_ascii=False)


_IGNORED = {
    "eval_results": {"id", "created_at", "updated_at"},
    "prompts": {"created_at"},
    "datasets": {"created_at"},
    "traces": {"id", "created_at"},
    "spans": {"id"},
}
_TABLES = ("evals", "eval_results", "prompts", "evals_to_prompts", "datasets",
           "evals_to_datasets", "tags", "evals_to_tags", "traces", "spans")  # fmt: skip


def _dump_db(config_dir: Path, ignore_eval_created_at: bool = False) -> dict[str, list[str]]:
    conn = sqlite3.connect(config_dir / "promptfoo.db")
    out = {}
    for table in _TABLES:
        ignored = set(_IGNORED.get(table, set()))
        if table == "evals" and ignore_eval_created_at:
            ignored.add("created_at")
        cols = [r[1] for r in conn.execute(f"pragma table_info({table})") if r[1] not in ignored]
        out[table] = sorted(repr(r) for r in conn.execute(f"select {','.join(cols)} from {table}"))
    conn.close()
    return out


@pytest.fixture(scope="module")
def node_eval(tmp_path_factory) -> tuple[Path, Path]:
    """A Node-run eval in its own DB, plus Node's export of it."""
    base = tmp_path_factory.mktemp("node_eval")
    (base / "cfg.yaml").write_text(CONFIG, encoding="utf-8")
    cfg = base / "db"
    run = node(cfg, "eval", "-c", str(base / "cfg.yaml"), "--no-cache", "--no-table", cwd=base)
    assert run.returncode == 100, run.stdout + run.stderr  # one test fails by design
    export = base / "node_export.json"
    _ok(node(cfg, "export", "eval", "latest", "-o", str(export)))
    return cfg, export


def test_python_export_matches_node_export_of_same_db_eval(node_eval, tmp_path):
    """Round trip 3: same DB eval, both exporters; only exportedAt/nodeVersion differ."""
    cfg, node_export = node_eval
    py_export = tmp_path / "py_export.json"
    _ok(py(cfg, "export", "eval", "latest", "-o", str(py_export)))
    assert _normalize(py_export, result_ids=False) == _normalize(node_export, result_ids=False)
    a = json.loads(py_export.read_text())["metadata"]
    b = json.loads(node_export.read_text())["metadata"]
    assert {k for k in a if a[k] != b.get(k)} <= {"exportedAt", "nodeVersion"}


@pytest.mark.parametrize("fmt", ["csv", "xml", "jsonl", "junit.xml", "yaml", "html"])
def test_other_formats_match_node(node_eval, tmp_path, fmt):
    cfg, _ = node_eval
    a, b = tmp_path / f"node.{fmt}", tmp_path / f"py.{fmt}"
    _ok(node(cfg, "export", "eval", "latest", "-o", str(a)))
    _ok(py(cfg, "export", "eval", "latest", "-o", str(b)))
    drop = ("nodeVersion:", "exportedAt:", "<li>Exported ")
    lines_a = [ln for ln in a.read_text().splitlines() if not ln.strip().startswith(drop)]
    lines_b = [ln for ln in b.read_text().splitlines() if not ln.strip().startswith(drop)]
    assert lines_a == lines_b


def test_stdout_export_matches_node(node_eval, tmp_path):
    cfg, _ = node_eval
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(_ok(node(cfg, "export", "eval", "latest")).stdout)
    b.write_text(_ok(py(cfg, "export", "eval", "latest")).stdout)
    assert _normalize(a, result_ids=False) == _normalize(b, result_ids=False)


@pytest.mark.parametrize(("importer", "exporter"), [("py", "py"), ("py", "node"),
                                                    ("node", "py"), ("node", "node")])  # fmt: skip
def test_cross_import_round_trips(node_eval, tmp_path, importer, exporter):
    """Round trips 1 and 2: a Node export imports into Python and a Python export into Node;
    re-exporting with either side reproduces the original export."""
    cfg, node_export = node_eval
    source = tmp_path / "source.json"
    if importer == "node":  # Node imports what Python exported
        _ok(py(cfg, "export", "eval", "latest", "-o", str(source)))
    else:  # Python imports what Node exported
        source.write_text(node_export.read_text())
    target = tmp_path / "target"
    _ok(RUNNERS[importer](target, "import", str(source)))
    again = tmp_path / "again.json"
    _ok(RUNNERS[exporter](target, "export", "eval", "latest", "-o", str(again)))
    assert _normalize(again) == _normalize(node_export)


@pytest.mark.parametrize(
    ("fixture", "ignore_created"),
    [
        ("node-v4-export.json", False),
        ("sample-export.json", False),
        ("legacy-v2-export.json", False),
        ("v4-with-traces.json", False),
        ("openai-evals-items.jsonl", True),  # created_at is the wall clock at import
        ("openai-evals-multi-run-items.jsonl", True),
    ],
)
def test_same_file_imports_to_same_rows(tmp_path, fixture, ignore_created):
    """Node and Python write the same evals/results/prompts/datasets/tags/traces rows."""
    path = FIXTURES / fixture
    a, b = tmp_path / "py", tmp_path / "node"
    pa, pb = _ok(py(a, "import", str(path))), _ok(node(b, "import", str(path)))
    assert pa.stdout.strip().splitlines()[-1] == pb.stdout.strip().splitlines()[-1]
    assert _dump_db(a, ignore_created) == _dump_db(b, ignore_created)


def test_legacy_eval_exports_match(tmp_path):
    """Round trip 4: a v3-layout (table-backed) eval exports identically from both sides."""
    cfg = tmp_path / "db"
    _ok(node(cfg, "import", str(FIXTURES / "legacy-v2-export.json")))
    a, b = tmp_path / "node.json", tmp_path / "py.json"
    _ok(node(cfg, "export", "eval", "eval-legacy-table-backed", "-o", str(a)))
    _ok(py(cfg, "export", "eval", "eval-legacy-table-backed", "-o", str(b)))
    assert _normalize(a) == _normalize(b)
    target = tmp_path / "py_db"
    _ok(py(target, "import", str(b)))
    c = tmp_path / "node_again.json"
    _ok(node(target, "export", "eval", "eval-legacy-table-backed", "-o", str(c)))
    assert _normalize(c) == _normalize(a)


@pytest.mark.parametrize(
    ("args", "make"),
    [
        (["import", "{missing}"], None),
        (["import", "{file}"], "invalid json"),
        (["import", "{file}"], '{"foo":1}\n{"bar":2}\n'),
        (["import", "{file}"], '{"id":"x","config":{},"results":{"results":[]}}'),
        (["export", "eval", "nope"], None),
    ],
)
def test_error_messages_match_node(tmp_path, args, make):
    file = tmp_path / "input.json"
    if make is not None:
        file.write_text(make)
    argv = [a.format(file=file, missing=tmp_path / "missing.json") for a in args]
    outs = []
    for name, run in RUNNERS.items():
        proc = run(tmp_path / name, *argv)
        assert proc.returncode == 1
        outs.append((proc.stdout + proc.stderr).strip().splitlines()[0])
    assert outs[0] == outs[1]


def test_collision_messages_match_node(tmp_path):
    sample = str(FIXTURES / "sample-export.json")
    lines = {}
    for name, run in RUNNERS.items():
        cfg = tmp_path / name
        _ok(run(cfg, "import", sample))
        dup = run(cfg, "import", sample)
        assert dup.returncode == 1
        forced = _ok(run(cfg, "import", sample, "--force"))
        lines[name] = ((dup.stdout + dup.stderr).splitlines()[0], forced.stdout.strip())
    assert lines["node"] == lines["py"]
