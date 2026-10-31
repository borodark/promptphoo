"""The OTLP receiver against Node's: the same traced eval, whose provider exports OTLP/JSON and
OTLP/protobuf spans to the eval's receiver, stores the same rows under both CLIs, and Node's
`export` of the Python-written eval equals Python's.

Skipped unless NODE_PROMPTFOO names a Node promptfoo command (as tools/parity.py uses). Each
CLI gets its own PROMPTFOO_CONFIG_DIR and a free localhost port for its receiver.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

NODE_PROMPTFOO = os.environ.get("NODE_PROMPTFOO")
pytestmark = pytest.mark.skipif(not NODE_PROMPTFOO, reason="NODE_PROMPTFOO is not set")

ROOT = Path(__file__).resolve().parent.parent
PROVIDER = Path(__file__).resolve().parent / "fixtures" / "tracing" / "otlp_posting_provider.py"

CONFIG = """
description: otlp receiver interop
tracing:
  enabled: true
  otlp:
    http:
      enabled: true
      host: 127.0.0.1
      port: PORT
      redactAttributes: [secret]
prompts:
  - "Trace {{idx}}"
providers:
  - id: file://prov.py
    label: poster
    config:
      receiver: http://127.0.0.1:PORT
tests:
  - vars: {idx: 0}
  - vars: {idx: 1}
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env(config_dir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PROMPTFOO_", "OTEL_"))}
    env.update(
        PROMPTFOO_CONFIG_DIR=str(config_dir),
        PROMPTFOO_DISABLE_TELEMETRY="1",
        PROMPTFOO_DISABLE_UPDATE="1",
        PYTHONPATH=str(ROOT / "src"),
    )
    return env


def _cli(runtime: str) -> list[str]:
    assert NODE_PROMPTFOO
    return shlex.split(NODE_PROMPTFOO) if runtime == "node" else [sys.executable, "-m",
                                                                  "promptphoo"]  # fmt: skip


def _run(runtime: str, args: list[str], cwd: Path, config_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run([*_cli(runtime), *args], cwd=cwd, env=_env(config_dir),
                          capture_output=True, text=True, timeout=300, check=False)  # fmt: skip


@pytest.fixture(scope="module")
def evals(tmp_path_factory) -> dict[str, Path]:
    out = {}
    for runtime in ("node", "py"):
        work = tmp_path_factory.mktemp(f"{runtime}_work")
        port = _free_port()
        (work / "cfg.yaml").write_text(CONFIG.replace("PORT", str(port)))
        shutil.copy(PROVIDER, work / "prov.py")
        config_dir = work / "home"
        result = _run(runtime, ["eval", "-c", "cfg.yaml", "--no-cache", "--no-table"], work,
                      config_dir)  # fmt: skip
        assert (config_dir / "promptfoo.db").exists(), result.stdout + result.stderr
        out[runtime] = config_dir
    return out


def _rows(config_dir: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(config_dir / "promptfoo.db")
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _posted_spans(config_dir: Path) -> list[dict[str, Any]]:
    """Spans the provider exported (service.name posting-provider), keyed by test case, with
    the per-run trace and root-span ids replaced."""
    traces = dict(_rows(config_dir, "SELECT trace_id, test_case_id FROM traces"))
    rows = _rows(
        config_dir,
        "SELECT trace_id, span_id, parent_span_id, name, start_time, "
        "end_time, attributes, status_code, status_message FROM spans",
    )
    posted_ids = {
        (r[0], r[1])
        for r in rows
        if json.loads(r[6] or "{}").get("service.name") == "posting-provider"
    }
    out = []
    for trace_id, span_id, parent, name, start, end, attrs, code, message in rows:
        if (trace_id, span_id) not in posted_ids:
            continue
        posted_parent = (trace_id, parent) in posted_ids
        out.append({
            "testCase": traces.get(trace_id), "spanId": span_id,
            "parent": parent if posted_parent else "<test case span>",
            "name": name, "startTime": start, "endTime": end,
            "attributes": json.loads(attrs) if attrs else None,
            "statusCode": code, "statusMessage": message,
        })  # fmt: skip
    return sorted(out, key=lambda s: (s["testCase"] or "", s["spanId"]))


def test_provider_posted_spans_are_stored_like_node(evals):
    node, py = _posted_spans(evals["node"]), _posted_spans(evals["py"])
    assert len(node) == 6, "Node stored no posted spans"
    assert py == node
    llm = next(s for s in py if s["name"] == "llm.chat")
    assert llm["attributes"]["api.secret_key"] == "[REDACTED]"
    assert llm["attributes"]["temperature"] == 0.25
    agent = next(s for s in py if s["name"] == "agent.run")
    assert agent["attributes"]["config"] == {"depth": 1.5, "client_secret": "[REDACTED]"}


def test_trace_rows_match_node(evals):
    sql = "SELECT test_case_id, metadata FROM traces ORDER BY test_case_id"
    assert _rows(evals["py"], sql) == _rows(evals["node"], sql)


def test_provider_saw_both_requests_accepted(evals):
    for config_dir in evals.values():
        outputs = _rows(config_dir, "SELECT response FROM eval_results")
        for (response,) in outputs:
            assert json.loads(json.loads(response)["output"]) == [
                [200, '{"partialSuccess":{}}'], [200, '{"partialSuccess":{}}'],
            ]  # fmt: skip


def _export(runtime: str, config_dir: Path, out: Path) -> dict[str, Any]:
    result = _run(runtime, ["export", "eval", "latest", "-o", str(out)], config_dir, config_dir)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    data["metadata"].pop("exportedAt", None)
    data["metadata"].pop("nodeVersion", None)
    return data


@pytest.mark.parametrize("writer", ["py", "node"])
def test_export_of_a_traced_eval_matches_node(evals, tmp_path, writer):
    node_export = _export("node", evals[writer], tmp_path / "node.json")
    py_export = _export("py", evals[writer], tmp_path / "py.json")
    assert node_export["traces"], "Node exported no traces"
    assert py_export["traces"] == node_export["traces"]
    assert py_export == node_export


def test_export_import_round_trip_keeps_traces(evals, tmp_path):
    exported = tmp_path / "export.json"
    original = _export("py", evals["py"], exported)
    for runtime in ("py", "node"):
        home = tmp_path / f"{runtime}_import"
        result = _run(runtime, ["import", str(exported)], tmp_path, home)
        assert result.returncode == 0, result.stdout + result.stderr
        again = _export("py", home, tmp_path / f"{runtime}_again.json")
        assert again["traces"] == original["traces"]
