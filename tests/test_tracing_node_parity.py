"""promptfoo's own spans: the same traced configs under Node and Python store the same spans.

Skipped unless NODE_PROMPTFOO names a Node promptfoo command (as tools/parity.py uses). Both
CLIs run against a canned local LLM server (tests/tracing_fake_llm.py) and an in-repo Python
provider, each with its own PROMPTFOO_CONFIG_DIR. The stored spans are compared with ids and
times dropped: span names, the parent chain, attributes (eval ids normalized) and status.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

NODE_PROMPTFOO = os.environ.get("NODE_PROMPTFOO")
pytestmark = pytest.mark.skipif(not NODE_PROMPTFOO, reason="NODE_PROMPTFOO is not set")

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

BASIC_CONFIG = """
description: span parity
tracing:
  enabled: true
prompts:
  - "Hello {{name}}"
providers:
  - id: echo
    label: echo-label
  - id: file://prov.py
tests:
  - vars: {name: World}
    assert:
      - type: contains
        value: World
      - type: llm-rubric
        value: says hello
        provider: echo
  - vars: {name: Bob}
    assert:
      - type: equals
        value: nope
"""

GENAI_CONFIG = """
description: genai span parity
tracing:
  enabled: true
prompts:
  - "Say hi to {{name}}"
providers:
  - id: openai:chat:gpt-4o-mini
    config: {apiBaseUrl: "BASE/v1", temperature: 0.5, max_tokens: 50}
  - id: openai:completion:gpt-3.5-turbo-instruct
    config: {apiBaseUrl: "BASE/v1"}
  - id: openai:responses:gpt-4.1
    config: {apiBaseUrl: "BASE/v1", max_output_tokens: 64}
  - id: anthropic:messages:claude-3-5-haiku-20241022
    config: {apiBaseUrl: "BASE", max_tokens: 100, temperature: 0.2}
  - id: ollama:completion:llama3
    config: {temperature: 0.1, num_predict: 20}
  - id: ollama:chat:llama3
    config: {top_p: 0.9, stop: ["END"]}
tests:
  - vars: {name: World}
    assert:
      - type: icontains
        value: hi
"""


@pytest.fixture(scope="module")
def llm_base_url():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tracing_fake_llm import serve

    server = serve(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _run(cmd: list[str], workdir: Path, config_dir: Path, base_url: str) -> None:
    env = {
        **os.environ,
        "PROMPTFOO_CONFIG_DIR": str(config_dir),
        "PROMPTFOO_DISABLE_TELEMETRY": "1",
        "PROMPTFOO_DISABLE_UPDATE": "1",
        "OPENAI_API_KEY": "sk-test",
        "ANTHROPIC_API_KEY": "test",
        "OLLAMA_BASE_URL": base_url,
        "PYTHONPATH": str(ROOT / "src"),
    }
    for key in ("PROMPTFOO_TRACING_ENABLED", "PROMPTFOO_OTEL_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT"):
        env.pop(key, None)
    subprocess.run(
        [*cmd, "eval", "-c", "cfg.yaml", "--no-cache", "--no-table"],
        cwd=workdir,
        env=env,
        capture_output=True,
        timeout=300,
        check=False,
    )


def _normalized_spans(db_path: Path) -> list[Any]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT trace_id, span_id, parent_span_id, name, attributes, status_code, "
        "status_message FROM spans"
    ).fetchall()
    conn.close()
    by_id = {(r[0], r[1]): r for r in rows}

    def path(row: Any) -> list[str]:
        names = [row[3]]
        while row[2] is not None:
            row = by_id[(row[0], row[2])]
            names.append(row[3])
        return names[::-1]

    def root_key(row: Any) -> list[Any]:
        while row[2] is not None:
            row = by_id[(row[0], row[2])]
        attrs = json.loads(row[4] or "{}")
        return [attrs.get("promptfoo.test_case.id"), attrs.get("promptfoo.provider.id")]

    out = []
    for row in rows:
        attrs = json.loads(row[4] or "{}")
        if "promptfoo.eval.id" in attrs:
            attrs["promptfoo.eval.id"] = "<eval>"
        out.append(
            json.dumps(
                {
                    "trace": root_key(row),
                    "path": path(row),
                    "attributes": attrs,
                    "statusCode": row[5],
                    "statusMessage": row[6],
                },
                sort_keys=True,
            )
        )
    return sorted(out)


@pytest.mark.parametrize("config", [BASIC_CONFIG, GENAI_CONFIG], ids=["basic", "genai"])
def test_stored_spans_match_node(tmp_path: Path, llm_base_url: str, config: str) -> None:
    assert NODE_PROMPTFOO
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "cfg.yaml").write_text(config.replace("BASE", llm_base_url))
    shutil.copy(FIXTURES / "tracing_span_provider.py", workdir / "prov.py")

    node_dir, py_dir = tmp_path / "node", tmp_path / "py"
    _run(shlex.split(NODE_PROMPTFOO), workdir, node_dir, llm_base_url)
    _run([sys.executable, "-m", "promptphoo"], workdir, py_dir, llm_base_url)

    node_spans = _normalized_spans(node_dir / "promptfoo.db")
    py_spans = _normalized_spans(py_dir / "promptfoo.db")
    assert node_spans, "Node stored no spans"
    assert py_spans == node_spans
