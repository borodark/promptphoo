#!/usr/bin/env python3
"""Golden output for `list`, `show` and `delete`, recorded from real Node promptfoo.

Node evaluates three echo-provider evals into a fresh config dir and imports one legacy v3
eval (a v2 summary with a stored table). The tool snapshots the DB rows after each stage and
records Node's output for every command in the scenario. `tests/test_cli_list_show_delete.py`
loads the same rows into a migrated DB and requires promptphoo's output to match byte for byte.

Node's "There were some errors during the operation. See logs for more details." footer (with
its per-run log file paths) is cut from the goldens: promptphoo writes no log files (see D41).

Usage:
  NODE_PROMPTFOO="node ../promptfoo/dist/src/main.js" tools/list_show_golden.py
Writes tests/fixtures/list_show/golden.json.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "list_show" / "golden.json"
NODE_CMD = shlex.split(os.environ.get("NODE_PROMPTFOO", "npx -y promptfoo@0.125.1"))
FOOTER = "\n\nThere were some errors during the operation."

# Tables holding eval data, parents first so the rows load with foreign keys on.
TABLES = (
    "evals", "prompts", "datasets", "tags", "evals_to_prompts", "evals_to_datasets",
    "evals_to_tags", "eval_results", "traces", "spans", "blob_assets", "blob_references",
)  # fmt: skip

CONFIG_A = """\
description: First echo eval
prompts:
  - "Hello {{name}}"
  - "Bye {{name}} from {{place}}"
providers: [echo]
tests:
  - vars: {name: World, place: Paris}
    assert: [{type: contains, value: World}]
  - vars: {name: Bob, place: Rome}
    assert: [{type: equals, value: nope}]
"""
CONFIG_B = """\
description: A second eval with a description that is quite long, long enough to exceed one \
hundred characters in total length for sure
prompts: ["Summarize {{text}}"]
providers: [echo]
tests:
  - vars: {text: "alpha beta"}
"""

STAGE1 = [
    ("list", ["list", "evals"]),
    ("list_ids", ["list", "evals", "--ids-only"]),
    ("list_n1", ["list", "evals", "-n", "1"]),
    ("prompts", ["list", "prompts"]),
    ("prompts_ids", ["list", "prompts", "--ids-only"]),
    ("prompts_n1", ["list", "prompts", "-n", "1"]),
    ("datasets", ["list", "datasets"]),
    ("datasets_ids", ["list", "datasets", "--ids-only"]),
    ("show", ["show"]),
    ("show_eval", ["show", "eval"]),
    ("show_missing", ["show", "nope"]),
    ("show_prompt", ["show", "prompt", "652b7c"]),
    ("show_prompt_top", ["show", "652b7c"]),
    ("show_dataset", ["show", "dataset", "aef222"]),
    ("show_dataset_top", ["show", "aef222"]),
    ("show_prefix", ["show", "eval-"]),
    ("show_prompt_missing", ["show", "prompt", "zzz"]),
    ("show_dataset_missing", ["show", "dataset", "zzz"]),
    ("show_eval_missing", ["show", "eval", "zzz"]),
]
STAGE2 = [
    ("v3_list", ["list", "evals"]),
    ("v3_show", ["show", "eval", "eval-v3L-2020-01-02T03:04:05"]),
    ("v3_show_top", ["show", "eval-v3L-2020-01-02T03:04:05"]),
    ("v3_prompts", ["list", "prompts"]),
    ("v3_datasets", ["list", "datasets"]),
    ("v3_show_prompt", ["show", "prompt", "b13387"]),
]
# Run in order on one copy of the stage-2 DB.
DELETES = [
    ("del_missing", ["delete", "nope"]),
    ("del_prefix", ["delete", "eval-"]),
    ("del_eval_missing", ["delete", "eval", "nope"]),
    ("del_v3", ["delete", "eval", "eval-v3L-2020-01-02T03:04:05"]),
    ("del_latest", ["delete", "eval", "latest"]),
    ("del_after_list", ["list", "evals", "--ids-only"]),
]
EMPTY = [
    ("empty_list", ["list", "evals"]),
    ("empty_prompts", ["list", "prompts"]),
    ("empty_datasets", ["list", "datasets"]),
    ("empty_show", ["show"]),
    ("empty_show_eval", ["show", "eval"]),
    ("empty_del_latest", ["delete", "eval", "latest"]),
]


def legacy_export() -> dict[str, Any]:
    """A v2 summary with a 27-row table: imported, it becomes a legacy v3 `evals` row."""
    rows = []
    for i in range(27):
        ok = i % 3 != 0
        output = {
            "pass": ok,
            "score": 1 if ok else 0,
            "text": f"echo {i}" if ok else f"bad {i}---detail",
            "failureReason": 0 if ok else (1 if i % 2 else 2),
            "prompt": f"Legacy {i}",
            "latencyMs": 1,
            "cost": 0,
            "namedScores": {},
        }
        rows.append({"outputs": [output], "vars": [f"v{i}"], "test": {"vars": {"q": f"v{i}"}}})
    metrics = {
        "score": 18.125, "testPassCount": 18, "testFailCount": 5, "testErrorCount": 4,
        "assertPassCount": 18, "assertFailCount": 9, "totalLatencyMs": 27, "tokenUsage": {},
        "namedScores": {}, "namedScoresCount": {}, "cost": 0,
    }  # fmt: skip
    prompt = {
        "raw": "Legacy {{q}}",
        "label": "Legacy {{q}}",
        "provider": "echo",
        "id": "legacyprompt0001",
        "metrics": metrics,
    }
    return {
        "evalId": "eval-v3L-2020-01-02T03:04:05",
        "createdAt": "2020-01-02T03:04:05.000Z",
        "author": "legacy@example.com",
        "config": {
            "description": "Legacy v3 eval",
            "prompts": ["Legacy {{q}}"],
            "providers": ["echo"],
            "tests": [{"vars": {"q": f"v{i}"}} for i in range(27)],
        },
        "results": {
            "version": 2,
            "timestamp": "2020-01-02T03:04:05.000Z",
            "results": [],
            "stats": {"successes": 18, "failures": 9, "tokenUsage": {}},
            "table": {"head": {"prompts": [prompt], "vars": ["q"]}, "body": rows},
        },
    }


def strip_footer(text: str) -> str:
    return text.split(FOOTER)[0] + "\n" if FOOTER in text else text


def node(config_dir: Path, cwd: Path, *args: str) -> tuple[int, str]:
    env = {
        **os.environ,
        "PROMPTFOO_CONFIG_DIR": str(config_dir),
        "PROMPTFOO_DISABLE_TELEMETRY": "1",
        "PROMPTFOO_DISABLE_UPDATE": "1",
    }
    env.pop("FORCE_COLOR", None)
    proc = subprocess.run(
        [*NODE_CMD, *args], cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=300, stdin=subprocess.DEVNULL,
    )  # fmt: skip
    return proc.returncode, strip_footer(proc.stdout)


def dump_rows(config_dir: Path) -> dict[str, Any]:
    conn = sqlite3.connect(config_dir / "promptfoo.db")
    out = {}
    for table in TABLES:
        cur = conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid')
        rows = cur.fetchall()
        if rows:
            out[table] = {"columns": [c[0] for c in cur.description], "rows": rows}
    conn.close()
    return out


def table_counts(config_dir: Path) -> dict[str, int]:
    conn = sqlite3.connect(config_dir / "promptfoo.db")
    counts = {t: conn.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in TABLES}
    conn.close()
    return counts


def record(
    cases: list[tuple[str, list[str]]], stage: str, config_dir: Path, cwd: Path
) -> list[dict[str, Any]]:
    out = []
    for name, args in cases:
        code, output = node(config_dir, cwd, *args)
        out.append({"name": name, "stage": stage, "args": args, "exitCode": code, "output": output})
        print(f"{stage}/{name}: exit {code}", file=sys.stderr)
    return out


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "a.yaml").write_text(CONFIG_A)
        (work / "b.yaml").write_text(CONFIG_B)
        (work / "legacy.json").write_text(json.dumps(legacy_export(), indent=1))
        cases = record(EMPTY, "empty", work / "empty", work)
        cfg = work / "cfg"
        for config in ("a.yaml", "b.yaml", "a.yaml"):
            node(cfg, work, "eval", "-c", config, "--no-cache", "--no-progress-bar")
            time.sleep(1.1)  # distinct second-resolution ids and created_at order
        stage1 = dump_rows(cfg)
        cases += record(STAGE1, "stage1", cfg, work)
        code, output = node(cfg, work, "import", "legacy.json")
        assert code == 0, output
        stage2 = dump_rows(cfg)
        cases += record(STAGE2, "stage2", cfg, work)
        shutil.copytree(cfg, work / "del")
        cases += record(DELETES, "deletes", work / "del", work)
        golden = {
            "stages": {"stage1": stage1, "stage2": stage2},
            "cases": cases,
            "countsAfterDeletes": table_counts(work / "del"),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(golden, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
