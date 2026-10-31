#!/usr/bin/env python3
"""Parity harness: run a config through Node promptfoo and promptphoo, diff normalized JSON output.

Usage:
  tools/parity.py <config> [<config> ...]
  tools/parity.py --all            # every fixture in the upstream smoke-config dir

Env:
  PROMPTFOO_SRC    path to the upstream repo (default: ../promptfoo next to this repo)
  NODE_PROMPTFOO   command for the reference CLI (default: "npx -y promptfoo@<UPSTREAM_VERSION>")
  PY_PROMPTFOO     command for the port (default: "<this venv python> -m promptphoo")
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from promptphoo import UPSTREAM_VERSION  # noqa: E402

UPSTREAM = Path(os.environ.get("PROMPTFOO_SRC", ROOT.parent / "promptfoo")).resolve()
CONFIGS_DIR = UPSTREAM / "test" / "smoke" / "fixtures" / "configs"
WORK_DIR = ROOT / ".parity"

NODE_CMD = shlex.split(os.environ.get("NODE_PROMPTFOO", f"npx -y promptfoo@{UPSTREAM_VERSION}"))
PY_CMD = shlex.split(os.environ.get("PY_PROMPTFOO", f"{sys.executable} -m promptphoo"))

# Values that legitimately differ between any two runs, wherever they appear.
VOLATILE_KEYS = {
    "id",
    "evalId",
    "timestamp",
    "createdAt",
    "updatedAt",
    "latencyMs",
    "durationMs",
    "generationDurationMs",
    "evaluationDurationMs",
    "exportedAt",
    "evaluationCreatedAt",
}
# Top-level keys describing the host / run rather than the results.
VOLATILE_TOP = {"metadata", "shareableUrl"}

# Configs that rely on hooks this port deliberately drops (JS/TS/Ruby/Go).
UNSUPPORTED = re.compile(
    r"type:\s*['\"]?(not-)?(javascript|ruby)\b"
    r"|file://[^\s'\"]+\.(js|cjs|mjs|ts|rb|go)\b"
    r"|\b(ruby|golang):"
    r"|\bredteam:",
)
UNSUPPORTED_SUFFIXES = {".js", ".cjs", ".mjs", ".ts"}


def normalize(obj, top=True):
    if isinstance(obj, dict):
        return {
            k: normalize(v, False)
            for k, v in sorted(obj.items())
            if k not in VOLATILE_KEYS and not (top and k in VOLATILE_TOP)
        }
    if isinstance(obj, list):
        return [normalize(v, False) for v in obj]
    return obj


def run(cmd: list[str], config: Path, out: Path, config_dir: Path) -> tuple[int, str]:
    env = {
        **os.environ,
        "NO_COLOR": "1",
        "PROMPTFOO_CONFIG_DIR": str(config_dir),
        "PROMPTFOO_DISABLE_TELEMETRY": "1",
        "PROMPTFOO_DISABLE_UPDATE": "1",
    }
    args = [*cmd, "eval", "-c", str(config), "-o", str(out), "--no-cache", "--no-progress-bar"]
    proc = subprocess.run(args, cwd=UPSTREAM, env=env, capture_output=True, text=True, timeout=300)
    return proc.returncode, proc.stdout + proc.stderr


def compare(config: Path) -> str:
    """Return one of: PASS, FAIL, SKIP, ERROR (with details printed)."""
    text = config.read_text(errors="replace")
    if config.suffix in UNSUPPORTED_SUFFIXES or UNSUPPORTED.search(text):
        return "SKIP"

    case_dir = WORK_DIR / config.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, cmd in (("node", NODE_CMD), ("py", PY_CMD)):
        out = case_dir / f"{name}.json"
        out.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"pf-{name}-") as cfg_dir:
            code, log = run(cmd, config, out, Path(cfg_dir))
        (case_dir / f"{name}.log").write_text(log)
        data = json.loads(out.read_text()) if out.exists() else None
        results[name] = (code, data)

    (node_code, node_data), (py_code, py_data) = results["node"], results["py"]
    if node_data is None:
        print(f"  node produced no output (exit {node_code}); see {case_dir / 'node.log'}")
        return "ERROR"
    if py_data is None:
        print(f"  py produced no output (exit {py_code}); see {case_dir / 'py.log'}")
        return "FAIL"

    a = json.dumps(normalize(node_data), indent=2, sort_keys=True).splitlines()
    b = json.dumps(normalize(py_data), indent=2, sort_keys=True).splitlines()
    if a == b and node_code == py_code:
        return "PASS"
    if node_code != py_code:
        print(f"  exit code differs: node={node_code} py={py_code}")
    diff = list(difflib.unified_diff(a, b, "node", "py", lineterm="", n=2))
    (case_dir / "diff.txt").write_text("\n".join(diff))
    for line in diff[:40]:
        print("  " + line)
    if len(diff) > 40:
        print(f"  ... {len(diff) - 40} more lines in {case_dir / 'diff.txt'}")
    return "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", help="run every upstream smoke fixture")
    args = parser.parse_args()

    configs = [p.resolve() for p in args.configs]
    if args.all:
        configs += sorted(p for p in CONFIGS_DIR.iterdir() if p.suffix in {".yaml", ".yml"})
    if not configs:
        parser.error("give config paths or --all")

    tally: dict[str, list[str]] = {}
    for config in configs:
        print(f"{config.name}")
        status = compare(config)
        print(f"  -> {status}")
        tally.setdefault(status, []).append(config.name)

    print("\nSummary: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(tally.items())))
    return 1 if tally.get("FAIL") or tally.get("ERROR") else 0


if __name__ == "__main__":
    sys.exit(main())
