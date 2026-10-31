"""Exact Node vs promptphoo comparison of a replay (phase 1) run, row by row.

    python tools/live/compare_replay.py OUT/p1 promptfooconfig features

Both runs saw the same model responses, so every field should match. Error texts are compared on
their first line, since a JS stack and a Python traceback differ below it.
"""

import json
import sys
from pathlib import Path


def rows(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text())
    out = {}
    for r in data["results"]["results"]:
        key = (r["provider"].get("label") or r["provider"]["id"], r["promptIdx"], r["testIdx"])
        resp = r.get("response") or {}
        comps = (r.get("gradingResult") or {}).get("componentResults", [])
        out[key] = {
            "success": r["success"],
            "score": r.get("score"),
            "error": (r.get("error") or "").split("\n")[0].removeprefix("Error: "),
            "output": resp.get("output"),
            "cost": r.get("cost"),
            "tokens": resp.get("tokenUsage"),
            "finish": resp.get("finishReason"),
            "asserts": [
                (c["assertion"]["type"], c["pass"], c.get("score"), c.get("reason")) for c in comps
            ],
            "named": r.get("namedScores"),
        }
    return out, data["results"]["stats"]


def tokens(stats: dict) -> tuple:
    usage = stats["tokenUsage"]
    return usage["total"], (usage.get("assertions") or {}).get("total")


def main(base: Path, configs: list[str]) -> None:
    for name in configs:
        node, ns = rows(base / "node" / f"{name}.json")
        py, ps = rows(base / "py" / f"{name}.json")
        print(
            f"== {name}: rows {len(node)}/{len(py)}  pass {ns['successes']}/{ps['successes']}"
            f"  fail {ns['failures']}/{ps['failures']}  err {ns['errors']}/{ps['errors']}"
            f"  tokens (target, grading) {tokens(ns)}/{tokens(ps)}  [node/py]"
        )
        for key in sorted(node):
            a, b = node[key], py.get(key) or {}
            for field in a:
                if a[field] != b.get(field):
                    print("   DIFF", key, field)
                    print("     node:", repr(a[field])[:400])
                    print("     py:  ", repr(b.get(field))[:400])
        same = sum(node[k] == py.get(k) for k in node)
        print(f"   identical rows: {same} of {len(node)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
