"""Summarize a live (phase 2) run: per-config pass/fail/error counts, cost and tokens, Node vs
promptphoo, and the rows whose outcome or structure differs. Model outputs vary between live
calls, so texts are not compared; see compare_replay.py for exact comparison.

    python tools/live/compare_live.py OUT/p2
"""

import json
import sys
from pathlib import Path

CONFIGS = [
    "matrix", "features", "sonnet-5", "sonnet-5-5", "structured", "thinking",
    "unknown-model", "bad-model", "bad-key",
]  # fmt: skip


def load(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text())
    rows = {}
    for r in data["results"]["results"]:
        key = (r["provider"].get("label") or r["provider"]["id"], r["promptIdx"], r["testIdx"])
        resp = r.get("response") or {}
        comps = (r.get("gradingResult") or {}).get("componentResults", [])
        rows[key] = {
            "ok": r["success"],
            "finish": resp.get("finishReason"),
            "error": (r.get("error") or "").split("\n")[0][:110],
            "cost": r.get("cost"),
            "asserts": [(c["assertion"]["type"], c["pass"]) for c in comps],
        }
    return rows, data["results"]["stats"]


def main(base: Path) -> None:
    for name in CONFIGS:
        node, ns = load(base / "node" / f"{name}.json")
        py, ps = load(base / "py" / f"{name}.json")
        cost = [sum(r["cost"] or 0 for r in rows.values()) for rows in (node, py)]
        print(
            f"{name:13} pass {ns['successes']}/{ps['successes']} fail {ns['failures']}/"
            f"{ps['failures']} err {ns['errors']}/{ps['errors']}"
            f"  cost ${cost[0]:.4f}/${cost[1]:.4f}"
            f"  tokens {ns['tokenUsage']['total']}/{ps['tokenUsage']['total']}  [node/py]"
        )
        for key in sorted(node):
            a, b = node[key], py.get(key, {})
            diff = {
                f: (a[f], b.get(f))
                for f in ("ok", "finish", "error", "asserts")
                if a[f] != b.get(f)
            }
            if diff:
                print("   ", key, diff)
    for i in (1, 2):
        usage = json.loads((base / "py" / f"cache{i}.json").read_text())["results"]["stats"][
            "tokenUsage"
        ]
        print(f"cache run {i}: total={usage.get('total')} cached={usage.get('cached')}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
