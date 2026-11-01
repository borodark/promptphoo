"""Generate docs/PORT_MAP.md: which promptphoo module ports each upstream source file.

Usage: `python tools/port_map.py [--upstream ../promptfoo] [--check]`

The source of truth is the module docstrings: a module names the upstream files it ports there
("Ports src/util/transform.ts."). This script lists upstream's `src/` at the ported tag (D2),
read from git so the checkout's working tree does not matter, and sorts every file into:

- **named**: some module's docstring names it. The tests column lists the test files that import
  that module.
- **out of scope**: no module names it and it sits under an area a decision rules out.
- **not named**: neither. This is a mix of types-only files, helpers folded into another module,
  and real gaps; docs/SCOPE.md is the inventory of features.

It also lists the modules whose docstrings name no upstream file. A docstring that names a file
which does not exist at the tag is an error (exit 1). With `--check`, exit 1 when
docs/PORT_MAP.md differs from what would be generated.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "PORT_MAP.md"
TAG = "0.125.1"
UPSTREAM_PATH = r"\bsrc/(?:[A-Za-z0-9_.*/-]|\{[^{}]*\})+\.tsx?\b"
TOKEN = re.compile(rf"({UPSTREAM_PATH})|(?<![/\w.])([A-Za-z]\w*\.tsx?)\b")

# (prefix, reason). Checked in order; a file that a module names is never out of scope.
OUT_OF_SCOPE = [
    ("app/", "web UI (D1)"),
    ("server/", "web UI server (D1)"),
    ("commands/view.ts", "web UI (D1)"),
    ("commands/evalSetup.ts", "web UI (D1)"),
    ("redteam/", "redteam (D1)"),
    ("commands/redteam/", "redteam (D1)"),
    ("codeScan/", "code-scan (D1)"),
    ("optimizer/", "optimizer (D1)"),
    ("commands/optimize.ts", "optimizer (D1)"),
    ("commands/mcp/", "MCP (D1)"),
    ("commands/modelScan.ts", "model-audit (D1)"),
    ("commands/auth.ts", "cloud account (D44)"),
    ("assertions/redteam.ts", "redteam (D1)"),
    ("assertions/javascript.ts", "JavaScript hooks (D3)"),
    ("prompts/processors/javascript.ts", "JavaScript hooks (D3)"),
    ("assertions/ruby.ts", "Ruby hooks (D3)"),
    ("ruby/", "Ruby hooks (D3)"),
    ("golang/", "Go hooks (D3)"),
]


def upstream_files(upstream: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(upstream), "ls-tree", "-r", "--name-only", TAG, "src"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return sorted(
        p
        for p in out
        if p.endswith((".ts", ".tsx"))
        and not p.endswith(".d.ts")
        and not re.search(r"\.test\.|__tests__/|__mocks__/", p)
    )


def expand(path: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", path)
    if not m:
        return [path]
    return [
        e
        for alt in m.group(1).split(",")
        for e in expand(path[: m.start()] + alt + path[m.end() :])
    ]


def glob(path: str, upstream: set[str]) -> list[str]:
    """`src/prompts/processors/*.ts` -> the upstream files it matches, minus out-of-scope ones
    (a module that ports a folder by glob does not port the JS processor in it). A glob that
    matches nothing is kept as is, so that it is reported."""
    if "*" not in path:
        return [path]
    pat = re.compile("".join("[^/]*" if c == "*" else re.escape(c) for c in path) + "$")
    return sorted(f for f in upstream if pat.match(f) and not scope(f)) or [path]


def module_name(path: Path) -> str:
    rel = path.relative_to(ROOT / "src").with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def named_by_modules(upstream: set[str]) -> tuple[dict[str, set[str]], list[str]]:
    """Upstream path -> modules whose docstring names it; and modules that name none."""
    named: dict[str, set[str]] = defaultdict(set)
    silent = []
    for path in sorted((ROOT / "src" / "promptphoo").rglob("*.py")):
        if "_data" in path.parts:
            continue
        doc = ast.get_docstring(ast.parse(path.read_text())) or ""
        # A brace list ("{equals,contains}.ts") may wrap across lines; join it, then expand it.
        doc = re.sub(r"\{[^{}]*\}", lambda m: re.sub(r"\s+", "", m.group()), doc)
        paths, folder = set(), None
        for m in TOKEN.finditer(doc):
            if m.group(1):
                found = [g for e in expand(m.group(1)) for g in glob(e, upstream)]
                paths.update(found)
                folder = found[-1].rsplit("/", 1)[0]
            elif folder and f"{folder}/{m.group(2)}" in upstream:
                # "Ports src/assertions/bleu.ts, gleu.ts and rouge.ts": a bare name continues
                # the folder of the full path before it.
                paths.add(f"{folder}/{m.group(2)}")
        for p in paths:
            named[p].add(module_name(path))
        if not paths and path.stat().st_size:
            silent.append(module_name(path))
    return named, silent


def test_imports() -> dict[str, set[str]]:
    """Module -> test files that import it (directly or as `from parent import module`)."""
    out: dict[str, set[str]] = defaultdict(set)
    for path in sorted((ROOT / "tests").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for a in node.names:
                    out[a.name].add(rel)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                out[node.module].add(rel)
                for a in node.names:
                    out[f"{node.module}.{a.name}"].add(rel)
    return out


def group(path: str) -> str:
    rest = path[len("src/") :]
    return rest.split("/", 1)[0] + "/" if "/" in rest else "(top level)"


def scope(path: str) -> str | None:
    rest = path[len("src/") :]
    return next((why for prefix, why in OUT_OF_SCOPE if rest.startswith(prefix)), None)


def render(
    files: list[str], named: dict[str, set[str]], silent: list[str], tests: dict[str, set[str]]
) -> str:
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        status = "named" if f in named else ("out" if scope(f) else "not")
        groups[group(f)][status].append(f)
    order = sorted(groups, key=lambda g: (g == "(top level)", g))
    total = {s: sum(len(groups[g][s]) for g in groups) for s in ("named", "out", "not")}

    md = [
        "# Port map",
        "",
        f"Which promptphoo module ports each file of upstream promptfoo's `src/` at {TAG} (D2).",
        "Generated by `tools/port_map.py` from the module docstrings; do not edit by hand.",
        "Regenerate after adding or renaming a module, and name the upstream files a new module",
        "ports in its docstring. Features, rather than files, are inventoried in docs/SCOPE.md.",
        "",
        '- **Named**: a module\'s docstring names the file. "Imported by" lists the test files',
        "  that import that module; a dash means its tests reach it through another module",
        "  (most assertions run through the dispatcher), not that it is untested.",
        "- **Out of scope**: no module names it, and a decision rules out its area.",
        "- **Not named**: neither. Many are types-only files or helpers folded into another",
        "  module; some are real gaps.",
        "",
        "| Upstream area | Files | Named | Out of scope | Not named |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in order:
        c = groups[g]
        md.append(
            f"| `{g}` | {sum(len(v) for v in c.values())} | {len(c['named'])} | "
            f"{len(c['out'])} | {len(c['not'])} |"
        )
    md += [
        f"| **Total** | **{len(files)}** | **{total['named']}** | **{total['out']}** | "
        f"**{total['not']}** |",
        "",
        "## Named",
        "",
    ]
    for g in order:
        if not groups[g]["named"]:
            continue
        md += [f"### `{g}`", "", "| Upstream | promptphoo | Imported by |", "|---|---|---|"]
        for f in groups[g]["named"]:
            mods = sorted(named[f])
            ts = sorted({t for m in mods for t in tests.get(m, ())})
            md.append(
                f"| `{f}` | "
                + ", ".join(f"`{m}`" for m in mods)
                + " | "
                + (", ".join(f"`{Path(t).name}`" for t in ts) or "-")
                + " |"
            )
        md.append("")
    md += ["## Out of scope", "", "| Area | Reason | Files |", "|---|---|---:|"]
    reasons: dict[tuple[str, str], int] = defaultdict(int)
    for f in files:
        if f not in named and (why := scope(f)):
            prefix = next(p for p, w in OUT_OF_SCOPE if f[len("src/") :].startswith(p))
            reasons[(prefix, why)] += 1
    for (prefix, why), n in sorted(reasons.items()):
        md.append(f"| `src/{prefix}` | {why} | {n} |")
    md += ["", "## Not named by any module", ""]
    for g in order:
        if groups[g]["not"]:
            md.append(
                f"- `{g}` ({len(groups[g]['not'])}): "
                + ", ".join(f"`{f[len('src/') :]}`" for f in groups[g]["not"])
            )
    md += [
        "",
        "## Modules that name no upstream file",
        "",
        "promptphoo-only code (JS semantics, Jinja-as-Nunjucks, the in-house tracer) and",
        "modules whose upstream counterpart is named elsewhere, such as in a package",
        "`__init__`.",
        "",
        ", ".join(f"`{m}`" for m in silent),
        "",
    ]
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--upstream", default=str(ROOT.parent / "promptfoo"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    files = upstream_files(Path(args.upstream))
    named, silent = named_by_modules(set(files))
    missing = sorted(set(named) - set(files))
    for p in missing:
        print(
            f"{p}: named by {', '.join(sorted(named[p]))}, but not in upstream {TAG}",
            file=sys.stderr,
        )
    text = render(files, named, silent, test_imports())
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is out of date; run tools/port_map.py", file=sys.stderr)
            return 1
    else:
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(ROOT)}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
