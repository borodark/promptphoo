"""Check the file paths and `D<n>` references written in docs, docstrings and comments.

Usage: `python tools/check_refs.py [--upstream ../promptfoo] [--all]`

What it scans:
- Markdown at the repo root, under `docs/` and `.claude/skills/`.
- Docstrings and `#` comments of every `.py` file under `src/`, `tests/` and `tools/`
  (vendored `_data/` excluded), and `//` comments of the `.js` / `.mjs` tools.

What it checks:
- Repo paths (`src/promptphoo/...`, `tests/...`, `tools/...`, `docs/...`, `examples/...`, or a
  package-relative `cli/eval.py`) must exist. A `*` is globbed, `{a,b}` is expanded.
- Upstream paths (`src/....ts`, `test/...`, `drizzle/...`) must exist in the upstream checkout,
  when one is found. Without it they are reported as unchecked.
- `D<n>` must name an entry of docs/DECISIONS.md that is not marked unused. With `--all`, every
  reference from a file the entry's `Modules:` line does not list is printed for review, since
  a wrong-but-existing number can only be judged by reading it.

Exit status is 1 when a reference is broken.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "promptphoo"

PATH_RE = re.compile(
    r"(?<![\w./@:-])"
    r"((?:\.\./promptfoo/)?(?:[\w.-]+/)*[\w.*{},-]+\."
    r"(?:py|ts|tsx|js|mjs|cjs|md|json|jsonl|yaml|yml|sh|sql|html|txt|toml))"
    r"(?![\w/-])"
)
DREF_RE = re.compile(r"(?<![\w-])D(\d{1,3})(?![\w])")
REPO_PREFIXES = ("src/promptphoo/", "tests/", "tools/", "docs/", "examples/", ".claude/")
UPSTREAM_PREFIXES = ("src/", "test/", "drizzle/", "examples/", "site/")
# Build output that exists only after `npm run build` upstream.
BUILD_PREFIXES = ("dist/", "../promptfoo/dist/")
ROOT_FILES = {"README.md", "NOTICE", "LICENSE", "FOREWORD.md", "CONTRIBUTING.md", "CHANGELOG.md"}
# Names that look like paths but are not references to a file in either tree.
IGNORE = re.compile(
    r"^(?:[\w-]+\.(?:json|yaml|yml|txt|md|html|csv|db|sql|jsonl)$"  # bare file names
    r"|.*\{\{.*|.*<.*|e\.g\..*|i\.e\..*|.*\.\.\..*|path/to/.*)"
)


def expand_braces(text: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", text)
    if not m:
        return [text]
    out: list[str] = []
    for part in m.group(1).split(","):
        out.extend(expand_braces(text[: m.start()] + part + text[m.end() :]))
    return out


def decision_index() -> tuple[dict[int, str], dict[int, set[str]]]:
    titles: dict[int, str] = {}
    modules: dict[int, set[str]] = {}
    current = None
    for line in (ROOT / "docs" / "DECISIONS.md").read_text().splitlines():
        m = re.match(r"## (\d+)\. (.*)", line)
        if m:
            current = int(m.group(1))
            titles[current] = m.group(2)
            modules[current] = set()
        elif current is not None and line.startswith("- Modules:"):
            modules[current] = {p.strip() for p in line[len("- Modules:") :].split(",")}
    return titles, modules


def python_text(path: Path) -> Iterator[tuple[int, str]]:
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                for i, line in enumerate(body[0].value.value.splitlines()):
                    yield body[0].lineno + i, line
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            yield tok.start[0], tok.string


def js_text(path: Path) -> Iterator[tuple[int, str]]:
    for i, line in enumerate(path.read_text().splitlines(), 1):
        idx = line.find("//")
        if idx >= 0 and "://" not in line[max(0, idx - 6) : idx + 2]:
            yield i, line[idx:]


def markdown_text(path: Path) -> Iterator[tuple[int, str]]:
    yield from enumerate(path.read_text().splitlines(), 1)


def sources() -> Iterator[tuple[Path, Iterator[tuple[int, str]]]]:
    for p in sorted(ROOT.glob("*.md")):
        yield p, markdown_text(p)
    for p in sorted([*(ROOT / "docs").glob("*.md"), *(ROOT / ".claude" / "skills").rglob("*.md")]):
        yield p, markdown_text(p)
    for top in ("src", "tests", "tools"):
        for p in sorted((ROOT / top).rglob("*")):
            if "_data" in p.parts or "__pycache__" in p.parts or not p.is_file():
                continue
            if p.suffix == ".py":
                yield p, python_text(p)
            elif p.suffix in (".js", ".mjs"):
                yield p, js_text(p)


def exists_in(base: Path, rel: str) -> bool:
    if "*" in rel:
        return any(base.glob(rel))
    return (base / rel).exists()


_SUFFIXES: dict[Path, list[str]] = {}


def exists_as_suffix(base: Path, rel: str) -> bool:
    """A shortened path such as `groq/util.ts` after `src/providers/groq/chat.ts`."""
    if base not in _SUFFIXES:
        _SUFFIXES[base] = [
            p.relative_to(base).as_posix()
            for p in base.rglob("*")
            if "node_modules" not in p.parts and ".git" not in p.parts and p.is_file()
        ]
    return any(f == rel or f.endswith("/" + rel) for f in _SUFFIXES[base])


def check_path(ref: str, here: Path, upstream: Path | None) -> str | None:
    """Return None if the reference resolves, else a reason."""
    candidates = expand_braces(ref)
    for cand in candidates:
        cand = cand.rstrip(".,;:)")
        if cand.startswith(BUILD_PREFIXES):
            continue
        if cand.startswith("../promptfoo/"):
            if upstream is None:
                return "unchecked (no upstream checkout)"
            if not exists_in(upstream, cand[len("../promptfoo/") :]):
                return "missing upstream"
            continue
        if cand.startswith(REPO_PREFIXES) or cand in ROOT_FILES:
            if not exists_in(ROOT, cand):
                return "missing in repo"
            continue
        if cand.startswith(UPSTREAM_PREFIXES) or cand.endswith((".ts", ".tsx")):
            if exists_in(ROOT, cand):
                continue
            if upstream is None:
                return "unchecked (no upstream checkout)"
            if cand.startswith(UPSTREAM_PREFIXES):
                if not exists_in(upstream, cand):
                    return "missing upstream"
            elif not exists_as_suffix(upstream, cand):
                return "missing upstream"
            continue
        if "/" in cand:
            # A package-relative module path such as `cli/watch.py`, or one relative to the
            # file that mentions it.
            bases = [PKG, ROOT, here.parent, ROOT / "tests" / "fixtures"]
            if not any(exists_in(b, cand) for b in bases) and not exists_as_suffix(PKG, cand):
                return "missing (package-relative)"
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--upstream", type=Path, default=ROOT.parent / "promptfoo")
    ap.add_argument("--all", action="store_true", help="also list D<n> refs to review")
    args = ap.parse_args()
    upstream = args.upstream if (args.upstream / "src").is_dir() else None
    titles, modules = decision_index()
    unused = {n for n, t in titles.items() if t.strip().lower().startswith("(unused)")}

    broken = 0
    review: list[str] = []
    counts = {"paths": 0, "drefs": 0}
    for path, lines in sources():
        rel = path.relative_to(ROOT).as_posix()
        for lineno, text in lines:
            for m in PATH_RE.finditer(text):
                ref = m.group(1)
                if re.fullmatch(r"[A-Z][A-Z_]+\.md", ref):
                    # A bare document name such as PLAN_PORT.md, next to the file or in docs/.
                    counts["paths"] += 1
                    if not any((b / ref).exists() for b in (path.parent, ROOT, ROOT / "docs")):
                        broken += 1
                        print(f"{rel}:{lineno}: {ref}: missing document")
                    continue
                if IGNORE.match(ref) or "/" not in ref and ref not in ROOT_FILES:
                    continue
                counts["paths"] += 1
                reason = check_path(ref, path, upstream)
                if reason:
                    if reason.startswith("unchecked"):
                        review.append(f"{rel}:{lineno}: {ref}: {reason}")
                    else:
                        broken += 1
                        print(f"{rel}:{lineno}: {ref}: {reason}")
            for m in DREF_RE.finditer(text):
                n = int(m.group(1))
                counts["drefs"] += 1
                if n not in titles:
                    broken += 1
                    print(f"{rel}:{lineno}: D{n}: no such decision")
                elif n in unused and rel != "docs/DECISIONS.md":
                    broken += 1
                    print(f"{rel}:{lineno}: D{n}: number is unused")
                elif args.all and rel != "docs/DECISIONS.md":
                    mods = modules.get(n, set())
                    if not any(rel == mod or rel.startswith(mod.rstrip("/") + "/") for mod in mods):
                        review.append(f"{rel}:{lineno}: D{n} ({titles[n][:60]})")
    print(f"checked {counts['paths']} path references and {counts['drefs']} D<n> references")
    if args.all and review:
        print("\nfor review:")
        print("\n".join(review))
    print(f"{broken} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
