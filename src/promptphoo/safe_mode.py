"""Safe mode: test data loaded from files is data, never code or file access (D74).

Upstream's `--safe-mode` only refuses inline JavaScript. A CSV `__expected` cell can still be a
`python:` assertion, a var can be `file:///any/path`, and a YAML test file can name an `exec:`
grader. In safe mode promptphoo refuses all of these in tests read from test files (CSV, JSON,
JSONL, YAML), before anything in them is loaded. Tests written in the config itself, and tests a
Python generator named by the config returns, stay trusted: the config is code.
"""

from __future__ import annotations

import re
from typing import Any

from promptphoo import cli_state

CODE_ASSERTION_TYPES = {"python", "javascript", "ruby"}
SCRIPT_PROVIDER = re.compile(r"^(exec:|python:|golang:|ruby:|file://|package:)", re.IGNORECASE)
PROVIDER_KEYS = {"provider", "providers"}


class SafeModeError(ValueError):
    pass


def check_test_data(cases: Any, source: str) -> None:
    """Raise SafeModeError when safe mode is on and `cases`, read from `source`, could run code
    or reach the filesystem."""
    if not cli_state.safe_mode():
        return
    items = cases if isinstance(cases, list) else [cases]
    for index, case in enumerate(items):
        where = f"{source}, test {index + 1}"
        if isinstance(case, str):
            _refuse(where, f'refers to another test file "{case}"')
        _walk(case, where, ())


def _walk(value: Any, where: str, path: tuple[str, ...]) -> None:
    if isinstance(value, dict):
        kind = value.get("type")
        if isinstance(kind, str):
            kind = kind.removeprefix("not-")  # not-python runs code too
        if "assert" in path and isinstance(kind, str) and kind in CODE_ASSERTION_TYPES:
            _refuse(where, f"has a {kind} assertion")
        ref = value.get("$ref")
        if isinstance(ref, str) and not ref.startswith("#"):
            _refuse(where, f'has a $ref to "{ref}"')
        for key, child in value.items():
            _walk(child, where, (*path, str(key)))
    elif isinstance(value, list):
        if path == ("vars",):
            _refuse(where, "loads its vars from files")
        for child in value:
            _walk(child, where, path)
    elif isinstance(value, str):
        if path == ("vars",):
            _refuse(where, f'loads its vars from "{value}"')
        if value.startswith("file://"):
            _refuse(where, f'refers to "{value}" at {".".join(path) or "top level"}')
        if PROVIDER_KEYS & set(path) and SCRIPT_PROVIDER.match(value):
            _refuse(where, f'names the script provider "{value}"')


def _refuse(where: str, what: str) -> None:
    raise SafeModeError(
        f"Safe mode: {where} {what}. In safe mode, test files are data and cannot run code or "
        "read files. Move code-bearing tests into the config itself, or run without --safe-mode "
        "if you trust this file."
    )
