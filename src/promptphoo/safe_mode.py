"""Test data loaded from files is data: safe mode (D74) and datasets by default (D77).

Upstream's `--safe-mode` only refuses inline JavaScript. A CSV `__expected` cell can still be a
`python:` assertion, a var can be `file:///any/path`, and a YAML test file can name an `exec:`
grader. In safe mode promptphoo refuses all of these in tests read from test files (CSV, JSON,
JSONL, YAML), before anything in them is loaded. Tests written in the config itself, and tests a
Python generator named by the config returns, stay trusted: the config is code.

Without safe mode, CSV and JSONL datasets still may not carry code (D77): code assertions,
script providers and `file://` references to code files are refused unless
PROMPTPHOO_TRUST_TEST_DATA=true. Their plain `file://` references (a document, an image) load as
in Node. YAML and JSON test files are trusted outside safe mode; config authors usually write
them.
"""

from __future__ import annotations

import os
import re
from typing import Any

from promptphoo import cli_state, envars
from promptphoo.files import confine_to_config_folder

CODE_ASSERTION_TYPES = {"python", "javascript", "ruby"}
SCRIPT_PROVIDER = re.compile(r"^(exec:|python:|golang:|ruby:|file://|package:)", re.IGNORECASE)
PROVIDER_KEYS = {"provider", "providers"}
CODE_FILE = re.compile(
    r"^file://.*\.(py|js|cjs|mjs|ts|cts|mts|jsx|tsx|sh|bash|rb|go|exe)(:[\w.]+)?$", re.IGNORECASE
)


class SafeModeError(ValueError):
    pass


def check_test_data(cases: Any, source: str, dataset: bool = False) -> None:
    """Raise SafeModeError when `cases`, read from `source`, could run code or reach the
    filesystem in safe mode, or, for a CSV/JSONL `dataset`, could run code at all."""
    items = cases if isinstance(cases, list) else [cases]
    if cli_state.safe_mode():
        strict: bool | None = True
    elif dataset and not envars.get_env_bool("PROMPTPHOO_TRUST_TEST_DATA"):
        strict = False
    else:
        strict = None
    for index, case in enumerate(items):
        where = f"{source}, test {index + 1}"
        if strict is not None:
            if strict and isinstance(case, str):
                _refuse(where, f'refers to another test file "{case}"', strict)
            _walk(case, where, (), strict)
        _confine_file_references(case, where, (), dataset)


def _confine_file_references(value: Any, where: str, path: tuple[str, ...], dataset: bool) -> None:
    """Var values in any test file, and every reference in a CSV or JSONL dataset, must stay
    inside the config folder (D78). Checked before nested references are loaded."""
    if isinstance(value, dict):
        for key, child in value.items():
            _confine_file_references(child, where, (*path, str(key)), dataset)
    elif isinstance(value, list):
        for child in value:
            _confine_file_references(child, where, path, dataset)
    elif (
        isinstance(value, str)
        and value.startswith("file://")
        and (dataset or path[:1] == ("vars",))
        and not CODE_FILE.match(value)  # code files are refused or trusted elsewhere
    ):
        reference = value[len("file://") :].split("#")[0]
        resolved = os.path.abspath(os.path.join(cli_state.base_path() or "", reference))
        confine_to_config_folder(resolved, where)


def _walk(value: Any, where: str, path: tuple[str, ...], strict: bool) -> None:
    if isinstance(value, dict):
        kind = value.get("type")
        if isinstance(kind, str):
            kind = kind.removeprefix("not-")  # not-python runs code too
        if "assert" in path and isinstance(kind, str) and kind in CODE_ASSERTION_TYPES:
            _refuse(where, f"has a {kind} assertion", strict)
        ref = value.get("$ref")
        if strict and isinstance(ref, str) and not ref.startswith("#"):
            _refuse(where, f'has a $ref to "{ref}"', strict)
        for key, child in value.items():
            _walk(child, where, (*path, str(key)), strict)
    elif isinstance(value, list):
        if strict and path == ("vars",):
            _refuse(where, "loads its vars from files", strict)
        for child in value:
            _walk(child, where, path, strict)
    elif isinstance(value, str):
        if strict and path == ("vars",):
            _refuse(where, f'loads its vars from "{value}"', strict)
        at = ".".join(path) or "top level"
        if strict and value.startswith("file://"):
            _refuse(where, f'refers to "{value}" at {at}', strict)
        if CODE_FILE.match(value):
            _refuse(where, f'refers to the code file "{value}" at {at}', strict)
        if PROVIDER_KEYS & set(path) and SCRIPT_PROVIDER.match(value):
            _refuse(where, f'names the script provider "{value}"', strict)


def _refuse(where: str, what: str, strict: bool) -> None:
    if strict:
        raise SafeModeError(
            f"Safe mode: {where} {what}. In safe mode, test files are data and cannot run code "
            "or read files. Move code-bearing tests into the config itself, or run without "
            "--safe-mode if you trust this file."
        )
    raise SafeModeError(
        f"{where} {what}. CSV and JSONL datasets are data and cannot run code (D77). Move "
        "code-bearing tests into the config or a YAML test file, or set "
        "PROMPTPHOO_TRUST_TEST_DATA=true if you trust this dataset."
    )
