"""Options every upstream command gets from addCommonOptionsRecursively (src/mainUtils.ts).

`-v/--verbose` and `--env-file/--env-path` are already applied from argv by `cli.main.main`
before any command runs, so the options here only have to be accepted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import click

F = TypeVar("F", bound=Callable[..., Any])


def common_options(f: F) -> F:
    f = click.option("-v", "--verbose", is_flag=True, default=False, help="Show debug logs")(f)
    f = click.option(
        "--env-file",
        "--env-path",
        "env_file",
        multiple=True,
        help="Path(s) to .env file(s). Repeat the flag or use comma-separated values for "
        "multiple files.",
    )(f)
    return f


def exit_with(code: int) -> None:
    """Node sets `process.exitCode` and returns; a non-zero code ends the click command."""
    if code:
        raise SystemExit(code)
