"""Ports src/util/promptfooCommand.ts: how to spell the next command for the user.

`process.execPath` is `sys.executable` here, so in practice only the npm environment variables
can select the npx or brew spellings.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Literal

InstallerType = Literal["npx", "brew", "npm-global", "unknown"]


def detect_installer() -> InstallerType:
    npm_exec_path = os.environ.get("npm_execpath", "")  # noqa: SIM112
    npm_lifecycle_script = os.environ.get("npm_lifecycle_script", "")  # noqa: SIM112
    exec_path = sys.executable or ""
    if "npx" in npm_exec_path or "npx" in exec_path or "npx" in npm_lifecycle_script:
        return "npx"
    prefix = os.environ.get("npm_config_prefix", "")  # noqa: SIM112
    ua = os.environ.get("npm_config_user_agent", "")  # noqa: SIM112
    if (
        re.search(r"Homebrew[/\\]Cellar", prefix, re.I)
        or re.search(r"Homebrew[/\\]Cellar", exec_path, re.I)
        or re.search(r"[/\\]Homebrew[/\\]", prefix, re.I)
        or re.search(r"[/\\]Homebrew[/\\]", exec_path, re.I)
    ):
        return "brew"
    if re.search(r"\bnpx/\d+", ua, re.I):
        return "npx"
    if re.search(r"\bnpm/\d+", ua, re.I):
        return "npm-global"
    return "unknown"


def promptfoo_command(subcommand: str) -> str:
    if detect_installer() == "npx":
        return f"npx promptfoo@latest {subcommand}" if subcommand else "npx promptfoo@latest"
    return f"promptfoo {subcommand}" if subcommand else "promptfoo"
