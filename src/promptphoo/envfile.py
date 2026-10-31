"""`.env` loading. Ports setupEnv (src/util/env.ts), loadEnvFiles (src/util/envFile.ts) and
setupEnvFilesFromArgv (src/mainUtils.ts). Parsing uses python-dotenv."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import MutableMapping, Sequence

from dotenv import dotenv_values

logger = logging.getLogger(__name__)

_FALSY = ("false", "0", "no", "off", "")


def normalize_env_paths(raw: str | Sequence[str] | None) -> list[str]:
    values = [raw] if isinstance(raw, str) else list(raw or [])
    return [p.strip() for v in values for p in v.split(",") if p.strip()]


def load_env_files(
    paths: list[str] | None = None,
    *,
    override: bool | None = None,
    process_env: MutableMapping[str, str] | None = None,
) -> None:
    configured_path = os.environ.get("DOTENV_PATH") or os.environ.get("DOTENV_CONFIG_PATH")
    encoding = os.environ.get("DOTENV_ENCODING") or os.environ.get("DOTENV_CONFIG_ENCODING")
    if override is None:
        configured = os.environ.get("DOTENV_OVERRIDE", os.environ.get("DOTENV_CONFIG_OVERRIDE"))
        override = configured is not None and configured.lower() not in _FALSY
    files = paths or [configured_path or os.path.join(os.getcwd(), ".env")]
    parsed: dict[str, str] = {}
    for file in files:
        resolved = os.path.expanduser(file) if file.startswith("~") else file
        if not os.path.isfile(resolved):
            continue
        for key, value in dotenv_values(resolved, encoding=encoding or "utf-8").items():
            if override or key not in parsed:
                parsed[key] = value or ""
    destination = process_env if process_env is not None else os.environ
    for key, value in parsed.items():
        if override or key not in destination:
            destination[key] = value


def setup_env(env_path: str | Sequence[str] | None) -> None:
    paths = normalize_env_paths(env_path)
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Environment file not found: {p}")
    if len(paths) == 1:
        logger.info("Loading environment variables from %s", paths[0])
    elif paths:
        logger.info("Loading environment variables from: %s", ", ".join(paths))
    load_env_files(paths or None, override=True if paths else None)


def env_paths_from_argv(argv: Sequence[str]) -> list[str]:
    values: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            break
        m = re.match(r"^--env-(?:file|path)=(.*)$", arg)
        if m:
            values.append(m.group(1))
        elif arg in ("--env-file", "--env-path"):
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt and nxt != "--" and not nxt.startswith("-"):
                values.append(nxt)
                i += 1
        i += 1
    return normalize_env_paths(values)
