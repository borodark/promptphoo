"""State directory resolution. Ports getConfigDirectoryPath (src/util/config/manage.ts)."""

from __future__ import annotations

import os

_override: str | None = None


def set_config_directory_path(path: str | None) -> None:
    global _override
    _override = path


def get_config_directory_path(create: bool = False) -> str:
    path = (
        _override
        or os.environ.get("PROMPTFOO_CONFIG_DIR")
        or os.path.join(os.path.expanduser("~"), ".promptfoo")
    )
    if create:
        os.makedirs(path, exist_ok=True)
    return path
