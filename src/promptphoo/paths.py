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
        from promptphoo.private_files import make_private_dirs

        make_private_dirs(path)  # 0700 when created (D75)
    return path
