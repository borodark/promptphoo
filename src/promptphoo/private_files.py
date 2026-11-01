"""Owner-only permissions for the files promptphoo keeps under the state directory (D75).

The database, cache, logs and blobs can hold prompts, model outputs and API keys written inline
in a config. Node creates them with the default umask, readable by every local user. promptphoo
creates new directories 0700 and new files 0600, and narrows existing ones it opens to 0600.
Node runs as the same user, so sharing the state directory with it is unaffected.
"""

from __future__ import annotations

import logging
import os
import stat
from typing import Any

logger = logging.getLogger(__name__)

DIR_MODE = 0o700
FILE_MODE = 0o600


def make_private_dirs(path: str) -> None:
    """os.makedirs with mode 0700 for any directory it creates. Existing directories keep their
    mode, since a state directory shared on purpose should stay shared."""
    os.makedirs(path, mode=DIR_MODE, exist_ok=True)


def restrict(path: str) -> None:
    """Narrow an existing file to 0600 when group or others can access it."""
    try:
        mode = os.stat(path).st_mode
        if stat.S_ISREG(mode) and mode & 0o077:
            os.chmod(path, FILE_MODE)
    except OSError as e:
        logger.debug("Could not restrict permissions of %s: %s", path, e)


def private_opener(path: str, flags: int) -> int:
    """An `opener` for open() that creates the file 0600."""
    return os.open(path, flags, FILE_MODE)


def open_private(path: str, mode: str = "w", **kwargs: Any) -> Any:
    return open(path, mode, opener=private_opener, **kwargs)  # noqa: SIM115 - caller closes
