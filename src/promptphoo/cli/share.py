"""`promptfoo share [id]`: registered with upstream's arguments, but sharing is cloud-only.

Upstream (src/commands/share.ts) uploads the eval to promptfoo.app or a self-hosted server.
promptphoo keeps results local (see D44), so the command fails with a pointer to `export`.
"""

from __future__ import annotations

import logging
from typing import Any

import click

from promptphoo.cli.common import common_options, exit_with

logger = logging.getLogger("promptphoo")

SHARE_NOT_SUPPORTED = (
    "Sharing is not supported by promptphoo: results stay local and are never uploaded. "
    "Use `promptphoo export <evalId> -o results.json` to save an eval to a file, or "
    "`npx promptfoo@latest view` from Node promptfoo to browse the same database."
)


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    @cli.command(
        "share",
        help="Create a shareable URL of an eval or a model audit (defaults to most recent)",
    )
    @click.argument("id_", metavar="[ID]", required=False)
    @click.option(
        "--show-auth",
        is_flag=True,
        default=False,
        help="Show username/password authentication information in the URL if exists",
    )
    @click.option(
        "-y",
        "--yes",
        is_flag=True,
        default=False,
        help="Flag does nothing (maintained for backwards compatibility only - shares are now "
        "private by default)",
    )
    @common_options
    def share_cmd(**_: Any) -> None:
        logger.error(SHARE_NOT_SUPPORTED)
        exit_with(1)

    return share_cmd
