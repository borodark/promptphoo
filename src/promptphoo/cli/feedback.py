"""`promptphoo feedback [message]`. Replaces src/commands/feedback.ts and src/feedback.ts.

Upstream posts the message, and optionally the user's email, to api.promptfoo.dev. promptphoo is
an independent port, so that would send feedback about this tool to a company that does not
maintain it. The command sends nothing and points to promptphoo's issue tracker instead (D61).
"""

from __future__ import annotations

import logging
from importlib import metadata
from typing import Any

import click

from promptphoo.cli.common import common_options

logger = logging.getLogger("promptphoo")

FALLBACK_TRACKER = "the issue tracker of the promptphoo repository"


def issues_url() -> str:
    """The `Issues` project URL from the installed package metadata, if it declares one."""
    try:
        urls = metadata.metadata("promptphoo").get_all("Project-URL") or []
    except metadata.PackageNotFoundError:
        return FALLBACK_TRACKER
    for entry in urls:
        label, _, url = entry.partition(",")
        if label.strip().lower() in ("issues", "bug tracker") and url.strip():
            return url.strip()
    return FALLBACK_TRACKER


def gather_feedback(message: str | None = None) -> None:
    logger.info("promptphoo does not send feedback anywhere.")
    logger.info(f"Please open an issue at {issues_url()}.")
    logger.info(
        "promptphoo is an unofficial port: report problems with it there, not to the promptfoo "
        "project."
    )
    if message and message.strip():
        logger.info(f"\nYour message, ready to paste:\n\n{message.strip()}")


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    @cli.command("feedback", help="Where to send feedback about promptphoo")
    @click.argument("message", required=False)
    @common_options
    def feedback_cmd(message: str | None, **_: Any) -> None:
        gather_feedback(message)

    return feedback_cmd
