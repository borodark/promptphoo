"""`promptfoo feedback [message]`. Ports src/commands/feedback.ts and src/feedback.ts.

Feedback goes to the same endpoint Node posts to (api.promptfoo.dev). It is the one utility
command that needs the network; without it, it reports Node's network-error message.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import click

from promptphoo import globalconfig
from promptphoo.cli.common import common_options
from promptphoo.jsvalues import json_stringify

logger = logging.getLogger("promptphoo")

FEEDBACK_URL = "https://api.promptfoo.dev/api/feedback"


async def send_feedback(feedback: str) -> None:
    from promptphoo.fetch import fetch_with_proxy

    if not feedback.strip():
        return
    try:
        resp = await fetch_with_proxy(
            FEEDBACK_URL,
            {
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": json_stringify({"message": feedback}),
            },
        )
        if resp.ok:
            logger.info("Feedback sent. Thank you!")
        else:
            logger.info("Failed to send feedback. Please try again or open an issue on GitHub.")
    except Exception:  # noqa: BLE001 - upstream reports any failure the same way
        logger.error("Network error while sending feedback")


def _prompt_user(text: str) -> str:
    return click.prompt(text, default="", show_default=False, prompt_suffix="")


async def gather_feedback(message: str | None = None) -> None:
    if message:
        await send_feedback(message)
        return
    try:
        user_email = globalconfig.get_user_email()
        logger.info("\n📝 promptfoo Feedback")
        logger.info("─" * 40)
        feedback_text = _prompt_user("Share your thoughts, bug reports, or feature requests:\n>")
        if not feedback_text.strip():
            logger.info("No feedback provided.")
            return
        final_feedback = feedback_text
        if user_email:
            contact_info = user_email
        else:
            contact_info = _prompt_user(
                "Email address (optional, if you'd like a response): "
            ).strip()
        if contact_info:
            final_feedback = f"{feedback_text}\n\nContact: {contact_info}"
        await send_feedback(final_feedback)
    except Exception:  # noqa: BLE001
        logger.error("Error gathering feedback")


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    @cli.command("feedback", help="Send feedback to the promptfoo developers")
    @click.argument("message", required=False)
    @common_options
    def feedback_cmd(message: str | None, **_: Any) -> None:
        asyncio.run(gather_feedback(message))

    return feedback_cmd
