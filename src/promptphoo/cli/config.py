"""`promptfoo config get|set|unset email`. Ports src/commands/config.ts.

The email lives in `<configDir>/promptfoo.yaml` under `account.email`, the same file and key
Node uses (see `promptphoo.globalconfig`).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import click

from promptphoo import globalconfig
from promptphoo.cli.common import common_options, exit_with

logger = logging.getLogger("promptphoo")

# zod v4's z.email() pattern.
EMAIL_RE = re.compile(
    r"(?:[A-Za-z0-9_'+\-]+\.)*[A-Za-z0-9_'+\-]*[A-Za-z0-9_+-]@"
    r"(?:[A-Za-z0-9][A-Za-z0-9\-]*\.)+[A-Za-z]{2,}"
)
LOGGED_IN_ERROR = (
    "Cannot update email while logged in. Email is managed through Node promptfoo's "
    "'promptfoo auth login'. Please use 'promptfoo auth logout' there first if you want to "
    "use a different email."
)


def get_email() -> int:
    email = globalconfig.get_user_email()
    if email:
        logger.info(email)
    elif globalconfig.get_cloud_api_key():
        logger.info(
            "Email is managed through Node promptfoo's 'promptfoo auth login'. Run 'promptfoo auth "
            "whoami' there to view the current account."
        )
    else:
        logger.info('No email set. Use "promptphoo config set email <email>" to set one.')
    return 0


def set_email(email: str) -> int:
    if globalconfig.get_cloud_api_key():
        logger.error(LOGGED_IN_ERROR)
        return 1
    if not EMAIL_RE.fullmatch(email):
        logger.error(f"Invalid email address: {email}")
        return 1
    globalconfig.set_user_email(email)
    logger.info(f"Email set to {email}")
    return 0


def unset_email(force: bool) -> int:
    if globalconfig.get_cloud_api_key():
        logger.error(LOGGED_IN_ERROR)
        return 1
    current = globalconfig.get_user_email()
    if not current:
        logger.info("No email is currently set.")
        return 0
    if not force:
        should_unset = click.confirm(
            f'Are you sure you want to unset the email "{current}"?', default=False
        )
        if not should_unset:
            logger.info("Operation cancelled.")
            return 0
    globalconfig.set_user_email("")
    logger.info("Email has been unset.")
    return 0


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    @cli.group("config", help="Edit configuration settings")
    def config_group() -> None:
        pass

    @config_group.group("get", help="Get configuration settings")
    def get_group() -> None:
        pass

    @config_group.group("set", help="Set configuration settings")
    def set_group() -> None:
        pass

    @config_group.group("unset", help="Unset configuration settings")
    def unset_group() -> None:
        pass

    @get_group.command("email", help="Get user email")
    @common_options
    def get_email_cmd(**_: Any) -> None:
        exit_with(get_email())

    @set_group.command("email", help="Set user email")
    @click.argument("email")
    @common_options
    def set_email_cmd(email: str, **_: Any) -> None:
        exit_with(set_email(email))

    @unset_group.command("email", help="Unset user email")
    @click.option(
        "-f", "--force", is_flag=True, default=False, help="Force unset without confirmation"
    )
    @common_options
    def unset_email_cmd(force: bool, **_: Any) -> None:
        exit_with(unset_email(force))

    return config_group
