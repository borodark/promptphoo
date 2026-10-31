"""`promptfoo delete <id>` and `delete eval <id|latest|all>`.

Ports src/commands/delete.ts. `delete eval all` asks for confirmation like upstream's
@inquirer/confirm prompt; `-y/--yes` is a promptphoo addition that answers it (see D41).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import click

from promptphoo.cli.show import ExitCode, IdOrSubcommandGroup, env_option, run_db_command

logger = logging.getLogger("promptphoo.cli")

CONFIRM_DELETE_ALL = (
    "Are you sure you want to delete all stored evaluations? This action cannot be undone."
)


def handle_eval_delete(eval_id: str, exit_code: ExitCode) -> None:
    from promptphoo.db.queries import delete_eval

    try:
        delete_eval(eval_id)
        logger.info("Evaluation with ID %s has been successfully deleted.", eval_id)
    except Exception as e:
        # `${error}` of an Error is "Error: <message>".
        logger.error("Could not delete evaluation with ID %s:\nError: %s", eval_id, _message(e))
        exit_code.fail()


def _message(e: BaseException) -> str:
    return str(e.args[0]) if isinstance(e, LookupError) and e.args else str(e)


def confirm(message: str) -> bool | None:
    """@inquirer/confirm with its default of yes: an empty answer is yes, any other answer is
    yes only when it starts with y. Inquirer's cursor redraws are not reproduced. At end of
    input there is no answer (None); Node then exits 1 on its unsettled prompt."""
    sys.stdout.write(f"? {message} (Y/n) ")
    sys.stdout.flush()
    answer = sys.stdin.readline()
    if not answer:
        sys.stdout.write("\n")
        return None
    answer = answer.strip()
    return not answer or answer[0] in "yY"


def handle_eval_delete_all(yes: bool = False, exit_code: ExitCode | None = None) -> None:
    from promptphoo.db.queries import delete_all_evals

    confirmed = True if yes else confirm(CONFIRM_DELETE_ALL)
    if confirmed is None and exit_code is not None:
        exit_code.fail()
    if not confirmed:
        return
    delete_all_evals()
    logger.info("All evaluations have been deleted.")


def handle_delete(id_: str, exit_code: ExitCode) -> None:
    from promptphoo.db.queries import get_eval_from_id

    if get_eval_from_id(id_):
        handle_eval_delete(id_, exit_code)
        return
    logger.error("No resource found with ID %s", id_)
    exit_code.fail()


def handle_delete_eval(eval_id: str, yes: bool, exit_code: ExitCode) -> None:
    from promptphoo.models.eval import Eval

    if eval_id == "latest":
        latest = Eval.latest()
        if latest:
            handle_eval_delete(latest.id, exit_code)
        else:
            logger.error("No eval found.")
            exit_code.fail()
    elif eval_id == "all":
        handle_eval_delete_all(yes, exit_code)
    else:
        handle_eval_delete(eval_id, exit_code)


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    @cli.group("delete", cls=IdOrSubcommandGroup, help="Delete various resources")
    def delete_cmd() -> None:
        pass

    @click.command("delete-id", hidden=True)
    @click.argument("id_", metavar="ID")
    @env_option
    def delete_id(id_: str, env_path: tuple[str, ...], verbose: bool) -> None:
        run_db_command(env_path, lambda code: handle_delete(id_, code))

    assert isinstance(delete_cmd, IdOrSubcommandGroup)
    delete_cmd.default_command = delete_id

    @delete_cmd.command(
        "eval",
        help='Delete an evaluation by ID. Use "latest" to delete the most recent evaluation, '
        'or "all" to delete all evaluations.',
    )
    @click.argument("eval_id", metavar="ID")
    @env_option
    @click.option(
        "-y", "--yes", is_flag=True, help="Skip the confirmation prompt for `delete eval all`"
    )
    def delete_eval_cmd(eval_id: str, env_path: tuple[str, ...], verbose: bool, yes: bool) -> None:
        run_db_command(env_path, lambda code: handle_delete_eval(eval_id, yes, code))

    return delete_cmd
