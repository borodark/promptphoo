"""`promptfoo show [id]`, `show eval [id]`, `show prompt <id>`, `show dataset <id>`.

Ports src/commands/show.ts. Telemetry is a no-op.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import click

from promptphoo.jsvalues import to_fixed
from promptphoo.output.ansi import chalk

logger = logging.getLogger("promptphoo.cli")


# --- shared with list_ and delete ------------------------------------------------------------


class ExitCode:
    """process.exitCode: handlers set it, the command exits with it after cleanup."""

    def __init__(self) -> None:
        self.code = 0

    def fail(self) -> None:
        self.code = 1


def run_db_command(env_path: Any, fn: Callable[[ExitCode], Any]) -> None:
    """setupEnv, the startup migrations main.ts runs for every command, then `fn`; exits with
    the code `fn` left in `process.exitCode`."""
    from promptphoo import db
    from promptphoo.envfile import setup_env

    setup_env(list(env_path) if env_path else None)
    exit_code = ExitCode()
    try:
        db.run_db_migrations()
        fn(exit_code)
    finally:
        db.close_db()
    if exit_code.code:
        raise SystemExit(exit_code.code)


def env_option(fn: Callable[..., Any]) -> Callable[..., Any]:
    fn = click.option(
        "--env-file", "--env-path", "env_path", multiple=True, help="Path to .env file"
    )(fn)
    return click.option("-v", "--verbose", is_flag=True, help="Show debug logs")(fn)


class IdOrSubcommandGroup(click.Group):
    """A commander command that takes `<id>` and also has subcommands (`show <id>` beside
    `show eval <id>`): a first argument that names no subcommand goes to the default command."""

    default_command: click.Command

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        if args and args[0] not in self.commands:
            return self.default_command.name, self.default_command, args
        return super().resolve_command(ctx, args)


def print_border() -> None:
    from promptphoo.output.table import terminal_max_width

    logger.info("%s", "=" * terminal_max_width())


def print_resource_hint(resource: str) -> None:
    noun = "evaluation" if resource == "eval" else resource
    shown = chalk(f"promptphoo show {resource} <id>", "green")
    logger.info("Run %s to see details of a specific %s.", shown, noun)


# --- formatting ------------------------------------------------------------------------------


def _count(metrics: Mapping[str, Any], key: str) -> Any:
    value = metrics.get(key)
    return 0 if value is None else value


def format_pass_rate(metrics: Mapping[str, Any] | None) -> str:
    if not metrics:
        return "-"
    counts: list[Any] = [
        metrics.get(k) for k in ("testPassCount", "testFailCount", "testErrorCount")
    ]
    if not all(isinstance(c, int | float) and not isinstance(c, bool) for c in counts):
        return "-"  # undefined + n is NaN, and NaN > 0 is false
    passed, total = counts[0], sum(counts)
    if total > 0:
        return f"{to_fixed((passed / total) * 100, 2)}%"
    return "-"


def format_pass_count(metrics: Mapping[str, Any] | None) -> str:
    if not metrics:
        return "-"
    return _js_number(_count(metrics, "testPassCount"))


def format_fail_count(metrics: Mapping[str, Any] | None) -> str:
    if not metrics:
        return "-"
    fail_count = _count(metrics, "testFailCount")
    error_count = _count(metrics, "testErrorCount")
    if error_count > 0:
        return f"{_js_number(fail_count)} (+{_js_number(error_count)} errors)"
    return _js_number(fail_count)


def _js_number(value: Any) -> str:
    from promptphoo.jsvalues import js_string

    return js_string(value)


def _raw_score(metrics: Mapping[str, Any] | None) -> str:
    """`metrics?.score?.toFixed(2) || '-'`."""
    score = (metrics or {}).get("score")
    if isinstance(score, int | float) and not isinstance(score, bool):
        return to_fixed(score, 2) or "-"
    return "-"


# --- handlers --------------------------------------------------------------------------------


def handle_prompt(id_: str, exit_code: ExitCode) -> None:
    from promptphoo.db.queries import get_prompt_from_hash, locale_compare_key
    from promptphoo.output.table import wrap_table

    prompt = get_prompt_from_hash(id_)
    if not prompt:
        logger.error("Prompt with ID %s not found.", id_)
        exit_code.fail()
        return

    print_border()
    logger.info("%s", chalk(prompt["prompt"]["raw"], "cyan"))
    print_border()
    logger.info("%s", chalk(f"Prompt {id_}", "bold"))
    print_border()

    logger.info("This prompt is used in the following evals:")
    evals = sorted(prompt["evals"], key=lambda e: locale_compare_key(e["id"]), reverse=True)
    table = [
        {
            "Eval ID": evl["id"][:6],
            "Dataset ID": evl["datasetId"][:6],
            "Raw score": _raw_score(evl.get("metrics")),
            "Pass rate": format_pass_rate(evl.get("metrics")),
            "Pass count": format_pass_count(evl.get("metrics")),
            "Fail count": format_fail_count(evl.get("metrics")),
        }
        for evl in evals[:10]
    ]
    logger.info("%s", wrap_table(table))
    print_border()
    print_resource_hint("eval")
    print_resource_hint("dataset")


AMBIGUOUS_LIST_LIMIT = 10


def resolve_eval_id(ref: str, exit_code: ExitCode) -> str | None:
    """The full id of the one eval `ref` names, exactly or by prefix. Upstream matches a prefix
    and then looks the prefix up as an exact id, which always fails (D71). Several matches are
    listed and fail; no match returns `ref` unchanged, so the caller's not-found message stays
    upstream's."""
    from promptphoo.db.queries import resolve_eval_ref

    matches = resolve_eval_ref(ref)
    if len(matches) == 1:
        return matches[0]["id"]
    if not matches:
        return ref
    lines = [f'ID prefix "{ref}" matches {len(matches)} evaluations:']
    for m in matches[:AMBIGUOUS_LIST_LIMIT]:
        description = m.get("description") or ""
        if len(description) > 60:
            description = description[:57] + "..."
        lines.append(f"  {m['id']}" + (f"  {description}" if description else ""))
    if len(matches) > AMBIGUOUS_LIST_LIMIT:
        lines.append(f"  ... and {len(matches) - AMBIGUOUS_LIST_LIMIT} more")
    lines.append("Use a longer prefix or the full ID.")
    logger.error("%s", "\n".join(lines))
    exit_code.fail()
    return None


def handle_eval(id_: str, exit_code: ExitCode) -> None:
    from promptphoo.models.eval import Eval
    from promptphoo.output.table import generate_table

    resolved = resolve_eval_id(id_, exit_code)
    if resolved is None:
        return
    id_ = resolved
    eval_ = Eval.find_by_id(id_)
    if not eval_:
        logger.error("No evaluation found with ID %s", id_)
        logger.info(
            "Run %s to see available evaluation IDs.", chalk("promptphoo list evals", "green")
        )
        exit_code.fail()
        return
    table = eval_.get_table()
    prompts = table["head"]["prompts"]
    vars_ = table["head"]["vars"]

    logger.info("%s", generate_table(table, 100, 25))
    if len(table["body"]) > 25:
        rows_left = len(table["body"]) - 25
        logger.info("... %d more row%s not shown ...\n", rows_left, "" if rows_left == 1 else "s")

    print_border()
    logger.info("%s", chalk(f"Eval {id_}", "cyan"))
    print_border()
    logger.info("%d prompts", len(prompts))
    prompt_ids = [p.get("id") for p in prompts if p.get("id")]
    if prompt_ids:
        unique = list(dict.fromkeys(prompt_ids))
        preview = ", ".join(unique[:5])
        more = f" (and {len(unique) - 5} more...)" if len(unique) > 5 else ""
        logger.info("Prompt IDs: %s%s", preview, more)
    more_vars = f" (and {len(vars_) - 5} more...)" if len(vars_) > 5 else ""
    logger.info("%d variables: %s%s", len(vars_), ", ".join(vars_[:5]), more_vars)


def handle_dataset(id_: str, exit_code: ExitCode) -> None:
    from promptphoo.db.queries import get_dataset_from_hash, locale_compare_key
    from promptphoo.output.table import wrap_table

    dataset = get_dataset_from_hash(id_)
    if not dataset:
        logger.error("Dataset with ID %s not found.", id_)
        exit_code.fail()
        return

    print_border()
    logger.info("%s", chalk(f"Dataset {id_}", "bold"))
    print_border()

    logger.info("This dataset is used in the following evals:")
    prompts = sorted(
        dataset["prompts"], key=lambda p: locale_compare_key(p["evalId"]), reverse=True
    )
    table = [
        {
            "Eval ID": p["evalId"][:6],
            "Prompt ID": p["id"][:6],
            "Raw score": _raw_score(p["prompt"].get("metrics")),
            "Pass rate": format_pass_rate(p["prompt"].get("metrics")),
            "Pass count": format_pass_count(p["prompt"].get("metrics")),
            "Fail count": format_fail_count(p["prompt"].get("metrics")),
        }
        for p in prompts[:10]
    ]
    logger.info("%s", wrap_table(table))
    print_border()
    print_resource_hint("prompt")
    print_resource_hint("eval")


def show_latest_eval(exit_code: ExitCode) -> None:
    from promptphoo.models.eval import Eval

    latest = Eval.latest()
    if latest:
        handle_eval(latest.id, exit_code)
        return
    logger.error("No eval found")
    logger.info("Run %s to create one.", chalk("promptphoo eval", "green"))
    exit_code.fail()


def handle_show(id_: str | None, exit_code: ExitCode) -> None:
    from promptphoo.db.queries import get_dataset_from_hash, get_prompt_from_hash, resolve_eval_ref

    if not id_:
        show_latest_eval(exit_code)
        return
    if resolve_eval_ref(id_):
        handle_eval(id_, exit_code)
        return
    if get_prompt_from_hash(id_):
        handle_prompt(id_, exit_code)
        return
    if get_dataset_from_hash(id_):
        handle_dataset(id_, exit_code)
        return
    logger.error("No resource found with ID %s", id_)
    green = [chalk(f"promptphoo list {r}", "green") for r in ("evals", "prompts", "datasets")]
    logger.info("Run %s, %s, or %s to see available IDs.", *green)
    exit_code.fail()


# --- registration ----------------------------------------------------------------------------


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    @cli.group(
        "show",
        cls=IdOrSubcommandGroup,
        invoke_without_command=True,
        help="Show details of a specific resource (defaults to most recent)",
    )
    @env_option
    @click.pass_context
    def show_cmd(ctx: click.Context, env_path: tuple[str, ...], verbose: bool) -> None:
        if ctx.invoked_subcommand is None:
            run_db_command(env_path, lambda code: handle_show(None, code))

    @click.command("show-id", hidden=True)
    @click.argument("id_", metavar="[ID]")
    @env_option
    def show_id(id_: str, env_path: tuple[str, ...], verbose: bool) -> None:
        run_db_command(env_path, lambda code: handle_show(id_, code))

    assert isinstance(show_cmd, IdOrSubcommandGroup)
    show_cmd.default_command = show_id

    @show_cmd.command(
        "eval", help="Show details of a specific evaluation (defaults to most recent)"
    )
    @click.argument("id_", metavar="[ID]", required=False)
    @env_option
    def show_eval(id_: str | None, env_path: tuple[str, ...], verbose: bool) -> None:
        def run(code: ExitCode) -> None:
            if not id_:
                show_latest_eval(code)
            else:
                handle_eval(id_, code)

        run_db_command(env_path, run)

    @show_cmd.command("prompt", help="Show details of a specific prompt")
    @click.argument("id_", metavar="ID")
    @env_option
    def show_prompt(id_: str, env_path: tuple[str, ...], verbose: bool) -> None:
        run_db_command(env_path, lambda code: handle_prompt(id_, code))

    @show_cmd.command("dataset", help="Show details of a specific dataset")
    @click.argument("id_", metavar="ID")
    @env_option
    def show_dataset(id_: str, env_path: tuple[str, ...], verbose: bool) -> None:
        run_db_command(env_path, lambda code: handle_dataset(id_, code))

    return show_cmd
