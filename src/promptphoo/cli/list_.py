"""`promptfoo list evals|prompts|datasets [-n <limit>] [--ids-only]`.

Ports src/commands/list.ts.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import click

from promptphoo.cli.show import (
    ExitCode,
    env_option,
    print_border,
    print_resource_hint,
    run_db_command,
)

logger = logging.getLogger("promptphoo.cli")


def parse_limit(n: str | None) -> int | float | None:
    """`Number(cmdObj.n) || undefined`: NaN and 0 mean the default limit."""
    if n is None:
        return None
    text = n.strip()
    try:
        if text.lower().startswith(("0x", "0o", "0b")):
            value: float = int(text, 0)
        elif text.lstrip("+-") in ("Infinity", ""):
            value = 0 if not text else (-math.inf if text.startswith("-") else math.inf)
        elif text.lower().lstrip("+-").startswith(("inf", "nan")):
            return None  # float() accepts these, Number() does not
        else:
            value = float(text)
    except ValueError:
        return None
    if not value or math.isnan(value):
        return None
    if math.isinf(value):
        return -1  # SQLite reads a negative LIMIT as "no limit"
    return int(value) if float(value).is_integer() else value


def _sha_prefix(raw: Any) -> str:
    from promptphoo.evaluator.run import sha256

    return sha256(raw)[:6]


def _cut(text: str) -> str:
    return text[:100] + ("..." if len(text) > 100 else "")


def list_evals(limit: int | float | None, ids_only: bool, _code: ExitCode) -> None:
    from promptphoo.db.queries import get_vars_from_evals
    from promptphoo.models.eval import Eval
    from promptphoo.output.table import wrap_table

    evals = Eval.get_many(limit)
    if ids_only:
        for evl in evals:
            logger.info("%s", evl.id)
        return

    vars_ = get_vars_from_evals(evals)
    rows = []
    for evl in sorted(evals, key=lambda e: e.created_at):
        prompts = evl.get_prompts()
        description = evl.config.get("description") or ""
        rows.append(
            {
                "eval id": evl.id,
                "description": _cut(description),
                "prompts": ", ".join(_sha_prefix(p["raw"]) for p in prompts),
                "vars": ", ".join(vars_.get(evl.id) or []),
            }
        )
    widths = {"eval id": 32, "description": 25, "prompts": 10, "vars": 12}
    logger.info("%s", wrap_table(rows, widths))
    print_border()
    print_resource_hint("eval")
    print_resource_hint("prompt")


def list_prompts(limit: int | float | None, ids_only: bool, _code: ExitCode) -> None:
    from promptphoo.db.queries import get_prompts, locale_compare_key
    from promptphoo.output.table import wrap_table

    prompts = sorted(get_prompts(limit), key=lambda p: locale_compare_key(p["recentEvalId"]))
    if ids_only:
        for prompt in prompts:
            logger.info("%s", prompt["id"])
        return

    rows = [
        {
            "prompt id": p["id"][:6],
            "raw": _cut(p["prompt"]["raw"]),
            "evals": p["count"],
            "recent eval": p["recentEvalId"],
        }
        for p in prompts
    ]
    widths = {"prompt id": 12, "raw": 30, "evals": 8, "recent eval": 30}
    logger.info("%s", wrap_table(rows, widths))
    print_border()
    print_resource_hint("prompt")
    print_resource_hint("eval")


def _score(prompt: dict[str, Any]) -> float:
    score = (prompt["prompt"].get("metrics") or {}).get("score")
    return score if isinstance(score, int | float) and score else 0


def _best_prompt(dataset: dict[str, Any]) -> str:
    if not dataset["prompts"]:
        return "N/A"
    best = sorted(dataset["prompts"], key=_score, reverse=True)
    # sorted(reverse=True) keeps equal scores in input order, like Array.prototype.sort.
    return best[0]["id"][:6] or "N/A"


def list_datasets(limit: int | float | None, ids_only: bool, _code: ExitCode) -> None:
    from promptphoo.db.queries import get_test_cases, locale_compare_key
    from promptphoo.output.table import wrap_table

    datasets = sorted(
        get_test_cases(limit), key=lambda d: locale_compare_key(d["recentEvalId"]), reverse=True
    )
    if ids_only:
        for dataset in datasets:
            logger.info("%s", dataset["id"])
        return

    rows = [
        {
            "dataset id": d["id"][:6],
            "best prompt": _best_prompt(d),
            "evals": d["count"],
            "prompts": len(d["prompts"]),
            "recent eval": d["recentEvalId"],
        }
        for d in datasets
    ]
    widths = {"dataset id": 12, "best prompt": 15, "evals": 8, "prompts": 10, "recent eval": 30}
    logger.info("%s", wrap_table(rows, widths))
    print_border()
    print_resource_hint("dataset")
    print_resource_hint("prompt")
    print_resource_hint("eval")


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    @cli.group("list", help="List various resources")
    def list_cmd() -> None:
        pass

    for name, noun, handler in (
        ("evals", "evaluations", list_evals),
        ("prompts", "prompts", list_prompts),
        ("datasets", "datasets", list_datasets),
    ):
        singular = {"evals": "evaluation", "prompts": "prompt", "datasets": "dataset"}[name]

        @list_cmd.command(name, help=f"List {noun}")
        @env_option
        @click.option("-n", "n", metavar="<limit>", help=f"Number of {noun} to display")
        @click.option("--ids-only", is_flag=True, help=f"Only show {singular} IDs")
        def sub(
            env_path: tuple[str, ...],
            verbose: bool,
            n: str | None,
            ids_only: bool,
            handler: Any = handler,
        ) -> None:
            limit = parse_limit(n)
            run_db_command(env_path, lambda code: handler(limit, ids_only, code))

    return list_cmd
