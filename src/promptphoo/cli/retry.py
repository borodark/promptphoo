"""`promptfoo retry <evalId>` and the retry-errors helpers shared with `eval --retry-errors`.
Ports src/commands/retry.ts and src/node/retry.ts.

A retry runs the saved eval again in resume + retry mode: completed pairs are skipped except
ERROR rows, which run again. The old ERROR rows are deleted only after the retry succeeded, then
the prompt metrics are recalculated from the rows that remain.
"""

from __future__ import annotations

import asyncio
import logging
import textwrap
from typing import Any

import click

from promptphoo import cli_state, db
from promptphoo.config.load import ConfigResolutionError, resolve_configs
from promptphoo.providers import filter_providers

logger = logging.getLogger("promptphoo")

ERROR = 2  # ResultFailureReason.ERROR
# Batch size of 1000 balances memory use against query overhead for large evals.
RECALCULATE_BATCH_SIZE = 1000


# --- filterProviders.ts (persisted provider filters) ----------------------------------------


def get_persisted_provider_filter_options(provider_filter: Any) -> dict[str, str]:
    if provider_filter is None:
        return {}
    if isinstance(provider_filter, str) and provider_filter:
        return {"filterProviders": provider_filter}
    from promptphoo.jsvalues import json_stringify

    raise ValueError(
        "Stored provider filter is invalid (expected a non-empty string, got "
        f"{json_stringify(provider_filter)}). Refusing to run without the original provider "
        "selection."
    )


def get_provider_filter_regex_error(pattern: str) -> str | None:
    from promptphoo.jsregex import compile_js_regex

    try:
        compile_js_regex(pattern)
        return None
    except (ValueError, Exception) as e:  # noqa: BLE001 - mirror `new RegExp` throwing
        return str(e)


# --- database helpers ------------------------------------------------------------------------


def get_error_result_ids(eval_id: str) -> list[str]:
    """getErrorResultIds: ids of the eval's ERROR results."""
    rows = (
        db.get_db()
        .execute(
            "SELECT id FROM eval_results WHERE eval_id = ? AND failure_reason = ?", (eval_id, ERROR)
        )
        .fetchall()
    )
    return [r["id"] if isinstance(r, dict) else r[0] for r in rows]


def delete_error_results(result_ids: list[str]) -> None:
    """deleteErrorResults: batch delete, then notify watchers of each affected eval."""
    if not result_ids:
        return
    conn = db.get_db()
    affected: list[str] = []
    with db.transaction(conn):
        for start in range(0, len(result_ids), 500):
            chunk = result_ids[start : start + 500]
            marks = ", ".join("?" * len(chunk))
            for r in conn.execute(
                f"SELECT DISTINCT eval_id FROM eval_results WHERE id IN ({marks})", chunk
            ).fetchall():
                eval_id = r["eval_id"] if isinstance(r, dict) else r[0]
                if eval_id not in affected:
                    affected.append(eval_id)
            conn.execute(f"DELETE FROM eval_results WHERE id IN ({marks})", chunk)
    for eval_id in affected:
        db.update_signal_file(eval_id)
    logger.debug("Deleted %d error results from database", len(result_ids))


def _accumulate_named_metric(
    metrics: dict[str, Any], name: str, value: float, grading: Any, test_vars: dict[str, Any]
) -> None:
    """util/namedMetrics.ts accumulateNamedMetric"""
    from promptphoo.assertions.index import render_metric_name

    count = (
        sum(
            1
            for c in (grading or {}).get("componentResults") or []
            if render_metric_name((c.get("assertion") or {}).get("metric"), test_vars) == name
        )
        or 1
    )
    weights = (grading or {}).get("namedScoreWeights") or {}
    if name in weights:
        weight_total = weights[name] if weights[name] is not None else 0
        weighted = value * weight_total
    else:
        weight_total = count
        weighted = value
    metrics["namedScores"][name] = metrics["namedScores"].get(name, 0) + weighted
    metrics["namedScoresCount"][name] = metrics["namedScoresCount"].get(name, 0) + count
    weights_acc = metrics.setdefault("namedScoreWeights", {})
    weights_acc[name] = weights_acc.get(name, 0) + weight_total


async def recalculate_prompt_metrics(eval_record: Any) -> None:
    """recalculatePromptMetrics: rebuild every column's metrics from the stored rows, streaming
    them in batches."""
    from promptphoo.evaluator.run import create_default_prompt_metrics
    from promptphoo.tokens import accumulate_grading_token_usage, accumulate_response_token_usage

    logger.debug("Recalculating prompt metrics after deleting ERROR results")
    by_prompt = {i: create_default_prompt_metrics() for i in range(len(eval_record.prompts))}
    for batch in eval_record.fetch_results_batched(RECALCULATE_BATCH_SIZE):
        for result in batch:
            metrics = by_prompt.get(result.prompt_idx)
            if metrics is None:
                logger.debug("Skipping result with invalid promptIdx: %s", result.prompt_idx)
                continue
            response = result.response or {}
            # updatePromptResultCounts
            if result.success:
                metrics["testPassCount"] += 1
            elif result.failure_reason == ERROR:
                metrics["testErrorCount"] += 1
            else:
                metrics["testFailCount"] += 1
            if response.get("cached"):
                metrics["testCacheHitCount"] = (metrics.get("testCacheHitCount") or 0) + 1
            metrics["score"] += result.score or 0
            metrics["totalLatencyMs"] += result.latency_ms or 0
            incurred = response.get("incurredCost")
            if incurred is None and response.get("cached"):
                incurred = 0
            if incurred is not None or metrics.get("incurredCost") is not None:
                base = metrics.get("incurredCost")
                base = metrics["cost"] if base is None else base
                metrics["incurredCost"] = base + (
                    incurred if incurred is not None else (result.cost or 0)
                )
            metrics["cost"] += result.cost or 0
            for name, value in (result.named_scores or {}).items():
                _accumulate_named_metric(
                    metrics, name, value, result.grading_result,
                    (result.test_case or {}).get("vars") or {},
                )  # fmt: skip
            components = (result.grading_result or {}).get("componentResults")
            if components:
                metrics["assertPassCount"] += sum(1 for c in components if c.get("pass"))
                metrics["assertFailCount"] += sum(1 for c in components if not c.get("pass"))
            if response.get("tokenUsage"):
                accumulate_response_token_usage(metrics["tokenUsage"], response)
            grading = result.grading_result or {}
            if grading.get("tokensUsed"):
                accumulate_grading_token_usage(
                    metrics["tokenUsage"], grading["tokensUsed"],
                    cached=(grading.get("metadata") or {}).get("cachedResponse"),
                )  # fmt: skip
    for idx, metrics in by_prompt.items():
        if idx < len(eval_record.prompts):
            eval_record.prompts[idx]["metrics"] = metrics
    if eval_record.persisted:
        await eval_record.append_prompts(eval_record.prompts)
    logger.debug("Prompt metrics recalculation completed")


# --- retry -----------------------------------------------------------------------------------


def _jsonl_output_paths(output_path: Any) -> list[str]:
    from promptphoo.output import get_output_file_format

    paths = output_path if isinstance(output_path, list) else [output_path]
    return [p for p in paths if isinstance(p, str) and get_output_file_format(p) == "jsonl"]


async def resolve_retry_configs(original_eval: Any, cmd: dict[str, Any]) -> dict[str, Any]:
    options = get_persisted_provider_filter_options(
        (original_eval.runtime_options or {}).get("providerFilter")
    )
    provider_filter = options.get("filterProviders")
    config_path = cmd.get("config")
    description = f'retry config "{config_path}"' if config_path else "saved evaluation config"
    regex_error = get_provider_filter_regex_error(provider_filter) if provider_filter else None
    if provider_filter and regex_error:
        raise ConfigResolutionError(
            f"Could not resolve the {description} using stored provider filter "
            f'"{provider_filter}": {regex_error}. Existing ERROR results were preserved.'
        )
    if config_path:
        configs = await resolve_configs({"config": [config_path], **options}, {})
    else:
        configs = await resolve_configs(dict(options), original_eval.config)
    # The original run filtered raw configs, then instantiated providers by id/label. Replay
    # both stages so the retried provider set matches the original.
    configs["test_suite"]["providers"] = filter_providers(
        configs["test_suite"]["providers"], provider_filter
    )
    if provider_filter and not configs["test_suite"]["providers"]:
        raise ConfigResolutionError(
            f'Stored provider filter "{provider_filter}" matched no providers in the '
            f"{description}. Existing ERROR results were preserved."
        )
    return configs


async def retry_command(eval_id: str, cmd: dict[str, Any]) -> Any:
    from promptphoo.models.eval import Eval

    logger.info("🔄 Retrying failed tests for evaluation: %s", eval_id)
    original = Eval.find_by_id(eval_id)
    if original is None:
        raise ValueError(f"Evaluation with ID {eval_id} not found")
    error_ids = get_error_result_ids(eval_id)
    if not error_ids:
        logger.info("✅ No ERROR results found in this evaluation")
        return original
    logger.info("Found %d ERROR results to retry", len(error_ids))
    resolved = await resolve_retry_configs(original, cmd)
    with (
        cli_state.with_config(resolved["config"]),
        cli_state.with_env(resolved["test_suite"].get("env") or cli_state.env()),
    ):
        return await _retry_with_config(original, error_ids, cmd, resolved)


async def _retry_with_config(
    original: Any, error_ids: list[str], cmd: dict[str, Any], resolved: dict[str, Any]
) -> Any:
    from promptphoo.evaluator.run import evaluate
    from promptphoo.output import write_multiple_outputs

    # ERROR rows are deleted only after a successful retry, so a failed retry can be re-run.
    logger.info("🔄 Running evaluation with resume mode to retry %d test cases...", len(error_ids))
    cli_state.resume = True
    cli_state.retry_mode = True
    cli_state.retry_error_result_ids = error_ids

    clo = resolved.get("command_line_options") or {}
    config_max = clo.get("maxConcurrency")
    max_concurrency = cmd.get("maxConcurrency")
    if max_concurrency is None and isinstance(config_max, int) and not isinstance(config_max, bool):
        max_concurrency = config_max
    config_delay = clo.get("delay")
    delay = cmd.get("delay")
    if delay is None and isinstance(config_delay, int) and not isinstance(config_delay, bool):
        delay = config_delay
    if delay and delay > 0:
        cli_state.set_max_concurrency(1)
        logger.info(
            "Running at concurrency=1 because %sms delay was requested between API calls", delay
        )
    elif max_concurrency is not None:
        cli_state.set_max_concurrency(max_concurrency)
    options = {
        "maxConcurrency": 1 if delay and delay > 0 else max_concurrency,
        "delay": delay,
        "eventSource": "cli",
        "restorePromptColumns": not cmd.get("config"),
        "showProgressBar": not cmd.get("verbose"),
    }
    try:
        retried: Any = await evaluate(
            resolved["test_suite"], original, {k: v for k, v in options.items() if v is not None}
        )
        jsonl_paths = _jsonl_output_paths(original.config.get("outputPath"))
        if retried.result_persistence_failed:
            if jsonl_paths:
                # Restore the pre-retry database view in the JSONL artifact.
                try:
                    write_multiple_outputs(jsonl_paths, retried, None)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "Retry results failed to persist, and restoring JSONL output failed. %s", e
                    )
            raise RuntimeError(
                "Retry results failed to persist. Existing ERROR rows were preserved."
            )
        rows_deleted = False
        try:
            delete_error_results(error_ids)
            rows_deleted = True
            await recalculate_prompt_metrics(retried)
        except Exception as e:  # noqa: BLE001 - cleanup failure is non-fatal
            logger.warning("Post-retry cleanup had issues. Retry results are saved. %s", e)
        if rows_deleted and jsonl_paths:
            try:
                retried.clear_results()
                write_multiple_outputs(jsonl_paths, retried, None)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Retry succeeded and the database is up to date, but rewriting JSONL output "
                    "failed. %s",
                    e,
                )
        logger.info("✅ Retry completed for evaluation: %s", original.id)
        if cmd.get("share"):
            logger.warning("--share is not supported by promptphoo; results stay local.")
        return retried
    finally:
        cli_state.resume = False
        cli_state.retry_mode = False
        cli_state.retry_error_result_ids = None
        cli_state.set_max_concurrency(None)


# --- click wiring ----------------------------------------------------------------------------

RECOVERY_HINT = textwrap.dedent(
    """

    Recovery options:
      - Run the same retry command again to continue
      - Check API credentials and network connectivity
      - Use --verbose for detailed error information
    """
)


def register(cli: click.Group, default_config: dict[str, Any], default_config_path: str | None):
    @cli.command("retry", help="Retry all ERROR results from a given evaluation")
    @click.argument("eval_id", metavar="<evalId>")
    @click.option(
        "-c",
        "--config",
        help="Path to configuration file (optional, uses original eval config if not provided)",
    )
    @click.option("-v", "--verbose", is_flag=True, default=None, help="Verbose output")
    @click.option("--max-concurrency", type=int, help="Maximum number of concurrent evaluations")
    @click.option("--delay", type=int, help="Delay between evaluations in milliseconds")
    @click.option(
        "--share/--no-share",
        default=None,
        help="Share results to cloud (auto-shares when cloud is configured)",
    )
    def retry_cmd(**opts: Any) -> None:
        cmd = {
            "config": opts["config"],
            "verbose": opts["verbose"],
            "maxConcurrency": opts["max_concurrency"],
            "delay": opts["delay"],
            "share": opts["share"],
        }
        cmd = {k: v for k, v in cmd.items() if v is not None}
        try:
            asyncio.run(retry_command(opts["eval_id"], cmd))
        except ConfigResolutionError as e:
            (logger.warning if e.log_level == "warn" else logger.error)("%s", e.cli_message)
            raise SystemExit(1) from e
        except Exception as e:  # noqa: BLE001 - upstream logs every failure and exits 1
            logger.error("Failed to retry evaluation: %s", e)
            logger.info("%s", RECOVERY_HINT)
            raise SystemExit(1) from e
        finally:
            db.close_db()

    return retry_cmd
