"""The comparison pass of the eval loop: select-best and max-score, which grade all outputs of
one test together after every row has run. Ports the comparison helpers and the
processComparisonAssertions / processSelectBestAssertions / processMaxScoreAssertions methods
of src/evaluator.ts, plus updateAssertionMetrics.

Results here are the store's EvalResult records (models/eval.py). They are mutated in place and,
when the eval is persisted, written back through `store.save_result` as upstream does. Resume,
retry-errors and grading cancellation are not ported yet (see D29).
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Any

from promptphoo import cache
from promptphoo.jsonutil import js_deep_equal, js_json_parse
from promptphoo.providers import canonicalize_provider_id
from promptphoo.providers.base import is_api_provider
from promptphoo.sanitizer import (
    REDACTED,
    is_non_credential_header,
    is_provider_type_map,
    is_secret_env_var_name,
    is_secret_field,
    safe_json_stringify,
    sanitize_object,
    sanitize_url,
)
from promptphoo.scheduler.provider_call_execution_context import (
    with_provider_call_execution_context,
)
from promptphoo.tokens import (
    accumulate_assertion_token_usage,
    accumulate_grading_request,
    clone_token_usage_breakdown,
    create_empty_assertions,
)

if TYPE_CHECKING:
    from promptphoo.evaluator.run import Evaluator, RunEvalOptions

logger = logging.getLogger(__name__)

PROMPTFOO_METADATA_KEY = "__promptfoo"
ABORTED_GRADING_PREFIX = "Aborted: "
COMPARISON_ERROR_PREFIX = "Error grading select-best"
COMPARISON_RESUME_ERROR = (
    "Cannot resume select-best with this grader configuration. Supply a grader configuration "
    "matching the saved result, or rerun the evaluation."
)
_ERROR = 2  # ResultFailureReason.ERROR
_ASSERT = 1
_NONE = 0


def get_result_index_key(result: Any) -> str:
    if isinstance(result, dict):
        return f"{result['testIdx']}:{result['promptIdx']}"
    test_idx = getattr(result, "test_idx", None)
    return f"{test_idx}:{result.prompt_idx}"


# --- token usage ----------------------------------------------------------------------------


def update_assertion_metrics(
    metrics: dict[str, Any], assertion_tokens: dict[str, Any] | None, cached: bool | None = None
) -> None:
    """updateAssertionMetrics: add assertion tokens to `metrics.tokenUsage`."""
    token_usage = metrics.get("tokenUsage")
    if token_usage is None or not assertion_tokens:
        return
    reported_total = assertion_tokens.get("total")
    if reported_total is None:
        reported_total = (assertion_tokens.get("prompt") or 0) + (
            assertion_tokens.get("completion") or 0
        )
    cached_tokens = assertion_tokens.get("cached") or 0
    cached_response = cached is True or (
        cached is None
        and assertion_tokens.get("numRequests") == 0
        and cached_tokens > 0
        and reported_total <= cached_tokens
    )
    if cached_response and not token_usage.get("incurredTokenUsage"):
        token_usage["incurredTokenUsage"] = clone_token_usage_breakdown(token_usage)
    if not token_usage.get("assertions"):
        token_usage["assertions"] = create_empty_assertions()
    accumulate_assertion_token_usage(token_usage["assertions"], assertion_tokens)
    incurred = token_usage.get("incurredTokenUsage")
    if incurred and not cached_response:
        if incurred.get("assertions") is None:
            incurred["assertions"] = create_empty_assertions()
        accumulate_assertion_token_usage(
            incurred["assertions"], assertion_tokens.get("incurredTokenUsage") or assertion_tokens
        )


def _result_has_model_graded_assertion(result: Any) -> bool:
    from promptphoo.assertions.index import MODEL_GRADED_ASSERTION_TYPES

    return any(
        (a or {}).get("type") in MODEL_GRADED_ASSERTION_TYPES
        for a in (result.test_case or {}).get("assert") or []
    )


def merge_comparison_token_usage(
    result: Any, grading_result: dict[str, Any], eval_token_usage: dict[str, Any]
) -> None:
    gr = result.grading_result
    if not gr or not grading_result.get("tokensUsed"):
        return
    if not gr.get("tokensUsed"):
        gr["tokensUsed"] = {"total": 0, "prompt": 0, "completion": 0}
    row_tokens_used = gr["tokensUsed"]
    row_metrics: dict[str, Any] = {"tokenUsage": {"assertions": row_tokens_used}}
    if row_tokens_used.get("incurredTokenUsage"):
        row_metrics["tokenUsage"]["incurredTokenUsage"] = {
            "assertions": row_tokens_used["incurredTokenUsage"]
        }
    cached = (grading_result.get("metadata") or {}).get("cachedResponse")
    update_assertion_metrics(row_metrics, grading_result["tokensUsed"], cached=cached)
    incurred = (row_metrics["tokenUsage"].get("incurredTokenUsage") or {}).get("assertions")
    if incurred:
        row_tokens_used["incurredTokenUsage"] = incurred
    if _result_has_model_graded_assertion(result):
        update_assertion_metrics(
            {"tokenUsage": eval_token_usage}, grading_result["tokensUsed"], cached=cached
        )


# --- merging a comparison verdict into a row ------------------------------------------------


def merge_select_best_grading_result(
    result: Any, grading_result: dict[str, Any], eval_token_usage: dict[str, Any]
) -> None:
    merge_comparison_token_usage(result, grading_result, eval_token_usage)
    gr = result.grading_result
    if gr:
        if (gr.get("metadata") or {}).get("cachedResponse") is True and (
            grading_result.get("metadata") or {}
        ).get("cachedResponse") is not True:
            metadata = {k: v for k, v in gr["metadata"].items() if k != "cachedResponse"}
            if metadata:
                gr["metadata"] = metadata
            else:
                gr.pop("metadata", None)
        gr["pass"] = bool(gr.get("pass")) and bool(grading_result.get("pass"))
        result.success = gr["pass"]
        if not grading_result.get("pass"):
            gr["reason"] = grading_result.get("reason")
            gr["score"] = grading_result.get("score")
            result.score = gr["score"]
        if not gr.get("componentResults"):
            gr["componentResults"] = []
        gr["componentResults"].append(grading_result)
        return
    new_pass = bool(result.success) and bool(grading_result.get("pass"))
    result.grading_result = {**grading_result, "pass": new_pass}
    result.success = new_pass
    if not grading_result.get("pass"):
        result.grading_result["score"] = grading_result.get("score")
        result.score = grading_result.get("score")


def merge_max_score_grading_result(result: Any, grading_result: dict[str, Any]) -> None:
    existing = result.grading_result
    existing_components = (existing or {}).get("componentResults") or []
    comparison_passed = bool(grading_result.get("pass"))
    previous_pass = (
        existing["pass"] if existing and existing.get("pass") is not None else result.success
    )
    next_pass = bool(previous_pass) and comparison_passed
    if comparison_passed:
        new_score = (
            existing["score"] if existing and existing.get("score") is not None else result.score
        )
    else:
        new_score = grading_result.get("score")
    if not comparison_passed and previous_pass:
        reason = grading_result.get("reason")
    else:
        reason = existing["reason"] if existing and existing.get("reason") is not None else ""
    result.grading_result = {
        **(existing or {}),
        "pass": next_pass,
        "score": new_score,
        "reason": reason,
        "componentResults": [*existing_components, grading_result],
        "namedScores": {
            **((existing or {}).get("namedScores") or {}),
            **(grading_result.get("namedScores") or {}),
        },
        "tokensUsed": (existing or {}).get("tokensUsed") or grading_result.get("tokensUsed"),
        "assertion": grading_result.get("assertion"),
    }
    result.success = next_pass
    if not comparison_passed:
        result.score = new_score


# --- comparison error state (metadata.__promptfoo.comparisonError) --------------------------


def has_completed_comparison(result: Any, type_: str) -> bool:
    # Target failures have no comparison component and cannot be regraded by resume.
    if result.failure_reason == _ERROR and not get_comparison_error(result):
        return True
    gr = result.grading_result or {}
    return (gr.get("assertion") or {}).get("type") == type_ or any(
        (c.get("assertion") or {}).get("type") == type_ for c in gr.get("componentResults") or []
    )


def get_comparison_error(result: Any) -> dict[str, Any] | None:
    metadata = result.metadata if not isinstance(result, dict) else result.get("metadata")
    internal = (metadata or {}).get(PROMPTFOO_METADATA_KEY)
    state = internal.get("comparisonError") if isinstance(internal, dict) else None
    if (
        isinstance(state, dict)
        and isinstance(state.get("success"), bool)
        and isinstance(state.get("score"), int | float)
        and not isinstance(state.get("score"), bool)
        and state.get("failureReason") in (_NONE, _ASSERT)
    ):
        return state
    return None


def set_comparison_error(result: Any, state: dict[str, Any] | None = None) -> None:
    is_row = isinstance(result, dict)
    metadata = result.get("metadata") if is_row else result.metadata
    internal = (metadata or {}).get(PROMPTFOO_METADATA_KEY)
    if not state and not (isinstance(internal, dict) and internal.get("comparisonError")):
        return
    fields = dict(internal) if isinstance(internal, dict) else {}
    if state:
        fields["comparisonError"] = state
    else:
        fields.pop("comparisonError", None)
    new_metadata = {**(metadata or {}), PROMPTFOO_METADATA_KEY: fields}
    if not fields:
        del new_metadata[PROMPTFOO_METADATA_KEY]
    if is_row:
        result["metadata"] = new_metadata
    else:
        result.metadata = new_metadata


# --- grader references ----------------------------------------------------------------------

_REDACTED_URL_ENCODED = re.compile(r"%(?:25)*5BREDACTED%(?:25)*5D", re.IGNORECASE)
_URL_LIKE = re.compile(r"^(?:[a-z][a-z0-9+.-]*://|/[^?#]*[?#])", re.IGNORECASE)
_URL_FIELD = re.compile(r"(?:url|uri|host|endpoint|proxy)$", re.IGNORECASE)
_INSTANCE_ID = re.compile(r"^\[.+ Instance\]$")


def has_comparison_redactions(
    value: Any,
    field: str | None = None,
    parent_field: str | None = None,
    seen: set[int] | None = None,
) -> bool:
    seen = set() if seen is None else seen
    if isinstance(value, str):
        masked = (
            REDACTED in value or bool(_REDACTED_URL_ENCODED.search(value)) or ("***:***@" in value)
        )
        return masked and (
            (field is None and bool(_URL_LIKE.search(value)))
            or field == "id"
            or (
                field is not None
                and (
                    is_secret_field(field)
                    or bool(_URL_FIELD.search(field))
                    or (parent_field == "env" and is_secret_env_var_name(field))
                    or (
                        parent_field is not None
                        and parent_field.lower() == "headers"
                        and not is_non_credential_header(field)
                    )
                )
            )
        )
    if not value or not isinstance(value, dict | list) or id(value) in seen:
        return False
    seen.add(id(value))
    entries = value.items() if isinstance(value, dict) else ((str(i), v) for i, v in
                                                            enumerate(value))  # fmt: skip
    return any(
        has_comparison_redactions(key, None, None, seen)
        or has_comparison_redactions(entry, key, field, seen)
        for key, entry in entries
    )


def comparison_provider_options(provider: Any) -> Any:
    if is_api_provider(provider):
        out: dict[str, Any] = {"id": provider.id()}
        for key, value in (("label", provider.label), ("config", provider.config),
                           ("env", provider.env)):  # fmt: skip
            if value is not None:
                out[key] = value
        return out
    return provider


def comparison_provider_id(provider: Any) -> str | None:
    if isinstance(provider, str):
        pid: Any = provider
    elif is_api_provider(provider):
        pid = provider.id()
    elif isinstance(provider, dict):
        pid = provider.get("id")
    else:
        pid = None
    if not isinstance(pid, str):
        return None
    canonical = canonicalize_provider_id(pid) or pid
    return sanitize_url(canonical) if "://" in canonical or canonical.startswith("/") else canonical


def persist_comparison_provider(provider: Any) -> Any:
    if is_provider_type_map(provider):
        options: Any = {**provider, "text": persist_comparison_provider(provider.get("text"))}
    else:
        options = comparison_provider_options(provider)
    return sanitize_object(options, sanitize_urls=True, max_depth=math.inf, throw_on_error=True)


def comparison_provider_settings(provider: Any) -> Any:
    persisted = persist_comparison_provider(provider)
    options = {"id": persisted} if isinstance(persisted, str) else (persisted or {})
    # Compare every nonsecret option, including endpoint/account settings and env overrides.
    merged = {k: v for k, v in {**options, "id": comparison_provider_id(provider)}.items()
              if k != "label"}  # fmt: skip
    text = safe_json_stringify(merged)
    return js_json_parse(text) if text is not None else None


def restore_comparison_provider(saved: Any, configured: Any, current_result: bool) -> Any:
    if not saved or is_api_provider(saved):
        return saved
    configured_text = configured.get("text") if is_provider_type_map(configured) else configured
    if is_provider_type_map(saved):
        return {
            **saved,
            "text": restore_comparison_provider(saved.get("text"), configured_text, current_result),
        }
    saved_id = comparison_provider_id(saved)
    if (
        not saved_id
        or saved_id == REDACTED
        or saved_id.startswith("[Function] ")
        or _INSTANCE_ID.search(saved_id)
    ):
        raise ValueError(
            "Cannot resume select-best because the saved runtime grader has no provider ID. "
            "Rerun the evaluation."
        )
    same_id = saved_id == comparison_provider_id(configured_text)
    can_restore = same_id and (
        current_result
        or js_deep_equal(
            comparison_provider_settings(saved), comparison_provider_settings(configured_text)
        )
    )
    if can_restore:
        if has_comparison_redactions(comparison_provider_options(configured_text)):
            raise ValueError(COMPARISON_RESUME_ERROR)
        # Use the actual descriptor or handle, including credentials embedded in URLs.
        return configured_text
    if has_comparison_redactions(saved) or (same_id and is_api_provider(configured_text)):
        raise ValueError(COMPARISON_RESUME_ERROR)
    return saved


def snapshot_comparison_provider(provider: Any, seen: dict[int, Any] | None = None) -> Any:
    if not isinstance(provider, dict | list):
        return provider
    seen = {} if seen is None else seen
    if id(provider) in seen:
        return seen[id(provider)]
    if isinstance(provider, list):
        out_list: list[Any] = []
        seen[id(provider)] = out_list
        out_list.extend(snapshot_comparison_provider(v, seen) for v in provider)
        return out_list
    out: dict[str, Any] = {}
    seen[id(provider)] = out
    for key, value in provider.items():
        out[key] = snapshot_comparison_provider(value, seen)
    return out


def get_comparison_providers(test: dict[str, Any], current_result: bool = False) -> dict:
    assertion = next((a for a in test.get("assert") or [] if a.get("type") == "select-best"), None)
    provider = (assertion or {}).get("provider") or (test.get("options") or {}).get("provider")
    return {"provider": snapshot_comparison_provider(provider), "currentResult": current_result}


def persist_comparison_test_case(test_case: dict[str, Any]) -> dict[str, Any]:
    """The test case as persistEvalRow stores it for a select-best row: grader references are
    replaced by their sanitized, serializable form."""
    out = dict(test_case)
    options = test_case.get("options")
    if options:
        new_options = dict(options)
        if "provider" in options:
            new_options["provider"] = persist_comparison_provider(options["provider"])
        out["options"] = new_options
    if test_case.get("assert") is not None:
        out["assert"] = [
            {**a, "provider": persist_comparison_provider(a.get("provider"))}
            if a.get("type") == "select-best" and "provider" in a
            else a
            for a in test_case["assert"]
        ]
    return out


# --- row bookkeeping ------------------------------------------------------------------------


def mark_comparison_rows(steps: list[RunEvalOptions]) -> tuple[dict[int, None], dict[int, None]]:
    """markComparisonRows: insertion-ordered sets (JS Set) of testIdx values."""
    select_best: dict[int, None] = {}
    max_score: dict[int, None] = {}
    for step in steps:
        asserts = step.test.get("assert") or []
        if any(a.get("type") == "select-best" for a in asserts):
            select_best[step.test_idx] = None
        if any(a.get("type") == "max-score" for a in asserts):
            max_score[step.test_idx] = None
    return select_best, max_score


def build_repeat_cache_context_by_test_idx(steps: list[RunEvalOptions]) -> dict[int, Any]:
    return {step.test_idx: step for step in steps}


# --- the pass -------------------------------------------------------------------------------


class ComparisonPass:
    """The comparison methods of upstream's Evaluator, bound to one evaluator."""

    def __init__(self, evaluator: Evaluator) -> None:
        self.ev = evaluator
        self.comparison_providers: dict[str, dict[str, Any]] = {}
        self.current_result_keys: set[str] = set()

    @property
    def store(self) -> Any:
        return self.ev.store

    # persistEvalRow (the comparison part)
    def prepare_row(self, row: dict[str, Any]) -> None:
        self.current_result_keys.add(get_result_index_key(row))
        set_comparison_error(row)
        test_case = row.get("testCase") or {}
        if any(a.get("type") == "select-best" for a in test_case.get("assert") or []):
            # Capture grader references before a later hook can replace shared nested fields.
            self.comparison_providers[get_result_index_key(row)] = get_comparison_providers(
                test_case, True
            )
            row["testCase"] = persist_comparison_test_case(test_case)

    def update_comparison_stats(
        self,
        result: Any,
        passed: bool,
        reason: str,
        tokens_used: dict[str, Any] | None,
        was_success: bool,
        was_score: float,
        metrics: dict[str, Any] | None,
        grading_cached: bool | None = None,
    ) -> None:
        if metrics:
            metrics["assertPassCount"] += 1 if passed else 0
            metrics["assertFailCount"] += 0 if passed else 1
            if tokens_used:
                update_assertion_metrics(metrics, tokens_used, cached=grading_cached)
            if not passed and result.score != was_score:
                metrics["score"] += result.score - was_score
        if was_success and not result.success:
            result.failure_reason = _ASSERT
            result.error = reason
            if metrics:
                metrics["testPassCount"] -= 1
                metrics["testFailCount"] += 1
            if get_result_index_key(result) in self.current_result_keys:
                self.ev.stats.successes -= 1
                self.ev.stats.failures += 1

    def update_comparison_result_counts(
        self, result: Any, previous: dict[str, Any], metrics: dict[str, Any] | None
    ) -> None:
        def outcome(success: Any, failure_reason: Any) -> str:
            if success:
                return "successes"
            return "errors" if failure_reason == _ERROR else "failures"

        before = outcome(previous["success"], previous["failureReason"])
        after = outcome(result.success, result.failure_reason)
        if before != after:
            if get_result_index_key(result) in self.current_result_keys:
                setattr(self.ev.stats, before, getattr(self.ev.stats, before) - 1)
                setattr(self.ev.stats, after, getattr(self.ev.stats, after) + 1)
            if metrics:
                counts = {"successes": "testPassCount", "failures": "testFailCount",
                          "errors": "testErrorCount"}  # fmt: skip
                metrics[counts[before]] -= 1
                metrics[counts[after]] += 1
        if metrics:
            metrics["score"] += result.score - previous["score"]

    async def process(
        self,
        prompts: list[dict[str, Any]],
        repeat_cache_context_by_test_idx: dict[int, Any],
        rows_with_select_best: dict[int, None],
        rows_with_max_score: dict[int, None],
    ) -> None:
        """processComparisonAssertions (without grading cancellation)."""
        compare_rows_count = len(rows_with_select_best) + len(rows_with_max_score)
        compare_count = 0
        for test_idx in list(rows_with_select_best):
            compare_count += 1
            await self._process_select_best_for_test(
                compare_count, compare_rows_count, prompts, repeat_cache_context_by_test_idx,
                test_idx,
            )  # fmt: skip
        if rows_with_max_score:
            logger.info("Processing %d max-score assertions...", len(rows_with_max_score))
        for test_idx in list(rows_with_max_score):
            compare_count += 1
            await self._process_max_score_for_test(prompts, test_idx)

    async def get_results_to_compare(self, test_idx: int) -> list[Any]:
        if self.store.persisted:
            return await self.store.read_results_by_test_idx(test_idx)
        return [r for r in self.store.results if r.test_idx == test_idx]

    def _comparison_call_api_context(
        self, first: Any, repeat_cache_context: Any | None
    ) -> dict[str, Any]:
        provider_id = (first.provider or {}).get("id")
        original = next((p for p in self.ev.test_suite["providers"] if p.id() == provider_id), None)
        context: dict[str, Any] = {"getCache": cache.get_cache}
        if original is not None:
            context["originalProvider"] = original
        context.update(
            prompt=first.prompt,
            promptIdx=first.prompt_idx,
            repeatIndex=repeat_cache_context.repeat_index if repeat_cache_context else None,
            testIdx=first.test_idx,
            vars=(first.test_case or {}).get("vars") or {},
        )
        if context["repeatIndex"] is None:
            del context["repeatIndex"]
        return context

    async def _process_select_best_for_test(
        self,
        compare_count: int,
        compare_rows_count: int,
        prompts: list[dict[str, Any]],
        repeat_cache_context_by_test_idx: dict[int, Any],
        test_idx: int,
    ) -> None:
        from promptphoo.assertions.index import run_compare_assertion
        from promptphoo.evaluator.run import get_repeat_cache_namespace

        results = await self.get_results_to_compare(test_idx)
        if not results:
            logger.warning("Expected results to be found for test index %s", test_idx)
            return
        first = results[0]
        saved_test = first.test_case or {}
        assertion = next(
            (a for a in saved_test.get("assert") or [] if a.get("type") == "select-best"), None
        )
        if not assertion:
            return
        try:
            providers = self.comparison_providers.get(get_result_index_key(first))
            # Persisted rows retain each column's criteria and vars. Only grader references
            # and redacted credentials need to come from the current configuration.
            comparison_test_case = {
                **saved_test,
                "options": {**(saved_test.get("options") or {}), "provider": None},
            }
            compare_assertion = {
                **assertion,
                "provider": restore_comparison_provider(
                    assertion.get("provider") or (saved_test.get("options") or {}).get("provider"),
                    (providers or {}).get("provider"),
                    (providers or {}).get("currentResult", False),
                ),
            }
            repeat_ctx = repeat_cache_context_by_test_idx.get(test_idx)
            outputs = [(r.response or {}).get("output") or "" for r in results]
            namespace = (
                get_repeat_cache_namespace(repeat_ctx.repeat_index, repeat_ctx.evaluate_options)
                if repeat_ctx
                else None
            )
            with cache.with_cache_namespace(namespace):
                grading_results = await with_provider_call_execution_context(
                    {"abortSignal": None, "rateLimitRegistry": self.ev.rate_limit_registry},
                    lambda: run_compare_assertion(
                        comparison_test_case, compare_assertion, outputs,
                        self._comparison_call_api_context(first, repeat_ctx),
                    ),
                )  # fmt: skip
        except Exception as error:
            grader_id = comparison_provider_id(
                assertion.get("provider")
                if assertion.get("provider") is not None
                else (saved_test.get("options") or {}).get("provider")
            )
            # Provider errors can contain credentials or config source snippets, so saved
            # results get a generic reason; the cause is logged only at debug level.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[Evaluator] select-best grading failed (graderId=%s, testIdx=%s): %r",
                    grader_id, test_idx, error,
                )  # fmt: skip
            message = (
                "Check the grader configuration and credentials. Supply a grader configuration "
                "matching the saved result to resume, or rerun the evaluation. Run with "
                "--verbose to log the underlying error."
            )
            reason = f"{COMPARISON_ERROR_PREFIX}{f' ({grader_id})' if grader_id else ''}: {message}"
            grading_results = []
            for result in results:
                if result.failure_reason == _ERROR and not get_comparison_error(result):
                    continue
                previous = {"success": result.success, "score": result.score,
                            "failureReason": result.failure_reason}  # fmt: skip
                # Keep the existing assertion verdict and outcome for a successful retry.
                if not get_comparison_error(result):
                    state = {**previous}
                    if result.error is not None:
                        state["error"] = result.error
                    set_comparison_error(result, state)
                result.error = reason
                result.failure_reason = _ERROR
                result.success = False
                result.score = 0
                self.update_comparison_result_counts(
                    result, previous, _metrics(prompts, result.prompt_idx)
                )
                if self.store.persisted:
                    await self.store.save_result(result)

        for index, grading_result in enumerate(grading_results):
            await self._apply_select_best(
                grading_result, _metrics(prompts, results[index].prompt_idx), results[index]
            )
        logger.debug("Model-graded comparison #%d of %d complete", compare_count,
                     compare_rows_count)  # fmt: skip

    async def _process_max_score_for_test(
        self, prompts: list[dict[str, Any]], test_idx: int
    ) -> None:
        from promptphoo.matchers.comparison import select_max_score

        results = await self.get_results_to_compare(test_idx)
        if not results:
            logger.warning("Expected results to be found for test index %s", test_idx)
            return
        assertion = next(
            (a for a in (results[0].test_case or {}).get("assert") or []
             if a.get("type") == "max-score"),
            None,
        )  # fmt: skip
        if not assertion:
            return
        outputs = [(r.response or {}).get("output") or "" for r in results]
        grading_results = await select_max_score(
            outputs,
            [
                {
                    "gradingResult": r.grading_result,
                    "unavailable": r.failure_reason == _ERROR and not get_comparison_error(r),
                }
                for r in results
            ],
            assertion,
        )
        logger.debug("Max-score assertion for test #%s complete", test_idx)
        for index, result in enumerate(results):
            await self._apply_max_score(
                {**grading_results[index], "assertion": assertion},
                _metrics(prompts, result.prompt_idx),
                result,
            )

    async def _finalize(
        self,
        grading_result: dict[str, Any],
        metrics: dict[str, Any] | None,
        result: Any,
        was_success: bool,
        was_score: float,
    ) -> None:
        cached = (grading_result.get("metadata") or {}).get("cachedResponse")
        self.update_comparison_stats(
            result,
            bool(grading_result.get("pass")),
            grading_result.get("reason") or "",
            grading_result.get("tokensUsed"),
            was_success,
            was_score,
            metrics,
            cached,
        )
        response = result.response
        if (
            response
            and response.get("cached")
            and response.get("tokenUsage")
            and (response["tokenUsage"].get("numRequests") or 0) == 0
            and grading_result.get("tokensUsed")
        ):
            comparison_usage = create_empty_assertions()
            accumulate_grading_request(
                comparison_usage, grading_result["tokensUsed"], cached=cached
            )
            if (comparison_usage.get("numRequests") or 0) > 0:
                response["tokenUsage"]["numRequests"] = 1
                self.ev.stats.token_usage["numRequests"] = (
                    self.ev.stats.token_usage.get("numRequests") or 0
                ) + 1
                if metrics:
                    metrics["tokenUsage"]["numRequests"] = (
                        metrics["tokenUsage"].get("numRequests") or 0
                    ) + 1
        if self.store.persisted:
            await self.store.save_result(result)

    def _restore_comparison_result(self, result: Any, metrics: dict[str, Any] | None) -> bool:
        if result.failure_reason == _ERROR:
            saved = get_comparison_error(result)
            if not saved:
                return False
            previous = {"success": result.success, "score": result.score,
                        "failureReason": result.failure_reason}  # fmt: skip
            result.success = saved["success"]
            result.score = saved["score"]
            result.failure_reason = saved["failureReason"]
            result.error = saved.get("error")
            set_comparison_error(result)
            self.update_comparison_result_counts(result, previous, metrics)
        return True

    async def _apply_select_best(
        self, grading_result: dict[str, Any], metrics: dict[str, Any] | None, result: Any
    ) -> None:
        if not self._restore_comparison_result(result, metrics):
            return
        was_success, was_score = result.success, result.score
        merge_select_best_grading_result(result, grading_result, self.ev.stats.token_usage)
        await self._finalize(grading_result, metrics, result, was_success, was_score)

    async def _apply_max_score(
        self, grading_result: dict[str, Any], metrics: dict[str, Any] | None, result: Any
    ) -> None:
        # A failed select-best grader must remain an error until that grader succeeds.
        if result.failure_reason == _ERROR and not (result.error or "").startswith(
            ABORTED_GRADING_PREFIX
        ):
            return
        if not self._restore_comparison_result(result, metrics):
            return
        was_success, was_score = result.success, result.score
        merge_max_score_grading_result(result, grading_result)
        await self._finalize(grading_result, metrics, result, was_success, was_score)


def _metrics(prompts: list[dict[str, Any]], prompt_idx: int) -> dict[str, Any] | None:
    if 0 <= prompt_idx < len(prompts):
        return prompts[prompt_idx].get("metrics")
    return None
