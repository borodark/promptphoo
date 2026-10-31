"""Weighted aggregation of assertion results. Ports src/assertions/assertionsResult.ts."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import Any

from promptphoo import envars
from promptphoo.assertions.script import as_grading_result
from promptphoo.jsonutil import js_deep_equal
from promptphoo.jsvalues import number_to_string, to_fixed
from promptphoo.tokens import add_completion_details

GUARDRAIL_BLOCKED_REASON = "Content failed guardrail safety checks"
_FIELDS = ("total", "prompt", "completion", "cached", "numRequests")


def default_tokens_used() -> dict[str, int]:
    return dict.fromkeys(_FIELDS, 0)


def _num(value: Any) -> float:
    return value if value is not None else 0


def _normalize_token_usage(result: dict[str, Any]) -> dict[str, Any] | None:
    used = result.get("tokensUsed")
    if not used:
        return None
    metadata = result.get("metadata") or {}
    if metadata.get("cachedResponse") is True:
        reported = used.get("total")
        if reported is None:
            reported = _num(used.get("prompt")) + _num(used.get("completion"))
        logical = reported or _num(used.get("cached"))
        return {
            **used,
            "total": logical,
            "prompt": _num(used.get("prompt")),
            "completion": _num(used.get("completion")),
            "cached": max(_num(used.get("cached")), logical),
            "numRequests": max(_num(used.get("numRequests")), 1),
            "incurredTokenUsage": default_tokens_used(),
        }
    out = dict(used)
    if metadata.get("renderedGradingPrompt") is not None and used.get("numRequests") == 0:
        out["numRequests"] = 1
    return out


def _clone_usage(usage: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in usage.items() if k != "incurredTokenUsage"}
    if out.get("completionDetails"):
        out["completionDetails"] = dict(out["completionDetails"])
    return out


def _accumulate(target: dict[str, Any], update: dict[str, Any]) -> None:
    track = bool(target.get("incurredTokenUsage") or update.get("incurredTokenUsage"))
    if track and not target.get("incurredTokenUsage"):
        target["incurredTokenUsage"] = _clone_usage(target)
    for f in _FIELDS:
        target[f] = _num(target.get(f)) + _num(update.get(f))
    if update.get("completionDetails"):
        target["completionDetails"] = add_completion_details(
            target.get("completionDetails"), update["completionDetails"]
        )
    if track and target.get("incurredTokenUsage"):
        _accumulate(
            target["incurredTokenUsage"], update.get("incurredTokenUsage") or _clone_usage(update)
        )


def _did_fresh_work(usage: dict[str, Any] | None) -> bool:
    return usage is not None and any(
        _num(usage.get(k)) > 0 for k in ("numRequests", "total", "prompt", "completion")
    )


def _merge_metadata(base: dict | None, incoming: dict | None) -> dict | None:
    if not base and not incoming:
        return None
    merged = {**(incoming or {}), **(base or {})}
    if base and base.get("assertionSet"):
        merged["assertionSet"] = base["assertionSet"]
    return merged


def _merge_scoring_metadata(base: dict | None, scoring: dict, base_used: dict | None) -> Any:
    metadata = _merge_metadata(base, scoring.get("metadata"))
    scoring_fresh = (scoring.get("metadata") or {}).get("cachedResponse") is not True and (
        _did_fresh_work(scoring.get("tokensUsed"))
    )
    components_fresh = (base or {}).get("cachedResponse") is not True and _did_fresh_work(
        base_used or {}
    )
    if (not scoring_fresh and not components_fresh) or (metadata or {}).get(
        "cachedResponse"
    ) is not True:
        return metadata
    assert metadata is not None
    remaining = {k: v for k, v in metadata.items() if k != "cachedResponse"}
    return remaining or None


def _merge_scoring_token_usage(base_used: dict | None, scoring: dict) -> dict | None:
    scoring_used = scoring.get("tokensUsed")
    if not scoring_used or scoring_used is base_used:
        return base_used
    smeta = scoring.get("metadata") or {}
    independent = (
        smeta.get("cachedResponse") is True or smeta.get("renderedGradingPrompt") is not None
    )
    if base_used and not independent and js_deep_equal(scoring_used, base_used):
        return base_used
    normalized = _normalize_token_usage(scoring)
    if not normalized:
        return base_used
    base_used = base_used or {}
    merged: dict[str, Any] = {f: _num(base_used.get(f)) for f in _FIELDS}
    if base_used.get("completionDetails"):
        merged["completionDetails"] = dict(base_used["completionDetails"])
    if base_used.get("incurredTokenUsage"):
        merged["incurredTokenUsage"] = _clone_usage(base_used["incurredTokenUsage"])
    used_tokens = any(_num(normalized.get(k)) > 0 for k in ("total", "prompt", "completion"))
    update = dict(normalized)
    if (
        smeta.get("cachedResponse") is not True
        and used_tokens
        and not normalized.get("numRequests")
    ):
        update["numRequests"] = 1
    _accumulate(merged, update)
    return merged


def _weighted(total: float, weight: float) -> float:
    if not math.isfinite(total) or not math.isfinite(weight):
        return math.nan
    return total / weight if weight > 0 else 0


def _set_metadata(assertion_set: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": assertion_set.get("type"),
        "assertionCount": len(assertion_set.get("assert") or []),
    }
    for key in ("metric", "threshold", "weight"):
        if assertion_set.get(key) is not None:
            out[key] = assertion_set[key]
    return out


class AssertionsResult:
    @staticmethod
    def no_asserts_result() -> dict[str, Any]:
        return {
            "pass": True,
            "score": 1,
            "reason": "No assertions",
            "tokensUsed": default_tokens_used(),
        }

    def __init__(
        self, threshold: float | None = None, parent_assertion_set: dict[str, Any] | None = None
    ) -> None:
        self.threshold = threshold
        self.parent_assertion_set = parent_assertion_set
        self.tokens_used: dict[str, Any] = default_tokens_used()
        self.total_score = 0.0
        self.total_weight = 0.0
        self.failed_reason: str | None = None
        self.component_results: dict[int, dict[str, Any]] = {}
        self.named_scores: dict[str, float] = {}
        self.named_score_weights: dict[str, float] = {}
        self.result: dict[str, Any] | None = None
        self.failed_content_safety_checks = False

    def add_result(
        self,
        index: int,
        result: dict[str, Any],
        metric: str | None = None,
        weight: float | None = None,
    ) -> None:
        weight = 1 if weight is None else weight
        self.total_score += result["score"] * weight
        self.total_weight += weight
        self.component_results[index] = result

        assertion = result.get("assertion") or {}
        if (
            assertion.get("type") == "guardrails"
            and (assertion.get("config") or {}).get("purpose") == "redteam"
            and not result.get("pass")
        ):
            self.failed_content_safety_checks = True

        metric_weight = 1 if weight == 0 else weight
        if metric:
            self.named_scores[metric] = (
                self.named_scores.get(metric, 0) + result["score"] * metric_weight
            )
            self.named_score_weights[metric] = self.named_score_weights.get(metric, 0) + (
                metric_weight
            )
        for name, score in (result.get("namedScores") or {}).items():
            if name == metric:
                continue
            incoming = result.get("namedScoreWeights") or {}
            w = (incoming[name] if incoming.get(name) is not None else 1) if name in incoming else 1
            w *= metric_weight
            self.named_scores[name] = self.named_scores.get(name, 0) + score * w
            self.named_score_weights[name] = self.named_score_weights.get(name, 0) + w

        used = _normalize_token_usage(result)
        if used:
            _accumulate(self.tokens_used, used)

        if result.get("pass"):
            return
        self.failed_reason = result.get("reason")
        if envars.get_env_bool("PROMPTFOO_SHORT_CIRCUIT_TEST_FAILURES"):
            raise RuntimeError(result.get("reason"))

    def _components(self) -> list[dict[str, Any]]:
        if not self.component_results:
            return []
        # Node assigns componentResults[index]; gaps (skipped select-*/max-score) become holes
        # that flatMap skips.
        ordered = [self.component_results[i] for i in sorted(self.component_results)]
        flat: list[dict[str, Any]] = []
        for r in ordered:
            flat.append(r)
            for sub in r.get("componentResults") or []:
                flat.append({**sub, "assertion": sub.get("assertion") or r.get("assertion")})
        return flat

    async def test_result(self, scoring_function: Callable[..., Any] | None = None) -> dict:
        if self.result is not None:
            return self.result
        score = _weighted(self.total_score, self.total_weight)
        passed = self.failed_reason is None
        reason = self.failed_reason if self.failed_reason is not None else "All assertions passed"
        if isinstance(self.threshold, int | float) and not (
            isinstance(self.threshold, float) and math.isnan(self.threshold)
        ):
            passed = score >= self.threshold
            comparator = "≥" if passed else "<"

            reason = (
                f"Aggregate score {to_fixed(score, 2)} {comparator} "
                f"{number_to_string(self.threshold)} threshold"
            )
        if self.failed_content_safety_checks:
            passed = True
            reason = GUARDRAIL_BLOCKED_REASON

        components = self._components()
        named = {
            k: _weighted(v, self.named_score_weights.get(k, 0))
            for k, v in self.named_scores.items()
        }
        cached = bool(self.component_results) and all(
            (r.get("metadata") or {}).get("cachedResponse") is True
            for r in self.component_results.values()
        )
        res: dict[str, Any] = {
            "pass": passed,
            "score": score,
            "reason": reason,
            "namedScores": named,
        }
        if self.named_score_weights:
            res["namedScoreWeights"] = dict(self.named_score_weights)
        res["tokensUsed"] = self.tokens_used
        res["componentResults"] = components
        if self.parent_assertion_set or cached:
            meta: dict[str, Any] = {}
            if self.parent_assertion_set:
                meta["assertionSet"] = _set_metadata(self.parent_assertion_set["assertionSet"])
            if cached:
                meta["cachedResponse"] = True
            res["metadata"] = meta
        self.result = res

        if scoring_function:
            try:
                value = scoring_function(
                    named,
                    {
                        "threshold": self.threshold,
                        "parentAssertionSet": self.parent_assertion_set,
                        "componentResults": components,
                        "tokensUsed": self.tokens_used,
                    },
                )
                if inspect.isawaitable(value):
                    value = await value
                scoring = as_grading_result(value)
                if not scoring:
                    raise ValueError("assertion scoring function must return a GradingResult")
                merged = {
                    **res,
                    **scoring,
                    "tokensUsed": _merge_scoring_token_usage(res.get("tokensUsed"), scoring),
                }
                if res.get("metadata") or scoring.get("metadata"):
                    merged["metadata"] = _merge_scoring_metadata(
                        res.get("metadata"), scoring, res.get("tokensUsed")
                    )
                self.result = res = merged
            except Exception as e:
                res["pass"] = False
                res["score"] = 0
                res["reason"] = f"Scoring function error: {e}"

        invalid = {
            m
            for field in ("namedScores", "namedScoreWeights")
            for m, v in (res.get(field) or {}).items()
            if not (isinstance(v, int | float) and math.isfinite(v))
        }
        if not math.isfinite(res["score"]) or invalid:
            res["pass"] = False
            res["score"] = 0
            res["reason"] = "Assertion aggregation error: scores or weights must remain finite"
            for field in ("namedScores", "namedScoreWeights"):
                if res.get(field) is not None:
                    res[field] = {k: v for k, v in res[field].items() if k not in invalid}
        return res
