"""Token usage accounting. Ports src/util/tokenUsageUtils.ts and
src/contracts/completionDetails.ts. Usage objects are plain dicts with the Node field names."""

from __future__ import annotations

from typing import Any

TokenUsage = dict[str, Any]

_DETAIL_KEYS = (
    "reasoning",
    "acceptedPrediction",
    "rejectedPrediction",
    "cacheReadInputTokens",
    "cacheCreationInputTokens",
)


def create_empty_completion_details() -> dict[str, int]:
    return dict.fromkeys(_DETAIL_KEYS, 0)


def create_empty_assertions() -> TokenUsage:
    return {
        "total": 0,
        "prompt": 0,
        "completion": 0,
        "cached": 0,
        "numRequests": 0,
        "completionDetails": create_empty_completion_details(),
    }


def create_empty_token_usage() -> TokenUsage:
    return {
        "prompt": 0,
        "completion": 0,
        "cached": 0,
        "total": 0,
        "numRequests": 0,
        "completionDetails": create_empty_completion_details(),
        "assertions": create_empty_assertions(),
    }


def add_completion_details(target: dict | None, update: dict) -> dict[str, int]:
    return {k: ((target or {}).get(k) or 0) + (update.get(k) or 0) for k in _DETAIL_KEYS}


def _add(a: Any, b: Any) -> Any:
    return (a if a is not None else 0) + (b if b is not None else 0)


def clone_token_usage_breakdown(usage: TokenUsage) -> TokenUsage:
    out = {k: v for k, v in usage.items() if k != "incurredTokenUsage"}
    if out.get("completionDetails"):
        out["completionDetails"] = dict(out["completionDetails"])
    for key in ("attacker", "assertions", "generation"):
        if out.get(key):
            sub = dict(out[key])
            if sub.get("completionDetails"):
                sub["completionDetails"] = dict(sub["completionDetails"])
            out[key] = sub
    return out


def _accumulated_total(usage: TokenUsage) -> int:
    if usage.get("total") is not None:
        return usage["total"]
    component = (usage.get("prompt") or 0) + (usage.get("completion") or 0)
    cached = usage.get("cached") or 0
    if usage.get("numRequests") == 0 and cached > 0 and component <= cached:
        return 0
    return component


def accumulate_token_usage(
    target: TokenUsage, update: TokenUsage | None, increment_requests: bool = False
) -> None:
    """Add `update` into `target` in place."""
    if not update:
        return
    track_incurred = bool(target.get("incurredTokenUsage") or update.get("incurredTokenUsage"))
    if track_incurred and not target.get("incurredTokenUsage"):
        target["incurredTokenUsage"] = clone_token_usage_breakdown(target)

    target["prompt"] = _add(target.get("prompt"), update.get("prompt"))
    target["completion"] = _add(target.get("completion"), update.get("completion"))
    target["cached"] = _add(target.get("cached"), update.get("cached"))
    target["total"] = _add(target.get("total"), _accumulated_total(update))

    if update.get("numRequests") is not None:
        target["numRequests"] = _add(target.get("numRequests"), update["numRequests"])
    elif increment_requests:
        target["numRequests"] = (target.get("numRequests") or 0) + 1

    if update.get("completionDetails"):
        target["completionDetails"] = add_completion_details(
            target.get("completionDetails"), update["completionDetails"]
        )

    ua = update.get("assertions")
    if ua:
        ta: dict[str, Any] | None = target.get("assertions")
        if not ta:
            ta = target["assertions"] = {
                "total": 0, "prompt": 0, "completion": 0, "cached": 0, "numRequests": 0,
            }  # fmt: skip
        ta["total"] = _add(ta.get("total"), _accumulated_total(ua))
        for key in ("prompt", "completion", "cached", "numRequests"):
            ta[key] = _add(ta.get(key), ua.get(key))
        if ua.get("completionDetails"):
            ta["completionDetails"] = add_completion_details(
                ta.get("completionDetails"), ua["completionDetails"]
            )

    for key in ("attacker", "generation"):
        if update.get(key):
            if not target.get(key):
                target[key] = create_empty_assertions()
            accumulate_token_usage(target[key], update[key])

    if track_incurred and target.get("incurredTokenUsage"):
        accumulate_token_usage(
            target["incurredTokenUsage"],
            update.get("incurredTokenUsage") or clone_token_usage_breakdown(update),
            increment_requests,
        )


def accumulate_response_token_usage(
    target: TokenUsage, response: dict[str, Any] | None, count_as_request: bool = True
) -> None:
    """Add a provider response's usage, counting cached replays as requests but not spend."""
    if not response:
        return
    reported: TokenUsage = response.get("tokenUsage") or {}
    cached = bool(response.get("cached"))
    reported_total = reported.get("total")
    if reported_total is None:
        reported_total = (reported.get("prompt") or 0) + (reported.get("completion") or 0)
    if not count_as_request:
        requests = 0
    elif cached:
        requests = max(reported.get("numRequests") or 0, 1)
    else:
        requests = reported["numRequests"] if reported.get("numRequests") is not None else 1

    logical: TokenUsage = dict(reported)
    if cached and reported_total == 0 and (reported.get("cached") or 0) > 0:
        logical["total"] = reported["cached"]
    if cached:
        logical["cached"] = max(reported.get("cached") or 0, reported_total)
    logical["numRequests"] = requests

    if reported.get("incurredTokenUsage"):
        incurred = clone_token_usage_breakdown(reported["incurredTokenUsage"])
    elif cached:
        incurred = {k: reported[k] for k in ("attacker", "assertions", "generation")
                    if reported.get(k)}  # fmt: skip
        incurred["numRequests"] = 0
    else:
        incurred = clone_token_usage_breakdown(logical)
    if not count_as_request:
        incurred["numRequests"] = 0

    if target.get("incurredTokenUsage") or reported.get("incurredTokenUsage") or cached:
        logical["incurredTokenUsage"] = incurred
    accumulate_token_usage(target, logical)


def accumulate_assertion_token_usage(target: TokenUsage, update: TokenUsage | None) -> None:
    if not update:
        return
    target["total"] = _add(target.get("total"), _accumulated_total(update))
    for key in ("prompt", "completion", "cached", "numRequests"):
        target[key] = _add(target.get(key), update.get(key))
    if update.get("completionDetails"):
        target["completionDetails"] = add_completion_details(
            target.get("completionDetails"), update["completionDetails"]
        )


def accumulate_grading_request(
    assertions: TokenUsage,
    tokens_used: TokenUsage | None,
    cached: bool | None = None,
    fresh: bool | None = None,
) -> None:
    if not tokens_used:
        if not cached:
            assertions["numRequests"] = (assertions.get("numRequests") or 0) + 1
        return
    total = tokens_used.get("total")
    if total is None:
        total = (tokens_used.get("prompt") or 0) + (tokens_used.get("completion") or 0)
    cached_tokens = tokens_used.get("cached") or 0
    inferred = (
        cached is None
        and tokens_used.get("numRequests") == 0
        and cached_tokens > 0
        and total <= cached_tokens
    )
    if cached is True or inferred:
        requests = 0
    elif tokens_used.get("numRequests") == 0:
        requests = 1 if (fresh is True or total > 0) else 0
    else:
        n = tokens_used.get("numRequests")
        requests = n if n is not None else 1
    accumulate_assertion_token_usage(assertions, {**tokens_used, "numRequests": requests})


def accumulate_grading_token_usage(
    target: TokenUsage, tokens_used: TokenUsage | None, cached: bool | None = None
) -> None:
    """Add a GradingResult's tokensUsed to a row's tokenUsage.assertions."""
    used = tokens_used or {}
    total = used.get("total")
    if total is None:
        total = (used.get("prompt") or 0) + (used.get("completion") or 0)
    cached_tokens = used.get("cached") or 0
    cached_response = cached is True or (
        cached is None
        and used.get("numRequests") == 0
        and cached_tokens > 0
        and total <= cached_tokens
    )
    logical = create_empty_assertions()
    logical_used = (
        {**used, "total": cached_tokens}
        if cached_response and total == 0 and cached_tokens > 0
        else tokens_used
    )
    accumulate_grading_request(
        logical, logical_used, cached=False if cached_response else cached,
        fresh=True if cached_response else None,
    )  # fmt: skip
    incurred = create_empty_assertions()
    if not cached_response:
        accumulate_grading_request(
            incurred, used.get("incurredTokenUsage") or tokens_used, cached=cached
        )
    update: TokenUsage = {"assertions": logical}
    if target.get("incurredTokenUsage") or used.get("incurredTokenUsage") or cached_response:
        update["incurredTokenUsage"] = {"assertions": incurred}
    accumulate_token_usage(target, update)
