"""The end-of-eval summary printed by `promptfoo eval`.

Ports src/util/eval/summary.ts (`generateEvalSummary`) and src/util/formatDuration.ts. Lines carry
chalk colours when the terminal supports them; strip ANSI for plain text.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypedDict

from promptphoo.jsvalues import to_fixed
from promptphoo.output.ansi import chalk


class TokenUsageTracker(Protocol):
    """What the summary needs from upstream's TokenUsageTracker singleton."""

    def get_provider_ids(self) -> list[str]: ...

    def get_provider_usage(self, provider_id: str) -> Mapping[str, Any] | None: ...


class EvalSummaryParams(TypedDict, total=False):
    evalId: str
    isRedteam: bool
    writeToDatabase: bool
    shareableUrl: str | None
    wantsToShare: bool
    hasExplicitDisable: bool
    cloudEnabled: bool
    activelySharing: bool
    tokenUsage: Mapping[str, Any]
    successes: int
    failures: int
    errors: int
    cacheHits: int
    duration: float
    maxConcurrency: int
    # A TokenUsageTracker, or a plain {providerId: tokenUsage} mapping, or None.
    tracker: Any
    targetErrorStatus: int | None
    repeatStability: Mapping[str, Any] | None


def format_duration(seconds: float) -> str:
    """formatDuration: "2h 5m 30s", "5m 0s" or "45s"."""
    total = int(seconds // 1)
    if total < 60:
        return f"{total}s"
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    result = ""
    if hours > 0:
        result += f"{hours}h "
    if minutes > 0 or hours > 0:
        result += f"{minutes}m "
    return result + f"{secs}s"


def _loc(value: Any) -> str:
    """Number.prototype.toLocaleString() in an en-US locale."""
    from promptphoo.output.writer import to_locale_string

    return to_locale_string(value)


def _total(usage: Mapping[str, Any] | None) -> float:
    if not usage:
        return 0
    total = usage.get("total")
    if total is not None:
        return total
    return (usage.get("prompt") or 0) + (usage.get("completion") or 0)


def _completion_message(
    completion_type: str,
    eval_id: str,
    shareable_url: str | None,
    was_aborted: bool,
    write_to_database: bool,
    actively_sharing: bool,
) -> str:
    if was_aborted:
        suffix = f" (ID: {chalk(eval_id, 'cyan')})" if write_to_database else ""
        return f"{chalk('✗', 'red')} {completion_type} aborted{suffix}"
    if write_to_database and shareable_url:
        return f"{chalk('✓', 'green')} {completion_type} complete: {shareable_url}"
    if write_to_database and actively_sharing:
        return f"{chalk('✓', 'green')} {completion_type} complete"
    if write_to_database:
        return f"{chalk('✓', 'green')} {completion_type} complete (ID: {chalk(eval_id, 'cyan')})"
    return f"{chalk('✓', 'green')} {completion_type} complete"


def _abort_lines(status: int | None) -> list[str]:
    if status is None:
        return []
    return [
        "",
        chalk("Scan stopped: Target is unavailable and will not recover on retry.", "red", "bold"),
        chalk(f"  Target returned HTTP {status}", "red"),
        "",
        chalk("Possible causes:", "yellow"),
        chalk("  • Invalid API key or authentication (401/403)", "yellow"),
        chalk("  • Target endpoint does not exist (404)", "yellow"),
        chalk("  • Server does not support the request (501)", "yellow"),
        "",
        chalk("To fix: Check your target configuration and credentials.", "cyan"),
    ]


def _guidance_lines(p: Mapping[str, Any]) -> list[str]:
    if (
        not p.get("writeToDatabase")
        or p.get("shareableUrl")
        or p.get("wantsToShare")
        or p.get("activelySharing")
    ):
        return []
    # Upstream also suggests `promptfoo share` / promptfoo.app and promptfoo.dev/feedback; the
    # port has no sharing and its feedback goes to its own tracker (D62).
    view = chalk("npx promptfoo@latest view", "green", "bold")
    return [
        "",
        f"» View results in Node promptfoo's web viewer (same database): {view}",
        f"» Feedback on promptphoo: {chalk('promptphoo feedback', 'green', 'bold')}",
    ]


def _usage_details(usage: Mapping[str, Any], total: float) -> list[str]:
    parts: list[str] = []
    if usage.get("prompt") and usage["prompt"] > 0:
        parts.append(f"{_loc(usage['prompt'])} prompt")
    if usage.get("completion") and usage["completion"] > 0:
        parts.append(f"{_loc(usage['completion'])} completion")
    cached = usage.get("cached")
    if cached and cached > 0:
        parts.append("cached" if cached == total and not parts else f"{_loc(cached)} cached")
    reasoning = (usage.get("completionDetails") or {}).get("reasoning")
    if reasoning and reasoning > 0:
        parts.append(f"{_loc(reasoning)} reasoning")
    return parts


def _grading_line(assertions: Mapping[str, Any] | None) -> str | None:
    total = _total(assertions)
    if not assertions or (total == 0 and (assertions.get("cached") or 0) == 0):
        return None
    details = _usage_details(assertions, total)
    return f"  {chalk('Grading:', 'gray')} {chalk(_loc(total), 'white')} ({', '.join(details)})"


def _provider_ids_and_usage(tracker: Any) -> list[tuple[str, Mapping[str, Any] | None]]:
    if tracker is None:
        return []
    if isinstance(tracker, Mapping):
        return list(tracker.items())
    return [(pid, tracker.get_provider_usage(pid)) for pid in tracker.get_provider_ids()]


def _provider_lines(tracker: Any) -> list[str]:
    entries = _provider_ids_and_usage(tracker)
    if len(entries) <= 1:
        return []
    providers = [(pid, u) for pid, u in entries if u is not None]
    providers.sort(key=lambda item: -(item[1].get("total") or 0))
    lines = ["", chalk("Providers:", "bold")]
    for pid, usage in providers:
        total = usage.get("total") or 0
        if total == 0 and (usage.get("prompt") or 0) + (usage.get("completion") or 0) == 0:
            continue
        display_total = total or (usage.get("prompt") or 0) + (usage.get("completion") or 0)
        display_id = pid[: pid.index(" (")] if " (" in pid else pid
        details = _usage_details(usage, display_total)
        request_info = f"{usage.get('numRequests') or 0} requests"
        sep = "; " if details else ""
        lines.append(
            f"  {chalk(f'{display_id}:', 'gray')} {chalk(_loc(display_total), 'white')} "
            f"({request_info}{sep}{', '.join(details)})"
        )
    return lines


def _token_usage_lines(usage: Mapping[str, Any], is_redteam: bool, tracker: Any) -> list[str]:
    primary = _total(usage)
    grading_line = _grading_line(usage.get("assertions"))
    attacker = _total(usage.get("attacker"))
    generation = _total(usage.get("generation"))
    generation_requests = (usage.get("generation") or {}).get("numRequests") or 0
    if (
        primary == 0
        and grading_line is None
        and attacker == 0
        and generation == 0
        and generation_requests == 0
    ):
        return []
    grand_total = primary + attacker + _total(usage.get("assertions")) + generation
    lines = [f"{chalk('Total Tokens:', 'bold')} {chalk(_loc(grand_total), 'white', 'bold')}"]
    if is_redteam and usage.get("numRequests"):
        lines.append(f"  {chalk('Probes:', 'gray')} {chalk(_loc(usage['numRequests']), 'white')}")
    if primary > 0:
        label = "Target" if is_redteam else "Provider"
        parts = _usage_details(usage, primary)
        lines.append(
            f"  {chalk(f'{label}:', 'gray')} {chalk(_loc(primary), 'white')} ({', '.join(parts)})"
        )
    if usage.get("generation") and generation > 0:
        parts = _usage_details(usage["generation"], generation)
        lines.append(
            f"  {chalk('Generation:', 'gray')} {chalk(_loc(generation), 'white')} "
            f"({', '.join(parts)})"
        )
    elif generation_requests > 0:
        noun = "request" if generation_requests == 1 else "requests"
        lines.append(
            f"  {chalk('Generation:', 'gray')} token usage unavailable "
            f"({_loc(generation_requests)} {noun})"
        )
    if usage.get("attacker") and attacker > 0:
        parts = _usage_details(usage["attacker"], attacker)
        lines.append(
            f"  {chalk('Attacker:', 'gray')} {chalk(_loc(attacker), 'white')} ({', '.join(parts)})"
        )
    if grading_line:
        lines.append(grading_line)
    incurred = usage.get("incurredTokenUsage")
    if incurred:
        incurred_tokens = (
            _total(incurred)
            + _total(incurred.get("attacker"))
            + _total(incurred.get("assertions"))
            + _total(incurred.get("generation"))
        )
        savings = max(grand_total - incurred_tokens, 0)
        if savings > 0:
            lines.extend(
                [
                    f"{chalk('Incurred Tokens:', 'bold')} "
                    f"{chalk(_loc(incurred_tokens), 'white', 'bold')}",
                    f"  {chalk('Cached Savings:', 'gray')} {chalk(_loc(savings), 'white')}",
                    f"  {chalk('Actual Target Requests:', 'gray')} "
                    f"{chalk(_loc(incurred.get('numRequests') or 0), 'white')}",
                ]
            )
    lines.extend(_provider_lines(tracker))
    return lines


def _percentage(count: float, total: float) -> str:
    pct = 0 if total == 0 else (count / total) * 100
    return f"{to_fixed(pct, 0)}%" if pct in (0, 100) else f"{to_fixed(pct, 2)}%"


def _result_line(count: int, label: str, icon: str | None, color: str, total: int) -> str:
    icon_part = f"{chalk(icon, color)} " if icon else ""
    return (
        f"  {icon_part}{chalk(_loc(count), 'white', 'bold')} {chalk(label, 'white')} "
        f"{chalk(f'({_percentage(count, total)})', 'gray')}"
    )


def _results_lines(p: Mapping[str, Any]) -> list[str]:
    successes = p.get("successes") or 0
    failures = p.get("failures") or 0
    errors = p.get("errors") or 0
    cache_hits = p.get("cacheHits")
    total = successes + failures + errors
    lines = [
        "",
        chalk("Results:", "bold"),
        _result_line(successes, "passed", "✓" if successes > 0 else None, "green", total),
        _result_line(failures, "failed", "✗" if failures > 0 else None, "red", total),
        _result_line(errors, "error" if errors == 1 else "errors", "✗" if errors > 0 else None,
                     "red", total),
    ]  # fmt: skip
    if cache_hits:
        lines.append(_result_line(cache_hits, "cache hits", None, "gray", total))
    lines.extend(
        [
            chalk(
                f"Duration: {format_duration(p.get('duration') or 0)} "
                f"(concurrency: {p.get('maxConcurrency')})",
                "gray",
            ),
            "",
        ]
    )
    return lines


def _pass_rate(value: float) -> str:
    return f"{to_fixed(value * 100, 1)}%"


def _repeat_stability_lines(summary: Mapping[str, Any] | None) -> list[str]:
    if not summary:
        return []
    groups = summary.get("totalGroups") or 0
    unstable = summary.get("unstableGroups") or 0
    lines = [
        "",
        chalk("Repeat stability:", "bold"),
        f"  {_loc(groups)} repeated group{'' if groups == 1 else 's'}; {_loc(unstable)} unstable",
    ]
    for group in [g for g in summary.get("groups") or [] if g.get("unstable")][:5]:
        scored = group.get("passed", 0) + group.get("failed", 0)
        provider = group.get("provider") or {}
        label = (
            group.get("description")
            or group.get("promptLabel")
            or provider.get("label")
            or provider.get("id")
        )
        interval = group.get("passRateConfidenceInterval")
        interval_text = (
            f"; 95% CI {_pass_rate(interval['lower'])}–{_pass_rate(interval['upper'])}"
            if interval
            else ""
        )
        lines.append(
            chalk(
                f"  ⚠ {label}: {group.get('passed')}/{scored} passed across "
                f"{group.get('repetitions')} runs{interval_text}",
                "yellow",
            )
        )
    if unstable > 5:
        lines.append(chalk(f"  … and {unstable - 5} more unstable groups", "gray"))
    with_errors = summary.get("groupsWithErrors") or 0
    if with_errors > 0:
        lines.append(
            chalk(
                f"  {with_errors} repeated group{'' if with_errors == 1 else 's'} contained errors",
                "red",
            )
        )
    cached = summary.get("cachedResults") or 0
    if cached > 0:
        lines.append(
            chalk(
                f"  {cached} cached result{'' if cached == 1 else 's'} detected; confidence "
                "intervals are omitted for affected groups",
                "gray",
            )
        )
    return lines


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def generate_eval_summary(params: Mapping[str, Any] | None = None, **kwargs: Any) -> list[str]:
    """generateEvalSummary: the lines doEval logs after the results table.

    Takes Node's EvalSummaryParams as a camelCase mapping, or the same fields as snake_case
    keyword arguments (`eval_id=..., token_usage=..., max_concurrency=...`); keyword arguments
    override mapping entries.
    """
    params = {**(params or {}), **{_camel(k): v for k, v in kwargs.items()}}
    completion_type = "Red team" if params.get("isRedteam") else "Eval"
    return [
        _completion_message(
            completion_type,
            params.get("evalId") or "",
            params.get("shareableUrl"),
            params.get("targetErrorStatus") is not None,
            bool(params.get("writeToDatabase")),
            bool(params.get("activelySharing", False)),
        ),
        *_abort_lines(params.get("targetErrorStatus")),
        *_guidance_lines(params),
        "",
        *_token_usage_lines(
            params.get("tokenUsage") or {}, bool(params.get("isRedteam")), params.get("tracker")
        ),
        *_repeat_stability_lines(params.get("repeatStability")),
        *_results_lines(params),
    ]
