"""Ports test/util/eval/summary.test.ts (generateEvalSummary) and formatDuration."""

from __future__ import annotations

import re
from typing import Any

import pytest

from promptphoo.output.ansi import chalk
from promptphoo.output.summary import format_duration, generate_eval_summary
from promptphoo.tokens import accumulate_token_usage, create_empty_token_usage

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


class MockTracker:
    def __init__(self, usage: dict[str, Any] | None = None) -> None:
        self.usage = usage or {}

    def get_provider_ids(self) -> list[str]:
        return list(self.usage)

    def get_provider_usage(self, provider_id: str) -> Any:
        return self.usage.get(provider_id)


def params(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "evalId": "eval-123",
        "isRedteam": False,
        "writeToDatabase": False,
        "shareableUrl": None,
        "wantsToShare": False,
        "hasExplicitDisable": False,
        "cloudEnabled": False,
        "tokenUsage": {"total": 0},
        "successes": 5,
        "failures": 0,
        "errors": 0,
        "duration": 5000,
        "maxConcurrency": 4,
        "tracker": MockTracker(),
    }
    base.update(overrides)
    return base


def render(**overrides: Any) -> str:
    return strip_ansi("\n".join(generate_eval_summary(params(**overrides))))


@pytest.fixture(autouse=True)
def _colors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")


# --- completion message ---------------------------------------------------------------------


def test_basic_completion_message_when_not_writing_to_database() -> None:
    output = render()
    assert "✓ Eval complete" in output
    assert "eval-123" not in output


def test_eval_id_when_writing_to_database_without_shareable_url() -> None:
    output = render(evalId="eval-456", writeToDatabase=True)
    assert "✓ Eval complete (ID: eval-456)" in output


def test_shareable_url_when_available() -> None:
    output = render(
        writeToDatabase=True, shareableUrl="https://promptfoo.app/eval/abc123", wantsToShare=True
    )
    assert "✓ Eval complete: https://promptfoo.app/eval/abc123" in output


def test_red_team_complete() -> None:
    output = render(evalId="eval-rt", isRedteam=True, writeToDatabase=True)
    assert "✓ Red team complete (ID: eval-rt)" in output


def test_explains_non_retryable_target_errors_when_aborted() -> None:
    output = render(
        evalId="eval-aborted",
        writeToDatabase=True,
        successes=0,
        errors=1,
        duration=1000,
        targetErrorStatus=401,
    )
    assert "✗ Eval aborted (ID: eval-aborted)" in output
    assert "Scan stopped: Target is unavailable and will not recover on retry." in output
    assert "Target returned HTTP 401" in output
    assert "Server error (500)" not in output
    assert "To fix: Check your target configuration and credentials." in output


# --- token usage ----------------------------------------------------------------------------


def test_eval_tokens() -> None:
    output = render(tokenUsage={"total": 1000, "prompt": 400, "completion": 600, "cached": 0})
    assert "Tokens:" in output
    assert "Provider: 1,000 (400 prompt, 600 completion)" in output
    assert "Target:" not in output


def test_grading_tokens_only_when_no_eval_tokens() -> None:
    output = render(
        tokenUsage={
            "total": 0,
            "assertions": {"total": 500, "prompt": 200, "completion": 300, "cached": 0},
        }
    )
    assert "Tokens:" in output
    assert "Grading: 500 (200 prompt, 300 completion)" in output
    assert "Provider:" not in output


def test_cached_only_grading_usage() -> None:
    output = render(
        tokenUsage={"total": 0, "assertions": {"total": 0, "cached": 97, "numRequests": 0}},
        successes=1,
        duration=1000,
        maxConcurrency=1,
    )
    assert "Total Tokens: 0" in output
    assert "Grading: 0 (97 cached)" in output
    assert "Provider:" not in output


def test_cached_only_grading_separate_from_target_totals() -> None:
    output = render(
        isRedteam=True,
        tokenUsage={
            "total": 100,
            "prompt": 60,
            "completion": 40,
            "numRequests": 1,
            "assertions": {"total": 0, "cached": 97, "numRequests": 0},
        },
        successes=1,
    )
    assert "Total Tokens: 100" in output
    assert "Target: 100 (60 prompt, 40 completion)" in output
    assert "Grading: 0 (97 cached)" in output
    assert "Total Tokens: 197" not in output


def test_eval_and_grading_tokens() -> None:
    output = render(
        tokenUsage={
            "total": 1000,
            "prompt": 400,
            "completion": 600,
            "assertions": {"total": 500, "prompt": 200, "completion": 300},
        }
    )
    assert "Provider: 1,000 (400 prompt, 600 completion)" in output
    assert "Grading: 500 (200 prompt, 300 completion)" in output


def test_separates_target_generation_attacker_grading() -> None:
    output = render(
        isRedteam=True,
        tokenUsage={
            "total": 100,
            "prompt": 60,
            "completion": 40,
            "numRequests": 2,
            "attacker": {"total": 50, "prompt": 30, "completion": 20, "numRequests": 3},
            "assertions": {"total": 25, "prompt": 15, "completion": 10},
            "generation": {"total": 40, "prompt": 25, "completion": 15, "numRequests": 4},
        },
        successes=1,
    )
    assert "Total Tokens: 215" in output
    assert "Target: 100 (60 prompt, 40 completion)" in output
    assert "Provider:" not in output
    assert "Generation: 40 (25 prompt, 15 completion)" in output
    assert "Attacker: 50 (30 prompt, 20 completion)" in output
    assert "Grading: 25 (15 prompt, 10 completion)" in output
    assert "Probes: 2" in output


def test_cached_footprint_vs_incurred() -> None:
    output = render(
        isRedteam=True,
        tokenUsage={
            "total": 295,
            "prompt": 201,
            "completion": 94,
            "cached": 295,
            "numRequests": 1,
            "assertions": {"total": 37, "prompt": 23, "completion": 14, "numRequests": 1},
            "incurredTokenUsage": {
                "total": 0,
                "numRequests": 0,
                "assertions": {"total": 37, "prompt": 23, "completion": 14, "numRequests": 1},
            },
        },
        successes=1,
    )
    assert "Total Tokens: 332" in output
    assert "Target: 295" in output
    assert "Grading: 37" in output
    assert "Incurred Tokens: 37" in output
    assert "Cached Savings: 295" in output
    assert "Actual Target Requests: 0" in output
    assert "Probes: 1" in output


def test_omits_incurred_when_all_fresh() -> None:
    output = render(
        isRedteam=True,
        tokenUsage={
            "total": 100,
            "numRequests": 1,
            "incurredTokenUsage": {"total": 100, "numRequests": 1},
        },
        successes=1,
    )
    assert "Total Tokens: 100" in output
    assert "Incurred Tokens:" not in output


def test_derives_category_totals_from_prompt_and_completion() -> None:
    usage = create_empty_token_usage()
    accumulate_token_usage(
        usage,
        {
            "prompt": 60,
            "completion": 40,
            "numRequests": 2,
            "attacker": {"prompt": 30, "completion": 20, "numRequests": 3},
            "assertions": {"prompt": 15, "completion": 10},
            "generation": {"prompt": 25, "completion": 15, "numRequests": 4},
        },
    )
    output = render(isRedteam=True, tokenUsage=usage, successes=1)
    assert "Total Tokens: 215" in output
    assert "Target: 100 (60 prompt, 40 completion)" in output
    assert "Generation: 40 (25 prompt, 15 completion)" in output
    assert "Attacker: 50 (30 prompt, 20 completion)" in output
    assert "Grading: 25 (15 prompt, 10 completion)" in output


def test_unmetered_generation_requests() -> None:
    output = render(
        isRedteam=True,
        tokenUsage={
            "total": 0,
            "numRequests": 0,
            "generation": {"total": 0, "prompt": 0, "completion": 0, "numRequests": 1},
        },
        successes=0,
        errors=1,
    )
    assert "Total Tokens: 0" in output
    assert "Generation: token usage unavailable (1 request)" in output
    assert "Probes:" not in output


def test_attacker_only_usage() -> None:
    output = render(
        isRedteam=True,
        tokenUsage={
            "total": 0,
            "numRequests": 0,
            "attacker": {"total": 73, "prompt": 45, "completion": 28, "numRequests": 3},
        },
        successes=0,
        errors=1,
    )
    assert "Total Tokens: 73" in output
    assert "Attacker: 73 (45 prompt, 28 completion)" in output
    assert "Target:" not in output
    assert "Grading:" not in output
    assert "Probes:" not in output


def test_fully_cached() -> None:
    assert "Provider: 1,000 (cached)" in render(tokenUsage={"total": 1000, "cached": 1000})


def test_partially_cached() -> None:
    output = render(tokenUsage={"total": 1000, "prompt": 400, "completion": 600, "cached": 200})
    assert "Provider: 1,000 (400 prompt, 600 completion, 200 cached)" in output


def test_no_token_section_without_tokens() -> None:
    assert "Tokens:" not in render(tokenUsage={"total": 0})


# --- provider breakdown ---------------------------------------------------------------------


def test_provider_breakdown_with_request_counts() -> None:
    tracker = MockTracker(
        {
            "openai:gpt-4": {
                "total": 1500, "prompt": 600, "completion": 900, "cached": 0, "numRequests": 5,
            },
            "anthropic:claude-3": {
                "total": 800, "prompt": 300, "completion": 500, "cached": 0, "numRequests": 3,
            },
        }
    )  # fmt: skip
    output = render(tokenUsage={"total": 2300}, successes=8, tracker=tracker)
    assert "Providers:" in output
    assert "openai:gpt-4: 1,500 (5 requests; 600 prompt, 900 completion)" in output
    assert "anthropic:claude-3: 800 (3 requests; 300 prompt, 500 completion)" in output
    assert output.index("openai:gpt-4") < output.index("anthropic:claude-3")


def test_provider_request_count_even_when_zero() -> None:
    tracker = MockTracker(
        {
            "openai:gpt-4": {"total": 1000, "cached": 1000, "numRequests": 0},
            "anthropic:claude-3": {
                "total": 500,
                "prompt": 200,
                "completion": 300,
                "numRequests": 2,
            },
        }
    )
    output = render(tokenUsage={"total": 1500}, tracker=tracker)
    assert "openai:gpt-4: 1,000 (0 requests; cached)" in output
    assert "anthropic:claude-3: 500 (2 requests; 200 prompt, 300 completion)" in output


def test_provider_with_undefined_usage_and_mapping_tracker() -> None:
    usage = {
        "openai:gpt-4": {"total": 1000, "prompt": 400, "completion": 600, "numRequests": 5},
        "missing-provider": None,
        "anthropic:claude-3": {"total": 500, "prompt": 200, "completion": 300, "numRequests": 3},
    }
    for tracker in (MockTracker(usage), usage):
        output = render(tokenUsage={"total": 1500}, successes=8, tracker=tracker)
        assert "Providers:" in output
        assert "openai:gpt-4" in output
        assert "anthropic:claude-3" in output
        assert "missing-provider" not in output


def test_provider_id_suffix_is_trimmed() -> None:
    tracker = MockTracker(
        {"echo (a)": {"total": 2, "numRequests": 1}, "other": {"total": 1, "numRequests": 1}}
    )
    output = render(tokenUsage={"total": 3}, tracker=tracker)
    assert "  echo: 2 (1 requests)" in output


# --- results --------------------------------------------------------------------------------


def test_percentages_at_100_percent() -> None:
    output = render(successes=10)
    assert "Results:" in output
    assert "10 passed (100%)" in output
    assert "0 failed (0%)" in output
    assert "0 errors (0%)" in output


def test_percentages_when_some_fail() -> None:
    output = render(successes=17, failures=3)
    assert "17 passed (85.00%)" in output
    assert "3 failed (15.00%)" in output


def test_percentages_split_evenly() -> None:
    output = render(successes=5, failures=5)
    assert "5 passed (50.00%)" in output
    assert "5 failed (50.00%)" in output


def test_includes_errors() -> None:
    lines = render(successes=8, failures=1, errors=1).split("\n")
    assert any(re.match(r"^\s*(✓\s+)?8 passed \(80\.00%\)$", line) for line in lines)
    assert any(re.match(r"^\s*(✗\s+)?1 failed \(10\.00%\)$", line) for line in lines)
    assert any(re.match(r"^\s*(✗\s+)?1 error \(10\.00%\)$", line) for line in lines)
    assert "1 errors" not in "\n".join(lines)


def test_cache_hits_only_when_present() -> None:
    assert "4 cache hits (100%)" in render(successes=4, cacheHits=4)
    assert "cache hits" not in render(successes=4)
    assert "cache hits" not in render(successes=4, cacheHits=0)


def test_colored_icons_with_muted_percentages() -> None:
    lines = generate_eval_summary(params(successes=8, failures=1, errors=1))
    g = chalk
    assert (
        f"  {g('✓', 'green')} {g('8', 'white', 'bold')} {g('passed', 'white')} "
        f"{g('(80.00%)', 'gray')}"
    ) in lines
    assert (
        f"  {g('✗', 'red')} {g('1', 'white', 'bold')} {g('failed', 'white')} "
        f"{g('(10.00%)', 'gray')}"
    ) in lines
    assert (
        f"  {g('✗', 'red')} {g('1', 'white', 'bold')} {g('error', 'white')} {g('(10.00%)', 'gray')}"
    ) in lines
    assert "\x1b[32m✓\x1b[39m \x1b[37m\x1b[1m8\x1b[22m\x1b[39m" in "\n".join(lines)


def test_no_colors_without_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORCE_COLOR")
    assert "\x1b[" not in "\n".join(generate_eval_summary(params(successes=1, failures=1)))


# --- guidance -------------------------------------------------------------------------------


def test_guidance_when_writing_without_shareable_url() -> None:
    output = render(writeToDatabase=True)
    assert "» View results: promptfoo view" in output
    assert "» Share with your team: https://promptfoo.app" in output
    assert "» Feedback: https://promptfoo.dev/feedback" in output


def test_share_guidance_with_cloud_enabled() -> None:
    output = render(writeToDatabase=True, cloudEnabled=True)
    assert "» View results: promptfoo view" in output
    assert "» Create shareable URL: promptfoo share" in output
    assert "https://promptfoo.app" not in output


def test_no_share_guidance_when_explicitly_disabled() -> None:
    output = render(writeToDatabase=True, hasExplicitDisable=True)
    assert "» View results: promptfoo view" in output
    assert "» Share with your team" not in output
    assert "» Create shareable URL" not in output


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {
            "writeToDatabase": True,
            "shareableUrl": "https://promptfoo.app/eval/abc123",
            "wantsToShare": True,
            "cloudEnabled": True,
        },
    ],
)
def test_no_guidance(overrides: dict[str, Any]) -> None:
    output = render(**overrides)
    assert "» View results:" not in output
    assert "» Share" not in output
    assert "» Feedback:" not in output


def test_duration_and_concurrency() -> None:
    output = render(duration=125, maxConcurrency=8)
    assert "Duration: 2m 5s (concurrency: 8)" in output


# --- repeat stability -----------------------------------------------------------------------


def test_repeat_stability() -> None:
    output = render(
        successes=4,
        failures=1,
        errors=1,
        duration=5,
        maxConcurrency=2,
        repeatStability={
            "totalGroups": 2,
            "unstableGroups": 1,
            "groupsWithErrors": 1,
            "cachedResults": 1,
            "groups": [
                {
                    "repeatGroupId": "test-0-vars-0",
                    "promptIdx": 0,
                    "provider": {"id": "provider-a"},
                    "description": "Flickering rubric",
                    "repetitions": 3,
                    "passed": 2,
                    "failed": 1,
                    "errors": 0,
                    "cached": 0,
                    "passRate": 2 / 3,
                    "passRateConfidenceInterval": {
                        "confidenceLevel": 0.95,
                        "lower": 0.2077,
                        "upper": 0.9385,
                    },
                    "unstable": True,
                },
                {
                    "repeatGroupId": "test-1-vars-0",
                    "promptIdx": 0,
                    "provider": {"id": "provider-a"},
                    "repetitions": 3,
                    "passed": 2,
                    "failed": 0,
                    "errors": 1,
                    "cached": 1,
                    "passRate": 1,
                    "unstable": False,
                },
            ],
        },
    )
    assert "Repeat stability:" in output
    assert "2 repeated groups; 1 unstable" in output
    assert "Flickering rubric: 2/3 passed across 3 runs" in output
    assert "95% CI 20.8%–93.8%" in output
    assert "1 repeated group contained errors" in output
    assert "1 cached result detected" in output


# --- edge cases -----------------------------------------------------------------------------


def test_error_plurals() -> None:
    assert "1 error" in render(errors=1) and "1 errors" not in render(errors=1)
    assert "3 errors" in render(errors=3)
    assert "0 errors" in render(errors=0)


def test_full_layout() -> None:
    lines = [
        strip_ansi(line)
        for line in generate_eval_summary(
            params(
                evalId="eval-abc",
                writeToDatabase=True,
                tokenUsage={"total": 1234, "prompt": 1000, "completion": 234},
                successes=1,
                failures=1,
                errors=1,
                duration=65,
            )
        )
    ]
    assert lines == [
        "✓ Eval complete (ID: eval-abc)",
        "",
        "» View results: promptfoo view",
        "» Share with your team: https://promptfoo.app",
        "» Feedback: https://promptfoo.dev/feedback",
        "",
        "Total Tokens: 1,234",
        "  Provider: 1,234 (1,000 prompt, 234 completion)",
        "",
        "Results:",
        "  ✓ 1 passed (33.33%)",
        "  ✗ 1 failed (33.33%)",
        "  ✗ 1 error (33.33%)",
        "Duration: 1m 5s (concurrency: 4)",
        "",
    ]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s"),
        (45.9, "45s"),
        (60, "1m 0s"),
        (125, "2m 5s"),
        (3600, "1h 0m 0s"),
        (7530, "2h 5m 30s"),
    ],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_snake_case_keyword_arguments() -> None:
    lines = generate_eval_summary(
        eval_id="eval-kw",
        write_to_database=True,
        token_usage={"total": 10, "prompt": 4, "completion": 6},
        successes=2,
        failures=0,
        errors=0,
        cache_hits=1,
        duration=3,
        max_concurrency=2,
    )
    output = strip_ansi("\n".join(lines))
    assert "✓ Eval complete (ID: eval-kw)" in output
    assert "Provider: 10 (4 prompt, 6 completion)" in output
    assert "1 cache hits (50.00%)" in output
    assert "Duration: 3s (concurrency: 2)" in output
