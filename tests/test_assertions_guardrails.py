"""The `guardrails` assertion. Ports test/assertions/guardrails.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from promptphoo.assertions.base import AssertionParams
from promptphoo.assertions.guardrails import handle_guardrails
from promptphoo.assertions.index import run_assertion

ASSERTION = {"type": "guardrails"}
INVERSE = {"type": "not-guardrails"}


def _run(response: dict[str, Any], inverse: bool = False) -> dict[str, Any]:
    return handle_guardrails(
        AssertionParams(
            assertion=INVERSE if inverse else ASSERTION,
            base_type="guardrails",
            inverse=inverse,
            output="",
            output_string="",
            provider_response=response,
        )
    )


@pytest.mark.parametrize(
    ("response", "passed", "reason"),
    [
        ({"guardrails": {"flagged": False}, "output": "x"}, True, "Content passed safety checks"),
        ({"guardrails": {}, "output": "x"}, True, "Content passed safety checks"),
        ({"guardrails": {"flagged": True}}, False, "Content failed safety checks"),
        (
            {"guardrails": {"flagged": True, "flaggedInput": True}},
            False,
            "Prompt failed safety checks",
        ),
        (
            {"guardrails": {"flagged": True, "flaggedOutput": True}},
            False,
            "Output failed safety checks",
        ),
        (
            {"output": "x", "metadata": {"redteamHistory": [{"guardrails": {"flagged": True}}]}},
            False,
            "Content failed safety checks",
        ),
        ({"output": "x"}, True, "Content passed safety checks"),
        (
            {"guardrails": {"flagged": True, "reason": "Custom safety violation reason"}},
            False,
            "Custom safety violation reason",
        ),
    ],
)
def test_guardrails(response, passed, reason):
    assert _run(response) == {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": reason,
        "assertion": ASSERTION,
    }


@pytest.mark.parametrize(
    ("response", "passed", "reason"),
    [
        (
            {"guardrails": {"flagged": True, "reason": "Prompt injection detected"}},
            True,
            "Guardrail correctly blocked: Prompt injection detected",
        ),
        (
            {"guardrails": {"flagged": False}},
            False,
            "Content was not blocked by guardrails (expected to be blocked)",
        ),
        ({"output": "x"}, False, "Content was not blocked by guardrails (expected to be blocked)"),
        (
            {"guardrails": {"flagged": True, "flaggedInput": True}},
            True,
            "Guardrail correctly blocked: Prompt failed safety checks",
        ),
    ],
)
def test_not_guardrails(response, passed, reason):
    assert _run(response, inverse=True) == {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": reason,
        "assertion": INVERSE,
    }


async def test_dispatched_by_run_assertion():
    result = await run_assertion(
        assertion={"type": "guardrails"},
        test={},
        provider_response={"output": "x", "guardrails": {"flagged": True, "flaggedOutput": True}},
    )
    assert result["pass"] is False
    assert result["reason"] == "Output failed safety checks"
