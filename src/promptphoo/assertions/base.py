"""Assertion handler interface. Ports AssertionParams / GradingResult from src/types/index.ts.

A handler takes AssertionParams and returns a GradingResult dict (or an awaitable of one):
    {"pass": bool, "score": float, "reason": str, "assertion": <assertion dict>, ...optional
     "namedScores", "tokensUsed", "componentResults", "metadata"}
Handler modules export HANDLERS: dict[base_type, handler]; assertions/index.py merges them.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from promptphoo.jsvalues import json_stringify

GradingResult = dict[str, Any]
Assertion = dict[str, Any]


@dataclass
class AssertionParams:
    assertion: Assertion
    base_type: str
    inverse: bool
    output: Any
    output_string: str
    rendered_value: Any = None
    value_from_script: Any = None
    prompt: str | None = None
    provider: Any = None
    provider_response: dict[str, Any] = field(default_factory=dict)
    test: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    cost: float | None = None
    log_probs: list[float] | None = None
    assertion_value_context: dict[str, Any] = field(default_factory=dict)
    provider_call_context: dict[str, Any] | None = None


Handler = Callable[[AssertionParams], "GradingResult | Awaitable[GradingResult]"]


class AssertionValueError(ValueError):
    """Equivalent of a failed `invariant(...)` in a Node handler: the assertion is misconfigured.
    The message is surfaced verbatim, so keep Node's wording."""


def invariant(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionValueError(message)


def coerce_string(value: Any) -> str:
    """coerceString: strings pass through, everything else is JSON.stringify'd."""
    return value if isinstance(value, str) else json_stringify(value)


def result(params: AssertionParams, passed: bool, reason_on_fail: str, **extra: Any) -> GradingResult:
    """The common `{pass, score: pass ? 1 : 0, reason, assertion}` shape."""
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": "Assertion passed" if passed else reason_on_fail,
        "assertion": params.assertion,
        **extra,
    }
