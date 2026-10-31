"""select-best and max-score. Ports test/matchers/comparison.test.ts,
test/matchers/max-score.test.ts, test/assertions/max-score.test.ts,
test/evaluator/select-best-minimal.integration.test.ts and the select-best / max-score cases
of test/evaluator/{metrics,tokenUsage,runtime,execution,selectBestRuntime}.test.ts.
Graders are always fakes."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from promptphoo import cli_state, db, paths
from promptphoo.assertions.index import run_assertions, run_compare_assertion
from promptphoo.evaluator.run import ResultFailureReason, evaluate
from promptphoo.matchers import comparison as comparison_matchers
from promptphoo.matchers.comparison import matches_select_best, select_max_score
from promptphoo.models.eval import Eval
from promptphoo.providers.base import ApiProvider


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("IS_TESTING", raising=False)
    paths.set_config_directory_path(None)
    db.close_db()
    db.run_db_migrations()
    yield tmp_path
    db.close_db()


class FakeProvider(ApiProvider):
    def __init__(self, provider_id: str, response: Any = None, fn: Any = None, config=None):
        super().__init__({"id": provider_id, "config": config})
        self.response = response
        self.fn = fn
        self.calls: list[tuple[str, Any]] = []

    async def call_api(self, prompt, context=None, options=None):
        self.calls.append((prompt, context))
        if self.fn is not None:
            result = self.fn(prompt, context)
            if isinstance(result, BaseException):
                raise result
            return result
        return dict(self.response)


def _select_best_provider(output: str) -> FakeProvider:
    return FakeProvider(
        "select-best-test-provider",
        {"output": output, "tokenUsage": {"total": 7, "prompt": 3, "completion": 4}},
    )


def run(coro):
    return asyncio.run(coro)


# --- matchesSelectBest ----------------------------------------------------------------------


def test_parses_multi_digit_verdict_indexes():
    provider = _select_best_provider("10")
    outputs = [f"Output {i}" for i in range(12)]
    result = run(matches_select_best("choose the best output", outputs, {"provider": provider}))
    assert result[10]["pass"] is True
    assert result[10]["score"] == 1
    assert result[10]["reason"] == "Output selected as the best: choose the best output"
    assert len([r for r in result if r["pass"]]) == 1
    assert result[0]["reason"] == "Output not selected: choose the best output"


def test_independent_failure_results_for_invalid_verdicts():
    provider = _select_best_provider("no verdict")
    result = run(matches_select_best("choose the best output", ["A", "B"], {"provider": provider}))
    assert len(result) == 2
    assert result[0] is not result[1]
    assert result[0]["pass"] is False
    assert result[0]["reason"] == "Invalid select-best verdict: NaN"
    assert result[0]["tokensUsed"]["total"] == 7
    assert result[0]["tokensUsed"]["prompt"] == 3
    assert result[0]["tokensUsed"]["completion"] == 4


def test_out_of_range_verdict():
    provider = _select_best_provider("The best is 5")
    result = run(matches_select_best("c", ["A", "B"], {"provider": provider}))
    assert result[0]["reason"] == "Invalid select-best verdict: 5"


def test_cached_comparison_keeps_provenance_and_logical_usage():
    provider = FakeProvider(
        "cached-select-best-provider",
        {
            "output": "0",
            "cached": True,
            "tokenUsage": {"total": 30, "prompt": 18, "completion": 12, "numRequests": 1},
        },
    )
    result = run(matches_select_best("choose the best output", ["A", "B"], {"provider": provider}))
    for r in result:
        assert r["metadata"] == {"cachedResponse": True}
        assert {k: r["tokensUsed"][k] for k in ("total", "prompt", "completion", "cached",
                                                "numRequests")} == {  # fmt: skip
            "total": 30, "prompt": 18, "completion": 12, "cached": 30, "numRequests": 1,
        }  # fmt: skip


def test_cached_malformed_comparison_keeps_provenance():
    provider = FakeProvider(
        "cached-invalid-select-best-provider",
        {
            "output": "not a verdict",
            "cached": True,
            "tokenUsage": {"total": 30, "prompt": 18, "completion": 12, "numRequests": 1},
        },
    )
    result = run(matches_select_best("choose the best output", ["A", "B"], {"provider": provider}))
    for r in result:
        assert r["pass"] is False
        assert r["metadata"] == {"cachedResponse": True}
        assert r["tokensUsed"]["total"] == 30
        assert r["tokensUsed"]["cached"] == 30


@pytest.mark.parametrize(
    "response",
    [
        {"output": "0"},
        {"output": "not a verdict"},
        {"error": "comparison provider failed", "output": ""},
    ],
    ids=["valid verdict", "invalid verdict", "provider error"],
)
def test_mixed_cached_and_incurred_usage(response):
    usage = {
        "total": 100,
        "prompt": 70,
        "completion": 30,
        "cached": 70,
        "numRequests": 2,
        "completionDetails": {"reasoning": 9},
        "incurredTokenUsage": {
            "total": 30,
            "prompt": 20,
            "completion": 10,
            "numRequests": 1,
            "completionDetails": {"reasoning": 4},
        },
    }
    provider = FakeProvider("mixed-cache-select-best-provider", {**response, "tokenUsage": usage})
    result = run(matches_select_best("choose the best output", ["A", "B"], {"provider": provider}))
    for r in result:
        assert r["tokensUsed"] == usage


def test_reserved_criteria_and_outputs_vars_win_over_user_vars():
    provider = _select_best_provider("0")
    grading = {
        "provider": provider,
        "rubricPrompt": "criteria={{ criteria }}\noutputs={{ outputs }}\nextra={{ extra }}",
    }
    run(
        matches_select_best(
            "criteria from assertion",
            ["first output", "second output"],
            grading,
            {
                "criteria": "vars criteria sentinel",
                "outputs": "vars outputs sentinel",
                "extra": "kept user var",
            },  # fmt: skip
        )
    )
    prompt, context = provider.calls[0]
    assert "criteria=criteria from assertion" in prompt
    assert "first output" in prompt
    assert "extra=kept user var" in prompt
    assert "vars criteria sentinel" not in prompt
    assert "vars outputs sentinel" not in prompt
    assert context["vars"]["criteria"] == "criteria from assertion"
    assert context["vars"]["extra"] == "kept user var"
    assert context["isGrading"] is True
    assert context["prompt"] == {"raw": prompt, "label": "select-best"}


def test_default_select_best_prompt_renders_like_nunjucks():
    provider = _select_best_provider("1")
    run(matches_select_best("be terse", ["A", '{"k": 1}'], {"provider": provider}))
    rendered = json.loads(provider.calls[0][0])
    content = rendered[0]["content"]
    assert content.startswith("You are comparing multiple pieces of text")
    assert "be terse" in content
    assert '<Text index="0">\nA\n</Text>' in content
    assert '<Text index="1">\n{"k":1}\n</Text>' in content


def test_select_best_requires_two_outputs():
    with pytest.raises(ValueError, match="at least two outputs to compare between"):
        run(matches_select_best("c", ["only"], {"provider": _select_best_provider("0")}))


def test_select_best_malformed_response():
    provider = FakeProvider("p", {"output": {"index": 0}})
    with pytest.raises(ValueError, match="select-best produced malformed response"):
        run(matches_select_best("c", ["A", "B"], {"provider": provider}))


# --- runCompareAssertion --------------------------------------------------------------------


def test_run_compare_assertion_passes_context_with_original_provider():
    grader = FakeProvider("test-grading-provider", {"output": "0", "tokenUsage": {}})
    original = FakeProvider("original-provider", {"output": ""})
    test = {"vars": {"foo": "bar"}, "options": {"provider": grader}}
    assertion = {"type": "select-best", "value": "Pick the best output"}
    context = {"originalProvider": original, "prompt": {"raw": "test prompt", "label": "test"},
               "vars": {"foo": "bar"}}  # fmt: skip
    run(run_compare_assertion(test, assertion, ["Output 1", "Output 2"], context))
    assert grader.calls
    assert grader.calls[0][1]["originalProvider"].id() == "original-provider"


def test_run_compare_assertion_requires_string_value():
    grader = FakeProvider("g", {"output": "0"})
    with pytest.raises(ValueError, match="select-best must have a string value"):
        run(run_compare_assertion({}, {"type": "select-best", "provider": grader}, ["a", "b"]))


def _make_secret_suite():
    secret = "select-best-grader-secret"
    seen: list[str] = []
    grader = FakeProvider("select-best-grader", config={"apiKey": secret})

    def grade(prompt, context):
        seen.append(grader.config["apiKey"])
        return {"output": "0"}

    grader.fn = grade
    target = FakeProvider("select-best-target", {"output": "candidate"})
    suite = {
        "providers": [target],
        "prompts": [{"raw": "first", "label": "first"}, {"raw": "second", "label": "second"}],
        "tests": [
            {
                "assert": [
                    {
                        "type": "select-best",
                        "value": "choose the best",
                        "provider": grader,
                        "config": {"apiKey": secret},
                    }
                ]
            }
        ],
    }
    return secret, seen, grader, target, suite


def test_run_compare_assertion_keeps_the_grader_key_out_of_results():
    secret, seen, grader, _target, suite = _make_secret_suite()
    test = suite["tests"][0]
    [result, _] = run(run_compare_assertion(test, test["assert"][0], ["first", "second"]))
    assert len(grader.calls) == 1
    assert seen == [secret]
    assert result["assertion"] == {"type": "select-best", "value": "choose the best"}
    assert grader.config["apiKey"] == secret


# --- selectMaxScore -------------------------------------------------------------------------


def _mock_result(score: float, test_idx: int) -> dict[str, Any]:
    return {
        "testIdx": test_idx,
        "gradingResult": {
            "pass": score > 0.5,
            "score": score,
            "reason": "Test result",
            "componentResults": [
                {"pass": True, "score": 1, "reason": "Contains apple",
                 "assertion": {"type": "contains", "value": "apple"}},
                {"pass": score == 1, "score": 1 if score == 1 else 0,
                 "reason": "Contains orange" if score == 1 else "Does not contain orange",
                 "assertion": {"type": "contains", "value": "orange"}},
            ],
        },
    }  # fmt: skip


def _with_components(result: dict[str, Any], *scores: float) -> dict[str, Any]:
    components = [
        {"pass": s > 0, "score": s, "reason": "r", "assertion": {"type": "contains", "value": v}}
        for s, v in zip(scores, ("apple", "orange"), strict=True)
    ]
    return {**result, "gradingResult": {**result["gradingResult"], "componentResults": components}}


MAX_SCORE = {"type": "max-score"}


def test_max_score_excludes_target_errors_even_with_only_negative_available():
    available = _with_components(_mock_result(0, 1), -1, -1)
    results = run(select_max_score(["unavailable", "available"],
                                   [{"unavailable": True, "gradingResult": None}, available],
                                   MAX_SCORE))  # fmt: skip
    assert (results[0]["pass"], results[0]["score"]) == (False, 0)
    assert results[0]["reason"] == "Target response unavailable for comparison"
    assert (results[1]["pass"], results[1]["score"]) == (True, 1)


def test_max_score_no_winner_when_every_target_failed():
    results = run(select_max_score(["", ""], [{"unavailable": True}, {"unavailable": True}],
                                   MAX_SCORE))  # fmt: skip
    assert [(r["pass"], r["score"]) for r in results] == [(False, 0), (False, 0)]


def test_max_score_selects_highest():
    outputs = ["Output 0", "Output 1", "Output 2"]
    results = [_mock_result(0.5, 0), _mock_result(1.0, 1), _mock_result(0.75, 2)]
    graded = run(select_max_score(outputs, results, MAX_SCORE))
    assert len(graded) == 3
    assert (graded[0]["pass"], graded[0]["score"]) == (False, 0)
    assert "Not selected" in graded[0]["reason"]
    assert (graded[1]["pass"], graded[1]["score"]) == (True, 1)
    assert graded[1]["reason"] == "Selected as highest scoring output (score: 1.000)"
    assert graded[2]["reason"] == "Not selected (score: 0.500, max: 1.000)"


def test_max_score_ties_select_first():
    results = [_mock_result(1.0, 0), _mock_result(1.0, 1), _mock_result(0.5, 2)]
    graded = run(select_max_score(["a", "b", "c"], results, MAX_SCORE))
    assert graded[0]["pass"] is True
    assert graded[1]["pass"] is False
    assert graded[1]["reason"] == "Not selected (score: 1.000, max: 1.000)"
    assert graded[2]["pass"] is False


def test_max_score_threshold():
    results = [_mock_result(0.6, 0), _mock_result(0.4, 1)]
    graded = run(select_max_score(["a", "b"], results,
                                  {"type": "max-score", "value": {"threshold": 0.7}}))  # fmt: skip
    assert graded[0]["pass"] is False
    assert graded[0]["reason"] == "Not selected - score 0.500 below threshold 0.7"
    assert graded[1]["pass"] is False


def test_max_score_requires_two_outputs():
    with pytest.raises(ValueError, match="max-score assertion must have at least two outputs"):
        run(select_max_score(["a"], [_mock_result(1.0, 0)], MAX_SCORE))


def test_max_score_requires_another_assertion():
    results = [
        {"gradingResult": {**_mock_result(1.0, 0)["gradingResult"], "componentResults": []}},
        {"gradingResult": {**_mock_result(0.5, 1)["gradingResult"], "componentResults": []}},
    ]
    with pytest.raises(ValueError, match="max-score requires at least one other assertion"):
        run(select_max_score(["a", "b"], results, MAX_SCORE))


def test_max_score_ignores_select_best_and_max_score_components():
    results = []
    for r in (_mock_result(1.0, 0), _mock_result(0.5, 1)):
        gr = r["gradingResult"]
        extra = [{"pass": True, "score": 1, "assertion": {"type": "select-best"}},
                 {"pass": False, "score": 0, "assertion": {"type": "max-score"}}]  # fmt: skip
        results.append(
            {"gradingResult": {**gr, "componentResults": gr["componentResults"] + extra}}
        )
    graded = run(select_max_score(["a", "b"], results, MAX_SCORE))
    assert [g["pass"] for g in graded] == [True, False]
    assert graded[0]["namedScores"] == {"maxScore": 1.0, "assertionCount": 2, "totalWeight": 2.0}


def test_max_score_zero_scores():
    results = [
        _with_components(_mock_result(0, 0), 0, 0),
        _with_components(_mock_result(0, 1), 0, 0),
        _with_components(_mock_result(0.1, 2), 0.1, 0),
    ]
    graded = run(select_max_score(["a", "b", "c"], results, MAX_SCORE))
    assert [g["pass"] for g in graded] == [False, False, True]
    assert graded[2]["reason"] == "Selected as highest scoring output (score: 0.050)"


def test_max_score_named_scores():
    graded = run(select_max_score(["a", "b"], [_mock_result(0.8, 0), _mock_result(0.6, 1)],
                                  MAX_SCORE))  # fmt: skip
    assert graded[0]["namedScores"]["maxScore"] == 0.5
    assert graded[1]["namedScores"]["maxScore"] == 0.5
    assert [g["pass"] for g in graded] == [True, False]


def test_max_score_weights_and_sum():
    results = [
        _with_components(_mock_result(0, 0), 1, 0),
        _with_components(_mock_result(0, 1), 0, 1),
    ]
    value = {"method": "sum", "weights": {"contains": 2}}
    graded = run(select_max_score(["a", "b"], results, {"type": "max-score", "value": value}))
    assert graded[0]["namedScores"] == {"maxScore": 2, "assertionCount": 2, "totalWeight": 4}
    assert graded[0]["reason"] == "Selected as highest scoring output (score: 2.000)"


def test_max_score_threshold_zero_differs_from_undefined():
    results = [_mock_result(0, 0), _mock_result(0.5, 1)]
    without = run(select_max_score(["a", "b"], results, {"type": "max-score", "value": {}}))
    assert [g["pass"] for g in without] == [True, False]
    zero = run(select_max_score(["a", "b"], results,
                                {"type": "max-score", "value": {"threshold": 0}}))  # fmt: skip
    assert [g["pass"] for g in zero] == [True, False]


def test_max_score_fails_all_when_max_is_below_threshold_zero():
    results = [_with_components(_mock_result(-0.5, 0), -1, 0),
               _with_components(_mock_result(-1, 1), -2, 0)]  # fmt: skip
    graded = run(select_max_score(["a", "b"], results,
                                  {"type": "max-score", "value": {"threshold": 0}}))  # fmt: skip
    assert [g["pass"] for g in graded] == [False, False]
    assert graded[0]["reason"] == "Not selected - score -0.500 below threshold 0"
    assert graded[1]["reason"] == "Not selected (score: -1.000, max: -0.500)"


def test_max_score_passes_winner_when_max_equals_threshold_zero():
    results = [_with_components(_mock_result(0, 0), 0, 0),
               _with_components(_mock_result(-1, 1), -2, 2)]  # fmt: skip
    graded = run(select_max_score(["a", "b"], results,
                                  {"type": "max-score", "value": {"threshold": 0}}))  # fmt: skip
    assert [g["pass"] for g in graded] == [True, False]


# --- runAssertions skips comparison assertions ----------------------------------------------


def test_run_assertions_excludes_max_score():
    test = {"assert": [{"type": "contains", "value": "test"}, {"type": "max-score"}]}
    result = run(run_assertions(prompt="test prompt", provider_response={"output": "test output"},
                                test=test))  # fmt: skip
    assert [c["assertion"]["type"] for c in result["componentResults"]] == ["contains"]


def test_run_assertions_excludes_select_best_and_max_score():
    test = {
        "assert": [
            {"type": "contains", "value": "test"},
            {"type": "select-best", "value": "best criteria"},
            {"type": "max-score"},
            {"type": "equals", "value": "test output"},
        ]
    }
    result = run(run_assertions(prompt="test prompt", provider_response={"output": "test output"},
                                test=test))  # fmt: skip
    assert [c["assertion"]["type"] for c in result["componentResults"]] == ["contains", "equals"]


# --- the evaluator --------------------------------------------------------------------------


def _pick_containing(needle: str):
    """A judge that picks the <Text> block containing `needle`. Persisted results come back in
    the SQLite index order (eval_id, test_idx, success), as upstream's, not in prompt order."""

    def grade(prompt, ctx):
        content = json.loads(prompt)[0]["content"]
        blocks = content.split('<Text index="')[1:]
        for block in blocks:
            index, _, body = block.partition('">')
            if needle in body.split("</Text>")[0]:
                return {"output": index}
        return {"output": "no verdict"}

    return grade


def _two_prompts() -> list[dict[str, Any]]:
    return [{"raw": "Prompt A", "label": "Prompt A"}, {"raw": "Prompt B", "label": "Prompt B"}]


def _sorted(record: Eval) -> list[dict[str, Any]]:
    summary = record.to_evaluate_summary()
    return sorted(summary["results"], key=lambda r: (r["testIdx"], r["promptIdx"]))


@pytest.mark.parametrize("persisted", [True, False])
def test_max_score_applies_to_pass_fail_and_stats(config_dir, persisted):
    provider = FakeProvider(
        "max-score-provider",
        {"output": "hello world", "tokenUsage": {"total": 1, "prompt": 1, "completion": 0,
                                                  "cached": 0, "numRequests": 1}},
    )  # fmt: skip
    suite = {
        "providers": [provider],
        "prompts": _two_prompts(),
        "tests": [{"assert": [{"type": "contains", "value": "hello"}, {"type": "max-score"}]}],
    }
    record = Eval.create({}, suite["prompts"]) if persisted else Eval({})
    run(evaluate(suite, record, {"cache": False}))
    if persisted:
        record = Eval.find_by_id(record.id)
        assert record is not None
        record.load_results()
    results = _sorted(record)
    assert len(results) == 2
    assert results[0]["success"] is True
    assert results[1]["success"] is False
    assert results[1]["failureReason"] == ResultFailureReason.ASSERT
    assert results[1]["error"] == "Not selected (score: 1.000, max: 1.000)"
    stats = record.get_stats()
    assert (stats["successes"], stats["failures"], stats["errors"]) == (1, 1, 0)
    gr = results[0]["gradingResult"]
    assert [c["assertion"]["type"] for c in gr["componentResults"]] == ["contains", "max-score"]
    assert gr["namedScores"] == {"maxScore": 1, "assertionCount": 1, "totalWeight": 1}
    assert results[1]["gradingResult"]["reason"] == "Not selected (score: 1.000, max: 1.000)"
    assert results[1]["score"] == 0
    metrics = [p["metrics"] for p in record.prompts]
    assert [m["testPassCount"] for m in metrics] == [1, 0]
    assert [m["testFailCount"] for m in metrics] == [0, 1]
    assert [m["assertPassCount"] for m in metrics] == [2, 1]
    assert [m["assertFailCount"] for m in metrics] == [0, 1]
    assert [m["score"] for m in metrics] == [1, 0]


def test_select_best_applies_to_pass_fail_and_stats(config_dir, monkeypatch):
    async def fake(criteria, outputs, grading=None, vars=None, context=None):
        return [{"pass": True, "score": 1, "reason": "Selected as best"},
                {"pass": False, "score": 0, "reason": "Not selected"}]  # fmt: skip

    monkeypatch.setattr(comparison_matchers, "matches_select_best", fake)
    provider = FakeProvider("select-best-provider", {"output": "hello world"})
    suite = {
        "providers": [provider],
        "prompts": _two_prompts(),
        "tests": [{"assert": [{"type": "contains", "value": "hello"},
                              {"type": "select-best", "value": "choose the best one"}]}],
    }  # fmt: skip
    record = Eval.create({}, suite["prompts"])
    run(evaluate(suite, record, {"cache": False}))
    record.load_results()
    results = _sorted(record)
    assert [r["success"] for r in results] == [True, False]
    assert results[1]["failureReason"] == ResultFailureReason.ASSERT
    assert results[1]["error"] == "Not selected"
    assert results[1]["gradingResult"]["reason"] == "Not selected"
    comp = results[0]["gradingResult"]["componentResults"][-1]
    assert comp["assertion"] == {"type": "select-best", "value": "choose the best one"}
    stats = record.get_stats()
    assert (stats["successes"], stats["failures"], stats["errors"]) == (1, 1, 0)


def test_cached_target_turns_count_as_probes_after_comparison(config_dir, monkeypatch):
    fresh = {
        "pass": True,
        "score": 1,
        "reason": "Fresh comparison grade",
        "tokensUsed": {"total": 13, "prompt": 8, "completion": 5, "numRequests": 1},
    }

    async def fake(criteria, outputs, grading=None, vars=None, context=None):
        return [dict(fresh), dict(fresh)]

    monkeypatch.setattr(comparison_matchers, "matches_select_best", fake)
    provider = FakeProvider(
        "cached-target-provider",
        {"output": "Cached target response", "cached": True,
         "tokenUsage": {"total": 295, "prompt": 201, "completion": 94, "numRequests": 1}},
    )  # fmt: skip
    suite = {
        "providers": [provider],
        "prompts": _two_prompts(),
        "tests": [{"assert": [{"type": "contains", "value": "Cached"},
                              {"type": "select-best", "value": "choose the best response"}]}],
    }  # fmt: skip
    record = Eval.create({}, suite["prompts"])
    run(evaluate(suite, record, {"cache": False}))
    record.load_results()
    stats = record.get_stats()["tokenUsage"]
    assert (stats["total"], stats["cached"], stats["numRequests"]) == (590, 590, 2)
    assert (stats["assertions"]["total"], stats["assertions"]["numRequests"]) == (26, 2)
    incurred = stats["incurredTokenUsage"]
    assert (incurred["total"], incurred["numRequests"]) == (0, 0)
    assert (incurred["assertions"]["total"], incurred["assertions"]["numRequests"]) == (26, 2)
    for result in _sorted(record):
        tu = result["tokenUsage"]
        assert (tu["total"], tu["cached"], tu["numRequests"]) == (295, 295, 1)
        assert (tu["assertions"]["total"], tu["assertions"]["numRequests"]) == (13, 1)
        assert (tu["incurredTokenUsage"]["assertions"]["total"],
                tu["incurredTokenUsage"]["assertions"]["numRequests"]) == (13, 1)  # fmt: skip
        assert result["response"]["tokenUsage"]["numRequests"] == 1


def test_select_best_skips_terminal_target_errors(config_dir):
    judge = FakeProvider("mixed-result-judge", fn=_pick_containing("Prompt A"))
    target = FakeProvider(
        "mixed-result-target",
        fn=lambda prompt, ctx: (
            {"error": "Target request failed"} if prompt == "broken" else {"output": prompt}
        ),
    )
    suite = {
        "providers": [target],
        "prompts": [*_two_prompts(), {"raw": "broken", "label": "broken"}],
        "tests": [{"assert": [{"type": "contains", "value": "Prompt A"},
                              {"type": "select-best", "value": "Choose first",
                               "provider": judge}]}],
    }  # fmt: skip
    record = Eval.create({}, suite["prompts"])
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    assert len(judge.calls) == 1
    record.load_results()
    stats = record.get_stats()
    assert (stats["successes"], stats["failures"], stats["errors"]) == (1, 1, 1)
    for row in _sorted(record):
        if row["failureReason"] == ResultFailureReason.ERROR:
            assert "Target request failed" in row["error"]
            continue
        components = row["gradingResult"]["componentResults"]
        assert len([c for c in components if c["assertion"]["type"] == "select-best"]) == 1


def test_max_score_skips_terminal_target_errors(config_dir):
    target = FakeProvider(
        "t",
        fn=lambda prompt, ctx: (
            {"error": "Target request failed"} if prompt == "broken" else {"output": prompt}
        ),
    )
    suite = {
        "providers": [target],
        "prompts": [{"raw": "broken", "label": "broken"}, *_two_prompts()],
        "tests": [{"assert": [{"type": "contains", "value": "Prompt B"}, {"type": "max-score"}]}],
    }
    record = Eval({})
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    rows = _sorted(record)
    assert rows[0]["failureReason"] == ResultFailureReason.ERROR
    assert rows[0]["error"] == "Target request failed"
    assert [r["success"] for r in rows] == [False, False, True]
    assert rows[1]["error"] == 'Expected output to contain "Prompt B"'
    stats = record.get_stats()
    assert (stats["successes"], stats["failures"], stats["errors"]) == (1, 1, 1)


def test_grader_exception_marks_rows_as_errors(config_dir, caplog):
    secret, _seen, grader, _target, suite = _make_secret_suite()
    grader.fn = lambda prompt, ctx: RuntimeError("temporary grader failure")
    record = Eval.create({}, suite["prompts"])
    with caplog.at_level(logging.DEBUG, logger="promptphoo.evaluator.comparison"):
        run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    messages = [r.getMessage() for r in caplog.records]
    assert any("select-best grading failed" in m and "temporary grader failure" in m
               for m in messages)  # fmt: skip
    rows = run(record.read_results_by_test_idx(0))
    assert len(rows) == 2
    for row in rows:
        assert row.failure_reason == ResultFailureReason.ERROR
        assert row.error.startswith("Error grading select-best (select-best-grader): ")
        assert "Check the grader configuration and credentials" in row.error
        assert "Run with --verbose to log the underlying error" in row.error
        assert "temporary grader failure" not in row.error
        assert row.grading_result["pass"] is True
        assert row.grading_result["componentResults"] == []
        state = row.metadata["__promptfoo"]["comparisonError"]
        # no other assertions: pass with score 0, as upstream's AssertionsResult
        assert state == {"success": True, "score": 0, "failureReason": 0}
    record.load_results()
    stats = record.get_stats()
    assert (stats["successes"], stats["failures"], stats["errors"]) == (0, 0, 2)
    assert secret not in json.dumps([r.test_case for r in rows], default=str)


def test_grader_exception_is_not_logged_without_debug(config_dir, caplog):
    _secret, _seen, grader, _target, suite = _make_secret_suite()
    grader.fn = lambda prompt, ctx: RuntimeError("fixture-non-url-grader-secret")
    record = Eval.create({}, suite["prompts"])
    with caplog.at_level(logging.INFO):
        run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    assert "fixture-non-url-grader-secret" not in caplog.text


def test_comparison_error_survives_max_score(config_dir):
    _secret, _seen, grader, target, suite = _make_secret_suite()
    suite["tests"][0]["assert"] = [
        {"type": "contains", "value": "first"},
        suite["tests"][0]["assert"][0],
        {"type": "max-score"},
    ]
    target.fn = lambda prompt, ctx: {"output": prompt}
    grader.fn = lambda prompt, ctx: RuntimeError("grader unavailable")
    record = Eval.create({}, suite["prompts"])
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    for row in run(record.read_results_by_test_idx(0)):
        assert row.success is False
        assert row.failure_reason == ResultFailureReason.ERROR
        assert "Check the grader configuration and credentials" in row.error
    record.load_results()
    stats = record.get_stats()
    assert (stats["successes"], stats["failures"], stats["errors"]) == (0, 0, 2)


def test_persisted_comparison_results_keep_the_grader_key_out(config_dir):
    secret, seen, grader, _target, suite = _make_secret_suite()
    record = Eval.create({}, suite["prompts"])
    run(evaluate(suite, record, {"maxConcurrency": 1, "cache": False}))
    rows = run(record.read_results_by_test_idx(0))
    assert len(grader.calls) == 1
    assert seen == [secret]
    assert len(rows) == 2
    assert secret not in json.dumps([r.grading_result for r in rows])
    assert secret not in json.dumps([r.test_case for r in rows])
    component = rows[0].grading_result["componentResults"][0]
    assert "provider" not in component["assertion"]
    assert "config" not in component["assertion"]
    assert sorted(r.success for r in rows) == [False, True]


def test_select_best_with_one_prompt_becomes_a_comparison_error(config_dir):
    grader = FakeProvider("g", {"output": "0"})
    suite = {
        "providers": [FakeProvider("t", {"output": "x"})],
        "prompts": [{"raw": "only", "label": "only"}],
        "tests": [{"assert": [{"type": "select-best", "value": "c", "provider": grader}]}],
    }
    record = Eval({})
    run(evaluate(suite, record, {"cache": False}))
    [row] = _sorted(record)
    assert row["failureReason"] == ResultFailureReason.ERROR
    assert row["error"].startswith("Error grading select-best (g): ")
    assert not grader.calls


def test_max_score_with_one_prompt_fails_the_eval(config_dir):
    suite = {
        "providers": [FakeProvider("t", {"output": "x"})],
        "prompts": [{"raw": "only", "label": "only"}],
        "tests": [{"assert": [{"type": "contains", "value": "x"}, {"type": "max-score"}]}],
    }
    with pytest.raises(ValueError, match="max-score assertion must have at least two outputs"):
        run(evaluate(suite, Eval({}), {"cache": False}))


def test_max_score_without_other_assertions_fails_the_eval(config_dir):
    suite = {
        "providers": [FakeProvider("t", {"output": "x"})],
        "prompts": _two_prompts(),
        "tests": [{"assert": [{"type": "max-score"}]}],
    }
    with pytest.raises(ValueError, match="max-score requires at least one other assertion"):
        run(evaluate(suite, Eval({}), {"cache": False}))


def test_select_best_inside_assert_set_is_ignored(config_dir):
    grader = FakeProvider("g", {"output": "0"})
    suite = {
        "providers": [FakeProvider("t", {"output": "x"})],
        "prompts": _two_prompts(),
        "tests": [{"assert": [{"type": "assert-set", "assert": [
            {"type": "contains", "value": "x"},
            {"type": "select-best", "value": "c", "provider": grader},
        ]}]}],
    }  # fmt: skip
    record = Eval({})
    run(evaluate(suite, record, {"cache": False}))
    assert not grader.calls
    assert [r["success"] for r in _sorted(record)] == [True, True]


def test_select_best_groups_by_test_across_providers(config_dir):
    def grade(prompt, ctx):
        # pick the output from provider "b"
        texts = json.loads(prompt)[0]["content"]
        return {"output": "1" if texts.index("b:") > texts.index("a:") else "0"}

    grader = FakeProvider("g", fn=grade)
    a = FakeProvider("a", fn=lambda p, c: {"output": f"a: {p}"})
    b = FakeProvider("b", fn=lambda p, c: {"output": f"b: {p}"})
    suite = {
        "providers": [a, b],
        "prompts": [{"raw": "{{q}}", "label": "q"}],
        "tests": [
            {"vars": {"q": "one"}, "assert": [{"type": "select-best", "value": "b wins",
                                                "provider": grader}]},
            {"vars": {"q": "two"}, "assert": [{"type": "select-best", "value": "b wins",
                                                "provider": grader}]},
        ],
    }  # fmt: skip
    record = Eval({})
    run(evaluate(suite, record, {"cache": False, "maxConcurrency": 1}))
    assert len(grader.calls) == 2
    rows = _sorted(record)
    winners = {(r["testIdx"], r["provider"]["id"]): r["success"] for r in rows}
    assert winners == {(0, "a"): False, (0, "b"): True, (1, "a"): False, (1, "b"): True}
    # the grader saw the test's vars and the first result's prompt
    ctx = grader.calls[0][1]
    assert ctx["vars"]["q"] == "one"
    assert ctx["originalProvider"].id() == "a"
    assert ctx["testIdx"] == 0


def test_comparison_without_select_best_grader_uses_default_test_provider(config_dir):
    grader = FakeProvider("g", {"output": "1"})
    suite = {
        "providers": [FakeProvider("t", {"output": "x"})],
        "prompts": _two_prompts(),
        "defaultTest": {"options": {"provider": grader}},
        "tests": [{"assert": [{"type": "select-best", "value": "c"}]}],
    }
    record = Eval({})
    with cli_state.with_config({}):
        run(evaluate(suite, record, {"cache": False}))
    assert len(grader.calls) == 1
    assert [r["success"] for r in _sorted(record)] == [False, True]
