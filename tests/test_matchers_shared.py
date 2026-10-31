"""Ports test/matchers/shared.test.ts, plus the vector and score helpers of shared.ts."""

from __future__ import annotations

import math

import pytest

from promptphoo.matchers.shared import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    fail,
    grader_fail,
    invert_score,
    normalize_matcher_token_usage,
    split_into_sentences,
    split_text_into_sentences,
    try_parse,
)


def _matches(actual: dict, expected: dict) -> bool:
    return all(actual.get(k) == v for k, v in expected.items())


class TestNormalizeMatcherTokenUsage:
    def test_derives_missing_totals(self):
        out = normalize_matcher_token_usage({"prompt": 7, "completion": 3})
        assert _matches(out, {"total": 10, "prompt": 7, "completion": 3, "numRequests": 0})

    def test_preserves_explicit_zero_total(self):
        out = normalize_matcher_token_usage({"total": 0, "prompt": 7, "completion": 3})
        assert _matches(out, {"total": 0, "prompt": 7, "completion": 3})

    def test_fully_cached_with_zero_requests_is_not_recharged(self):
        out = normalize_matcher_token_usage(
            {"prompt": 10, "completion": 0, "cached": 10, "numRequests": 0}
        )
        assert _matches(out, {"total": 0, "cached": 10, "numRequests": 0})

    def test_fresh_cached_prompts_without_request_count(self):
        out = normalize_matcher_token_usage({"prompt": 10, "completion": 0, "cached": 10})
        assert _matches(out, {"total": 10, "cached": 10})

    def test_preserves_incurred_usage_and_completion_details(self):
        incurred = {
            "total": 30,
            "prompt": 20,
            "completion": 10,
            "numRequests": 1,
            "completionDetails": {"reasoning": 4},
        }
        out = normalize_matcher_token_usage(
            {
                "total": 100,
                "prompt": 70,
                "completion": 30,
                "cached": 70,
                "numRequests": 2,
                "completionDetails": {"reasoning": 9},
                "incurredTokenUsage": incurred,
            }
        )
        assert _matches(out, {
            "total": 100, "prompt": 70, "completion": 30, "cached": 70, "numRequests": 2,
            "completionDetails": {"reasoning": 9}, "incurredTokenUsage": incurred,
        })  # fmt: skip

    def test_no_incurred_accounting_when_not_reported(self):
        assert "incurredTokenUsage" not in normalize_matcher_token_usage(
            {"total": 30, "numRequests": 1}
        )

    def test_undefined_usage(self):
        assert normalize_matcher_token_usage(None) == {
            "total": 0,
            "prompt": 0,
            "completion": 0,
            "cached": 0,
            "numRequests": 0,
            "completionDetails": {"reasoning": 0, "acceptedPrediction": 0, "rejectedPrediction": 0},
        }


class TestSplitIntoSentences:
    def test_splits_on_newlines_and_drops_blank_lines(self):
        assert split_into_sentences("a\n\nb\n  \nc") == ["a", "b", "c"]

    def test_single_paragraph_is_one_unit(self):
        assert split_into_sentences("One. Two. Three.") == ["One. Two. Three."]

    def test_numbered_list_markers_stay_attached(self):
        assert split_into_sentences("1. Paris is the capital.\n2. France is in Europe.") == [
            "1. Paris is the capital.",
            "2. France is in Europe.",
        ]


class TestSplitTextIntoSentences:
    def test_segments_prose_on_sentence_boundaries(self):
        assert split_text_into_sentences(
            "Paris is the capital of France. France is in Europe. Nice weather."
        ) == ["Paris is the capital of France.", "France is in Europe.", "Nice weather."]

    @pytest.mark.parametrize(
        "text",
        [
            "Paris is the capital of France. France is in Europe.\n",
            "\nParis is the capital of France. France is in Europe.",
            "Paris is the capital of France. France is in Europe.\r\n",
        ],
    )
    def test_incidental_leading_or_trailing_newline(self, text):
        assert split_text_into_sentences(text) == [
            "Paris is the capital of France.",
            "France is in Europe.",
        ]

    def test_splits_bang_and_question_boundaries(self):
        assert split_text_into_sentences("Really? Yes! Absolutely.") == [
            "Really?",
            "Yes!",
            "Absolutely.",
        ]

    def test_trims_single_line_prose(self):
        assert split_text_into_sentences(
            "   Paris is the capital of France. France is in Europe.   "
        ) == ["Paris is the capital of France.", "France is in Europe."]

    def test_does_not_split_decimals(self):
        assert split_text_into_sentences("Pi is about 3.14 in value. It is irrational.") == [
            "Pi is about 3.14 in value.",
            "It is irrational.",
        ]
        assert split_text_into_sentences("The value is 3.14.") == ["The value is 3.14."]

    def test_multi_line_text_is_pre_segmented(self):
        assert split_text_into_sentences("All employees i.e. engineers.\nThey get leave.") == [
            "All employees i.e. engineers.",
            "They get leave.",
        ]
        assert split_text_into_sentences("Line one.\n\n\nLine two.") == ["Line one.", "Line two."]

    def test_drops_bare_enumeration_markers(self):
        assert split_text_into_sentences("1. Paris is the capital. 2. France is in Europe.") == [
            "Paris is the capital.",
            "France is in Europe.",
        ]
        assert split_text_into_sentences("1. First fact. 2. Second fact.") == [
            "First fact.",
            "Second fact.",
        ]

    def test_keeps_numbers_inside_sentences(self):
        assert split_text_into_sentences("Pi is 3.14 here. There are 42 items.") == [
            "Pi is 3.14 here.",
            "There are 42 items.",
        ]

    def test_empty_or_whitespace_only(self):
        assert split_text_into_sentences("") == []
        assert split_text_into_sentences("   \n  \n ") == []


class TestScoreAndVectorHelpers:
    def test_fail_shape(self):
        out = fail("nope", {"prompt": 1, "completion": 2})
        assert out["pass"] is False and out["score"] == 0 and out["reason"] == "nope"
        assert out["tokensUsed"]["total"] == 3
        assert "metadata" not in out

    def test_grader_fail_is_tagged(self):
        out = grader_fail("transport error")
        assert out["metadata"] == {"graderError": True}
        assert out["pass"] is False

    @pytest.mark.parametrize(
        ("score", "expected"),
        [(0.25, 0.75), (1, 0), (0, 1), (-1, 1), (2, 0), (math.nan, 1), (math.inf, 1)],
    )
    def test_invert_score_is_clamped(self, score, expected):
        assert invert_score(score) == expected

    def test_vectors(self):
        assert dot_product([1, 2, 3], [4, 5, 6]) == 32
        assert cosine_similarity([1, 0], [0, 1]) == 0
        assert cosine_similarity([1, 1], [2, 2]) == pytest.approx(1)
        assert cosine_similarity([0, 0], [1, 1]) == 0
        assert euclidean_distance([0, 0], [3, 4]) == 5

    @pytest.mark.parametrize("fn", [dot_product, euclidean_distance, cosine_similarity])
    def test_unequal_vectors_raise(self, fn):
        with pytest.raises(ValueError, match="Vectors must be of equal length"):
            fn([1, 2], [1])

    def test_try_parse(self):
        assert try_parse('{"a": 1}') == {"a": 1}
        assert try_parse("not json") == "not json"
        assert try_parse("NaN") == "NaN"


def test_cosine_similarity_matches_node_bit_for_bit():
    # Expected values printed by Node with upstream's reduce-based cosineSimilarity. Python 3.12's
    # sum() uses compensated summation and gave 0.9999994978908286 here.
    seed = 7

    def rnd() -> float:
        nonlocal seed
        seed = (seed * 75 + 74) % 65537
        return seed / 65537 - 0.5

    a = [rnd() for _ in range(768)]
    b = [x + rnd() * 1e-3 for x in a]
    assert cosine_similarity(a, b) == 0.9999994978908282
    assert cosine_similarity(a, a) == 1


def test_js_sum_adds_left_to_right_like_reduce():
    from promptphoo.jsvalues import js_sum

    assert js_sum([0.1] * 10) == 0.9999999999999999  # [0.1 x10].reduce((a, b) => a + b, 0)
    assert js_sum([]) == 0
