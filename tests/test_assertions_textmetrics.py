# ruff: noqa: E501
"""Tests for assertions/textmetrics.py: ports of upstream's bleu/gleu/ngrams/levenshtein/rouge/
meteor specs, plus golden values from real Node (tools/textmetrics_golden.js)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from promptphoo.assertions import textmetrics as tm
from promptphoo.assertions.base import AssertionParams, AssertionValueError

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "textmetrics_golden.json").read_text(encoding="utf-8")
)


def params(
    output: str,
    value: Any,
    base_type: str = "rouge-n",
    inverse: bool = False,
    **extra: Any,
) -> AssertionParams:
    assertion = {"type": base_type, "value": value, **extra}
    return AssertionParams(
        assertion=assertion,
        base_type=base_type,
        inverse=inverse,
        output=output,
        output_string=output,
        rendered_value=value,
    )


def run(output: str, value: Any, base_type: str, inverse: bool = False, **extra: Any) -> Any:
    result = tm.HANDLERS[base_type](params(output, value, base_type, inverse, **extra))
    assert isinstance(result, dict)
    return result


# ---------------------------------------------------------------------------------------------
# ngrams.test.ts
# ---------------------------------------------------------------------------------------------


def test_get_ngrams() -> None:
    words = ["hello", "world", "how", "are", "you"]
    assert tm.get_ngrams(words, 1) == words
    assert tm.get_ngrams(words, 2) == ["hello world", "world how", "how are", "are you"]
    assert tm.get_ngrams(words, 3) == ["hello world how", "world how are", "how are you"]
    assert tm.get_ngrams(["hello", "world"], 3) == []
    assert tm.get_ngrams(["hello", "world", "how"], 3) == ["hello world how"]
    assert tm.get_ngrams([], 1) == []
    assert tm.get_ngrams(["the", "cat", "the", "cat"], 2) == ["the cat", "cat the", "the cat"]
    assert tm.get_ngrams(["hello"], 1) == ["hello"]
    assert tm.get_ngrams(["hello", "world"], 0) == []
    assert tm.get_ngrams(["hello,", "world!", "how?"], 2) == ["hello, world!", "world! how?"]


# ---------------------------------------------------------------------------------------------
# levenshtein.test.ts
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "value", "extra", "inverse", "passed", "reason"),
    [
        ("test", "test", {}, False, True, "Assertion passed"),
        ("tast", "test", {}, False, True, "Assertion passed"),
        ("tost", "test", {"threshold": 2}, False, True, "Assertion passed"),
        ("toast", "test", {"threshold": 1}, False, False, "Levenshtein distance 2 is greater than threshold 1"),
        ("completely different", "test", {}, False, False, None),
        ("", "", {}, False, True, "Assertion passed"),
        ("this is a vary long string to test", "this is a very long string to test", {"threshold": 10}, False, True, None),
        ("tast", "test", {"threshold": 5}, True, False, "Levenshtein distance 1 is less than or equal to threshold 5"),
        ("completely different", "test", {"threshold": 2}, True, True, "Assertion passed"),
        ("toast", "test", {"threshold": 2}, True, False, "Levenshtein distance 2 is less than or equal to threshold 2"),
        ("tast", "test", {}, True, False, "Levenshtein distance 1 is less than or equal to threshold 5"),
    ],
)  # fmt: skip
def test_levenshtein(
    output: str, value: str, extra: dict[str, Any], inverse: bool, passed: bool, reason: str | None
) -> None:
    result = run(output, value, "levenshtein", inverse, **extra)
    assert result["pass"] is passed
    assert result["score"] == (1 if passed else 0)
    if reason is not None:
        assert result["reason"] == reason


def test_levenshtein_rejects_non_string() -> None:
    with pytest.raises(
        AssertionValueError, match='"levenshtein" assertion type must have a string'
    ):
        run("test", 123, "levenshtein")


def test_levenshtein_counts_utf16_code_units() -> None:
    # JS strings are UTF-16: an astral character is two code units.
    assert tm.levenshtein_distance("😀", "") == 2
    assert tm.levenshtein_distance("kitten", "sitting") == 3
    a, b = "ab" * 70, "ba" * 70
    assert tm.levenshtein_distance(a, b) == 2


# ---------------------------------------------------------------------------------------------
# bleu.test.ts
# ---------------------------------------------------------------------------------------------

bleu = tm.calculate_bleu_score


def test_bleu_scores() -> None:
    assert bleu("The cat sat on the mat.", ["The cat sat on the mat."]) == 1
    different = bleu("Dogs run in the park.", ["The cat sat on the mat."])
    assert 0 < different < 0.001
    assert 0.5 < bleu("The dog sat on the mat.", ["The cat sat on the mat."]) < 1
    assert bleu("cat", ["cat"]) > 0
    for text in ("cat", "good dog", "the good dog"):
        assert bleu(text, [text]) == pytest.approx(1, abs=1e-5)
    expected = math.exp(0.5 * math.log(0.5) + 0.5 * math.log(1e-7))
    assert bleu("good cat", ["good dog"]) == pytest.approx(expected, abs=1e-10)
    assert bleu("cat", ["cat sat down"]) == pytest.approx(math.exp(1 - 3 / 1), abs=1e-10)
    assert bleu("good dog", ["good dog runs fast"]) == pytest.approx(math.exp(1 - 4 / 2), abs=1e-10)
    assert bleu("good dog", ["bad dog", "good dog", "the dog"]) == pytest.approx(1, abs=1e-5)
    expected = math.exp(
        0.4 * math.log(5 / 6)
        + 0.3 * math.log(3 / 5)
        + 0.2 * math.log(2 / 4)
        + 0.1 * math.log(1 / 3)
    )
    score = bleu("the dog sat on the mat", ["the cat sat on the mat"], [0.4, 0.3, 0.2, 0.1])
    assert score == pytest.approx(expected, abs=1e-10)
    assert bleu("cat", ["cat"], [0, 0, 0, 1]) == 0
    assert bleu("the good dog", ["the good dog"], [0, 0, 0, 1]) == 0
    assert bleu("the very good dog", ["the very good dog"], [0, 0, 0, 1]) == pytest.approx(1)
    assert not math.isnan(bleu("good dog", ["good dog"], [0.5, 0.5, 0, 0]))
    assert 0 < bleu("The cat sat.", ["The cat sat on the mat."]) < 1
    refs = ["The cat sat on the mat.", "There is a cat on the mat.", "A cat is sitting on the mat."]
    assert bleu("The cat was sitting on the mat.", refs) > 0.25
    refs = ["The cat sat on mat.", "Cat mat.", "A cat is sitting on a mat."]
    assert bleu("The cat sat on mat.", refs) > 0.999
    assert bleu("a b c d", ["a b c", "a b c d e"]) == bleu("a b c d", ["a b c d e", "a b c"])
    assert bleu("a b c d", ["a b", "a b c d e f"]) == 1
    assert bleu("a b c d", ["a b c d e f", "a b"]) == 1
    assert bleu("a b c d e", ["a b c d e f g"]) == pytest.approx(math.exp(1 - 7 / 5), abs=1e-10)
    assert bleu("a b c d e", ["a b c d e"]) == 1
    refs = ["The small cat sat.", "A cat was sitting.", "The cat is on the mat."]
    assert bleu("The small cat sat.", refs) > 0.999
    for refs in ([""], ["   "], ["", "\n\t"]):
        assert bleu("the cat sat on the mat", refs) == 0
    assert bleu("hello", ["", "hello world"]) == pytest.approx(math.exp(-1))
    assert bleu("", ["some reference"]) == 0
    assert bleu("   ", ["some reference"]) == 0
    assert bleu("\n\t", ["some reference"]) == 0


def test_bleu_errors() -> None:
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        bleu("a b", ["a b"], [1, -1, 0.5, 0.5])
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        bleu("a b c x", ["a b c d"], [0.5, 0.5, 1, -1])
    with pytest.raises(ValueError, match="Invalid inputs"):
        bleu("test", [])
    with pytest.raises(ValueError, match="Invalid inputs"):
        bleu(None, ["some reference"])
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        bleu("", ["some reference"], [1, -1, 0.5, 0.5])
    with pytest.raises(ValueError, match="Weights must sum to 1"):
        bleu("", ["some reference"], [0.5, 0.5, 0.5, 0.5])
    for weights in ([math.nan, 0, 0, 1], [math.inf, 0, 0, 0], [-math.inf, 1, 0, 0],
                    ["1", 0, 0, 0], [True, 0, 0, 0], [None, 0, 0, 1]):  # fmt: skip
        for candidate in ("", "a b c d"):
            with pytest.raises(ValueError, match="Weights must be finite numbers"):
                bleu(candidate, ["some reference"], weights)
    with pytest.raises(ValueError, match="Weights must sum to 1"):
        bleu("test", ["x"], [0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="Invalid inputs"):
        bleu("test", ["x"], [0.5, 0.5])


def test_handle_bleu() -> None:
    r = run("The cat sat on the mat.", "The cat sat on the mat.", "bleu")
    assert r["pass"] is True and r["reason"] == "Assertion passed"
    r = run("The cat sat on mat.", ["The cat sat on mat.", "The cat is sitting on mat."], "bleu")
    assert r["pass"] is True
    r = run("The dog sat on the mat.", "The cat sat on the mat.", "bleu", threshold=0.8)
    assert r["pass"] is False
    assert r["reason"].startswith("BLEU score 0.") and r["reason"].endswith(
        "less than threshold 0.8"
    )
    r = run("The dog ran in the park.", "The cat sat on the mat.", "bleu", True, threshold=0.8)
    assert r["pass"] is True and r["reason"] == "Assertion passed"
    r = run("a b c d e", "a b c d e f g", "bleu", True)
    assert r["score"] == pytest.approx(1 - math.exp(1 - 7 / 5), abs=1e-10)
    r = run("The dog ran in the park.", "The cat sat on the mat.", "bleu")
    assert r["pass"] is False and r["reason"].endswith("is less than threshold 0.5")
    assert run("", "some reference", "bleu")["score"] == 0
    assert run("   \n\t", "some reference", "bleu")["pass"] is False
    r = run("", "some reference", "bleu", True)
    assert r["pass"] is True and r["score"] == 1
    r = run("hello", ["", "hello world"], "bleu", threshold=0.9)
    assert r["pass"] is False and r["score"] == pytest.approx(math.exp(-1))
    with pytest.raises(AssertionValueError, match="string or array of strings"):
        run("x", 1, "bleu")


# ---------------------------------------------------------------------------------------------
# gleu.test.ts
# ---------------------------------------------------------------------------------------------

gleu = tm.calculate_gleu_score


def test_gleu_scores() -> None:
    assert gleu("The cat sat on the mat", ["The cat sat on the mat"]) == 1
    assert gleu("The cat sat on the mat.", ["The cat sat on the mat"]) > 0.95
    dots = "." * 50_000
    assert gleu(f"The cat{dots}", ["The cat"]) == 1
    assert gleu("The cat", [f"The cat{dots}"]) == 1
    punct = f"{dots}!"
    assert gleu(f"The cat{punct}", [f"The cat{punct}"]) == 1
    assert gleu(f"The cat{punct}", ["The cat!"]) < 1
    assert gleu("the the the the the the the", ["The cat sat on the mat"]) == pytest.approx(
        0.09, abs=0.005
    )
    s = gleu(
        "It is a guide to action which ensures that the military always obeys the commands of the party",
        ["It is a guide to action that ensures that the military will forever heed Party commands"],
    )
    assert 0.35 < s <= 0.46
    assert gleu("One two three", ["One two three four five"]) == pytest.approx(0.429, abs=0.05)
    assert gleu("One two three four five", ["One two three"]) == pytest.approx(0.429, abs=0.05)
    assert gleu("cat", ["cat"]) == 1
    assert gleu("The cat sat.", ["The cat sat on the mat."]) == pytest.approx(0.33, abs=0.005)
    refs = ["The cat sat on the mat.", "There is a cat on the mat.", "A cat is sitting on the mat."]
    assert gleu("The cat was sitting on the mat.", refs) >= 0.5
    for blank in ("", "   \n\t", "..."):
        assert gleu(blank, ["The cat sat on the mat."]) == 0
        assert gleu("The cat sat on the mat.", [blank]) == 0
    assert gleu("   ", ["..."]) == 0 and gleu("...", ["   "]) == 0 and gleu("...", ["..."]) == 0
    assert gleu("a . b", ["a b"]) == pytest.approx(1 / 3)
    assert gleu("a b", ["a . b"]) == pytest.approx(1 / 3)
    assert gleu("...", ["a . b"]) == 0 and gleu("a . b", ["..."]) == 0
    refs = ["The small cat sat.", "A cat was sitting.", "The cat is on the mat."]
    assert gleu("The small cat sat.", refs) == 1
    assert gleu("the the the the the the.", ["The cat sat on the mat."], 2) == 0
    assert gleu("the the the the the the the.", ["The cat sat on the mat."], 1, 1) == pytest.approx(
        0.286, abs=0.05
    )
    assert gleu("The cat on the mat", ["The cat sat on the mat"], 1, 2) == pytest.approx(
        0.727, abs=0.05
    )
    with pytest.raises(ValueError, match="Invalid inputs"):
        gleu("test", [])


def test_handle_gleu() -> None:
    assert run("The cat sat on the mat.", "The cat sat on the mat.", "gleu")["pass"] is True
    r = run("The dog sat on the mat.", "The cat sat on the mat.", "gleu", threshold=0.8)
    assert r["pass"] is False and r["reason"].endswith("is less than threshold 0.8")
    assert run("The dog ran in the park.", "The cat sat on the mat.", "gleu", True, threshold=0.8)[
        "pass"
    ]
    r = run("", "The cat sat on the mat.", "gleu", threshold=0.5)
    assert r == {"pass": False, "score": 0, "reason": "GLEU score 0.0000 is less than threshold 0.5",
                 "assertion": r["assertion"]}  # fmt: skip
    r = run("", "The cat sat on the mat.", "gleu", True, threshold=0.5)
    assert r["pass"] is True and r["score"] == 1 and r["reason"] == "Assertion passed"
    r = run("", "The cat sat on the mat.", "gleu", True, threshold=0)
    assert r["pass"] is False and r["score"] == 1
    assert r["reason"] == "GLEU score 0.0000 is greater than or equal to threshold 0"


# ---------------------------------------------------------------------------------------------
# rouge.test.ts
# ---------------------------------------------------------------------------------------------


def test_rouge_basic() -> None:
    r = run("the cat sat on the mat", "the cat sat on the mat", "rouge-n")
    assert r["pass"] is True and r["score"] == 1
    assert r["reason"] == "ROUGE-N score 1.00 is greater than or equal to threshold 0.75"
    r = run("some different output", "This is the expected output.", "rouge-n")
    assert r["pass"] is False and r["score"] == pytest.approx(0.2222, abs=1e-4)
    assert r["reason"] == "ROUGE-N score 0.22 is less than threshold 0.75"
    r = run("some different output", "This is the expected output.", "rouge-n", threshold=0.2)
    assert r["pass"] is True
    assert r["reason"] == "ROUGE-N score 0.22 is greater than or equal to threshold 0.2"
    high = run("the cat sat on the mat", "the cat sat on the mat", "rouge-n", True)
    assert high["pass"] is False and high["score"] == 0
    low = run("some different output", "This is the expected output.", "rouge-n", True)
    assert low["pass"] is True and low["score"] == pytest.approx(0.7778, abs=1e-4)
    assert run("The CAT Sat", "the cat sat", "rouge-n")["score"] == 1
    assert run("The cat sat on the mat", "The cat sat on the mat", "rouge-n")["score"] == 1
    assert run("Hello hello", "Hello hello", "rouge-n")["score"] == 1
    r = run("Hello hello", "Hello hello", "rouge-n", True)
    assert r["pass"] is False and r["score"] == 0
    assert run("completely unrelated sentence", "the cat sat on the mat", "rouge-n")["score"] < 0.75
    for base, label in (("rouge-l", "ROUGE-L"), ("rouge-s", "ROUGE-S")):
        r = run("The Quick Brown Fox", "the quick brown fox", base)
        assert r["pass"] is True and r["score"] == 1
        assert r["reason"] == f"{label} score 1.00 is greater than or equal to threshold 0.75"
    assert run("hello world. next", "hello world next", "rouge-n")["score"] == pytest.approx(
        2 / 3, abs=1e-12
    )


@pytest.mark.parametrize(
    ("base", "label"), [("rouge-n", "ROUGE-N"), ("rouge-l", "ROUGE-L"), ("rouge-s", "ROUGE-S")]
)
def test_rouge_each_type(base: str, label: str) -> None:
    text = "the cat sat on the mat"
    r = run(text, text, base)
    assert r["score"] == 1 and r["pass"] is True
    assert r["reason"] == f"{label} score 1.00 is greater than or equal to threshold 0.75"
    r = run("alpha beta gamma", "delta epsilon zeta", base)
    assert r["score"] == 0 and r["pass"] is False
    for inverse in (False, True):
        r = run("", "reference", base, inverse, threshold=0)
        assert r["pass"] is (not inverse) and r["score"] == (1 if inverse else 0)
        assert r["reason"] == f"{label} score 0.00 is greater than or equal to threshold 0"
    assert 0 < run("the dog sat on the mat", "the cat sat on the mat", base)["score"] < 1
    for blank in ("", " ", "\n", "\u0085", " \u0085\n"):
        assert run(blank, "the cat sat", base)["score"] == 0
        assert run("the cat sat", blank, base)["score"] == 0
    for output, reference in (("", ""), (" ", "\n"), ("\u0085", " \u0085\n")):
        assert run(output, reference, base)["pass"] is False
    with pytest.raises(AssertionValueError, match='"rouge" assertion type must be a string value'):
        run("actual text", 123, base)


def test_rouge_l_s_scoring() -> None:
    for base in ("rouge-l", "rouge-s"):
        for text in ("the the cat", "a b a b a"):
            assert run(text, text, base)["score"] == 1
    r = run("hello world. next", "hello world next", "rouge-l")
    assert r["score"] == pytest.approx(6 / 7, abs=1e-12) and r["pass"] is True
    assert r["reason"] == "ROUGE-L score 0.86 is greater than or equal to threshold 0.75"
    r = run("hello world. next", "hello world next", "rouge-s")
    assert r["score"] == pytest.approx(2 / 3, abs=1e-12) and r["pass"] is False
    assert r["reason"] == "ROUGE-S score 0.67 is less than threshold 0.75"
    assert run("the cat sat. a dog ran.", "a dog ran. the cat sat.", "rouge-l")["score"] == 1
    r = run("the cat sat. a dog ran.", "a dog ran. the cat sat.", "rouge-s")
    assert r["score"] == pytest.approx(13 / 28, abs=1e-12)
    for inverse in (False, True):
        r = run('"The cat sat." "A dog ran."', '"A dog ran." "The cat sat."', "rouge-l", inverse)
        assert r["score"] == (0 if inverse else 1) and r["pass"] is (not inverse)
    for base, expected in (("rouge-l", 5 / 7), ("rouge-s", 2 / 3)):
        for sep in ("\n", "\r\n"):
            ref = "Sat on the mat the cat."
            assert run("The cat sat on the mat.", ref, base)["score"] == pytest.approx(
                expected, abs=1e-12
            )
            assert run(f"The cat{sep}sat on the mat.", ref, base)["score"] == pytest.approx(
                expected, abs=1e-12
            )
    assert run("hello", "hello", "rouge-l")["score"] == 1
    assert run("hello", "hello", "rouge-s")["score"] == 0


# ---------------------------------------------------------------------------------------------
# meteor.test.ts / meteorMatching.test.ts (natural mocked, as upstream does)
# ---------------------------------------------------------------------------------------------


class _StubWordNet:
    def __init__(self, synonyms: dict[str, list[str]], always_record: bool) -> None:
        self.synonyms = synonyms
        self.always_record = always_record

    def lookup(self, word: str) -> list[list[str]]:
        if word in self.synonyms:
            return [self.synonyms[word]]
        return [[]] if self.always_record else []


@pytest.fixture
def meteor_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    wn = _StubWordNet({"cat": ["feline"], "mat": ["rug"], "sat": ["sitting"]}, True)
    monkeypatch.setattr(tm, "_stem", lambda w: w[:-1] if w.endswith("s") else w)
    monkeypatch.setattr(tm, "_get_wordnet", lambda: wn)


@pytest.fixture
def meteor_matching_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    wn = _StubWordNet({"feline": ["cat", "house_cat"], "canine": ["dog"]}, False)
    stems = {"cats": "cat", "running": "run"}
    monkeypatch.setattr(tm, "_stem", lambda w: stems.get(w, w))
    monkeypatch.setattr(tm, "_get_wordnet", lambda: wn)


@pytest.mark.usefixtures("meteor_mock")
def test_meteor_spec() -> None:
    def m(output: str, value: Any, inverse: bool = False, **extra: Any) -> Any:
        return run(output, value, "meteor", inverse, **extra)

    assert m("The cat sat on the mat", "The cat sat on the mat")["score"] > 0.95
    assert m("The cat sat on the mat.", "The cat sat on the mat")["score"] > 0.95
    r = m("It is a guide to action that ensures the military will forever heed Party commands",
          "It is a guide to action which ensures that the military always obeys the commands of the party")  # fmt: skip
    assert r["pass"] is True and r["score"] > 0.6
    assert m("non matching hypothesis", "The cat sat on the mat")["score"] < 0.5
    r = m("The dog ran in the park", "The cat sat on the mat")
    assert r["pass"] is False and r["score"] < 0.5
    assert r["reason"].startswith("METEOR score 0.") and r["reason"].endswith(
        "did not meet threshold 0.5"
    )
    refs = [
        "It is a guide to action that ensures that the military will forever heed Party commands",
        "It is the guiding principle which guarantees the military forces always being under the command of the Party",
        "It is the practical guide for the army always to heed the directions of the party",
    ]
    out = "It is a guide to action which ensures that the military always obeys the commands of the party"
    assert m(out, refs)["score"] > 0.69
    refs = ["The cat sat on the mat.", "There is a cat on the mat.", "A cat is sitting on the mat."]
    assert m("The cat was sitting on the mat.", refs)["score"] > 0.5
    r = m(
        "The cat is sitting on the mat",
        "The cat sat on the mat",
        threshold=0.75,
        alpha=0.85,
        beta=2.0,
        gamma=0.4,
    )
    assert r["pass"] is True
    r = m("The dog ran in the park", "The cat sat on the mat", True, threshold=0.8)
    assert r["pass"] is True and r["score"] > 0.5
    with pytest.raises(ValueError, match="Invalid inputs"):
        m("test", [])
    assert m("one, two, three", "one, two, three")["score"] > 0.95
    assert m("the cat sat on the mat", "THE CAT SAT ON THE MAT")["score"] > 0.95
    assert m("The cat is sitting on the mat", "The cats are sitting on the mats", threshold=0.5)[
        "pass"
    ]
    assert m("The cat sat on the mat", "The feline sat on the rug")["score"] > 0.7


@pytest.mark.usefixtures("meteor_matching_mock")
def test_meteor_matching() -> None:
    def m(output: str, value: Any, inverse: bool = False, **extra: Any) -> Any:
        return run(output, value, "meteor", inverse, **{"threshold": 0.5, **extra})

    for output in ("cat dog cat", "cats dog cats", "feline dog feline"):
        r = m(output, "cat dog")
        assert r["score"] == pytest.approx(10 / 21, abs=1e-12) and r["pass"] is False
        assert r["reason"] == "METEOR score 0.4762 did not meet threshold 0.5"
    assert m("cats running canine", "cat run dog")["score"] == pytest.approx(53 / 54, abs=1e-12)
    assert m("feline", "house_cat")["score"] == 0
    assert m("cats running canine", ["unrelated", "cat run dog"])["score"] == pytest.approx(
        53 / 54, abs=1e-12
    )
    r = m("feline dog feline", "cat dog", True)
    assert r["score"] == pytest.approx(11 / 21, abs=1e-12) and r["pass"] is True
    assert r["reason"] == "METEOR assertion passed"
    table: list[tuple[dict[str, float], float]] = [
        ({}, 75 / 116), ({"alpha": 0}, 15 / 16), ({"beta": 0}, 10 / 29), ({"gamma": 0}, 20 / 29),
        ({"alpha": 0.25, "beta": 2, "gamma": 0.4}, 0.8),
    ]  # fmt: skip
    for extra, expected in table:
        assert m("cat dog", "cat dog bird", False, **extra)["score"] == pytest.approx(
            expected, abs=1e-12
        )
    for inverse in (False, True):
        r = m("cat", "cat", inverse)
        assert r["score"] == 0.5 and r["pass"] is (not inverse)
    for value in (None, 42, {}, ["cat", 42]):
        with pytest.raises(AssertionValueError, match="string or array of strings"):
            m("cat", value)


def test_meteor_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> Any:
        raise tm.MissingNlpDependency("no wordnet")

    monkeypatch.setattr(tm, "_get_wordnet", missing)
    r = run("Actual output", "Expected output", "meteor", threshold=0.7)
    assert r == {
        "pass": False,
        "score": 0,
        "reason": "METEOR assertion requires the nltk package and WordNet data. Please install them "
        "using: pip install promptphoo[nlp] && python -m nltk.downloader wordnet31",
        "assertion": r["assertion"],
    }


def test_porter_stemmer() -> None:
    # Values from natural@8.1.1's PorterStemmer.
    cases = {
        "cats": "cat", "running": "run", "relational": "relat", "conditional": "condit",
        "happiness": "happi", "generalization": "gener", "sitting": "sit", "caresses": "caress",
        "ponies": "poni", "agreed": "agre", "hopping": "hop", "filing": "file", "sky": "sky",
        "is": "is", "the": "the", "motoring": "motor", "controlling": "control", "rolling": "roll",
    }  # fmt: skip
    assert {w: tm.porter_stem(w) for w in cases} == cases


# ---------------------------------------------------------------------------------------------
# Golden values from real Node
# ---------------------------------------------------------------------------------------------

# Inputs whose sentence segmentation the Python port does not reproduce (structures from the
# un-ported parts of js-rouge's segmenter). Only ROUGE-L/ROUGE-S depend on segmentation.
SEGMENT_DIVERGENCES: set[str] = set()


def _wordnet_for_golden() -> Any:
    dict_dir = os.environ.get("PROMPTPHOO_TEST_WORDNET_DIR")
    if dict_dir:
        files = {}
        for pos in tm.WordNet.POS:
            for kind in ("index", "data"):
                files[f"{kind}.{pos}"] = (Path(dict_dir) / f"{kind}.{pos}").read_bytes()
        return tm.WordNet(files)
    try:
        return tm._load_wordnet()
    except tm.MissingNlpDependency:
        return None


def _golden_ids() -> list[str]:
    return [f"{i}-{c['type']}" for i, c in enumerate(GOLDEN["cases"])]


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=_golden_ids())
def test_golden(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    if case["type"] in ("rouge-l", "rouge-s") and (
        case["output"] in SEGMENT_DIVERGENCES or case["value"] in SEGMENT_DIVERGENCES
    ):
        pytest.skip("known sentence-segmentation divergence")
    if case["type"] == "meteor":
        wordnet = _wordnet_for_golden()
        if wordnet is None:
            pytest.skip("no WordNet data (pip install promptphoo[nlp]; nltk wordnet31)")
        monkeypatch.setattr(tm, "_get_wordnet", lambda: wordnet)
    expected = case["result"]
    if "error" in expected:
        with pytest.raises(ValueError) as info:
            run(case["output"], case["value"], case["type"], case["inverse"], **case["extra"])
        assert str(info.value) == expected["error"]
        return
    result = run(case["output"], case["value"], case["type"], case["inverse"], **case["extra"])
    assert {k: result[k] for k in ("pass", "score", "reason")} == expected


@pytest.mark.parametrize("seg", GOLDEN["segments"], ids=range(len(GOLDEN["segments"])))
def test_golden_tokenize(seg: dict[str, Any]) -> None:
    assert tm.tree_bank_tokenize(seg["text"].lower()) == seg["tokens"]


@pytest.mark.parametrize("seg", GOLDEN["segments"], ids=range(len(GOLDEN["segments"])))
def test_golden_segment(seg: dict[str, Any]) -> None:
    if seg["text"] in SEGMENT_DIVERGENCES:
        pytest.skip("known sentence-segmentation divergence")
    assert tm._sentence_segment(seg["text"]) == seg["sentences"]
