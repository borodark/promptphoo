# ruff: noqa: E501
"""RAG matchers. Ports the deterministic cases of upstream test/matchers/{answer-relevance,
context-recall,context-relevance,context-faithfulness}.test.ts, plus byte-exact grader prompts
rendered by real Nunjucks (tests/fixtures/rag_prompts_golden.json, tools/rag_prompts_golden.js)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rag_fakes import TOKENS, FakeEmbedding, FakeText, embedding_result, use_defaults

from promptphoo import cache
from promptphoo.matchers.rag import (
    matches_answer_relevance,
    matches_context_faithfulness,
    matches_context_recall,
    matches_context_relevance,
)
from promptphoo.matchers.rubric import render_llm_rubric_prompt
from promptphoo.matchers.shared import split_text_into_sentences
from promptphoo.prompts import grading

GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "rag_prompts_golden.json").read_text(encoding="utf-8")
)
GOLDEN_BY_NAME = {c["name"]: c for c in GOLDEN}
ANY_USAGE_KEYS = {"total", "prompt", "completion", "cached", "completionDetails", "numRequests"}


def out(text: str, **kw: Any) -> dict[str, Any]:
    return {"output": text, "tokenUsage": dict(TOKENS), **kw}


# ---- golden prompt rendering -------------------------------------------------------------------


@pytest.mark.parametrize("case", GOLDEN, ids=[c["name"] for c in GOLDEN])
async def test_grader_prompt_matches_nunjucks(case):
    rendered = await render_llm_rubric_prompt(getattr(grading, case["prompt"]), case["vars"])
    assert rendered == case["rendered"]


async def test_answer_relevance_sends_golden_prompt(monkeypatch):
    grader = FakeText(out("What is the capital of France?"))
    use_defaults(
        monkeypatch,
        gradingProvider=grader,
        embeddingProvider=FakeEmbedding(lambda t: embedding_result(1, 0)),
    )
    await matches_answer_relevance("q", "Paris is the capital of France.", 0.5)
    assert grader.sent == [GOLDEN_BY_NAME["answer_relevance"]["rendered"]] * 3
    _, context = grader.calls[0]
    assert context == {
        "isGrading": True,
        "prompt": {
            "raw": GOLDEN_BY_NAME["answer_relevance"]["rendered"],
            "label": "answer-relevance",
        },
        "vars": {"answer": "Paris is the capital of France."},
    }


async def test_answer_relevance_parses_json_output_for_the_prompt(monkeypatch):
    grader = FakeText(out("Q?"))
    use_defaults(
        monkeypatch,
        gradingProvider=grader,
        embeddingProvider=FakeEmbedding(lambda t: embedding_result(1, 0)),
    )
    await matches_answer_relevance(
        "q", '{"city": "Paris", "country": "France", "tags": [1, 2.5]}', 0.5
    )
    assert grader.sent[0] == GOLDEN_BY_NAME["answer_relevance_json_answer"]["rendered"]


async def test_context_recall_sends_golden_prompt(monkeypatch):
    case = GOLDEN_BY_NAME["context_recall"]
    grader = FakeText(out("1. Paris is the capital of France. [Attributed]"))
    use_defaults(monkeypatch, gradingProvider=grader)
    await matches_context_recall(
        ["Paris is the capital of France.", "The Eiffel Tower is in Paris."],
        "Paris is the capital of France.",
        0.5,
        None,
        {"extra": "kept user var"},
    )
    assert grader.sent == [case["rendered"]]
    assert grader.calls[0][1]["prompt"]["label"] == "context-recall"
    assert grader.calls[0][1]["vars"] == case["vars"]


async def test_context_relevance_sends_golden_prompt(monkeypatch):
    case = GOLDEN_BY_NAME["context_relevance"]
    grader = FakeText(out("Paris is the capital of France."))
    use_defaults(monkeypatch, gradingProvider=grader)
    await matches_context_relevance(case["vars"]["query"], case["vars"]["context"], 0.5)
    assert grader.sent == [case["rendered"]]
    assert grader.calls[0][1]["prompt"]["label"] == "context-relevance"


async def test_context_faithfulness_sends_golden_prompts(monkeypatch):
    longform = GOLDEN_BY_NAME["context_faithfulness_longform"]
    nli = GOLDEN_BY_NAME["context_faithfulness_nli"]
    grader = FakeText(
        [
            out("Paris is the capital.\nIt has the Eiffel Tower."),
            out("Final verdict for each statement in order: Yes. Yes."),
        ]
    )
    use_defaults(monkeypatch, gradingProvider=grader)
    result = await matches_context_faithfulness(
        "What is the capital of France?",
        longform["vars"]["answer"],
        ["Paris is the capital of France.", "The Eiffel Tower is in Paris."],
        0.5,
        None,
        {"query": "What is the capital of France?", "context": "Paris is the capital of France."},
    )
    assert grader.sent == [longform["rendered"], nli["rendered"]]
    assert [c[1]["prompt"]["label"] for c in grader.calls] == [
        "context-faithfulness-longform",
        "context-faithfulness-nli",
    ]
    assert grader.calls[1][1]["vars"]["statements"] == nli["vars"]["statements"]
    assert result["score"] == 1
    assert result["reason"] == "Faithfulness 1.00 is >= 0.5"


# ---- matchesAnswerRelevance --------------------------------------------------------------------


@pytest.fixture
def relevance(monkeypatch):
    grader = FakeText(out("foobar"))
    embedder = FakeEmbedding(lambda t: embedding_result(1, 0))
    use_defaults(monkeypatch, gradingProvider=grader, embeddingProvider=embedder)
    return grader, embedder


async def test_answer_relevance_passes_above_threshold(relevance):
    grader, embedder = relevance
    result = await matches_answer_relevance("Input text", "Sample output", 0.5)
    assert result["pass"] is True
    assert result["reason"] == "Relevance 1.00 is greater than threshold 0.5"
    assert result["score"] == 1
    assert set(result["tokensUsed"]) == ANY_USAGE_KEYS
    assert result["tokensUsed"]["numRequests"] == 0
    assert result["metadata"]["averageSimilarity"] == 1
    assert result["metadata"]["threshold"] == 0.5
    assert all(
        set(q) == {"question", "similarity"} for q in result["metadata"]["generatedQuestions"]
    )
    assert grader.sent[0].startswith(grading.ANSWER_RELEVANCY_GENERATE[:50])
    assert embedder.inputs_seen[0] == "Input text"


async def test_answer_relevance_fails_below_threshold(monkeypatch):
    grader = FakeText(lambda text: out(text))

    def embed(text: str) -> Any:
        if "Input text" in text:
            return embedding_result(1, 0)
        if "Different output" in text:
            return {
                "embedding": [0, 1, 0],
                "tokenUsage": {"total": 5, "prompt": 2, "completion": 3},
            }
        raise AssertionError(f"Unexpected input {text}")

    embedder = FakeEmbedding(embed)
    use_defaults(monkeypatch, gradingProvider=grader, embeddingProvider=embedder)
    result = await matches_answer_relevance("Input text", "Different output", 0.5)
    assert result["pass"] is False
    assert result["reason"] == "Relevance 0.00 is less than threshold 0.5"
    assert result["score"] == 0
    assert result["metadata"]["averageSimilarity"] == 0
    assert embedder.inputs_seen[1].startswith(grading.ANSWER_RELEVANCY_GENERATE[:50])


async def test_answer_relevance_tags_grading_errors(relevance):
    grader, _ = relevance
    grader.responses = {"error": "grading provider exploded", "tokenUsage": {"total": 0}}
    result = await matches_answer_relevance("q", "a", 0.5)
    assert (result["pass"], result["score"]) == (False, 0)
    assert "grading provider exploded" in result["reason"]
    assert result["metadata"] == {"graderError": True}


async def test_answer_relevance_tags_embedding_errors(relevance):
    _, embedder = relevance
    embedder.fn = lambda t: {"error": "embedding provider exploded", "tokenUsage": {"total": 0}}
    result = await matches_answer_relevance("q", "a", 0.5)
    assert (result["pass"], result["score"]) == (False, 0)
    assert result["metadata"] == {"graderError": True}


async def test_answer_relevance_tracks_token_usage(relevance):
    result = await matches_answer_relevance("Input text", "Sample output", 0.5)
    usage = result["tokensUsed"]
    assert usage["total"] == 50 == usage["prompt"] + usage["completion"]
    assert usage["cached"] == 0
    assert usage["completionDetails"]


async def test_answer_relevance_metadata(monkeypatch):
    questions = iter(
        [
            "What is the capital city of France?",
            "Which city is the capital of France?",
            "What is France's capital?",
        ]
    )
    grader = FakeText(lambda _p: out(next(questions)))
    question_input = "What is the capital of France?"

    def embed(text: str) -> Any:
        if text == question_input:
            return embedding_result(1, 0)
        if "capital" in text and "France" in text:
            return {"embedding": [0.9, 0.1, 0], "tokenUsage": {"total": 5}}
        return {"embedding": [0.8, 0.2, 0], "tokenUsage": {"total": 5}}

    use_defaults(monkeypatch, gradingProvider=grader, embeddingProvider=FakeEmbedding(embed))
    result = await matches_answer_relevance(question_input, "The capital of France is Paris.", 0.7)
    assert len(result["metadata"]["generatedQuestions"]) == 3
    assert "capital" in result["metadata"]["generatedQuestions"][0]["question"]
    assert result["metadata"]["averageSimilarity"] == pytest.approx(0.99, abs=0.005)
    assert result["metadata"]["threshold"] == 0.7
    assert result["pass"] is True


@pytest.mark.parametrize("deterministic", [False, True])
async def test_answer_relevance_caches_three_independent_samples(monkeypatch, deterministic):
    monkeypatch.setattr(cache, "_instance", cache.Cache(None, cache.get_cache_ttl_ms()))
    questions = (
        ["How long do refunds take?"] * 3
        if deterministic
        else [
            "How long do refunds take?",
            "What is the capital of France?",
            "Which planet is the largest?",
        ]
    )
    store: dict[str, str] = {}
    generated = 0

    def respond(prompt: str) -> dict[str, Any]:
        nonlocal generated
        key = cache.get_scoped_cache_key(prompt)
        if key in store:
            return {"output": store[key], "cached": True}
        store[key] = questions[generated]
        generated += 1
        return {"output": store[key]}

    use_defaults(
        monkeypatch,
        gradingProvider=FakeText(respond),
        embeddingProvider=FakeEmbedding(lambda t: embedding_result(1, 0)),
    )
    for _ in range(2):
        result = await matches_answer_relevance(
            "How long do refunds take?", "Refunds are processed within 5 business days.", 0.5
        )
        assert generated == 3
        assert [q["question"] for q in result["metadata"]["generatedQuestions"]] == questions


# ---- matchesContextRecall ----------------------------------------------------------------------

ATTRIBUTION = "foo [Attributed]\nbar [Not attributed]\nbaz [Attributed]\n"


async def recall(monkeypatch, output: str, threshold: float, context: Any = "Context text", **kw):
    grader = FakeText(out(output))
    use_defaults(monkeypatch, gradingProvider=grader)
    result = await matches_context_recall(context, "Ground truth text", threshold, **kw)
    return result, grader


async def test_context_recall_tags_provider_failures(monkeypatch):
    use_defaults(monkeypatch, gradingProvider=FakeText({"error": "grading provider unavailable"}))
    result = await matches_context_recall("Context text", "Ground truth text", 0.5)
    assert (result["pass"], result["score"]) == (False, 0)
    assert result["reason"] == "grading provider unavailable"
    assert result["metadata"] == {"graderError": True}


@pytest.mark.parametrize(
    ("threshold", "passed", "reason"),
    [(0.5, True, "Recall 0.67 is >= 0.5"), (0.9, False, "Recall 0.67 is < 0.9")],
)
async def test_context_recall_threshold(monkeypatch, threshold, passed, reason):
    result, _ = await recall(monkeypatch, ATTRIBUTION, threshold)
    assert result["pass"] is passed
    assert result["reason"] == reason
    assert result["score"] == pytest.approx(2 / 3)
    assert result["tokensUsed"]["numRequests"] == 0
    meta = result["metadata"]
    assert (meta["totalSentences"], meta["attributedSentences"]) == (3, 2)
    assert meta["score"] == pytest.approx(2 / 3)


async def test_context_recall_sentence_attributions(monkeypatch):
    result, _ = await recall(
        monkeypatch,
        "Paris is the capital of France. [Attributed]\nThe Eiffel Tower is located there. "
        "[Attributed]\nLondon is the capital of UK. [Not attributed]",
        0.6,
    )
    assert result["metadata"]["sentenceAttributions"] == [
        {"sentence": "Paris is the capital of France", "attributed": True},
        {"sentence": "The Eiffel Tower is located there", "attributed": True},
        {"sentence": "London is the capital of UK", "attributed": False},
    ]
    assert result["pass"] is True


async def test_context_recall_reserved_vars_win(monkeypatch):
    result, grader = await recall(
        monkeypatch,
        "1. Statement from the expected answer. [Attributed]",
        0,
        "context from contextTransform",
        grading={
            "rubricPrompt": "context={{ context }}\ngroundTruth={{ groundTruth }}\nextra={{ extra }}"
        },
        vars={
            "context": "vars context sentinel",
            "groundTruth": "sentinel",
            "extra": "kept user var",
        },
    )
    prompt, context = grader.calls[0]
    assert (
        prompt
        == "context=context from contextTransform\ngroundTruth=Ground truth text\nextra=kept user var"
    )
    assert context["vars"] == {
        "context": "context from contextTransform",
        "groundTruth": "Ground truth text",
        "extra": "kept user var",
    }


@pytest.mark.parametrize(
    ("output", "score", "total", "attributed"),
    [
        (
            "Let me analyze each sentence in the answer.\n\n1. The signature from the service owner "
            "on the DR plan is requested. The context clearly states this requirement. [Attributed]",
            1.0,
            1,
            1,
        ),
        (
            "I'll analyze each sentence to determine if it can be attributed to the context.\n\n"
            "Here is my analysis:\n\n1. Einstein was German. This can be inferred from the context. "
            "[Attributed]\n2. He won the Nobel Prize. There is no mention of this in the context. "
            "[Not Attributed]",
            0.5,
            2,
            1,
        ),
        (
            "1. First sentence [ATTRIBUTED]\n2. Second sentence [not attributed]\n"
            "3. Third sentence [Attributed]",
            2 / 3,
            3,
            2,
        ),
    ],
)
async def test_context_recall_ignores_preamble(monkeypatch, output, score, total, attributed):
    result, _ = await recall(monkeypatch, output, 0.4)
    assert result["score"] == pytest.approx(score)
    assert result["metadata"]["totalSentences"] == total
    assert result["metadata"]["attributedSentences"] == attributed


async def test_context_recall_not_attributed_wins(monkeypatch):
    result, _ = await recall(
        monkeypatch,
        "First sentence. [Attributed]\nSecond sentence. [Not Attributed] [Attributed]",
        0.5,
    )
    assert result["score"] == 0.5
    assert result["metadata"]["sentenceAttributions"] == [
        {"sentence": "First sentence", "attributed": True},
        {"sentence": "Second sentence", "attributed": False},
    ]


async def test_context_recall_no_classifications_is_grader_error(monkeypatch):
    result, _ = await recall(
        monkeypatch,
        "I cannot determine if the statements are attributed to the context.\n"
        "The context does not provide enough information.",
        0.5,
    )
    assert (result["pass"], result["score"]) == (False, 0)
    assert result["reason"] == "Grader returned no [Attributed]/[Not Attributed] classifications"
    assert result["metadata"]["totalSentences"] == 0
    assert result["metadata"]["graderError"] is True


async def test_context_recall_joins_chunks(monkeypatch):
    chunks = [
        "Paris is the capital of France and has many landmarks.",
        "The Eiffel Tower is a famous iron lattice tower in Paris.",
        "France is located in Western Europe.",
    ]
    result, grader = await recall(
        monkeypatch,
        "Paris is the capital of France [Attributed]\nThe Eiffel Tower is located there [Attributed]",
        0.5,
        chunks,
    )
    assert result["pass"] is True
    assert result["score"] == 1.0
    assert "Recall 1.00 is >= 0.5" in result["reason"]
    assert "\n\n".join(chunks) in grader.sent[0]


# ---- matchesContextRelevance -------------------------------------------------------------------

THREE_LINES = "Paris is the capital of France.\nFrance is in Europe.\nThe weather is nice today."
THREE_PROSE = "Paris is the capital of France. France is in Europe. The weather is nice today."


async def relevance_of(monkeypatch, context: Any, output: str, threshold: float = 0.5):
    use_defaults(monkeypatch, gradingProvider=FakeText(out(output)))
    return await matches_context_relevance("What is the capital of France?", context, threshold)


async def test_context_relevance_tags_provider_failures(monkeypatch):
    use_defaults(monkeypatch, gradingProvider=FakeText({"error": "grading provider unavailable"}))
    result = await matches_context_relevance("What is X?", "Some context", 0.5)
    assert (result["pass"], result["score"], result["reason"]) == (
        False,
        0,
        "grading provider unavailable",
    )
    assert result["metadata"] == {"graderError": True}


async def test_context_relevance_line_based(monkeypatch):
    result = await relevance_of(monkeypatch, THREE_LINES, "Paris is the capital of France", 0.3)
    assert result["score"] == pytest.approx(1 / 3)
    assert result["pass"] is True
    assert result["reason"] == "Context relevance 0.33 is >= 0.3"
    meta = result["metadata"]
    assert meta["graderOutputs"] == {"final": "Paris is the capital of France"}
    assert meta["extractedSentences"] == ["Paris is the capital of France"]
    assert meta["totalContextUnits"] == meta["totalContextSentences"] == 3
    assert meta["contextUnits"] == THREE_LINES.split("\n")
    assert meta["relevantSentenceCount"] == 1
    assert meta["insufficientInformation"] is False


@pytest.mark.parametrize(
    ("context", "output", "score", "total", "relevant"),
    [
        (THREE_PROSE, "Paris is the capital of France.", 1 / 3, 3, 1),
        (THREE_PROSE, "Paris is the capital of France. France is in Europe.", 2 / 3, 3, 2),
        (THREE_PROSE + "\n", "Paris is the capital of France.", 1 / 3, 3, 1),
        (THREE_PROSE, "1. Paris is the capital of France.\n2. France is in Europe.", 2 / 3, 3, 2),
        (THREE_PROSE, "1. Paris is the capital of France. 2. France is in Europe.", 2 / 3, 3, 2),
        (
            "Paris is the capital of France. France is in Europe.\nBerlin is the capital of "
            "Germany. Germany is in Europe.",
            "Paris is the capital of France. France is in Europe.",
            0.5,
            2,
            1,
        ),
        (
            "Paris is the capital of France.\nBerlin is the capital of Germany.\nThe weather is "
            "nice.\nFrance and Germany are in Europe.",
            "Paris is the capital of France.\nBerlin is the capital of Germany.\nFrance and Germany "
            "are in Europe.",
            0.75,
            4,
            3,
        ),
    ],
)
async def test_context_relevance_segmentation(monkeypatch, context, output, score, total, relevant):
    result = await relevance_of(monkeypatch, context, output)
    assert result["score"] == pytest.approx(score)
    assert result["metadata"]["totalContextUnits"] == total
    assert result["metadata"]["relevantSentenceCount"] == relevant


async def test_context_relevance_insufficient_information(monkeypatch):
    result = await relevance_of(monkeypatch, THREE_PROSE, "Insufficient Information")
    assert (result["score"], result["pass"]) == (0, False)
    meta = result["metadata"]
    assert meta["insufficientInformation"] is True
    assert meta["graderOutputs"] == {"final": "Insufficient Information"}
    assert meta["extractedSentences"] == []
    assert meta["relevantSentenceCount"] == 0


async def test_context_relevance_formatted_document(monkeypatch):
    policy = """Company Leave Policy & Guidelines
HR System **May 2024**
Scope and Applicability
The revised leave policy is applicable from 1st May 2024. The scope of this policy covers
entitlement and guidelines for all categories of leaves in the office. This policy will
include people transferred into the office under all incoming structured programs, short-
term/permanent transfers
All permanent employees i.e. Engineering Team, Business Services Team and includes all employees who go out of office on Cross Office Staffing
This policy will include people transferred into the office under all incoming structured programs, short term / permanent transfers
This policy excludes all staff going on any outgoing structured programs, short term/ permanent transfers, people going out of office under any of the other global mobility programs"""  # noqa: E501
    grader_output = (
        "All permanent employees i.e. Engineering Team, Business Services Team\nThis policy will "
        "include people transferred into the office under all incoming structured programs\nThe "
        "scope of this policy covers entitlement and guidelines for all categories of leaves in "
        "the office"
    )
    result = await relevance_of(monkeypatch, policy, grader_output, 0.1)
    assert result["score"] > 0.05
    assert result["pass"] is True
    assert result["metadata"]["totalContextUnits"] > 5
    assert result["metadata"]["relevantSentenceCount"] == 3


async def test_context_relevance_chunks(monkeypatch):
    chunks = [
        "RAG systems improve factual accuracy by incorporating external knowledge sources.",
        "They reduce hallucinations in large language models through grounded responses.",
        "RAG enables up-to-date information retrieval beyond training data cutoffs.",
        "The weather forecast shows rain this weekend.",
    ]
    result = await relevance_of(monkeypatch, chunks, "\n".join(chunks[:3]))
    assert (result["score"], result["pass"]) == (0.75, True)
    assert result["metadata"]["totalContextUnits"] == 4
    assert result["metadata"]["contextUnits"] == chunks
    assert result["metadata"]["relevantSentenceCount"] == 3


async def test_context_relevance_dedupes_repeats(monkeypatch):
    result = await relevance_of(
        monkeypatch, "The answer is 42.", "The answer is 42.\nThe answer is 42.\nThe answer is 42."
    )
    assert (result["score"], result["pass"]) == (1, True)
    assert result["metadata"]["relevantSentenceCount"] == 1
    assert result["metadata"]["extractedSentences"] == ["The answer is 42."]


PYTHON_SENTENCES = [
    "Python is a high-level, general-purpose programming language known for readability.",
    "It has a large standard library.",
    "It was created by Guido van Rossum and released in 1991.",
    "Python is widely used in data science, web development, and automation.",
]
GUIDO = "It was created by Guido van Rossum and released in 1991."


@pytest.mark.parametrize(
    "context",
    [" ".join(PYTHON_SENTENCES), "\n".join(PYTHON_SENTENCES), PYTHON_SENTENCES],
    ids=["prose", "multiline", "array"],
)
@pytest.mark.parametrize(
    ("output", "extracted", "score", "passed"),
    [
        (f"candidate sentences:\n{GUIDO}", [GUIDO], 0.25, False),
        (
            f"1. {GUIDO}\n- it was created by Guido van  Rossum and released in 1991.",
            [f"1. {GUIDO}"],
            0.25,
            False,
        ),
        (
            f"{GUIDO}\nThis passage answers your question.\nIt has a large standard library.",
            [GUIDO, "It has a large standard library."],
            0.5,
            True,
        ),
    ],
    ids=["header", "reformatted-repeat", "commentary"],
)
async def test_context_relevance_counts_only_distinct_quotes(
    monkeypatch, context, output, extracted, score, passed
):
    use_defaults(monkeypatch, gradingProvider=FakeText(out(output)))
    result = await matches_context_relevance("Who created Python?", context, 0.5)
    assert (result["score"], result["pass"]) == (score, passed)
    assert result["metadata"]["totalContextUnits"] == 4
    assert result["metadata"]["relevantSentenceCount"] == len(extracted)
    assert result["metadata"]["extractedSentences"] == extracted
    assert result["metadata"].get("graderError") is not True


@pytest.mark.parametrize(
    "sentences",
    [["1.2 is the value.", "2.2 is the value."], ["-5 is the value.", "5 is the value."]],
)
@pytest.mark.parametrize("separator", [" ", "\n", None])
async def test_context_relevance_keeps_numbers(monkeypatch, sentences, separator):
    context = sentences if separator is None else separator.join(sentences)
    use_defaults(monkeypatch, gradingProvider=FakeText({"output": "\n".join(sentences)}))
    result = await matches_context_relevance("What are the values?", context, 0.75)
    assert (result["score"], result["pass"]) == (1, True)
    assert result["metadata"]["extractedSentences"] == sentences


@pytest.mark.parametrize("fmt", ["array", "multiline"])
@pytest.mark.parametrize(
    ("selection", "extracted", "score", "passed"),
    [
        ("Fact B. Fact C.", ["Fact A.", "Fact B. Fact C."], 1, True),
        ("Fact B. Invented detail.", ["Fact A."], 0.5, False),
    ],
)
async def test_context_relevance_selected_units(
    monkeypatch, fmt, selection, extracted, score, passed
):
    chunks = ["Fact A.", "Fact B. Irrelevant detail. Fact C."]
    context = chunks if fmt == "array" else "\n".join(chunks)
    use_defaults(monkeypatch, gradingProvider=FakeText({"output": f"Fact A.\n{selection}"}))
    result = await matches_context_relevance("What are the facts?", context, 0.75)
    assert (result["score"], result["pass"]) == (score, passed)
    assert result["metadata"]["totalContextUnits"] == 2
    assert result["metadata"]["extractedSentences"] == extracted


async def test_context_relevance_edge_cases(monkeypatch):
    empty = await relevance_of(monkeypatch, "", "Insufficient Information")
    assert (empty["score"], empty["pass"]) == (0, False)
    assert empty["metadata"]["totalContextSentences"] == 0
    assert empty["metadata"]["insufficientInformation"] is True

    short = await relevance_of(monkeypatch, "The answer is 42.", "The answer is 42")
    assert (short["score"], short["pass"]) == (1.0, True)

    malformed = await relevance_of(
        monkeypatch, "Fragment without end Proper sentence. Another fragment", "Proper sentence"
    )
    assert malformed["score"] > 0
    assert malformed["metadata"]["relevantSentenceCount"] == 1


async def test_context_relevance_ungrounded_output_is_grader_error(monkeypatch):
    output = "I am sorry, but I cannot answer that from the provided context."
    result = await relevance_of(
        monkeypatch, "Paris is the capital of France. France is in Europe.", output
    )
    assert (result["pass"], result["score"]) == (False, 0)
    assert result["reason"] == "Grader output does not quote the provided context"
    assert result["metadata"] == {"graderError": True, "graderOutputs": {"final": output}}


def test_split_text_into_sentences():
    assert split_text_into_sentences("A b. C d!  E f? g") == ["A b.", "C d!", "E f?", "g"]
    assert split_text_into_sentences("1. One. 2. Two.") == ["One.", "Two."]
    assert split_text_into_sentences("x. y.\n\nz") == ["x. y.", "z"]


# ---- matchesContextFaithfulness ----------------------------------------------------------------


async def faithfulness(monkeypatch, *responses: dict[str, Any], threshold: float = 0.5, **kw):
    grader = FakeText(list(responses))
    use_defaults(monkeypatch, gradingProvider=grader)
    result = await matches_context_faithfulness(
        "Query text", "Output text", kw.pop("context", "Context text"), threshold, **kw
    )
    return result, grader


@pytest.mark.parametrize(
    ("verdicts", "threshold", "passed", "reason"),
    [
        ("Yes. No. Yes.", 0.5, True, "Faithfulness 0.67 is >= 0.5"),
        ("Yes. Yes. No.", 0.7, False, "Faithfulness 0.67 is < 0.7"),
        ("Yes.", 0.5, False, "Faithfulness 0.33 is < 0.5"),
    ],
)
async def test_context_faithfulness_final_verdicts(
    monkeypatch, verdicts, threshold, passed, reason
):
    result, _ = await faithfulness(
        monkeypatch,
        out("Statement 1\nStatement 2\nStatement 3\n"),
        out(f"Final verdict for each statement in order: {verdicts}"),
        threshold=threshold,
    )
    assert result["pass"] is passed
    assert result["reason"] == reason
    assert set(result) == {"pass", "score", "reason", "tokensUsed"}


async def test_context_faithfulness_tracks_token_usage(monkeypatch):
    result, _ = await faithfulness(
        monkeypatch,
        out("Statement 1\nStatement 2\nStatement 3\n"),
        out("Final verdict for each statement in order: Yes. No. Yes."),
    )
    usage = result["tokensUsed"]
    assert (usage["total"], usage["prompt"], usage["completion"], usage["cached"]) == (
        20,
        10,
        10,
        0,
    )
    assert usage["numRequests"] == 0


async def test_context_faithfulness_tags_provider_errors(monkeypatch):
    result, _ = await faithfulness(monkeypatch, {"error": "grading provider failed"})
    assert (result["pass"], result["score"], result["reason"]) == (
        False,
        0,
        "grading provider failed",
    )
    assert result["metadata"] == {"graderError": True}


async def test_context_faithfulness_empty_statements(monkeypatch):
    result, grader = await faithfulness(monkeypatch, {"output": "   "})
    assert result["reason"] == "Could not extract context-faithfulness statements"
    assert result["metadata"] == {"graderError": True}
    assert len(grader.calls) == 1


async def test_context_faithfulness_reserved_vars_win(monkeypatch):
    grader = FakeText(
        [
            out("Statement from answer."),
            out("Final verdict for each statement in order: Yes."),
        ]
    )
    use_defaults(monkeypatch, gradingProvider=grader)
    await matches_context_faithfulness(
        "question from assertion",
        "answer from provider",
        "context from contextTransform",
        0,
        {
            "rubricPrompt": [
                "question={{ question }}\nanswer={{ answer }}\nextra={{ extra }}",
                "context={{ context }}\nstatements={{ statements }}\nextra={{ extra }}",
            ]
        },
        {
            "question": "vars question sentinel",
            "answer": "vars answer sentinel",
            "context": "vars context sentinel",
            "statements": "vars statements sentinel",
            "extra": "kept user var",
        },
    )
    (longform, longform_ctx), (nli, nli_ctx) = grader.calls
    assert (
        longform
        == "question=question from assertion\nanswer=answer from provider\nextra=kept user var"
    )
    # Nunjucks prints an array by joining its items with ","
    assert (
        nli
        == "context=context from contextTransform\nstatements=Statement from answer.\nextra=kept user var"
    )
    assert longform_ctx["vars"]["answer"] == "answer from provider"
    assert nli_ctx["vars"]["statements"] == ["Statement from answer."]


@pytest.mark.parametrize(
    "verdict",
    [
        "I apologize, but I cannot create statements or provide an analysis based on the given "
        "context.",
        "Final verdict for each statement in order: Unable to determine.",
    ],
)
async def test_context_faithfulness_unparseable_verdicts(monkeypatch, verdict):
    result, _ = await faithfulness(
        monkeypatch, out("Statement 1\nStatement 2\nStatement 3"), out(verdict)
    )
    assert result["pass"] is False
    assert result["score"] == 0
    assert result["reason"] == "Could not parse context-faithfulness verdicts"
    assert result["metadata"] == {"graderError": True}


async def test_context_faithfulness_line_verdicts(monkeypatch):
    missing, _ = await faithfulness(
        monkeypatch, out("Statement 1\nStatement 2\nStatement 3"), out("Statement 1\nverdict: yes")
    )
    assert missing["reason"] == "Faithfulness 0.33 is < 0.5"
    clamped, _ = await faithfulness(
        monkeypatch,
        out("Statement 1\nStatement 2\nStatement 3"),
        out("verdict: no\nverdict: no\nverdict: no\nverdict: no\nverdict: no"),
    )
    assert (clamped["score"], clamped["reason"]) == (0, "Faithfulness 0.00 is < 0.5")


async def test_context_faithfulness_prefers_resolved_context(monkeypatch):
    _, grader = await faithfulness(
        monkeypatch,
        out("Statement 1"),
        out("Final verdict for each statement in order: Yes."),
        context="TRANSFORMED_CONTEXT_BETA",
        vars={"context": "RAW_CONTEXT_ALPHA"},
    )
    assert "TRANSFORMED_CONTEXT_BETA" in grader.sent[1]
    assert "RAW_CONTEXT_ALPHA" not in grader.sent[1]


async def test_context_faithfulness_requires_array_rubric(monkeypatch):
    use_defaults(monkeypatch, gradingProvider=FakeText(out("x")))
    with pytest.raises(ValueError, match="Invariant failed: rubricPrompt must be an array"):
        await matches_context_faithfulness("q", "o", "c", 0.5, {"rubricPrompt": "single"})
