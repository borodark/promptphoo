# ruff: noqa: E501
"""matchesClassification, matchesModeration and matchesSearchRubric. Ports the deterministic
cases of upstream test/matchers/{classification,moderation,search-rubric}.test.ts."""

from __future__ import annotations

import json
from typing import Any

import pytest
from rag_fakes import FakeText, use_defaults

from promptphoo import providers as providers_pkg
from promptphoo.matchers.classification import matches_classification
from promptphoo.matchers.moderation import LLAMA_GUARD_REPLICATE_PROVIDER, matches_moderation
from promptphoo.matchers.search import matches_search_rubric
from promptphoo.providers.base import ApiProvider
from promptphoo.providers.openai.moderation import OpenAiModerationProvider
from promptphoo.providers.web_search_utils import has_web_search_capability

# ---- matchesClassification ---------------------------------------------------------------------


class Classifier(ApiProvider):
    def __init__(self, response: Any = None, id: str = "TestClassificationProvider") -> None:
        super().__init__({})
        self._id = id
        self.response = response or {"classification": {"classA": 0.6, "classB": 0.5}}
        self.calls: list[tuple[Any, ...]] = []

    def default_id(self) -> str:
        return self._id

    async def call_api(self, prompt, context=None, options=None):
        raise NotImplementedError("Not implemented")

    async def call_classification_api(self, *args: Any) -> Any:
        self.calls.append(args)
        return self.response


@pytest.mark.parametrize(
    ("expected", "threshold", "result"),
    [
        ("classA", 0.5, {"pass": True, "score": 0.6, "reason": "Classification classA has score 0.60 >= 0.5"}),
        ("classA", 0.9, {"pass": False, "score": 0.6, "reason": "Classification classA has score 0.60 < 0.9"}),
        (None, 0.55, {"pass": True, "score": 0.6, "reason": "Maximum classification score 0.60 >= 0.55"}),
        (None, 0.9, {"pass": False, "score": 0.6, "reason": "Maximum classification score 0.60 < 0.9"}),
        ("missing", 0.5, {"pass": False, "score": 0, "reason": "Classification missing has score 0.00 < 0.5"}),
    ],
)  # fmt: skip
async def test_classification_thresholds(expected, threshold, result):
    grader = Classifier()
    assert (
        await matches_classification(expected, "Sample output", threshold, {"provider": grader})
        == result
    )
    assert grader.calls == [("Sample output",)]


@pytest.mark.parametrize(("threshold", "passed", "op"), [(0.5, True, ">="), (0.75, False, "<")])
async def test_classification_preserves_token_usage(threshold, passed, op):
    incurred = {"total": 16, "prompt": 12, "completion": 4, "numRequests": 1}
    provider = Classifier(
        {
            "classification": {"classA": 0.625},
            "tokenUsage": {
                "prompt": 12,
                "completion": 4,
                "numRequests": 1,
                "completionDetails": {"reasoning": 2},
                "incurredTokenUsage": incurred,
            },
        }
    )
    assert await matches_classification(
        "classA", "Sample output", threshold, {"provider": provider}
    ) == {
        "pass": passed,
        "score": 0.625,
        "reason": f"Classification classA has score 0.63 {op} {threshold}",
        "tokensUsed": {
            "total": 16,
            "prompt": 12,
            "completion": 4,
            "cached": 0,
            "numRequests": 1,
            "completionDetails": {"reasoning": 2},
            "incurredTokenUsage": incurred,
        },
    }


async def test_classification_cached_tokens():
    provider = Classifier(
        {
            "classification": {"classA": 0.625},
            "tokenUsage": {"prompt": 12, "completion": 4, "cached": 16, "numRequests": 0},
            "cached": True,
        }
    )
    assert await matches_classification(None, "Sample output", 0.5, {"provider": provider}) == {
        "pass": True,
        "score": 0.625,
        "reason": "Maximum classification score 0.63 >= 0.5",
        "tokensUsed": {
            "total": 0,
            "prompt": 12,
            "completion": 4,
            "cached": 16,
            "numRequests": 0,
            "completionDetails": {"reasoning": 0, "acceptedPrediction": 0, "rejectedPrediction": 0},
        },
        "metadata": {"cachedResponse": True},
    }


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ({"error": "Request timed out"}, "Request timed out"),
        ({}, "Unknown error fetching classification"),
        ({"classification": {}}, "No classification scores returned"),
    ],
)
@pytest.mark.parametrize("cached", [False, True])
async def test_classification_failures_are_grader_errors(response, reason, cached):
    provider = Classifier(
        {
            **response,
            "cached": cached,
            "tokenUsage": {"total": 16, "prompt": 12, "completion": 4, "numRequests": 1},
        }
    )
    metadata: dict[str, Any] = {"graderError": True}
    if cached:
        metadata["cachedResponse"] = True
    assert await matches_classification("classA", "Sample output", 0.5, {"provider": provider}) == {
        "pass": False,
        "score": 0,
        "reason": reason,
        "tokensUsed": {
            "total": 16,
            "prompt": 12,
            "completion": 4,
            "cached": 0,
            "numRequests": 1,
            "completionDetails": {"reasoning": 0, "acceptedPrediction": 0, "rejectedPrediction": 0},
        },
        "metadata": metadata,
    }


async def test_classification_undefined_token_usage():
    provider = Classifier({"classification": {"classA": 0.625}, "tokenUsage": None})
    assert await matches_classification("classA", "Sample output", 0.5, {"provider": provider}) == {
        "pass": True,
        "score": 0.625,
        "reason": "Classification classA has score 0.63 >= 0.5",
    }


async def test_classification_requires_a_classification_provider():
    with pytest.raises(
        ValueError, match="No provider of type classification found for 'classification check'"
    ):
        await matches_classification("classA", "output", 0.5)
    with pytest.raises(
        ValueError, match="Provider fake-grader is not a valid classification provider"
    ):
        await matches_classification("classA", "output", 0.5, {"provider": FakeText()})


# ---- matchesModeration -------------------------------------------------------------------------

NORMALIZED = {
    "total": 5,
    "prompt": 2,
    "completion": 3,
    "cached": 0,
    "numRequests": 0,
    "completionDetails": {"reasoning": 0, "acceptedPrediction": 0, "rejectedPrediction": 0},
}


@pytest.fixture
def moderation_env(monkeypatch):
    for key in ("OPENAI_API_KEY", "REPLICATE_API_KEY", "REPLICATE_API_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list[tuple[Any, ...]] = []
    response: dict[str, Any] = {
        "flags": [],
        "tokenUsage": {"total": 5, "prompt": 2, "completion": 3},
    }

    async def fake(self, *args):
        calls.append(args)
        return response

    monkeypatch.setattr(OpenAiModerationProvider, "call_moderation_api", fake)
    return calls, response


async def test_moderation_skips_empty_output(moderation_env):
    calls, _ = moderation_env
    assert await matches_moderation("test prompt", "") == {
        "pass": True,
        "score": 1,
        "reason": "No output to moderate",
    }
    assert calls == []


async def test_moderation_uses_openai_by_default(moderation_env):
    calls, _ = moderation_env
    result = await matches_moderation("test prompt", "test response")
    assert calls == [("test prompt", "test response")]
    assert result == {
        "pass": True,
        "score": 1,
        "reason": "No moderation flags detected",
        "tokensUsed": NORMALIZED,
    }


async def test_moderation_replicate_fallback_is_loud(moderation_env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("REPLICATE_API_KEY", "test-key")
    loaded: list[str] = []

    async def fake_load(path, **kw):
        loaded.append(path)
        raise ValueError(f"Unknown provider: {path}")

    monkeypatch.setattr(providers_pkg, "load_api_provider", fake_load)
    with pytest.raises(ValueError, match="Unknown provider"):
        await matches_moderation("test prompt", "test response")
    assert loaded == [LLAMA_GUARD_REPLICATE_PROVIDER]


async def test_moderation_respects_grading_provider():
    class Custom(ApiProvider):
        def default_id(self) -> str:
            return "custom-moderation"

        async def call_moderation_api(self, prompt, response):
            return {"flags": [{"code": "hate", "description": "Hate"}]}

    result = await matches_moderation("p", "r", [], {"provider": Custom()})
    assert result == {"pass": False, "score": 0, "reason": "Moderation flags detected: Hate"}


async def test_moderation_api_error_is_grader_error(moderation_env):
    _, response = moderation_env
    response.clear()
    response.update(
        {"error": "provider unavailable", "tokenUsage": {"total": 5, "prompt": 2, "completion": 3}}
    )
    assert await matches_moderation("test prompt", "test response") == {
        "pass": False,
        "score": 0,
        "reason": "Moderation API error: provider unavailable",
        "tokensUsed": NORMALIZED,
        "metadata": {"graderError": True},
    }


async def test_moderation_categories(moderation_env):
    _, response = moderation_env
    response.clear()
    response["flags"] = [
        {"code": "violence", "description": "Violence", "confidence": 1},
        {"code": "hate", "description": "Hate", "confidence": 1},
    ]
    assert await matches_moderation("p", "r", ["hate"]) == {
        "pass": False,
        "score": 0,
        "reason": "Moderation flags detected: Hate",
    }
    assert (await matches_moderation("p", "r"))[
        "reason"
    ] == "Moderation flags detected: Violence, Hate"
    response["flags"] = [{"code": "violence", "description": "Violence", "confidence": 1}]
    assert await matches_moderation("p", "r", ["hate"]) == {
        "pass": True,
        "score": 1,
        "reason": "No relevant moderation flags detected",
    }


# ---- matchesSearchRubric -----------------------------------------------------------------------


class OpenAiResponsesProvider(FakeText):
    """Named like the real class so hasWebSearchCapability recognises it."""

    def __init__(self, responses: Any = None) -> None:
        super().__init__(
            responses
            or {
                "output": json.dumps({"pass": True, "score": 1, "reason": "web search ok"}),
                "tokenUsage": {"total": 5, "prompt": 3, "completion": 2},
            },
            id="openai:gpt-5.5-2026-04-23",
        )
        self.config = {"tools": [{"type": "web_search_preview"}]}


@pytest.fixture
def search(monkeypatch):
    configured = FakeText({"output": "should not be used"}, id="openai:gpt-4o")
    web = OpenAiResponsesProvider()
    loaded: list[str] = []
    target = {"provider": configured}

    async def fake_load(path, **kw):
        loaded.append(path)
        return target["provider"]

    monkeypatch.setattr(providers_pkg, "load_api_provider", fake_load)
    use_defaults(monkeypatch, webSearchProvider=web)
    return configured, web, loaded, target


async def test_search_rubric_falls_back_to_web_search_default(search):
    configured, web, loaded, _ = search
    result = await matches_search_rubric(
        "Confirm current facts", "output", {"provider": "openai:responses:gpt-5.5-2026-04-23"}
    )
    assert loaded == ["openai:responses:gpt-5.5-2026-04-23"]
    assert configured.calls == []
    assert len(web.calls) == 1
    assert (result["pass"], result["score"], result["reason"]) == (True, 1, "web search ok")
    assert result["tokensUsed"] == {"total": 5, "prompt": 3, "completion": 2}
    assert web.calls[0][1]["prompt"]["label"] == "search-rubric"


async def test_search_rubric_reserved_vars_win(search):
    _, web, _, target = search
    target["provider"] = web
    await matches_search_rubric(
        "rubric from assertion",
        "output from provider",
        {
            "provider": "openai:responses:x",
            "rubricPrompt": "output={{ output }}\nrubric={{ rubric }}\nextra={{ extra }}",
        },
        {
            "output": "vars output sentinel",
            "rubric": "vars rubric sentinel",
            "extra": "kept user var",
        },
    )
    prompt, context = web.calls[0]
    assert (
        prompt == "output=output from provider\nrubric=rubric from assertion\nextra=kept user var"
    )
    assert context["vars"] == {
        "output": "output from provider",
        "rubric": "rubric from assertion",
        "extra": "kept user var",
    }


async def test_search_rubric_requires_grading_config(search):
    with pytest.raises(ValueError, match="Cannot grade output without grading config"):
        await matches_search_rubric("Confirm current facts", "output")


async def test_search_rubric_metadata(search):
    _, web, _, target = search
    target["provider"] = web
    result = await matches_search_rubric("Confirm", "output", {"provider": "openai:responses:x"})
    assert result["metadata"] == {
        "searchResults": [],
        "searchProvider": "openai:gpt-5.5-2026-04-23",
    }


async def test_search_rubric_threshold(search):
    _, web, _, _ = search
    web.responses = {"output": json.dumps({"pass": True, "score": 0.4, "reason": "too weak"})}
    assertion = {"type": "search-rubric", "value": "Confirm current facts", "threshold": 0.5}
    result = await matches_search_rubric("Confirm current facts", "output", {}, {}, assertion)
    assert (result["pass"], result["score"], result["reason"]) == (False, 0.4, "too weak")
    assert result["assertion"] is assertion
    assert "tokensUsed" not in result


async def test_search_rubric_provider_error(search):
    _, web, _, _ = search
    web.responses = {"error": "search unavailable"}
    result = await matches_search_rubric("Confirm current facts", "output", {})
    assert (result["pass"], result["score"]) == (False, 0)
    assert result["reason"] == "Search rubric evaluation failed: search unavailable"
    assert result["metadata"] == {"graderError": True}


@pytest.mark.parametrize(
    "output",
    ["service unavailable", 'verdict includes "pass": true', "{}", '{"score": 0}', '{"pass": "false"}', '{"pass": null}'],
)  # fmt: skip
async def test_search_rubric_unusable_verdicts(search, output):
    _, web, _, _ = search
    web.responses = {"output": output, "tokenUsage": {"total": 5, "prompt": 3, "completion": 2}}
    result = await matches_search_rubric("Confirm current facts", "output", {})
    assert result == {
        "pass": False,
        "score": 0,
        "reason": 'Search rubric evaluation failed: Expected a JSON object with a boolean "pass" field',
        "tokensUsed": {"total": 5, "prompt": 3, "completion": 2},
        "assertion": None,
        "metadata": {"graderError": True},
    }


async def test_search_rubric_verdict_inside_text(search):
    _, web, _, _ = search
    web.responses = {
        "output": 'Search complete.\n```json\n{"pass": false, "reason": "Fact is incorrect"}\n```'
    }
    result = await matches_search_rubric("Confirm current facts", "output", {})
    assert (result["pass"], result["score"], result["reason"]) == (False, 0, "Fact is incorrect")
    assert result["metadata"] == {
        "searchResults": [],
        "searchProvider": "openai:gpt-5.5-2026-04-23",
    }


async def test_search_rubric_without_any_web_search_provider(search, monkeypatch):
    configured, _, loaded, target = search
    use_defaults(monkeypatch, gradingProvider=configured)
    target["provider"] = None
    with pytest.raises(ValueError, match="anthropic:messages:claude-sonnet-5"):
        await matches_search_rubric("Confirm current facts", "output", {})
    assert loaded[0] == "anthropic:messages:claude-opus-5-5"
    assert loaded[1] == "openai:responses:gpt-6-sol"


class _Probe(ApiProvider):
    def __init__(self, id: str, config: dict[str, Any] | None = None, **attrs: Any) -> None:
        super().__init__({"config": config or {}})
        self._id = id
        for k, v in attrs.items():
            setattr(self, k, v)

    def default_id(self) -> str:
        return self._id


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (_Probe("perplexity:sonar"), True),
        (_Probe("google:gemini-2.5-pro", {"tools": [{"googleSearch": {}}]}), True),
        (_Probe("google:gemini-2.5-pro"), False),
        (_Probe("xai:grok-3", {"search_parameters": {"mode": "on"}}), True),
        (_Probe("xai:responses:grok-4", {"tools": [{"type": "web_search"}]}), True),
        (_Probe("openai:responses:o4-mini", {"tools": [{"type": "web_search_preview"}]}), True),
        (_Probe("openai:responses:o4-mini"), False),
        (_Probe("openai:chat:gpt-4o-search-preview"), True),
        (_Probe("openai:chat:gpt-4o"), False),
        (_Probe("openai:chat:x", {"passthrough": {"model": "gpt-5-search-api"}}), True),
        (_Probe("openai:codex", {"web_search_mode": "live"}), True),
        (_Probe("anthropic:messages:claude", {"tools": [{"type": "web_search_20250305"}]}), True),
        (_Probe("anthropic:messages:claude", {"tools": [{"type": "web_search"}]}), False),
        (None, False),
    ],
)
def test_has_web_search_capability(provider, expected):
    assert has_web_search_capability(provider) is expected
