"""Fake grading providers for the RAG/similarity/classifier/moderation/search-rubric tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from promptphoo.providers.base import ApiProvider

TOKENS = {"total": 10, "prompt": 5, "completion": 5}


def embedding_result(a: float, b: float) -> dict[str, Any]:
    """createEmbeddingResult from upstream's test/factories/literalFixtures.ts."""
    return {"embedding": [a, b, 0], "tokenUsage": {"total": 5, "prompt": 2, "completion": 3}}


class FakeText(ApiProvider):
    """call_api returns the queued responses in order (the last one repeats), or the result of
    `fn(prompt)`. Calls are recorded as (prompt, context)."""

    def __init__(
        self,
        responses: list[dict[str, Any]] | dict[str, Any] | Callable[[str], Any] | None = None,
        id: str = "fake-grader",
    ) -> None:
        super().__init__({})
        self._id = id
        self.responses = responses if responses is not None else {"output": "foobar"}
        self.calls: list[tuple[str, Any]] = []

    def default_id(self) -> str:
        return self._id

    def _next(self, prompt: str) -> Any:
        r = self.responses
        if callable(r):
            return r(prompt)
        if isinstance(r, list):
            index = min(len(self.calls) - 1, len(r) - 1)
            return dict(r[index])
        return dict(r)

    async def call_api(self, prompt, context=None, options=None):
        self.calls.append((prompt, context))
        return self._next(prompt)

    @property
    def sent(self) -> list[str]:
        return [c[0] for c in self.calls]


class FakeEmbedding(ApiProvider):
    def __init__(self, fn: Callable[[str], Any], id: str = "fake-embedding") -> None:
        super().__init__({})
        self._id = id
        self.fn = fn
        self.inputs_seen: list[str] = []

    def default_id(self) -> str:
        return self._id

    async def call_api(self, prompt, context=None, options=None):
        raise NotImplementedError

    async def call_embedding_api(self, text: str) -> dict[str, Any]:
        self.inputs_seen.append(text)
        result = self.fn(text)
        if isinstance(result, Exception):
            raise result
        return result


class FakeTextAndEmbedding(FakeText):
    def __init__(self, responses: Any, embed: Callable[[str], Any], id: str = "fake-both") -> None:
        super().__init__(responses, id)
        self.embed = embed
        self.inputs_seen: list[str] = []

    async def call_embedding_api(self, text: str) -> dict[str, Any]:
        self.inputs_seen.append(text)
        return self.embed(text)


def use_defaults(monkeypatch: pytest.MonkeyPatch, **providers: Any) -> None:
    """Make get_default_providers() return the given providers (others None)."""
    from promptphoo.providers import defaults

    keys = ("embeddingProvider", "gradingProvider", "gradingJsonProvider", "moderationProvider",
            "webSearchProvider", "llmRubricProvider", "suggestionsProvider")  # fmt: skip
    value = {k: providers.get(k) for k in keys}

    async def fake(env: Any = None) -> dict[str, Any]:
        return value

    monkeypatch.setattr(defaults, "get_default_providers", fake)
