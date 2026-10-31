"""Shared fixtures for the OpenAI provider tests: an isolated in-memory cache, a clean OpenAI
environment and a fresh HTTP client per test (respx mocks the shared httpx client)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from promptphoo import cache, fetch

OPENAI_ENV_VARS = [
    "OPENAI_API_KEY",
    "OPENAI_API_HOST",
    "OPENAI_API_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_ORGANIZATION",
    "OPENAI_MAX_TOKENS",
    "OPENAI_MAX_COMPLETION_TOKENS",
    "OPENAI_TEMPERATURE",
    "OPENAI_TOP_P",
    "OPENAI_PRESENCE_PENALTY",
    "OPENAI_FREQUENCY_PENALTY",
    "OPENAI_BEST_OF",
    "OPENAI_STOP",
    "PROMPTFOO_REQUIRE_JSON_PROMPTS",
    "PROMPTFOO_RETRY_5XX",
]

CHAT_URL = "https://api.openai.com/v1/chat/completions"


@pytest.fixture
def openai_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any):
    for name in OPENAI_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("PROMPTFOO_REQUEST_BACKOFF_MS", "0")
    monkeypatch.setattr(cache, "_instance", cache.Cache(None, cache.get_cache_ttl_ms()))
    monkeypatch.setattr(cache, "_enabled", True)
    monkeypatch.setattr(fetch, "_client", None)
    yield monkeypatch


def chat_response(content: Any = "Test output", **extra: Any) -> dict[str, Any]:
    message = {"content": content} if not isinstance(content, dict) else content
    return {
        "choices": [{"message": message, **extra}],
        "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
    }


def ok(payload: Any, status: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json=payload, headers=headers)


def sent_json(route: Any, index: int = -1) -> dict[str, Any]:
    return json.loads(route.calls[index].request.content)
