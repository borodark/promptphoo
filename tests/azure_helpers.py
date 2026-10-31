"""Shared fixtures for the Azure provider tests: a clean Azure/OpenAI environment, an isolated
in-memory cache, a fresh HTTP client per test (respx mocks the shared httpx client) and an Azure
CLI stub, so no test can reach a real host or run a real `az`."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from promptphoo import cache, fetch
from promptphoo.providers.azure import identity

AZURE_ENV_VARS = [
    "AZURE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_API_HOST",
    "AZURE_OPENAI_API_HOST",
    "AZURE_API_BASE_URL",
    "AZURE_OPENAI_API_BASE_URL",
    "AZURE_OPENAI_BASE_URL",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID",
    "AZURE_AUTHORITY_HOST",
    "AZURE_TOKEN_SCOPE",
    "OPENAI_MAX_TOKENS",
    "OPENAI_MAX_COMPLETION_TOKENS",
    "OPENAI_TEMPERATURE",
    "OPENAI_TOP_P",
    "OPENAI_PRESENCE_PENALTY",
    "OPENAI_FREQUENCY_PENALTY",
    "OPENAI_BEST_OF",
    "OPENAI_STOP",
    "PROMPTFOO_REQUIRE_JSON_PROMPTS",
]

HOST = "test.azure.com"
API_VERSION = "2024-12-01-preview"


def chat_url(deployment: str = "test-deployment", path: str = "chat/completions") -> str:
    return f"https://{HOST}/openai/deployments/{deployment}/{path}"


class FakeAz:
    """Stands in for `identity.run_az`. Records calls; fails unless `replies` are queued."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.replies: list[Any] = []

    def reply(self, token: str = "cli-token", expires_on: float | None = None) -> None:
        payload: dict[str, Any] = {"accessToken": token, "tokenType": "Bearer"}
        if expires_on is not None:
            payload["expires_on"] = expires_on
        self.replies.append((0, json.dumps(payload), ""))

    async def __call__(self, args: list[str]) -> tuple[int, str, str]:
        self.calls.append(args)
        if not self.replies:
            return (1, "", "ERROR: Please run 'az login' to setup account.")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


@pytest.fixture
def azure_env(monkeypatch: pytest.MonkeyPatch):
    for name in AZURE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PROMPTFOO_REQUEST_BACKOFF_MS", "0")
    monkeypatch.setattr(cache, "_instance", cache.Cache(None, cache.get_cache_ttl_ms()))
    monkeypatch.setattr(cache, "_enabled", True)
    monkeypatch.setattr(fetch, "_clients", {})
    az = FakeAz()
    monkeypatch.setattr(identity, "run_az", az)
    yield monkeypatch


@pytest.fixture
def fake_az(azure_env: pytest.MonkeyPatch) -> FakeAz:
    az = identity.run_az
    assert isinstance(az, FakeAz)
    return az


def ok(payload: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def sent_json(route: Any, index: int = -1) -> dict[str, Any]:
    return json.loads(route.calls[index].request.content)


def chat_response(content: Any = "test response", **choice: Any) -> dict[str, Any]:
    return {
        "id": "mock-id",
        "choices": [{"message": {"role": "assistant", "content": content}, **choice}],
        "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
    }


def api_options(**config: Any) -> dict[str, Any]:
    return {"config": {"apiHost": HOST, "apiKey": "test-key", "maxRetries": 0, **config}}
