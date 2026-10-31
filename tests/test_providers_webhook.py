"""The `webhook:` provider. Ports test/providers/webhook.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from promptphoo import fetch
from promptphoo.cache import FetchWithCacheResult
from promptphoo.providers import load_api_provider
from promptphoo.providers.webhook import WebhookProvider


class FakeFetch:
    """Stands in for fetch_with_cache: records each call and returns (or raises) `response`."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.response: Any = {"output": "test response"}

    async def __call__(self, url, options=None, timeout=None, fmt="json", *args, **kwargs):
        self.calls.append((url, options, timeout, fmt))
        if isinstance(self.response, Exception):
            raise self.response
        return FetchWithCacheResult(data=self.response, cached=False, status=200, status_text="OK")


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeFetch:
    fake_fetch = FakeFetch()
    monkeypatch.setattr(fetch, "fetch_with_cache", fake_fetch)
    return fake_fetch


def test_ids_and_string():
    provider = WebhookProvider("http://test.com")
    assert provider.webhook_url == "http://test.com"
    assert provider.id() == "webhook:http://test.com"
    assert str(provider) == "[Webhook Provider http://test.com]"
    with_options = WebhookProvider("http://test.com", {"id": "test-id", "config": {"foo": "bar"}})
    assert with_options.raw_config == {"foo": "bar"}
    assert with_options.id() == "test-id"


async def test_calls_webhook_and_returns_output(fake):
    result = await WebhookProvider("http://test.com").call_api("test prompt")
    assert fake.calls == [
        (
            "http://test.com",
            {
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": '{"prompt":"test prompt"}',
            },
            300000,
            "json",
        )
    ]
    assert result == {"output": "test response", "cached": False}


async def test_includes_config(fake):
    await WebhookProvider("http://test.com", {"config": {"foo": "bar"}}).call_api("test prompt")
    assert fake.calls[0][1]["body"] == '{"prompt":"test prompt","config":{"foo":"bar"}}'


async def test_fetch_error(fake):
    fake.response = RuntimeError("Network error")
    result = await WebhookProvider("http://test.com").call_api("test prompt")
    assert result == {"error": "Webhook call error: Error: Network error"}


async def test_invalid_response_format(fake):
    fake.response = {"foo": "bar"}
    result = await WebhookProvider("http://test.com").call_api("test prompt")
    assert result == {"error": 'Webhook response error: Unexpected response format: {"foo":"bar"}'}


async def test_registry_routes_webhook_prefix():
    provider = await load_api_provider("webhook:http://localhost:9999/hook")
    assert isinstance(provider, WebhookProvider)
    assert provider.webhook_url == "http://localhost:9999/hook"


# Two other small registry routes from upstream.


async def test_palm_prefix_is_an_alias_for_google():
    from promptphoo.providers.google.ai_studio import AIStudioChatProvider

    provider = await load_api_provider(
        "palm:gemini-2.5-flash", options={"env": {"GOOGLE_API_KEY": "k"}}
    )
    assert isinstance(provider, AIStudioChatProvider)


async def test_openai_codex_security_is_rejected():
    for path in ("openai:codex-security", "openai:codex-security:gpt-5"):
        with pytest.raises(ValueError, match="Agents/Codex"):
            await load_api_provider(path)
