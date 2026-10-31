"""The scheduler wired into the eval loop (evaluator.ts callActiveProvider / Evaluator)."""

from __future__ import annotations

import pytest
from scheduler_fakes import FakeTime

from promptphoo import db, paths
from promptphoo.evaluator.run import evaluate
from promptphoo.models.eval import Eval
from promptphoo.providers.base import ApiProvider
from promptphoo.scheduler import clock
from promptphoo.scheduler.provider_call_execution_context import (
    get_provider_call_execution_context,
)

RATE_LIMITED = {
    "error": "API error: 429 Too Many Requests",
    "metadata": {"http": {"status": 429, "headers": {"retry-after": "3"}}},
}


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("PROMPTFOO_DISABLE_ADAPTIVE_SCHEDULER", raising=False)
    paths.set_config_directory_path(None)
    db.close_db()
    db.run_db_migrations()
    yield tmp_path
    db.close_db()


class ScriptedProvider(ApiProvider):
    def __init__(self, responses, provider_id="scripted", config=None):
        super().__init__({"id": provider_id, "config": config})
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def call_api(self, prompt, context=None, options=None):
        execution = get_provider_call_execution_context() or {}
        self.calls.append({"options": options, "registry": execution.get("rateLimitRegistry")})
        item = self.responses.pop(0) if self.responses else {"output": prompt}
        if isinstance(item, BaseException):
            raise item
        return item


def _run(fake, provider, options=None):
    suite = {
        "providers": [provider],
        "prompts": [{"raw": "Hello {{name}}", "label": "greet"}],
        "tests": [{"vars": {"name": "World"}}],
    }
    record = Eval({"tests": suite["tests"]}, author=None)

    async def body():
        return await evaluate(suite, record, {"cache": False, **(options or {})})

    fake.run(body)
    return record.results


def test_rate_limited_response_is_retried(config_dir, monkeypatch):
    fake = FakeTime(monkeypatch)
    provider = ScriptedProvider([RATE_LIMITED, RATE_LIMITED, {"output": "recovered"}])
    results = _run(fake, provider)
    assert len(provider.calls) == 3
    assert results[0].success
    assert results[0].response["output"] == "recovered"
    # retry-after: 3 on both 429s, on the virtual clock
    assert clock.date_now() - fake.start_ms >= 6000
    # the evaluator passes the scheduler's header observer and scopes the registry
    assert callable(provider.calls[0]["options"]["onResponseHeaders"])
    assert provider.calls[0]["registry"] is not None


def test_retries_exhausted_keep_the_provider_error(config_dir, monkeypatch):
    fake = FakeTime(monkeypatch)
    provider = ScriptedProvider([RATE_LIMITED] * 10)
    results = _run(fake, provider)
    assert len(provider.calls) == 4  # 1 + DEFAULT_RETRY_POLICY.maxRetries
    assert results[0].error == "API error: 429 Too Many Requests"
    assert not results[0].success


def test_provider_max_retries_zero(config_dir, monkeypatch):
    fake = FakeTime(monkeypatch)
    provider = ScriptedProvider([RATE_LIMITED, {"output": "never"}], config={"maxRetries": 0})
    results = _run(fake, provider)
    assert len(provider.calls) == 1
    assert results[0].error == "API error: 429 Too Many Requests"


def test_handles_own_retries(config_dir, monkeypatch):
    fake = FakeTime(monkeypatch)
    provider = ScriptedProvider([RATE_LIMITED, {"output": "never"}])
    provider.handles_own_retries = True
    results = _run(fake, provider)
    assert len(provider.calls) == 1
    assert results[0].error == "API error: 429 Too Many Requests"


def test_scheduler_disabled_by_env(config_dir, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_DISABLE_ADAPTIVE_SCHEDULER", "true")
    fake = FakeTime(monkeypatch)
    provider = ScriptedProvider([RATE_LIMITED, {"output": "never"}])
    results = _run(fake, provider)
    assert len(provider.calls) == 1
    assert provider.calls[0]["options"] is None
    assert results[0].error == "API error: 429 Too Many Requests"


def test_ordinary_errors_are_not_retried(config_dir, monkeypatch):
    fake = FakeTime(monkeypatch)
    provider = ScriptedProvider([{"error": "Invalid API key"}, {"output": "never"}])
    results = _run(fake, provider)
    assert len(provider.calls) == 1
    assert results[0].error == "Invalid API key"
    assert clock.date_now() == fake.start_ms


def test_transient_exceptions_are_retried(config_dir, monkeypatch):
    fake = FakeTime(monkeypatch)
    provider = ScriptedProvider([Exception("HTTP 503: Service Unavailable"), {"output": "ok"}])
    results = _run(fake, provider)
    assert len(provider.calls) == 2
    assert results[0].success


def test_non_transient_exception_becomes_error_row(config_dir, monkeypatch):
    fake = FakeTime(monkeypatch)
    provider = ScriptedProvider([ValueError("boom"), {"output": "never"}])
    results = _run(fake, provider)
    assert len(provider.calls) == 1
    assert "boom" in results[0].error
