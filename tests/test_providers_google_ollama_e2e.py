"""End-to-end evals through the google: and ollama: providers, with the network mocked."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from promptphoo import db, paths
from promptphoo.evaluator.run import evaluate
from promptphoo.models.eval import Eval
from promptphoo.providers import load_api_provider


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("IS_TESTING", raising=False)
    for name in ("GEMINI_API_KEY", "OLLAMA_BASE_URL", "OLLAMA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    paths.set_config_directory_path(None)
    db.close_db()
    db.run_db_migrations()
    yield tmp_path
    db.close_db()


def _suite(provider):
    return {
        "providers": [provider],
        "prompts": [{"raw": "What is the capital of {{country}}?", "label": "capital"}],
        "tests": [
            {"vars": {"country": "France"}, "assert": [{"type": "contains", "value": "Paris"}]},
            {"vars": {"country": "Peru"}, "assert": [{"type": "contains", "value": "Paris"}]},
        ],
    }


def _answer(prompt: str) -> str:
    return "Paris." if "France" in prompt else "Lima."


@respx.mock
async def test_google_gemini_eval(config_dir, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    def reply(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "test-key"
        prompt = json.loads(request.content)["contents"][0]["parts"][0]["text"]
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": _answer(prompt)}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 2,
                                  "totalTokenCount": 18, "thoughtsTokenCount": 8},
            },
        )  # fmt: skip

    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    ).mock(side_effect=reply)
    provider = await load_api_provider("google:gemini-2.5-flash")
    assert provider.id() == "google:gemini-2.5-flash"
    suite = _suite(provider)
    record = Eval({"tests": suite["tests"]}, author=None)
    await evaluate(suite, record, {"cache": False})

    assert route.call_count == 2
    results = sorted(record.results, key=lambda r: r.test_idx)
    assert [r.success for r in results] == [True, False]
    response = results[0].response
    assert response["output"] == "Paris."
    assert response["tokenUsage"]["completionDetails"]["reasoning"] == 8
    # gemini-2.5-flash: 0.3/1M input, 2.5/1M output (thinking billed as output).
    assert results[0].cost == pytest.approx((8 * 0.3 + 10 * 2.5) / 1e6)
    assert record.to_evaluate_summary()["stats"]["successes"] == 1


@respx.mock
async def test_ollama_chat_eval(config_dir):
    def reply(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "llama3"
        answer = _answer(body["messages"][0]["content"])
        lines = [
            {"message": {"role": "assistant", "content": answer[:2]}, "done": False},
            {"message": {"role": "assistant", "content": answer[2:]}, "done": True,
             "done_reason": "stop", "prompt_eval_count": 12, "eval_count": 3},
        ]  # fmt: skip
        return httpx.Response(200, text="\n".join(json.dumps(line) for line in lines) + "\n")

    route = respx.post("http://localhost:11434/api/chat").mock(side_effect=reply)
    provider = await load_api_provider("ollama:chat:llama3")
    assert provider.id() == "ollama:chat:llama3"
    suite = _suite(provider)
    record = Eval({"tests": suite["tests"]}, author=None)
    await evaluate(suite, record, {"cache": False})

    assert route.call_count == 2
    results = sorted(record.results, key=lambda r: r.test_idx)
    assert [r.success for r in results] == [True, False]
    assert results[0].response["output"] == "Paris."
    assert results[0].response["finishReason"] == "stop"
    usage = results[0].response["tokenUsage"]
    assert (usage["prompt"], usage["completion"], usage["total"], usage["numRequests"]) == (
        12,
        3,
        15,
        1,
    )
