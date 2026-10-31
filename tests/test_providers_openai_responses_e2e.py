"""End-to-end: `promptfoo eval` with openai:responses:gpt-5-mini, and a bare GPT-5.6+ id that
routes to the Responses API, against a respx-mocked API."""

from __future__ import annotations

import json

import pytest
import respx
from click.testing import CliRunner
from openai_helpers import CHAT_URL, ok, openai_env

from promptphoo.cli.main import build_cli

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

RESPONSES_URL = "https://api.openai.com/v1/responses"

CONFIG = """\
description: openai responses e2e
prompts:
  - "Say hello to {{{{name}}}}"
providers:
  - {provider}
tests:
  - vars: {{name: Alice}}
    assert:
      - type: contains
        value: Alice
  - vars: {{name: Bob}}
    assert:
      - type: equals
        value: nope
"""


def _reply(request):
    body = json.loads(request.content)
    name = body["input"].removeprefix("Say hello to ")
    return ok(
        {
            "id": f"resp_{name}",
            "status": "completed",
            "model": body["model"],
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"Hello {name}!"}],
                },
            ],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
                "output_tokens_details": {"reasoning_tokens": 1},
            },
        }
    )


def _run_eval(tmp_path, monkeypatch, provider: str):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "promptfooconfig.yaml").write_text(CONFIG.format(provider=provider))
    args = ["eval", "-c", "promptfooconfig.yaml", "-o", "out.json", "--no-write", "--no-table"]
    result = CliRunner().invoke(build_cli(args), args, catch_exceptions=False)
    data = json.loads((tmp_path / "out.json").read_text())
    return result, sorted(data["results"]["results"], key=lambda r: r["vars"]["name"]), data


@respx.mock
def test_eval_openai_responses_end_to_end(tmp_path, monkeypatch):
    route = respx.post(RESPONSES_URL).mock(side_effect=_reply)
    chat = respx.post(CHAT_URL)
    result, results, data = _run_eval(tmp_path, monkeypatch, "openai:responses:gpt-5-mini")

    assert route.call_count == 2 and chat.call_count == 0
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "gpt-5-mini"
    assert body["text"] == {"format": {"type": "text"}}
    assert "temperature" not in body  # gpt-5-mini is a reasoning model
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-api-key"
    assert result.exit_code == 100, result.output

    assert [r["response"]["output"] for r in results] == ["Hello Alice!", "Hello Bob!"]
    assert [r["success"] for r in results] == [True, False]
    assert [r["provider"]["id"] for r in results] == ["openai:gpt-5-mini"] * 2
    usage = results[0]["response"]["tokenUsage"]
    assert {k: usage[k] for k in ("numRequests", "total", "prompt", "completion")} == {
        "numRequests": 1,
        "total": 10,
        "prompt": 7,
        "completion": 3,
    }
    assert usage["completionDetails"]["reasoning"] == 1
    assert results[0]["response"]["metadata"]["responseId"] == "resp_Alice"
    assert results[0]["cost"] == pytest.approx(7 * 0.25e-6 + 3 * 2e-6, abs=1e-12)
    stats = data["results"]["stats"]
    assert (stats["successes"], stats["failures"]) == (1, 1)


@respx.mock
def test_eval_bare_gpt56_routes_to_responses(tmp_path, monkeypatch):
    route = respx.post(RESPONSES_URL).mock(side_effect=_reply)
    chat = respx.post(CHAT_URL)
    result, results, _ = _run_eval(tmp_path, monkeypatch, "openai:gpt-5.6")

    assert route.call_count == 2 and chat.call_count == 0
    assert json.loads(route.calls[0].request.content)["model"] == "gpt-5.6"
    assert result.exit_code == 100, result.output
    assert [r["response"]["output"] for r in results] == ["Hello Alice!", "Hello Bob!"]
    assert [r["provider"]["id"] for r in results] == ["openai:gpt-5.6"] * 2
