"""End-to-end: `promptfoo eval` with openai:chat:gpt-4o-mini against a respx-mocked API."""

from __future__ import annotations

import json

import pytest
import respx
from click.testing import CliRunner
from openai_helpers import CHAT_URL, ok, openai_env

from promptphoo.cli.main import build_cli

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

CONFIG = """\
description: openai e2e
prompts:
  - "Say hello to {{name}}"
providers:
  - openai:chat:gpt-4o-mini
tests:
  - vars: {name: Alice}
    assert:
      - type: contains
        value: Alice
  - vars: {name: Bob}
    assert:
      - type: equals
        value: nope
"""


def _reply(request):
    body = json.loads(request.content)
    name = body["messages"][0]["content"].removeprefix("Say hello to ")
    return ok(
        {
            "choices": [{"message": {"content": f"Hello {name}!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }
    )


@respx.mock
def test_eval_openai_chat_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "promptfooconfig.yaml").write_text(CONFIG)
    route = respx.post(CHAT_URL).mock(side_effect=_reply)
    args = ["eval", "-c", "promptfooconfig.yaml", "-o", "out.json", "--no-write", "--no-table"]
    result = CliRunner().invoke(build_cli(args), args, catch_exceptions=False)

    assert route.call_count == 2
    sent = sorted(json.loads(c.request.content)["messages"][0]["content"] for c in route.calls)
    assert sent == ["Say hello to Alice", "Say hello to Bob"]
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-api-key"
    assert result.exit_code == 100, result.output

    data = json.loads((tmp_path / "out.json").read_text())
    results = sorted(data["results"]["results"], key=lambda r: r["vars"]["name"])
    assert [r["response"]["output"] for r in results] == ["Hello Alice!", "Hello Bob!"]
    assert [r["success"] for r in results] == [True, False]
    # Upstream OpenAiGenericProvider.id() is `openai:<model>`, not the configured path.
    assert [r["provider"]["id"] for r in results] == ["openai:gpt-4o-mini"] * 2
    usage = results[0]["response"]["tokenUsage"]
    assert {k: usage[k] for k in ("numRequests", "total", "prompt", "completion")} == {
        "numRequests": 1,
        "total": 10,
        "prompt": 7,
        "completion": 3,
    }
    assert results[0]["cost"] == pytest.approx(7 * 0.15e-6 + 3 * 0.6e-6, abs=1e-12)
    stats = data["results"]["stats"]
    assert (stats["successes"], stats["failures"]) == (1, 1)
    assert stats["tokenUsage"]["total"] == 20
