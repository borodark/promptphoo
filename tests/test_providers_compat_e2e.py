"""End-to-end: `promptfoo eval` with deepseek:deepseek-chat against a respx-mocked API."""

from __future__ import annotations

import json

import pytest
import respx
from click.testing import CliRunner
from openai_helpers import ok, openai_env

from promptphoo.cli.main import build_cli

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

CONFIG = """\
description: deepseek e2e
prompts:
  - "Say hello to {{name}}"
providers:
  - id: deepseek:deepseek-chat
    label: DeepSeek
    config:
      temperature: 0.2
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
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_cache_hit_tokens": 40,
            },
        }
    )


@respx.mock
def test_eval_deepseek_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "promptfooconfig.yaml").write_text(CONFIG)
    route = respx.post(DEEPSEEK_URL).mock(side_effect=_reply)
    args = ["eval", "-c", "promptfooconfig.yaml", "-o", "out.json", "--no-write", "--no-table"]
    result = CliRunner().invoke(build_cli(args), args, catch_exceptions=False)

    assert route.call_count == 2
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-deepseek-key"
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "deepseek-chat"
    assert body["temperature"] == 0.2
    assert "max_tokens" not in body
    assert "thinking" not in body
    assert result.exit_code == 100, result.output

    data = json.loads((tmp_path / "out.json").read_text())
    results = sorted(data["results"]["results"], key=lambda r: r["vars"]["name"])
    assert [r["response"]["output"] for r in results] == ["Hello Alice!", "Hello Bob!"]
    assert [r["success"] for r in results] == [True, False]
    # configuredProviderFactory drops the provider-level id, so id() is the class default.
    assert [r["provider"]["id"] for r in results] == ["deepseek:deepseek-chat"] * 2
    assert [r["provider"]["label"] for r in results] == ["DeepSeek"] * 2
    # 60 uncached + 40 cache-hit prompt tokens and 20 completion tokens at deepseek-chat rates.
    expected = 60 * 0.14e-6 + 40 * 0.0028e-6 + 20 * 0.28e-6
    assert results[0]["cost"] == pytest.approx(expected, abs=1e-12)
    stats = data["results"]["stats"]
    assert (stats["successes"], stats["failures"]) == (1, 1)
    assert stats["tokenUsage"]["total"] == 240
