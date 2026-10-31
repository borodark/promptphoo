"""End-to-end: `promptfoo eval` with azure:chat:my-deployment against a respx-mocked endpoint,
the host coming from AZURE_API_HOST."""

from __future__ import annotations

import json

import pytest
import respx
from azure_helpers import API_VERSION, azure_env, ok
from click.testing import CliRunner

from promptphoo.cli.main import build_cli

pytestmark = pytest.mark.usefixtures(azure_env.__name__)

CONFIG = """\
description: azure e2e
prompts:
  - "Say hello to {{name}}"
providers:
  - id: azure:chat:my-deployment
    config:
      modelName: gpt-4o
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

URL = "https://my-resource.openai.azure.com/openai/deployments/my-deployment/chat/completions"


def _reply(request):
    body = json.loads(request.content)
    name = body["messages"][0]["content"].removeprefix("Say hello to ")
    return ok(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": f"Hello {name}!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }
    )


@respx.mock
def test_eval_azure_chat_end_to_end(tmp_path, azure_env):
    azure_env.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    azure_env.setenv("AZURE_API_HOST", "my-resource.openai.azure.com")
    azure_env.setenv("AZURE_API_KEY", "azure-test-key")
    azure_env.chdir(tmp_path)
    (tmp_path / "promptfooconfig.yaml").write_text(CONFIG)
    route = respx.post(URL).mock(side_effect=_reply)
    args = ["eval", "-c", "promptfooconfig.yaml", "-o", "out.json", "--no-write", "--no-table"]
    result = CliRunner().invoke(build_cli(args), args, catch_exceptions=False)

    assert route.call_count == 2
    request = route.calls[0].request
    assert request.url.params["api-version"] == API_VERSION
    assert request.headers["api-key"] == "azure-test-key"
    assert "Authorization" not in request.headers
    assert json.loads(request.content)["model"] == "my-deployment"
    assert result.exit_code == 100, result.output

    data = json.loads((tmp_path / "out.json").read_text())
    results = sorted(data["results"]["results"], key=lambda r: r["vars"]["name"])
    assert [r["response"]["output"] for r in results] == ["Hello Alice!", "Hello Bob!"]
    assert [r["success"] for r in results] == [True, False]
    assert [r["provider"]["id"] for r in results] == ["azure:chat:my-deployment"] * 2
    assert results[0]["response"]["guardrails"] == {
        "flagged": False,
        "flaggedInput": False,
        "flaggedOutput": False,
    }
    # gpt-4o on Azure: $2.50 / $10 per million tokens.
    assert results[0]["cost"] == pytest.approx((7 * 2.5 + 3 * 10) / 1e6, abs=1e-12)
    stats = data["results"]["stats"]
    assert (stats["successes"], stats["failures"]) == (1, 1)
    assert stats["tokenUsage"]["total"] == 20
