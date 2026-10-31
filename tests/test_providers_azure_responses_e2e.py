"""End-to-end: `promptfoo eval` with azure:responses:my-deployment against a respx-mocked
endpoint, the host coming from AZURE_API_HOST."""

from __future__ import annotations

import json

import pytest
import respx
from azure_helpers import azure_env, ok
from click.testing import CliRunner

from promptphoo.cli.main import build_cli

pytestmark = pytest.mark.usefixtures(azure_env.__name__)

CONFIG = """\
description: azure responses e2e
prompts:
  - "Say hello to {{name}}"
providers:
  - id: azure:responses:my-deployment
    config:
      passthrough:
        model: gpt-4.1
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

URL = "https://my-resource.openai.azure.com/openai/v1/responses"


def _reply(request):
    body = json.loads(request.content)
    name = body["input"].removeprefix("Say hello to ")
    return ok(
        {
            "id": f"resp_{name}",
            "status": "completed",
            "model": body["model"],
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"Hello {name}!"}],
                }
            ],
            "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        }
    )


@respx.mock
def test_eval_azure_responses_end_to_end(tmp_path, azure_env):
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
    assert request.url.params["api-version"] == "preview"
    assert request.headers["api-key"] == "azure-test-key"
    assert "Authorization" not in request.headers
    body = json.loads(request.content)
    assert body["model"] == "gpt-4.1"
    assert body["text"] == {"format": {"type": "text"}}
    assert result.exit_code == 100, result.output

    data = json.loads((tmp_path / "out.json").read_text())
    results = sorted(data["results"]["results"], key=lambda r: r["vars"]["name"])
    assert [r["response"]["output"] for r in results] == ["Hello Alice!", "Hello Bob!"]
    assert [r["success"] for r in results] == [True, False]
    assert [r["provider"]["id"] for r in results] == ["azure:responses:my-deployment"] * 2
    assert results[0]["response"]["metadata"]["responseId"] == "resp_Alice"
    # gpt-4.1 on Azure: $2 / $8 per million tokens.
    assert results[0]["cost"] == pytest.approx((7 * 2 + 3 * 8) / 1e6, abs=1e-12)
    stats = data["results"]["stats"]
    assert (stats["successes"], stats["failures"]) == (1, 1)
    assert stats["tokenUsage"]["total"] == 20
