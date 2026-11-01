"""End-to-end: model-graded-closedqa and select-best with an Anthropic grader (D68).

Both grading prompts are a single system message, which the Anthropic API rejects in Node
(`400 at least one message is required`). promptphoo sends it as the user turn."""

from __future__ import annotations

import json

import httpx
import respx
from click.testing import CliRunner

from promptphoo import db
from promptphoo.cli.main import build_cli

MESSAGES_URL = "https://api.anthropic.com/v1/messages"

CONFIG = """\
prompts:
  - "Say hi to {{name}}"
  - "Greet {{name}}"
providers: [echo]
defaultTest:
  options:
    provider: anthropic:messages:claude-sonnet-4-5
tests:
  - vars: {name: Ann}
    assert:
      - type: model-graded-closedqa
        value: Is it a greeting?
      - type: select-best
        value: Which greeting is warmest?
"""


def _reply(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if not body.get("messages"):
        return httpx.Response(400, json={"error": {"message": "at least one message is required"}})
    text = body["messages"][0]["content"][0]["text"]
    answer = "0" if "warmest" in text else "Y"
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": answer}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    )


@respx.mock
def test_closedqa_and_select_best_grade_with_anthropic(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "promptfooconfig.yaml").write_text(CONFIG)
    route = respx.post(MESSAGES_URL).mock(side_effect=_reply)
    args = ["eval", "-c", "promptfooconfig.yaml", "-o", "out.json", "--no-write", "--no-cache"]
    try:
        result = CliRunner().invoke(build_cli(args), args, catch_exceptions=False)
    finally:
        db.close_db()
    assert result.exit_code in (0, 100), result.output

    bodies = [json.loads(c.request.content) for c in route.calls]
    assert bodies and all(b["messages"] and "system" not in b for b in bodies)
    rows = json.loads((tmp_path / "out.json").read_text())["results"]["results"]
    components = [c for r in rows for c in r["gradingResult"]["componentResults"]]
    closedqa = [c for c in components if c["assertion"]["type"] == "model-graded-closedqa"]
    select_best = [c for c in components if c["assertion"]["type"] == "select-best"]
    assert len(closedqa) == 2 and all(c["pass"] for c in closedqa)
    assert sorted(c["pass"] for c in select_best) == [False, True]
    assert not any("at least one message" in (c.get("reason") or "") for c in components)
