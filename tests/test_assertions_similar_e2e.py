"""End-to-end: `promptfoo eval` with `similar` assertions graded by upstream's default embedding
provider (openai:embedding:text-embedding-3-large) against a respx-mocked API."""

from __future__ import annotations

import json

import pytest
import respx
from click.testing import CliRunner
from openai_helpers import ok, openai_env

from promptphoo.cli.main import build_cli

pytestmark = pytest.mark.usefixtures(openai_env.__name__)

EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"

CONFIG = """\
description: similar e2e
prompts:
  - "{{text}}"
providers:
  - echo
tests:
  - vars: {text: The cat sat on the mat}
    assert:
      - type: similar
        value: A cat is sitting on a mat
        threshold: 0.8
      - type: not-similar
        value: Stock prices fell sharply
  - vars: {text: Stock prices fell sharply}
    assert:
      - type: similar:dot
        value: A cat is sitting on a mat
        threshold: 0.8
"""

VECTORS = {
    "The cat sat on the mat": [1.0, 0.0, 0.0],
    "A cat is sitting on a mat": [0.96, 0.28, 0.0],
    "Stock prices fell sharply": [0.0, 0.0, 1.0],
}


def _embed(request):
    body = json.loads(request.content)
    return ok(
        {
            "data": [{"embedding": VECTORS[body["input"]]}],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
    )


@respx.mock
def test_eval_similar_with_default_embedding_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "promptfooconfig.yaml").write_text(CONFIG)
    route = respx.post(EMBEDDINGS_URL).mock(side_effect=_embed)
    args = ["eval", "-c", "promptfooconfig.yaml", "-o", "out.json", "--no-write", "--no-table"]
    result = CliRunner().invoke(build_cli(args), args, catch_exceptions=False)
    assert result.exit_code == 100, result.output

    bodies = [json.loads(c.request.content) for c in route.calls]
    assert {b["model"] for b in bodies} == {"text-embedding-3-large"}
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-api-key"
    assert sorted({b["input"] for b in bodies}) == sorted(VECTORS)

    data = json.loads((tmp_path / "out.json").read_text())
    rows = sorted(data["results"]["results"], key=lambda r: r["vars"]["text"])
    stocks, cat = rows
    assert cat["success"] is True
    similar, not_similar = cat["gradingResult"]["componentResults"]
    assert similar["reason"] == "Similarity 0.96 is greater than or equal to threshold 0.8"
    assert similar["score"] == pytest.approx(0.96)
    assert not_similar["pass"] is True
    assert not_similar["reason"] == "Similarity 0.00 is less than threshold 0.75"
    assert not_similar["score"] == 1
    assert similar["tokensUsed"]["prompt"] == 8

    assert stocks["success"] is False
    (dot,) = stocks["gradingResult"]["componentResults"]
    assert dot["assertion"]["type"] == "similar:dot"
    assert (dot["pass"], dot["score"]) == (False, 0)
    assert dot["reason"] == "Similarity 0.00 is less than threshold 0.8"
