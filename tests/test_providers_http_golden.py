"""HttpProvider parity with real promptfoo 0.125.1 (fixtures from tools/http_golden.js)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from promptphoo import cache
from promptphoo.providers.http import HttpProvider

FIXTURES = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "http_golden_cases.json").read_text())
GOLDEN = {g["name"]: g for g in json.loads((FIXTURES / "http_golden.json").read_text())}
# Headers added by the transport (Node's undici or httpx), not by the provider.
TRANSPORT_HEADERS = {
    "host", "connection", "content-length", "user-agent", "accept", "accept-encoding",
    "accept-language", "sec-fetch-mode", "x-promptfoo-version",
}  # fmt: skip
HOST = "golden.test"


@pytest.fixture(autouse=True)
def _no_cache():
    with cache.with_cache_enabled(False):
        yield


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
async def test_matches_node(case):
    golden = GOLDEN[case["name"]]
    seen: dict = {}
    resp = case["response"]

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = request.url.raw_path.decode()
        seen["headers"] = {
            k: v for k, v in request.headers.items() if k.lower() not in TRANSPORT_HEADERS
        }
        seen["body"] = request.content.decode()
        return httpx.Response(
            resp["status"],
            headers={"content-type": "application/json", **resp.get("headers", {})},
            content=resp["body"].encode(),
            extensions={"reason_phrase": (resp.get("statusText") or "").encode()}
            if resp.get("statusText")
            else {},
        )

    raw = bool(case.get("raw"))
    provider_id = "http" if raw else f"http://{HOST}{case['path']}"
    vars = {**case.get("vars", {}), **({"path": case["path"], "host": HOST} if raw else {})}
    with respx.mock(assert_all_called=False) as mock:
        mock.route(host=HOST).mock(side_effect=handler)
        provider = HttpProvider(provider_id, {"config": case["config"]})
        context = {"vars": vars, "prompt": {"raw": case["prompt"], "label": "p"}}
        if case.get("debug"):
            context["debug"] = True
        try:
            result = await provider.call_api(case["prompt"], context)
        except Exception as e:
            assert "error" in golden, f"unexpected error {e}"
            assert str(e) == golden["error"]
        else:
            assert "response" in golden, f"expected error {golden.get('error')}"
            result.pop("latencyMs", None)
            result.get("metadata", {}).get("http", {}).pop("headers", None)
            result.get("metadata", {}).pop("headers", None)
            assert json.loads(json.dumps(result)) == golden["response"]
            assert list(result) == list(golden["response"])
    if golden["request"] is None:
        assert not seen
    else:
        expected = dict(golden["request"])
        expected["headers"] = {
            k: v for k, v in expected["headers"].items() if k not in TRANSPORT_HEADERS
        }
        assert seen == expected
