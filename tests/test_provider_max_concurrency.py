"""provider.config.maxConcurrency caps one provider's pool below the run's concurrency (D80)."""

import logging

import pytest

from promptphoo.api import evaluate
from promptphoo.scheduler.rate_limit_registry import _get_provider_max_concurrency

GRADER = """
import json, os, threading, time
_lock = threading.Lock()
_inflight = 0
def call_api(prompt, options, context):
    global _inflight
    path = options["config"]["log"]
    with _lock:
        _inflight += 1
        now = _inflight
    with open(path, "a") as f:
        f.write(f"{now}\\n")
    time.sleep(0.15)
    with _lock:
        _inflight -= 1
    return {"output": json.dumps({"pass": True, "score": 1, "reason": "graded"})}
"""


def _run(tmp_path, grader_config):
    (tmp_path / "grader.py").write_text(GRADER)
    log = tmp_path / "inflight.log"
    record = evaluate(
        {
            "basePath": str(tmp_path),
            "prompts": ["{{w}}"],
            "providers": ["echo"],
            "tests": [
                {
                    "vars": {"w": str(i)},
                    "assert": [
                        {
                            "type": "llm-rubric",
                            "value": "anything",
                            "provider": {
                                "id": "file://grader.py",
                                "config": {"log": str(log), **grader_config},
                            },
                        }
                    ],
                }
                for i in range(8)
            ],
        },
        {"cache": False, "maxConcurrency": 8},
    )
    summary = record.to_evaluate_summary()
    assert summary["stats"]["successes"] == 8, summary["stats"]
    return max(int(line) for line in log.read_text().split())


def test_grader_pool_is_capped_by_its_own_config(tmp_path):
    assert _run(tmp_path, {"maxConcurrency": 2}) == 2


def test_without_a_cap_the_grader_follows_the_run(tmp_path):
    assert _run(tmp_path, {}) > 2


class _Provider:
    def __init__(self, config):
        self.config = config

    def id(self):
        return "p"


@pytest.mark.parametrize(
    "raw, expected",
    [(3, 3), (3.0, 3), ("4", 4), (" 2 ", 2), (None, None)],
)
def test_reads_valid_caps(raw, expected):
    assert _get_provider_max_concurrency(_Provider({"maxConcurrency": raw})) == expected


@pytest.mark.parametrize("raw", [0, -1, 1.5, True, "two", ""])
def test_rejects_invalid_caps_with_a_warning(raw, caplog):
    with caplog.at_level(logging.WARNING):
        assert _get_provider_max_concurrency(_Provider({"maxConcurrency": raw})) is None
    assert "Ignoring invalid provider.config.maxConcurrency" in caplog.text
    assert '"providerId":"p"' in caplog.text
