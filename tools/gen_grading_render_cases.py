"""Write tests/fixtures/grading_render_cases.json: grading prompts and template contexts that
tools/grading_render_golden.js renders with real Nunjucks.

    PYTHONPATH=src .venv/bin/python tools/gen_grading_render_cases.py
"""

from __future__ import annotations

import json
from pathlib import Path

from promptphoo.matchers import llm_grading
from promptphoo.matchers.shared import try_parse
from promptphoo.prompts import grading

TRAJECTORY = json.dumps(
    {
        "traceId": "t-1",
        "stepCount": 2,
        "compactedStepCount": 2,
        "steps": [
            {"index": 1, "type": "tool", "name": "search_orders", "spanName": "tool.call"},
            {"index": 2, "type": "command", "name": "ls -la", "status": {"code": 2}},
        ],
    },
    indent=2,
)

CASES = [
    {
        "name": "llm-rubric default",
        "template": grading.DEFAULT_GRADING_PROMPT,
        "context": {"topic": "cats", "output": "Hello <b>world</b> \"quoted\"\nline2",
                    "rubric": "Is friendly & polite"},
    },
    {
        "name": "llm-rubric object output and array var",
        "template": grading.DEFAULT_GRADING_PROMPT,
        "context": {"items": [{"a": 1}, "x", 2, [3, 4]], "output": try_parse('{"answer": 42}'),
                    "rubric": {"criteria": "mentions 42"}},
    },
    {
        "name": "factuality default",
        "template": grading.PROMPTFOO_FACTUALITY_PROMPT,
        "context": {"input": "What is the capital of France?", "ideal": "Paris",
                    "completion": "The capital of France is Paris."},
    },
    {
        "name": "closedqa default",
        "template": grading.OPENAI_CLOSED_QA_PROMPT,
        "context": {"input": "Tell me a joke", "criteria": "is funny",
                    "completion": try_parse("[1, 2]")},
    },
    {
        "name": "g-eval steps",
        "template": llm_grading.GEVAL_PROMPT_STEPS,
        "context": {"criteria": "Coherence - the collective quality of all sentences"},
    },
    {
        "name": "g-eval evaluate",
        "template": llm_grading.GEVAL_PROMPT_EVALUATE,
        "context": {"criteria": "Coherence", "steps": "Read it\n- Score it", "maxScore": "10",
                    "input": "Write a poem", "output": try_parse('{"poem": "roses"}')},
    },
    {
        "name": "trajectory goal success default",
        "template": grading.TRAJECTORY_GOAL_SUCCESS_PROMPT,
        "context": {"goal": "Refund the order", "output": "Refunded order 123",
                    "trajectory": TRAJECTORY},
    },
    {
        "name": "agent-rubric default",
        "template": grading.DEFAULT_AGENT_GRADING_PROMPT,
        "context": {"output": "Created file foo.py", "rubric": "Creates foo.py"},
    },
    {
        "name": "legacy non-JSON prompt",
        "template": "Grade {{ output }} against {{ rubric }}.\n{% for i in items %}- {{ i }}\n"
        "{% endfor %}{{ items | join(', ') }}",
        "context": {"output": "out", "rubric": "rub", "items": [{"k": "v"}, "plain"]},
    },
    {
        "name": "JSON custom prompt with nested values",
        "template": json.dumps(
            [{"role": "system", "content": ["{{ a }}", {"nested": "{{ b }}"}, 3, True, None]},
             {"role": "user", "content": "{{ obj }}"}]
        ),
        "context": {"a": "A", "b": "B", "obj": {"x": [1, {"y": 2}]}},
    },
]  # fmt: skip


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    (out / "grading_render_cases.json").write_text(
        json.dumps(CASES, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(CASES)} cases")


if __name__ == "__main__":
    main()
