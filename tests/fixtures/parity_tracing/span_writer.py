"""A traced target for the trace-assertion parity fixture.

It writes its spans straight into the `spans` table of the shared promptfoo database, under the
trace id of the traceparent it was called with. The stdlib sqlite3 module is used rather than
promptphoo's trace store, so Node's Python provider (a separate process) and promptphoo (in
process) run exactly the same code.
"""

import json
import os
import sqlite3
import uuid

SPANS = [
    ("agent.run", None, 1000, 1120, None, None, {"gen_ai.operation.name": "invoke_agent"}),
    ("llm.chat", "agent", 1005, 1045, None, None, {"gen_ai.request.model": "fake-model"}),
    ("tool.search", "agent", 1050, 1080, None, None,
     {"tool.name": "search", "tool.arguments": '{"q": "paris"}'}),
    ("tool.fetch", "agent", 1085, 1110, 2, "fetch failed: timeout", {"tool.name": "fetch"}),
]  # fmt: skip


def _db_path():
    config_dir = os.environ.get("PROMPTFOO_CONFIG_DIR") or os.path.expanduser("~/.promptfoo")
    return os.path.join(config_dir, "promptfoo.db")


def call_api(prompt, options, context):
    traceparent = (context or {}).get("traceparent")
    if not traceparent:
        return {"output": "untraced"}
    _, trace_id, parent_span_id, _ = traceparent.split("-")
    span_ids = {"agent": "a1" * 8}
    conn = sqlite3.connect(_db_path(), timeout=30)
    try:
        with conn:
            for index, (name, parent, start, end, code, message, attrs) in enumerate(SPANS):
                span_id = span_ids["agent"] if index == 0 else f"{index:016x}"
                conn.execute(
                    "INSERT INTO spans (id, trace_id, span_id, parent_span_id, name, start_time, "
                    "end_time, attributes, status_code, status_message) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        trace_id,
                        span_id,
                        span_ids[parent] if parent else parent_span_id,
                        name,
                        start,
                        end,
                        json.dumps(attrs),
                        code,
                        message,
                    ),
                )
    finally:
        conn.close()
    return {"output": "Paris"}
