"""A canned LLM HTTP server for span parity runs: python fake_llm.py PORT"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

USAGE = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
RESPONSES = {
    "/v1/chat/completions": {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello there"},
                     "finish_reason": "stop"}],
        "usage": USAGE,
    },
    "/v1/completions": {
        "id": "cmpl-1", "object": "text_completion", "created": 1,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [{"text": "Hi", "index": 0, "finish_reason": "stop"}], "usage": USAGE,
    },
    "/v1/responses": {
        "id": "resp_1", "object": "response", "model": "gpt-4.1", "status": "completed",
        "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "Hi", "annotations": []}]}],
        "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    },
    "/v1/messages": {
        "id": "msg_1", "type": "message", "role": "assistant",
        "model": "claude-3-5-haiku-20241022", "content": [{"type": "text", "text": "Hi"}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 2},
    },
    "/api/generate": {"model": "llama3", "response": "Hi", "done": True,
                      "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 2},
    "/api/chat": {"model": "llama3", "message": {"role": "assistant", "content": "Hi"},
                  "done": True, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 2},
}  # fmt: skip


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        self.rfile.read(length)
        path = self.path.split("?")[0]
        body = RESPONSES.get(path)
        data = json.dumps(body if body is not None else {"error": "not found"}).encode()
        self.send_response(200 if body is not None else 404)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def serve(port: int = 0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    serve(int(sys.argv[1])).serve_forever()
