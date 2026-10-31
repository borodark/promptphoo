"""Recording / replaying proxy for api.anthropic.com.

  proxy.py record <port> <dir>   forward to Anthropic, save each exchange to <dir>/NNNN.json
  proxy.py replay <port> <dir>   answer from <dir> by (path, canonical body); misses -> 599

The API key is never written: x-api-key and authorization headers are dropped from records.
"""

import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

UPSTREAM = "https://api.anthropic.com"
SECRET_HEADERS = {"x-api-key", "authorization"}
HOP = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}

mode, port, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
counter = [len(list(out.glob("[0-9]*.json")))]
client = httpx.Client(timeout=600)
REPLAY_MISS = b'{"type":"error","error":{"type":"replay_miss","message":"no recorded exchange"}}'


def key(path: str, body: bytes) -> str:
    try:
        canon = json.dumps(json.loads(body), sort_keys=True, separators=(",", ":"))
    except ValueError:
        canon = body.decode("utf-8", "replace")
    return hashlib.sha256(f"{path}\n{canon}".encode()).hexdigest()


index: dict[str, list[dict]] = {}
if mode == "replay":
    for f in sorted(out.glob("[0-9]*.json")):
        rec = json.loads(f.read_text())
        index.setdefault(rec["key"], []).append(rec)
    (out / "misses").mkdir(exist_ok=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - quiet
        pass

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        k = key(self.path, body)
        headers = {h.lower(): v for h, v in self.headers.items()}
        safe_headers = {h: v for h, v in headers.items() if h not in SECRET_HEADERS | HOP}
        if mode == "record":
            fwd = {h: v for h, v in headers.items() if h not in HOP}
            r = client.post(UPSTREAM + self.path, content=body, headers=fwd)
            rec = {
                "key": k,
                "path": self.path,
                "requestHeaders": safe_headers,
                "requestBody": body.decode("utf-8", "replace"),
                "status": r.status_code,
                "responseHeaders": {
                    h: v
                    for h, v in r.headers.items()
                    if h.lower().startswith(
                        ("content-type", "anthropic-", "retry-after", "request-id")
                    )
                },
                "responseBody": r.text,
            }
            with lock:
                counter[0] += 1
                (out / f"{counter[0]:04d}.json").write_text(json.dumps(rec, indent=1))
        else:
            with lock:
                # Identical requests (e.g. g-eval's step generation) get their recordings in the
                # order they were recorded; the last one repeats if the replayer asks again.
                recs = index.get(k)
                rec = (recs.pop(0) if len(recs) > 1 else recs[0]) if recs else None
                if rec is None:
                    counter[0] += 1
                    (out / "misses" / f"{counter[0]:04d}.json").write_text(
                        json.dumps(
                            {
                                "path": self.path,
                                "requestHeaders": safe_headers,
                                "requestBody": body.decode("utf-8", "replace"),
                            },
                            indent=1,
                        )
                    )
            if rec is None:
                self.send_response(599)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(REPLAY_MISS)
                return
            (out / "replayed_headers.jsonl").open("a").write(
                json.dumps({"key": k, "requestHeaders": safe_headers}) + "\n"
            )
        payload = rec["responseBody"].encode()
        self.send_response(rec["status"])
        for h, v in rec["responseHeaders"].items():
            self.send_header(h, v)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
