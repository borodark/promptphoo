"""Fuzz promptphoo's OTLP protobuf decoder against protobufjs, through tools/otlp_node.mjs.

    PYTHONPATH=src python tools/otlp_protobuf_golden.py <promptfoo dist/src> [N] [SEED] [--write F]

Random ExportTraceServiceRequests are encoded by protobufjs, then mutated (truncated, bit
flipped, spliced, random bytes). Every payload is decoded by protobufjs and by
`tracing/protobuf.py`, and the results (or error messages) must agree. `--write F` also saves
the payloads with Node's results as JSONL, the golden file tests/test_tracing_protobuf.py
replays offline.
"""

import base64
import json
import math
import random
import subprocess
import sys

from promptphoo.jsvalues import json_stringify
from promptphoo.tracing.protobuf import decode_export_trace_service_request

ARGS = [a for a in sys.argv[1:] if a != "--write"]
WRITE = sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None
if WRITE:
    ARGS.remove(WRITE)
DIST = ARGS[0] if ARGS else ""
N = int(ARGS[1]) if len(ARGS) > 1 else 3000
SEED = int(ARGS[2]) if len(ARGS) > 2 else 1
rng = random.Random(SEED)


def node(mode, text):
    proc = subprocess.run(
        ["node", "tools/otlp_node.mjs", DIST, mode],
        input=text,
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        print(proc.stderr[-3000:])
        raise SystemExit(1)
    return proc.stdout


def rand_value(depth=0):
    kinds = ["s", "b", "i", "d", "bytes"] + (["arr", "kv"] if depth < 3 else [])
    k = rng.choice(kinds)
    if k == "s":
        return {"stringValue": rng.choice(["", "x", "héllo", "a" * 50, "☃"])}
    if k == "b":
        return {"boolValue": rng.random() < 0.5}
    if k == "i":
        return {
            "intValue": str(
                rng.choice(
                    [
                        0,
                        1,
                        -1,
                        2**31,
                        -(2**63),
                        2**63 - 1,
                        2**53 + 1,
                        rng.randint(-(10**12), 10**12),
                    ]
                )
            )
        }
    if k == "d":
        return {"doubleValue": rng.choice([0.5, -0.0, 1e300, 3.14159, -2.0, 1e-7])}
    if k == "bytes":
        return {"bytesValue": base64.b64encode(rng.randbytes(rng.randint(0, 5))).decode()}
    if k == "arr":
        return {"arrayValue": {"values": [rand_value(depth + 1) for _ in range(rng.randint(0, 3))]}}
    return {
        "kvlistValue": {
            "values": [
                {"key": f"k{i}", "value": rand_value(depth + 1)} for i in range(rng.randint(0, 3))
            ]
        }
    }


def rand_attrs():
    return [
        {"key": rng.choice(["a", "b", "evaluation.id", "x.y"]), "value": rand_value()}
        for _ in range(rng.randint(0, 4))
    ]


def rand_span():
    span = {
        "traceId": base64.b64encode(rng.randbytes(rng.choice([16, 16, 8, 0]))).decode(),
        "spanId": base64.b64encode(rng.randbytes(rng.choice([8, 8, 4]))).decode(),
        "name": rng.choice(["op", "", "llm.call"]),
        "kind": rng.randint(0, 7),
        "startTimeUnixNano": str(rng.choice([0, 1700000000123456789, 2**64 - 1, 12345])),
        "attributes": rand_attrs(),
    }
    if rng.random() < 0.5:
        span["endTimeUnixNano"] = str(rng.randint(0, 2**63))
    if rng.random() < 0.5:
        span["parentSpanId"] = base64.b64encode(rng.randbytes(8)).decode()
    if rng.random() < 0.5:
        span["status"] = {"code": rng.randint(0, 2), "message": rng.choice(["", "boom"])}
    if rng.random() < 0.3:
        span["events"] = [{"name": "e", "timeUnixNano": "5", "attributes": rand_attrs()}]
    if rng.random() < 0.3:
        span["links"] = [{"traceId": base64.b64encode(rng.randbytes(16)).decode(), "flags": 3}]
    if rng.random() < 0.3:
        span["flags"] = rng.randint(0, 2**32 - 1)
    if rng.random() < 0.2:
        span["traceState"] = "k=v"
    return span


def rand_request():
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": rand_attrs()} if rng.random() < 0.7 else None,
                "scopeSpans": [
                    {
                        "scope": {"name": "s", "version": rng.choice(["", "1"])}
                        if rng.random() < 0.7
                        else None,
                        "spans": [rand_span() for _ in range(rng.randint(0, 3))],
                    }
                    for _ in range(rng.randint(0, 2))
                ],
                "schemaUrl": rng.choice(["", "https://x"]),
            }
            for _ in range(rng.randint(0, 2))
        ]
    }


def strip_none(v):
    if isinstance(v, dict):
        return {k: strip_none(x) for k, x in v.items() if x is not None}
    if isinstance(v, list):
        return [strip_none(x) for x in v]
    return v


def mutate(data: bytes) -> bytes:
    b = bytearray(data)
    op = rng.randint(0, 5)
    if op == 0 and b:
        del b[rng.randrange(len(b)) :]
    elif op == 1 and b:
        b[rng.randrange(len(b))] = rng.randrange(256)
    elif op == 2:
        pos = rng.randrange(len(b) + 1)
        b[pos:pos] = rng.randbytes(rng.randint(1, 4))
    elif op == 3 and b:
        for _ in range(rng.randint(1, 4)):
            b[rng.randrange(len(b))] ^= 1 << rng.randrange(8)
    elif op == 4:
        return rng.randbytes(rng.randint(0, 40))
    else:
        b += bytes([rng.choice([0x0A, 0x12, 0x08, 0x7F, 0x80, 0xFF]), rng.randrange(256)])
    return bytes(b)


def to_plain(v):
    if isinstance(v, bytes):
        return {"$bytes": v.hex()}
    if isinstance(v, dict):
        return {k: to_plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [to_plain(x) for x in v]
    if isinstance(v, float):
        if not math.isfinite(v):
            return {"$number": "Infinity" if v > 0 else ("-Infinity" if v < 0 else "NaN")}
        if v == 0 and math.copysign(1, v) < 0:
            return {"$number": "-0"}
    return v


def canon(text):
    return json.dumps(json.loads(text), sort_keys=True)


def main() -> None:
    requests = [strip_none(rand_request()) for _ in range(max(1, N // 10))]
    valid = []
    for req in requests:
        valid.append(base64.b64decode(node("encode", json.dumps(req))))
    payloads = list(valid)
    while len(payloads) < N:
        payloads.append(mutate(rng.choice(valid)))
    lines = [base64.b64encode(p).decode() for p in payloads]
    out = node("decode-lines", "\n".join(lines) + "\n").splitlines()
    node_out = [line[2:] for line in out if line.startswith("@@")]
    mismatches = 0
    for payload, line, expected in zip(payloads, lines, node_out, strict=True):
        try:
            got = json_stringify({"ok": to_plain(decode_export_trace_service_request(payload))})
        except ValueError as e:
            got = json_stringify({"error": str(e)})
        if canon(got) != canon(expected):
            mismatches += 1
            if mismatches <= 5:
                print("MISMATCH", line)
                print("  node:", expected[:400])
                print("  py:  ", got[:400])
    print(f"{len(payloads)} payloads, {mismatches} mismatches")
    if WRITE:
        with open(WRITE, "w", encoding="utf-8") as f:
            for line, expected in zip(lines, node_out, strict=True):
                f.write(json.dumps({"payload": line, "node": json.loads(expected)}) + "\n")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
