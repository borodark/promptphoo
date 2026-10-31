"""Deterministic media provider for the blob-storage interop tests (tests/test_blobs_node.py).

Every mode returns a fixed payload in one of the response shapes src/blobs/extractor.ts scans.
"""

import base64
import struct
import zlib


def _png(width=32, height=32, seed=7):
    rows = []
    x = seed
    for _ in range(height):
        row = bytearray([0])
        for _ in range(width * 3):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            row.append(x >> 23)
        rows.append(bytes(row))

    def chunk(tag, data):
        crc = struct.pack(">I", zlib.crc32(tag + data))
        return struct.pack(">I", len(data)) + tag + data + crc

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"".join(rows), 9)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _wav(n=4000):
    fmt = struct.pack("<IHHIIHH", 16, 1, 1, 8000, 8000, 1, 8)
    data = bytes((i * 37) & 0xFF for i in range(n))
    header = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " + fmt
    return header + b"data" + struct.pack("<I", n) + data


PNG = base64.b64encode(_png()).decode()
WAV = base64.b64encode(_wav()).decode()
SVG = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="4"/></svg>'


def call_api(prompt, options, context):
    mode = (options.get("config") or {}).get("mode", "image")
    if mode == "image":
        return {"output": "data:image/png;base64," + PNG}
    if mode == "images":
        return {
            "output": "see image",
            "images": [{"data": "data:image/png;base64," + PNG, "mimeType": "image/png"}],
            "metadata": {"preview": "data:image/png;base64," + PNG},
        }
    if mode == "audio":
        return {"output": "spoken", "audio": {"data": WAV, "format": "wav", "transcript": "hi"}}
    if mode == "b64json":
        output = '{"data":[{"b64_json":"' + PNG + '"}]}'
        return {"output": output, "isBase64": True, "format": "json"}
    if mode == "svg":
        return {"output": SVG}
    if mode == "small":
        return {"output": "data:image/png;base64,iVBORw0KGgo="}
    return {"output": "plain"}
