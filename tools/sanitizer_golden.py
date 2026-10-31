"""Regenerate tests/fixtures/sanitizer_golden.json from upstream sanitizer.ts under real Node.

Usage: python tools/sanitizer_golden.py [../promptfoo]
Needs Node >= 22.6 (runs the TypeScript source with --experimental-strip-types; the two npm
imports are replaced by small shims, so no `npm ci` is required).
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STUBS = r"""
export function safeStringify(obj: any, replacer: any): string {
  const stack: any[] = [];
  try {
    return JSON.stringify(obj, function (this: any, key, val) {
      val = replacer.call(this, key, val);
      if (typeof val === 'object' && val !== null) {
        while (stack.length > 0 && stack[stack.length - 1] !== this) stack.pop();
        if (stack.includes(val)) return '[Circular]';
        stack.push(val);
      }
      return val;
    });
  } catch {
    return JSON.stringify('[unable to serialize, circular reference is too complex to analyze]');
  }
}
const isApiProvider = (p: any) =>
  typeof p === 'object' && p != null && typeof p.id === 'function' &&
  typeof p.callApi === 'function';
export const GRADING_PROVIDER_TYPE_KEYS =
  ['text', 'embedding', 'classification', 'moderation'] as const;
function isTypedProviderValue(value: unknown): boolean {
  if (typeof value === 'string') return value.length > 0;
  if (isApiProvider(value)) return true;
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const id = (value as any).id;
  return typeof id === 'string' && id.length > 0;
}
export function isProviderTypeMap(provider: any): boolean {
  return Boolean(provider && typeof provider === 'object' && !Array.isArray(provider) &&
    !isApiProvider(provider) && !Object.prototype.hasOwnProperty.call(provider, 'id') &&
    GRADING_PROVIDER_TYPE_KEYS.some((t) => Object.prototype.hasOwnProperty.call(provider, t) &&
      isTypedProviderValue(provider[t])));
}
"""

RUNNER = r"""
import * as s from './sanitizer.ts';
import { readFileSync } from 'node:fs';
const cases = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const out = cases.map(([fn, args]: [string, any[]]) => {
  try {
    if (args[1] && args[1].maxDepth === 'inf') args[1].maxDepth = Infinity;
    return { ok: (s as any)[fn](...args) };
  } catch (e: any) {
    return { err: String(e) };
  }
});
console.log(JSON.stringify(out));
"""

FUZZ_PARTS = (
    ["http://", "https://", "HTTPS://", "ws://", "ftp://", "foo://", "file://", "az://", "", "/",
     "//", "http:", "x:"],
    ["", "u@", "u:p@", ":p@", "a b:c%20d@", "@"],
    ["example.com", "EXAMPLE.COM", "127.0.0.1", "0x7f.0.0.1", "1.2.3", "[::1]", "[2001:db8::0:1]",
     "h:443", "h:80", "h:99999", "h:abc", "exa mple.com", "xn--caf-dma.com", "café.com",
     "h%41.com", "", "a..b", "192.168.0.256"],
    ["", "/", "/a/b", "/a/../b", "/a/./b/", "/%2e%2e/x", '/a b/c"d', "/x/{y}/`z`", "\\a\\b",
     "/v1/sk-abcdefghijklmnopqrstuvwxyz", "/token/abcdef123456789", "/café", "/%zz",
     "/AKIAABCDEFGHIJKLMNOP"],
    ["", "?", "?a=1", "?token=abc", "?api_key=sk-abcdefghijklmnopqrstuv", "?a=b c&d='e'",
     "?x=1;password=2", "?q=%E2%9C%93&sig=xyz", "?a&b=&c=1", "?includeCredentials=true",
     "?pageToken=1&max_tokens=3", "?data=%7B%22apiKey%22%3A%22v%22%7D", "?a=%zz&key=1",
     "?subscription-key=abc", "?a=1+2&token=a+b", "?{{x}}=1"],
    ["", "#", "#top", "#access_token=abc&x=1", "#a b", "#id_token=eyJhbGciOi.eyJzdWIi.sig"],
)  # fmt: skip


def build_cases(upstream: Path) -> list[list]:
    src = (upstream / "test/util/sanitizer.test.ts").read_text()
    lits = {m.group(1) for m in re.finditer(r"'((?:[^'\\\n]|\\.)*)'", src)}
    lits |= {m.group(1) for m in re.finditer(r"`([^`\\$\n]*)`", src)}
    lits = sorted(s for s in lits if "\\" not in s and "${" not in s)
    cases: list[list] = []
    for s in lits:
        cases += [
            ["sanitizeUrl", [s]],
            ["sanitizeUrlForLogging", [s]],
            ["sanitizeUrlEncodedString", [s]],
            ["looksLikeSecret", [s]],
            ["isSecretEnvVarName", [s]],
            ["isCredentialHeader", [s, s]],
        ]
        if len(s) <= 120 and re.search(r"[:/=?#]", s):
            cases += [
                ["sanitizeObject", [{"url": s, "apiHost": s, "headers": {s: s}}]],
                ["sanitizeObject", [{"x": {s: "v"}, "endpoint": s}, {"sanitizeUrls": True}]],
                ["sanitizeObject", [{"env": {s: "v1", "A_HOST": s, "B_URL": s}}]],
            ]
    rng = random.Random(7)
    for _ in range(600):
        u = "".join(rng.choice(part) for part in FUZZ_PARTS)
        cases += [
            ["sanitizeUrl", [u]],
            ["sanitizeUrlForLogging", [u]],
            ["sanitizeObject", [{"apiHost": u, "endpoint": u, u: 1}, {"sanitizeUrls": True}]],
        ]
    return cases


def main() -> None:
    upstream = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT.parent / "promptfoo").resolve()
    source = (upstream / "src/util/sanitizer.ts").read_text()
    source = source.replace(
        "import deepEqual from 'fast-deep-equal';",
        "const deepEqual = (a: any, b: any) => JSON.stringify(a) === JSON.stringify(b);",
    )
    source = source.replace(
        "import safeStringify from 'fast-safe-stringify';",
        "import { safeStringify } from './stubs.ts';",
    )
    source = source.replace("from './gradingProvider';", "from './stubs.ts';")
    source = re.sub(r"^import type .*$", "", source, flags=re.M)
    cases = build_cases(upstream)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "sanitizer.ts").write_text(source)
        (d / "stubs.ts").write_text(STUBS)
        (d / "run.ts").write_text(RUNNER)
        (d / "cases.json").write_text(json.dumps(cases))
        out = subprocess.run(
            ["node", "--experimental-strip-types", "--no-warnings", "run.ts", "cases.json"],
            cwd=d,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    golden = json.loads(out)
    rows = [[fn, args, res] for (fn, args), res in zip(cases, golden, strict=True)]
    target = ROOT / "tests/fixtures/sanitizer_golden.json"
    target.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} cases to {target}")


if __name__ == "__main__":
    main()
