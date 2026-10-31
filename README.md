# promptphoo

Python 3 port of the [promptfoo](https://github.com/promptfoo/promptfoo) eval core, tracking upstream
v0.125.1. It aims to be a drop-in replacement: it reads the same `promptfooconfig.yaml`, takes the same
`eval` flags, and shares the `~/.promptfoo` SQLite schema and migrations with the Node version.

Out of scope: web UI, redteam, code-scan, and JS/TS/Ruby hooks (use `file://*.py`).

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/python tools/parity.py --all   # Node vs Python diff on upstream smoke fixtures
```

`tools/parity.py` uses `npx -y promptfoo@0.125.1` as the reference by default; override with
`NODE_PROMPTFOO`. Per-fixture outputs, logs and diffs land in `.parity/`.

`src/promptphoo/_data/` vendors upstream's `drizzle/` migrations, `tableOutput.html` and the Python
worker wrappers unchanged; refresh them when bumping the upstream version.
