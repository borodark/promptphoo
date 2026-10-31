# Verification report

This page records how promptphoo 0.1.1 was checked against Node promptfoo 0.125.1: what ran, how,
what came out, and what was **not** checked. Everything below was run on 2026-10-31 on Linux
(Python 3.12, one GPU for the local models), at the commits named in each row. Every result can
be rerun with the commands under [How to reproduce](#how-to-reproduce).

A difference from Node that this page or [`DECISIONS.md`](DECISIONS.md) does not explain is a
bug. Please report it.

## Summary

| Area | What was checked | How | Result | Commit |
|---|---|---|---|---|
| Static checks | Lint, formatting, types | `ruff check`, `ruff format --check`, `pyright` over `src`, `tests`, `tools` | Clean | `9d64258` |
| Unit and golden tests | The whole suite, with the Node interop suites on | `pytest -q -n auto` with `NODE_PROMPTFOO` set | 6,543 passed, 161 skipped, 0 failed | `9d64258` |
| Node parity | Upstream's smoke fixtures through both CLIs, output JSON diffed after normalizing ids, timestamps and latency | `tools/parity.py --all` | 63 pass, 24 skip, 4 expected-error, 2 expected-fail (each listed with its reason in `EXPECTED`); 0 unexpected | `9d64258` |
| Doc references | Every file path and `D<n>` cited in docs, docstrings and comments | `tools/check_refs.py --upstream ../promptfoo` | 1,209 path refs and 311 decision refs, 0 broken | `9d64258` |
| Clean install | Built wheel in a fresh venv with every extra, `HOME` pointed at an empty directory | `pip install dist/*.whl[all]`, then the suite | 6,534 passed; nothing written to `HOME`; only `promptphoo` and `ph` on PATH | `8cf5fe0` |
| Base install | The suite without any optional extra | `pip install .[dev]`, then the suite | 6,447 passed, 226 skipped, 0 failed | `8cf5fe0` |
| Local models, run 1 | Two Ollama models under test, a local grader and local embeddings, Node then Python into one database | `examples/local-ollama/` | 8/8 tests pass in both; all 18 assertion scores identical; 1,616 tokens in both | `8cf5fe0` |
| Local models, run 2 | The same, repeated for the retest | as above | 8/8 in both; 17 of 18 scores identical. The odd one out: gpt-oss phrased one answer two ways, scoring 0.865 vs 0.835. Each CLI listed both evals. 184 s Node, 175 s Python | `8cf5fe0` |
| Anthropic wire parity | Every request promptphoo sends to Anthropic matches one Node sent | Node recorded live calls through a proxy; promptphoo replayed them with a dummy key ([phase 1](#phase-1-wire-parity)) | 69 of 69 requests matched, 0 misses; identical `anthropic-version` and beta headers | `9d64258` |
| Anthropic replay results | Given the same responses, every result row matches | `tools/live/compare_replay.py` | features 12 of 12 rows identical; matrix 18 of 20 (the 2 are a replay artefact, below) | `9d64258` |
| Anthropic live | Both CLIs call api.anthropic.com directly on nine configs | [phase 2](#phase-2-live) | Same pass/error outcome on every config except one live-variance row; identical error texts; costs equal to the float | `9d64258` |
| Response cache | A second identical run is served from cache | features config run twice | 4,072 of 4,072 tokens cached on the rerun | `9d64258` |
| Key handling | The API key never lands on disk | `grep -rlF -f <keyfile>` over every output, log, recording and database | Found in no file | `9d64258` |
| Test isolation | A flaky failure in `test_trace_assertions_e2e.py` (about 1 full run in 5) | Reproduced deterministically: a test that leaves the shared DB connection open makes the next eval on that worker write into the wrong database | Fixed in `tests/conftest.py` (D60); 3 consecutive full runs clean | after `3e6099e` |

Total spend for the Anthropic tests was about $1.

## Phase 1: wire parity

`tools/live/anthropic_proxy.py` sits between the CLI and Anthropic. In record mode it forwards
each request and saves the exchange, keyed by path and canonical (sorted-key) JSON body, without
the `x-api-key` or `authorization` headers. In replay mode it answers from those recordings and
saves any request it has never seen as a miss.

Node ran the matrix (`tools/live/configs/matrix/`: Haiku 4.5 and Sonnet 5, a plain and a chat
prompt, `icontains`, `llm-rubric`, `similar`, `cost`, `latency`, `factuality`,
`model-graded-closedqa`, `g-eval`, a transform, `select-best`, `word-count`, `not-is-refusal`,
`answer-relevance`) and the features config (tool use with `tool-call-f1`, JSON-schema
structured output, extended thinking, `finish-reason`) live through the recorder. promptphoo then
ran both against the recordings with a dummy key.

- **Requests:** promptphoo sent 69 requests and every one matched a Node recording. Headers
  matched too, once promptphoo stopped sending `x-promptfoo-version` to Anthropic (D64).
- **Results:** the first replay matched 13 of 20 matrix rows. Two port bugs explained part of it
  and were fixed in `9d64258`: Python 3.12's `sum()` uses compensated summation, so cosine
  similarity differed from Node's in the last bit (D63), and the extra header. After the fixes,
  features matches 12 of 12 rows and the matrix 18 of 20.
- **The last two rows** are a harness artefact, not a port difference. Both models answered
  "Bubbles." to the same prompt, so `answer-relevance` sent byte-identical grader requests for
  the two rows, and the live API gave different answers each time. The proxy serves duplicate
  recordings in recorded order, and promptphoo issues its grading requests in a different order
  from Node under `-j 1`, so the two rows received each other's grader answers. Pass/fail and
  token totals still match exactly.

## Phase 2: live

Both CLIs called the API directly, with caching off. Model outputs vary between live calls, so
`tools/live/compare_live.py` compares outcomes and structure, not text.

| Config | What it covers | Pass (Node/Py) | Fail | Errors | Notes |
|---|---|---|---|---|---|
| matrix | as in phase 1 | 7 / 8 | 13 / 12 | 0 / 0 | The one differing row is live variance: Sonnet 5 answered the goldfish question in slightly different words, and `answer-relevance` scored 0.36 (Node) vs 0.51 (Python) against a 0.5 threshold. Target cost $0.0049 / $0.0050 |
| features | tools, structured output, thinking | 4 / 4 | 8 / 8 | 0 / 0 | Target cost $0.0218 / $0.0240 |
| sonnet-5 | upstream `examples/anthropic/sonnet-5` | 2 / 2 | | | 6,258 / 6,303 tokens; `temperature` omitted for Sonnet 5 in both |
| sonnet-5-5 | upstream `examples/anthropic/sonnet-5-5` | 4 / 4 | | | |
| structured | upstream `examples/anthropic/structured-outputs` | 4 / 4 | | | |
| thinking | upstream `examples/claude-thinking` (Anthropic providers) | 8 / 8 | | | The config asks for `stream: true`; promptphoo sends non-streaming and gets the same outcome |
| unknown-model | `claude-haiku-5-5`, missing from 0.125.1's price table | 1 / 1 | | | Cost 0 in both |
| bad-model | a model id that does not exist, next to a real one | 1 / 1 | | 1 / 1 | Same `404 not_found_error` text |
| bad-key | an invalid key | | | 2 / 2 | Same `401 authentication_error` text |

Where costs are comparable row by row they are equal to the last digit (for example
`8.900000000000001e-05` in both). Target cost over all configs: $0.284 Node, $0.278 Python, the
difference coming from different live output lengths.

## Upstream bugs reproduced faithfully

promptphoo ports upstream's behaviour, including these bugs found during the live runs. They are
listed in [`SCOPE.md`](SCOPE.md#known-upstream-bugs-reproduced).

- **`model-graded-closedqa` always errors with Anthropic graders.** Its prompt is a single system
  message, and Anthropic rejects a request with no user message (`400 at least one message is
  required`). Both runtimes fail it identically.
- **A grader inherits a target's config when the ids match.** With `claude-sonnet-5` as both
  target and grader, the grader runs with the target's `max_tokens: 200`.
- **Node crashes where promptphoo does not:** an assertion-level `provider: {text, embedding}`
  on `answer-relevance` makes Node throw `RangeError: Maximum call stack size exceeded` while
  cloning the test. promptphoo runs it. The live configs set that provider on `defaultTest`, which
  works in both.

## Bugs found and fixed by this verification

| Bug | Found by | Fix |
|---|---|---|
| A `?` inside a flow mapping (`{ q: What is it? }`) failed to parse; js-yaml reads it as one string | Validating the live configs before spending | `3c13f67` |
| `x-promptfoo-version` was sent to Anthropic; upstream's Anthropic SDK path never sends it | Phase 1 header comparison | `9d64258` (D64) |
| Float sums differed from JS in the last bit (cosine similarity, averages, perplexity, latency, JUnit totals) | Phase 1 row comparison | `9d64258` (D63) |

## Not verified

- Python 3.11 and 3.13 (declared supported; only 3.12 was run). macOS and Windows.
- Live OpenAI, Azure OpenAI, Google and Bedrock calls. These providers are covered by mocked HTTP
  tests (respx) and Node goldens, not by a live run.
- Remote test sources (URLs, `https://` CSVs) and heavy concurrency against rate limits.

## How to reproduce

```bash
# Static checks and the suite (the interop suites need NODE_PROMPTFOO)
export NODE_PROMPTFOO="node /path/to/promptfoo/dist/src/main.js"
.venv/bin/pytest -q -n auto && .venv/bin/ruff check src tests tools \
  && .venv/bin/ruff format --check src tests tools && .venv/bin/pyright
.venv/bin/python tools/parity.py --all
.venv/bin/python tools/check_refs.py --upstream ../promptfoo

# Local models (needs Ollama with the models the config lists)
promptphoo eval -c examples/local-ollama/promptfooconfig.yaml

# Live Anthropic, both phases (spends about $1; needs Ollama with nomic-embed-text)
ANTHROPIC_KEY_FILE=~/path/to/key tools/live/run_anthropic.sh      # writes .live/
```

`run_anthropic.sh` reads the key from the file only inside the commands that need it, never
echoes it, and ends with the leak scan.
