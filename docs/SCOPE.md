# Scope inventory

promptphoo ports the eval core of promptfoo 0.125.1 (D1). This page compares that scope with
the code, item by item, against upstream's own lists:

- the assertion handlers in `src/assertions/index.ts` and the special types in `src/types/index.ts`;
- the options of `src/commands/eval.ts`, plus the `--env-file` and `--verbose` options that
  `src/main.ts` adds to every command;
- the commands registered in `src/main.ts`;
- the provider factories in `src/providers/registry.ts` and `src/providers/families/`.

Each item is **implemented**, **out of scope** (with the decision that rules it out), or
**missing**. An out-of-scope or missing provider fails with a clear error when it is loaded;
nothing silently falls back to another provider, except where noted under Known gaps.

| Table | Implemented | Out of scope | Missing |
|---|---:|---:|---:|
| Commands | 16 | 10 | 0 |
| Eval flags | 39 | 2 | 0 |
| Assertion types | 68 | 4 | 0 |
| Provider families and routes | 51 | 10 | 25 |
| **Total** | **174** | **26** | **25** |

## Commands

| Command | Status | Note |
|---|---|---|
| `eval` | implemented | All flags below |
| `init` | implemented | `--no-interactive` writes Node's files byte for byte (D45) |
| `validate`, `validate config`, `validate target` | implemented | Local-only target tests (D45) |
| `list evals` / `prompts` / `datasets` | implemented | D40, D41 |
| `show eval` / `prompt` / `dataset` | implemented | D40, D41 |
| `delete eval` | implemented | D40, D41 |
| `export eval`, `export logs` | implemented | D42, D49, D53 |
| `import` | implemented | D42, D49 |
| `retry` | implemented | D36 |
| `cache clear` | implemented | |
| `config get` / `set` / `unset email` | implemented | D45 |
| `logs`, `logs list` | implemented | D45 |
| `debug` | implemented | Reports `pythonVersion` instead of `nodeVersion` (D45) |
| `feedback` | implemented | Sends nothing; points to the issue tracker (D61) |
| `generate dataset` | implemented | D47 |
| `generate assertions` | implemented | D47 |
| `share` | out of scope | Registered; always fails, results stay local (D44) |
| `auth` | out of scope | Cloud account login (D44) |
| `view` | out of scope | Web UI (D1) |
| `eval setup` | out of scope | Web UI (D1) |
| `mcp` | out of scope | D1 |
| `code-scans` | out of scope | D1 |
| `model-scan` | out of scope | D1 |
| `optimize` | out of scope | D1 |
| `redteam` and its subcommands | out of scope | D1 |
| `generate redteam` | out of scope | D1 |

## Eval flags

Every option of upstream's `eval` is accepted.

| Flag | Status | Note |
|---|---|---|
| `-c/--config`, `-a/--assertions`, `-p/--prompts`, `-r/--providers`, `-t/--tests`, `-v/--vars` | implemented | 6 flags |
| `--model-outputs`, `--prompt-prefix`, `--prompt-suffix` | implemented | 3 flags |
| `--var`, `--tag` | implemented | `--tag` merges `commandLineOptions.tags` (D46) |
| `-j/--max-concurrency`, `--repeat`, `--delay`, `--no-cache`, `--safe-mode` | implemented | 5 flags |
| `-n/--filter-first-n`, `--filter-pattern`, `--filter-range`, `--filter-prompts`, `--filter-providers/--filter-targets`, `--filter-sample`, `--filter-sample-seed`, `--filter-metadata` | implemented | 8 flags |
| `--filter-failing`, `--filter-failing-only`, `--filter-errors-only` | implemented | Results file or eval id (D37) |
| `-o/--output`, `--table/--no-table`, `--table-cell-max-length`, `--description`, `--no-write` | implemented | 5 flags |
| `--resume`, `--retry-errors` | implemented | D36 |
| `--grader`, `--suggest-prompts` | implemented | D34, D46 |
| `-w/--watch` | implemented | Polls file stamps (D39) |
| `-x/--extension` | implemented | Python hooks only (D38) |
| `--no-progress-bar` | implemented | Accepted. promptphoo draws no progress bar (D36) |
| `--env-file/--env-path`, `--verbose` | implemented | Common options from `main.ts` |
| `--share/--no-share` | out of scope | `--share` warns, and the eval still runs (D44) |
| `--remote` | out of scope | Red-team remote generation; accepted and ignored (D1) |

## Assertion types

Upstream has 68 handler keys, the special types `select-best`, `max-score` and `human`, and
`assert-set`: 72 types in all. Every implemented type also accepts the `not-` prefix where
upstream does.

| Types | Status | Note |
|---|---|---|
| `equals`, `contains`, `icontains`, `contains-all`, `contains-any`, `icontains-all`, `icontains-any`, `starts-with`, `regex`, `word-count` | implemented | 10 |
| `is-json`, `contains-json`, `is-xml`, `contains-xml`, `is-html`, `contains-html`, `is-sql`, `contains-sql` | implemented | 8; `is-sql`/`contains-sql` need the `sql` extra |
| `levenshtein`, `rouge-n`, `rouge-l`, `rouge-s`, `bleu`, `gleu`, `meteor` | implemented | 7; `meteor` needs the `nlp` extra |
| `cost`, `latency`, `finish-reason`, `perplexity`, `perplexity-score`, `is-refusal` | implemented | 6 |
| `is-valid-function-call`, `is-valid-openai-function-call`, `is-valid-openai-tools-call`, `tool-call-f1`, `skill-used` | implemented | 5 |
| `trajectory:tool-used`, `trajectory:tool-sequence`, `trajectory:tool-args-match`, `trajectory:step-count`, `trajectory:goal-success` | implemented | 5 |
| `trace-span-count`, `trace-span-duration`, `trace-error-spans` | implemented | 3 (D54) |
| `python`, `webhook` | implemented | 2 |
| `llm-rubric`, `factuality`, `model-graded-factuality`, `model-graded-closedqa`, `g-eval`, `search-rubric` | implemented | 6 (D25) |
| `answer-relevance`, `context-faithfulness`, `context-recall`, `context-relevance`, `conversation-relevance` | implemented | 5 (D27, D35) |
| `similar`, `similar:cosine`, `similar:dot`, `similar:euclidean`, `classifier`, `moderation` | implemented | 6 (D28) |
| `agent-rubric` | implemented | Needs an explicitly configured agentic grader (D26) |
| `guardrails` | implemented | Reads the provider response's `guardrails` |
| `select-best`, `max-score`, `assert-set` | implemented | 3 (D29, D36) |
| `javascript`, `ruby` | out of scope | Fail with "not supported, use Python" (D3) |
| `pi` | out of scope | Needs Promptfoo's remote Pi scoring API (D25) |
| `human` | out of scope | Set by manual grading in the web viewer (D1) |

## Providers

One row per upstream family or route group, and each row counts once in the summary, except the
first row (4 providers) and the OpenAI-compatible row (37 families).

| Provider ids | Status | Note |
|---|---|---|
| `echo`, `sequence`, `exec:`, `python:` and `file://*.py` | implemented | 4 |
| `http:`, `https:` | implemented | Including `signatureAuth`. Streaming is not ported |
| `openai:` chat, completion, responses, embedding(s), moderation | implemented | Responses rejects `stream` and background polling (D23) |
| `anthropic:` messages, completion | implemented | |
| `azure:`, `azureopenai:` chat, completion, embedding(s), responses | implemented | Entra ID through two credentials (D24) |
| `bedrock:` InvokeModel, converse, messages, embeddings | implemented | Needs the `bedrock` extra. Only part of the legacy InvokeModel handlers |
| `google:` (AI Studio) chat and embedding, and its `palm:` alias | implemented | |
| `vertex:` chat (Gemini, Llama) and embedding | implemented | Needs the `vertex` extra |
| `ollama:` chat, completion, embedding(s) | implemented | |
| `github:` | implemented | Raises upstream's own "GitHub Models was retired" error |
| `webhook:` | implemented | |
| OpenAI-compatible: `abliteration`, `aimlapi`, `alibaba` (with `alicloud`, `aliyun`, `dashscope`), `atlascloud`, `cerebras`, `cloudera`, `cloudflare-ai`, `cloudflare-gateway`, `cometapi`, `databricks`, `deepseek`, `docker`, `envoy`, `fireworks`, `groq`, `helicone`, `hyperbolic`, `jfrog`/`qwak`, `litellm`, `llama`, `llamaapi`, `localai`, `meta`, `minimax`, `mlflow-gateway`, `moonshot`, `novita`, `nscale`, `nvidia`, `openrouter`, `orcarouter`, `perplexity`, `portkey`, `snowflake`, `togetherai`, `truefoundry`, `xai` | implemented | 37 families (D31) |
| `golang:`, `ruby:` | out of scope | D3 |
| `file://*.js` and other JS/TS provider files | out of scope | D3 |
| `package:` | out of scope | Loads an npm package (D3) |
| `vercel:` | out of scope | Needs the Vercel AI SDK (D31) |
| Media and voice routes of `xai`, `nscale`, `hyperbolic`, `cometapi`; `xai:responses:` | out of scope | D31 |
| `azure:` assistant, foundry-agent, image, video, moderation, realtime | out of scope | D24 |
| `openai:` agents, agents-api, codex, codex-sdk, codex-app-server, codex-desktop, codex-security, realtime, live, decisions, chatkit | out of scope | Agent and realtime providers (plan's out-of-scope list, D26). `chatkit` raises upstream's removal error |
| `anthropic:claude-agent-sdk`, `anthropic:claude-code`, `opencode`, `openinterpreter`, `openclaw`, `a2a` | out of scope | Agent runtimes (D26) |
| `mcp` | out of scope | D1 |
| `promptfoo:simulated-user`, `promptfoo:model:`, `promptfoo:redteam:*`, `agentic:memory-poisoning` | out of scope | Red team and cloud (D1) |
| `openai:` image, tts/speech, transcription, video, assistant | missing | Fail with a not-supported error |
| `mistral:`, `ai21:` | missing | Planned. Upstream gives them their own request code; the error points to `openai:chat` with their base URL (D31) |
| `vertex:` Claude models | missing | Return an error pointing to the `anthropic:` provider |
| `google:`/`vertex:` image (Imagen, Gemini image), video (Veo), live, interactions | missing | Fail with a not-supported error |
| `bedrock:` agents, kb, mantle, responses, video, nova-sonic | missing | Fail with a not-supported error |
| `bedrock-agent:` | missing | |
| `sagemaker:` | missing | |
| `huggingface:`, `hf:` | missing | |
| `replicate:` | missing | Also blocks the moderation fallback in D28 |
| `cohere:` | missing | |
| `voyage:` | missing | Default embedding grader branch is skipped (D33) |
| `watsonx:` | missing | |
| `bam:` | missing | |
| `elevenlabs:` | missing | |
| `fal:` | missing | |
| `modelslab:` | missing | |
| `quiverai:` | missing | |
| `f5:` | missing | |
| `typesafe:` | missing | |
| `n8n:` | missing | |
| `slack:` | missing | |
| `transformers:` | missing | |
| `ws:`, `wss:` (WebSocket) | missing | |
| `browser` | missing | |
| `promptfoo:manual-input` | missing | |

## Decided before the alpha

These were open when the inventory was first taken, and have since been decided.

- The TLS default: certificates are verified unless `PROMPTFOO_INSECURE_SSL=true` (D61, which
  supersedes D21).
- Jinja sandboxing: templates and inline expressions render in a `SandboxedEnvironment` (D61).
- Console scripts: only `promptphoo` and `ph` are installed; `promptfoo` and `pf` are not (D62).
- `promptphoo feedback` sends nothing and points to the issue tracker (D61).
- The eval summary names Node's viewer of the shared database and `promptphoo feedback`, and no
  longer mentions promptfoo.app or promptfoo.dev (D62).

## Known gaps

- **Providers:** the 25 missing rows above. The largest are Mistral and AI21, HuggingFace,
  Replicate, Cohere, Voyage, watsonx, SageMaker and Bedrock agents, WebSocket and browser
  targets, and the OpenAI and Google image, audio, video and live routes.
- **Provider env aliases:** upstream's `mergeProviderEnv` drops a lower layer's alias of a
  credential (for example `GEMINI_API_KEY` when a higher layer sets `GOOGLE_API_KEY`).
  promptphoo merges every layer key by key, so a conflicting alias from a lower layer can win.
- **xAI Responses:** `xai:responses:` is not ported, so search-rubric's xAI candidate and the
  xAI default graders are skipped (D28, D33).
- **Test sources:** Excel (`.xlsx`, `.xls`), Google Sheets, Hugging Face datasets, Azure Blob
  and SharePoint fail with a clear error.
- **`derivedMetrics`** are skipped with a warning.
- **Streaming:** Responses `stream: true` and background polling (D23), HTTP streaming, and
  Bedrock ConverseStream are not ported.
- **Progress bar:** none is drawn, and `--no-progress-bar` has nothing to turn off (D36).
- **Resume and tracing leftovers** listed under D36 and D57: extension-hook prompt recovery,
  streaming JSONL during a resume, the repeat-stability summary, and tracing in the
  comparison pass.
- **Default graders:** the Mistral, xAI, Voyage and Azure Content Safety branches are skipped,
  so those credentials fall through to the OpenAI defaults (D33).

## Known upstream bugs reproduced

promptphoo ports upstream's behaviour, bugs included, so a config behaves the same in both. These
were confirmed against the live Anthropic API on 2026-10-31 (see
[`VERIFICATION.md`](VERIFICATION.md)).

- **`model-graded-closedqa` with an Anthropic grader always errors** (`400 at least one message
  is required`): its grading prompt is a single system message. Use `llm-rubric` with an
  Anthropic grader, or an OpenAI grader for closed QA.
- **A grader whose provider id matches a target reuses that target's config**, such as
  `max_tokens`.
- **Not reproduced:** Node throws `RangeError: Maximum call stack size exceeded` when
  `answer-relevance` has an assertion-level `provider: {text, embedding}`; promptphoo runs it.
