#!/usr/bin/env node
// Regenerates src/promptphoo/providers/openai/tables.py from upstream promptfoo's model and
// pricing tables (src/providers/openai/{util,billing,gpt6}.ts). The values are evaluated by Node
// itself, so float arithmetic such as `2.5 / 1e6` and `perMillion(2.5)` is bit-for-bit upstream's.
//
// Usage: node tools/gen_openai_tables.mjs [../promptfoo]   (Node >= 22.6 for --experimental-strip-types)
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const upstream = resolve(process.argv[2] ?? join(here, '..', '..', 'promptfoo'));
const src = (name) => readFileSync(join(upstream, 'src/providers/openai', name), 'utf8');
const stripImports = (text) => text.replace(/^import[\s\S]*?;\n/gm, '');

const stubs = `
const OpenAI: any = { APIError: class {} };
class InvalidToolCallError extends Error {}
class InvalidToolSchemaError extends Error {}
const logger: any = console;
const getAjv: any = () => undefined;
const safeJsonStringify: any = JSON.stringify;
const isSafeCost: any = () => true;
const isSafeTokenCount: any = () => true;
const isNonCredentialHeader: any = () => false;
const looksLikeSecret: any = () => false;
const sanitizeUrl: any = (u: string) => u;
const calculateCost: any = () => undefined;
const prepareOpenAiFunctions: any = () => undefined;
`;

const exported = {
  util: [
    'GPT_LONG_CONTEXT_THRESHOLD',
    'OPENAI_DAYBREAK_ALIASES',
    'RETIRED_OPENAI_MODEL_IDS',
    'LEGACY_OPENAI_DISCOVERY_EXCLUSIONS',
    'OPENAI_TTS_AND_RETIRED_MODELS',
    'OPENAI_CHAT_AND_RETIRED_MODELS',
    'LEGACY_OPENAI_AUDIO_MODELS',
    'OPENAI_CODEX_ONLY_MODELS',
    'OPENAI_FIRST_PARTY_API_HOSTNAMES',
    'OPENAI_RESPONSES_ONLY_AND_RETIRED_MODELS',
    'OPENAI_DEEP_RESEARCH_AND_RETIRED_MODELS',
    'OPENAI_COMPLETION_MODELS',
    'NON_CONVERSATIONAL_REALTIME_MODELS',
    'OPENAI_REALTIME_AND_RETIRED_MODELS',
    'RETIRED_OPENAI_REALTIME_MODELS',
    'OPENAI_TRANSCRIPTION_MODELS',
  ],
  billing: [
    'STANDARD_CACHED_INPUT_RATES',
    'FINE_TUNED_TEXT_RATES',
    'FINE_TUNED_BATCH_OVERRIDES',
    'LONG_CONTEXT_CACHED_INPUT_RATES',
    'FLEX_SUPPORTED_TEXT_MODELS',
    'FAST_TEXT_RATES',
    'IMAGE_MODEL_RATES',
    'REALTIME_MODAL_RATES',
    'EMBEDDING_RATES',
    'GPT_5_6_MODELS',
    'BEDROCK_MANTLE_TEXT_RATES',
    'CACHE_WRITE_MODELS',
    'BEDROCK_GOVCLOUD_MODELS',
  ],
};

// The billing module's helpers would collide with util's, so each table module is its own file.
const dir = mkdtempSync(join(tmpdir(), 'openai-tables-'));
writeFileSync(join(dir, 'gpt6.ts'), stripImports(src('gpt6.ts')));
writeFileSync(
  join(dir, 'util.ts'),
  `${stubs}${stripImports(src('util.ts'))}\nexport { ${exported.util.join(', ')} };\n`.replace(
    /export const (GPT_LONG_CONTEXT_THRESHOLD|OPENAI_DAYBREAK_ALIASES|RETIRED_OPENAI_MODEL_IDS|OPENAI_CODEX_ONLY_MODELS|OPENAI_COMPLETION_MODELS|NON_CONVERSATIONAL_REALTIME_MODELS|OPENAI_TRANSCRIPTION_MODELS)\b/g,
    'const $1',
  ),
);
writeFileSync(
  join(dir, 'billing.ts'),
  `import { getGpt6Variant, isGpt6Model } from './gpt6.ts';
import { GPT_LONG_CONTEXT_THRESHOLD, OPENAI_DAYBREAK_ALIASES } from './util.ts';
const OPENAI_BILLING_MODELS: any[] = [];
const getOpenAICacheWriteInputTokens: any = () => undefined;
const isAzureOpenAiEndpoint: any = () => false;
${stripImports(src('billing.ts'))}
export { ${exported.billing.join(', ')} };
`,
);
writeFileSync(
  join(dir, 'dump.ts'),
  `import * as util from './util.ts';
import * as billing from './billing.ts';
const plain = (v: unknown): unknown =>
  v instanceof Map ? Object.fromEntries(v) : v instanceof Set ? [...v] : v;
const out: Record<string, unknown> = {};
for (const name of ${JSON.stringify(exported.util)}) out[name] = plain((util as any)[name]);
for (const name of ${JSON.stringify(exported.billing)}) out[name] = plain((billing as any)[name]);
process.stdout.write(JSON.stringify(out));
`,
);
const json = execFileSync(
  process.execPath,
  ['--experimental-strip-types', '--no-warnings', join(dir, 'dump.ts')],
  { encoding: 'utf8' },
);
const tables = JSON.parse(json);

// OpenAiResponsesProvider.OPENAI_RESPONSES_MODEL_NAMES is a literal array on the class.
{
  const text = src('responses.ts');
  const start = text.indexOf('[', text.indexOf('static OPENAI_RESPONSES_MODEL_NAMES = ['));
  let depth = 0;
  let end = start;
  for (; end < text.length; end++) {
    if (text[end] === '[') depth++;
    if (text[end] === ']' && --depth === 0) break;
  }
  tables.OPENAI_RESPONSES_MODEL_NAMES = new Function(`return ${text.slice(start, end + 1)}`)();
}

// Python literals: JSON numbers already print as the shortest round-trip form.
const py = (value, indent = '') => {
  if (value === null || value === undefined) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  if (typeof value === 'number') {
    const s = String(value);
    return Number.isInteger(value) ? s : /e/.test(s) && !/\./.test(s) ? s.replace('e', '.0e') : s;
  }
  if (typeof value === 'string') return JSON.stringify(value);
  if (Array.isArray(value)) {
    if (value.every((v) => typeof v !== 'object' || v === null)) {
      return `[${value.map((v) => py(v)).join(', ')}]`;
    }
    const inner = indent + '    ';
    return `[\n${value.map((v) => `${inner}${py(v, inner)},`).join('\n')}\n${indent}]`;
  }
  const entries = Object.entries(value);
  if (entries.every(([, v]) => typeof v !== 'object' || v === null)) {
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}: ${py(v)}`).join(', ')}}`;
  }
  const inner = indent + '    ';
  return `{\n${entries.map(([k, v]) => `${inner}${JSON.stringify(k)}: ${py(v, inner)},`).join('\n')}\n${indent}}`;
};
const sets = new Set([
  'RETIRED_OPENAI_MODEL_IDS',
  'LEGACY_OPENAI_DISCOVERY_EXCLUSIONS',
  'OPENAI_FIRST_PARTY_API_HOSTNAMES',
  'NON_CONVERSATIONAL_REALTIME_MODELS',
  'FLEX_SUPPORTED_TEXT_MODELS',
  'GPT_5_6_MODELS',
  'CACHE_WRITE_MODELS',
  'BEDROCK_GOVCLOUD_MODELS',
]);
const body = Object.entries(tables)
  .map(([name, value]) => {
    const literal = py(value);
    return sets.has(name) ? `${name} = frozenset(${literal})` : `${name} = ${literal}`;
  })
  .join('\n\n');
const header = `"""OpenAI model and pricing tables. Generated by tools/gen_openai_tables.mjs from upstream
src/providers/openai/{util,billing,responses}.ts (v0.125.1). Do not edit by hand; regenerate.
"""

# ruff: noqa: E501
# fmt: off
from typing import Any

`;
const target = join(here, '..', 'src/promptphoo/providers/openai/tables.py');
writeFileSync(
  target,
  `${header}${body.replace(/^(\w+) = /gm, '$1: Any = ')}\n`,
);
console.log(`wrote ${target}`);
