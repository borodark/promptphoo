// Render upstream's synthesis prompts (src/testCase/synthesis.ts and
// src/assertions/synthesis.ts) with the real npm dedent, for a set of inputs, and write
// tests/fixtures/synthesis_golden.json. The prompt builders are cut out of the upstream
// sources verbatim and run under --experimental-strip-types; sampleArray is replaced by an
// order-preserving slice (the dataset prompt sorts the sampled tests anyway).
//
// Usage: node --experimental-strip-types --no-warnings tools/synthesis_golden.mjs \
//   /path/to/promptfoo /path/to/node_modules/dedent
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const upstream = path.resolve(process.argv[2] || path.join(root, '..', 'promptfoo'));
const dedentDir = path.resolve(process.argv[3]);

function cutFunction(src, name) {
  const start = src.indexOf(`export function ${name}(`);
  if (start < 0) throw new Error(`missing ${name}`);
  const end = src.indexOf('\n}\n', start);
  return src.slice(start, end + 3);
}

const tc = fs.readFileSync(path.join(upstream, 'src/testCase/synthesis.ts'), 'utf8');
const as = fs.readFileSync(path.join(upstream, 'src/assertions/synthesis.ts'), 'utf8');
const module = [
  `import dedent from '${pathToFileURL(path.join(dedentDir, 'dist', 'dedent.mjs')).href}';`,
  'const sampleArray = (array, n) => array.slice(0, Math.min(n, array.length));',
  cutFunction(tc, 'generatePersonasPrompt'),
  cutFunction(tc, 'testCasesPrompt'),
  cutFunction(as, 'generateNewQuestionsPrompt'),
  cutFunction(as, 'convertQuestionToPythonPrompt'),
].join('\n');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'synthesis-golden-'));
const file = path.join(tmp, 'synthesis.mts');
fs.writeFileSync(file, module);
const m = await import(pathToFileURL(file).href);
fs.rmSync(tmp, { recursive: true });

const one = ['You are a helpful assistant. Answer {{question}} about {{topic}}.'];
const two = [
  'Write a poem about {{topic}}',
  '  indented\n    multi-line prompt with \\n and `ticks` for {{ name | upper }}\n',
];
const tests = [
  { vars: { topic: 'cats', question: 'why?' } },
  { description: 'no vars' },
  { vars: { topic: 'dogs' }, assert: [{ type: 'contains', value: 'dog' }] },
  { vars: {} },
];

const cases = {
  personas: [
    [one, 5],
    [two, 3],
  ],
  testCases: [
    [one, 'A curious student', [], 3, ['question', 'topic'], undefined],
    [one, 'A busy parent', tests, 20, ['question', 'topic'], 'Use short values.'],
    [two, 'A poet\n  who writes', tests.slice(0, 1), 2, ['topic', 'name'], ''],
  ],
  newQuestions: [
    [one, [], 5],
    [two, tests, 0],
    [two, [{ assert: [{ type: 'llm-rubric', value: 'Is it\nkind?' }] }], 3],
  ],
  pythonPrompt: [
    [one, 'Is the output under 50 words?'],
    [two, 'Does it contain \\b or a `code` block?'],
  ],
};
const out = {
  personas: cases.personas.map((args) => ({ args, out: m.generatePersonasPrompt(...args) })),
  testCases: cases.testCases.map((args) => ({ args, out: m.testCasesPrompt(...args) })),
  newQuestions: cases.newQuestions.map((args) => ({
    args,
    out: m.generateNewQuestionsPrompt(...args),
  })),
  pythonPrompt: cases.pythonPrompt.map((args) => ({
    args,
    out: m.convertQuestionToPythonPrompt(...args),
  })),
};
fs.writeFileSync(
  path.join(root, 'tests', 'fixtures', 'synthesis_golden.json'),
  JSON.stringify(out, null, 1) + '\n',
);
