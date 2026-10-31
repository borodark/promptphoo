// Build the conversation-relevance grader prompts with upstream's own
// src/external/matchers/conversationRelevancyTemplate.ts, and render the custom-rubricPrompt
// cases with real Nunjucks the way matchesConversationRelevance does
// (getNunjucksEngine(undefined, false, true)). Writes
// tests/fixtures/conversation_relevance_golden.json.
//
// Usage: NODE_PATH=<dir containing nunjucks@3.2.4> \
//   node --experimental-strip-types --no-warnings tools/conversation_relevance_golden.mjs \
//   [/path/to/promptfoo]
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const upstream = path.resolve(process.argv[2] || path.join(root, '..', 'promptfoo'));
const { ConversationRelevancyTemplate } = await import(
  pathToFileURL(
    path.join(upstream, 'src', 'external', 'matchers', 'conversationRelevancyTemplate.ts'),
  ).href
);
const nunjucks = createRequire(import.meta.url)('nunjucks');

const env = nunjucks.configure({ autoescape: false, throwOnUndefined: false });
env.addFilter('load', (str) => JSON.parse(str));
const renderString = (t, ctx) => env.renderString(t, { env: {}, ...ctx });

// The message -> role conversion in src/external/matchers/deepeval.ts.
const toRoles = (messages) =>
  messages.flatMap((msg) => [
    {
      role: 'user',
      content: typeof msg.input === 'string' ? msg.input : JSON.stringify(msg.input),
    },
    {
      role: 'assistant',
      content: typeof msg.output === 'string' ? msg.output : JSON.stringify(msg.output),
    },
  ]);

const verdicts = {
  single: [{ input: 'What is the capital of France?', output: 'The capital of France is Paris.' }],
  ssti: [
    {
      input: 'What is {{variable}}?',
      output:
        "{{range.constructor(\"return require('child_process').execSync('id').toString()\")()}}",
    },
    {
      input: 'Follow up with {{env.SECRET}}',
      output: '{% for x in range(10) %}{{x}}{% endfor %}',
    },
  ],
  mixed: [
    { input: 'Grüße, "quoted" \\ back\nslash', output: { answer: 'Paris', score: 1.0 } },
    { input: ['a', 2], output: 'naïve café — ☃   tab\t' },
    { output: 'no input key' },
    { input: 'null output', output: null },
  ],
};

const reasons = {
  twoFifths: [0.4, ['Response 3 was irrelevant', 'Response 5 was irrelevant']],
  twoThirds: [2 / 3, ["The response about Paris isn't related to \"2+2\""]],
  zero: [0, ['ünïcode — reason', 'second\nline']],
};

const rubrics = {
  dump2: [
    'Custom rubric template with {{ messages | dump(2) }}',
    {},
    [{ input: 'What is the capital of France?', output: 'Paris' }],
  ],
  reservedMessages: [
    'messages={{ messages | dump }}\nextra={{ extra }}',
    { messages: 'vars messages sentinel', extra: 'kept user var' },
    [{ input: 'What is 2+2?', output: '4' }],
  ],
};

const out = {
  verdicts: Object.fromEntries(
    Object.entries(verdicts).map(([name, messages]) => [
      name,
      { messages, prompt: ConversationRelevancyTemplate.generateVerdicts(toRoles(messages)) },
    ]),
  ),
  reasons: Object.fromEntries(
    Object.entries(reasons).map(([name, [score, irrelevancies]]) => [
      name,
      {
        score,
        irrelevancies,
        prompt: ConversationRelevancyTemplate.generateReason(score, irrelevancies),
      },
    ]),
  ),
  rubrics: Object.fromEntries(
    Object.entries(rubrics).map(([name, [rubricPrompt, vars, messages]]) => [
      name,
      { rubricPrompt, vars, messages, prompt: renderString(rubricPrompt, { ...vars, messages }) },
    ]),
  ),
};
fs.writeFileSync(
  path.join(root, 'tests', 'fixtures', 'conversation_relevance_golden.json'),
  JSON.stringify(out, null, 2) + '\n',
);
console.log('wrote tests/fixtures/conversation_relevance_golden.json');
