// Render the RAG and search-rubric grading prompts the way upstream's renderLlmRubricPrompt
// (src/matchers/rubric.ts) does, and write tests/fixtures/rag_prompts_golden.json.
//
// Usage: NODE_PATH=<dir containing nunjucks@3.2.4 node_modules> node tools/rag_prompts_golden.js
const fs = require('fs');
const path = require('path');
const nunjucks = require('nunjucks');

const root = path.join(__dirname, '..');
const prompts = JSON.parse(
  fs.readFileSync(path.join(root, 'src', 'promptphoo', '_data', 'grading_prompts.json'), 'utf-8'),
);

// getNunjucksEngine(undefined, false, true)
const env = nunjucks.configure({ autoescape: false, throwOnUndefined: false });
env.addFilter('load', (str) => JSON.parse(str));
const renderString = (t, ctx) => env.renderString(t, { env: {}, ...ctx });

// processContextForTemplating with PROMPTFOO_DISABLE_OBJECT_STRINGIFY unset
function processContext(context) {
  return Object.fromEntries(
    Object.entries(context).map(([key, value]) => {
      if (value && typeof value === 'object') {
        if (Array.isArray(value)) {
          return [key, value.map((i) => (i && typeof i === 'object' ? JSON.stringify(i) : i))];
        }
        return [key, JSON.stringify(value)];
      }
      return [key, value];
    }),
  );
}

function renderLlmRubricPrompt(rubricPrompt, context) {
  const processed = processContext(context);
  try {
    const parsed = JSON.parse(rubricPrompt, (_k, v) =>
      typeof v === 'string' ? renderString(v, processed) : v,
    );
    return JSON.stringify(parsed);
  } catch {}
  return renderString(rubricPrompt, processed);
}

const PARIS = 'Paris is the capital of France.';
const TOWER = 'The Eiffel Tower is in Paris.';
const cases = [
  { name: 'answer_relevance', prompt: 'ANSWER_RELEVANCY_GENERATE', vars: { answer: PARIS } },
  {
    name: 'answer_relevance_json_answer',
    prompt: 'ANSWER_RELEVANCY_GENERATE',
    vars: { answer: { city: 'Paris', country: 'France', tags: [1, 2.5] } },
  },
  {
    name: 'context_recall',
    prompt: 'CONTEXT_RECALL',
    vars: { extra: 'kept user var', context: `${PARIS}\n\n${TOWER}`, groundTruth: PARIS },
  },
  {
    name: 'context_relevance',
    prompt: 'CONTEXT_RELEVANCE',
    vars: { context: `${PARIS} ${TOWER}`, query: 'What is the capital of France?' },
  },
  {
    name: 'context_faithfulness_longform',
    prompt: 'CONTEXT_FAITHFULNESS_LONGFORM',
    vars: {
      query: 'What is the capital of France?',
      context: PARIS,
      question: 'What is the capital of France?',
      answer: `Paris is the capital. It has "the" <Eiffel> & Tower's view.`,
    },
  },
  {
    name: 'context_faithfulness_nli',
    prompt: 'CONTEXT_FAITHFULNESS_NLI_STATEMENTS',
    vars: {
      query: 'What is the capital of France?',
      context: `${PARIS}\n\n${TOWER}`,
      statements: ['Paris is the capital.', 'It has the Eiffel Tower.'],
    },
  },
  {
    name: 'search_rubric',
    prompt: 'DEFAULT_WEB_SEARCH_PROMPT',
    vars: {
      city: 'Tokyo',
      output: 'It is "sunny" in Tokyo\nand 25°C',
      rubric: 'States the current weather in {{ city }}',
    },
  },
  {
    name: 'search_rubric_object_output',
    prompt: 'DEFAULT_WEB_SEARCH_PROMPT',
    vars: { output: { ceo: 'Satya Nadella' }, rubric: 'Names the current CEO' },
  },
];

const out = cases.map((c) => ({ ...c, rendered: renderLlmRubricPrompt(prompts[c.prompt], c.vars) }));
const dest = path.join(root, 'tests', 'fixtures', 'rag_prompts_golden.json');
fs.writeFileSync(dest, JSON.stringify(out, null, 2) + '\n');
console.log(`wrote ${out.length} golden renders`);
