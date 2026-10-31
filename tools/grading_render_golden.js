// Render tests/fixtures/grading_render_cases.json with real Nunjucks through a copy of upstream's
// renderLlmRubricPrompt (src/matchers/rubric.ts) and write tests/fixtures/grading_render_golden.json.
//
// Usage: NODE_PATH=<dir containing nunjucks@3.2.4 node_modules> node tools/grading_render_golden.js
const fs = require('fs');
const path = require('path');
const nunjucks = require('nunjucks');

const fixtures = path.join(__dirname, '..', 'tests', 'fixtures');
const cases = JSON.parse(
  fs.readFileSync(path.join(fixtures, 'grading_render_cases.json'), 'utf-8'),
);

const env = nunjucks.configure({ autoescape: false, throwOnUndefined: false });
env.addFilter('load', (str) => JSON.parse(str));

function processContextForTemplating(context) {
  return Object.fromEntries(
    Object.entries(context).map(([key, value]) => {
      if (value && typeof value === 'object') {
        if (Array.isArray(value)) {
          return [
            key,
            value.map((item) => (item && typeof item === 'object' ? JSON.stringify(item) : item)),
          ];
        }
        return [key, JSON.stringify(value)];
      }
      return [key, value];
    }),
  );
}

function renderLlmRubricPrompt(rubricPrompt, context) {
  const processedContext = processContextForTemplating(context);
  try {
    const parsed = JSON.parse(rubricPrompt, (_k, v) =>
      typeof v === 'string' ? env.renderString(v, processedContext) : v,
    );
    return JSON.stringify(parsed);
  } catch {}
  return env.renderString(rubricPrompt, processedContext);
}

const out = cases.map(({ name, template, context }) => ({
  name,
  rendered: renderLlmRubricPrompt(template, context),
}));
fs.writeFileSync(
  path.join(fixtures, 'grading_render_golden.json'),
  JSON.stringify(out, null, 2) + '\n',
);
console.log(`wrote ${out.length} golden grading prompts`);
