// Render tests/fixtures/templating_cases.json with real Nunjucks (configured the way promptfoo's
// getNunjucksEngine does) and write tests/fixtures/templating_golden.json.
//
// Usage: NODE_PATH=<dir containing nunjucks@3.2.4 node_modules> node tools/nunjucks_golden.js
const fs = require('fs');
const path = require('path');
const nunjucks = require('nunjucks');

const fixtures = path.join(__dirname, '..', 'tests', 'fixtures');
const cases = JSON.parse(fs.readFileSync(path.join(fixtures, 'templating_cases.json'), 'utf-8'));

const env = nunjucks.configure({ autoescape: false, throwOnUndefined: false });
env.addFilter('load', (str) => JSON.parse(str));

const out = cases.map(({ t, c }) => {
  try {
    return { ok: env.renderString(t, c) };
  } catch (e) {
    return { error: String(e.message).split('\n')[0] };
  }
});
fs.writeFileSync(path.join(fixtures, 'templating_golden.json'), JSON.stringify(out, null, 2) + '\n');
console.log(`wrote ${out.length} golden renders`);
