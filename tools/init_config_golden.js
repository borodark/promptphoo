// Render onboarding's CONFIG_TEMPLATE (from src/promptphoo/_data/init_templates.json) with real
// Nunjucks for each `promptfoo init` wizard path and write tests/fixtures/init_config_golden.json.
//
// Usage: NODE_PATH=<dir containing nunjucks node_modules> node tools/init_config_golden.js
const fs = require('fs');
const path = require('path');
const nunjucks = require('nunjucks');

const root = path.join(__dirname, '..');
const { CONFIG_TEMPLATE } = JSON.parse(
  fs.readFileSync(path.join(root, 'src/promptphoo/_data/init_templates.json'), 'utf-8'),
);
const env = nunjucks.configure({ autoescape: false, throwOnUndefined: false });

const tweet = ['Write a tweet about {{topic}}', 'Write a concise, funny tweet about {{topic}}'];
const rag = [
  'Write a customer service response to:\n\n{{inquiry}}\n\nUse these documents:\n\n{{context}}',
];
const agent = ['Fulfill this user helpdesk ticket: {{inquiry}}'];
const openaiAgent = {
  id: 'openai:chat:gpt-6-sol',
  config: {
    reasoning_effort: 'none',
    tools: [{ type: 'function', function: { name: 'get_current_weather' } }],
  },
};
const vertex = [
  { id: 'vertex:gemini-3.8-flash', config: { region: 'global' } },
  'vertex:gemini-2.5-pro',
];
const contexts = [
  { prompts: tweet, providers: ['openai:gpt-6-luna', 'openai:gpt-6-sol'], type: 'compare' },
  { prompts: tweet.slice(0, 1), providers: vertex.concat(['x']), type: 'compare' },
  { prompts: rag, providers: ['file://provider.py'], type: 'rag', language: 'python' },
  { prompts: rag, providers: ['exec:provider.sh'], type: 'rag', language: 'javascript' },
  { prompts: rag, providers: ['https://example.com/api/generate'], type: 'rag' },
  { prompts: agent, providers: [openaiAgent], type: 'agent', language: 'not_sure' },
  { prompts: agent, providers: vertex, type: 'agent', language: 'python' },
  { prompts: agent, providers: ['file://provider.js'], type: 'agent', language: 'javascript' },
];
for (const c of contexts) {
  c.language = c.language || 'not_sure';
}
const out = contexts.map((ctx) => ({ ctx, out: env.renderString(CONFIG_TEMPLATE, ctx) }));
fs.writeFileSync(
  path.join(root, 'tests/fixtures/init_config_golden.json'),
  JSON.stringify(out, null, 1) + '\n',
);
