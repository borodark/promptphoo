// Regenerate src/promptphoo/_data/init_templates.json from upstream's src/onboarding.ts and
// src/commands/exampleAliases.ts. The template literals are evaluated by Node itself, so escapes
// (\`, \${) come out exactly as `promptfoo init` writes them.
//   node --experimental-strip-types --no-warnings tools/gen_init_templates.mjs /path/to/promptfoo
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const upstream = process.argv[2];
if (!upstream) {
  console.error('usage: gen_init_templates.mjs /path/to/promptfoo');
  process.exit(2);
}
const src = fs.readFileSync(path.join(upstream, 'src/onboarding.ts'), 'utf8');

// Return the raw text of the template literal that starts at src[start] (a backtick).
function literalAt(text, start) {
  let i = start + 1;
  while (i < text.length) {
    const c = text[i];
    if (c === '\\') {
      i += 2;
      continue;
    }
    if (c === '`') {
      return text.slice(start, i + 1);
    }
    i++;
  }
  throw new Error('unterminated template literal');
}

const templates = {};
for (const name of [
  'CONFIG_TEMPLATE',
  'PYTHON_PROVIDER',
  'JAVASCRIPT_PROVIDER',
  'BASH_PROVIDER',
  'WINDOWS_PROVIDER',
  'PYTHON_VAR',
  'JAVASCRIPT_VAR',
]) {
  const marker = `const ${name} = \``;
  const at = src.indexOf(marker);
  if (at < 0) {
    throw new Error(`${name} not found`);
  }
  templates[name] = (0, eval)(literalAt(src, at + marker.length - 1));
}

// getDefaultReadme: evaluate the returned literal with placeholders the port fills in at runtime.
const fnAt = src.indexOf('function getDefaultReadme(');
const retAt = src.indexOf('return `', fnAt);
const readmeLiteral = literalAt(src, retAt + 'return '.length);
templates.README = new Function('useCase', 'promptfooCommand', `return ${readmeLiteral};`)(
  '@@USE_CASE@@',
  (sub) => `@@PROMPTFOO_COMMAND:${sub}@@`,
);

const aliases = await import(
  pathToFileURL(path.join(upstream, 'src/commands/exampleAliases.ts')).href
);
templates.EXAMPLE_ALIASES = aliases.EXAMPLE_ALIASES;
templates.EXAMPLE_REPLACEMENTS = aliases.EXAMPLE_REPLACEMENTS;
templates.REMOVED_EXAMPLES = aliases.REMOVED_EXAMPLES;

const out = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
  'src/promptphoo/_data/init_templates.json',
);
fs.writeFileSync(out, JSON.stringify(templates, null, 1) + '\n');
