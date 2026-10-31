// Golden output for promptphoo.output: js-yaml dump, fast-xml-parser XMLBuilder, csv-stringify
// and the cli-table3 console table (src/table.ts), rendered by the real Node libraries.
//
// Usage (needs js-yaml@5.5.0 cli-table3@0.6.5 fast-xml-parser@5.7.1 csv-stringify@6.9.0
// chalk@6 under NODE_PATH; Node >= 22.12 for require(esm)):
//   FORCE_COLOR=0 NODE_PATH=<dir>/node_modules node tools/output_golden.js plain
//   FORCE_COLOR=1 NODE_PATH=<dir>/node_modules node tools/output_golden.js color
// writes tests/fixtures/output_golden_<mode>.json.
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const Table = require('cli-table3');
const { XMLBuilder } = require('fast-xml-parser');
const { stringify } = require('csv-stringify/sync');
const chalk = require('chalk').default;

const mode = process.argv[2] || 'plain';
const TERMINAL_MAX_WIDTH = 120;

function ellipsize(str, maxLen) {
  if (str.length > maxLen) {
    if (maxLen < 3) return str.slice(0, Math.max(0, maxLen));
    return str.slice(0, maxLen - 3) + '...';
  }
  return str;
}

// src/table.ts generateTable, verbatim apart from the import of TERMINAL_MAX_WIDTH.
function generateTable(evaluateTable, tableCellMaxLength = 250, maxRows = 25) {
  const head = evaluateTable.head;
  const headLength = head.prompts.length + head.vars.length;
  const table = new Table({
    head: [...head.vars, ...head.prompts.map((p) => `[${p.provider}] ${p.label}`)].map((h) =>
      ellipsize(h, tableCellMaxLength),
    ),
    colWidths: Array(headLength).fill(Math.floor(TERMINAL_MAX_WIDTH / headLength)),
    wordWrap: true,
    wrapOnWordBoundary: true,
    style: { head: ['blue', 'bold'] },
  });
  for (const row of evaluateTable.body.slice(0, maxRows)) {
    table.push([
      ...row.vars.map((v) => ellipsize(v, tableCellMaxLength)),
      ...row.outputs.map(({ pass, text, failureReason }) => {
        text = ellipsize(text, tableCellMaxLength);
        if (pass) return chalk.green('[PASS] ') + text;
        return (
          chalk.red(failureReason === 1 ? '[FAIL] ' : '[ERROR] ') +
          text
            .split('---')
            .map((c, idx) => (idx === 0 ? chalk.red.bold(c) : c))
            .join('---')
        );
      }),
    ]);
  }
  return table.toString();
}

// src/table.ts wrapTable, verbatim.
function wrapTable(rows, columnWidths) {
  if (rows.length === 0) return 'No data to display';
  const head = Object.keys(rows[0]);
  const defaultWidth = Math.floor(TERMINAL_MAX_WIDTH / head.length);
  const colWidths = head.map((column) => columnWidths?.[column] || defaultWidth);
  const table = new Table({ head, colWidths, wordWrap: true, wrapOnWordBoundary: true });
  for (const row of rows) table.push(Object.values(row));
  return table.toString();
}

const shared = { a: 1 };
const yamlCases = [
  { a: 1, b: 'two', c: true, d: null, e: 1.5, f: [1, 2, { g: 'h' }], i: {}, j: [] },
  {
    s: ['yes', 'no', 'on', 'off', 'y', 'N', '~', 'null', '', ' ', '123', '0x1F', '1_000', '1:20',
      '1.5', '.inf', '2024-01-01', '2024-01-01T10:00:00Z', '<<', '=', '-', '- a', ': x', 'a: b',
      'a #b', '#c', "it's", '"q"', '@x', '`b`', '%p', '!t', '&a', '*a', '|', '>', '[x]', '{y}',
      'a,b', 'tab\there', 'nbsp\u00a0x', 'line\nbreak', 'trailing\n', 'two\n\n', '\n',
      ' leading', 'trailing ', '---', '--- x', '...', 'ünïcödé', '日本語', 'emoji 😀',
      'ctrl\u0001x', 'del\u007f', 'bom\ufeff', 'true', 'False', 'NULL', '+1', '-1', '1e3',
      '1.0e+3', '.5', '0o17', '0b101', '017', '+.inf', '.NaN'],
  },
  {
    long: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
    nested: { deeper: { longer: 'word '.repeat(40).trim() } },
    multi: 'first line\nsecond line\n\n  indented line\nlast',
    multiLong: 'short\n' + 'x '.repeat(60) + '\nend\n',
    nolongspace: 'a'.repeat(120),
    keyWith: { 'key: colon': 1, '- dash': 2, 'yes': 3, '123': 4, 'with space': 5, '': 6 },
  },
  { numbers: [0, 1e21, 1e-7, 0.1, 123456789012, -5, 3.14159, 1e100, Number.MAX_SAFE_INTEGER] },
  { ref1: shared, ref2: shared, list: [shared] },
  [{ a: 1, b: [1, [2, 3]] }, 'x', [], {}],
  'just a string',
  { emptyList: [], emptyObj: {}, listOfEmpty: [[], {}], nul: [null, null] },
];

const xmlCases = [
  { promptfoo: { evalId: 'eval-1', results: { a: '1', list: ['x', 'y'], obj: { k: '<&>"\'' } }, config: {}, shareableUrl: '' } },
  {
    '?xml': { '@_version': '1.0', '@_encoding': 'UTF-8' },
    testsuites: {
      '@_errors': 1, '@_name': 'promptfoo', '@_time': '4.25',
      testsuite: [
        { '@_name': '[echo] prompt 1', '@_tests': 2, testcase: [
          { '@_name': 'test 1', '@_time': '1.25', '@_classname': '[echo] prompt 1' },
          { '@_name': 'test 2 <&>', '@_time': '0', failure: { '#text': 'Score: 0\nReason: Assertion failed', '@_message': 'Assertion failed', '@_type': 'assertion' }, '@_classname': 'c' },
        ] },
      ],
    },
  },
  { root: { empty: '', nested: [{ a: '1' }, { b: { c: '' } }], arr: [['p', 'q']], t: 'true', '@_flag': 'true' } },
];

const csvCases = [
  [['a', 'b,c', 'd"e', 'f\ng', 'h\ri', '', 'x'], [1, 2.5, true, false, null, undefined, 'z']],
  [["'=1", ' lead', 'trail ', '', 'é']],
];

const tableCases = [
  {
    head: { vars: ['var1', 'var2'], prompts: [{ provider: 'test-provider', label: 'test-label' }] },
    body: [
      { vars: ['value1', 'value2'], outputs: [{ pass: true, text: 'passing test' }] },
      { vars: ['value3', 'value4'], outputs: [{ pass: false, text: 'failing test', failureReason: 1 }] },
      { vars: ['v5', ''], outputs: [{ pass: false, text: 'error test---more', failureReason: 2 }] },
    ],
  },
  {
    head: { vars: ['question'], prompts: [{ provider: 'openai:gpt', label: 'Answer: {{question}}' }, { provider: 'echo', label: 'p2' }] },
    body: [
      {
        vars: ['What is the capital of France and why is it so famous among travellers around the world? Please explain in detail.'],
        outputs: [
          { pass: true, text: 'Paris is the capital.\nIt is famous for the Eiffel Tower, museums, cafes and a supercalifragilisticexpialidociousword-that-is-too-long-to-fit-in-a-column-of-forty.' },
          { pass: false, text: 'Expected output to contain "Berlin"---\nParis', failureReason: 1 },
        ],
      },
      { vars: ['日本語のテキスト 😀'], outputs: [{ pass: true, text: '' }, { pass: false, text: 'x'.repeat(300), failureReason: 2 }] },
    ],
  },
  { head: { vars: [], prompts: [{ provider: 'p', label: 'only head' }] }, body: [] },
];

const out = {
  yaml: yamlCases.map((input) => ({ input, output: yaml.dump(input) })),
  xml: xmlCases.map((input) => ({
    input,
    output: new XMLBuilder({ ignoreAttributes: false, format: true, indentBy: '  ' }).build(input),
  })),
  csv: csvCases.map((input) => ({ input, output: stringify(input) })),
  table: tableCases.map((input) => ({ input, output: generateTable(input) })),
  tableSmall: [{ input: tableCases[0], output: generateTable(tableCases[0], 5, 1) }],
  wrapTable: [
    [[{ col1: 'value1', col2: 'value2' }, { col1: 'value3', col2: 'value4' }], undefined],
    [[{ col1: 'a long value that wraps', col2: 'b', col3: 'c' }], { col1: 10, col3: 15 }],
    [[{ name: 'John', age: 30 }], undefined],
  ].map(([rows, widths]) => ({ input: { rows, widths: widths ?? null }, output: wrapTable(rows, widths) })),
};
const dest = path.join(__dirname, '..', 'tests', 'fixtures', `output_golden_${mode}.json`);
fs.writeFileSync(dest, JSON.stringify(out, null, 2) + '\n');
console.log(`wrote ${dest}`);
