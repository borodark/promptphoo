// Score tests/fixtures/textmetrics_golden.json cases with promptfoo's real assertion handlers
// (src/assertions/{levenshtein,bleu,gleu,rouge,meteor}.ts) and their npm dependencies.
//
// Setup (in a scratch dir, no install scripts):
//   npm install js-rouge@3.2.3 natural@8.1.1 fastest-levenshtein@1.0.16 esbuild@0.24.0 --ignore-scripts
//   printf '%s\n' \
//     "export * from '<promptfoo>/src/assertions/levenshtein';" \
//     "export * from '<promptfoo>/src/assertions/bleu';" \
//     "export * from '<promptfoo>/src/assertions/gleu';" \
//     "export * from '<promptfoo>/src/assertions/rouge';" \
//     "export * from '<promptfoo>/src/assertions/meteor';" > entry.ts
//   ./node_modules/.bin/esbuild entry.ts --bundle --platform=node --format=cjs \
//     --supported:dynamic-import=false \
//     --external:js-rouge --external:natural --external:fastest-levenshtein --outfile=bundle.js
// --supported:dynamic-import=false turns meteor.ts's `await import('natural')` into an interop
// require. Under native ESM import() (how promptfoo's CLI loads it), natural@8.1.1's namespace
// only has `default`, so PorterStemmer is undefined and METEOR throws
// "Invariant failed: PorterStemmer should be loaded" (an upstream bug); the golden values are
// for the intended behaviour.
// Usage:
//   NODE_PATH=<scratch>/node_modules node tools/textmetrics_golden.js <scratch>/bundle.js
const fs = require('fs');
const path = require('path');

const h = require(path.resolve(process.argv[2]));
const rouge = require('js-rouge');

// Prose of the kind LLM outputs contain. Used for rouge-l/rouge-s (which segment sentences) and
// for every other metric.
const TEXTS = [
  'The cat sat on the mat.',
  'the cat sat on the mat',
  'A cat was sitting on the mat.',
  'The quick brown fox jumps over the lazy dog.',
  'The cat sat. A dog ran.',
  'A dog ran. The cat sat.',
  'Hello world. next',
  'Paris is the capital of France. It is known for the Eiffel Tower.',
  'The capital of France is Paris, which is famous for the Eiffel Tower.',
  'Dr. Smith went to Washington. He arrived at 5 p.m. on Monday.',
  'Dr. Smith arrived in Washington on Monday at 5 p.m.',
  'The U.S. government said so. Then it ended.',
  'I paid $3.50 for it! Really? Yes.',
  "It's John's book. Don't touch it, he can't share.",
  'Wait... what happened? Nothing much.',
  'See e.g. the docs. They help a lot.',
  'Version 2.0 is out. Upgrade now!',
  'Results: accuracy 95.5%, recall 0.8. Good enough.',
  'Line one\nline two\n\nNew paragraph here.',
  'First line.\nSecond line.\nThird line.',
  'He said "Stop." Then he left.',
  '"The cat sat." "A dog ran."',
  'Photosynthesis converts light energy into chemical energy. Plants use it to make glucose.',
  'Plants make glucose from light through photosynthesis, storing chemical energy.',
  'Machine learning is a subset of AI. It learns patterns from data (without explicit rules).',
  'In short, the answer is 42. The question, however, remains unknown.',
  'Water boils at 100 degrees Celsius at sea level. At higher altitudes it boils at lower temperatures.',
  'To install, run npm install. Then start the server with npm start.',
  'Mr. and Mrs. Jones live on Elm St. in Springfield. They moved there in 2019.',
  'The meeting is at 10 a.m. tomorrow. Please be on time.',
  'Yes. No. Maybe.',
  'hello',
  'The the the the the the the',
  'It is a guide to action which ensures that the military always obeys the commands of the party',
  'It is a guide to action that ensures that the military will forever heed Party commands',
  'Why is the sky blue? Because of Rayleigh scattering. Shorter wavelengths scatter more.',
  'The sky appears blue because shorter wavelengths of light are scattered more by the atmosphere.',
  'Note: this is important. Read it twice.',
  'Use the --force flag carefully -- it overwrites files.',
  "The students' essays were graded. Each one got a score.",
  'Einstein (1879-1955) developed relativity. He won the Nobel Prize in 1921.',
  'The result was ok... but not great. We will try again.',
  'Step one is done. step two is next.',
  'The price rose 5%. Analysts expected 3%.',
  'Call me at 555-1234. Or email me.',
  'UPPER CASE TEXT. MORE UPPER CASE.',
  'Café au lait. Crème brûlée!',
  'Emoji test 😀 works. Another sentence.',
  '  leading and trailing spaces.  ',
  'tabs\tand\tspaces mixed. ok',
  'Gonna wanna gotta lemme gimme. Cannot stop.',
  "'Tis the season. 'Twas the night.",
];

// Lists and other structures the Python segmenter ports only partly; recorded so the test can
// report divergences explicitly.
const SEGMENT_ONLY = [
  '1. First item\n2. Second item\n3. Third item',
  'Steps:\n1. Open the file.\n2. Edit it.\n3. Save it.',
  '- bullet one\n- bullet two',
  'Visit example.com. It is great.',
  'Mr. J. R. R. Tolkien wrote books.',
  'The cat sat.The dog ran.',
  'She asked, "Are you coming?" and left.',
  'He said “hello.” Then he left.',
  'See Smith et al. (2020). The results hold.',
  'Use `npm install`. Then run it.',
];

const PAIRS = [];
for (let i = 0; i < TEXTS.length; i++) {
  PAIRS.push([TEXTS[i], TEXTS[i]]);
  PAIRS.push([TEXTS[i], TEXTS[(i + 1) % TEXTS.length]]);
  PAIRS.push([TEXTS[i], TEXTS[(i + 7) % TEXTS.length]]);
}

// Deterministic pseudo-random strings for levenshtein (ASCII, accented, astral, long).
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}
const ALPHABET = ['a', 'b', 'c', 'd', ' ', 'é', '😀', 'z', 'Q', '\n'];
const rand = rng(42);
function randomString(len) {
  let out = '';
  for (let i = 0; i < len; i++) {
    out += ALPHABET[Math.floor(rand() * ALPHABET.length)];
  }
  return out;
}
const LEV = [
  ['test', 'test'],
  ['tast', 'test'],
  ['', ''],
  ['', 'abc'],
  ['kitten', 'sitting'],
  ['😀', ''],
  ['😀a', 'a😀'],
];
for (const len of [3, 10, 31, 32, 33, 64, 65, 100, 300]) {
  LEV.push([randomString(len), randomString(len + Math.floor(rand() * 5))]);
  LEV.push([randomString(len), randomString(Math.floor(len / 2))]);
}

const assertionFor = (type, extra = {}) => ({ type, ...extra });

async function main() {
  const cases = [];
  const record = async (type, output, value, extra = {}, inverse = false) => {
    const assertion = assertionFor(type, extra);
    const params = { assertion, baseType: type, inverse, outputString: output, renderedValue: value };
    let result;
    try {
      const fn = {
        levenshtein: h.handleLevenshtein,
        bleu: h.handleBleuScore,
        gleu: h.handleGleuScore,
        'rouge-n': h.handleRougeScore,
        'rouge-l': h.handleRougeScore,
        'rouge-s': h.handleRougeScore,
        meteor: h.handleMeteorAssertion,
      }[type];
      const r = await fn(params);
      result = { pass: r.pass, score: r.score, reason: r.reason };
    } catch (e) {
      result = { error: String(e.message) };
    }
    cases.push({ type, output, value, extra, inverse, result });
  };

  for (const [a, b] of LEV) {
    await record('levenshtein', a, b);
    await record('levenshtein', a, b, { threshold: 40 }, true);
  }
  for (const [a, b] of PAIRS) {
    for (const type of ['bleu', 'gleu', 'rouge-n', 'rouge-l', 'rouge-s', 'meteor']) {
      await record(type, a, b);
    }
  }
  // Multiple references, thresholds and inverse.
  const multi = [TEXTS[0], TEXTS[2], TEXTS[3]];
  for (const type of ['bleu', 'gleu', 'meteor']) {
    await record(type, TEXTS[1], multi);
    await record(type, TEXTS[1], multi, { threshold: 0.9 }, true);
    await record(type, '', TEXTS[0]);
    await record(type, '   ', [TEXTS[0], '']);
  }
  await record('meteor', TEXTS[1], TEXTS[2], { alpha: 0.85, beta: 2, gamma: 0.4, threshold: 0.6 });
  for (const type of ['rouge-n', 'rouge-l', 'rouge-s']) {
    await record(type, TEXTS[0], TEXTS[2], { threshold: 0.3 }, true);
    await record(type, '\u0085', TEXTS[2]);
  }

  const segments = [...TEXTS, ...SEGMENT_ONLY].map((text) => ({
    text,
    sentences: rouge.sentenceSegment(text, { caseNeutral: true }),
    tokens: rouge.treeBankTokenize(text.toLowerCase()),
  }));

  const out = { versions: { 'js-rouge': '3.2.3', natural: '8.1.1', 'fastest-levenshtein': '1.0.16' }, cases, segments };
  const file = path.join(__dirname, '..', 'tests', 'fixtures', 'textmetrics_golden.json');
  fs.writeFileSync(file, JSON.stringify(out, null, 1) + '\n');
  console.log(`wrote ${cases.length} cases and ${segments.length} segmentations`);
}

main();
