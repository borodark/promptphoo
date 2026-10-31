#!/bin/sh
# Regenerate src/promptphoo/_data/grading_prompts.json from upstream's prompts/grading.ts and
# external/prompts/ragas.ts, evaluated by Node so dedent and JSON.stringify output is exact.
#   tools/gen_grading_prompts.sh /path/to/promptfoo /path/to/node_modules/dedent
set -eu
UP=$1; DEDENT=$2; OUT=$(cd "$(dirname "$0")/.." && pwd)/src/promptphoo/_data/grading_prompts.json
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
{ echo "import dedent from '$DEDENT/dist/dedent.mjs';"
  grep -v "^import\|^export \* from" "$UP/src/external/prompts/ragas.ts"
  grep -v "^import\|^export \* from" "$UP/src/prompts/grading.ts"; } > "$TMP/all.mts"
echo "import * as m from './all.mts'; process.stdout.write(JSON.stringify(m, null, 1));" > "$TMP/dump.mts"
node --experimental-strip-types --no-warnings "$TMP/dump.mts" > "$OUT"
