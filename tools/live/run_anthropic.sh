#!/bin/bash
# Live Anthropic verification: Node promptfoo vs promptphoo. Spends real API credit (about $1).
#
#   ANTHROPIC_KEY_FILE=~/path/to/key NODE_PROMPTFOO="node .../dist/src/main.js" \
#     tools/live/run_anthropic.sh [OUT_DIR]
#
# Phase 1 (wire parity): Node runs the matrix through anthropic_proxy.py in record mode;
#   promptphoo then runs it against the recordings with a dummy key. Every request promptphoo
#   sends must match one Node sent; compare_replay.py diffs the results row by row.
# Phase 2 (live): both CLIs call api.anthropic.com directly on the matrix, upstream's Anthropic
#   examples (PROMPTFOO_SRC, default ../promptfoo), an unpriced model and the error paths; then
#   a cached rerun, and a scan of every output for the key.
# Needs a local Ollama with nomic-embed-text for the `similar`/`answer-relevance` checks.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT=${1:-$REPO/.live}
KEY_FILE=${ANTHROPIC_KEY_FILE:?set ANTHROPIC_KEY_FILE to a file holding the API key}
NODE=${NODE_PROMPTFOO:?set NODE_PROMPTFOO to the Node promptfoo command}
PY=${PY_PROMPTFOO:-$REPO/.venv/bin/promptphoo}
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
UPSTREAM=${PROMPTFOO_SRC:-$REPO/../promptfoo}
unset OPENAI_API_KEY AZURE_OPENAI_API_KEY GOOGLE_API_KEY GEMINI_API_KEY ANTHROPIC_BASE_URL
export PROMPTFOO_DISABLE_TELEMETRY=1
rm -rf "$OUT" && mkdir -p "$OUT"
cp -r "$HERE/configs" "$OUT/configs"
for d in sonnet-5 sonnet-5-5 structured-outputs; do cp -r "$UPSTREAM/examples/anthropic/$d" "$OUT/configs/$d"; done
cp -r "$UPSTREAM/examples/claude-thinking" "$OUT/configs/claude-thinking"
C=$OUT/configs

echo "== phase 1: record (Node, live) and replay (promptphoo, dummy key)"
mkdir -p "$OUT/p1/node" "$OUT/p1/py"
"$PYTHON" "$HERE/anthropic_proxy.py" record 8765 "$OUT/p1/rec" & PX=$!; sleep 1
for c in promptfooconfig features; do
  ( cd "$C/matrix" && ANTHROPIC_API_KEY="$(cat "$KEY_FILE")" ANTHROPIC_BASE_URL=http://127.0.0.1:8765 \
    PROMPTFOO_CONFIG_DIR="$OUT/p1/state-node" $NODE eval -c $c.yaml --no-cache -j 1 \
    -o "$OUT/p1/node/$c.json" > "$OUT/p1/node/$c.log" 2>&1 ); echo "node $c exit $?"
done
kill $PX; wait $PX 2>/dev/null
"$PYTHON" "$HERE/anthropic_proxy.py" replay 8766 "$OUT/p1/rec" & PX=$!; sleep 1
for c in promptfooconfig features; do
  ( cd "$C/matrix" && ANTHROPIC_API_KEY=sk-ant-replay-dummy ANTHROPIC_BASE_URL=http://127.0.0.1:8766 \
    PROMPTFOO_CONFIG_DIR="$OUT/p1/state-py" "$PY" eval -c $c.yaml --no-cache -j 1 \
    -o "$OUT/p1/py/$c.json" > "$OUT/p1/py/$c.log" 2>&1 ); echo "py $c exit $?"
done
kill $PX; wait $PX 2>/dev/null
echo "replay misses: $(ls "$OUT/p1/rec/misses" 2>/dev/null | wc -l)"
"$PYTHON" "$HERE/compare_replay.py" "$OUT/p1" promptfooconfig features

echo "== phase 2: live, direct"
mkdir -p "$OUT/p2/node" "$OUT/p2/py"
run() { # name dir config extra-args...
  local name=$1 dir=$2 cfg=$3; shift 3
  for side in node py; do
    local cmd; [ $side = node ] && cmd=$NODE || cmd=$PY
    ( cd "$dir" && ANTHROPIC_API_KEY="${KEY_OVERRIDE:-$(cat "$KEY_FILE")}" PROMPTFOO_CONFIG_DIR="$OUT/p2/state-$side" \
      $cmd eval -c "$cfg" --no-cache -o "$OUT/p2/$side/$name.json" "$@" > "$OUT/p2/$side/$name.log" 2>&1 )
    echo "$side $name exit $?"
  done
}
run matrix "$C/matrix" promptfooconfig.yaml
run features "$C/matrix" features.yaml
run sonnet-5 "$C/sonnet-5" promptfooconfig.yaml
run sonnet-5-5 "$C/sonnet-5-5" promptfooconfig.yaml
run structured "$C/structured-outputs" promptfooconfig.yaml
run thinking "$C/claude-thinking" promptfooconfig.yaml --filter-providers '^anthropic:'
run unknown-model "$C/unknown-model" promptfooconfig.yaml
run bad-model "$C/errors" promptfooconfig.yaml
KEY_OVERRIDE=sk-ant-api03-invalid-key-for-testing run bad-key "$C/errors" promptfooconfig.yaml
for i in 1 2; do
  ( cd "$C/matrix" && ANTHROPIC_API_KEY="$(cat "$KEY_FILE")" PROMPTFOO_CONFIG_DIR="$OUT/p2/state-cache" \
    "$PY" eval -c features.yaml -o "$OUT/p2/py/cache$i.json" > "$OUT/p2/py/cache$i.log" 2>&1 ); echo "py cache run $i exit $?"
done
"$PYTHON" "$HERE/compare_live.py" "$OUT/p2"
echo "== key leak scan (files containing the key):"
grep -rlF -f "$KEY_FILE" "$OUT" 2>/dev/null || echo "none"
