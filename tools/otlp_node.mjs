#!/usr/bin/env node
// Drives upstream promptfoo's own tracing modules for promptphoo's golden and interop tests.
//
//   node tools/otlp_node.mjs <dist/src dir> encode       stdin: OTLP JSON  -> stdout: base64 protobuf
//   node tools/otlp_node.mjs <dist/src dir> decode       stdin: base64     -> stdout: decoded JSON
//   node tools/otlp_node.mjs <dist/src dir> serve <port> [options JSON]
//        starts upstream's OTLP receiver (startOTLPReceiver) on 127.0.0.1:<port>, prints
//        "ready" once it is listening and stops when stdin closes.
//
// <dist/src dir> is the directory of promptfoo 0.125.1's dist/src/main.js. Its chunk file names
// are content hashes, so the modules are found by their exports.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [distDir, mode, ...rest] = process.argv.slice(2);

async function findChunk(prefix, marker) {
  for (const name of fs.readdirSync(distDir).sort()) {
    if (!name.startsWith(prefix) || !name.endsWith('.js')) continue;
    const text = fs.readFileSync(path.join(distDir, name), 'utf8');
    if (text.includes(marker)) return import(pathToFileURL(path.join(distDir, name)).href);
  }
  throw new Error(`no ${prefix} chunk exporting ${marker}`);
}

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString('utf8');
}

function toPlain(value) {
  if (value instanceof Uint8Array) return { $bytes: Buffer.from(value).toString('hex') };
  if (Array.isArray(value)) return value.map(toPlain);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, toPlain(v)]));
  }
  if (typeof value === 'number' && !Number.isFinite(value)) return { $number: String(value) };
  if (Object.is(value, -0)) return { $number: '-0' };
  return value;
}

async function main() {
  if (mode === 'encode' || mode === 'decode' || mode === 'decode-lines') {
    const protobuf = await findChunk('protobuf-', 'encodeExportTraceServiceRequest as r');
    if (mode === 'encode') {
      const buf = await protobuf.r(JSON.parse((await readStdin())));
      process.stdout.write(buf.toString('base64'));
      return;
    }
    if (mode === 'decode-lines') {
      const out = [];
      const lines = (await readStdin()).split('\n');
      if (lines.at(-1) === '') lines.pop();
      for (const line of lines) {
        try {
          out.push(JSON.stringify({ ok: toPlain(await protobuf.n(Buffer.from(line, 'base64'))) }));
        } catch (error) {
          out.push(JSON.stringify({ error: error.message }));
        }
      }
      // The logger writes to stdout too, so each result line carries a marker.
      process.stdout.write(out.map((line) => `@@${line}`).join('\n') + '\n');
      return;
    }
    try {
      const decoded = await protobuf.n(Buffer.from((await readStdin()).trim(), 'base64'));
      process.stdout.write(JSON.stringify({ ok: toPlain(decoded) }));
    } catch (error) {
      process.stdout.write(JSON.stringify({ error: error.message }));
    }
    return;
  }
  if (mode === 'serve') {
    const receiver = await findChunk('otlpReceiver-', 'export { deregisterOTLPReceiverTracePolicy');
    const port = Number(rest[0]);
    const options = rest[1] ? JSON.parse(rest[1]) : {};
    await receiver.startOTLPReceiver(port, '127.0.0.1', options.acceptFormats, options.extra);
    process.stdout.write('ready\n');
    process.stdin.resume();
    process.stdin.on('end', async () => {
      await receiver.stopOTLPReceiver();
      process.exit(0);
    });
    return;
  }
  throw new Error(`unknown mode ${mode}`);
}

main().catch((error) => {
  process.stderr.write(String(error?.stack ?? error));
  process.exit(1);
});
