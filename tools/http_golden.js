// Run tests/fixtures/http_golden_cases.json through promptfoo's real HttpProvider against a local
// HTTP server and write tests/fixtures/http_golden.json (the request the server saw, plus the
// provider response or error).
//
// Usage: PROMPTFOO_CACHE_ENABLED=false NODE_PATH=<dir with promptfoo@0.125.1 node_modules> \
//          node tools/http_golden.js
const fs = require('fs');
const http = require('http');
const path = require('path');
const pf = require('promptfoo');

const fixtures = path.join(__dirname, '..', 'tests', 'fixtures');
const cases = JSON.parse(fs.readFileSync(path.join(fixtures, 'http_golden_cases.json'), 'utf-8'));
const KEEP_HEADERS = new Set(['host', 'connection', 'content-length', 'user-agent', 'accept',
  'accept-encoding', 'accept-language', 'sec-fetch-mode']);

let current;
const server = http.createServer((req, res) => {
  let body = '';
  req.setEncoding('utf8');
  req.on('data', (c) => (body += c));
  req.on('end', () => {
    const headers = {};
    for (const [k, v] of Object.entries(req.headers)) {
      if (!KEEP_HEADERS.has(k)) {
        headers[k] = v;
      }
    }
    current.seen = { method: req.method, url: req.url, headers, body };
    const r = current.response;
    res.writeHead(r.status, r.statusText || undefined, {
      'content-type': 'application/json',
      ...(r.headers || {}),
    });
    res.end(r.body);
  });
});

server.listen(0, '127.0.0.1', async () => {
  const host = `127.0.0.1:${server.address().port}`;
  const out = [];
  for (const c of cases) {
    current = { response: c.response };
    const id = c.raw ? 'http' : `http://${host}${c.path}`;
    const vars = { ...(c.vars || {}), ...(c.raw ? { path: c.path, host } : {}) };
    let result;
    try {
      const [provider] = await pf.loadApiProviders([{ id, config: c.config }]);
      const ctx = { vars, prompt: { raw: c.prompt, label: 'p' }, ...(c.debug ? { debug: true } : {}) };
      const resp = await provider.callApi(c.prompt, ctx);
      delete resp.latencyMs;
      if (resp.metadata && resp.metadata.http) {
        delete resp.metadata.http.headers;
        delete resp.metadata.headers;
      }
      result = { response: JSON.parse(JSON.stringify(resp)) };
    } catch (e) {
      result = { error: String(e.message) };
    }
    const seen = current.seen && JSON.parse(JSON.stringify(current.seen).split(host).join('golden.test'));
    const scrubbed = JSON.parse(JSON.stringify(result).split(host).join('golden.test'));
    out.push({ name: c.name, request: seen || null, ...scrubbed });
  }
  server.close();
  fs.writeFileSync(path.join(fixtures, 'http_golden.json'), JSON.stringify(out, null, 2) + '\n');
  console.log(`wrote ${out.length} golden HTTP cases`);
});
