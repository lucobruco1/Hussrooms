/**
 * Hussrooms static server - zero dependencies.
 *
 *   node server.js            -> http://0.0.0.0:5173
 *   node server.js --port 8080
 *
 * It exists because ES modules cannot be loaded over file://, and because the
 * preview environment proxies it through a hostname. Deliberately no
 * X-Frame-Options / CSP frame-ancestors header, so the game can be embedded.
 */

import { createServer } from 'node:http';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)));
const argv = process.argv.slice(2);
const portArg = argv.indexOf('--port');
const PORT = Number(process.env.PORT || (portArg >= 0 ? argv[portArg + 1] : 0) || 5173);
const HOST = process.env.HOST || '0.0.0.0';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.wasm': 'application/wasm',
  '.txt': 'text/plain; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
};

function safeJoin(root, urlPath) {
  const decoded = decodeURIComponent(urlPath.split('?')[0].split('#')[0]);
  const rel = normalize(decoded).replace(/^([/\\])+/, '');
  const full = join(root, rel);
  return full.startsWith(root) ? full : null;
}

const server = createServer((req, res) => {
  let file = safeJoin(ROOT, req.url === '/' ? '/index.html' : req.url);
  if (!file) {
    res.writeHead(403).end('forbidden');
    return;
  }
  if (existsSync(file) && statSync(file).isDirectory()) file = join(file, 'index.html');
  if (!existsSync(file)) {
    res.writeHead(404, { 'content-type': 'text/plain' }).end(`404 ${req.url}`);
    return;
  }
  const type = MIME[extname(file).toLowerCase()] || 'application/octet-stream';
  const size = statSync(file).size;
  res.writeHead(200, {
    'content-type': type,
    'content-length': size,
    'cache-control': 'no-cache',
    'access-control-allow-origin': '*',
  });
  if (req.method === 'HEAD') {
    res.end();
    return;
  }
  createReadStream(file).pipe(res);
});

server.listen(PORT, HOST, () => {
  console.log(`Hussrooms -> http://${HOST}:${PORT}`);
  console.log('  index.html          game (WebGL, three.js)');
  console.log('  src/ai/lab.html     headless lab: train and inspect brains in a tab');
  console.log('  public/brain.json   pretrained checkpoint (served when present)');
});
