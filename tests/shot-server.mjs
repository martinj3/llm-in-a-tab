// Static file server for the repo that also accepts screenshots back from
// the page: POST /shot with a data: URL body writes a PNG next to it.
//
// The Browser pane cannot composite in this environment, so there is no
// way to take a screenshot of a rendered page from outside it -- but
// canvas 2D still draws perfectly well without a compositor, so the page
// can hand its own pixels back. That is the only way to actually look at
// what js/viz/stack.js produces rather than inferring it from pixel
// statistics.
//
// Usage: node tests/shot-server.mjs [port]   (default 8124)
//        then open http://localhost:8124/tests/viz-harness.html
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PORT = Number(process.argv[2] || 8124);

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
};

createServer(async (req, res) => {
  if (req.method === "POST" && req.url.startsWith("/shot")) {
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const body = Buffer.concat(chunks).toString("utf-8");
    const name = (new URL(req.url, "http://x").searchParams.get("name") || "shot") + ".png";
    const b64 = body.slice(body.indexOf(",") + 1);
    const out = path.join(ROOT, "tests", "shots", name);
    await writeFile(out, Buffer.from(b64, "base64"));
    res.writeHead(200, { "content-type": "text/plain" });
    res.end(out);
    console.log(`wrote ${out} (${(b64.length * 0.75 / 1024).toFixed(0)} KB)`);
    return;
  }

  const rel = decodeURIComponent(new URL(req.url, "http://x").pathname);
  const file = path.join(ROOT, rel === "/" ? "index.html" : rel);
  if (!file.startsWith(ROOT)) {
    res.writeHead(403).end("forbidden");
    return;
  }
  try {
    const data = await readFile(file);
    res.writeHead(200, { "content-type": TYPES[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  } catch {
    res.writeHead(404).end("not found");
  }
}).listen(PORT, () => console.log(`serving ${ROOT} on http://localhost:${PORT}`));
