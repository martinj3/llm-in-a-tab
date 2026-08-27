// Dumps real activation frames from a real model so the visualization can
// be tuned against what it will actually be fed, rather than against a
// guess at the distribution. Writes a flat binary of N frames plus a JSON
// sidecar, for tests/viz-harness.html to load with ?real=1.
//
// Usage: node tests/dump-frames.mjs [135M|360M] [outDir]
import { writeFile, mkdir } from "node:fs/promises";
import path from "node:path";

import { resolveUrl } from "../js/models.js";
import { rangedFetch } from "../js/model/download.js";
import { parseHeader, decodeBf16, quantizeRowsInt8 } from "../js/model/safetensors.js";
import { fetchModelConfig } from "../js/model/config.js";
import { loadTokenizer, StreamDecoder } from "../js/model/tokenizer.js";
import { renderSystemTurn, renderUserTurn } from "../js/model/template.js";
import { prefill, decodeStep } from "../js/model/transformer.js";
import { createKVCache } from "../js/model/kvcache.js";
import { createProbe } from "../js/model/probe.js";
import { argmax } from "../js/model/sample.js";

const MAX_CTX = 1024;
const STEPS = 16;
const modelId = process.argv[2] || "135M";
const outDir = process.argv[3] || path.join("tests", "shots");

async function loadTensors(id) {
  const url = resolveUrl(id, "model.safetensors");
  const header = parseHeader(new Uint8Array(await rangedFetch(url, 0, 200_000)));
  const names = [...header.tensors.keys()];
  const tensors = new Map();
  let i = 0;
  async function one() {
    while (i < names.length) {
      const name = names[i++];
      const info = header.tensors.get(name);
      const raw = await rangedFetch(url, header.dataStart + info.offsets[0], header.dataStart + info.offsets[1]);
      const f32 = decodeBf16(raw, info.shape.reduce((a, b) => a * b, 1));
      if (info.shape.length === 2) {
        const { qweight, scales } = quantizeRowsInt8(f32, info.shape[0], info.shape[1]);
        tensors.set(name, { kind: "i8", shape: info.shape, qweight, scales });
      } else {
        tensors.set(name, { kind: "f32", shape: info.shape, f32 });
      }
    }
  }
  await Promise.all([one(), one(), one(), one()]);
  return tensors;
}

const config = await fetchModelConfig(modelId);
const tokenizer = await loadTokenizer(modelId);
console.log("fetching weights...");
const tensors = await loadTensors(modelId);

const cache = createKVCache(config, MAX_CTX);
const probe = createProbe(config, MAX_CTX);
const ids = tokenizer.encode(renderSystemTurn() + renderUserTurn("Explain gravity in one sentence."));
let logits = prefill(tensors, config, cache, ids);

const decoder = new StreamDecoder(tokenizer);
const frames = [];
const texts = [];
for (let s = 0; s < STEPS; s++) {
  const id = argmax(logits);
  texts.push(decoder.push(id) || "");
  probe.beginToken();
  logits = decodeStep(tensors, config, cache, id, probe);
  const f = probe.endToken();
  frames.push({ bytes: f.bytes, attnUsed: f.attnUsed, pos: cache.seqLen - 1 });
}

// Histogram of |activation| across every captured MLP cell, which is the
// number that decides whether the colour ramp reads as a sparse
// constellation or as television static.
const { mlpCells, stride, layers } = probe.geometry;
const hist = new Array(16).fill(0);
let total = 0;
for (const f of frames) {
  for (let l = 0; l < layers; l++) {
    for (let c = 0; c < mlpCells; c++) {
      const mag = Math.abs(f.bytes[l * stride + c] - 128) / 127;
      hist[Math.min(15, (mag * 16) | 0)]++;
      total++;
    }
  }
}
console.log("\n|activation| distribution (fraction of cells per 1/16 bucket):");
hist.forEach((n, i) =>
  console.log(
    `  ${(i / 16).toFixed(3)}-${((i + 1) / 16).toFixed(3)}  ${((n / total) * 100).toFixed(2)}%  ` +
      "#".repeat(Math.round((n / total) * 200))
  )
);
let neg = 0;
for (const f of frames) {
  for (let l = 0; l < layers; l++) {
    for (let c = 0; c < mlpCells; c++) if (f.bytes[l * stride + c] < 128) neg++;
  }
}
console.log(`negative fraction: ${((neg / total) * 100).toFixed(1)}%`);

await mkdir(outDir, { recursive: true });
const buf = Buffer.concat(frames.map((f) => Buffer.from(f.bytes)));
await writeFile(path.join(outDir, "frames.bin"), buf);
await writeFile(
  path.join(outDir, "frames.json"),
  JSON.stringify(
    { modelId, ...probe.geometry, count: frames.length,
      attnUsed: frames.map((f) => f.attnUsed), pos: frames.map((f) => f.pos), texts },
    null, 1
  )
);
console.log(`\nwrote ${frames.length} frames (${(buf.length / 1024).toFixed(0)} KB) to ${outDir}`);
console.log(`tokens: ${JSON.stringify(texts.join(""))}`);
