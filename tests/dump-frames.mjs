// Dumps real activation frames from a real model so the visualization can
// be tuned against what it will actually be fed, rather than against a
// guess at the distribution. Writes a flat binary of N frames plus a JSON
// sidecar, for tests/viz-harness.html to load with ?real=1.
//
// Usage: node tests/dump-frames.mjs [135M|360M] [outDir] [prompt]
//
// The prompt matters more than it looks: the attention block is
// heads x context, so a short prompt dumps frames whose attention plates
// are 40 columns of fat blocks and a long one dumps the pixel-dense case.
// The default is deliberately wordy for that reason.
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
const prompt = process.argv[4] ||
  "Here is a passage I would like you to think about carefully before answering. " +
  "Gravity is one of the four fundamental interactions, alongside electromagnetism " +
  "and the strong and weak nuclear forces. It is by far the weakest of the four, " +
  "yet it is the one that shapes the large-scale structure of the universe, because " +
  "it is always attractive and has unlimited range, so it accumulates over enormous " +
  "distances where the others cancel out. Newton described it as a force between " +
  "masses; Einstein described it instead as the curvature of spacetime, with masses " +
  "following the straightest available paths through a geometry they themselves bend. " +
  "Both descriptions agree closely in weak fields, and the second is needed near very " +
  "dense objects and for the universe as a whole. Now, in one sentence: what is gravity?";
const ids = tokenizer.encode(renderSystemTurn() + renderUserTurn(prompt));
let logits = prefill(tensors, config, cache, ids);

const decoder = new StreamDecoder(tokenizer);
const frames = [];
const texts = [];
for (let s = 0; s < STEPS; s++) {
  const id = argmax(logits);
  texts.push(decoder.push(id) || "");
  probe.beginToken(cache.seqLen + 1);
  logits = decodeStep(tensors, config, cache, id, probe);
  const f = probe.endToken();
  frames.push({ bytes: f.bytes, cols: f.cols, stride: f.stride, pos: cache.seqLen - 1 });
}

const { mlpCells, residCells, heads, layers, attnBase } = probe.geometry;

// Distribution of a byte field across every captured frame. These are the
// numbers that decide whether a colour ramp reads as a sparse
// constellation or as television static, and guessing them wrong is how
// the first pass at this looked like noise.
function histogram(label, pick) {
  const hist = new Array(16).fill(0);
  let total = 0;
  for (const f of frames) {
    for (let l = 0; l < layers; l++) pick(f, l, (mag) => { hist[Math.min(15, (mag * 16) | 0)]++; total++; });
  }
  console.log(`\n${label} (fraction per 1/16 bucket):`);
  hist.forEach((n, i) =>
    console.log(
      `  ${(i / 16).toFixed(3)}-${((i + 1) / 16).toFixed(3)}  ${((n / total) * 100).toFixed(2)}%  ` +
        "#".repeat(Math.round((n / total) * 200))
    )
  );
  return total;
}

const total = histogram("|mlp activation|", (f, l, add) => {
  for (let c = 0; c < mlpCells; c++) add(Math.abs(f.bytes[l * f.stride + c] - 128) / 127);
});
let neg = 0;
for (const f of frames) {
  for (let l = 0; l < layers; l++) {
    for (let c = 0; c < mlpCells; c++) if (f.bytes[l * f.stride + c] < 128) neg++;
  }
}
console.log(`negative fraction: ${((neg / total) * 100).toFixed(1)}%`);

histogram("attention weight (sqrt of own-peak fraction)", (f, l, add) => {
  const base = l * f.stride + attnBase;
  for (let i = 0; i < heads * f.cols; i++) add(f.bytes[base + i] / 255);
});

await mkdir(outDir, { recursive: true });
// Frames vary in length now (the attention block grows with the context),
// so the sidecar carries per-frame byte offsets rather than one stride.
const buf = Buffer.concat(frames.map((f) => Buffer.from(f.bytes)));
await writeFile(path.join(outDir, "frames.bin"), buf);
let off = 0;
const offsets = frames.map((f) => {
  const at = off;
  off += f.bytes.length;
  return at;
});
await writeFile(
  path.join(outDir, "frames.json"),
  JSON.stringify(
    { modelId, layers, mlpCells, residCells, heads, attnBase,
      maxCols: probe.geometry.maxCols, count: frames.length,
      cols: frames.map((f) => f.cols), stride: frames.map((f) => f.stride),
      offsets, pos: frames.map((f) => f.pos), texts },
    null, 1
  )
);
console.log(`\nwrote ${frames.length} frames (${(buf.length / 1024).toFixed(0)} KB) to ${outDir}`);
console.log(`tokens: ${JSON.stringify(texts.join(""))}`);
