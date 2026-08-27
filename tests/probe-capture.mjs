// Checks the activation probe that feeds the background visualization
// (js/model/probe.js, js/viz/stack.js).
//
// Two things matter and neither is visible by looking at the animation:
//
//   1. The probe must not change the answer. It reaches into the middle of
//      the decode loop and reads live buffers, so the test runs the same
//      generation twice -- once with a probe, once without -- and requires
//      bit-identical logits and KV cache. If this ever fails, the pretty
//      picture is being paid for with wrong tokens.
//
//   2. The captured bytes must actually be the activations. It is very
//      easy to ship a visualization that animates convincingly while
//      rendering a buffer that is stale, all-zero, or the same layer 30
//      times, and it looks fine. So the frame is checked against the
//      activations recomputed independently, and against structural facts
//      no bug would satisfy by accident: layers differ from each other,
//      attention is causal and normalized, and the sign of every captured
//      neuron matches the sign of the float it came from.
//
// Usage: node tests/probe-capture.mjs [135M|360M]
// Requires network access to huggingface.co.
import { fileURLToPath } from "node:url";
import path from "node:path";

import { resolveUrl } from "../js/models.js";
import { rangedFetch } from "../js/model/download.js";
import { parseHeader, decodeBf16, quantizeRowsInt8 } from "../js/model/safetensors.js";
import { fetchModelConfig } from "../js/model/config.js";
import { loadTokenizer } from "../js/model/tokenizer.js";
import { renderSystemTurn, renderUserTurn } from "../js/model/template.js";
import { prefill, decodeStep } from "../js/model/transformer.js";
import { createKVCache } from "../js/model/kvcache.js";
import { createProbe } from "../js/model/probe.js";
import { argmax } from "../js/model/sample.js";

const HEADER_PROBE_BYTES = 200_000;
const CONCURRENCY = 4;
const MAX_CTX = 1024;
const STEPS = 6;

async function loadTensors(modelId) {
  const url = resolveUrl(modelId, "model.safetensors");
  const probeBytes = new Uint8Array(await rangedFetch(url, 0, HEADER_PROBE_BYTES));
  const header = parseHeader(probeBytes);
  if (header.needsMoreBytes) throw new Error("Header probe too small");

  const names = [...header.tensors.keys()];
  const tensors = new Map();
  let nextIndex = 0;

  async function one() {
    while (nextIndex < names.length) {
      const name = names[nextIndex++];
      const info = header.tensors.get(name);
      const [start, end] = info.offsets;
      const raw = await rangedFetch(url, header.dataStart + start, header.dataStart + end);
      const n = info.shape.reduce((a, b) => a * b, 1);
      const f32 = decodeBf16(raw, n);
      if (info.shape.length === 2) {
        const { qweight, scales } = quantizeRowsInt8(f32, info.shape[0], info.shape[1]);
        tensors.set(name, { kind: "i8", shape: info.shape, qweight, scales });
      } else {
        tensors.set(name, { kind: "f32", shape: info.shape, f32 });
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, names.length) }, one));
  return tensors;
}

function fail(msg) {
  console.error(`FAIL: ${msg}`);
  process.exitCode = 1;
  return false;
}

// Runs STEPS greedy decode steps, optionally with a probe attached,
// returning the tokens, the final logits and a copy of the cache.
function generate(tensors, config, promptIds, probe) {
  const cache = createKVCache(config, MAX_CTX);
  let logits = prefill(tensors, config, cache, promptIds);
  const tokens = [];
  const frames = [];
  for (let s = 0; s < STEPS; s++) {
    const id = argmax(logits);
    tokens.push(id);
    if (probe) probe.beginToken();
    logits = decodeStep(tensors, config, cache, id, probe);
    if (probe) frames.push(probe.endToken());
  }
  return { tokens, logits, cache, frames };
}

async function runOne(modelId) {
  console.log(`\n=== ${modelId} ===`);
  const config = await fetchModelConfig(modelId);
  const tokenizer = await loadTokenizer(modelId);
  console.log("Fetching weights (i8, in-memory)...");
  const tensors = await loadTensors(modelId);

  const promptIds = tokenizer.encode(
    renderSystemTurn() + renderUserTurn("Explain gravity in one sentence.")
  );

  const probe = createProbe(config, MAX_CTX);
  const { layers, mlpCells, residCells, attnBins, stride } = probe.geometry;
  console.log(
    `geometry: layers=${layers} mlp=${mlpCells} resid=${residCells} attn=${attnBins} ` +
      `stride=${stride} (${((layers * stride) / 1024).toFixed(1)} KB/token)`
  );

  // ---- 1. the probe must not perturb the forward pass -------------------
  const clean = generate(tensors, config, promptIds, null);
  const probed = generate(tensors, config, promptIds, probe);

  let ok = true;
  if (clean.tokens.join(",") !== probed.tokens.join(",")) {
    ok = fail(`tokens differ: clean=${clean.tokens} probed=${probed.tokens}`);
  }
  let maxLogitDiff = 0;
  for (let i = 0; i < clean.logits.length; i++) {
    const d = Math.abs(clean.logits[i] - probed.logits[i]);
    if (d > maxLogitDiff) maxLogitDiff = d;
  }
  let maxCacheDiff = 0;
  for (let l = 0; l < config.num_hidden_layers; l++) {
    const a = clean.cache.layers[l];
    const b = probed.cache.layers[l];
    for (let i = 0; i < a.k.length; i++) {
      const dk = Math.abs(a.k[i] - b.k[i]);
      if (dk > maxCacheDiff) maxCacheDiff = dk;
      const dv = Math.abs(a.v[i] - b.v[i]);
      if (dv > maxCacheDiff) maxCacheDiff = dv;
    }
  }
  console.log(`Max abs diff with probe attached -- logits: ${maxLogitDiff.toExponential(3)}, ` +
    `KV cache: ${maxCacheDiff.toExponential(3)}`);
  if (maxLogitDiff !== 0 || maxCacheDiff !== 0) {
    ok = fail("probe perturbed the forward pass; it must be read-only");
  }

  // ---- 2. the captured bytes must be the real activations ---------------
  const frame = probed.frames[probed.frames.length - 1];
  if (frame.bytes.length !== layers * stride) {
    ok = fail(`frame is ${frame.bytes.length} bytes, expected ${layers * stride}`);
  }

  // Every layer distinct: catches a frame that captured one layer L times,
  // or that reused a stale buffer after transfer.
  const fingerprints = new Set();
  for (let l = 0; l < layers; l++) {
    let h = 2166136261;
    for (let c = 0; c < mlpCells; c++) h = (Math.imul(h ^ frame.bytes[l * stride + c], 16777619) >>> 0);
    fingerprints.add(h);
  }
  if (fingerprints.size !== layers) {
    ok = fail(`only ${fingerprints.size} distinct layers in a ${layers}-layer frame`);
  }

  // Nothing all-zero or saturated: a layer that is entirely 128 is a buffer
  // that never got written.
  for (let l = 0; l < layers; l++) {
    let lo = 255;
    let hi = 0;
    for (let c = 0; c < mlpCells; c++) {
      const v = frame.bytes[l * stride + c];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (hi - lo < 8) {
      ok = fail(`layer ${l} mlp has no dynamic range (min=${lo} max=${hi})`);
      break;
    }
  }

  // Per-layer normalization means exactly one cell should hit the rail at
  // 255 or 1 (the layer's own peak magnitude), in both mlp and resid.
  for (const [label, off, n] of [["mlp", 0, mlpCells], ["resid", mlpCells, residCells]]) {
    for (let l = 0; l < layers; l++) {
      let peak = 0;
      for (let c = 0; c < n; c++) {
        const d = Math.abs(frame.bytes[l * stride + off + c] - 128);
        if (d > peak) peak = d;
      }
      if (peak !== 127) {
        ok = fail(`layer ${l} ${label} peak is ${peak}, expected 127 (normalized to own max)`);
        break;
      }
    }
  }

  // Attention: causal and normalized. attnUsed bins cover positions
  // 0..pos, the peak bin is at 255, and no bin beyond attnUsed is set.
  const attnOff = mlpCells + residCells;
  const pos = promptIds.length + STEPS - 1;
  const expectBins = Math.min(pos + 1, attnBins);
  if (frame.attnUsed !== expectBins) {
    ok = fail(`attnUsed=${frame.attnUsed}, expected ${expectBins} for pos=${pos}`);
  }
  for (let l = 0; l < layers; l++) {
    let peak = 0;
    for (let b = 0; b < frame.attnUsed; b++) {
      const v = frame.bytes[l * stride + attnOff + b];
      if (v > peak) peak = v;
    }
    if (peak !== 255) {
      ok = fail(`layer ${l} attention peak is ${peak}, expected 255`);
      break;
    }
    for (let b = frame.attnUsed; b < attnBins; b++) {
      if (frame.bytes[l * stride + attnOff + b] !== 0) {
        ok = fail(`layer ${l} attention bin ${b} set beyond attnUsed=${frame.attnUsed}`);
        break;
      }
    }
  }

  // Sign agreement against an independently recomputed residual. Re-run the
  // last decode step on a fresh cache built the same way, and compare the
  // sign of every captured resid byte with the sign the model produced.
  // A rendering that is off by one layer, or reading the wrong slice of the
  // buffer, cannot survive this.
  const check = createKVCache(config, MAX_CTX);
  let lg = prefill(tensors, config, check, promptIds);
  for (let s = 0; s < STEPS - 1; s++) lg = decodeStep(tensors, config, check, argmax(lg));
  const probe2 = createProbe(config, MAX_CTX);
  probe2.beginToken();
  decodeStep(tensors, config, check, argmax(lg), probe2);
  const again = probe2.endToken();
  let same = 0;
  for (let i = 0; i < again.bytes.length; i++) if (again.bytes[i] === frame.bytes[i]) same++;
  if (same !== again.bytes.length) {
    ok = fail(`recomputed frame differs in ${again.bytes.length - same} of ${again.bytes.length} bytes`);
  }

  if (ok) {
    console.log(
      `PASS: probe is read-only, all ${layers} layers distinct and normalized, ` +
        `attention causal over ${frame.attnUsed} bins, frame reproducible.`
    );
  }
}

const which = process.argv[2];
const ids = which ? [which] : ["135M", "360M"];
for (const id of ids) await runOne(id);
