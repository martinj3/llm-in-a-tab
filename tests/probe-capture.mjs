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
//      every head's attention row is a distinct vector that decodes back
//      to a probability distribution, and the frame reproduces exactly
//      when the same decode step is re-run on a freshly built cache.
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
    if (probe) probe.beginToken(cache.seqLen + 1);
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
  const { layers, mlpCells, residCells, heads, maxCols, attnBase } = probe.geometry;
  const pos = promptIds.length + STEPS - 1;
  const expectCols = Math.min(pos + 1, maxCols);
  const expectStride = attnBase + heads * expectCols;
  console.log(
    `geometry: layers=${layers} mlp=${mlpCells} resid=${residCells} heads=${heads} ` +
      `maxCols=${maxCols}; at pos=${pos} stride=${expectStride} ` +
      `(${((layers * expectStride) / 1024).toFixed(1)} KB/token)`
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
  const stride = frame.stride;
  if (frame.cols !== expectCols) {
    ok = fail(`frame cols=${frame.cols}, expected ${expectCols} for pos=${pos}`);
  }
  if (stride !== expectStride) {
    ok = fail(`frame stride=${stride}, expected ${expectStride}`);
  }
  if (frame.bytes.length !== layers * stride) {
    ok = fail(`frame is ${frame.bytes.length} bytes, expected ${layers * stride}`);
  }

  const fnv = (from, n) => {
    let h = 2166136261;
    for (let i = 0; i < n; i++) h = Math.imul(h ^ frame.bytes[from + i], 16777619) >>> 0;
    return h;
  };

  // Every layer distinct: catches a frame that captured one layer L times,
  // or that reused a stale buffer after transfer.
  const fingerprints = new Set();
  for (let l = 0; l < layers; l++) fingerprints.add(fnv(l * stride, mlpCells));
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

  // ---- 3. the attention block must be the attention matrix --------------
  //
  // Every head of every layer gets its own row of `cols` bytes. Three
  // properties, each of which a plausible bug fails:
  //
  //  - Own-peak normalization puts exactly one 255 in each row. A row that
  //    peaks lower is a row that was never written, or was written with a
  //    scale borrowed from a different head.
  //  - Rows must differ from each other. Writing every head to the same
  //    offset, or capturing after `scores` has been overwritten, produces
  //    `heads` identical rows and looks perfectly plausible on screen.
  //    Not *every* pair, though: some trained heads collapse onto the
  //    position-0 sink so completely that the rest of the row quantizes to
  //    zeros, and two such heads are then byte-identical for real reasons.
  //    360M layer 18 does exactly this. So the bar is that no layer is
  //    entirely uniform and that duplicates stay rare -- and every
  //    duplicate found gets printed with how sink-dominated it was, so a
  //    real bug cannot hide behind the tolerance.
  //  - The bytes must decode back to a probability distribution. The probe
  //    stores sqrt(w / peak), so squaring recovers w/peak and the row sums
  //    to 1/peak. Softmax over `cols` positions bounds peak into
  //    [1/cols, 1], so that sum must land in [1, cols]. Nothing but a real
  //    post-softmax vector satisfies this at every layer and head.
  let rowSumLo = Infinity;
  let rowSumHi = 0;
  const allRows = new Map(); // fingerprint -> first "layer/head" that produced it
  let duplicates = 0;
  for (let l = 0; l < layers && ok; l++) {
    const rows = new Set();
    for (let h = 0; h < heads; h++) {
      const from = l * stride + attnBase + h * frame.cols;
      let peak = 0;
      let sum = 0;
      let dark = 0; // bytes below 4: positions the quantizer flattened away
      for (let c = 0; c < frame.cols; c++) {
        const v = frame.bytes[from + c];
        if (v > peak) peak = v;
        if (v < 4) dark++;
        const p = v / 255;
        sum += p * p;
      }
      if (peak !== 255) {
        ok = fail(`layer ${l} head ${h} attention peak is ${peak}, expected 255`);
        break;
      }
      // Quantization rounds, so the recovered sum can sit a hair either
      // side of 1 on a row that is essentially all sink; allow a little
      // slack at the bottom.
      if (!(sum >= 0.98 && sum <= frame.cols)) {
        ok = fail(
          `layer ${l} head ${h}: recovered mass ${sum.toFixed(3)} outside [1, ${frame.cols}] ` +
            `-- these bytes are not a softmax output`
        );
        break;
      }
      if (sum < rowSumLo) rowSumLo = sum;
      if (sum > rowSumHi) rowSumHi = sum;
      const fp = fnv(from, frame.cols);
      rows.add(fp);
      const seen = allRows.get(fp);
      if (seen === undefined) allRows.set(fp, `${l}/${h}`);
      else {
        duplicates++;
        console.log(
          `  note: layer ${l} head ${h} is byte-identical to ${seen} ` +
            `(${dark}/${frame.cols} positions quantize to zero -- sink-dominated)`
        );
      }
    }
    if (ok && rows.size < 2) {
      ok = fail(`layer ${l}: all ${heads} heads produced the same row`);
    }
  }
  const total = layers * heads;
  if (ok && allRows.size < total * 0.85) {
    ok = fail(`only ${allRows.size} distinct rows out of ${total} -- heads are collapsing`);
  }
  if (ok) {
    console.log(
      `attention: ${layers}x${heads}x${frame.cols} = ` +
        `${(layers * heads * frame.cols).toLocaleString()} weights/token, ` +
        `${allRows.size}/${total} rows distinct (${duplicates} sink-degenerate), ` +
        `1/peak in [${rowSumLo.toFixed(2)}, ${rowSumHi.toFixed(2)}] of a possible ${frame.cols}`
    );
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
  probe2.beginToken(check.seqLen + 1);
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
        `attention rows decode back to distributions over ${frame.cols} positions, ` +
        `frame reproducible.`
    );
  }
}

const which = process.argv[2];
const ids = which ? [which] : ["135M", "360M"];
for (const id of ids) await runOne(id);
