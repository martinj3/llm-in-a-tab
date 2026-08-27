// safetensors header parsing, bf16 decode, int8 quantization, and the
// per-tensor streaming loader that replaces Phase 1's whole-file download.
//
// File layout (see plan.md 1.2): first 8 bytes are a little-endian u64
// giving the header length, then that many bytes of JSON describing every
// tensor's dtype/shape/byte-range, then the raw tensor data back to back.
import { resolveUrl } from "../models.js";
import { headContentLength, rangedFetch } from "./download.js";
import { getManifest, putManifest, putTensor } from "./store.js";

const HEADER_PROBE_BYTES = 200_000;
const CONCURRENCY = 4;

// Parses a buffer that starts at byte 0 of the file. If the buffer doesn't
// yet contain the full header, returns { needsMoreBytes: N } so the caller
// can fetch up to byte N and call this again.
export function parseHeader(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLen = Number(view.getBigUint64(0, true));
  const dataStart = 8 + headerLen;
  if (bytes.byteLength < dataStart) {
    return { needsMoreBytes: dataStart };
  }
  const headerJson = JSON.parse(new TextDecoder().decode(bytes.subarray(8, dataStart)));
  const tensors = new Map();
  for (const [name, info] of Object.entries(headerJson)) {
    if (name === "__metadata__") continue;
    tensors.set(name, { dtype: info.dtype, shape: info.shape, offsets: info.data_offsets });
  }
  return { headerLen, dataStart, tensors };
}

// bf16 is the top 16 bits of an f32: same sign/exponent layout, truncated
// mantissa. Reconstructing f32 is `bits << 16` with no rebiasing needed
// (plan.md 1.2). Writing through a Uint32Array view and reading back through
// a same-buffer Float32Array view does the reinterpretation for free.
export function decodeBf16(rawBytes, numElements) {
  const src = new DataView(rawBytes);
  const out = new Float32Array(numElements);
  const bits = new Uint32Array(out.buffer);
  for (let i = 0; i < numElements; i++) {
    const bf16 = src.getUint16(i * 2, true);
    bits[i] = bf16 << 16;
  }
  return out;
}

// Per-row int8 quantization (plan.md 1.3): one scale per output row, not
// one for the whole tensor, so a single outlier weight can't crush every
// other row into a handful of levels.
export function quantizeRowsInt8(f32, numRows, numCols) {
  const qweight = new Int8Array(numRows * numCols);
  const scales = new Float32Array(numRows);
  for (let r = 0; r < numRows; r++) {
    const rowStart = r * numCols;
    let maxAbs = 0;
    for (let c = 0; c < numCols; c++) {
      const v = Math.abs(f32[rowStart + c]);
      if (v > maxAbs) maxAbs = v;
    }
    const scale = maxAbs === 0 ? 1 : maxAbs / 127;
    scales[r] = scale;
    for (let c = 0; c < numCols; c++) {
      let q = Math.round(f32[rowStart + c] / scale);
      if (q > 127) q = 127;
      if (q < -127) q = -127;
      qweight[rowStart + c] = q;
    }
  }
  return { qweight, scales };
}

// Every tensor name and shape the architecture requires, derived purely
// from config.json -- nothing here is hardcoded per-model (plan.md 0).
// There is deliberately no lm_head.weight: tie_word_embeddings means the
// output projection reuses model.embed_tokens.weight (gotcha 3).
export function expectedTensorShapes(config) {
  const H = config.hidden_size;
  const L = config.num_hidden_layers;
  const nH = config.num_attention_heads;
  const nKV = config.num_key_value_heads;
  const I = config.intermediate_size;
  const V = config.vocab_size;
  const headDim = H / nH;
  if (!Number.isInteger(headDim)) {
    throw new Error(`hidden_size ${H} is not divisible by num_attention_heads ${nH}`);
  }

  const shapes = new Map();
  shapes.set("model.embed_tokens.weight", [V, H]);
  for (let i = 0; i < L; i++) {
    const p = `model.layers.${i}`;
    shapes.set(`${p}.self_attn.q_proj.weight`, [nH * headDim, H]);
    shapes.set(`${p}.self_attn.k_proj.weight`, [nKV * headDim, H]);
    shapes.set(`${p}.self_attn.v_proj.weight`, [nKV * headDim, H]);
    shapes.set(`${p}.self_attn.o_proj.weight`, [H, nH * headDim]);
    shapes.set(`${p}.mlp.gate_proj.weight`, [I, H]);
    shapes.set(`${p}.mlp.up_proj.weight`, [I, H]);
    shapes.set(`${p}.mlp.down_proj.weight`, [H, I]);
    shapes.set(`${p}.input_layernorm.weight`, [H]);
    shapes.set(`${p}.post_attention_layernorm.weight`, [H]);
  }
  shapes.set("model.norm.weight", [H]);
  return shapes;
}

// Phase 2 exit criterion: every tensor named in config.json's architecture
// is present with the expected shape. Throws with every problem listed at
// once, rather than failing on the first mismatch.
export function validateHeader(header, config) {
  const expected = expectedTensorShapes(config);
  const missing = [];
  const mismatched = [];
  for (const [name, shape] of expected) {
    const actual = header.tensors.get(name);
    if (!actual) {
      missing.push(name);
      continue;
    }
    const shapesEqual =
      actual.shape.length === shape.length && actual.shape.every((d, i) => d === shape[i]);
    if (!shapesEqual) {
      mismatched.push(`${name}: expected [${shape}], got [${actual.shape}]`);
    }
  }
  if (missing.length || mismatched.length) {
    throw new Error(
      `safetensors header does not match config.json architecture. ` +
        `Missing tensors: ${missing.join(", ") || "none"}. ` +
        `Shape mismatches: ${mismatched.join("; ") || "none"}.`
    );
  }
  return expected;
}

function sameNameSet(namesA, namesB) {
  if (namesA.length !== namesB.length) return false;
  const setB = new Set(namesB);
  return namesA.every((n) => setB.has(n));
}

async function loadOneTensor(url, dataStart, modelId, dtype, name, info) {
  if (info.dtype !== "BF16") {
    throw new Error(`Tensor ${name} has unexpected dtype ${info.dtype} (expected BF16)`);
  }
  const [start, end] = info.offsets;
  const raw = await rangedFetch(url, dataStart + start, dataStart + end);
  const numElements = info.shape.reduce((a, b) => a * b, 1);
  const f32 = decodeBf16(raw, numElements);

  // Same decode path regardless of dtype flag; only the post-decode branch
  // differs (plan.md Phase 2: "same code path, one branch"). RMSNorm
  // weights (1D) always stay f32 -- quantizing them can visibly break
  // output for very little size saved (plan.md 1.3).
  let record;
  if (dtype === "i8" && info.shape.length === 2) {
    const { qweight, scales } = quantizeRowsInt8(f32, info.shape[0], info.shape[1]);
    record = { kind: "i8", shape: info.shape, qweight, scales };
  } else {
    record = { kind: "f32", shape: info.shape, f32 };
  }
  await putTensor(modelId, dtype, name, record);
}

// Orchestrates the whole Phase 2 flow: probe the header, validate it
// against config.json, then stream each tensor in with a small bounded
// concurrency (peak memory stays a handful of tensors, never the whole
// file -- required for mobile, see plan.md Phase 2). Skips the network
// entirely when a matching manifest is already cached.
export async function loadModelTensors(modelId, config, dtype, onProgress) {
  const url = resolveUrl(modelId, "model.safetensors");
  const remoteLength = await headContentLength(url);

  const expectedNames = [...expectedTensorShapes(config).keys()];
  const cachedManifest = await getManifest(modelId, dtype);
  if (
    cachedManifest &&
    cachedManifest.complete &&
    cachedManifest.sourceContentLength === remoteLength &&
    sameNameSet(cachedManifest.tensorNames, expectedNames)
  ) {
    return { cached: true, tensorNames: cachedManifest.tensorNames, remoteLength };
  }

  let probe = new Uint8Array(await rangedFetch(url, 0, HEADER_PROBE_BYTES));
  let header = parseHeader(probe);
  if (header.needsMoreBytes) {
    const rest = new Uint8Array(await rangedFetch(url, probe.length, header.needsMoreBytes));
    const combined = new Uint8Array(header.needsMoreBytes);
    combined.set(probe, 0);
    combined.set(rest, probe.length);
    probe = combined;
    header = parseHeader(probe);
  }

  validateHeader(header, config);

  const names = [...header.tensors.keys()];
  let bytesLoaded = 0;
  let nextIndex = 0;

  async function worker() {
    while (nextIndex < names.length) {
      const name = names[nextIndex++];
      const info = header.tensors.get(name);
      await loadOneTensor(url, header.dataStart, modelId, dtype, name, info);
      bytesLoaded += info.offsets[1] - info.offsets[0];
      if (onProgress) onProgress(bytesLoaded, remoteLength, name, nextIndex, names.length);
    }
  }

  const workerCount = Math.min(CONCURRENCY, names.length);
  await Promise.all(Array.from({ length: workerCount }, worker));

  await putManifest(modelId, dtype, {
    tensorNames: names,
    sourceContentLength: remoteLength,
    complete: true,
    downloadedAt: Date.now(),
  });

  return { cached: false, tensorNames: names, remoteLength };
}
