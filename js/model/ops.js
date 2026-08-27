// Core numeric kernels: matvec (f32 and int8 weights), embedding lookup,
// RMSNorm, RoPE, softmax, SiLU. Plain scalar loops throughout -- manual
// unrolling measured *slower* than the naive version (plan.md 1.6 / gotcha
// 18), so there is nothing clever here on purpose.

// No mean subtraction, no bias -- RMSNorm, not LayerNorm (gotcha 5). eps is
// applied inside the sqrt.
export function rmsnorm(out, x, weight, eps) {
  const n = x.length;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[i] * x[i];
  const scale = 1 / Math.sqrt(ss / n + eps);
  for (let i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

// out[j] = sum_k W[j*inDim+k] * x[k]. W is row-major [outDim, inDim]
// (gotcha 4: safetensors Linear layout, contiguous per output row).
export function matvecF32(out, W, x, outDim, inDim) {
  for (let j = 0; j < outDim; j++) {
    const rowStart = j * inDim;
    let acc = 0;
    for (let k = 0; k < inDim; k++) acc += W[rowStart + k] * x[k];
    out[j] = acc;
  }
}

// Same layout, per-row-quantized (plan.md 1.3): out[j] = scale[j] *
// sum_k qW[j*inDim+k] * x[k]. x stays f32; JS widens int8*f32 to double
// automatically, no overflow handling needed (unlike C).
export function matvecI8(out, qW, scales, x, outDim, inDim) {
  for (let j = 0; j < outDim; j++) {
    const rowStart = j * inDim;
    let acc = 0;
    for (let k = 0; k < inDim; k++) acc += qW[rowStart + k] * x[k];
    out[j] = acc * scales[j];
  }
}

// Dispatches on the tensor record shape produced by safetensors.js /
// store.js: { kind: 'f32', f32 } or { kind: 'i8', qweight, scales }. Using
// this everywhere means the forward pass is identical whether the model
// was loaded as f32 or int8 (plan.md Phase 6: "same code path, one
// branch").
export function linear(out, tensor, x, outDim, inDim) {
  if (tensor.kind === "f32") {
    matvecF32(out, tensor.f32, x, outDim, inDim);
  } else {
    matvecI8(out, tensor.qweight, tensor.scales, x, outDim, inDim);
  }
}

// Row `tok` of an [V, H] embedding tensor, dequantized if needed. Also
// used as the lm_head input weight is never read this way -- lm_head goes
// through linear() instead, since tie_word_embeddings makes it a matvec
// against the same table (gotcha 3), not a row lookup.
export function embedRow(out, tensor, tok, H) {
  const base = tok * H;
  if (tensor.kind === "f32") {
    out.set(tensor.f32.subarray(base, base + H));
  } else {
    const scale = tensor.scales[tok];
    for (let i = 0; i < H; i++) out[i] = tensor.qweight[base + i] * scale;
  }
}

// Precomputed per-position rotation angles, split-half convention (gotcha
// 2: pairs are (i, i + headDim/2), not (2i, 2i+1) -- this is also why
// GGUF was rejected, see plan.md 1.2). One theta^(-2i/headDim) frequency
// per pair, shared by every layer at a given position -- callers should
// compute this once per position, not once per layer.
export function ropeTables(pos, headDim, theta) {
  const half = headDim / 2;
  const cos = new Float32Array(half);
  const sin = new Float32Array(half);
  for (let i = 0; i < half; i++) {
    const freq = Math.pow(theta, (-2 * i) / headDim);
    const angle = pos * freq;
    cos[i] = Math.cos(angle);
    sin[i] = Math.sin(angle);
  }
  return { cos, sin };
}

// Rotates one head's slice of `vec` (in place) starting at `headOffset`.
export function applyRope(vec, headOffset, headDim, cosTab, sinTab) {
  const half = headDim / 2;
  for (let i = 0; i < half; i++) {
    const a = vec[headOffset + i];
    const b = vec[headOffset + i + half];
    const cos = cosTab[i];
    const sin = sinTab[i];
    vec[headOffset + i] = a * cos - b * sin;
    vec[headOffset + i + half] = a * sin + b * cos;
  }
}

// In-place softmax over out[0, len).
export function softmax(out, len) {
  let max = -Infinity;
  for (let i = 0; i < len; i++) if (out[i] > max) max = out[i];
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const e = Math.exp(out[i] - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < len; i++) out[i] /= sum;
}

export function silu(x) {
  return x / (1 + Math.exp(-x));
}
