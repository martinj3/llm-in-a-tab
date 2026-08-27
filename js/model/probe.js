// Captures a fixed-size summary of every layer's activations during a
// decode step, for the background visualization (js/viz/stack.js).
//
// transformer.js's header calls this out as the reason its forward pass
// is kept as named ops in sequence rather than fused: observing
// intermediate activations was always meant to cost a hook, not a
// rewrite. The hook is three `if (probe)` branches in runWithCache().
//
// The work added per token is ~2k float reads and ~2k byte writes per
// layer against a forward pass doing ~135M multiply-accumulates -- about
// 0.05%, and it is skipped entirely when the visualization is off.
//
// Everything is quantized to bytes and packed into one flat buffer per
// token so the whole frame crosses to the main thread as a single
// transferable ArrayBuffer: one postMessage, no structured-clone copy,
// no per-layer message storm (the worker does not yield between layers,
// so 30 separate messages would arrive in a burst anyway).

const ATTN_BINS = 128;

// Ceilings, not expected sizes: SmolLM2's intermediate_size is 1536
// (135M) / 2560 (360M) and hidden_size 576 / 960, so nothing is actually
// subsampled for either model. They exist so a larger model degrades to a
// coarser picture instead of a megabyte-per-token firehose.
const MAX_MLP_CELLS = 4096;
const MAX_RESID_CELLS = 2048;

// Signed activations -> bytes with 128 meaning zero, normalized by this
// vector's own peak magnitude.
//
// Per-layer normalization rather than one shared scale is deliberate:
// residual-stream magnitudes in a pre-norm transformer grow by more than
// an order of magnitude from layer 0 to the last layer, so a global scale
// would render the early layers as a uniformly black field and clip the
// late ones. Each layer is shown against its own dynamic range, which is
// what makes the *pattern* legible; absolute magnitude is carried
// separately by the per-layer energy the renderer computes.
function encodeSigned(out, off, src, n, cells) {
  const step = n / cells;
  let peak = 1e-9;
  for (let c = 0; c < cells; c++) {
    const v = src[(c * step) | 0];
    const a = v < 0 ? -v : v;
    if (a > peak) peak = a;
  }
  const k = 127 / peak;
  for (let c = 0; c < cells; c++) {
    let q = Math.round(src[(c * step) | 0] * k);
    if (q > 127) q = 127;
    else if (q < -127) q = -127;
    out[off + c] = 128 + q;
  }
}

// Attention weights are already non-negative and sum to nH (the scratch
// holds a sum over heads, not a mean -- normalizing by the peak below
// makes the difference irrelevant).
//
// Binned with max rather than mean: attention onto a long context is
// spiky, and averaging a 1000-position context down to 128 bins smears
// every spike into the background. Max keeps "this layer is staring at
// token 14" visible, which is the only thing worth showing.
function encodeAttention(out, off, src, count, scratch) {
  const bins = Math.min(count, ATTN_BINS);
  const per = count / bins;
  let peak = 1e-9;
  for (let b = 0; b < bins; b++) {
    const lo = (b * per) | 0;
    const hi = b === bins - 1 ? count : ((b + 1) * per) | 0;
    let v = 0;
    for (let j = lo; j < hi; j++) if (src[j] > v) v = src[j];
    scratch[b] = v;
    if (v > peak) peak = v;
  }
  const k = 255 / peak;
  for (let b = 0; b < bins; b++) out[off + b] = Math.round(scratch[b] * k);
  return bins;
}

export function createProbe(config, maxCtx) {
  const layers = config.num_hidden_layers;
  const H = config.hidden_size;
  const I = config.intermediate_size;

  const mlpCells = Math.min(I, MAX_MLP_CELLS);
  const residCells = Math.min(H, MAX_RESID_CELLS);
  const stride = mlpCells + residCells + ATTN_BINS;

  // runWithCache() zeroes this, accumulates each head's post-softmax
  // weights into it, and captureLayer() reads it back. It lives here
  // rather than in the transformer so the transformer owns no
  // visualization state.
  const attn = new Float32Array(maxCtx);
  const binScratch = new Float32Array(ATTN_BINS);

  let bytes = null;
  let attnUsed = 0;

  return {
    // Sent once, in the worker's "ready" message: the renderer needs the
    // shape of a frame to reshape it into grids, and it never changes
    // while a model is resident.
    geometry: { layers, mlpCells, residCells, attnBins: ATTN_BINS, stride },

    attn,

    beginToken() {
      // Fresh each token because the previous one's buffer was
      // transferred away and is now detached. 67KB per token at a few
      // tokens per second is nothing to allocate.
      bytes = new Uint8Array(layers * stride);
      attnUsed = 0;
    },

    // Called at the end of each layer's body, where `resid` is the
    // residual stream leaving the layer and `mlp` still holds
    // silu(gate)*up -- the closest thing a transformer has to "neurons
    // firing", and the vector interpretability work actually calls
    // neurons.
    captureLayer(l, resid, mlp, count) {
      if (!bytes) return;
      const base = l * stride;
      encodeSigned(bytes, base, mlp, I, mlpCells);
      encodeSigned(bytes, base + mlpCells, resid, H, residCells);
      attnUsed = encodeAttention(bytes, base + mlpCells + residCells, attn, count, binScratch);
    },

    endToken() {
      const out = bytes;
      bytes = null;
      return { bytes: out, attnUsed };
    },
  };
}
