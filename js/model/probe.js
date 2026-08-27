// Captures a fixed-size summary of every layer's activations during a
// decode step, for the background visualization (js/viz/stack.js).
//
// transformer.js's header calls this out as the reason its forward pass
// is kept as named ops in sequence rather than fused: observing
// intermediate activations was always meant to cost a hook, not a
// rewrite. The hook is two `if (probe)` branches in runWithCache().
//
// Everything is quantized to bytes and packed into one flat buffer per
// token so the whole frame crosses to the main thread as a single
// transferable ArrayBuffer: one postMessage, no structured-clone copy,
// no per-layer message storm (the worker does not yield between layers,
// so 30 separate messages would arrive in a burst anyway).
//
// WHAT IS AND IS NOT SUBSAMPLED. The MLP and residual vectors are small
// enough to send whole. The attention matrix -- heads x context -- is the
// one genuinely 2D thing a decode step computes, and it is sent whole
// too: every (head, position) weight the model actually used gets its own
// byte, so the renderer can put one screen pixel on one matrix entry. The
// price is a frame that grows with the conversation, which is why the
// buffer is sized per token from the live context length rather than
// fixed at the ceiling: at position 200 a 135M frame is 115KB, not the
// 290KB a maxCtx-sized allocation would cost every token from the first.

// Ceilings, not expected sizes: SmolLM2's intermediate_size is 1536
// (135M) / 2560 (360M) and hidden_size 576 / 960, so nothing is actually
// subsampled for either model. They exist so a larger model degrades to a
// coarser picture instead of a megabyte-per-token firehose.
const MAX_MLP_CELLS = 4096;
const MAX_RESID_CELLS = 2048;

// Likewise a ceiling. The worker caps context at 1024, so at present this
// never bites and attention is captured at true 1:1. Above it, columns are
// max-pooled rather than dropped.
const MAX_ATTN_COLS = 1024;

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

// One query head's post-softmax distribution over the whole context ->
// one row of bytes. `src` is the live `scores` buffer inside the attention
// loop, which the next head overwrites, so this has to run per head rather
// than once per layer.
//
// TWO CHOICES WORTH THE WORDS.
//
// Normalized per head, by that head's own peak. Heads differ in
// concentration by orders of magnitude -- some put 0.9 of their mass on
// one position, others spread 0.002 across five hundred -- so a shared
// scale renders every diffuse head as a black row. Own-peak is the same
// argument encodeSigned() makes per layer, and what it buys is that each
// row shows *where that head is looking*, which is the question attention
// answers.
//
// Square-rooted in the quantizer, not in the palette. Real attention is
// dominated by the sink at position 0, and under a linear 8-bit scale the
// entire rest of the row -- the local window, the induction spikes, the
// actual content of the picture -- lands in bytes 0-3 and is gone before
// the renderer ever sees it. Shaping here is a precision decision;
// shaping in the palette could not recover what quantization threw away.
// `+ 0.5 | 0` rather than a bare truncation, and not only for accuracy:
// k is 1/peak, so peak*k is 1 give or take one ulp, and on the wrong side
// of it a truncating cast puts the peak at 254 instead of 255. Everything
// downstream -- the renderer's own-peak assumption, the test that checks
// it -- would then be quietly off by a code on some rows and not others.
const q255 = (x) => (255 * Math.sqrt(x) + 0.5) | 0;

function encodeAttentionRow(out, off, src, count, cols) {
  let peak = 1e-9;
  for (let j = 0; j < count; j++) if (src[j] > peak) peak = src[j];
  const k = 1 / peak;
  if (count <= cols) {
    for (let j = 0; j < count; j++) out[off + j] = q255(src[j] * k);
    return;
  }
  // Pooled with max, not mean: attention onto a long context is spiky, and
  // averaging smears every spike into the background. Max keeps "this head
  // is staring at token 14" visible, which is the only thing worth showing.
  const per = count / cols;
  for (let c = 0; c < cols; c++) {
    const lo = (c * per) | 0;
    const hi = c === cols - 1 ? count : ((c + 1) * per) | 0;
    let v = 0;
    for (let j = lo; j < hi; j++) if (src[j] > v) v = src[j];
    out[off + c] = q255(v * k);
  }
}

export function createProbe(config, maxCtx) {
  const layers = config.num_hidden_layers;
  const H = config.hidden_size;
  const I = config.intermediate_size;
  const heads = config.num_attention_heads;

  const mlpCells = Math.min(I, MAX_MLP_CELLS);
  const residCells = Math.min(H, MAX_RESID_CELLS);
  const maxCols = Math.min(maxCtx, MAX_ATTN_COLS);
  // Where a layer's attention block starts, relative to that layer's base.
  const attnBase = mlpCells + residCells;

  let bytes = null;
  let cols = 0;
  let stride = 0;

  return {
    // Sent once, in the worker's "ready" message: the renderer needs the
    // shape of a frame to reshape it into grids, and it never changes
    // while a model is resident. `stride` is *not* here -- it depends on
    // the context length and so travels with each frame instead.
    geometry: { layers, mlpCells, residCells, heads, maxCols, attnBase },

    // `count` is the number of cached positions this token will attend
    // over, i.e. pos + 1. Known before the pass starts, and it fixes the
    // frame's width.
    beginToken(count) {
      cols = Math.min(Math.max(count, 1), maxCols);
      stride = attnBase + heads * cols;
      // Fresh each token because the previous one's buffer was
      // transferred away and is now detached.
      bytes = new Uint8Array(layers * stride);
    },

    // Called inside the attention loop, once per head, while `scores`
    // still holds this head's distribution.
    captureHead(l, h, scores, count) {
      if (!bytes) return;
      encodeAttentionRow(bytes, l * stride + attnBase + h * cols, scores, count, cols);
    },

    // Called at the end of each layer's body, where `resid` is the
    // residual stream leaving the layer and `mlp` still holds
    // silu(gate)*up -- the closest thing a transformer has to "neurons
    // firing", and the vector interpretability work actually calls
    // neurons.
    captureLayer(l, resid, mlp) {
      if (!bytes) return;
      const base = l * stride;
      encodeSigned(bytes, base, mlp, I, mlpCells);
      encodeSigned(bytes, base + mlpCells, resid, H, residCells);
    },

    endToken() {
      const out = bytes;
      bytes = null;
      return { bytes: out, cols, stride };
    },
  };
}
