// The forward pass: embedding lookup, then per layer (RMSNorm -> QKV ->
// RoPE -> causal attention -> output proj -> residual -> RMSNorm -> SwiGLU
// MLP -> residual), then final norm and the tied lm_head projection.
//
// Phase 5's `forward()` below is a single full-sequence pass with no KV
// cache -- kept as-is because tests/golden-forward.mjs verifies against it
// directly and it's the simplest version to read. Phase 6 adds
// `prefill()`/`decodeStep()`, which share `runWithCache()`: same per-layer
// math, but K/V read from and write into a persistent kvcache.js buffer
// instead of a per-call local array, so a multi-turn conversation's cache
// survives across calls (plan.md gotcha 22).
//
// Kept as named ops called in sequence, not fused into one closure, so
// later phases can observe intermediate activations without a rewrite
// (plan.md section 4's design constraint).
import { rmsnorm, linear, linearB4, embedRow, ropeTables, applyRope, softmax, silu } from "./ops.js";

function getTensor(tensors, name) {
  const t = tensors.get(name);
  if (!t) throw new Error(`Missing tensor: ${name}`);
  return t;
}

// tensors: Map<name, record> in the { kind: 'f32'|'i8', ... } shape
// produced by safetensors.js / store.js.
// config: parsed+validated config.json.
// tokenIds: token ids for the whole prompt.
// Returns Float32Array[vocab_size]: logits at the final position.
export function forward(tensors, config, tokenIds) {
  const H = config.hidden_size;
  const L = config.num_hidden_layers;
  const nH = config.num_attention_heads;
  const nKV = config.num_key_value_heads;
  const headDim = H / nH;
  const ratio = nH / nKV; // GQA: `ratio` query heads share each KV head (gotcha 6)
  const I = config.intermediate_size;
  const V = config.vocab_size;
  const eps = config.rms_norm_eps;
  const theta = config.rope_theta;
  const T = tokenIds.length;

  const embed = getTensor(tensors, "model.embed_tokens.weight");

  const hidden = [];
  for (let t = 0; t < T; t++) {
    const row = new Float32Array(H);
    embedRow(row, embed, tokenIds[t], H);
    hidden.push(row);
  }

  // Shared across every layer at a given position (RoPE angles don't
  // depend on the layer), so computed once here rather than per layer.
  const ropeCache = tokenIds.map((_, t) => ropeTables(t, headDim, theta));

  const normBuf = new Float32Array(H);
  const qBuf = new Float32Array(nH * headDim);
  const attnOut = new Float32Array(nH * headDim);
  const oBuf = new Float32Array(H);
  const gateBuf = new Float32Array(I);
  const upBuf = new Float32Array(I);
  const mlpOut = new Float32Array(H);
  const scores = new Float32Array(T);
  const invSqrtHeadDim = 1 / Math.sqrt(headDim);

  for (let l = 0; l < L; l++) {
    const p = `model.layers.${l}`;
    const wIn = getTensor(tensors, `${p}.input_layernorm.weight`);
    const wQ = getTensor(tensors, `${p}.self_attn.q_proj.weight`);
    const wK = getTensor(tensors, `${p}.self_attn.k_proj.weight`);
    const wV = getTensor(tensors, `${p}.self_attn.v_proj.weight`);
    const wO = getTensor(tensors, `${p}.self_attn.o_proj.weight`);
    const wPostAttn = getTensor(tensors, `${p}.post_attention_layernorm.weight`);
    const wGate = getTensor(tensors, `${p}.mlp.gate_proj.weight`);
    const wUp = getTensor(tensors, `${p}.mlp.up_proj.weight`);
    const wDown = getTensor(tensors, `${p}.mlp.down_proj.weight`);

    // K/V accumulate left-to-right within this layer: at position t we
    // already have K/V for 0..t-1 and only need to add t's before
    // attending, since attention is causal (gotcha 1: K is stored
    // post-RoPE, rotated once here rather than re-rotated every step).
    const allK = [];
    const allV = [];

    for (let t = 0; t < T; t++) {
      rmsnorm(normBuf, hidden[t], wIn.f32, eps);

      linear(qBuf, wQ, normBuf, nH * headDim, H);
      const k = new Float32Array(nKV * headDim);
      const v = new Float32Array(nKV * headDim);
      linear(k, wK, normBuf, nKV * headDim, H);
      linear(v, wV, normBuf, nKV * headDim, H);

      const { cos, sin } = ropeCache[t];
      for (let h = 0; h < nH; h++) applyRope(qBuf, h * headDim, headDim, cos, sin);
      for (let kvh = 0; kvh < nKV; kvh++) applyRope(k, kvh * headDim, headDim, cos, sin);
      allK.push(k);
      allV.push(v);

      for (let h = 0; h < nH; h++) {
        const kvh = (h / ratio) | 0;
        const qOff = h * headDim;
        const kvOff = kvh * headDim;
        for (let j = 0; j <= t; j++) {
          const kRow = allK[j];
          let dot = 0;
          for (let d = 0; d < headDim; d++) dot += qBuf[qOff + d] * kRow[kvOff + d];
          scores[j] = dot * invSqrtHeadDim;
        }
        softmax(scores, t + 1);
        const outOff = h * headDim;
        for (let d = 0; d < headDim; d++) attnOut[outOff + d] = 0;
        for (let j = 0; j <= t; j++) {
          const w = scores[j];
          const vRow = allV[j];
          for (let d = 0; d < headDim; d++) attnOut[outOff + d] += w * vRow[kvOff + d];
        }
      }

      linear(oBuf, wO, attnOut, H, nH * headDim);
      for (let i = 0; i < H; i++) hidden[t][i] += oBuf[i];

      rmsnorm(normBuf, hidden[t], wPostAttn.f32, eps);
      linear(gateBuf, wGate, normBuf, I, H);
      linear(upBuf, wUp, normBuf, I, H);
      for (let i = 0; i < I; i++) gateBuf[i] = silu(gateBuf[i]) * upBuf[i];
      linear(mlpOut, wDown, gateBuf, H, I);
      for (let i = 0; i < H; i++) hidden[t][i] += mlpOut[i];
    }
  }

  const finalNorm = getTensor(tensors, "model.norm.weight");
  const normed = new Float32Array(H);
  rmsnorm(normed, hidden[T - 1], finalNorm.f32, eps);

  // lm_head is the tied embedding table (gotcha 3): logits[v] is the dot
  // product of the final hidden state with embedding row v, which is
  // exactly what linear() computes against an [V, H] weight.
  const logits = new Float32Array(V);
  linear(logits, embed, normed, V, H);
  return logits;
}

// Shared by prefill() and decodeStep(): runs `tokenIds.length` new tokens
// through every layer, appending their K/V into `cache` at
// cache.seqLen, cache.seqLen+1, ... and attending against everything
// already in the cache plus what this call writes (causal, since a
// position's attention only ever reads positions <= itself, which have
// already been written earlier in the same j loop or in a prior call).
// K is rotated once at write time and stored post-RoPE (gotcha 1); Q is
// rotated fresh per call. Advances cache.seqLen exactly once, after every
// layer has finished writing every new token, whether tokenIds.length is
// the whole prompt (prefill) or a single generated token (decode) --
// gotcha 20/21.
// Returns logits at the final new position only (the only one a decode
// loop ever needs).
//
// `probe` is optional (js/model/probe.js). When present it is handed each
// layer's activations on the way past, for the background visualization.
// Only decodeStep() passes one -- prefillBatched() is the perf-critical
// path verified against prefillSequential() by tests/prefill-batch.mjs and
// is left alone, and a prefill is over in well under a second anyway.
function runWithCache(tensors, config, cache, tokenIds, probe = null) {
  const H = config.hidden_size;
  const L = config.num_hidden_layers;
  const nH = config.num_attention_heads;
  const nKV = config.num_key_value_heads;
  const headDim = H / nH;
  const ratio = nH / nKV;
  const I = config.intermediate_size;
  const V = config.vocab_size;
  const eps = config.rms_norm_eps;
  const theta = config.rope_theta;
  const T = tokenIds.length;
  const startPos = cache.seqLen;

  if (startPos + T > cache.maxCtx) {
    throw new Error(`KV cache full: ${startPos + T} exceeds maxCtx ${cache.maxCtx}`);
  }

  const embed = getTensor(tensors, "model.embed_tokens.weight");

  const hidden = [];
  for (let i = 0; i < T; i++) {
    const row = new Float32Array(H);
    embedRow(row, embed, tokenIds[i], H);
    hidden.push(row);
  }

  const ropeCache = tokenIds.map((_, i) => ropeTables(startPos + i, headDim, theta));

  const normBuf = new Float32Array(H);
  const qBuf = new Float32Array(nH * headDim);
  const kBuf = new Float32Array(nKV * headDim);
  const vBuf = new Float32Array(nKV * headDim);
  const attnOut = new Float32Array(nH * headDim);
  const oBuf = new Float32Array(H);
  const gateBuf = new Float32Array(I);
  const upBuf = new Float32Array(I);
  const mlpOut = new Float32Array(H);
  const scores = new Float32Array(cache.maxCtx);
  const invSqrtHeadDim = 1 / Math.sqrt(headDim);
  const kvStride = nKV * headDim;

  for (let l = 0; l < L; l++) {
    const p = `model.layers.${l}`;
    const wIn = getTensor(tensors, `${p}.input_layernorm.weight`);
    const wQ = getTensor(tensors, `${p}.self_attn.q_proj.weight`);
    const wK = getTensor(tensors, `${p}.self_attn.k_proj.weight`);
    const wV = getTensor(tensors, `${p}.self_attn.v_proj.weight`);
    const wO = getTensor(tensors, `${p}.self_attn.o_proj.weight`);
    const wPostAttn = getTensor(tensors, `${p}.post_attention_layernorm.weight`);
    const wGate = getTensor(tensors, `${p}.mlp.gate_proj.weight`);
    const wUp = getTensor(tensors, `${p}.mlp.up_proj.weight`);
    const wDown = getTensor(tensors, `${p}.mlp.down_proj.weight`);
    const layerCache = cache.layers[l];

    for (let i = 0; i < T; i++) {
      const pos = startPos + i;
      rmsnorm(normBuf, hidden[i], wIn.f32, eps);

      linear(qBuf, wQ, normBuf, nH * headDim, H);
      linear(kBuf, wK, normBuf, nKV * headDim, H);
      linear(vBuf, wV, normBuf, nKV * headDim, H);

      const { cos, sin } = ropeCache[i];
      for (let h = 0; h < nH; h++) applyRope(qBuf, h * headDim, headDim, cos, sin);
      for (let kvh = 0; kvh < nKV; kvh++) applyRope(kBuf, kvh * headDim, headDim, cos, sin);

      const writeOff = pos * kvStride;
      layerCache.k.set(kBuf, writeOff);
      layerCache.v.set(vBuf, writeOff);

      if (probe) probe.attn.fill(0, 0, pos + 1);

      for (let h = 0; h < nH; h++) {
        const kvh = (h / ratio) | 0;
        const qOff = h * headDim;
        const kvHeadOff = kvh * headDim;
        for (let j = 0; j <= pos; j++) {
          const kBase = j * kvStride + kvHeadOff;
          let dot = 0;
          for (let d = 0; d < headDim; d++) dot += qBuf[qOff + d] * layerCache.k[kBase + d];
          scores[j] = dot * invSqrtHeadDim;
        }
        softmax(scores, pos + 1);
        // Summed across heads here, while `scores` is still this head's
        // distribution -- one pass over pos+1 floats against the
        // pos*headDim*2 the surrounding loop already does, so under 1%.
        if (probe) {
          const acc = probe.attn;
          for (let j = 0; j <= pos; j++) acc[j] += scores[j];
        }
        const outOff = h * headDim;
        for (let d = 0; d < headDim; d++) attnOut[outOff + d] = 0;
        for (let j = 0; j <= pos; j++) {
          const w = scores[j];
          const vBase = j * kvStride + kvHeadOff;
          for (let d = 0; d < headDim; d++) attnOut[outOff + d] += w * layerCache.v[vBase + d];
        }
      }

      linear(oBuf, wO, attnOut, H, nH * headDim);
      for (let d = 0; d < H; d++) hidden[i][d] += oBuf[d];

      rmsnorm(normBuf, hidden[i], wPostAttn.f32, eps);
      linear(gateBuf, wGate, normBuf, I, H);
      linear(upBuf, wUp, normBuf, I, H);
      for (let d = 0; d < I; d++) gateBuf[d] = silu(gateBuf[d]) * upBuf[d];
      linear(mlpOut, wDown, gateBuf, H, I);
      for (let d = 0; d < H; d++) hidden[i][d] += mlpOut[d];

      // After the residual add, so `hidden[i]` is the stream *leaving*
      // this layer, and before the next iteration overwrites gateBuf.
      if (probe) probe.captureLayer(l, hidden[i], gateBuf, pos + 1);
    }
  }

  cache.seqLen = startPos + T;

  const finalNorm = getTensor(tensors, "model.norm.weight");
  const normed = new Float32Array(H);
  rmsnorm(normed, hidden[T - 1], finalNorm.f32, eps);

  const logits = new Float32Array(V);
  linear(logits, embed, normed, V, H);
  return logits;
}

// Sequential (unbatched) prefill: processes a whole prompt one token at a
// time through runWithCache. Kept as its own export purely so
// tests/prefill-batch.mjs can assert prefillBatched() produces an
// identical KV cache and logits (plan.md Phase 7's exit criterion) --
// production code should call prefill(), not this.
export function prefillSequential(tensors, config, cache, tokenIds) {
  return runWithCache(tensors, config, cache, tokenIds);
}

// Processes exactly one new token -- the generation-loop step -- appending
// it to the cache at cache.seqLen exactly like a prompt token (gotcha 21).
// Batching needs >1 token to pay for itself, so decode always goes through
// the same unbatched path as a length-1 prefill.
export function decodeStep(tensors, config, cache, tokenId, probe = null) {
  return runWithCache(tensors, config, cache, [tokenId], probe);
}

const BATCH = 4;

// Batched prefill (plan.md Phase 7): processes new tokens BATCH-at-a-time
// through each layer's linear projections (q/k/v, o_proj, gate/up/down),
// reusing each weight row across up to 4 tokens before moving to the next
// row -- the one place batching beats loop overhead, because each weight
// byte is read once and reused (plan.md section 3, ~1.9x measured). RoPE,
// attention, and the SiLU*up elementwise step stay per-token: there is no
// shared weight to reuse there, so nothing to batch.
//
// A chunk smaller than BATCH (the final remainder) is padded with a fixed
// all-zero buffer for the unused slots so the kernel's inner loop never
// branches on chunk size (gotcha 17) -- their outputs are simply never
// read back into `hidden` or the KV cache.
export function prefillBatched(tensors, config, cache, tokenIds) {
  const H = config.hidden_size;
  const L = config.num_hidden_layers;
  const nH = config.num_attention_heads;
  const nKV = config.num_key_value_heads;
  const headDim = H / nH;
  const ratio = nH / nKV;
  const I = config.intermediate_size;
  const V = config.vocab_size;
  const eps = config.rms_norm_eps;
  const theta = config.rope_theta;
  const T = tokenIds.length;
  const startPos = cache.seqLen;

  if (startPos + T > cache.maxCtx) {
    throw new Error(`KV cache full: ${startPos + T} exceeds maxCtx ${cache.maxCtx}`);
  }

  const embed = getTensor(tensors, "model.embed_tokens.weight");
  const hidden = [];
  for (let i = 0; i < T; i++) {
    const row = new Float32Array(H);
    embedRow(row, embed, tokenIds[i], H);
    hidden.push(row);
  }
  const ropeCache = tokenIds.map((_, i) => ropeTables(startPos + i, headDim, theta));

  const zeroH = new Float32Array(H); // fixed padding input; never written
  const zeroI = new Float32Array(I); // fixed padding input; never written
  const normBuf = Array.from({ length: BATCH }, () => new Float32Array(H));
  const qBuf = Array.from({ length: BATCH }, () => new Float32Array(nH * headDim));
  const kBuf = Array.from({ length: BATCH }, () => new Float32Array(nKV * headDim));
  const vBuf = Array.from({ length: BATCH }, () => new Float32Array(nKV * headDim));
  const attnOut = Array.from({ length: BATCH }, () => new Float32Array(nH * headDim));
  const oBuf = Array.from({ length: BATCH }, () => new Float32Array(H));
  const gateBuf = Array.from({ length: BATCH }, () => new Float32Array(I));
  const upBuf = Array.from({ length: BATCH }, () => new Float32Array(I));
  const mlpOut = Array.from({ length: BATCH }, () => new Float32Array(H));
  const scores = new Float32Array(cache.maxCtx);
  const invSqrtHeadDim = 1 / Math.sqrt(headDim);
  const kvStride = nKV * headDim;

  for (let l = 0; l < L; l++) {
    const p = `model.layers.${l}`;
    const wIn = getTensor(tensors, `${p}.input_layernorm.weight`);
    const wQ = getTensor(tensors, `${p}.self_attn.q_proj.weight`);
    const wK = getTensor(tensors, `${p}.self_attn.k_proj.weight`);
    const wV = getTensor(tensors, `${p}.self_attn.v_proj.weight`);
    const wO = getTensor(tensors, `${p}.self_attn.o_proj.weight`);
    const wPostAttn = getTensor(tensors, `${p}.post_attention_layernorm.weight`);
    const wGate = getTensor(tensors, `${p}.mlp.gate_proj.weight`);
    const wUp = getTensor(tensors, `${p}.mlp.up_proj.weight`);
    const wDown = getTensor(tensors, `${p}.mlp.down_proj.weight`);
    const layerCache = cache.layers[l];

    for (let chunkStart = 0; chunkStart < T; chunkStart += BATCH) {
      const n = Math.min(BATCH, T - chunkStart);

      for (let s = 0; s < n; s++) rmsnorm(normBuf[s], hidden[chunkStart + s], wIn.f32, eps);
      const n0 = 0 < n ? normBuf[0] : zeroH;
      const n1 = 1 < n ? normBuf[1] : zeroH;
      const n2 = 2 < n ? normBuf[2] : zeroH;
      const n3 = 3 < n ? normBuf[3] : zeroH;
      linearB4(qBuf[0], qBuf[1], qBuf[2], qBuf[3], wQ, n0, n1, n2, n3, nH * headDim, H);
      linearB4(kBuf[0], kBuf[1], kBuf[2], kBuf[3], wK, n0, n1, n2, n3, nKV * headDim, H);
      linearB4(vBuf[0], vBuf[1], vBuf[2], vBuf[3], wV, n0, n1, n2, n3, nKV * headDim, H);

      for (let s = 0; s < n; s++) {
        const i = chunkStart + s;
        const pos = startPos + i;
        const { cos, sin } = ropeCache[i];
        for (let h = 0; h < nH; h++) applyRope(qBuf[s], h * headDim, headDim, cos, sin);
        for (let kvh = 0; kvh < nKV; kvh++) applyRope(kBuf[s], kvh * headDim, headDim, cos, sin);

        const writeOff = pos * kvStride;
        layerCache.k.set(kBuf[s], writeOff);
        layerCache.v.set(vBuf[s], writeOff);

        for (let h = 0; h < nH; h++) {
          const kvh = (h / ratio) | 0;
          const qOff = h * headDim;
          const kvHeadOff = kvh * headDim;
          for (let j = 0; j <= pos; j++) {
            const kBase = j * kvStride + kvHeadOff;
            let dot = 0;
            for (let d = 0; d < headDim; d++) dot += qBuf[s][qOff + d] * layerCache.k[kBase + d];
            scores[j] = dot * invSqrtHeadDim;
          }
          softmax(scores, pos + 1);
          const outOff = h * headDim;
          for (let d = 0; d < headDim; d++) attnOut[s][outOff + d] = 0;
          for (let j = 0; j <= pos; j++) {
            const w = scores[j];
            const vBase = j * kvStride + kvHeadOff;
            for (let d = 0; d < headDim; d++) attnOut[s][outOff + d] += w * layerCache.v[vBase + d];
          }
        }
      }

      const a0 = 0 < n ? attnOut[0] : zeroH; // nH*headDim === H, so zeroH pads this too
      const a1 = 1 < n ? attnOut[1] : zeroH;
      const a2 = 2 < n ? attnOut[2] : zeroH;
      const a3 = 3 < n ? attnOut[3] : zeroH;
      linearB4(oBuf[0], oBuf[1], oBuf[2], oBuf[3], wO, a0, a1, a2, a3, H, nH * headDim);

      for (let s = 0; s < n; s++) {
        const hrow = hidden[chunkStart + s];
        for (let d = 0; d < H; d++) hrow[d] += oBuf[s][d];
        rmsnorm(normBuf[s], hrow, wPostAttn.f32, eps);
      }
      const m0 = 0 < n ? normBuf[0] : zeroH;
      const m1 = 1 < n ? normBuf[1] : zeroH;
      const m2 = 2 < n ? normBuf[2] : zeroH;
      const m3 = 3 < n ? normBuf[3] : zeroH;
      linearB4(gateBuf[0], gateBuf[1], gateBuf[2], gateBuf[3], wGate, m0, m1, m2, m3, I, H);
      linearB4(upBuf[0], upBuf[1], upBuf[2], upBuf[3], wUp, m0, m1, m2, m3, I, H);
      for (let s = 0; s < n; s++) {
        for (let d = 0; d < I; d++) gateBuf[s][d] = silu(gateBuf[s][d]) * upBuf[s][d];
      }
      const g0 = 0 < n ? gateBuf[0] : zeroI;
      const g1 = 1 < n ? gateBuf[1] : zeroI;
      const g2 = 2 < n ? gateBuf[2] : zeroI;
      const g3 = 3 < n ? gateBuf[3] : zeroI;
      linearB4(mlpOut[0], mlpOut[1], mlpOut[2], mlpOut[3], wDown, g0, g1, g2, g3, H, I);
      for (let s = 0; s < n; s++) {
        const hrow = hidden[chunkStart + s];
        for (let d = 0; d < H; d++) hrow[d] += mlpOut[s][d];
      }
    }
  }

  cache.seqLen = startPos + T;

  const finalNorm = getTensor(tensors, "model.norm.weight");
  const normed = new Float32Array(H);
  rmsnorm(normed, hidden[T - 1], finalNorm.f32, eps);

  const logits = new Float32Array(V);
  linear(logits, embed, normed, V, H);
  return logits;
}

// Processes a whole prompt (or, for a persisted multi-turn cache, just the
// new turn's tokens -- gotcha 22) and returns logits at the last position.
// Production entry point -- batched (Phase 7), verified equivalent to
// prefillSequential() by tests/prefill-batch.mjs.
export function prefill(tensors, config, cache, tokenIds) {
  return prefillBatched(tensors, config, cache, tokenIds);
}
