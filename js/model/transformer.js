// The forward pass: embedding lookup, then per layer (RMSNorm -> QKV ->
// RoPE -> causal attention -> output proj -> residual -> RMSNorm -> SwiGLU
// MLP -> residual), then final norm and the tied lm_head projection.
//
// Phase 5 scope (plan.md): a single full-sequence forward pass over the
// whole prompt, no KV cache -- that arrives in Phase 6 for the decode
// loop. Kept as named ops called in sequence, not fused into one closure,
// so later phases can observe intermediate activations without a rewrite
// (plan.md section 4's design constraint).
import { rmsnorm, linear, embedRow, ropeTables, applyRope, softmax, silu } from "./ops.js";

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
