// KV cache: allocation and indexing only (plan.md 1.4). Position-major
// layout, `[pos][kv_head][head_dim]` -- appending a token is one contiguous
// write, and GQA reads a kv-head's slice without materializing the
// expanded per-query-head K/V (gotcha 6). Preallocated per layer at
// `maxCtx`; `seqLen` is the single fill counter transformer.js advances
// once per call, after every layer has written that call's tokens
// (gotcha 20: appending and advancing position must not be able to
// diverge, so nothing outside transformer.js's forward loop ever writes
// into these arrays or touches seqLen).
export function createKVCache(config, maxCtx) {
  const nKV = config.num_key_value_heads;
  const headDim = config.hidden_size / config.num_attention_heads;
  const L = config.num_hidden_layers;
  const layers = Array.from({ length: L }, () => ({
    k: new Float32Array(maxCtx * nKV * headDim),
    v: new Float32Array(maxCtx * nKV * headDim),
  }));
  return { layers, maxCtx, nKV, headDim, seqLen: 0 };
}
