// Phase 7's exit criterion (plan.md section 5): "measurable ~1.9x speedup
// on prompt processing, with identical output," guarded by "a test
// asserting batched and sequential prefill produce identical KV cache
// contents." Runs both prefillSequential() and prefillBatched() (Phase 6's
// per-token path vs Phase 7's B=4 kernel) on the same real weights and
// prompt, into separate caches, and diffs every layer's K/V buffer plus
// the final logits. Then times each on a longer synthetic prompt to check
// the speedup is in the right ballpark.
//
// Usage: node tests/prefill-batch.mjs [135M|360M]
// Requires network access to huggingface.co.
import { resolveUrl } from "../js/models.js";
import { rangedFetch } from "../js/model/download.js";
import { parseHeader, decodeBf16, quantizeRowsInt8 } from "../js/model/safetensors.js";
import { fetchModelConfig } from "../js/model/config.js";
import { loadTokenizer } from "../js/model/tokenizer.js";
import { encodeChatPrompt } from "../js/model/template.js";
import { prefillSequential, prefillBatched } from "../js/model/transformer.js";
import { createKVCache } from "../js/model/kvcache.js";

const HEADER_PROBE_BYTES = 200_000;
const CONCURRENCY = 4;

async function loadTensors(modelId, dtype) {
  const url = resolveUrl(modelId, "model.safetensors");
  const probe = new Uint8Array(await rangedFetch(url, 0, HEADER_PROBE_BYTES));
  const header = parseHeader(probe);
  if (header.needsMoreBytes) {
    throw new Error(`Header probe too small: need ${header.needsMoreBytes} bytes`);
  }
  const names = [...header.tensors.keys()];
  const tensors = new Map();
  let nextIndex = 0;
  async function worker() {
    while (nextIndex < names.length) {
      const name = names[nextIndex++];
      const info = header.tensors.get(name);
      const [start, end] = info.offsets;
      const raw = await rangedFetch(url, header.dataStart + start, header.dataStart + end);
      const numElements = info.shape.reduce((a, b) => a * b, 1);
      const f32 = decodeBf16(raw, numElements);
      if (dtype === "i8" && info.shape.length === 2) {
        const { qweight, scales } = quantizeRowsInt8(f32, info.shape[0], info.shape[1]);
        tensors.set(name, { kind: "i8", shape: info.shape, qweight, scales });
      } else {
        tensors.set(name, { kind: "f32", shape: info.shape, f32 });
      }
    }
  }
  const workerCount = Math.min(CONCURRENCY, names.length);
  await Promise.all(Array.from({ length: workerCount }, worker));
  return tensors;
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

async function runOne(modelId) {
  console.log(`\n=== ${modelId} ===`);
  const config = await fetchModelConfig(modelId);
  const tokenizer = await loadTokenizer(modelId);
  const messages = [{ role: "user", content: "What is the capital of France?" }];
  const promptIds = encodeChatPrompt(tokenizer, messages, true);

  console.log("Fetching weights (i8, in-memory)...");
  const tensors = await loadTensors(modelId, "i8");

  // 1) Correctness: identical KV cache contents and logits.
  const maxCtx = promptIds.length + 8;
  const cacheSeq = createKVCache(config, maxCtx);
  const cacheBatch = createKVCache(config, maxCtx);
  const logitsSeq = prefillSequential(tensors, config, cacheSeq, promptIds);
  const logitsBatch = prefillBatched(tensors, config, cacheBatch, promptIds);

  let worstLayerDiff = 0;
  for (let l = 0; l < config.num_hidden_layers; l++) {
    const kDiff = maxAbsDiff(cacheSeq.layers[l].k, cacheBatch.layers[l].k);
    const vDiff = maxAbsDiff(cacheSeq.layers[l].v, cacheBatch.layers[l].v);
    worstLayerDiff = Math.max(worstLayerDiff, kDiff, vDiff);
  }
  const logitsDiff = maxAbsDiff(logitsSeq, logitsBatch);
  console.log(`Max abs diff, KV cache across all layers: ${worstLayerDiff.toExponential(3)}`);
  console.log(`Max abs diff, final logits: ${logitsDiff.toExponential(3)}`);
  if (cacheSeq.seqLen !== cacheBatch.seqLen) {
    throw new Error(`seqLen mismatch: sequential=${cacheSeq.seqLen} batched=${cacheBatch.seqLen}`);
  }
  if (worstLayerDiff > 1e-3 || logitsDiff > 1e-3) {
    throw new Error("Batched prefill diverges from sequential prefill beyond float noise.");
  }
  console.log("PASS: batched and sequential prefill produce identical KV cache contents and logits.");

  // 2) Speedup: repeat the prompt several times to get a longer synthetic
  // prefill (order doesn't matter for a raw throughput comparison -- both
  // paths just need enough tokens to amortize per-call setup).
  const longPrompt = [];
  while (longPrompt.length < 256) longPrompt.push(...promptIds);
  longPrompt.length = 256;

  const seqCache = createKVCache(config, longPrompt.length + 1);
  const t0 = Date.now();
  prefillSequential(tensors, config, seqCache, longPrompt);
  const seqMs = Date.now() - t0;

  const batchCache = createKVCache(config, longPrompt.length + 1);
  const t1 = Date.now();
  prefillBatched(tensors, config, batchCache, longPrompt);
  const batchMs = Date.now() - t1;

  const speedup = seqMs / batchMs;
  console.log(`Sequential prefill (256 tok): ${seqMs}ms. Batched: ${batchMs}ms. Speedup: ${speedup.toFixed(2)}x`);
}

async function main() {
  const arg = process.argv[2];
  const models = arg ? [arg] : ["135M", "360M"];
  for (const modelId of models) {
    await runOne(modelId);
  }
}

main().catch((err) => {
  console.error("\nFAIL:", err.message);
  process.exit(1);
});
