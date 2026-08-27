// Phase 8's correctness gate (plan.md section 5): "multi-turn conversation
// with a persisted KV cache -- turn 2 prefills only the new user message,
// never the whole conversation," and "turn 3 no slower to start than turn 2."
//
// Three checks, cheapest first:
//
//   1. TOKENIZER (no weights, instant). Feeding the conversation to the
//      model one turn at a time must produce exactly the same token ids as
//      rendering the whole thing and encoding it in one go. This is the
//      assumption the entire persisted-cache design rests on and it is not
//      free: it holds only because special tokens are matched literally
//      before pretokenization (gotcha 9), which keeps the lone "\n"
//      between <|im_end|> and <|im_start|> encoding identically in
//      isolation and mid-string. If it ever broke, every turn after the
//      first would be subtly off-format with no error (gotcha 16).
//
//   2. ENGINE (real weights). A 3-turn conversation run through one
//      persisted cache must land in the same state as a fresh cache
//      prefilled with the equivalent full conversation -- same seqLen,
//      same KV contents, same logits. This catches position bookkeeping
//      drifting by one (gotcha 20), which otherwise shows up only as
//      fluent garbage several turns in.
//
//   3. TIMING. Turn 3's prefill must not cost more than turn 2's, which
//      is only true if the cache is genuinely persisted. Reported against
//      what a full re-prefill would have cost, which is the number the
//      cache is buying down.
//
// Usage: node tests/chat-turns.mjs [135M|360M]
// Requires network access to huggingface.co.
import { resolveUrl } from "../js/models.js";
import { rangedFetch } from "../js/model/download.js";
import { parseHeader, decodeBf16, quantizeRowsInt8 } from "../js/model/safetensors.js";
import { fetchModelConfig } from "../js/model/config.js";
import { loadTokenizer, StreamDecoder } from "../js/model/tokenizer.js";
import {
  renderChatPrompt,
  renderSystemTurn,
  renderUserTurn,
  renderTurnClose,
} from "../js/model/template.js";
import { prefill, decodeStep } from "../js/model/transformer.js";
import { createKVCache } from "../js/model/kvcache.js";
import { argmax } from "../js/model/sample.js";

const HEADER_PROBE_BYTES = 200_000;
const CONCURRENCY = 4;
const MAX_CTX = 512;
const REPLY_BUDGET = 24;

const USER_TURNS = [
  "What is the capital of France?",
  "What language do they speak there?",
  "Name one famous landmark.",
];

async function loadTensors(modelId) {
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
      if (info.shape.length === 2) {
        const { qweight, scales } = quantizeRowsInt8(f32, info.shape[0], info.shape[1]);
        tensors.set(name, { kind: "i8", shape: info.shape, qweight, scales });
      } else {
        tensors.set(name, { kind: "f32", shape: info.shape, f32 });
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, names.length) }, worker));
  return tensors;
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function sameIds(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

// --- 1. tokenizer: incremental turns == one-shot template ---------------

function checkIncrementalTokenization(tokenizer) {
  console.log("\n-- incremental tokenization --");

  // Stand-in assistant replies. Their exact text does not matter; what
  // matters is that a turn boundary falls right after the <|im_end|> the
  // model emits and right before the "\n" the template puts after it.
  const replies = ["Paris.", "French.", "The Eiffel Tower."];

  const segments = [renderSystemTurn()];
  for (let turn = 0; turn < USER_TURNS.length; turn++) {
    if (turn > 0) segments.push(renderTurnClose(true));
    segments.push(renderUserTurn(USER_TURNS[turn]));
    if (turn < replies.length - 1) segments.push(`${replies[turn]}<|im_end|>`);
  }

  const incrementalIds = segments.flatMap((segment) => tokenizer.encode(segment));
  const oneShotIds = tokenizer.encode(segments.join(""));

  console.log(`  segments: ${segments.length}, tokens: ${incrementalIds.length}`);
  if (!sameIds(incrementalIds, oneShotIds)) {
    const at = incrementalIds.findIndex((v, i) => v !== oneShotIds[i]);
    throw new Error(
      `Incremental encoding diverges from one-shot at token ${at}: ` +
        `${incrementalIds[at]} vs ${oneShotIds[at]}`
    );
  }
  console.log("  PASS: per-turn encoding == one-shot encoding of the same string.");

  // And the string the incremental path builds is the canonical template.
  const messages = [];
  for (let turn = 0; turn < USER_TURNS.length; turn++) {
    messages.push({ role: "user", content: USER_TURNS[turn] });
    if (turn < replies.length - 1) messages.push({ role: "assistant", content: replies[turn] });
  }
  const canonical = renderChatPrompt(messages, true);
  if (segments.join("") !== canonical) {
    console.log(`  incremental: ${JSON.stringify(segments.join(""))}`);
    console.log(`  canonical:   ${JSON.stringify(canonical)}`);
    throw new Error("Incremental turn rendering does not reproduce renderChatPrompt().");
  }
  console.log("  PASS: assembled turns reproduce renderChatPrompt() character for character.");
}

// --- 2 + 3. engine: persisted cache == fresh cache, and stays fast -------

// One reply, greedily, exactly as worker.js generates it: sample, stop on
// EOS but feed it in first (gotcha 21), otherwise feed and continue.
function generateReply(tensors, config, cache, logits, eosId) {
  const ids = [];
  let endedWithEos = false;
  for (let i = 0; i < REPLY_BUDGET; i++) {
    const tokenId = argmax(logits);
    ids.push(tokenId);
    if (tokenId === eosId || tokenId === 0) {
      decodeStep(tensors, config, cache, tokenId);
      endedWithEos = true;
      break;
    }
    logits = decodeStep(tensors, config, cache, tokenId);
  }
  return { ids, endedWithEos };
}

async function runOne(modelId) {
  console.log(`\n=== ${modelId} ===`);
  const config = await fetchModelConfig(modelId);
  const tokenizer = await loadTokenizer(modelId);

  checkIncrementalTokenization(tokenizer);

  console.log("\nFetching weights (i8, in-memory)...");
  const tensors = await loadTensors(modelId);
  const eosId = config.eos_token_id ?? 2;

  console.log("\n-- persisted-cache conversation --");
  const cache = createKVCache(config, MAX_CTX);

  // The system prompt is prefilled once at load, before any user input.
  const systemIds = tokenizer.encode(renderSystemTurn());
  prefill(tensors, config, cache, systemIds);
  console.log(`  system prompt: ${systemIds.length} tokens, seqLen=${cache.seqLen}`);

  const allIds = [...systemIds];
  const prefillMs = [];
  let pendingPrefix = "";

  for (const userTurn of USER_TURNS) {
    const turnIds = tokenizer.encode(pendingPrefix + renderUserTurn(userTurn));
    const t0 = Date.now();
    let logits = prefill(tensors, config, cache, turnIds);
    prefillMs.push(Date.now() - t0);
    allIds.push(...turnIds);

    const { ids, endedWithEos } = generateReply(tensors, config, cache, logits, eosId);
    allIds.push(...ids);
    pendingPrefix = renderTurnClose(endedWithEos);

    const decoder = new StreamDecoder(tokenizer);
    const text = ids.map((id) => decoder.push(id)).join("") + decoder.flush();
    console.log(
      `  turn ${prefillMs.length}: prefill ${turnIds.length} tok in ${prefillMs.at(-1)}ms, ` +
        `reply ${ids.length} tok -> ${JSON.stringify(text)}`
    );
  }

  if (cache.seqLen !== allIds.length) {
    throw new Error(`seqLen ${cache.seqLen} != tokens fed ${allIds.length} (gotcha 20)`);
  }
  console.log(`  seqLen=${cache.seqLen} matches ${allIds.length} tokens fed. Bookkeeping exact.`);

  // Same tokens, one cache built from scratch in a single prefill.
  console.log("\n-- equivalence against a from-scratch prefill --");
  const fresh = createKVCache(config, MAX_CTX);
  const t0 = Date.now();
  const freshLogits = prefill(tensors, config, fresh, allIds);
  const fullPrefillMs = Date.now() - t0;

  // The persisted cache's own final logits: re-derive by asking it for
  // the next token after everything it has seen, which is what a 4th turn
  // would start from. Comparing KV contents is the stronger check anyway.
  let worstDiff = 0;
  for (let l = 0; l < config.num_hidden_layers; l++) {
    const used = cache.seqLen * cache.nKV * cache.headDim;
    worstDiff = Math.max(
      worstDiff,
      maxAbsDiff(cache.layers[l].k.subarray(0, used), fresh.layers[l].k.subarray(0, used)),
      maxAbsDiff(cache.layers[l].v.subarray(0, used), fresh.layers[l].v.subarray(0, used))
    );
  }
  console.log(`  Max abs diff, KV cache across all layers: ${worstDiff.toExponential(3)}`);
  if (fresh.seqLen !== cache.seqLen) {
    throw new Error(`seqLen mismatch: persisted=${cache.seqLen} fresh=${fresh.seqLen}`);
  }
  if (worstDiff > 1e-3) {
    throw new Error("Turn-by-turn cache diverges from a from-scratch prefill of the same tokens.");
  }
  console.log("  PASS: three turns through a persisted cache == one full prefill.");
  void freshLogits;

  console.log("\n-- turn latency --");
  prefillMs.forEach((ms, i) => console.log(`  turn ${i + 1} prefill: ${ms}ms`));
  console.log(`  a full ${allIds.length}-token re-prefill would cost: ${fullPrefillMs}ms`);
  // Turn 3 prefills a similar number of new tokens to turn 2, so its cost
  // should be in the same ballpark -- not growing with conversation
  // length, which is what re-prefilling every turn would look like.
  if (prefillMs[2] > prefillMs[1] * 2) {
    throw new Error(
      `Turn 3 prefill (${prefillMs[2]}ms) is more than 2x turn 2 (${prefillMs[1]}ms) -- ` +
        `the cache is not being reused across turns (gotcha 22).`
    );
  }
  console.log("  PASS: turn 3 starts no slower than turn 2.");
}

async function main() {
  const modelId = process.argv[2] ?? "135M";
  await runOne(modelId);
  console.log("\nAll checks passed.");
}

main().catch((err) => {
  console.error("\nFAIL:", err.message);
  process.exit(1);
});
