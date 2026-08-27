// Phase 5's exit criterion (plan.md section 5): "top-10 logits for a fixed
// prompt match the reference to within f32 noise." Loads the real weights
// straight from Hugging Face into memory as f32 (skipping IndexedDB, which
// only exists in a browser), runs transformer.js's forward pass, and
// compares against reference/golden/logits_<model>.json -- generated once
// with real `transformers` (see notes.md).
//
// Usage: node tests/golden-forward.mjs [135M|360M]
// Requires network access to huggingface.co.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { resolveUrl } from "../js/models.js";
import { headContentLength, rangedFetch } from "../js/model/download.js";
import { parseHeader, decodeBf16 } from "../js/model/safetensors.js";
import { fetchModelConfig } from "../js/model/config.js";
import { loadTokenizer } from "../js/model/tokenizer.js";
import { encodeChatPrompt, renderChatPrompt } from "../js/model/template.js";
import { forward } from "../js/model/transformer.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HEADER_PROBE_BYTES = 200_000;
const CONCURRENCY = 4;

// f32-only in-memory loader for testing: same header/decode logic as
// safetensors.js's loadModelTensors, but skipping quantization and
// IndexedDB entirely -- this harness only needs the tensors for one
// forward-pass call, not persistence.
async function loadTensorsF32(modelId) {
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
      tensors.set(name, { kind: "f32", shape: info.shape, f32 });
    }
  }

  const workerCount = Math.min(CONCURRENCY, names.length);
  await Promise.all(Array.from({ length: workerCount }, worker));
  return tensors;
}

function loadGolden(modelId) {
  const p = path.join(__dirname, "..", "reference", "golden", `logits_${modelId}.json`);
  return JSON.parse(readFileSync(p, "utf-8"));
}

async function runOne(modelId) {
  console.log(`\n=== ${modelId} ===`);
  const golden = loadGolden(modelId);

  const config = await fetchModelConfig(modelId);
  const tokenizer = await loadTokenizer(modelId);

  const promptText = renderChatPrompt(golden.messages, true);
  const promptIds = encodeChatPrompt(tokenizer, golden.messages, true);

  if (promptText !== golden.prompt_text) {
    throw new Error(
      `Prompt text mismatch.\nExpected: ${JSON.stringify(golden.prompt_text)}\nGot:      ${JSON.stringify(promptText)}`
    );
  }
  if (JSON.stringify(promptIds) !== JSON.stringify(golden.prompt_ids)) {
    throw new Error(
      `Token ids mismatch.\nExpected: ${JSON.stringify(golden.prompt_ids)}\nGot:      ${JSON.stringify(promptIds)}`
    );
  }
  console.log(`Prompt text and ${promptIds.length} token ids match golden exactly.`);

  console.log("Fetching weights (f32, in-memory)...");
  const t0 = Date.now();
  const tensors = await loadTensorsF32(modelId);
  console.log(`Loaded ${tensors.size} tensors in ${((Date.now() - t0) / 1000).toFixed(1)}s.`);

  console.log("Running forward pass...");
  const t1 = Date.now();
  const logits = forward(tensors, config, promptIds);
  console.log(`Forward pass took ${((Date.now() - t1) / 1000).toFixed(1)}s.`);

  const ranked = Array.from(logits)
    .map((v, i) => [i, v])
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);
  const gotIndices = ranked.map(([i]) => i);
  const gotValues = ranked.map(([, v]) => v);

  console.log("Golden top-10:", golden.top10_indices);
  console.log("Got    top-10:", gotIndices);

  const indicesMatch = JSON.stringify(gotIndices) === JSON.stringify(golden.top10_indices);
  let maxAbsDiff = 0;
  for (let i = 0; i < 10; i++) {
    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(gotValues[i] - golden.top10_values[i]));
  }
  console.log(`Max abs diff on top-10 values: ${maxAbsDiff.toExponential(3)}`);

  if (!indicesMatch) {
    throw new Error("Top-10 token ranking does not match golden reference.");
  }
  if (maxAbsDiff > 0.05) {
    throw new Error(`Top-10 values diverge by more than f32 noise tolerance (0.05): ${maxAbsDiff}`);
  }
  console.log(`PASS: ${modelId} matches golden reference (top-10 ranking exact, values within f32 noise).`);
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
