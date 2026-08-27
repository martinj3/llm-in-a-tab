// The inference worker. Owns the weights, the KV cache, and the
// generation loop; the main thread owns only the DOM (plan.md 1.5).
//
// A scalar JS forward pass pins a core for 300-800ms per token. On the
// main thread that is a page that does not scroll, does not repaint, and
// shows the browser's "page unresponsive" dialog. Everything below runs
// here so the UI stays alive while a reply is being generated.
//
// One worker, never several: GitHub Pages cannot set the COOP/COEP
// headers cross-origin isolation needs, so SharedArrayBuffer is
// unavailable and a second worker would need its own 135-360MB copy of
// the weights (gotcha 23). That also means exactly one model is resident
// at a time -- loading the other one replaces it.
import { MODELS } from "./models.js";
import { fetchModelConfig } from "./model/config.js";
import { loadModelTensors } from "./model/safetensors.js";
import { getAllTensors } from "./model/store.js";
import { loadTokenizer, StreamDecoder } from "./model/tokenizer.js";
import { renderSystemTurn, renderUserTurn, renderTurnClose } from "./model/template.js";
import { createKVCache } from "./model/kvcache.js";
import { prefill, decodeStep } from "./model/transformer.js";
import { sampleToken } from "./model/sample.js";
import { createProbe } from "./model/probe.js";

const DTYPE = "i8";

// plan.md section 8 lists contexts above 1024 as a non-goal for v1, and
// section 1.4's table is why: the 360M's KV cache alone is 84MB at 1k.
const MAX_CTX = 1024;

// plan.md section 7: hard stop at the limit, and reserve headroom, so a
// prompt that would leave no room to answer is rejected up front rather
// than accepted and then truncated mid-reply.
const REPLY_RESERVE = 256;

let model = null; // { modelId, config, tensors, tokenizer, cache, pendingPrefix, probe }
let stopRequested = false;
let generating = false;

// Owned by the main thread (the user can switch the visualization off in
// the composer). When false the probe is never handed to decodeStep, so
// the forward pass runs exactly as it did before this existed.
let vizEnabled = true;

function post(message) {
  self.postMessage(message);
}

function postTransfer(message, transfer) {
  self.postMessage(message, transfer);
}

function log(text) {
  post({ type: "log", text });
}

function status(text) {
  post({ type: "status", text });
}

// What the model actually occupies, in MB, measured rather than probed.
//
// There is no way to ask for this tab's real memory use: the standardized
// measureUserAgentSpecificMemory() requires cross-origin isolation, which
// GitHub Pages cannot give us (gotcha 23), and Chrome's non-standard
// performance.memory is not exposed in workers at all -- it exists only
// on Window, where it would report the *main thread's* isolate and
// therefore miss every byte of this. So instead of a heap probe that
// would be either unavailable or wrong, sum the byteLengths we allocated.
// That is the number worth showing anyway: it is the one that decides
// whether a phone kills the tab.
function residentMB() {
  if (!model) return null;
  let bytes = 0;
  for (const tensor of model.tensors.values()) {
    bytes += tensor.kind === "f32"
      ? tensor.f32.byteLength
      : tensor.qweight.byteLength + tensor.scales.byteLength;
  }
  for (const layer of model.cache.layers) bytes += layer.k.byteLength + layer.v.byteLength;
  return bytes / (1024 * 1024);
}

// Lets the worker's message queue drain between tokens so a "stop"
// message can actually be delivered. A message handler runs to
// completion, so without this the generation loop would be
// uninterruptible and the stop button would only take effect after the
// whole reply finished. One macrotask per token against a ~300ms token is
// free.
function yieldToMessages() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function handleLoad({ modelId }) {
  const label = MODELS[modelId].label;

  status(`Fetching config for ${label}`);
  const config = await fetchModelConfig(modelId);
  log(
    `${label}: hidden=${config.hidden_size} layers=${config.num_hidden_layers} ` +
      `heads=${config.num_attention_heads} kv_heads=${config.num_key_value_heads} ` +
      `rope_theta=${config.rope_theta}`
  );

  // Free the previous model before allocating the next one, so a switch
  // does not transiently hold both sets of weights.
  model = null;

  status("Checking storage");
  if (navigator.storage?.estimate) {
    const { usage, quota } = await navigator.storage.estimate();
    const mb = (n) => (n / (1024 * 1024)).toFixed(0);
    log(`Storage: ${mb(usage)} MB used of ${mb(quota)} MB quota.`);
  }

  const loadResult = await loadModelTensors(modelId, config, DTYPE, (loaded, total, name, i, n) => {
    post({ type: "progress", phase: "download", loaded, total, name, i, n });
    status(`Downloading tensor ${i}/${n}`);
  });

  // Persistence is requested by main.js before this runs -- StorageManager
  // .persist() is exposed on Window only, not in workers.
  if (loadResult.cached) {
    log(`Cache hit: ${loadResult.tensorNames.length} tensors already stored, skipped the network.`);
  } else {
    log(
      `Downloaded and quantized ${loadResult.tensorNames.length} tensors from ` +
        `${(loadResult.remoteLength / 1e6).toFixed(1)} MB of bf16 source data.`
    );
  }

  status("Reading weights into memory");
  const tensors = await getAllTensors(modelId, DTYPE, loadResult.tensorNames, (done, total) => {
    post({ type: "progress", phase: "unpack", loaded: done, total, i: done, n: total });
  });

  status("Loading tokenizer");
  const tokenizer = await loadTokenizer(modelId);
  log(`Tokenizer: ${tokenizer.tokenToId.size} vocab entries.`);

  status("Allocating KV cache");
  const cache = createKVCache(config, MAX_CTX);
  const cacheMB =
    (2 * MAX_CTX * config.num_key_value_heads * (config.hidden_size / config.num_attention_heads) *
      4 * config.num_hidden_layers) /
    (1024 * 1024);
  log(`KV cache: ${MAX_CTX} positions, ${cacheMB.toFixed(0)} MB.`);

  // Prefill the system prompt now, at load time, so the first user
  // message does not pay for it (gotcha 22 -- the whole point of a
  // persisted cache). This is why "start chat" only appears once loading
  // is fully done: by then the model has already read its system prompt.
  status("Processing system prompt");
  const systemIds = tokenizer.encode(renderSystemTurn());
  const t0 = performance.now();
  prefill(tensors, config, cache, systemIds);
  const prefillMs = performance.now() - t0;
  log(
    `System prompt prefilled: ${systemIds.length} tokens in ${prefillMs.toFixed(0)}ms ` +
      `(${(prefillMs / systemIds.length).toFixed(0)}ms/token).`
  );

  const probe = createProbe(config, MAX_CTX);
  model = { modelId, config, tensors, tokenizer, cache, pendingPrefix: "", probe };

  post({
    type: "ready",
    modelId,
    label,
    seqLen: cache.seqLen,
    maxCtx: MAX_CTX,
    residentMB: residentMB(),
    // Frame layout for the visualization. Sent once here rather than with
    // every token: it is fixed by the architecture and the renderer needs
    // it to reshape a flat frame into grids.
    viz: probe.geometry,
  });
  status("Ready");
}

async function handleGenerate({ text, temperature, topP, maxNewTokens }) {
  if (!model) throw new Error("No model loaded.");
  const { config, tensors, tokenizer, cache, probe } = model;

  const turnText = model.pendingPrefix + renderUserTurn(text);
  const turnIds = tokenizer.encode(turnText);

  const budget = Math.min(maxNewTokens ?? REPLY_RESERVE, REPLY_RESERVE);
  if (cache.seqLen + turnIds.length + budget > MAX_CTX) {
    const room = MAX_CTX - cache.seqLen - budget;
    post({
      type: "rejected",
      message:
        `Out of context: this turn is ${turnIds.length} tokens and only ${Math.max(room, 0)} fit ` +
        `while reserving ${budget} for the reply (${cache.seqLen}/${MAX_CTX} used). ` +
        `Reload to start a fresh conversation.`,
    });
    return;
  }

  // The user's turn is now committed to the cache, so the context must
  // end up in canonical template form no matter how generation ends --
  // see renderTurnClose().
  model.pendingPrefix = renderTurnClose(false);

  generating = true;
  stopRequested = false;
  post({ type: "reply-start" });

  const prefillStart = performance.now();
  let logits = prefill(tensors, config, cache, turnIds);
  const prefillMs = performance.now() - prefillStart;
  post({
    type: "prefill-done",
    tokens: turnIds.length,
    ms: prefillMs,
    seqLen: cache.seqLen,
    maxCtx: MAX_CTX,
  });

  // Stop on <|im_end|>, and on <|endoftext|> defensively (gotcha 14).
  const eosId = config.eos_token_id ?? 2;
  const stopIds = new Set([eosId, 0]);

  const decoder = new StreamDecoder(tokenizer);
  const decodeStart = performance.now();
  let generated = 0;
  let endedWithEos = false;
  let reason = "budget";

  while (generated < budget) {
    const tokenId = sampleToken(logits, { temperature, topP });

    if (stopIds.has(tokenId)) {
      // Feed the stop token in anyway. It is part of the template, so it
      // belongs in the cache at seqLen++ exactly like any other token
      // (gotcha 21) -- the next turn's tokens have to follow it, not
      // replace it. Its logits are simply discarded.
      decodeStep(tensors, config, cache, tokenId);
      endedWithEos = true;
      reason = "stop-token";
      break;
    }

    const piece = decoder.push(tokenId);
    generated++;

    const watching = vizEnabled;
    if (watching) probe.beginToken();
    logits = decodeStep(tensors, config, cache, tokenId, watching ? probe : null);
    if (watching) {
      // The forward pass that just ran was *of* this token, so the frame
      // and the text belong together -- which is what lets the renderer
      // caption each pass of the tower with the token it processed.
      const frame = probe.endToken();
      postTransfer(
        {
          type: "activations",
          bytes: frame.bytes,
          attnUsed: frame.attnUsed,
          pos: cache.seqLen - 1,
          text: piece,
        },
        [frame.bytes.buffer]
      );
    }

    // seqLen/tok-s are worth showing live rather than only once the reply
    // finishes -- a reply can run tens of seconds on a phone, and "ctx --/--"
    // sitting frozen the whole time looks like the HUD stopped working
    // rather than like it is just waiting to report at the end.
    const elapsedMs = performance.now() - decodeStart;
    if (piece) {
      post({
        type: "token",
        text: piece,
        seqLen: cache.seqLen,
        maxCtx: MAX_CTX,
        tokensPerSecond: elapsedMs > 0 ? generated / (elapsedMs / 1000) : 0,
      });
    }

    await yieldToMessages();
    if (stopRequested) {
      reason = "stopped";
      break;
    }
  }

  const tail = decoder.flush();
  if (tail) post({ type: "token", text: tail });

  model.pendingPrefix = renderTurnClose(endedWithEos);

  const decodeMs = performance.now() - decodeStart;
  generating = false;
  post({
    type: "reply-done",
    reason,
    generated,
    decodeMs,
    tokensPerSecond: generated > 0 ? generated / (decodeMs / 1000) : 0,
    seqLen: cache.seqLen,
    maxCtx: MAX_CTX,
    residentMB: residentMB(),
  });
}

self.onmessage = async (event) => {
  const message = event.data;

  // Handled outside the try/dispatch below because it has to be
  // processed *while* a generate handler is suspended at its
  // yieldToMessages() await, not queued behind it.
  if (message.type === "stop") {
    if (generating) stopRequested = true;
    return;
  }

  // Same reasoning as "stop": toggling the visualization mid-reply has to
  // land while the generate handler is parked at yieldToMessages(), not
  // after the whole reply has finished.
  if (message.type === "viz") {
    vizEnabled = message.enabled;
    return;
  }

  try {
    if (message.type === "load") {
      await handleLoad(message);
    } else if (message.type === "generate") {
      await handleGenerate(message);
    } else {
      throw new Error(`Unknown message type: ${message.type}`);
    }
  } catch (err) {
    generating = false;
    post({ type: "error", message: err.message, phase: message.type });
  }
};
