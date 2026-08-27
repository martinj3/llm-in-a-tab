import { MODELS } from "./models.js";
import { fetchModelConfig } from "./model/config.js";
import { loadModelTensors } from "./model/safetensors.js";
import { getTensor } from "./model/store.js";
import { loadTokenizer } from "./model/tokenizer.js";
import { renderChatPrompt, encodeChatPrompt } from "./model/template.js";

const DTYPE = "i8";

const statusLine = document.getElementById("status-line");
const logEl = document.getElementById("log");
const progressWrap = document.getElementById("progress-wrap");
const progressBar = document.getElementById("progress-bar");
const progressLabel = document.getElementById("progress-label");
const buttons = document.querySelectorAll(".model-buttons button");

function setStatus(text) {
  statusLine.textContent = text;
}

function log(text) {
  const time = new Date().toLocaleTimeString();
  logEl.textContent += `[${time}] ${text}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

function setButtonsDisabled(disabled) {
  buttons.forEach((b) => (b.disabled = disabled));
}

function showProgress(loaded, total) {
  progressWrap.hidden = false;
  const mb = (n) => (n / (1024 * 1024)).toFixed(1);
  if (total && !Number.isNaN(total)) {
    progressBar.max = total;
    progressBar.value = loaded;
    progressLabel.textContent = `${mb(loaded)} / ${mb(total)} MB`;
  } else {
    progressBar.removeAttribute("value");
    progressLabel.textContent = `${mb(loaded)} MB`;
  }
}

function hideProgress() {
  progressWrap.hidden = true;
}

async function reportStorageEstimate() {
  if (!navigator.storage?.estimate) return;
  const { usage, quota } = await navigator.storage.estimate();
  const mb = (n) => (n / (1024 * 1024)).toFixed(0);
  log(`Storage: ${mb(usage)} MB used of ${mb(quota)} MB quota.`);
}

async function requestPersistence() {
  if (!navigator.storage?.persist) return;
  const granted = await navigator.storage.persist();
  log(`Storage persistence ${granted ? "granted" : "NOT granted"}.`);
}

// Phase 2 exit criterion: "a weight histogram looks like a bell curve
// centered near zero." Dequantizes one row of a real weight matrix and
// renders a quick ASCII histogram so a human can eyeball the distribution
// without any charting library.
async function logWeightHistogram(modelId, dtype) {
  const record = await getTensor(modelId, dtype, "model.layers.0.self_attn.q_proj.weight");
  if (!record || record.kind !== "i8") return;

  const [, numCols] = record.shape;
  const scale = record.scales[0];
  const row = new Float32Array(numCols);
  for (let c = 0; c < numCols; c++) row[c] = record.qweight[c] * scale;

  const maxAbs = Math.max(...row.map(Math.abs), 1e-8);
  const bins = 11;
  const counts = new Array(bins).fill(0);
  for (const v of row) {
    const bucket = Math.min(bins - 1, Math.floor(((v + maxAbs) / (2 * maxAbs)) * bins));
    counts[bucket]++;
  }
  const maxCount = Math.max(...counts);
  const lines = counts.map((c, i) => {
    const lo = (-maxAbs + (2 * maxAbs * i) / bins).toFixed(3);
    const bar = "#".repeat(Math.round((c / maxCount) * 30));
    return `  ${lo.padStart(8)}  ${bar}`;
  });
  log(`Histogram of layer 0 q_proj row 0 (${numCols} values, dequantized):\n${lines.join("\n")}`);
}

async function handleDownload(modelId) {
  const model = MODELS[modelId];
  setButtonsDisabled(true);
  try {
    setStatus(`Fetching config for ${model.label}...`);
    const config = await fetchModelConfig(modelId);
    log(
      `${model.label}: hidden_size=${config.hidden_size}, layers=${config.num_hidden_layers}, ` +
        `heads=${config.num_attention_heads}, kv_heads=${config.num_key_value_heads}, ` +
        `rope_theta=${config.rope_theta}`
    );

    await reportStorageEstimate();

    setStatus(`Loading ${model.label} tensors...`);
    const result = await loadModelTensors(modelId, config, DTYPE, (loaded, total, name, i, n) => {
      showProgress(loaded, total);
      setStatus(`Loading ${model.label}: tensor ${i}/${n} (${name})`);
    });
    hideProgress();

    if (result.cached) {
      log(`Cache hit: ${result.tensorNames.length} tensors already stored, skipped the network.`);
    } else {
      log(
        `Loaded and quantized ${result.tensorNames.length} tensors from ` +
          `${(result.remoteLength / 1e6).toFixed(1)} MB of source data.`
      );
      await requestPersistence();
    }

    await logWeightHistogram(modelId, DTYPE);

    setStatus(`Loading tokenizer for ${model.label}...`);
    const tokenizer = await loadTokenizer(modelId);
    log(`Tokenizer loaded: ${tokenizer.tokenToId.size} vocab entries.`);

    const sampleMessages = [{ role: "user", content: "Hello!" }];
    const promptText = renderChatPrompt(sampleMessages, true);
    const promptIds = encodeChatPrompt(tokenizer, sampleMessages, true);
    log(`Sample chat prompt (${promptIds.length} tokens): ${JSON.stringify(promptText)}`);
    log(`Round-trip decode: ${JSON.stringify(tokenizer.decode(promptIds))}`);

    setStatus(`${model.label} ready.`);
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
    log(`ERROR: ${err.message}`);
  } finally {
    hideProgress();
    setButtonsDisabled(false);
  }
}

buttons.forEach((button) => {
  button.addEventListener("click", () => {
    const modelId = button.dataset.model;
    log(`Selected model: ${modelId}`);
    handleDownload(modelId);
  });
});
