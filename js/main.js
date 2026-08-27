import { MODELS, resolveUrl } from "./models.js";
import { fetchModelConfig } from "./model/config.js";
import { headContentLength, downloadWithProgress } from "./model/download.js";
import { getModelRecord, putModelRecord } from "./model/store.js";

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

    const weightsUrl = resolveUrl(modelId, "model.safetensors");
    setStatus(`Checking cache for ${model.label}...`);
    const remoteLength = await headContentLength(weightsUrl);
    log(`Remote model.safetensors size: ${(remoteLength / 1e6).toFixed(1)} MB`);

    const cached = await getModelRecord(modelId);
    if (cached && cached.contentLength === remoteLength) {
      log(`Cache hit: using stored copy (${(cached.bytes.byteLength / 1e6).toFixed(1)} MB), skipping network.`);
      setStatus(`${model.label} ready (from cache).`);
      setButtonsDisabled(false);
      return;
    }

    if (cached) {
      log("Cached copy size mismatch with remote -- re-downloading.");
    }

    setStatus(`Downloading ${model.label}...`);
    const { buffer, byteLength } = await downloadWithProgress(weightsUrl, showProgress);
    hideProgress();

    if (remoteLength !== null && byteLength !== remoteLength) {
      throw new Error(
        `Downloaded ${byteLength} bytes but Content-Length said ${remoteLength} -- truncated download.`
      );
    }
    log(`Downloaded ${(byteLength / 1e6).toFixed(1)} MB. Storing in IndexedDB...`);

    await putModelRecord({
      id: modelId,
      contentLength: remoteLength,
      bytes: buffer,
      downloadedAt: Date.now(),
    });

    await requestPersistence();
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
