// Main thread: DOM only. Every byte of model work happens in worker.js
// (plan.md 1.5); this file never imports anything from js/model/ except
// the two small pure helpers below, so there is no way for a heavy
// operation to sneak onto the UI thread.
import { MODELS } from "./models.js";
import { getManifest } from "./model/store.js";

const DTYPE = "i8";

const worker = new Worker("js/worker.js", { type: "module" });

const landing = document.getElementById("landing");
const chat = document.getElementById("chat");
const logEl = document.getElementById("log");
const transcript = document.getElementById("transcript");
const composer = document.getElementById("composer");
const input = document.getElementById("input");
const sendButton = document.getElementById("send");
const stopButton = document.getElementById("stop");
const greedyBox = document.getElementById("greedy");
const temperatureSlider = document.getElementById("temperature");
const topPSlider = document.getElementById("top-p");
const temperatureOut = document.getElementById("temperature-out");
const topPOut = document.getElementById("top-p-out");

const hud = {
  model: document.getElementById("hud-model"),
  ctx: document.getElementById("hud-ctx"),
  speed: document.getElementById("hud-speed"),
  prefill: document.getElementById("hud-prefill"),
  mem: document.getElementById("hud-mem"),
};

// ---------------------------- mobile viewport ---------------------------
// iOS Safari (and Chrome-on-iOS, which is Safari underneath) does not
// resize position:fixed elements when the on-screen keyboard opens -- the
// layout viewport stays full-height and the page scrolls to keep the
// focused field visible, carrying fixed content like the HUD strip up out
// of view with it. window.visualViewport reports the part of the page
// actually visible above the keyboard, so screens are pinned to that
// instead of to 100vh.
function syncViewport() {
  const vv = window.visualViewport;
  if (!vv) return;
  document.documentElement.style.setProperty("--vv-height", `${vv.height}px`);
  document.documentElement.style.setProperty("--vv-top", `${vv.offsetTop}px`);
}
window.visualViewport?.addEventListener("resize", syncViewport);
window.visualViewport?.addEventListener("scroll", syncViewport);
syncViewport();

function setMem(residentMB) {
  hud.mem.textContent = residentMB ? `mem ${residentMB.toFixed(0)}MB` : "mem --";
}

// StorageManager.persist() is exposed on Window only, not in workers, so
// it has to happen here rather than next to the download it protects.
// Without it the browser may evict the whole ~135MB cache under storage
// pressure, turning the next visit back into a full re-download.
async function requestPersistence() {
  if (!navigator.storage?.persist) return;
  const granted = await navigator.storage.persist();
  log(`Storage persistence ${granted ? "granted" : "NOT granted"}.`);
}

const panels = new Map(
  [...document.querySelectorAll(".model-panel")].map((el) => [
    el.dataset.model,
    {
      el,
      status: el.querySelector(".panel-status"),
      action: el.querySelector(".panel-action"),
      progress: el.querySelector(".panel-progress"),
      bar: el.querySelector(".bar > span"),
      barLabel: el.querySelector(".bar-label"),
    },
  ])
);

// activeModel is set the moment a load starts and stays set: exactly one
// model is resident in the worker at a time (gotcha 23), so the other
// panel is locked out rather than allowed to queue a second load.
let activeModel = null;
let readyModel = null;
let streamingBubble = null;

function log(text) {
  const time = new Date().toLocaleTimeString();
  logEl.textContent += `[${time}] ${text}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

// ------------------------------ landing ---------------------------------

function setPanelStatus(modelId, text, isError = false) {
  const panel = panels.get(modelId);
  panel.status.textContent = text;
  panel.status.classList.toggle("error", isError);
}

function setPanelAction(modelId, label, role, disabled = false) {
  const { action } = panels.get(modelId);
  action.textContent = label;
  action.dataset.role = role;
  action.disabled = disabled;
}

function showPanelProgress(modelId, fraction, label) {
  const panel = panels.get(modelId);
  panel.progress.hidden = false;
  panel.bar.style.width = `${Math.min(100, Math.max(0, fraction * 100)).toFixed(1)}%`;
  panel.barLabel.textContent = label;
}

function hidePanelProgress(modelId) {
  panels.get(modelId).progress.hidden = true;
}

// Both models cached at once is a normal state, but only one can be
// loaded, so a cached model still shows "Load" rather than jumping
// straight to "Start chat": there is real work between a cached download
// and a usable model (reading ~135MB back out of IndexedDB, then
// prefilling the system prompt). Promising "start chat" before that would
// be a lie about a five-second wait.
async function refreshCacheStates() {
  for (const modelId of Object.keys(MODELS)) {
    const manifest = await getManifest(modelId, DTYPE);
    if (manifest?.complete) {
      setPanelStatus(modelId, `Cached locally (${manifest.tensorNames.length} tensors). No download needed.`);
      setPanelAction(modelId, "Load (cached)", "load");
    }
  }
}

function lockOtherPanels(activeId) {
  for (const [modelId, panel] of panels) {
    if (modelId === activeId) {
      panel.el.dataset.state = "active";
    } else {
      panel.el.dataset.disabled = "true";
      panel.action.disabled = true;
    }
  }
}

for (const [modelId, panel] of panels) {
  panel.action.addEventListener("click", () => {
    if (panel.action.dataset.role === "chat") {
      enterChat();
      return;
    }
    activeModel = modelId;
    lockOtherPanels(modelId);
    setPanelAction(modelId, "Loading", "loading", true);
    log(`Loading ${MODELS[modelId].label}...`);
    requestPersistence().finally(() => worker.postMessage({ type: "load", modelId }));
  });
}

// ------------------------------- chat ----------------------------------

function enterChat() {
  landing.classList.add("fading");
  // Matches the .screen opacity transition in style.css; the landing
  // screen stays in the layout until it has finished fading so the two
  // screens cross-fade instead of one popping in.
  setTimeout(() => {
    landing.hidden = true;
    chat.hidden = false;
    chat.classList.add("entering");
    requestAnimationFrame(() => chat.classList.remove("entering"));
    input.focus();
  }, 600);
}

function addBubble(role, text = "") {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  transcript.appendChild(bubble);
  return bubble;
}

// Only auto-scroll when the user is already at the bottom, so reading
// back through a long reply is not fought by every incoming token.
function scrollTranscript() {
  const nearBottom =
    transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 80;
  if (nearBottom) transcript.scrollTop = transcript.scrollHeight;
}

function setGenerating(isGenerating) {
  sendButton.hidden = isGenerating;
  stopButton.hidden = !isGenerating;
  input.disabled = isGenerating;
}

function samplingSettings() {
  if (greedyBox.checked) return { temperature: 0, topP: 1 };
  return {
    temperature: Number(temperatureSlider.value),
    topP: Number(topPSlider.value),
  };
}

function syncSamplingControls() {
  const greedy = greedyBox.checked;
  temperatureSlider.disabled = greedy;
  topPSlider.disabled = greedy;
  temperatureOut.textContent = Number(temperatureSlider.value).toFixed(2);
  topPOut.textContent = Number(topPSlider.value).toFixed(2);
}

greedyBox.addEventListener("change", syncSamplingControls);
temperatureSlider.addEventListener("input", syncSamplingControls);
topPSlider.addEventListener("input", syncSamplingControls);
syncSamplingControls();

// Grow the textarea with its content, up to the max-height in CSS.
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
});

// Enter sends, Shift+Enter makes a newline -- the convention every chat
// app uses, and the reason the composer is a <textarea> and not an
// <input>.
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || input.disabled) return;

  addBubble("user", text);
  input.value = "";
  input.style.height = "auto";
  scrollTranscript();

  setGenerating(true);
  worker.postMessage({ type: "generate", text, ...samplingSettings() });
});

stopButton.addEventListener("click", () => {
  stopButton.disabled = true;
  worker.postMessage({ type: "stop" });
});

// ---------------------------- worker events -----------------------------

function updateCtx(seqLen, maxCtx) {
  hud.ctx.textContent = `ctx ${seqLen}/${maxCtx}`;
}

worker.onmessage = (event) => {
  const message = event.data;

  switch (message.type) {
    case "log":
      log(message.text);
      break;

    case "status":
      if (activeModel && !readyModel) setPanelStatus(activeModel, `${message.text}...`);
      break;

    case "progress": {
      const fraction = message.total ? message.loaded / message.total : 0;
      const label =
        message.phase === "download"
          ? `${(message.loaded / 1e6).toFixed(0)} / ${(message.total / 1e6).toFixed(0)} MB` +
            `  ·  tensor ${message.i}/${message.n}`
          : `unpacking ${message.i}/${message.n}`;
      showPanelProgress(activeModel, fraction, label);
      break;
    }

    case "ready": {
      readyModel = message.modelId;
      hidePanelProgress(message.modelId);
      setPanelStatus(
        message.modelId,
        `Ready. System prompt processed (${message.seqLen} tokens in context).`
      );
      setPanelAction(message.modelId, "Start chat", "chat");
      hud.model.textContent = message.label;
      updateCtx(message.seqLen, message.maxCtx);
      setMem(message.residentMB);
      log(`${message.label} ready.`);
      break;
    }

    case "reply-start":
      streamingBubble = addBubble("assistant");
      streamingBubble.classList.add("streaming");
      scrollTranscript();
      break;

    case "prefill-done":
      hud.prefill.textContent =
        `prefill ${message.tokens}tok ${(message.ms / 1000).toFixed(1)}s`;
      updateCtx(message.seqLen, message.maxCtx);
      break;

    case "token":
      if (streamingBubble) {
        streamingBubble.append(message.text);
        scrollTranscript();
      }
      break;

    case "reply-done": {
      if (streamingBubble) {
        streamingBubble.classList.remove("streaming");
        if (message.reason === "stopped") streamingBubble.append(" [stopped]");
        if (message.reason === "budget") streamingBubble.append(" [token limit]");
        streamingBubble = null;
      }
      hud.speed.textContent = `${message.tokensPerSecond.toFixed(2)} tok/s`;
      setMem(message.residentMB);
      updateCtx(message.seqLen, message.maxCtx);
      stopButton.disabled = false;
      setGenerating(false);
      input.focus();
      break;
    }

    // The turn was refused before anything entered the cache, so the
    // conversation is still intact -- this is a notice, not an error.
    case "rejected":
      addBubble("system", message.message);
      scrollTranscript();
      stopButton.disabled = false;
      setGenerating(false);
      break;

    case "error":
      log(`ERROR (${message.phase}): ${message.message}`);
      if (message.phase === "load" && activeModel) {
        setPanelStatus(activeModel, `Failed: ${message.message}`, true);
        setPanelAction(activeModel, "Retry", "load");
        hidePanelProgress(activeModel);
      } else {
        if (streamingBubble) {
          streamingBubble.classList.remove("streaming");
          streamingBubble = null;
        }
        addBubble("system", `Error: ${message.message}`);
        stopButton.disabled = false;
        setGenerating(false);
      }
      break;
  }
};

worker.onerror = (event) => {
  log(`Worker error: ${event.message}`);
};

refreshCacheStates();
