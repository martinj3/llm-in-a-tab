// Main thread: DOM only. Every byte of model work happens in worker.js
// (plan.md 1.5); this file never imports anything from js/model/ except
// the two small pure helpers below, so there is no way for a heavy
// operation to sneak onto the UI thread.
import { MODELS } from "./models.js";
import { getManifest } from "./model/store.js";
import { createStackViz } from "./viz/stack.js";

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
const vizToggle = document.getElementById("viz-toggle");
const sampling = document.getElementById("sampling");
const settingsToggle = document.getElementById("settings-toggle");
const candidatesEl = document.getElementById("candidates");
const hintModal = document.getElementById("hint-modal");
const hintModalOk = document.getElementById("hint-modal-ok");

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
// A zero height is never a viewport worth honouring, but it is a value
// visualViewport really does report -- while the tab is hidden, and while
// an embedding pane is still collapsed. Writing it through sets .screen to
// height:0, and since .screen is overflow:auto that clips the entire UI to
// nothing: the page still lays out, getBoundingClientRect still returns
// plausible rects, and every control silently stops hit-testing. Same
// class of failure as the invisible-overlay bug in fix.md, so it gets the
// same treatment -- keep the last good size and let the next real resize
// correct it.
function syncViewport() {
  const vv = window.visualViewport;
  if (!vv || vv.height <= 0) return;
  document.documentElement.style.setProperty("--vv-height", `${vv.height}px`);
  document.documentElement.style.setProperty("--vv-top", `${vv.offsetTop}px`);
}
window.visualViewport?.addEventListener("resize", syncViewport);
window.visualViewport?.addEventListener("scroll", syncViewport);
syncViewport();

// --------------------------- visualization ------------------------------

const viz = createStackViz(document.getElementById("viz"));

// The whole thing is a camera falling through a tower, so it is exactly
// what prefers-reduced-motion is asking about. Default it off there rather
// than removing the control: the preference is a default, not a ban, and
// someone who set it system-wide for carousels may still want this.
if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  vizToggle.checked = false;
}

function syncViz() {
  viz.setEnabled(vizToggle.checked);
  // Tells the worker whether to run the probe at all -- with it off the
  // forward pass is byte-for-byte what it was before the visualization
  // existed, which is also what makes the toggle a usable A/B for its
  // cost on a phone.
  worker.postMessage({ type: "viz", enabled: vizToggle.checked });
}

vizToggle.addEventListener("change", syncViz);
syncViz();

// --------------------------- candidate column ---------------------------
// The worker sends 15 alternatives per token; a phone has no room for a
// column that tall next to the reply, so the count is a media query
// rather than a message to the worker. Live, not stored -- so it is read
// per frame and a rotated phone picks up the new count on the next token.
const wideEnough = window.matchMedia("(min-width: 46rem)");

// Whitespace-only tokens are extremely common and would otherwise render
// as blank rows. Making the space and the newline visible is the
// difference between "the model was choosing between seven things" and
// "the model was choosing between two things and five bugs".
function tokenGlyphs(text) {
  return text.replace(/\n/g, "⏎").replace(/\t/g, "→").replace(/ /g, "·");
}

// A bar chart drawn in text. The left-eighth block characters let one
// monospace cell render eight widths, so a 2-cell bar has 16 steps and a
// 4-cell bar has 32 -- enough resolution to read a shape, in a strip
// narrow enough to sit beside the token without becoming the widest thing
// in the column. Drawing it as characters rather than as a styled <div>
// keeps it on the same monospace grid as everything else, and keeps the
// column selectable and copyable as plain text.
const EIGHTHS = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];

// Scaled against the most likely candidate, not against 1.0. The
// alternative -- full bar means p=1 -- spends almost the whole width on a
// range the model is rarely in, and collapses every row of a confident
// step into the same empty sliver, which is exactly the case the column
// exists to distinguish from a flat one. Here the top row is always full
// and the others are read against it: one long bar over slivers is a
// confident step, a stack of similar bars is a coin-flip.
//
// sqrt because the tail is where the shape lives. Linear ratios put
// everything below a few percent of the leader into the same first
// eighth; the square root spreads that range out without reordering
// anything. It is a perceptual scale, not a quantity to measure off --
// the exact number is printed immediately to its right.
function probBar(ratio, cells) {
  const eighths = Math.max(1, Math.round(Math.sqrt(ratio) * cells * 8));
  const full = Math.floor(eighths / 8);
  const bar = "█".repeat(Math.min(full, cells)) +
    (full < cells ? EIGHTHS[eighths % 8] : "");
  // Padded so the percentages stay in a column of their own rather than
  // shuffling left and right with the bar under them.
  return bar.padEnd(cells, " ");
}

// Shared by the live column and the inspector below, which differ only in
// how many rows they have room for and how wide the bars are.
function candidateRows({ chosen, items }, { count, cells }) {
  // Every bar is relative to the leader, and the worker sends the list
  // already sorted, so that is items[0] -- guarded anyway because a zero
  // here would put NaN in every row.
  const top = items[0]?.prob || 1;
  const rows = document.createDocumentFragment();

  for (const item of items.slice(0, count)) {
    const row = document.createElement("div");
    row.className = item.id === chosen ? "candidate chosen" : "candidate";

    const mark = document.createElement("span");
    mark.className = "candidate-mark";
    mark.textContent = item.id === chosen ? ">" : " ";

    const text = document.createElement("span");
    text.className = "candidate-token";
    text.textContent = tokenGlyphs(item.text);

    const bar = document.createElement("span");
    bar.className = "candidate-bar";
    bar.textContent = probBar(item.prob / top, cells);

    const prob = document.createElement("span");
    prob.className = "candidate-prob";
    prob.textContent = `${(item.prob * 100).toFixed(1)}%`;

    row.append(mark, text, bar, prob);
    rows.append(row);
  }

  return rows;
}

function renderCandidates(message) {
  const wide = wideEnough.matches;
  candidatesEl.replaceChildren(
    candidateRows(message, { count: wide ? 15 : 7, cells: wide ? 4 : 2 })
  );
}

// ------------------------------ inspector -------------------------------
// The live column is gone the moment the reply finishes, which is exactly
// when someone reads the reply and wonders how close a particular word was
// to being some other word. So each token of the most recent reply keeps
// the distribution it was drawn from, and asking for it -- hovering with a
// pointer, tapping on a touchscreen -- puts that column back, next to the
// word instead of at the edge of the screen.
//
// Only the most recent reply: the distributions are ~15 objects per token
// and a long conversation would accumulate them for every word ever
// generated, for a question nobody asks about a reply five turns back.
const inspectorEl = document.getElementById("inspector");

// Long enough to read a dozen rows, short enough that a stray tap does not
// leave a panel sitting over the text. Touch only -- a pointer dismisses
// it by moving away, which needs no timer.
const TAP_LINGER_MS = 10000;

// Keyed by the token's <span>, so a reply dropped from the DOM takes its
// distributions with it rather than needing to be swept.
const distributions = new WeakMap();

// The candidates message for a token arrives before the token message that
// carries its text, and one visible piece can take more than one token
// (the stream decoder holds a multi-byte character back until it is
// complete). So they queue here and are handed to the next piece that
// appears; sets[0] is the draw that produced its first byte, which is the
// choice the word is actually about.
let pendingCandidates = [];
let inspectableBubble = null;
let inspecting = null;
let lingerTimer = null;

function hideInspector() {
  clearTimeout(lingerTimer);
  lingerTimer = null;
  inspecting?.classList.remove("inspecting");
  inspecting = null;
  inspectorEl.hidden = true;
}

// Above the token by preference -- a panel below it covers the rest of the
// line being read. Clamped into the viewport on both axes, because a token
// near an edge is the common case, not the exception.
function positionInspector(span) {
  const gap = 8;
  const target = span.getBoundingClientRect();
  const panel = inspectorEl.getBoundingClientRect();

  const left = Math.min(
    Math.max(gap, target.left + target.width / 2 - panel.width / 2),
    window.innerWidth - panel.width - gap
  );
  let top = target.top - panel.height - gap;
  if (top < gap) {
    top = Math.min(target.bottom + gap, window.innerHeight - panel.height - gap);
  }

  inspectorEl.style.left = `${Math.max(gap, left)}px`;
  inspectorEl.style.top = `${Math.max(gap, top)}px`;
}

function showInspector(span, linger) {
  const sets = distributions.get(span);
  if (!sets?.length) return;

  if (span !== inspecting) {
    inspecting?.classList.remove("inspecting");
    inspecting = span;
    span.classList.add("inspecting");
    // Always the full percentages here, unlike the live column: this one
    // was asked for, and it is not competing with the reply for width.
    inspectorEl.replaceChildren(
      candidateRows(sets[0], { count: wideEnough.matches ? 15 : 8, cells: 4 })
    );
    // Unhidden before measuring -- a display:none element has no size.
    inspectorEl.hidden = false;
    positionInspector(span);
  }

  clearTimeout(lingerTimer);
  lingerTimer = linger ? setTimeout(hideInspector, TAP_LINGER_MS) : null;
}

function tokenAt(event) {
  const el = event.target instanceof Element ? event.target : null;
  return el?.closest(".bubble.inspectable .token") ?? null;
}

// Hover, for pointers. Delegated to the transcript rather than bound per
// token: a reply is hundreds of spans, and they are created one per token
// while the page is already busy running a forward pass.
transcript.addEventListener("pointerover", (event) => {
  if (event.pointerType !== "mouse") return;
  const span = tokenAt(event);
  if (span) showInspector(span, false);
  else if (!lingerTimer) hideInspector();
});

transcript.addEventListener("pointerleave", () => {
  if (!lingerTimer) hideInspector();
});

// Tap, for touchscreens -- and a click on a pointer, which just pins the
// panel that hovering already showed. Anywhere else in the transcript
// dismisses it, so it never has to be waited out.
transcript.addEventListener("click", (event) => {
  const span = tokenAt(event);
  if (span) showInspector(span, true);
  else hideInspector();
});

// The panel is positioned against the viewport, so anything that moves the
// token out from under it invalidates it.
transcript.addEventListener("scroll", hideInspector, { passive: true });
window.addEventListener("resize", hideInspector);

// -------------------------------- hint modal -----------------------------
// The hover/tap-to-inspect affordance above has no other visual cue, so it
// gets pointed out once, right after the first reply makes it available.
// sessionStorage rather than a module-level flag: a reload mid-session
// should not show it again, but a new tab starts the hint fresh.
const HINT_SHOWN_KEY = "llm-in-a-tab:inspector-hint-shown";

function maybeShowInspectorHint() {
  if (sessionStorage.getItem(HINT_SHOWN_KEY)) return;
  sessionStorage.setItem(HINT_SHOWN_KEY, "1");
  hintModal.hidden = false;
}

hintModalOk.addEventListener("click", () => {
  hintModal.hidden = true;
});

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

// Drives both the button swap and the CSS that dims the transcript and
// composer down to near-nothing, so the visualization behind them has the
// screen while the model is actually working.
function setGenerating(isGenerating) {
  sendButton.hidden = isGenerating;
  stopButton.hidden = !isGenerating;
  input.disabled = isGenerating;
  chat.dataset.generating = isGenerating ? "true" : "false";
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

// Collapsed by default -- the sliders are rarely touched, and hiding them
// gives the visualization behind the composer that much more screen.
settingsToggle.addEventListener("click", () => {
  const open = sampling.hidden;
  sampling.hidden = !open;
  settingsToggle.setAttribute("aria-expanded", String(open));
  settingsToggle.classList.toggle("active", open);
});

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
      viz.configure(message.viz);
      log(`${message.label} ready.`);
      break;
    }

    case "reply-start":
      // Clear rather than leave the last reply's final distribution to
      // flash for the ~500ms until the first token of this one arrives.
      candidatesEl.replaceChildren();
      // The previous reply stops being the one you can ask about the
      // moment a new one starts, not when it finishes.
      hideInspector();
      inspectableBubble?.classList.remove("inspectable");
      inspectableBubble = null;
      pendingCandidates = [];
      streamingBubble = addBubble("assistant");
      streamingBubble.classList.add("streaming");
      scrollTranscript();
      viz.start();
      break;

    case "activations":
      viz.push(message);
      break;

    case "candidates":
      renderCandidates(message);
      pendingCandidates.push(message);
      break;

    case "prefill-done":
      hud.prefill.textContent =
        `prefill ${message.tokens}tok ${(message.ms / 1000).toFixed(1)}s`;
      updateCtx(message.seqLen, message.maxCtx);
      break;

    case "token": {
      if (streamingBubble) {
        // A span per token rather than one growing text node, so each
        // piece of the reply is a thing the pointer can be over. Marked
        // .token only when it has a distribution to show -- the flushed
        // tail after the loop ends does not.
        const span = document.createElement("span");
        span.textContent = message.text;
        if (pendingCandidates.length) {
          span.className = "token";
          distributions.set(span, pendingCandidates);
          pendingCandidates = [];
        }
        streamingBubble.append(span);
        scrollTranscript();
      }
      updateCtx(message.seqLen, message.maxCtx);
      hud.speed.textContent = `${message.tokensPerSecond.toFixed(2)} tok/s`;
      break;
    }

    case "reply-done": {
      if (streamingBubble) {
        streamingBubble.classList.remove("streaming");
        inspectableBubble = streamingBubble;
        streamingBubble.classList.add("inspectable");
        if (message.reason === "stopped") streamingBubble.append(" [stopped]");
        if (message.reason === "budget") streamingBubble.append(" [token limit]");
        streamingBubble = null;
        maybeShowInspectorHint();
      }
      hud.speed.textContent = `${message.tokensPerSecond.toFixed(2)} tok/s`;
      setMem(message.residentMB);
      updateCtx(message.seqLen, message.maxCtx);
      stopButton.disabled = false;
      setGenerating(false);
      viz.stop();
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
      viz.stop();
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
        viz.stop();
      }
      break;
  }
};

worker.onerror = (event) => {
  log(`Worker error: ${event.message}`);
};

refreshCacheStates();
