// The background visualization: a tower of transformer layers that the
// camera falls through, one full stack per generated token.
//
// Nothing here is decorative noise. Every cell in a layer's grid is one
// entry of that layer's SwiGLU activation vector for the token being
// generated, every filament in the fan above it is that layer's
// head-averaged attention onto a slice of the live context, and the trace
// below it is the residual stream leaving the layer. The bytes come from
// js/model/probe.js, one 30- or 32-layer frame per token, and the descent
// rate is set by measured token time -- so the speed of the fall is an
// honest readout of tok/s.
//
// WHY THE CAMERA NEVER CUTS. A token takes 150ms-1s, which is 5-30ms per
// layer. Cutting to each layer in turn at that rate is a strobe, and
// panning down the stack and then snapping back to the top once per token
// is a strobe with extra steps. So the tower is treated as endless
// instead: each token's layers are laid out below the previous token's,
// separated by a band naming the token that pass emitted, and the camera
// simply falls at whatever speed keeps it about three-quarters of a token
// behind the worker. No cuts anywhere, and you watch tokens stream past
// as you descend.
//
// EVERYTHING IS drawImage, NOT fillRect. A layer's grid is painted into a
// tiny offscreen canvas at one device pixel per activation, then blitted
// with an affine transform and smoothing off. That is one drawImage per
// layer instead of ~1500 fillRects, which is the difference between this
// being free on a phone and it being the reason the phone gets hot.
import { NEURON_LUT, COLORS } from "./palette.js";

const GAP = 1; // layer slots between one token's stack and the next
const EDGE_SLICES = 6; // depth of the extruded side face, in layers
const SLOPE_Y = 0.05; // grid tips down to the right
const SLOPE_X = 0.34; // rows lean right as they descend
// Threshold for a bloom sprite, chosen against the measured activation
// distribution rather than by eye: ~0.5% of neurons exceed half their
// layer's peak, which is about eight per layer -- a constellation. At 232
// it was the top 0.08%, or one.
const HOT_BYTE = 192;
const MAX_HOT = 40;
const DRAW_SPAN = 2; // layer slots drawn either side of the camera

const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);

// Reshape a flat activation vector into the widest grid that divides it
// exactly, preferring an aspect near `target`. Exactness matters: a
// padded grid puts a ragged strip of dead cells along one edge of every
// layer, which reads as a rendering bug. SmolLM2 cooperates -- 1536 gives
// 96x16 at target 6, and 2560 gives 128x20.
function gridDims(n, target) {
  const ideal = Math.sqrt(n * target);
  let best = null;
  for (let w = Math.max(1, Math.round(ideal * 0.62)); w <= Math.round(ideal * 1.6); w++) {
    if (n % w) continue;
    const err = Math.abs(Math.log(w / (n / w) / target));
    if (!best || err < best.err) best = { w, h: n / w, err };
  }
  if (best) return { w: best.w, h: best.h };
  const w = Math.max(1, Math.round(ideal));
  return { w, h: Math.ceil(n / w) };
}

// Soft additive bloom for the hottest neurons, pre-rendered once. Drawn
// with 'lighter' rather than using shadowBlur, which would recompute a
// gaussian per sprite per frame.
function makeBloom(size) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, "rgba(226, 255, 226, 0.95)");
  grad.addColorStop(0.22, "rgba(130, 255, 168, 0.42)");
  grad.addColorStop(1, "rgba(75, 240, 122, 0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  return c;
}

export function createStackViz(canvas) {
  const ctx = canvas.getContext("2d");
  const bloom = makeBloom(64);

  let cfg = null; // { layers, mlpCells, residCells, attnBins, stride }
  let slot = 0; // layers + GAP: one token's worth of tower
  let grid = { w: 0, h: 0 };
  let cellCanvas = null;
  let cellCtx = null;
  let cellImage = null;

  let dpr = 1;
  let W = 0;
  let Hpx = 0;
  let compact = false;

  const frames = []; // newest last, pruned to what the camera still needs
  let nextIndex = 0;
  let camG = 0; // camera position, in layer slots, monotonically increasing
  let arrivalEma = 380; // ms per token, smoothed
  let lastArrival = 0;
  // Token arrivals are timed against the same clock the render loop
  // advances on, not against performance.now() directly. They are the same
  // thing in the app, but measuring production on one clock and playback on
  // another means the camera's speed controller is comparing incomparable
  // numbers the moment anything drives the loop differently -- which is
  // exactly what tests/viz-harness.html does, and it pinned the camera to
  // the bottom of the stack.
  let animClock = 0;

  let enabled = true;
  let running = false;
  let fade = 0; // global opacity, eased in on reply-start and out on done
  let raf = 0;
  let last = 0;

  // ------------------------------ sizing --------------------------------

  function resize() {
    // Capped rather than raw: a phone at devicePixelRatio 3 would have us
    // shading three times the fragments for a background nobody is
    // reading pixel-peeped, while competing with the worker for the CPU.
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    W = Math.max(1, rect.width);
    Hpx = Math.max(1, rect.height);
    compact = W < 720;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(Hpx * dpr);
    if (cfg) layoutGrid();
  }

  function layoutGrid() {
    // Portrait phones get a much squarer grid. A 6:1 ribbon works on a wide
    // desktop, where it reads as one of the stacked plates of a tower, but
    // on a 375px-wide screen it is a 120px-tall sliver adrift in a very tall
    // viewport. 1536 cells resolve to 96x16 on desktop and 48x32 here.
    grid = gridDims(cfg.mlpCells, compact ? 1.6 : 6);
    if (!cellCanvas) {
      cellCanvas = document.createElement("canvas");
      cellCtx = cellCanvas.getContext("2d");
    }
    if (cellCanvas.width !== grid.w || cellCanvas.height !== grid.h) {
      cellCanvas.width = grid.w;
      cellCanvas.height = grid.h;
      cellImage = cellCtx.createImageData(grid.w, grid.h);
    }
  }

  // ------------------------------- data ---------------------------------

  function configure(geometry) {
    cfg = geometry;
    slot = cfg.layers + GAP;
    frames.length = 0;
    nextIndex = 0;
    camG = 0;
    lastArrival = 0;
    arrivalEma = 380;
    layoutGrid();
  }

  function push(msg) {
    if (!cfg || !enabled) return;

    // Clamped to a plausible token time before it is folded in. The first
    // arrival of a *second* reply is measured against a clock last updated
    // during the first one, so the raw interval includes however long the
    // user spent typing -- which would otherwise slow the camera to a crawl
    // for the next several tokens.
    if (lastArrival) {
      const gap = clamp(animClock - lastArrival, 60, 4000);
      arrivalEma = arrivalEma * 0.72 + gap * 0.28;
    }
    lastArrival = animClock;

    // Mean |activation| per layer, for the stack rail. Computed once on
    // arrival rather than per frame: the same layer is on screen for
    // dozens of frames as the camera passes it.
    const energy = new Float32Array(cfg.layers);
    for (let l = 0; l < cfg.layers; l++) {
      const base = l * cfg.stride;
      let sum = 0;
      for (let c = 0; c < cfg.mlpCells; c++) sum += Math.abs(msg.bytes[base + c] - 128);
      energy[l] = sum / cfg.mlpCells / 127;
    }

    const frame = {
      index: nextIndex++,
      bytes: msg.bytes,
      attnUsed: msg.attnUsed,
      pos: msg.pos,
      text: msg.text,
      energy,
    };
    if (frames.length === 0) camG = frame.index * slot;
    frames.push(frame);
    prune();
    if (running) ensureRaf();
  }

  function prune() {
    const oldestNeeded = Math.floor(camG / slot) - 1;
    while (frames.length > 6 || (frames.length > 1 && frames[0].index < oldestNeeded)) {
      frames.shift();
    }
  }

  function frameAt(index) {
    for (let i = frames.length - 1; i >= 0; i--) if (frames[i].index === index) return frames[i];
    return null;
  }

  // ----------------------------- animation ------------------------------

  function step(dt) {
    if (running) fade = Math.min(1, fade + dt / 260);
    else fade = Math.max(0, fade - dt / 620);

    if (!frames.length) return;

    // The camera chases the newest frame but stays deliberately behind it,
    // so a token that arrives slightly late does not leave the camera
    // stalled at a wall with nothing below it. The catch-up term drains a
    // backlog (a fast burst of tokens) without ever running the camera
    // backwards.
    // The floor is the newest token's *last layer*, not the end of its
    // slot. The gap after it holds the token band, and descending past
    // that means descending into a stack the worker has not produced yet:
    // the camera parks below the band with nothing under it and the screen
    // goes empty. Stopping on the last real layer means a stall looks like
    // waiting on a layer, which is what is actually happening.
    const newest = frames[frames.length - 1];
    const maxG = newest.index * slot + cfg.layers - 1;
    const behind = (maxG - camG) / slot;
    const factor = 1 + Math.max(0, behind - 0.75) * 0.9;
    const speed = (slot / Math.max(arrivalEma, 80)) * factor;
    camG = Math.min(camG + speed * dt, maxG);
    prune();
  }

  // ----------------------------- rendering ------------------------------

  // Paints one layer's activation bytes into the offscreen cell canvas,
  // one device pixel per neuron.
  function paintCells(frame, layer) {
    const px = cellImage.data;
    const base = layer * cfg.stride;
    const n = cfg.mlpCells;
    for (let c = 0; c < n; c++) {
      const o = frame.bytes[base + c] * 4;
      const q = c * 4;
      px[q] = NEURON_LUT[o];
      px[q + 1] = NEURON_LUT[o + 1];
      px[q + 2] = NEURON_LUT[o + 2];
      px[q + 3] = NEURON_LUT[o + 3];
    }
    // Only reachable if a model's activation count has no exact grid and
    // gridDims() had to pad; the trailing cells are transparent, not black.
    for (let c = n * 4; c < px.length; c++) px[c] = 0;
    cellCtx.putImageData(cellImage, 0, 0);
  }

  // The affine map from grid space (columns, rows) to screen space. Both
  // shears are small; together they read as a plate turned a few degrees
  // in two axes, which is enough depth cue without the heavy 2:1 isometric
  // squash that would waste half the screen on empty diagonal.
  function gridMatrix(cx, cy, wPx, hPx) {
    const a = wPx / grid.w;
    const d = hPx / grid.h;
    const b = a * SLOPE_Y;
    const c = d * SLOPE_X;
    // Centre the parallelogram's bounding box on (cx, cy).
    const ox = cx - (wPx + hPx * SLOPE_X) / 2;
    const oy = cy - (wPx * SLOPE_Y + hPx) / 2;
    return { a, b, c, d, ox, oy };
  }

  // Width of an unscaled plate. Leaves room for the shear: the row lean
  // adds hPx*SLOPE_X to the bounding box, so a plate sized to the full
  // viewport runs off both edges once it is tilted.
  function plateWidth() {
    return Math.min(compact ? W * 0.86 : 1500, W * (compact ? 0.86 : 0.78));
  }

  function drawLayerSlab(frame, layer, y, s, alpha) {
    const wPx = plateWidth() * s;
    const hPx = (wPx / grid.w) * grid.h;
    const cx = W * (compact ? 0.5 : 0.53);
    const m = gridMatrix(cx, y, wPx, hPx);
    const focused = s > 0.82;

    // Extruded side face: the same layer's predecessors in this same
    // forward pass, offset up-left so only a sliver of each shows past the
    // front plate. It is the striped edge of a stack of 2D slices, and the
    // stripes are real -- slice i is layer (layer - 1 - i).
    if (focused && layer > 0) {
      const ex = 10 * s;
      const ey = ex * 0.44;
      for (let i = EDGE_SLICES; i >= 1; i--) {
        const src = layer - i;
        if (src < 0) continue;
        paintCells(frame, src);
        ctx.save();
        ctx.globalAlpha = alpha * 0.3 * (1 - i / (EDGE_SLICES + 2));
        ctx.transform(m.a, m.b, m.c, m.d, m.ox - ex * i, m.oy - ey * i);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(cellCanvas, 0, 0);
        ctx.restore();
      }
    }

    const P = (gx, gy) => [m.ox + m.a * gx + m.c * gy, m.oy + m.b * gx + m.d * gy];
    const corners = [P(0, 0), P(grid.w, 0), P(grid.w, grid.h), P(0, grid.h)];
    const outline = () => {
      ctx.beginPath();
      ctx.moveTo(corners[0][0], corners[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(corners[i][0], corners[i][1]);
      ctx.closePath();
    };

    // Opaque backing under the front plate. Without it the extruded
    // slices behind show through every transparent cell of the front face
    // -- which is most of them, since near-zero activations are near-
    // transparent by design -- and the layer turns into a grey haze
    // instead of a solid object with a stack of slices behind it.
    ctx.save();
    ctx.globalAlpha = alpha * 0.93;
    ctx.fillStyle = "#05070a";
    outline();
    ctx.fill();
    ctx.restore();

    paintCells(frame, layer);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.transform(m.a, m.b, m.c, m.d, m.ox, m.oy);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(cellCanvas, 0, 0);
    ctx.restore();

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = focused ? COLORS.frame : COLORS.frameDim;
    ctx.lineWidth = focused ? 1.1 : 0.8;
    outline();
    ctx.stroke();
    ctx.restore();

    if (focused) {
      drawHotNeurons(frame, layer, m, alpha, wPx / grid.w);
      drawAttentionFan(frame, layer, corners, alpha, s);
      drawResidualTrace(frame, layer, corners, alpha, s);
      drawLayerLabel(frame, layer, corners, alpha);
    }
    return corners;
  }

  // A bloom sprite on the cells above a fixed fraction of the layer's own
  // peak. Capped, because on a layer where everything fires the cap is the
  // difference between a highlight and a white screen.
  function drawHotNeurons(frame, layer, m, alpha, cellPx) {
    const base = layer * cfg.stride;
    const size = Math.max(9, cellPx * 3.4);
    let drawn = 0;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = alpha * 0.85;
    for (let c = 0; c < cfg.mlpCells && drawn < MAX_HOT; c++) {
      if (frame.bytes[base + c] < HOT_BYTE) continue;
      const gx = c % grid.w;
      const gy = (c / grid.w) | 0;
      const x = m.ox + m.a * (gx + 0.5) + m.c * (gy + 0.5);
      const y = m.oy + m.b * (gx + 0.5) + m.d * (gy + 0.5);
      ctx.drawImage(bloom, x - size / 2, y - size / 2, size, size);
      drawn++;
    }
    ctx.restore();
  }

  // Filaments converging from a rail spanning the context onto the token
  // being generated -- what this layer is reading, and how hard.
  //
  // Batched into alpha tiers so the whole fan is a handful of strokes
  // rather than one per filament.
  function drawAttentionFan(frame, layer, corners, alpha, s) {
    const bins = frame.attnUsed;
    if (!bins) return;
    const off = layer * cfg.stride + cfg.mlpCells + cfg.residCells;
    const x0 = corners[0][0];
    const x1 = corners[1][0];
    const railY = Math.min(corners[0][1], corners[1][1]) - 96 * s;
    // The focal point rides *on* the plate's top edge rather than at a
    // fixed height above it. The edge is sheared, so a flat offset leaves
    // the fan converging in mid-air next to the layer it is feeding.
    const k = 0.78;
    const fx = x0 + (x1 - x0) * k;
    const fy = corners[0][1] + (corners[1][1] - corners[0][1]) * k - 2 * s;

    const TIERS = 5;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (let t = TIERS; t >= 1; t--) {
      const lo = (t - 1) / TIERS;
      const hi = t / TIERS;
      ctx.beginPath();
      let any = false;
      for (let b = 0; b < bins; b++) {
        const w = frame.bytes[off + b] / 255;
        const v = Math.pow(w, 0.55);
        if (v < lo || v >= hi || v < 0.04) continue;
        const rx = x0 + (x1 - x0) * ((b + 0.5) / bins);
        ctx.moveTo(rx, railY);
        ctx.quadraticCurveTo((rx + fx) / 2, railY + (fy - railY) * 0.68, fx, fy);
        any = true;
      }
      if (!any) continue;
      // Deliberately near-invisible per filament. There are up to 128 of
      // them converging on one point under 'lighter', so anything that
      // reads as a visible line on its own composites into a solid white
      // wedge where they meet.
      ctx.strokeStyle = `rgba(${COLORS.fan}, ${(alpha * 0.05 * t).toFixed(3)})`;
      ctx.lineWidth = 0.6;
      ctx.stroke();
    }
    ctx.restore();

    // The rail itself, so the fan reads as reaching across a context
    // rather than floating.
    ctx.save();
    ctx.globalAlpha = alpha * 0.5;
    ctx.strokeStyle = COLORS.labelDim;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, railY);
    ctx.lineTo(x1, railY);
    ctx.stroke();
    ctx.restore();
  }

  // The residual stream leaving the layer, as an oscilloscope trace. It is
  // the one vector every layer writes into and reads back out of, so it
  // gets a different visual language from the neuron grid: a continuous
  // cyan signal under the plate rather than a field of discrete cells.
  function drawResidualTrace(frame, layer, corners, alpha, s) {
    const base = layer * cfg.stride + cfg.mlpCells;
    const n = cfg.residCells;
    const x0 = corners[3][0];
    const x1 = corners[2][0];
    const y0 = Math.max(corners[2][1], corners[3][1]) + 16 * s;
    const amp = 26 * s;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = x0 + ((x1 - x0) * i) / (n - 1);
      const y = y0 - ((frame.bytes[base + i] - 128) / 127) * amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = COLORS.trace;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.lineTo(x1, y0);
    ctx.lineTo(x0, y0);
    ctx.closePath();
    ctx.fillStyle = COLORS.traceFill;
    ctx.fill();
    ctx.restore();
  }

  function drawLayerLabel(frame, layer, corners, alpha) {
    const x = corners[3][0];
    const y = Math.max(corners[2][1], corners[3][1]) + 58;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = "11px ui-monospace, Menlo, Consolas, monospace";
    ctx.textBaseline = "top";
    ctx.fillStyle = COLORS.label;
    ctx.fillText(`L${String(layer).padStart(2, "0")}/${cfg.layers}`, x, y);
    ctx.fillStyle = COLORS.labelDim;
    ctx.fillText(
      compact
        ? `attn ${frame.pos + 1}`
        : `mlp ${cfg.mlpCells}  resid ${cfg.residCells}  attn ${frame.pos + 1}`,
      x + 74,
      y
    );
    ctx.restore();
  }

  // The band between one token's stack and the next, captioned with the
  // token that pass emitted. This is the thing that makes the endless
  // tower legible: without it a fall through 30 layers looks the same as a
  // fall through 60.
  function drawTokenBand(frame, y, alpha) {
    const w = Math.min(compact ? W * 0.9 : 1120, W * 0.9);
    const x = W * (compact ? 0.5 : 0.53) - w / 2;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = COLORS.goldDim;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x + w, y - 5);
    ctx.moveTo(x, y + 5);
    ctx.lineTo(x + w, y + 5);
    ctx.stroke();

    const raw = (frame.text || "").replace(/\n/g, "\\n");
    const label = raw ? `token  "${raw}"` : "token  ...";
    ctx.font = "12px ui-monospace, Menlo, Consolas, monospace";
    ctx.textBaseline = "middle";
    const tw = ctx.measureText(label).width + 20;
    ctx.fillStyle = "rgba(6, 8, 10, 0.9)";
    ctx.fillRect(x + w / 2 - tw / 2, y - 9, tw, 18);
    ctx.fillStyle = COLORS.gold;
    ctx.textAlign = "center";
    ctx.fillText(label, x + w / 2, y);
    ctx.restore();
  }

  // A miniature of the whole stack down the left edge, brightness per
  // layer from its mean activation energy, with a caret at the camera.
  // Instrumentation, and the only place the full shape of the model is
  // visible at once while the camera is inside it.
  function drawRail(frame, activeLayer, alpha) {
    if (!frame) return;
    const x = compact ? 13 : 24;
    const top = Hpx * 0.17;
    const bot = Hpx * 0.85;
    const n = cfg.layers;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = COLORS.labelDim;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bot);
    ctx.stroke();

    for (let l = 0; l < n; l++) {
      const y = top + ((bot - top) * l) / (n - 1);
      const e = clamp(frame.energy[l] * 3.2, 0.06, 1);
      const here = l === activeLayer;
      ctx.globalAlpha = alpha * (here ? 1 : 0.4 + 0.6 * e);
      ctx.strokeStyle = here ? COLORS.gold : COLORS.frame;
      ctx.lineWidth = here ? 2.5 : 1.4;
      ctx.beginPath();
      ctx.moveTo(x - (here ? 8 : 3), y);
      ctx.lineTo(x + 5 + e * (compact ? 10 : 22), y);
      ctx.stroke();
    }

    ctx.globalAlpha = alpha;
    ctx.font = "9px ui-monospace, Menlo, Consolas, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillStyle = COLORS.labelDim;
    ctx.fillText("STACK", x - 4, top - 8);
    ctx.textBaseline = "top";
    ctx.fillText(`${n}L`, x - 4, bot + 8);
    ctx.restore();
  }

  function render() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (fade <= 0.004 || !cfg || !frames.length) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const cy = Hpx * 0.47;
    // Derived from how tall a plate actually is rather than from the
    // viewport alone, so the stack stays a stack: a fixed fraction of the
    // height leaves the short desktop plates crowded and the tall phone
    // plates floating in empty space.
    const spacing = Math.max(((plateWidth() / grid.w) * grid.h) * 1.7, Hpx * 0.33);
    const centre = Math.round(camG);

    // Farthest first, so nearer layers overlay them.
    const order = [];
    for (let d = -DRAW_SPAN; d <= DRAW_SPAN; d++) order.push(centre + d);
    order.sort((p, q) => Math.abs(q - camG) - Math.abs(p - camG));

    let activeFrame = null;
    let activeLayer = 0;

    for (const g of order) {
      if (g < 0) continue;
      const z = g - camG;
      const s = 1 / (1 + Math.abs(z) * 0.42);
      const a = clamp(1.12 - Math.abs(z) * 0.5, 0, 1) * fade;
      if (a <= 0.012) continue;
      const y = cy + z * spacing;

      const index = Math.floor(g / slot);
      const layer = g - index * slot;
      const frame = frameAt(index);
      if (!frame) continue;

      if (layer >= cfg.layers) {
        drawTokenBand(frame, y, a);
        continue;
      }
      drawLayerSlab(frame, layer, y, s, a);
      if (Math.abs(z) < 0.5) {
        activeFrame = frame;
        activeLayer = layer;
      }
    }

    drawRail(activeFrame || frames[frames.length - 1], activeLayer, fade * 0.85);
  }

  // ------------------------------- loop ---------------------------------

  function tick(now) {
    const dt = Math.min(now - last, 80);
    last = now;
    animClock = now;
    step(dt);
    render();
    // Idle costs nothing: once faded out with no reply in flight the loop
    // stops entirely rather than clearing an empty canvas 60 times a
    // second behind a page nobody is generating on.
    if (!running && fade <= 0.004) {
      raf = 0;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    raf = requestAnimationFrame(tick);
  }

  function ensureRaf() {
    if (raf) return;
    last = animClock;
    raf = requestAnimationFrame(tick);
  }

  // ------------------------------- api ----------------------------------

  function start() {
    if (!enabled) return;
    lastArrival = 0;
    running = true;
    ensureRaf();
  }

  function stop() {
    running = false;
    // Left running so the camera finishes falling through whatever the
    // worker already produced instead of freezing mid-plate, and fades out
    // from there.
    ensureRaf();
  }

  function setEnabled(on) {
    enabled = on;
    if (!on) {
      running = false;
      fade = 0;
      frames.length = 0;
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  // A ResizeObserver on the canvas itself rather than a window/
  // visualViewport resize listener: the canvas is sized in CSS from
  // --vv-height, which main.js writes from a visualViewport event, so a
  // listener here would be racing that write and would read a stale box
  // exactly when the size changed. Observing the element measures the
  // result instead of predicting it, and it also covers the first layout,
  // which happens after this module runs.
  resize();
  new ResizeObserver(resize).observe(canvas);

  return { configure, push, start, stop, setEnabled, resize };
}
