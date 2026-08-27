// The background visualization: a tower of transformer layers that the
// camera falls through, one full stack per generated token.
//
// Nothing here is decorative noise. Each layer contributes two plates,
// in the order the block computes them:
//
//   ATTN -- the real attention matrix for this token. Rows are query
//           heads, columns are cached context positions, and every entry
//           is one post-softmax weight the model actually used. One
//           screen pixel per matrix entry, no binning.
//   MLP  -- every entry of that layer's SwiGLU activation vector, the
//           thing interpretability work calls neurons, with the residual
//           stream leaving the layer traced underneath.
//
// The bytes come from js/model/probe.js, one frame per token, and the
// descent rate is set by measured token time -- so the speed of the fall
// is an honest readout of tok/s.
//
// WHY THE CAMERA CUTS. The tower is endless -- each token's plates are
// laid out below the previous token's, separated by a band naming the
// token that pass emitted -- and the camera only ever moves down it. But
// it moves in whole layers. It holds, then jumps two plates, rather than
// sliding.
//
// This is the second attempt. Falling smoothly was the obvious reading of
// "descend through the layers", and it is unreadable: everything on screen
// is in motion, so the eye tracks the motion instead of the cells, and the
// thing you are meant to be watching -- which neurons fired -- goes past as
// a blur. Cutting fixes it by making the frame the stationary part. A
// plate's transform depends only on its integer distance from the camera,
// so the next layer's cells land on exactly the pixels the last one used
// and the plate reads as a panel of lights that changes rather than a
// surface that travels. Motion is reserved for the one thing it is good
// at: signalling that something happened.
//
// A layer and not a plate, because plates alternate ATTN, MLP, ATTN, MLP.
// Stepping one at a time swaps the type of every plate on screen at every
// cut -- a tall green neuron grid trades places with a short amber
// attention matrix, twice per layer -- which flickers worse than the slide
// it replaced. Stepping two keeps each slot's type fixed, so a cut changes
// only which layer's numbers are in it. The camera then sits half a slot
// low (EYE), framing the current layer's attention plate and MLP plate
// together rather than centring one and demoting the other.
//
// The camera still tracks a continuous position internally (camG) -- the
// pacing controller needs fractions of a plate to chase the worker with --
// and only the rendered position (camV) is quantized.
//
// WHY IT DOES NOT SHOW EVERY LAYER OF EVERY TOKEN. It cannot. Thirty
// layers is sixty plates, and 135M runs at up to 6 tok/s, which is 2.7ms
// per plate -- a sixth of one 60Hz frame. A camera that honestly visited
// all of them would cross six plates between two rendered frames, and
// consecutive frames would show unrelated layers: a strobe, and one that
// is *worse* the faster the model runs, which is exactly backwards.
//
// So the descent rate is the fixed thing and the depth per token is the
// variable one. Each token contributes a run of plates long enough to
// take about PLATE_MS each at the measured token time -- one layer per
// token at 6 tok/s, four or five at the 360M's ~1 tok/s -- and the next
// token's run picks up at the layer where the last one stopped, wrapping
// at the end of the model. Nothing is faked: every plate is real data
// from the token whose band follows it, at the layer its label names, and
// the STACK rail shows which slice of the model is currently on screen.
// What you lose is the guarantee that one token means one lap of the
// tower. What you get is a fall you can actually watch, and the faster
// the model runs the more of the model streams past per second, which is
// the right way round.
//
// EVERYTHING IS drawImage, NOT fillRect. A plate is painted into a tiny
// offscreen canvas at one device pixel per value, then blitted with an
// affine transform and smoothing off. That is one drawImage per plate
// instead of thousands of fillRects, which is the difference between this
// being free on a phone and it being the reason the phone gets hot.
import { NEURON_LUT, ATTN_LUT, COLORS } from "./palette.js";

// The token band is drawn *between* two plates rather than being given a
// slot of its own. At 6 tok/s a run is one layer -- two plates -- so a
// band that cost a slot would leave the camera staring at an empty screen
// a third of the time.
const GAP = 0;
// Target time on screen for one plate. The whole pacing scheme exists to
// hold this roughly constant whatever the model's token rate. The camera
// cuts a layer -- two plates -- at a time, so this is half the dwell: the
// picture holds for 170-330ms across the range of token rates the two
// models produce, which is long enough to read and short enough to feel
// like the model is working.
const PLATE_MS = 110;
const EDGE_SLICES = 6; // depth of the extruded side face, in layers
const SLOPE_Y = 0.05; // grid tips down to the right
const SLOPE_X = 0.34; // rows lean right as they descend
// Threshold for a bloom sprite, chosen against the measured activation
// distribution rather than by eye: ~0.5% of neurons exceed half their
// layer's peak, which is about eight per layer -- a constellation. At 232
// it was the top 0.08%, or one.
const HOT_BYTE = 192;
const MAX_HOT = 40;
// Plate slots drawn either side of the camera. Three, not two, because a
// layer is now two plates: at two you can never see a whole layer plus
// its neighbours, which is the shape the tower is supposed to have.
const DRAW_SPAN = 3;
const ATTN_ROW_PX = 15; // height of one head's row, desktop
const ATTN_ROW_PX_COMPACT = 10;
const FAN_FILAMENTS = 128;

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

  let cfg = null; // { layers, mlpCells, residCells, heads, maxCols, attnBase }
  let plates = 0; // 2 * layers: one lap of the model, attn+mlp per layer
  let grid = { w: 0, h: 0 };
  let cellCanvas = null;
  let cellCtx = null;
  let cellImage = null;
  let attnCanvas = null;
  let attnCtx = null;
  let attnImage = null;
  // Max over heads per column, reused by the fan and the sink/focus
  // markers so the heads x context pass happens once per plate.
  let profile = null;

  let dpr = 1;
  let W = 0;
  let Hpx = 0;
  let compact = false;

  const frames = []; // newest last, pruned to what the camera still needs
  let nextBase = 0; // global slot where the next token's run starts
  let nextStart = 0; // plate index within the model where it picks up
  let camG = 0; // camera position, in plate slots, monotonically increasing
  // What the renderer actually uses: camG floored to a whole plate. The
  // camera advances continuously so the pacing controller keeps working in
  // fractions of a plate, but it is *drawn* only on plate boundaries, so
  // the tower cuts from one plate to the next instead of sliding past.
  // See WHY THE CAMERA CUTS at the top of this file.
  let camV = 0;
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
    // viewport. 1536 cells resolve to 96x16 on desktop and 64x24 here,
    // which comes out very close to square cells at phone widths without
    // making the plate so tall that its attention plate has nowhere to go.
    grid = gridDims(cfg.mlpCells, compact ? 2.6 : 6);
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

  // Sized once at the ceiling rather than per frame, because the frame's
  // width grows with the conversation and reallocating an ImageData every
  // token would churn a megabyte a second. Only the used sub-rect is ever
  // uploaded or blitted.
  function layoutAttn() {
    if (!attnCanvas) {
      attnCanvas = document.createElement("canvas");
      attnCtx = attnCanvas.getContext("2d");
    }
    if (attnCanvas.width !== cfg.maxCols || attnCanvas.height !== cfg.heads) {
      attnCanvas.width = cfg.maxCols;
      attnCanvas.height = cfg.heads;
      attnImage = attnCtx.createImageData(cfg.maxCols, cfg.heads);
    }
    profile = new Float32Array(cfg.maxCols);
  }

  // ------------------------------- data ---------------------------------

  function configure(geometry) {
    cfg = geometry;
    plates = cfg.layers * 2;
    frames.length = 0;
    nextBase = 0;
    nextStart = 0;
    camG = 0;
    camV = 0;
    lastArrival = 0;
    arrivalEma = 380;
    layoutGrid();
    layoutAttn();
  }

  // How many plates this token's run gets: as many as fit at PLATE_MS
  // apiece in the time the last token took, minus the slot the token band
  // costs. Rounded down to a whole number of layers so a run never ends
  // between a layer's attention and its MLP, and never fewer than one
  // layer or more than the whole model.
  function runLength() {
    const budget = Math.round(arrivalEma / PLATE_MS) - GAP;
    const even = budget - (budget % 2);
    return clamp(even, 2, plates);
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
      const base = l * msg.stride;
      let sum = 0;
      for (let c = 0; c < cfg.mlpCells; c++) sum += Math.abs(msg.bytes[base + c] - 128);
      energy[l] = sum / cfg.mlpCells / 127;
    }

    const count = runLength();
    const frame = {
      bytes: msg.bytes,
      // Both per frame, not per model: the attention block is heads x cols
      // and cols is the live context length (js/model/probe.js).
      cols: msg.cols,
      stride: msg.stride,
      pos: msg.pos,
      text: msg.text,
      energy,
      base: nextBase, // first slot of this token's run, in tower coordinates
      start: nextStart, // plate index within the model that run begins at
      count,
    };
    // The next run picks up where this one stopped, wrapping at the last
    // layer, so the descent through the model is continuous across tokens
    // even though each token only covers part of it.
    nextStart = (nextStart + count) % plates;
    nextBase = frame.base + count + GAP;
    if (frames.length === 0) camG = camV = frame.base;
    frames.push(frame);
    prune();
    if (running) ensureRaf();
  }

  function prune() {
    // Keyed off the *rendered* position, which quantization can leave up to
    // two plates behind camG. Pruning on camG would drop a frame that is
    // still on screen.
    const cutoff = camV - DRAW_SPAN - 1;
    while (frames.length > 1 && frames[0].base + frames[0].count + GAP < cutoff) frames.shift();
    while (frames.length > 8) frames.shift();
  }

  // What lives at tower slot `g`: a plate of some token's run, or nothing
  // yet. `last` marks the final plate of a run, which is where the token
  // band gets drawn.
  function plateFor(g) {
    for (let i = frames.length - 1; i >= 0; i--) {
      const f = frames[i];
      if (g < f.base) continue;
      if (g >= f.base + f.count) return null; // the worker has not got here yet
      return {
        frame: f,
        plate: (f.start + (g - f.base)) % plates,
        last: g === f.base + f.count - 1,
      };
    }
    return null;
  }

  // The inclusive layer range a run covers, for captions.
  function runLayers(frame) {
    const first = frame.start >> 1;
    const lastPlate = (frame.start + frame.count - 1) % plates;
    return { first, last: lastPlate >> 1 };
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
    // The floor is the newest run's last plate. Below it is a stack the
    // worker has not produced yet, so descending past it parks the camera
    // over nothing at all. Stopping there means a stall looks like waiting
    // on a layer, which is what is actually happening.
    const newest = frames[frames.length - 1];
    const maxG = newest.base + newest.count - 1;
    const span = newest.count + GAP;
    const behind = (maxG - camG) / span;
    const factor = 1 + Math.max(0, behind - 0.75) * 0.9;
    // span/arrivalEma is PLATE_MS by construction, give or take the
    // rounding in runLength(); expressing it this way keeps the camera
    // locked to the worker rather than to a constant that would drift.
    const speed = (span / Math.max(arrivalEma, 80)) * factor;
    // Parked just short of the newest plate rather than on it. Sitting
    // exactly on the floor puts the bottom third of the screen in the
    // stack the worker has not produced yet, and at 6 tok/s the camera
    // reaches that floor most of the time -- so the steady state would be
    // a screen that is empty below the middle.
    camG = Math.min(camG + speed * dt, Math.max(maxG - 0.85, frames[0].base));
    // The cut. Everything above tracks a continuous position, and this is
    // the one place it becomes discrete.
    //
    // Quantized to a whole *layer*, not a whole plate. Plates alternate
    // ATTN, MLP, ATTN, MLP, so cutting a single plate at a time swaps the
    // type of everything on screen at every cut -- a tall green neuron grid
    // becomes a short amber attention matrix and back, twice a layer. That
    // is not a panel of lights changing, it is two different panels
    // trading places, and it flickers worse than the slide it replaced.
    //
    // Stepping two plates keeps every slot's type fixed: the slot at the
    // centre is always that layer's attention matrix, the one below it is
    // always its MLP, and a cut changes only which layer's numbers are in
    // them. Runs are a whole number of layers and always start on an even
    // plate (see runLength), so this parity holds for every frame.
    camV = Math.floor(camG / 2) * 2;
    prune();
  }

  // ----------------------------- rendering ------------------------------

  // Paints one layer's MLP activation bytes into the offscreen cell
  // canvas, one device pixel per neuron.
  function paintCells(frame, layer) {
    const px = cellImage.data;
    const base = layer * frame.stride;
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

  // Paints one layer's attention matrix -- heads down, context across --
  // and returns how many columns wide the result is.
  //
  // One byte becomes one pixel of the offscreen image, and the offscreen
  // image is never scaled *down* on the way to the screen: the column
  // count is capped at the plate's width in device pixels, so at most one
  // matrix entry lands on one pixel. Above that cap columns are pooled
  // with max rather than dropped, because a dropped column loses a spike
  // outright and the spikes are the picture. In practice the cap does not
  // bite -- 1024 positions against a 1500px plate at dpr 2 is 3000 device
  // pixels of room, and a phone at dpr 2 has 640 for a context that is
  // rarely that long.
  function paintAttn(frame, layer) {
    const rows = cfg.heads;
    const src = frame.cols;
    const out = Math.min(src, Math.max(64, Math.floor(plateWidth() * dpr)));
    const px = attnImage.data;
    const rowStride = attnImage.width * 4;
    const base = layer * frame.stride + cfg.attnBase;

    for (let r = 0; r < rows; r++) {
      const from = base + r * src;
      let q = r * rowStride;
      if (out === src) {
        for (let c = 0; c < out; c++, q += 4) {
          const o = frame.bytes[from + c] * 4;
          px[q] = ATTN_LUT[o];
          px[q + 1] = ATTN_LUT[o + 1];
          px[q + 2] = ATTN_LUT[o + 2];
          px[q + 3] = ATTN_LUT[o + 3];
        }
      } else {
        const per = src / out;
        for (let c = 0; c < out; c++, q += 4) {
          const lo = (c * per) | 0;
          const hi = c === out - 1 ? src : ((c + 1) * per) | 0;
          let v = 0;
          for (let j = lo; j < hi; j++) {
            const b = frame.bytes[from + j];
            if (b > v) v = b;
          }
          const o = v * 4;
          px[q] = ATTN_LUT[o];
          px[q + 1] = ATTN_LUT[o + 1];
          px[q + 2] = ATTN_LUT[o + 2];
          px[q + 3] = ATTN_LUT[o + 3];
        }
      }
    }
    attnCtx.putImageData(attnImage, 0, 0, 0, 0, out, rows);
    return out;
  }

  // Max over heads for each context position, into `profile`. "Is any head
  // looking here", which is what both the fan and the sink/focus markers
  // want; a mean would let one sharply-focused head vanish under eight
  // diffuse ones.
  function buildProfile(frame, layer) {
    const cols = frame.cols;
    const base = layer * frame.stride + cfg.attnBase;
    for (let c = 0; c < cols; c++) profile[c] = 0;
    for (let r = 0; r < cfg.heads; r++) {
      const from = base + r * cols;
      for (let c = 0; c < cols; c++) {
        const v = frame.bytes[from + c];
        if (v > profile[c]) profile[c] = v;
      }
    }
  }

  // The affine map from grid space (columns, rows) to screen space. Both
  // shears are small; together they read as a plate turned a few degrees
  // in two axes, which is enough depth cue without the heavy 2:1 isometric
  // squash that would waste half the screen on empty diagonal.
  function gridMatrix(cx, cy, wPx, hPx, gw, gh) {
    const a = wPx / gw;
    const d = hPx / gh;
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

  function plateCx() {
    return W * (compact ? 0.5 : 0.53);
  }

  function cornersOf(m, gw, gh) {
    const P = (gx, gy) => [m.ox + m.a * gx + m.c * gy, m.oy + m.b * gx + m.d * gy];
    return [P(0, 0), P(gw, 0), P(gw, gh), P(0, gh)];
  }

  function outlinePath(corners) {
    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(corners[i][0], corners[i][1]);
    ctx.closePath();
  }

  // Opaque backing under a plate. Without it whatever is behind shows
  // through every transparent cell of the front face -- which is most of
  // them, since near-zero values are near-transparent by design -- and the
  // plate turns into a grey haze instead of a solid object.
  function fillBacking(corners, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha * 0.93;
    ctx.fillStyle = "#05070a";
    outlinePath(corners);
    ctx.fill();
    ctx.restore();
  }

  // ---- attention plate --------------------------------------------------

  function drawAttnSlab(frame, layer, y, s, alpha) {
    if (!frame.cols) return;
    const drawCols = paintAttn(frame, layer);
    const rows = cfg.heads;
    const wPx = plateWidth() * s;
    // Height from a fixed row height rather than from the matrix aspect:
    // the matrix is heads x context, so its true aspect runs from 20:1 to
    // 100:1 over the course of a conversation and the plate would silently
    // thin to a hairline as the chat got longer. A head is a row you can
    // see, whatever the context length.
    const hPx = rows * (compact ? ATTN_ROW_PX_COMPACT : ATTN_ROW_PX) * s;
    const m = gridMatrix(plateCx(), y, wPx, hPx, drawCols, rows);
    const corners = cornersOf(m, drawCols, rows);
    const focused = s > 0.82;

    fillBacking(corners, alpha);

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.transform(m.a, m.b, m.c, m.d, m.ox, m.oy);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(attnCanvas, 0, 0, drawCols, rows, 0, 0, drawCols, rows);
    ctx.restore();

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = focused ? COLORS.attnFrame : COLORS.attnFrameDim;
    ctx.lineWidth = focused ? 1.1 : 0.8;
    outlinePath(corners);
    ctx.stroke();
    // Head separators, so the rows read as nine distinct heads rather than
    // one texture. Only when they are far enough apart to be lines.
    const rowPx = (corners[3][1] - corners[0][1]) / rows;
    if (focused && Math.abs(rowPx) > 6) {
      ctx.globalAlpha = alpha * 0.28;
      ctx.lineWidth = 0.6;
      ctx.beginPath();
      for (let r = 1; r < rows; r++) {
        ctx.moveTo(m.ox + m.c * r, m.oy + m.d * r);
        ctx.lineTo(m.ox + m.a * drawCols + m.c * r, m.oy + m.b * drawCols + m.d * r);
      }
      ctx.stroke();
    }
    ctx.restore();

    if (!focused) return;
    buildProfile(frame, layer);
    drawAttentionFan(frame, corners, alpha, s);
    drawContextMarkers(frame, m, corners, drawCols, rows, alpha);
    drawPlateLabel(
      corners,
      alpha,
      `L${String(layer).padStart(2, "0")} ATTN`,
      `${cfg.heads}h x ${frame.cols} ctx  ${(cfg.heads * frame.cols).toLocaleString()} weights`,
      COLORS.attnLabel,
      COLORS.attnLabelDim
    );
  }

  // Two verticals through the plate. Position 0 is the attention sink --
  // every head dumps mass there and it means nothing positional, so it is
  // marked and then discounted. The bright one is the strongest position
  // that is *not* the sink, which is the one that says what this layer is
  // actually reading.
  function drawContextMarkers(frame, m, corners, drawCols, rows, alpha) {
    const cols = frame.cols;
    let focusAt = -1;
    let best = 0;
    for (let c = 1; c < cols; c++) {
      if (profile[c] > best) {
        best = profile[c];
        focusAt = c;
      }
    }
    const scale = drawCols / cols;
    const vline = (col, style, width, dash) => {
      const gx = (col + 0.5) * scale;
      ctx.beginPath();
      ctx.setLineDash(dash);
      ctx.moveTo(m.ox + m.a * gx, m.oy + m.b * gx);
      ctx.lineTo(m.ox + m.a * gx + m.c * rows, m.oy + m.b * gx + m.d * rows);
      ctx.strokeStyle = style;
      ctx.lineWidth = width;
      ctx.stroke();
      ctx.setLineDash([]);
      return [m.ox + m.a * gx, m.oy + m.b * gx];
    };

    ctx.save();
    ctx.globalAlpha = alpha * 0.65;
    const sinkTop = vline(0, COLORS.attnLabelDim, 1, [2, 3]);
    let focusTop = null;
    if (focusAt > 0 && best > 24) {
      ctx.globalAlpha = alpha;
      focusTop = vline(focusAt, COLORS.gold, 1.2, []);
    }

    ctx.font = "9px ui-monospace, Menlo, Consolas, monospace";
    ctx.textBaseline = "bottom";
    ctx.textAlign = "left";
    ctx.globalAlpha = alpha * 0.9;
    ctx.fillStyle = COLORS.attnLabelDim;
    ctx.fillText("sink", sinkTop[0] + 3, sinkTop[1] - 3);
    if (focusTop) {
      ctx.globalAlpha = alpha;
      ctx.fillStyle = COLORS.gold;
      ctx.fillText(`focus ${focusAt}`, focusTop[0] + 3, focusTop[1] - 3);
    }
    ctx.restore();
  }

  // Filaments converging from a rail spanning the context onto the plate
  // -- the whole context funnelling into the one token being generated.
  // Sourced from the same per-position profile the markers use, so what
  // the fan shows and what the matrix shows cannot disagree.
  //
  // Batched into alpha tiers so the whole fan is a handful of strokes
  // rather than one per filament.
  function drawAttentionFan(frame, corners, alpha, s) {
    const cols = frame.cols;
    const n = Math.min(cols, FAN_FILAMENTS);
    const per = cols / n;
    const x0 = corners[0][0];
    const x1 = corners[1][0];
    const railY = Math.min(corners[0][1], corners[1][1]) - 42 * s;
    // The focal point rides *on* the plate's top edge rather than at a
    // fixed height above it. The edge is sheared, so a flat offset leaves
    // the fan converging in mid-air next to the plate it is feeding.
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
      for (let b = 0; b < n; b++) {
        const from = (b * per) | 0;
        const to = b === n - 1 ? cols : ((b + 1) * per) | 0;
        let peak = 0;
        for (let j = from; j < to; j++) if (profile[j] > peak) peak = profile[j];
        const v = peak / 255;
        if (v < lo || v >= hi || v < 0.06) continue;
        const rx = x0 + (x1 - x0) * ((b + 0.5) / n);
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
    ctx.strokeStyle = COLORS.attnLabelDim;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, railY);
    ctx.lineTo(x1, railY);
    ctx.stroke();
    ctx.restore();
  }

  // ---- MLP plate --------------------------------------------------------

  function drawLayerSlab(frame, layer, y, s, alpha) {
    const wPx = plateWidth() * s;
    const hPx = (wPx / grid.w) * grid.h;
    const m = gridMatrix(plateCx(), y, wPx, hPx, grid.w, grid.h);
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

    const corners = cornersOf(m, grid.w, grid.h);
    fillBacking(corners, alpha);

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
    outlinePath(corners);
    ctx.stroke();
    ctx.restore();

    if (!focused) return;
    drawHotNeurons(frame, layer, m, alpha, wPx / grid.w);
    drawResidualTrace(frame, layer, corners, alpha, s);
    drawPlateLabel(
      corners,
      alpha,
      `L${String(layer).padStart(2, "0")} MLP`,
      compact
        ? `${cfg.mlpCells}n`
        : `${cfg.mlpCells} neurons  ${grid.w}x${grid.h}  resid ${cfg.residCells}`,
      COLORS.label,
      COLORS.labelDim
    );
  }

  // A bloom sprite on the cells above a fixed fraction of the layer's own
  // peak. Capped, because on a layer where everything fires the cap is the
  // difference between a highlight and a white screen.
  function drawHotNeurons(frame, layer, m, alpha, cellPx) {
    const base = layer * frame.stride;
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

  // The residual stream leaving the layer, as an oscilloscope trace. It is
  // the one vector every layer writes into and reads back out of, so it
  // gets a different visual language from the neuron grid: a continuous
  // cyan signal under the plate rather than a field of discrete cells.
  function drawResidualTrace(frame, layer, corners, alpha, s) {
    const base = layer * frame.stride + cfg.mlpCells;
    const n = cfg.residCells;
    const x0 = corners[3][0];
    const x1 = corners[2][0];
    const y0 = Math.max(corners[2][1], corners[3][1]) + 12 * s;
    const amp = 20 * s;

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

  // ---- chrome -----------------------------------------------------------

  // Above the plate's top-left corner, not below it. Below is where the
  // residual trace and then the token band live, and the vertical budget
  // between two plates is only about 250px at desktop spacing -- a label
  // down there gets written over by one or the other.
  //
  // Queued rather than drawn in place. The token band closing the previous
  // run floats at a position that depends on the camera's sub-slot phase,
  // so it slides across this label about once per plate; drawing every
  // label last, over its own backing, is what keeps it readable through
  // that rather than half the time.
  let pendingLabel = null;

  function drawPlateLabel(corners, alpha, name, detail, nameColor, detailColor) {
    // -20 rather than -7 so the backing box clears the "sink"/"focus"
    // callouts, which sit just above the plate's top edge.
    pendingLabel = { x: corners[0][0], y: corners[0][1] - 20, alpha, name, detail, nameColor, detailColor };
  }

  function flushLabel() {
    if (!pendingLabel) return;
    const { x, y, alpha, name, detail, nameColor, detailColor } = pendingLabel;
    pendingLabel = null;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = "11px ui-monospace, Menlo, Consolas, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    const w = 78 + ctx.measureText(detail).width;
    ctx.fillStyle = "rgba(6, 8, 10, 0.95)";
    ctx.fillRect(x - 4, y - 13, w + 8, 18);
    ctx.fillStyle = nameColor;
    ctx.fillText(name, x, y);
    ctx.fillStyle = detailColor;
    ctx.fillText(detail, x + 78, y);
    ctx.restore();
  }

  // The band between one token's stack and the next, captioned with the
  // token that pass emitted. This is the thing that makes the endless
  // tower legible: without it a fall through 60 plates looks the same as a
  // fall through 120.
  function drawTokenBand(frame, y, alpha) {
    const w = Math.min(compact ? W * 0.9 : 1120, W * 0.9);
    const x = plateCx() - w / 2;
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
    // Named with the layers the run above it covered, so the band says
    // both which token that pass emitted and how deep into the model this
    // stretch of the fall got.
    const { first, last } = runLayers(frame);
    const span = first === last ? `L${first}` : `L${first}-L${last}`;
    const label = raw ? `token  "${raw}"   ${span}` : `token  ...   ${span}`;
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
      // One caret, not one per plate: the camera now frames a whole layer
      // at a time, so both halves of the block are always on screen and
      // "which half is being drawn" has stopped being a question.
      ctx.strokeStyle = here ? COLORS.gold : COLORS.frame;
      ctx.lineWidth = here ? 2.5 : 1.4;
      ctx.beginPath();
      ctx.moveTo(x - (here ? 8 : 3), y);
      ctx.lineTo(x + 5 + e * (compact ? 10 : 22), y);
      ctx.stroke();
    }

    // A bracket over the layers this token's run covers. With only part of
    // the model on screen per token, this is what says which part -- and
    // watching it walk down the rail and wrap is the clearest view of the
    // descent there is.
    const { first, last } = runLayers(frame);
    const yOf = (l) => top + ((bot - top) * l) / (n - 1);
    const drawBracket = (a2, b2) => {
      const y1 = yOf(a2) - 3;
      const y2 = yOf(b2) + 3;
      ctx.beginPath();
      ctx.moveTo(x - 13, y1);
      ctx.lineTo(x - 16, y1);
      ctx.lineTo(x - 16, y2);
      ctx.lineTo(x - 13, y2);
      ctx.stroke();
    };
    ctx.globalAlpha = alpha * 0.8;
    ctx.strokeStyle = COLORS.goldDim;
    ctx.lineWidth = 1;
    // A run that wrapped past the last layer draws as two brackets, which
    // is what actually happened.
    if (last >= first) drawBracket(first, last);
    else {
      drawBracket(first, n - 1);
      drawBracket(0, last);
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
    // Derived from how tall the plates actually are rather than from the
    // viewport alone, so the stack stays a stack: a fixed fraction of the
    // height leaves the short desktop plates crowded and the tall phone
    // plates floating in empty space.
    //
    // The mean of the two heights, not the larger: the plates alternate
    // tall-short-tall-short, so pitching every gap for the tall one puts a
    // screen's worth of empty space either side of every attention plate.
    //
    // The multiplier is set by what hangs off a plate rather than by taste.
    // Between one plate's centre and the next there has to be room for the
    // residual trace under the MLP, the token band, and then the attention
    // plate's label and fan -- about 250px at desktop sizes, and at
    // anything tighter the band ends up written across the plate below it.
    const mlpH = (plateWidth() / grid.w) * grid.h;
    const attnH = cfg.heads * (compact ? ATTN_ROW_PX_COMPACT : ATTN_ROW_PX);
    const spacing = Math.max(((mlpH + attnH) / 2) * 2.05, Hpx * 0.26);
    // The camera sits half a slot below the plate it is "on", so the two
    // plates of the current layer straddle the middle of the screen -- its
    // attention matrix just above, its MLP just below -- instead of one of
    // them being centred and the other pushed to a neighbour's depth. The
    // unit the camera moves in is a layer, so the unit it frames should be
    // a layer too, and both halves want to be readable at once.
    const EYE = 0.5;

    // Mildly perspective rather than linear in z: distance compresses, so
    // the tower recedes into the top and bottom of the frame instead of
    // the third plate simply being off screen.
    const yOf = (z) => cy + spacing * (z / (1 + Math.abs(z) * 0.28));

    // Farthest first, so nearer plates overlay them. Asymmetric by one
    // because the eye is offset by half a slot: the extra plate is on the
    // side the camera is leaning toward.
    const order = [];
    for (let d = -DRAW_SPAN; d <= DRAW_SPAN + 1; d++) order.push(camV + d);
    order.sort((p, q) => Math.abs(q - camV - EYE) - Math.abs(p - camV - EYE));

    let activeFrame = null;
    let activeLayer = 0;

    for (const g of order) {
      if (g < 0) continue;
      // Fixed by construction now -- an integer offset by the constant EYE
      // -- so a plate's transform is bit-identical from one cut to the next
      // *and* the plate that lands in this slot is always the same type.
      // Successive layers put their cells on exactly the same pixels: the
      // plate stops being a thing sliding past and becomes a panel of
      // lights that changes.
      const z = g - camV - EYE;
      const az = Math.abs(z);
      const s = 1 / (1 + az * 0.34);
      const a = clamp(1.15 - az * 0.3, 0, 1) * fade;
      if (a <= 0.012) continue;
      const y = yOf(z);

      const at = plateFor(g);
      if (!at) continue;

      // Two plates per layer, attention first: the order the block runs in.
      const layer = at.plate >> 1;
      const isAttn = (at.plate & 1) === 0;
      if (isAttn) drawAttnSlab(at.frame, layer, y, s, a);
      else drawLayerSlab(at.frame, layer, y, s, a);
      // The band closing this run, in the space before the next run's
      // first plate. Nudged past the midpoint so it reads as belonging to
      // the run above it rather than floating between two.
      if (at.last) drawTokenBand(at.frame, y + (yOf(z + 1) - y) * 0.55, a);
      // Both centre plates sit at az === EYE and belong to the same layer,
      // so either one names it.
      if (az <= EYE) {
        activeFrame = at.frame;
        activeLayer = layer;
      }
    }

    flushLabel();
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
