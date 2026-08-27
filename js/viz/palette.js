// Colour ramps for the activation grids, baked into a 256-entry RGBA
// lookup table.
//
// Activations arrive as bytes with 128 meaning zero (js/model/probe.js),
// so one array index turns a value into a colour. Up to ~10k cells get
// coloured per frame at 60fps once the extruded side face is counted, and
// that is not a place for a branchy colour function.
//
// Positive goes green -> white-hot and negative goes amber, which keeps
// the sign of every activation readable inside the terminal palette. The
// alpha ramp matters as much as the hue: near-zero cells stay almost
// transparent, so a layer reads as a sparse constellation of firing
// neurons over the background rather than a solid rectangle of colour.

// Piecewise-linear ramp through [stop, r, g, b] control points.
function ramp(stops, t) {
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0] || i === stops.length - 1) {
      const [p0, r0, g0, b0] = stops[i - 1];
      const [p1, r1, g1, b1] = stops[i];
      const f = p1 === p0 ? 0 : (t - p0) / (p1 - p0);
      const k = f < 0 ? 0 : f > 1 ? 1 : f;
      return [r0 + (r1 - r0) * k, g0 + (g1 - g0) * k, b0 + (b1 - b0) * k];
    }
  }
  return [0, 0, 0];
}

const POSITIVE = [
  [0.0, 10, 30, 18],
  [0.25, 20, 102, 58],
  [0.55, 63, 216, 115],
  [0.8, 157, 255, 171],
  [1.0, 242, 255, 240],
];

const NEGATIVE = [
  [0.0, 21, 14, 5],
  [0.3, 107, 66, 7],
  [0.65, 217, 138, 30],
  [1.0, 255, 208, 137],
];

// 4 bytes per entry, RGBA. Written a byte at a time rather than through a
// Uint32Array view of the ImageData: the packed-word trick is faster but
// assumes little-endian byte order, and at these cell counts the honest
// version is already far inside budget.
export const NEURON_LUT = new Uint8Array(256 * 4);

for (let v = 0; v < 256; v++) {
  const signed = (v - 128) / 127;
  const mag = Math.min(1, Math.abs(signed));
  // Gamma below 1 lifts the mid-range: activation distributions are
  // heavy-tailed, so a linear ramp leaves all but a handful of cells
  // sitting in the bottom tenth of the scale and the grid looks dead.
  const t = Math.pow(mag, 0.45);
  const [r, g, b] = ramp(signed >= 0 ? POSITIVE : NEGATIVE, t);
  // Alpha, unlike hue, is ramped steeply and off a floor near zero. The
  // measured distribution of real SwiGLU activations (tests/dump-frames.mjs)
  // is severe: 75% of neurons sit below 1/16 of their layer's peak and 90%
  // below 1/8, with under 1% above half. A gentle alpha ramp therefore
  // paints three quarters of every layer at a clearly visible level and the
  // grid reads as noise. Held down here, the bulk becomes faint texture and
  // the few percent that actually fired are what you see.
  //
  // Negatives are held back further again so the green positives carry the
  // composition and the amber reads as accent rather than a second,
  // competing field of colour.
  // The negative ramp is deliberately much steeper than the positive one,
  // not merely dimmer. 42% of neurons are negative, so at a matched ramp
  // the weak half of both signs paints the whole layer in overlapping dark
  // green and dark amber, which averages to olive mud. Steep here means
  // amber appears only where a neuron is *strongly* negative, and the field
  // stays green with amber as punctuation.
  const a = signed >= 0 ? 20 + 235 * Math.pow(mag, 0.55) : 8 + 178 * Math.pow(mag, 1.25);
  const o = v * 4;
  NEURON_LUT[o] = r | 0;
  NEURON_LUT[o + 1] = g | 0;
  NEURON_LUT[o + 2] = b | 0;
  NEURON_LUT[o + 3] = a | 0;
}

export const COLORS = {
  frame: "rgba(75, 240, 122, 0.5)",
  frameDim: "rgba(75, 240, 122, 0.16)",
  trace: "rgba(96, 210, 255, 0.85)",
  traceFill: "rgba(96, 210, 255, 0.09)",
  fan: "215, 255, 222",
  gold: "rgba(255, 200, 92, 0.92)",
  goldDim: "rgba(255, 200, 92, 0.3)",
  label: "rgba(75, 240, 122, 0.52)",
  labelDim: "rgba(75, 240, 122, 0.24)",
};
