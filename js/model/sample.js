// Token selection. Greedy (argmax) is the default and the only path used
// by the golden-vector tests, which need deterministic, reproducible
// output; temperature and top-p are the Phase 8 chat UI's sampling
// controls (plan.md Phase 8).

export function argmax(logits) {
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > bestVal) {
      bestVal = logits[i];
      best = i;
    }
  }
  return best;
}

// The k highest-probability tokens, descending, as [{ id, prob }].
//
// Display only -- nothing downstream of generation reads this, so it is
// safe for it to disagree with sampleToken's own draw. It does not sort
// the vocabulary: a k-element insertion scan is a few hundred thousand
// comparisons against the full sort's ~49k*log(49k), which matters here
// in a way it does not in sampleToken because this runs on every token
// including the greedy path, where sampleToken is just an argmax.
//
// temperature <= 0 (greedy) reports the model's own distribution at T=1
// rather than the degenerate one-hot the sampler is actually using --
// which is the interesting thing to look at, and the reason the chosen
// token is marked separately instead of inferred from being first.
export function topCandidates(logits, k = 15, temperature = 0) {
  const n = logits.length;
  const t = temperature > 0 ? temperature : 1;

  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < n; i++) sum += Math.exp((logits[i] - max) / t);

  // Insertion into a k-slot array kept in descending logit order. The
  // first test rejects the overwhelming majority of the vocabulary in one
  // comparison once the array is full.
  const top = [];
  for (let i = 0; i < n; i++) {
    const v = logits[i];
    if (top.length === k && v <= top[k - 1].v) continue;
    let at = top.length;
    while (at > 0 && top[at - 1].v < v) at--;
    top.splice(at, 0, { id: i, v });
    if (top.length > k) top.pop();
  }

  return top.map(({ id, v }) => ({ id, prob: Math.exp((v - max) / t) / sum }));
}

// Nucleus sampling: softmax(logits / temperature), keep the smallest set
// of tokens whose probabilities sum to >= topP, renormalize, draw one.
//
// temperature <= 0 means greedy, and is the default -- not a special case
// bolted on, but the honest limit of this function as T -> 0.
//
// This sorts all 49152 vocabulary entries every token, which is
// deliberate. A partial selection of the top few hundred would give the
// same answer for any sane topP and be much faster in isolation, but a
// token here already costs 300-800ms in the forward pass; a ~10ms sort is
// a few percent, and "softmax, sort, accumulate, draw" is the version you
// can read and confirm is right (plan.md's clearer-over-faster rule).
export function sampleToken(logits, { temperature = 0, topP = 1, random = Math.random } = {}) {
  if (!(temperature > 0)) return argmax(logits);

  const n = logits.length;
  const probs = new Float64Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp((logits[i] - max) / temperature);
    probs[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) probs[i] /= sum;

  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  // Int32Array.prototype.sort is numeric by default, so the comparator is
  // only needed to sort by probability rather than by index.
  const sorted = Array.prototype.sort.call(order, (a, b) => probs[b] - probs[a]);

  // Cut at the first token that carries the cumulative mass past topP --
  // inclusive, so the nucleus is never empty even if one token already
  // exceeds topP on its own.
  let cutoff = n;
  let cumulative = 0;
  for (let i = 0; i < n; i++) {
    cumulative += probs[sorted[i]];
    if (cumulative >= topP) {
      cutoff = i + 1;
      break;
    }
  }

  const target = random() * cumulative;
  let running = 0;
  for (let i = 0; i < cutoff; i++) {
    running += probs[sorted[i]];
    if (running >= target) return sorted[i];
  }
  return sorted[cutoff - 1]; // float rounding fell off the end
}
