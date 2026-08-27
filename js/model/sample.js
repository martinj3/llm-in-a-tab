// Greedy sampling only, for now -- Phase 6's exit criterion is deterministic,
// reproducible generation matching a reference greedy run. Temperature/top-p
// are Phase 8 scope (plan.md's chat UI "sampling controls"), not needed yet.
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
