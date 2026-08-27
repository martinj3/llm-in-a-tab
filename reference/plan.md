# llm-in-a-tab — Implementation Plan

A from-scratch transformer inference engine in plain JavaScript, running entirely
client-side in a browser tab. Hosted as a static site on GitHub Pages. No server,
no backend, no inference framework.

This is a **learning project**. When a choice is between "faster" and "clearer",
pick clearer. When it is between "clever" and "debuggable", pick debuggable.
Several decisions below deliberately reject the more efficient option because the
efficient option obscures the thing we are trying to learn.

---

## 0. Constraints

These are hard requirements, not preferences.

- **Pure JavaScript for inference.** No ML libraries, no ONNX, no WASM modules, no
  WebGL/WebGPU. Typed arrays and hand-written loops.
- **No external libraries** unless one meaningfully beats what the platform gives
  us for free. See §1.6 — as of now, nothing qualifies.
- **Static hosting only.** GitHub Pages. No server-side anything, no custom HTTP
  headers, no build step required to run.
- **Must not be broken on mobile** for the 135M model at 1k context. The 360M
  model is desktop-first and may be gated behind a warning.
- **Both models supported from day one.** Everything is driven by `config.json`.
  Nothing about a specific model is hardcoded.

---

## 1. Decisions already made, and why

### 1.1 Model: SmolLM2-Instruct, 135M and 360M

`HuggingFaceTB/SmolLM2-135M-Instruct` and `HuggingFaceTB/SmolLM2-360M-Instruct`.

Both are `LlamaForCausalLM` — RMSNorm, SwiGLU, RoPE, GQA — which is the canonical
modern decoder architecture. What you learn building this transfers directly.
They are small enough to run in a tab, public (no auth), and served with
permissive CORS.

**135M is the default and the development target.** It is ~2.6x cheaper per token
and iterates far faster while debugging. 360M is the "make it nicer" upgrade.

Note: it must be the **-Instruct** variant. The base models have no chat template
and no notion of conversational turns; they will just continue your text.

### 1.2 Weights: safetensors (bf16), NOT GGUF

GGUF Q8_0 is tempting — half the download (~386MB vs ~723MB for 360M) and it
carries a genuinely good block-wise quantization scheme already computed.

**We are not using it, for one decisive reason.** llama.cpp's GGUF conversion
permutes the Q and K projection weights. From `conversion/llama.py` in the
llama.cpp repo, `LlamaModel` sets `undo_permute = True` and applies:

```python
weights.reshape(n_head, 2, head_dim // 2, ...).swapaxes(1, 2).reshape(...)
```

to every `q_proj.weight` and `k_proj.weight`. This rewrites them into llama.cpp's
*interleaved* RoPE convention (rotating pairs `2i, 2i+1`) instead of HuggingFace's
*split-half* convention (pairs `i, i + head_dim/2`).

SmolLM2's `config.json` states `"rope_interleaved": false` — it is a split-half
model. If we loaded GGUF weights and wrote RoPE the way every reference
implementation and tutorial writes it, the model would produce **fluent-looking
garbage**: no crash, no error, plausible tokens, silently wrong. That is the worst
possible failure mode for a from-scratch project, because it looks like your
attention math is broken when it isn't.

safetensors also wins on parser simplicity:

| | safetensors bf16 | GGUF Q8_0 |
|---|---|---|
| Header | u64 LE length, then JSON, then raw bytes (~20 lines) | magic + 12 typed KV value types + arrays + alignment (~200 lines) |
| Weight decode | bf16 -> f32 is `u16 << 16` reinterpreted. One line. | Q8_0 blocks + fp16 scale decode |
| Tensor names | match `config.json` and every tutorial | `blk.0.attn_q.weight`, needs a mapping table |
| RoPE convention | matches the reference | **permuted** |

**bf16 is the easy case.** It has fp32's exponent layout, so widening is a left
shift and nothing else — no exponent rebiasing, no subnormals, no infinity
special-casing. If these models had shipped fp16 this would be more work.

### 1.3 In-memory format: int8 weights with PER-ROW scales

Weights are quantized during load and kept as `Int8Array` plus a `Float32Array` of
scales, one scale per **output row** of each matrix.

For each weight matrix stored `[out_features, in_features]` row-major:

```
scale[r] = max(|W[r, :]|) / 127
q[r, k]  = round(W[r, k] / scale[r])
```

In the matmul, accumulate `int8 * f32` (JS widens to double automatically — no
overflow concerns, unlike C) and multiply the finished dot product by `scale[r]`.

**Per-row, not per-tensor.** The danger in 8-bit quantization is not the 8 bits, it
is a shared scale: one outlier weight sets the scale for millions of others and
crushes them into a handful of levels. Per-row scaling costs one extra array read
hoisted out of the inner loop and recovers most of the gap to fp16. llama.cpp's
`Q8_0` uses one scale per 32 weights for exactly this reason — a per-tensor scheme
is *not* what "int8 is industry standard" refers to.

**Why not int16 or f32?** Measured (see §3), int8 is only ~13% faster than f32 in
a scalar JS loop — we are bound by loop overhead and bounds-checked typed-array
loads, not memory bandwidth. So int8's real payoff is **footprint**, which is the
binding constraint on mobile. int16 would cost 2x the memory for precision we
cannot perceive once per-row scaling is in place.

**Do not quantize:**
- **RMSNorm weights** — keep f32. ~60KB total, multiplicative gains applied to
  every activation, and quantizing them can visibly break output.
- **KV cache and activations** — f32. See §1.4.

**Do quantize the embedding table.** It is ~47M params (13-35% of the model) and,
because `tie_word_embeddings: true`, it is *also* the `lm_head` — the single
largest matmul per token. Per-row here means one scale per vocabulary entry, which
is a natural grouping.

### 1.4 KV cache

- **Store K *after* applying RoPE.** Rotate K by its absolute position at write
  time and store the rotated vector. Q is rotated fresh each step. Storing
  un-rotated K means re-rotating the entire cache every token — O(context) wasted
  work per step, for nothing. Both versions "work" on token 1, which is why this
  is easy to get wrong.
- **Store `num_key_value_heads`, not `num_attention_heads`.** 3 for the 135M, 5
  for the 360M. The GQA expansion is index arithmetic (`kvh = (h / ratio) | 0`),
  never a materialized tensor. Getting this wrong silently costs 3x the memory.
- **Layout: position-major**, `[pos][kv_head][head_dim]`. Measured, head-major is
  only ~2.5% faster on reads (218ms vs 224ms at 1024 ctx on the 360M) — within
  noise, because the hardware prefetcher handles the stride fine when each run is
  256 contiguous bytes. Position-major makes appending a token one contiguous
  write and the indexing simpler. Take the simpler one.
- **Preallocate at max context, per layer.** `Float32Array(maxCtx * kv_heads *
  head_dim)` for K and the same for V, allocated once at model load. The cache is
  a fixed buffer plus a `seqLen` fill counter — never a growing array, never an
  allocation inside the generation loop.
- **Precision: f32 for v1.** `Float16Array` (available in current Chrome) would
  halve it and is likely close to perf-neutral since attention is access-bound.
  That is the upgrade that makes 4k/8k contexts viable. Not now.

KV cache memory is exactly linear in context:

| context | 135M | 360M |
|---|---|---|
| 1k | 47 MB | 84 MB |
| 2k | 94 MB | 168 MB |
| 4k | 189 MB | 336 MB |
| 8k | 378 MB | 672 MB |

At 8k the 360M's cache alone exceeds its weights. This table is the feasibility
answer for the future context selector.

### 1.5 Threading and memory discipline

- **All model work runs in a Web Worker.** A scalar JS forward pass pins a core
  for hundreds of milliseconds per token; on the main thread that is an
  unresponsive page. The worker owns the weights, the KV cache, and the
  generation loop. The main thread only does UI and posts messages.
- **One worker, not several.** GitHub Pages cannot set the COOP/COEP headers that
  cross-origin isolation requires, so **`SharedArrayBuffer` is unavailable**.
  Multiple workers could not share the weight buffer — each would need its own
  copy at 135–360MB. Single worker is the design, not a limitation to work around.
- **Preallocate every buffer once; never allocate inside a loop.** All scratch
  space (activations, scores, logits, per-layer temporaries) is allocated at model
  load and mutated in place. Functions take an output buffer parameter rather than
  returning a new array. Allocation inside the token loop means the GC runs during
  generation, which shows up as visible stutter.

### 1.6 External libraries: none, and here is the evaluation

- **Matrix math** — nothing to add. The whole point is writing the loops.
  Measured evidence that hand-tuning is counterproductive: a 4-way unrolled inner
  loop was *slower* than the naive one (0.36 vs 0.43 GMAC/s). V8's optimizer does
  not want help.
- **Downloading** — `fetch` with `Range` headers is exactly what we need. No
  library does this better.
- **Caching** — raw IndexedDB is a slightly clunky API, and a tiny wrapper
  (e.g. `idb-keyval`, ~600 bytes) would be the *only* dependency with a defensible
  case. Still: we store a handful of large binary blobs under fixed keys. That is
  ~40 lines of vanilla IndexedDB, written once, never touched again. Skip it.

**Verdict: zero dependencies.** Revisit only if something concrete justifies it.

---

## 2. Confirmed model facts

Verified against the actual files in `reference/`. Do not re-derive these; do not
trust any value remembered from elsewhere. **But still load them from
`config.json` at runtime — nothing here gets hardcoded.**

| | 135M | 360M |
|---|---|---|
| `hidden_size` | 576 | 960 |
| `num_hidden_layers` | 30 | 32 |
| `num_attention_heads` | 9 | 15 |
| `num_key_value_heads` | 3 | 5 |
| head_dim (derived) | 64 | 64 |
| GQA ratio | 3 | 3 |
| `intermediate_size` | 1536 | 2560 |
| `vocab_size` | 49152 | 49152 |
| `rope_theta` | **100000** | **100000** |
| `max_position_embeddings` | 8192 | 8192 |
| `rms_norm_eps` | 1e-05 | 1e-05 |
| `tie_word_embeddings` | true | true |
| `hidden_act` | silu | silu |
| `attention_bias` / `mlp_bias` | false | false |
| `rope_scaling` | null | null |
| `rope_interleaved` | false | false |
| `torch_dtype` | bfloat16 | bfloat16 |

`rope_theta` is **100000**, not the Llama default of 10000. Assuming the default
produces fluent garbage with no error.

**The tokenizers are byte-identical between the two models** — verified by diffing
`tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`, and
`special_tokens_map.json`. One implementation, one vendored copy, one cache entry.

Special tokens: `0 = <|endoftext|>` (registered as **unk**, not EOS),
`1 = <|im_start|>`, `2 = <|im_end|>`. `eos_token_id: 2`.

Chat template (ChatML), verbatim from `tokenizer_config.json`:

```
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
{content}<|im_end|>
<|im_start|>assistant
```

The default system message is injected **only** when the first message is not
already a system message — supplying your own replaces it, it does not append.
The trailing `<|im_start|>assistant\n` is the "generation prompt"; sampling starts
after it.

**System prompt guidance:** shorter is better, and near-default is best. A 360M
model has no capacity to track many instructions; a long system prompt makes it
more confused, not more helpful. Keep custom prompts under ~30 tokens. The
formatting wrapper costs only ~4-5 tokens per message because `<|im_start|>` and
`<|im_end|>` are single tokens — and with a persisted KV cache the system prompt
is prefilled once and never repeated, so optimizing it for speed is pointless.
Optimize it for staying on the training distribution.

---

## 3. Measured performance baseline

Benchmarked in Node 22 (same V8 as Chrome), single-threaded, on a 2.8GHz Xeon,
with correctly-sized matrices so cache behavior is realistic. Scalar throughput
came out at **~0.6 GMAC/s**.

| | linear layers | attention @1024 ctx | total | tok/s |
|---|---|---|---|---|
| 135M | 227 ms | 141 ms | **368 ms** | **2.7** |
| 360M | 599 ms | 221 ms | **820 ms** | **1.2** |

Those are at *full* 1024 context. Early in a conversation (~300 tokens) it is
closer to 4.4 and 1.7 tok/s.

**Flagship phone estimate: roughly 0.6–1.0x of that** — comparable single-core
peak, but this workload pins a core for minutes and phones throttle hard. So
~2–4 tok/s for the 135M, ~0.8–1.7 tok/s for the 360M. Treat as ±2x.

**Attention is more expensive than its MAC count suggests** — 38% of a 135M token
and 27% of a 360M token at 1024 ctx, running at under half the matvec rate because
of strided KV access and two softmax passes. It scales linearly with context while
the linear layers stay flat. Rough projection for the 135M: ~2.7 tok/s at 1k,
~1.9 at 2k, ~1.2 at 4k, ~0.7 at 8k.

**Prefill batching is worth ~1.9x and saturates at B=4:**

| kernel | ms/token |
|---|---|
| sequential matvec | 5.55 |
| batched B=4, local accumulators | 2.93 (**1.90x**) |
| batched B=8, local accumulators | 2.88 (1.93x) |
| batched B=8, token loop innermost | 12.02 (**0.46x** — worse than not batching) |

This is the one place batching beats the loop-overhead ceiling, because each
weight byte is reused across tokens. **It only works with explicit local
accumulators** (`a0..a3`); writing the token loop innermost is 2x slower than not
batching at all. See §6.

---

## 4. Module layout

```
index.html
css/style.css
js/
  main.js            UI, button wiring, worker messaging, streaming output
  worker.js          owns model + KV cache + generation loop
  model/
    config.js        fetch + validate config.json, assert invariants
    safetensors.js   header parse, ranged streaming fetch, bf16 decode, quantize
    store.js         IndexedDB persistence for quantized weights + tokenizer
    tokenizer.js     byte-level BPE from vocab.json + merges.txt
    template.js      ChatML formatting, stop-token logic
    kvcache.js       allocation, append, indexing
    ops.js           matvec, matmatB4, rmsnorm, rope, softmax, silu
    transformer.js   forward pass (prefill + decode paths)
    sample.js        greedy / temperature / top-p
tools/
  fetch-reference.ps1
reference/
  plan.md, 135M/, 360M/
```

**Design constraint:** keep per-operation boundaries intact. `transformer.js`
should call named ops in sequence rather than fusing the whole forward pass into
one closure. Later work will want to observe intermediate activations, and that
must not require rewriting the engine.

---

## 5. Build order

Each phase has an exit criterion. **Do not start the next phase until the current
one demonstrably passes.** The failure mode this ordering exists to prevent is
having three unverified components at once and no way to tell which is broken.

### Phase 0 — Static shell
`index.html` with basic CSS, a status area, and two buttons: **Download 135M** and
**Download 360M**. No model code. Confirm GitHub Pages serves it.

*Exit: the page loads at the Pages URL and the buttons log the model they select.*

### Phase 1 — Download and cache, no parsing
Fetch `config.json` for the selected model. Fetch `model.safetensors` with a
progress bar driven by `ReadableStream`. Store the raw bytes in IndexedDB. On
reload, detect the cached copy and skip the network.

Call `navigator.storage.estimate()` before starting and `persist()` after, so a
quota problem surfaces up front instead of at 90%.

*Exit: download shows progress, survives a page reload without re-downloading,
and byte length matches `content-length`.*

### Phase 2 — safetensors parsing and quantization
Parse the header: first 8 bytes are a little-endian u64 JSON length, then that
many bytes of JSON giving every tensor's dtype, shape, and byte range, then the
raw data.

Then replace Phase 1's whole-file download with **per-tensor ranged fetches**:
read the first ~100KB, parse the header, then issue `Range` requests per tensor,
convert and quantize each as it arrives, and discard the raw bytes. Peak memory
becomes `final size + one tensor` instead of `final size + whole file`.

**This is required for mobile, not an optimization.** Buffering 723MB and then
converting peaks around 1.1GB and will get the tab killed on iOS.

Store the *quantized* arrays in IndexedDB, not the raw download — repeat visits
then skip both the network and the conversion pass, and store 135MB instead of
723MB.

Support a `dtype` flag: `'f32'` (dequantize only, for debugging) or `'i8'`
(quantize). Same code path, one branch.

*Exit: every tensor named in `config.json`'s architecture is present with the
expected shape; a weight histogram looks like a bell curve centered near zero;
peak memory during load stays near the final size.*

### Phase 3 — Tokenizer
Byte-level BPE from `vocab.json` + `merges.txt` (see §6 for why not
`tokenizer.json`). Needs: the GPT-2 `bytes_to_unicode` table, the digit-splitting
pretokenizer, the ByteLevel regex, literal special-token matching, and the merge
loop.

*Exit: encoding a set of test strings reproduces reference token IDs exactly,
including strings with numbers, emoji, leading/trailing spaces, and literal
`<|im_start|>` text. Round-trip decode is lossless.*

### Phase 4 — Chat template
Build the ChatML string from a message array. Hardcode the structure — **do not
write a Jinja interpreter.**

*Exit: output matches `apply_chat_template` output character for character.*

### Phase 5 — Forward pass, f32, single token
`ops.js` and `transformer.js`. Embedding lookup, then per layer: RMSNorm ->
Q/K/V projections -> RoPE -> attention -> output projection -> residual ->
RMSNorm -> SwiGLU MLP -> residual. Then final norm and the `lm_head` projection
(which is the tied embedding matrix).

Run in **f32** first. Validate against golden logits.

*Exit: top-10 logits for a fixed prompt match the reference to within f32 noise.
This is the single most important gate in the project.*

### Phase 6 — int8 and the KV cache
Switch the loader to `'i8'` and re-check: numeric tolerance loosens, but the top-k
token ranking should stay stable. Then add the KV cache and the decode loop, and
generate a multi-token continuation greedily.

*Exit: greedy generation from a fixed prompt is deterministic and reproducible,
and matches a reference greedy generation for at least the first several tokens.*

### Phase 7 — Prefill batching
Add the B=4 batched kernel for prompt processing. Guard it with a test asserting
batched and sequential prefill produce identical KV cache contents.

*Exit: measurable ~1.9x speedup on prompt processing, with identical output.*

### Phase 8 — Chat UI
Multi-turn conversation with a **persisted KV cache** — turn 2 prefills only the
new user message, never the whole conversation. Token-by-token streaming to the
DOM. Stop on `<|im_end|>`. Sampling controls (temperature, top-p), defaulting to
greedy.

*Exit: a coherent multi-turn conversation, with turn 3 no slower to start than
turn 2.*

### Phase 9 — Mobile hardening
Test the 135M at 1k context on a real phone. Verify load peak memory, IndexedDB
quota behavior, and that the UI stays responsive. Gate the 360M behind a size
warning.

*Exit: a full conversation completes on a phone without the tab being killed.*

---

## 6. Gotchas

Traps that produce **silently wrong output rather than errors**. Every one of
these has been verified against the files in `reference/`.

### Model / math

1. **`rope_theta` is 100000, not 10000.** Both models. Read it from config.
2. **RoPE is split-half, not interleaved.** `rope_interleaved: false`. Rotate
   dimension pairs `(i, i + head_dim/2)`, not `(2i, 2i+1)`. This is also why we
   rejected GGUF (§1.2).
3. **There is no `lm_head.weight` tensor.** `tie_word_embeddings: true` — the
   output projection *is* `model.embed_tokens.weight`. A loader looking for
   `lm_head.weight` will fail to find it.
4. **safetensors stores Linear weights as `[out_features, in_features]`
   row-major.** The dot product for output `j` is `sum_k x[k] * W[j*K + k]` —
   contiguous. Writing `W[k*N + j]` is both wrong for this layout and strides
   catastrophically through memory.
5. **RMSNorm, not LayerNorm.** No mean subtraction, no bias. `eps = 1e-5`, applied
   inside the sqrt.
6. **GQA: 3 query heads share each KV head.** `kvh = (h / 3) | 0`. Never
   materialize the expanded K/V.

### Tokenizer

7. **Digits are split individually.** The pretokenizer is a `Sequence`:
   `Digits(individual_digits: true)` then `ByteLevel`. Verified: there are **zero**
   multi-digit tokens in the vocab, and `0`–`9` are ids 32–41. A textbook GPT-2
   BPE will merge digit pairs and produce wrong ids. Because it only affects
   numbers, this ships easily and is not noticed until someone types a date.
8. **Use `vocab.json` + `merges.txt`, not `tokenizer.json`.** Verified identical
   content: `vocab.json` byte-matches `tokenizer.json`'s `model.vocab`, and
   `merges.txt` carries the same 48,900 merges. But 1.24MB vs 2.1MB, and
   line-oriented text maps directly to a merge-priority table. `merges.txt` has a
   `#version: 0.2` header line to skip.
9. **Special tokens are matched literally, before pretokenization.** Split the
   input on `<|im_start|>` etc. first, then run BPE only on the text between them.
   They are already present in `vocab.json` (ids 0–16), so no separate merge step.
10. **`add_prefix_space: false`.** Do not prepend a space. (Note the *decoder*
    declares `add_prefix_space: true` — check the round-trip against golden
    vectors rather than assuming symmetry.)
11. **No normalizer.** `"normalizer": null` — no Unicode normalization step.
12. **No UNK is reachable.** `byte_fallback: false`, `unk_token: null`, and
    byte-level coverage means every input maps to tokens. If id 0 appears in
    encoder output, something upstream is wrong.

### Chat format

13. **Do not prepend BOS.** `tokenizer_config.json` declares
    `bos_token = '<|im_start|>'` and config has `bos_token_id: 1`, which strongly
    tempts you. But `"post_processor": null` and the template never emits a BOS —
    the leading `<|im_start|>` belongs to the template's first message.
14. **Stop on `<|im_end|>` (id 2).** Without it the model generates
    `<|im_start|>user` and hallucinates the user's next turn, cheerfully and
    indefinitely. Also stop on id 0 defensively, and cap max new tokens.
15. **Whitespace is exact.** The `\n` after the role name and after `<|im_end|>`
    are part of the format. A missing newline is a silent quality regression.
16. **Do not invent prefixes** like `"User says:"`. These models saw exactly one
    format during instruction tuning and cannot generalize across variations the
    way a large model can. Off-format input collapses them to base-model behavior:
    rambling continuation instead of an assistant reply.

### Performance / implementation

17. **Batched prefill needs local accumulators.** `a0..a3` as locals, weight load
    in the outer position. Putting the token loop innermost is **2x slower than
    not batching** (measured: 12.02 vs 5.55 ms/token). B=4 captures the full
    benefit; B=8 adds nothing.
18. **Do not manually unroll the matvec inner loop.** Measured *slower* than the
    naive version (0.36 vs 0.43 GMAC/s). V8 handles it better alone.
19. **Never allocate inside the token loop.** All buffers preallocated at load,
    functions mutate an output parameter.
20. **Position bookkeeping must be exact.** `seqLen` counts every token fed
    through the model, chat template tokens included. Drift by one and RoPE
    desynchronizes into fluent garbage with no error. Make appending to the KV
    cache and advancing the position the same function so they cannot diverge.
21. **Generated tokens go into the KV cache too**, at `seqLen++`, exactly like
    prompt tokens.
22. **Persist the KV cache across turns.** Rebuilding it per request means every
    turn costs a full re-prefill, and the app is unusable by turn three.
23. **No `SharedArrayBuffer` on GitHub Pages.** COOP/COEP headers cannot be set on
    Pages, so cross-origin isolation is unavailable. Do not design for multiple
    workers sharing weight memory.

### Hosting

24. **Never commit weight files.** `.gitignore` blocks `*.safetensors`, `*.gguf`,
    `*.bin`. GitHub rejects any file over 100MB, and Git LFS does not work with
    Pages (it serves the pointer file, not the object).
25. **Hugging Face CORS works fine.** `huggingface.co/{repo}/resolve/main/{file}`
    sends `Access-Control-Allow-Origin: *` and supports `Range`. Do **not** route
    downloads through a third-party mirror; there is no CORS problem to solve.

---

## 7. Context limit policy

For v1: **hard stop at the configured limit**, with a clear message, and reserve
headroom — reject a prompt that leaves no room to generate a reply.

The sliding-window version is a later upgrade with a non-obvious catch worth
knowing now: because RoPE encodes absolute position, you can drop the oldest
entries and keep the rest valid (it becomes windowed attention), **but the first
~4 tokens must be retained or output quality collapses** — the "attention sink"
effect. A naive drop-the-oldest ring buffer appears to work and then degrades.
Don't build it yet, but don't build v1 assuming position 0 is always evictable.

---

## 8. Non-goals for v1

- WebGPU / WebGL / WASM acceleration
- Multi-threaded inference
- Context sizes above 1024 (the selector for 1k/2k/4k/8k comes later)
- Pre-converted weights hosted on a Release or our own HF repo (would cut the
  download roughly in half and remove the browser-side conversion path entirely —
  a good later optimization, not a v1 concern)
- Models other than SmolLM2 135M / 360M

## 9. Testing approach

**Golden vectors are the highest-value thing in this project.** Generate reference
outputs once with Python + `transformers` and commit them to `reference/golden/`:

- the templated prompt string for a fixed message list
- its exact token IDs
- the top-10 logit indices and values for the final position

Then Phase 3 matches token IDs (tokenizer correct), and Phase 5 matches top-10
logits (everything else correct). When they diverge you immediately know which
half is broken. Debugging a from-scratch transformer without this is guesswork —
it is the difference between a weekend and a month.
