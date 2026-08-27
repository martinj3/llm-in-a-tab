// Fetch and sanity-check config.json. Every architecture fact used by the
// forward pass is read from this file at runtime -- nothing is hardcoded.
// See plan.md section 2 for the confirmed values these assertions guard.
import { resolveUrl } from "../models.js";
import { fetchJson } from "./download.js";

const REQUIRED_FIELDS = [
  "hidden_size",
  "num_hidden_layers",
  "num_attention_heads",
  "num_key_value_heads",
  "intermediate_size",
  "vocab_size",
  "rope_theta",
  "max_position_embeddings",
  "rms_norm_eps",
  "tie_word_embeddings",
];

export async function fetchModelConfig(modelId) {
  const url = resolveUrl(modelId, "config.json");
  const config = await fetchJson(url);
  validateConfig(config);
  return config;
}

// Assert the invariants the whole plan is built on. If HuggingFace ever
// changes these, we want a loud failure here, not fluent garbage three
// layers deep (see plan.md's "silently wrong output" gotchas).
export function validateConfig(config) {
  for (const field of REQUIRED_FIELDS) {
    if (config[field] === undefined) {
      throw new Error(`config.json missing required field: ${field}`);
    }
  }
  if (config.model_type !== "llama") {
    throw new Error(`Expected model_type "llama", got "${config.model_type}"`);
  }
  if (config.tie_word_embeddings !== true) {
    throw new Error("Expected tie_word_embeddings: true (no separate lm_head)");
  }
  if (config.rope_interleaved) {
    throw new Error(
      "Expected split-half RoPE (rope_interleaved: false) -- this codebase assumes it"
    );
  }
  if (config.hidden_act !== "silu") {
    throw new Error(`Expected hidden_act "silu", got "${config.hidden_act}"`);
  }
}
