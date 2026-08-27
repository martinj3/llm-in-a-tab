// Registry of supported models. Everything else in this project reads
// architecture facts from config.json at runtime -- this table only knows
// where to find each model, never its internals.
export const MODELS = {
  "135M": {
    id: "135M",
    label: "SmolLM2-135M-Instruct",
    repo: "HuggingFaceTB/SmolLM2-135M-Instruct",
  },
  "360M": {
    id: "360M",
    label: "SmolLM2-360M-Instruct",
    repo: "HuggingFaceTB/SmolLM2-360M-Instruct",
  },
};

const HF_BASE = "https://huggingface.co";

export function resolveUrl(modelId, filename) {
  const model = MODELS[modelId];
  if (!model) throw new Error(`Unknown model id: ${modelId}`);
  return `${HF_BASE}/${model.repo}/resolve/main/${filename}`;
}
