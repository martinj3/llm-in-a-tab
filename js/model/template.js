// ChatML formatting, hardcoded rather than run through a Jinja interpreter
// (plan.md Phase 4). This is a direct translation of tokenizer_config.json's
// chat_template for these models -- verified character-for-character
// against a real Jinja2 rendering of that template (reference/golden/
// chat_template_cases.json).
export const DEFAULT_SYSTEM_MESSAGE =
  "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

// The default system message is injected only when there is at least one
// message and the first one isn't already a system message -- supplying
// your own replaces it, it does not append (plan.md 2).
export function renderChatPrompt(messages, addGenerationPrompt) {
  let out = "";
  if (messages.length > 0 && messages[0].role !== "system") {
    out += `<|im_start|>system\n${DEFAULT_SYSTEM_MESSAGE}<|im_end|>\n`;
  }
  for (const message of messages) {
    out += `<|im_start|>${message.role}\n${message.content}<|im_end|>\n`;
  }
  if (addGenerationPrompt) {
    out += "<|im_start|>assistant\n";
  }
  return out;
}

// No BOS is prepended (gotcha 13): tokenizer_config.json's post_processor
// is null, and the leading <|im_start|> above already belongs to the
// template's first message.
export function encodeChatPrompt(tokenizer, messages, addGenerationPrompt) {
  return tokenizer.encode(renderChatPrompt(messages, addGenerationPrompt));
}

// --- Incremental (per-turn) rendering, for the persisted KV cache -------
//
// The template above is pure concatenation, so a conversation can be fed
// to the model one turn at a time and the tokens are identical to
// rendering the whole thing and encoding it in one go (gotcha 22: turn 2
// must prefill only the new turn, never the whole conversation). That
// equivalence is exactly what tests/chat-turns.mjs asserts -- it is not
// obvious for free, because it relies on special tokens being matched
// literally before pretokenization (gotcha 9), which is what keeps the
// lone "\n" between <|im_end|> and <|im_start|> encoding the same way in
// isolation as it does mid-string.

// Prefilled once at model load, before the user has typed anything.
export function renderSystemTurn(systemMessage = DEFAULT_SYSTEM_MESSAGE) {
  return `<|im_start|>system\n${systemMessage}<|im_end|>\n`;
}

// One user turn plus the generation prompt. The model picks up right
// after the trailing "<|im_start|>assistant\n" and stops when it emits
// <|im_end|> (gotcha 14).
export function renderUserTurn(content) {
  return `<|im_start|>user\n${content}<|im_end|>\n<|im_start|>assistant\n`;
}

// What has to be fed to the cache after a reply, before the next user
// turn, to leave the context in canonical template form. If the model
// emitted <|im_end|> itself, only the newline the template puts after it
// is missing; if generation was cut short (token budget, stop button),
// the closing tag is missing too and we supply it -- leaving the context
// mid-turn would put the next prompt off-format, which collapses these
// models to base-model rambling (gotcha 16).
export function renderTurnClose(endedWithEos) {
  return endedWithEos ? "\n" : "<|im_end|>\n";
}
