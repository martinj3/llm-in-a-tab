// ChatML formatting, hardcoded rather than run through a Jinja interpreter
// (plan.md Phase 4). This is a direct translation of tokenizer_config.json's
// chat_template for these models -- verified character-for-character
// against a real Jinja2 rendering of that template (reference/golden/
// chat_template_cases.json).
const DEFAULT_SYSTEM_MESSAGE =
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
