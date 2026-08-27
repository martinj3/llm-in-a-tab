// Byte-level BPE tokenizer built from vocab.json + merges.txt, not
// tokenizer.json (gotcha 8: byte-identical content, 40% smaller, and the
// line-oriented merges format maps directly to a priority table).
import { resolveUrl } from "../models.js";
import { fetchJson } from "./download.js";

// Standard GPT-2 pretokenizer regex, used by the ByteLevel pretokenizer
// (verified against tokenizer.json: pre_tokenizer.pretokenizers[1],
// "use_regex": true). Applied per-segment, after digit splitting below.
const GPT2_SPLIT_REGEX =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

// The 17 added_tokens from tokenizer.json (ids 0-16), hardcoded rather than
// fetched from the 2.1MB tokenizer.json -- verified identical between both
// models. Matched literally, before pretokenization (gotcha 9); everything
// here is plain ASCII punctuation/letters, which self-maps under
// bytesToUnicode, so decode needs no special-casing for them either.
const SPECIAL_TOKENS = [
  "<|endoftext|>",
  "<|im_start|>",
  "<|im_end|>",
  "<repo_name>",
  "<reponame>",
  "<file_sep>",
  "<filename>",
  "<gh_stars>",
  "<issue_start>",
  "<issue_comment>",
  "<issue_closed>",
  "<jupyter_start>",
  "<jupyter_text>",
  "<jupyter_code>",
  "<jupyter_output>",
  "<jupyter_script>",
  "<empty_output>",
];

// GPT-2's byte <-> printable-unicode mapping (Radford et al.,
// bytes_to_unicode()). Printable ASCII/Latin-1 bytes map to themselves;
// every other byte (control chars, space, high bytes of multi-byte UTF-8
// sequences) maps into an unused printable range starting at U+0100, so
// every byte has a visible stand-in that can appear literally in
// vocab.json and merges.txt.
function buildByteToUnicode() {
  const bytes = [];
  for (let b = 0x21; b <= 0x7e; b++) bytes.push(b);
  for (let b = 0xa1; b <= 0xac; b++) bytes.push(b);
  for (let b = 0xae; b <= 0xff; b++) bytes.push(b);
  const chars = bytes.slice();
  let extra = 0;
  for (let b = 0; b < 256; b++) {
    if (!bytes.includes(b)) {
      bytes.push(b);
      chars.push(256 + extra);
      extra++;
    }
  }
  const byteToChar = new Map();
  const charToByte = new Map();
  for (let i = 0; i < bytes.length; i++) {
    const ch = String.fromCodePoint(chars[i]);
    byteToChar.set(bytes[i], ch);
    charToByte.set(ch, bytes[i]);
  }
  return { byteToChar, charToByte };
}

// Digits(individual_digits: true): every ASCII digit becomes its own
// segment, even inside a run of digits -- there are zero multi-digit
// tokens in the vocab (gotcha 7). Splitting on a capturing group keeps the
// digit characters; consecutive digits leave empty strings between them,
// which are dropped.
function splitIndividualDigits(text) {
  return text.split(/(\d)/).filter((s) => s.length > 0);
}

function bpeMerge(symbols, mergeRanks) {
  let current = symbols;
  while (current.length > 1) {
    let bestRank = Infinity;
    let bestIndex = -1;
    for (let i = 0; i < current.length - 1; i++) {
      const rank = mergeRanks.get(current[i] + " " + current[i + 1]);
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestIndex = i;
      }
    }
    if (bestIndex === -1) break;
    current = [
      ...current.slice(0, bestIndex),
      current[bestIndex] + current[bestIndex + 1],
      ...current.slice(bestIndex + 2),
    ];
  }
  return current;
}

export function parseMerges(mergesText) {
  const ranks = new Map();
  let rank = 0;
  for (const line of mergesText.split("\n")) {
    if (line === "" || line.startsWith("#version")) continue;
    ranks.set(line.trimEnd(), rank++);
  }
  return ranks;
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export class Tokenizer {
  constructor(vocab, mergeRanks) {
    this.tokenToId = vocab; // Map<string, number>
    this.idToToken = new Map([...vocab].map(([tok, id]) => [id, tok]));
    this.mergeRanks = mergeRanks; // Map<"A B", rank>
    const { byteToChar, charToByte } = buildByteToUnicode();
    this.byteToChar = byteToChar;
    this.charToByte = charToByte;
    this.specialTokenRegex = new RegExp(
      SPECIAL_TOKENS.slice()
        .sort((a, b) => b.length - a.length)
        .map(escapeRegex)
        .join("|"),
      "g"
    );
  }

  encode(text) {
    const ids = [];
    let lastIndex = 0;
    this.specialTokenRegex.lastIndex = 0;
    let match;
    while ((match = this.specialTokenRegex.exec(text)) !== null) {
      this._encodePlainText(text.slice(lastIndex, match.index), ids);
      ids.push(this.tokenToId.get(match[0]));
      lastIndex = match.index + match[0].length;
    }
    this._encodePlainText(text.slice(lastIndex), ids);
    return ids;
  }

  _encodePlainText(text, ids) {
    for (const digitSegment of splitIndividualDigits(text)) {
      const chunks = digitSegment.match(GPT2_SPLIT_REGEX);
      if (!chunks) continue;
      for (const chunk of chunks) {
        const bytes = new TextEncoder().encode(chunk);
        const symbols = Array.from(bytes, (b) => this.byteToChar.get(b));
        for (const symbol of bpeMerge(symbols, this.mergeRanks)) {
          const id = this.tokenToId.get(symbol);
          if (id === undefined) {
            throw new Error(`No vocab entry for BPE symbol ${JSON.stringify(symbol)}`);
          }
          ids.push(id);
        }
      }
    }
  }

  decode(ids) {
    const bytes = [];
    for (const id of ids) {
      const token = this.idToToken.get(id);
      if (token === undefined) throw new Error(`Unknown token id ${id}`);
      for (const ch of token) bytes.push(this.charToByte.get(ch));
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }
}

export async function loadTokenizer(modelId) {
  const [vocabJson, mergesText] = await Promise.all([
    fetchJson(resolveUrl(modelId, "vocab.json")),
    fetch(resolveUrl(modelId, "merges.txt")).then((r) => {
      if (!r.ok) throw new Error(`GET merges.txt failed: ${r.status}`);
      return r.text();
    }),
  ]);
  const vocab = new Map(Object.entries(vocabJson));
  const mergeRanks = parseMerges(mergesText);
  return new Tokenizer(vocab, mergeRanks);
}
