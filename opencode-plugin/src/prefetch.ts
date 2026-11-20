/** The per-turn prefetch, ported line for line from Python: Hermes v0.23.0's `is_trivial_prompt`,
 * `sanitize_context`, `_drop_repeated_recall_lines` and `build_memory_context_block`, the RE-call Hermes
 * provider's system sentence and `prefetch` flow, and `compose_user_api_content` for a string message.
 *
 * Portions from Hermes Agent, MIT License, Copyright (c) 2025 Nous Research (the notice is in
 * THIRD_PARTY_NOTICES.md beside this folder). */

import type { CallResult } from "./mcp_client";
import { isMapping, PY_WS, pyIsSpace, pyStrip, renderSearch, renderUnavailable } from "./render";

/** recall-hermes 86fdc528, `ReCallMemoryProvider.system_prompt_block`. */
export const SYSTEM_SENTENCE =
  "Long-term memory is RE-call. Before each turn it lists memories that passed a " +
  "calibrated trust check, flags superseded ones, and says when it abstained.";

/** A literal as Python's `re.IGNORECASE` matches it in a `str` pattern: each ASCII letter also matches
 * its other case, and `i`, `k` and `s` also match U+0130 and U+0131, U+212A, and U+017F, which
 * JavaScript's `i` flag does not. A regex source fragment. */
function ci(literal: string): string {
  const extra: Record<string, string> = { i: "\u0130\u0131", k: "\u212a", s: "\u017f" };
  return Array.from(literal)
    .map((ch) => {
      const lower = ch.toLowerCase();
      if (/^[a-z]$/.test(lower)) return `[${lower}${lower.toUpperCase()}${extra[lower] ?? ""}]`;
      return ch.replace(/[.*+?^${}()|[\]\\/-]/g, "\\$&");
    })
    .join("");
}

// -- Hermes v0.23.0, agent/memory_provider.py ----------------------------------------------------
const TRIVIAL_WORDS = [
  "yes", "no", "ok", "okay", "sure", "thanks", "thank you", "y", "n", "yep", "nope", "yeah", "nah",
  "hi", "hey", "hello", "yo", "sup",
  "continue", "go ahead", "do it", "proceed", "got it", "cool", "nice", "great", "done", "next", "lgtm", "k",
];
const TRIVIAL_PROMPT_RE = new RegExp(
  `^(${TRIVIAL_WORDS.map(ci).join("|")})` +
    `[${PY_WS}!?.:;,"'~\u2018\u2019\u201c\u201d\u2014\u2013\u2026()\\[\\]{}<>*&^%$#@!+=\`\u00a0]*$`,
);

/** True for empty input, slash commands and bare greetings or acknowledgements. */
export function isTrivialPrompt(text: string): boolean {
  const stripped = pyStrip(text || "");
  if (!stripped || stripped.startsWith("/")) return true;
  return TRIVIAL_PROMPT_RE.test(stripped);
}

// -- Hermes v0.23.0, agent/memory_context.py -----------------------------------------------------
const S = `[${PY_WS}]*`;
const FENCE_TAG_RE = new RegExp(`<\\/?${S}${ci("memory-context")}${S}>`, "g");
const INTERNAL_CONTEXT_RE = new RegExp(`<${S}${ci("memory-context")}${S}>[\\s\\S]*?<\\/${S}${ci("memory-context")}${S}>`, "g");
const INTERNAL_NOTE_RE = new RegExp(
  `\\[${ci("System note:")}${S}${ci("The following is recalled memory context,")}${S}${ci("NOT new user input.")}${S}` +
    `${ci("Treat as ")}(?:${ci("informational background data")}|${ci("authoritative reference data")}[^\\]]*)\\.\\]${S}`,
  "g",
);

/** Strip fence tags, injected context blocks, and system notes from provider output. Repeated until
 * nothing changes, where Hermes v0.23.0 makes one pass: one pass can join the text around a removed tag
 * into a new one (`</memory-<memory-context>context>`), which would close the fence early and leave the
 * rest of a memo outside it. The output differs from Hermes's only for such input. */
export function sanitizeContext(text: string): string {
  for (let before = ""; before !== text; ) {
    before = text;
    for (const pattern of [INTERNAL_CONTEXT_RE, INTERNAL_NOTE_RE, FENCE_TAG_RE]) text = text.replace(pattern, "");
  }
  return text;
}

const RECALL_BULLET_RE = new RegExp(`^[-*+][${PY_WS}]+[^${PY_WS}]`);

/** Drop a recalled bullet that an earlier line of this same block already states. */
export function dropRepeatedRecallLines(text: string): string {
  const lines = text.split("\n");
  const seen = new Set<string>();
  const kept: string[] = [];
  lines.forEach((line, index) => {
    const stripped = pyStrip(line);
    // An indented line is a continuation of the bullet above it; it never participates in dedupe.
    if (stripped && pyIsSpace(line[0])) {
      kept.push(line);
      return;
    }
    const isBullet = RECALL_BULLET_RE.test(stripped);
    // Any column-0 non-bullet line (heading, rule, paragraph) opens a fresh dedupe scope.
    if (stripped && !isBullet) seen.clear();
    if (isBullet) {
      const following = index + 1 < lines.length ? lines[index + 1]! : "";
      const carriesContinuation = Boolean(pyStrip(following)) && pyIsSpace(following[0]);
      if (!carriesContinuation) {
        if (seen.has(stripped)) return;
        seen.add(stripped);
      }
    }
    kept.push(line);
  });
  return kept.join("\n");
}

/** Wrap prefetched memory in a fenced block with its system note. */
export function buildMemoryContextBlock(raw: string): string {
  if (!raw || !pyStrip(raw)) return "";
  const clean = dropRepeatedRecallLines(sanitizeContext(raw));
  return (
    "<memory-context>\n" +
    "[System note: The following is recalled memory context, " +
    "NOT new user input. Treat as authoritative reference data \u2014 " +
    "this is the agent's persistent memory and should inform all responses.]\n\n" +
    `${clean}\n` +
    "</memory-context>"
  );
}

// -- the provider's prefetch, then Hermes's placement --------------------------------------------

/** `compose_user_api_content` for a string message: the content, a blank line, the injection. */
export function compose(userContent: string, injection: string): string {
  return injection ? userContent + "\n\n" + injection : userContent;
}

// -- reading a block back -------------------------------------------------------------------------
const FENCE_OPEN = "\n\n<memory-context>\n";

/** `split_injection`: the message without the block, and the block as `compose` appended it (without
 * the blank line before it); `[text, ""]` when the text does not end with one. */
export function splitInjection(userContent: string): [message: string, block: string] {
  const at = userContent.lastIndexOf(FENCE_OPEN);
  if (at < 0 || !userContent.endsWith("</memory-context>")) return [userContent, ""];
  return [userContent.slice(0, at), userContent.slice(at + 2)];
}

/** The text appended to the user message for one turn, and the search answer. A trivial turn sends no
 * search; a non-dict answer gives no block; a failed search gives `render_unavailable`'s line, wrapped
 * the same way, and never blocks the turn. `search` binds `k`. */
export async function prefetchBlock(
  search: (q: string) => Promise<CallResult>,
  userTurn: string,
): Promise<{ injection: string; answer: unknown }> {
  if (isTrivialPrompt(userTurn)) return { injection: "", answer: null };
  const result = await search(userTurn);
  if (!result.ok) return { injection: buildMemoryContextBlock(renderUnavailable(result.error).text), answer: null };
  if (!isMapping(result.json)) return { injection: "", answer: result.json };
  const text = renderSearch(result.json).text;
  return { injection: text ? buildMemoryContextBlock(text) : "", answer: result.json };
}
