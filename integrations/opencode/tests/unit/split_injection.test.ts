import { expect, test } from "bun:test";
import { stripBlocks } from "../../src/plugin";
import { buildMemoryContextBlock, compose, splitInjection } from "../../src/prefetch";
import { fixture } from "./fixtures";

/**
 * Invariant: `splitInjection` gives exactly what the Python original `split_injection` gives on every
 * fixture in `tests/fixtures/render`: the block from the last
 * "\n\n<memory-context>\n" without its two newlines, when the text ends with "</memory-context>",
 * and `[text, ""]` otherwise. The fixtures include every composed prefetch turn, a fence pasted
 * mid-text, two fences, a closing tag with no fenced opening, and a trailing newline.
 * Red proof, row SPLIT-SLICE: `userContent.slice(at + 2)]` replaced by `userContent.slice(at)]` in
 * src/prefetch.ts; fails at `expect(cases.map(...)).toEqual(cases.map((c) => c.output))`.
 * Row SPLIT-TRAILING: `!userContent.endsWith("</memory-context>")` replaced by `false`; fails at the
 * same `expect`. Row SPLIT-NO-OPEN: `at < 0 || ` removed; fails at the same `expect`.
 */
test("splitInjection: every fixture agrees with split_injection", () => {
  const cases = fixture<{ input: string; output: [string, string] }[]>("split_injection.json");
  expect(cases.length).toBeGreaterThan(20);
  expect(cases.map((c) => splitInjection(c.input))).toEqual(cases.map((c) => c.output));
});

/** Invariant (audit H-34): the title hook takes out only the block `compose` appended. A user's own text
 * that holds the block's opening marker comes back whole; the block, and so every memo line, is gone,
 * also when an unclosed marker follows the block.
 *
 * Node: `tests/unit/split_injection.test.ts > stripBlocks keeps user text that holds the opening
 * marker`. Red proof, row PLUGIN-STRIP-LAST: in src/plugin.ts `stripBlocks`,
 * `text.lastIndexOf(BLOCK_OPEN, from)` replaced by `text.indexOf(BLOCK_OPEN)` (the pre-fix forward
 * search); fails at `expect(stripBlocks(composed)).toBe(user)` (the user's text is cut at its marker). Row
 * PLUGIN-STRIP-UNCLOSED: `from = at - 1;\n      continue;` replaced by `return text;` (the search stops
 * at the unclosed marker); fails at `expect(stripBlocks(composed + marker)).toBe(user + marker)`. */
test("stripBlocks keeps user text that holds the opening marker", () => {
  const block = buildMemoryContextBlock("- [ok] memo m-1: The user works at Acme.");
  const lead = "NOT new user input.";
  const marker = "\n\n" + block.slice(0, block.indexOf(lead) + lead.length);
  const user = `Please explain this string:${marker} and what it does.`;
  const composed = compose(user, block);
  expect(composed).toContain("works at Acme");
  expect(stripBlocks(composed)).toBe(user);
  expect(stripBlocks(composed + marker)).toBe(user + marker);
});
