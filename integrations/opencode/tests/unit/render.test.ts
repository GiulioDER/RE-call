import { describe, expect, test } from "bun:test";
import { pyJsonDumps } from "../../src/plugin";
import { type Rendered, renderSearch, renderUnavailable } from "../../src/render";
import { fixture } from "./fixtures";

describe("render", () => {
  /** Invariant: `renderSearch` gives, for every fixture, exactly the text and count recall-hermes
   * 86fdc528 `render_search` gives.
   *
   * Node: `tests/unit/render.test.ts > render > renderSearch: every fixture renders byte-identically`.
   * Red proof (2026-11-10, local Windows, `bun run mutations RENDER-CITABLE RENDER-RETIRED
   * RENDER-ABSTAIN RENDER-CLIP RENDER-CLIP-BOUND RENDER-CODEPOINT RENDER-PYWS RENDER-TRUTHY
   * RENDER-PYSTR RENDER-VALIDITY`): each mutation failed the test at
   * `expect(cases.map(...renderSearch...)).toEqual(...)`, and the test was green again after each
   * restore. RENDER-CITABLE: in src/render.ts (row RENDER-CITABLE), `&& (ids === null || ids.has(pyStr(hit.chunk_id)))`
   * became `&& true`. RENDER-RETIRED: in src/render.ts (row RENDER-RETIRED), `if (RETIRED_VERDICTS.has(verdict)) {`
   * became `if (false) {`. RENDER-ABSTAIN: in src/render.ts (row RENDER-ABSTAIN), `!abstained && verdict === "ok"`
   * became `verdict === "ok"`. RENDER-CLIP: in src/render.ts (row RENDER-CLIP), `points.slice(0, MAX_HIT_CHARS - 3)`
   * became `points.slice(0, MAX_HIT_CHARS)`. RENDER-CLIP-BOUND: in src/render.ts (row RENDER-CLIP-BOUND), `points.length <=
   * MAX_HIT_CHARS ?` became `points.length < MAX_HIT_CHARS ?`. RENDER-CODEPOINT: in src/render.ts (row RENDER-CODEPOINT),
   * `const points = Array.from(flat);` became `const points = flat.split("");`. RENDER-PYWS: in
   * src/render.ts (row RENDER-PYWS), `\\x1c-\\x20\\x85` became `\\x20`. RENDER-TRUTHY: in src/render.ts (row RENDER-TRUTHY), `if
   * (isMapping(v)) return Object.keys(v).length > 0;` became `(nothing)`. RENDER-PYSTR: in
   * src/render.ts (row RENDER-PYSTR), `if (v === true) return "True";` became `if (v === true) return "true";`.
   * RENDER-VALIDITY: in src/render.ts (row RENDER-VALIDITY), `parts.join(", ")` became `parts.join("; ")`. */
  test("renderSearch: every fixture renders byte-identically", () => {
    const cases = fixture<{ name: string; input: unknown; output: Rendered }[]>("render_search.json");
    expect(cases.length).toBeGreaterThan(25);
    expect(cases.map((c) => ({ name: c.name, output: renderSearch(c.input) }))).toEqual(
      cases.map((c) => ({ name: c.name, output: c.output })),
    );
  });

  /** Invariant: `renderUnavailable` gives, for every fixture, exactly recall-hermes's line.
   *
   * Node: `tests/unit/render.test.ts > render > renderUnavailable: every fixture renders
   * byte-identically`. Red proof (2026-11-10, local Windows, `bun run mutations RENDER-UNAVAILABLE`):
   * in src/render.ts (row RENDER-UNAVAILABLE), `could not answer this turn: ${clip(reason)}` became `could not answer this
   * turn: ${reason}`; the test failed at `expect(cases.map((c) =>
   * renderUnavailable(c.input))).toEqual(...)`. Restored, green. */
  test("renderUnavailable: every fixture renders byte-identically", () => {
    const cases = fixture<{ input: string; output: Rendered }[]>("render_unavailable.json");
    expect(cases.map((c) => renderUnavailable(c.input))).toEqual(cases.map((c) => c.output));
  });
});

/** Invariant (audit H-04): a tool result's string is escaped as Python's `json.dumps` escapes it, so DEL
 * (U+007F) comes out as `\u007f`, like every other character outside 0x20 to 0x7e. The expected bytes
 * are what `python -I -c "import json; print(json.dumps('a\x7fb\x1fé~'))"` printed on 2026-11-11.
 *
 * Node: `tests/unit/render.test.ts > pyJsonDumps escapes DEL as Python does`. Red proof, row
 * PLUGIN-DUMPS-DEL: in src/plugin.ts `pyJsonDumps`, `code >= 0x7f` replaced by `code > 0x7f`; fails at
 * `expect(pyJsonDumps("a\x7fb\x1fé~")).toBe(...)`. */
test("pyJsonDumps escapes DEL as Python does", () => {
  expect(pyJsonDumps("a\x7fb\x1fé~")).toBe('"a\\u007fb\\u001f\\u00e9~"');
});
