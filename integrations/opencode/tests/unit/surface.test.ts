import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { childEnv } from "../../src/child";
import { loadConfig } from "../../src/config";
import type { ToolDef } from "../../src/mcp_client";
import { cleandoc, DROPPED, EXTRA, normalizeTool, selectTools, TRAINED } from "../../src/surface";

/** RE-call's own pin of the six tools, `tests/fixtures/assistant_surface_contract.json` at the repository
 * root, which `tests/test_assistant_surface.py` checks against `recall_mcp.assistant`. */
const CONTRACT = resolve(import.meta.dir, "..", "..", "..", "..", "tests", "fixtures", "assistant_surface_contract.json");

/** The six tools as the contract holds them, in the order the plugin offers them. */
function vendored(): ToolDef[] {
  const tools = (JSON.parse(readFileSync(CONTRACT, "utf-8")) as { tools: Record<string, { description: string; parameters: object }> }).tools;
  return TRAINED.map((t) => t.name).filter((name) => name in tools).map((name) => ({
    name,
    description: tools[name]!.description,
    inputSchema: tools[name]!.parameters,
  }));
}

const def = (name: string): ToolDef => ({ name, description: `the server's own ${name}`, inputSchema: { type: "object", title: name } });
/** What a Lite server with every tool on lists: the trained six as served, and every other tool. */
const SERVER: ToolDef[] = [...TRAINED, ...EXTRA.map((e) => def(e.name)), ...DROPPED.map((d) => def(d.name))];
const WRITES = ["recall_apply_fact", "recall_remember", "recall_forget_memo", ...EXTRA.filter((e) => e.write).map((e) => e.name)];
const config = (options: Record<string, unknown> = {}) => loadConfig(options, resolve("/data"));
const names = (tools: ToolDef[]) => tools.map((t) => t.name);

describe("surface", () => {
  /** Invariant (spec anti-goal): the six assistant tools are, name, description and schema, exactly RE-call's
   * pinned assistant surface, compared as RE-call's own surface test compares them.
   *
   * Node: `tests/unit/surface.test.ts > surface > the six are byte-identical to RE-call's assistant
   * surface contract`. Red proof (`bun run mutations SURFACE-VENDORED SURFACE-VENDORED-SCHEMA`): each
   * mutation fails the test at `expect(ours).toEqual(theirs)`, and the test is green again after each
   * restore. SURFACE-VENDORED: in src/surface.ts, `"Delete one memo from long-term memory by memo_id.`
   * becomes `"Delete a memo from long-term memory by memo_id.`. SURFACE-VENDORED-SCHEMA: `"title": "Memo
   * Id",` becomes `"title": "Memo ID",`. Re-run against the repository contract on 2026-11-20 when the
   * comparison moved off a vendored copy. */
  test("the six are byte-identical to RE-call's assistant surface contract", () => {
    const ours = TRAINED.map((t) => JSON.stringify(normalizeTool(t)));
    const theirs = vendored().map((t) => JSON.stringify(normalizeTool(t)));
    expect(ours).toEqual(theirs);
    expect(names(TRAINED)).toEqual([
      "recall_search", "recall_evidence", "recall_current_facts", "recall_apply_fact", "recall_remember", "recall_forget_memo",
    ]);
  });

  /** Invariant: on a store that is not Lite, no tool that writes is offered, trained or extra.
   *
   * Node: `tests/unit/surface.test.ts > surface > write tools are absent when not Lite`. Red proof
   * (2026-11-10, local Windows, `bun run mutations SURFACE-NOT-LITE`): in src/surface.ts:290,
   * `!writes(name) || (isLite && config.writeTools)` became `!writes(name) || config.writeTools`; the
   * test failed at `expect(offered.filter((n) => WRITES.includes(n))).toEqual([])`. Restored, green. */
  test("write tools are absent when not Lite", () => {
    const offered = names(selectTools(SERVER, config(), false));
    expect(offered.filter((n) => WRITES.includes(n))).toEqual([]);
    expect(offered).toContain("recall_search");
    expect(names(selectTools(SERVER, config({ write_tools: false }), true)).filter((n) => WRITES.includes(n))).toEqual([]);
  });

  /** Invariant: `extra_tools: false` leaves exactly the six.
   *
   * Node: `tests/unit/surface.test.ts > surface > extras are absent when extraTools is false`. Red
   * proof (2026-11-10, local Windows, `bun run mutations SURFACE-EXTRA-OFF`): in src/surface.ts:289,
   * `...(config.extraTools ? EXTRA.map((e) => e.name) : [])` became `...EXTRA.map((e) => e.name)`; the
   * test failed at `expect(names(selectTools(SERVER, config({ extra_tools: false }),
   * true))).toEqual(names(TRAINED))`. Restored, green. */
  test("extras are absent when extraTools is false", () => {
    expect(names(selectTools(SERVER, config({ extra_tools: false }), true))).toEqual(names(TRAINED));
    expect(names(selectTools(SERVER, config(), true))).toEqual([...names(TRAINED), ...EXTRA.map((e) => e.name)]);
  });

  /** Invariant: an extra comes from the server's own definition, and one the server does not list is
   * not offered.
   *
   * Node: `tests/unit/surface.test.ts > surface > an extra the server does not list is not offered`.
   * Red proof (2026-11-10, local Windows, `bun run mutations SURFACE-UNLISTED`): in src/surface.ts:299,
   * `allowed.has(e.name) && served.has(e.name) ? [served.get(e.name)!] : []` became
   * `allowed.has(e.name) ? [served.get(e.name) ?? { name: e.name, description: e.reason, inputSchema:
   * {} }] : []`; the test failed at `expect(names(offered)).not.toContain("recall_withdraw")`.
   * Restored, green. */
  test("an extra the server does not list is not offered", () => {
    const server = SERVER.filter((t) => t.name !== "recall_withdraw");
    const offered = selectTools(server, config(), true);
    expect(names(offered)).not.toContain("recall_withdraw");
    expect(offered.find((t) => t.name === "recall_related")).toBe(server.find((t) => t.name === "recall_related")!);
  });

  /** Invariant: no tool the spec keeps off is selected, whatever the server lists.
   *
   * Node: `tests/unit/surface.test.ts > surface > no operator tool is ever selected`. Red proof
   * (2026-11-10, local Windows, `bun run mutations SURFACE-OPERATOR`): in src/surface.ts:300,
   * `...extras];` became `...serverTools.filter((t) => !TRAINED.some((x) => x.name === t.name))];`; the
   * test failed at `expect(names(selectTools(SERVER, config(options), lite)).filter((n) =>
   * dropped.includes(n))).toEqual([])`. Restored, green. */
  test("no operator tool is ever selected", () => {
    const dropped = DROPPED.map((d) => d.name);
    for (const options of [{}, { write_tools: false }, { extra_tools: false }]) {
      for (const lite of [true, false]) {
        expect(names(selectTools(SERVER, config(options), lite)).filter((n) => dropped.includes(n))).toEqual([]);
      }
    }
  });

  /** Invariant: the child is told to serve exactly the tools the plugin offers, so the model and the
   * server agree on the list.
   *
   * Node: `tests/unit/surface.test.ts > surface > the child is told to serve exactly the offered
   * tools`. Red proof (2026-11-10, local Windows, `bun run mutations SURFACE-CHILD-TOOLS`): in
   * src/child.ts:97, `RECALL_MCP_TOOLS: offeredNames(config, isLiteDsn(config.dsn)).join(","),` became
   * `RECALL_MCP_TOOLS: "assistant",`; the test failed at `expect(childEnv(c,
   * {}).RECALL_MCP_TOOLS!.split(",")).toEqual(...)`. Restored, green. */
  test("the child is told to serve exactly the offered tools", () => {
    for (const options of [{}, { write_tools: false }, { extra_tools: false }]) {
      const c = config(options);
      expect(childEnv(c, {}).RECALL_MCP_TOOLS!.split(",")).toEqual(names(selectTools(SERVER, c, true)));
    }
  });

  /** Invariant: `cleandoc` gives what Python's `inspect.cleandoc` gives (expected values printed by
   * Python 3.14 `inspect.cleandoc` on these inputs).
   *
   * Node: `tests/unit/surface.test.ts > surface > cleandoc matches inspect.cleandoc`. Red proof
   * (2026-11-10, local Windows, `bun run mutations SURFACE-CLEANDOC`): in src/surface.ts:314,
   * `...lines.slice(1).map((l) => l.slice(margin))]` became `...lines.slice(1)]`; the test failed at
   * `expect(cleandoc("First line. ...")).toBe(...)`. Restored, green. */
  test("cleandoc matches inspect.cleandoc", () => {
    expect(cleandoc("First line.\n\n    Indented body\n        deeper\n    back\n\n")).toBe("First line.\n\nIndented body\n    deeper\nback");
    expect(cleandoc("\n\n  Leading blank lines\n  and two\n")).toBe("Leading blank lines\nand two");
    expect(cleandoc("tab\n\tone\n  \ttwo\n")).toBe("tab\none\ntwo");
  });
});
