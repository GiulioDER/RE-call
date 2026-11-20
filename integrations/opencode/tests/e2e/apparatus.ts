/** The precondition every end to end test checks first: the plugin was active in each turn's request. An
 * inactive plugin (no store, no server, a setup that failed) looks like a run without memory, so a block check
 * would fail for the wrong reason, and run set 1's red proof check was red that way. This throws an `Error`,
 * on purpose outside any `expect`: `tests/mutations.ts` counts only a failure at an `expect` as red proof, so
 * a run with an inactive plugin can never pass for one. */

import type { TurnRecord } from "../../harness/driver";
import { SYSTEM_SENTENCE } from "../../src/prefetch";
import { TRAINED } from "../../src/surface";

export function assertPluginActive(records: TurnRecord[]): void {
  if (records.length === 0) throw new Error("apparatus failure: plugin inactive (no turn was recorded)");
  for (const r of records) {
    const problems: string[] = [];
    if (r.request === null) {
      problems.push("the turn made no model request");
    } else {
      const system = r.request.system.map((part) => (typeof part === "string" ? part : JSON.stringify(part))).join("\n");
      if (!system.includes(SYSTEM_SENTENCE)) problems.push("no system sentence");
      const names = new Set(((r.request.tools ?? []) as { function?: { name?: string } }[]).map((t) => t.function?.name));
      const missing = TRAINED.map((t) => t.name).filter((name) => !names.has(name));
      if (missing.length > 0) problems.push(`no ${missing.join(", ")}`);
    }
    if (problems.length > 0) {
      throw new Error(`apparatus failure: plugin inactive (session ${r.session}, turn ${r.turn}: ${problems.join("; ")})`);
    }
  }
}

/** The folder holding the pinned `recall` and `recall-mcp` (RECALL_OPENCODE_BIN), for a live run. Unset or
 * empty, the run would take whatever `recall` PATH finds, an unpinned version, while its results are
 * written up as 0.15.0 (audit H-43), so it is refused before any setup starts. */
export function recallBin(env: Record<string, string | undefined>): string {
  const bin = env.RECALL_OPENCODE_BIN ?? "";
  if (!bin) throw new Error("apparatus failure: RECALL_OPENCODE_BIN is unset; name the folder holding recall and recall-mcp");
  return bin;
}
