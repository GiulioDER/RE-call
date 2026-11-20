/** The typecheck's scope (`bun run typecheck`, `tsc --noEmit` over tsconfig.json's `include`). Bun runs
 * TypeScript without checking types, so a file outside `include` is checked only if an included file happens
 * to import it. Red proof row: TSC-INCLUDE in `tests/mutations.integration.ts`. */

import { expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = join(import.meta.dir, "..", "..");

/** Invariant: every top-level directory that holds TypeScript (other than node_modules, out and probes) is in
 * tsconfig.json's `include`, so the bridge plugin's entry point and the CLIs no test imports are type
 * checked. Red: TSC-INCLUDE (the include list back to src, harness, tests and scripts; bridge, memory, port,
 * sandbox and tasks are missing), audit J CODE-001, 2026-11-13. */
test("the typecheck includes every top-level directory that holds TypeScript", () => {
  const include = (JSON.parse(readFileSync(join(ROOT, "tsconfig.json"), "utf8")) as { include: string[] }).include;
  const dirs = readdirSync(ROOT, { withFileTypes: true })
    // probes/ holds Task 1's throwaway OpenCode V2 probes, kept as evidence; nothing imports or runs them.
    .filter((e) => e.isDirectory() && !e.name.startsWith(".") && !["node_modules", "out", "probes"].includes(e.name))
    .map((e) => e.name)
    .filter((d) => readdirSync(join(ROOT, d), { recursive: true }).some((f) => String(f).endsWith(".ts")));
  expect(dirs.filter((d) => !include.includes(d))).toEqual([]);
});
