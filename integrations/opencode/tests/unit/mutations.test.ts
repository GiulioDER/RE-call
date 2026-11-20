import { afterEach, describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { applyMutation, failedAssertions, JOURNAL, recoverInterrupted, verdict } from "../mutations";
import { tempDir } from "./fake";

const dirs: string[] = [];
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true });
});

/** A throwaway root with one source file, so no test touches the repository's own `src/`. */
function root(): string {
  const dir = tempDir();
  dirs.push(dir);
  mkdirSync(join(dir, "src"));
  writeFileSync(join(dir, "src", "x.ts"), "original\n");
  return dir;
}

describe("mutation runner", () => {
  /** Invariant (audit H-29): a row left applied by a run that was killed before its restore is undone by
   * the next run, so a mutation (a write gate turned off, say) does not stay in the source OpenCode loads.
   *
   * Node: `tests/unit/mutations.test.ts > mutation runner > a row an interrupted run left applied is
   * undone by the next run`. Red proof (2026-11-11, local Windows, `bun run mutations MUT-RECOVER`): in
   * tests/mutations.ts `recoverInterrupted`, `restoreMutation(root, file, original);\n  return file;`
   * became `return file;`; the test failed at `expect(readFileSync(file, "utf-8")).toBe("original\n")`.
   * Restored, green. */
  test("a row an interrupted run left applied is undone by the next run", () => {
    const dir = root();
    const file = join(dir, "src", "x.ts");
    applyMutation(dir, "src/x.ts", "original\n", "mutated\n");
    expect(readFileSync(file, "utf-8")).toBe("mutated\n"); // the run is killed here: no restore
    expect(recoverInterrupted(dir)).toBe("src/x.ts");
    expect(readFileSync(file, "utf-8")).toBe("original\n");
    expect(existsSync(join(dir, JOURNAL))).toBe(false);
    expect(recoverInterrupted(dir)).toBeUndefined();
  });

  /** Invariant (audit H-29): recovery never overwrites a file changed since the interruption; it refuses
   * and says where the original is.
   *
   * Node: `tests/unit/mutations.test.ts > mutation runner > a file changed since the interruption is
   * not overwritten`. Red proof (2026-11-11, local Windows, `bun run mutations MUT-RECOVER-CHANGED`): in
   * tests/mutations.ts `recoverInterrupted`, `if (now !== mutated) {` became `if (now === "never") {`; the test failed
   * at `expect(() => recoverInterrupted(dir)).toThrow(...)`. Restored, green. */
  test("a file changed since the interruption is not overwritten", () => {
    const dir = root();
    const file = join(dir, "src", "x.ts");
    applyMutation(dir, "src/x.ts", "original\n", "mutated\n");
    writeFileSync(file, "edited after\n");
    expect(() => recoverInterrupted(dir)).toThrow("src/x.ts changed after an interrupted mutation run");
    expect(readFileSync(file, "utf-8")).toBe("edited after\n");
  });

  /** Invariant (audit H-29, differential review): a run killed after its journal was written but before
   * the mutation reached the file (or after the restore, before the journal was removed) leaves the file
   * as it was; the next run clears the journal and goes on, instead of refusing.
   *
   * Node: `tests/unit/mutations.test.ts > mutation runner > a journal whose file is already the original
   * is cleared`. Red proof (2026-11-11, local Windows, `bun run mutations MUT-RECOVER-CLEAN`): in
   * tests/mutations.ts `recoverInterrupted`, `if (now === original) {` became `if (now === "never") {`;
   * the test failed at `expect(refused).toBeUndefined()`. Restored, green. */
  test("a journal whose file is already the original is cleared", () => {
    const dir = root();
    applyMutation(dir, "src/x.ts", "original\n", "mutated\n");
    writeFileSync(join(dir, "src", "x.ts"), "original\n"); // the restore ran, the journal's removal did not
    let recovered: string | undefined = "not called";
    let refused: string | undefined;
    try {
      recovered = recoverInterrupted(dir);
    } catch (error) {
      refused = (error as Error).message;
    }
    expect(refused).toBeUndefined();
    expect(recovered).toBeUndefined();
    expect(existsSync(join(dir, JOURNAL))).toBe(false);
  });

  // -- the intended assertion (owner, 2026-11-11: every row names the expect that must fail) ----------

  /** A fixture test file in a throwaway folder, run by bun the way `runTest` runs a row's test. */
  function runFixture(name: string): string {
    const dir = tempDir();
    dirs.push(dir);
    const file = join(dir, "fixture.test.ts");
    writeFileSync(
      file,
      [
        'import { expect, test } from "bun:test";',
        'test("fixture fails at its second expect", () => {',
        "  expect(1).toBe(1);",
        "  expect([1, 2]).toEqual([1, 3]);",
        "});",
        'test("fixture fails at its first expect", () => {',
        "  expect(1).toBe(2);",
        "  expect([1, 2]).toEqual([1, 3]);",
        "});",
      ].join("\n"),
      "utf-8",
    );
    const proc = Bun.spawnSync(["bun", "test", file, "-t", name], { cwd: dir, stdout: "pipe", stderr: "pipe" });
    return proc.stdout.toString() + proc.stderr.toString();
  }
  const SECOND = "expect([1, 2]).toEqual([1, 3]);";

  /** Invariant (DEF-50): the assertion a failure is read at is the source line bun's code frame prints
   * above its `^`, trimmed of the line number.
   *
   * Node: `tests/unit/mutations.test.ts > mutation runner > the failing assertion is read from bun's code
   * frame`. Red proof, row MUT-ASSERT-FRAME: in tests/mutations.ts `failedAssertions`, `lines[i - 2]`
   * replaced by `lines[i - 3]` (the line above it); fails at
   * `expect(failedAssertions(out)).toEqual([SECOND])`. */
  test("the failing assertion is read from bun's code frame", () => {
    const out = runFixture("fixture fails at its second expect");
    expect(failedAssertions(out)).toEqual([SECOND]);
  });

  /** Invariant (DEF-50): a row is red only where it says: a failure at its own assertion is `expect`, and
   * a failure at another expect of the same test (a precondition) is `elsewhere`, a survivor.
   *
   * Node: `tests/unit/mutations.test.ts > mutation runner > a failure at another expect is not red`. Red
   * proof, row MUT-ASSERT-PIN: in tests/mutations.ts `verdict`, `? "expect" : "elsewhere";` replaced by
   * `? "expect" : "expect";`; fails at `expect(verdict(early, "fixture fails at its first expect",
   * SECOND)).toBe("elsewhere")`. */
  test("a failure at another expect is not red", () => {
    const late = runFixture("fixture fails at its second expect");
    expect(verdict(late, "fixture fails at its second expect", SECOND)).toBe("expect");
    const early = runFixture("fixture fails at its first expect");
    expect(failedAssertions(early)).toEqual(["expect(1).toBe(2);"]);
    expect(verdict(early, "fixture fails at its first expect", SECOND)).toBe("elsewhere");
  });

  /** Invariant (DEF-50): a row with no assertion recorded is never red; a failure only at expects is
   * `unpinned`, which the runner lists as UNKNOWN with the line for the backfill.
   *
   * Node: `tests/unit/mutations.test.ts > mutation runner > a row without an assertion is never red`.
   * Red proof, row MUT-ASSERT-UNPINNED: in tests/mutations.ts `verdict`, the line
   * `if (assertion === undefined) return "unpinned";` removed; fails at
   * `expect(verdict(late, "fixture fails at its second expect")).toBe("unpinned")`. */
  test("a row without an assertion is never red", () => {
    const late = runFixture("fixture fails at its second expect");
    expect(verdict(late, "fixture fails at its second expect")).toBe("unpinned");
  });
});
