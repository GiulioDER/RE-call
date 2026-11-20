import { describe, expect, test } from "bun:test";

import { CREDENTIAL, stripCredentials } from "../credentials";
import { realHome } from "../real_home";

describe("test preload", () => {
  /** Invariant: every credential-shaped name is removed and nothing else is.
   * Node: `tests/unit/preload.test.ts > test preload > provider keys and tokens are removed, other variables kept`.
   * Red proof (2026-11-11): mutating `tests/credentials.ts` `for (const k of removed) delete env[k];` to a no-op
   * failed at the first `toBeUndefined()`; restored, green. */
  test("provider keys and tokens are removed, other variables kept", () => {
    const env: Record<string, string | undefined> = {
      OPENROUTER_API_KEY: "x", VOYAGE_API_KEY: "x", GITHUB_PERSONAL_ACCESS_TOKEN: "x", ANTHROPIC_BASE_URL: "x",
      AWS_SECRET_ACCESS_KEY: "x", PATH: "p", USERPROFILE: "u", RECALL_TRUST_MODE: "strict",
    };
    const removed = stripCredentials(env);
    expect(env.OPENROUTER_API_KEY).toBeUndefined();
    expect(env.VOYAGE_API_KEY).toBeUndefined();
    expect(env.GITHUB_PERSONAL_ACCESS_TOKEN).toBeUndefined();
    expect(env.ANTHROPIC_BASE_URL).toBeUndefined();
    expect(env.AWS_SECRET_ACCESS_KEY).toBeUndefined();
    expect(env.PATH).toBe("p");
    expect(env.USERPROFILE).toBe("u");
    expect(env.RECALL_TRUST_MODE).toBe("strict");
    expect(removed.sort()).toEqual(["ANTHROPIC_BASE_URL", "AWS_SECRET_ACCESS_KEY", "GITHUB_PERSONAL_ACCESS_TOKEN", "OPENROUTER_API_KEY", "VOYAGE_API_KEY"]);
  });

  /** Invariant: the preload ran before this file, so the test process itself holds no credential.
   * Node: `tests/unit/preload.test.ts > test preload > the test process holds no credential`.
   * Red proof (2026-11-11): removing the preload line from `bunfig.toml` failed at `toEqual([])` on this
   * workstation, whose environment carries OPENROUTER_API_KEY; restored, green. */
  test("the test process holds no credential", () => {
    expect(Object.keys(process.env).filter((k) => CREDENTIAL.test(k))).toEqual([]);
  });

  /** Invariant: OpenCode, imported in a test process, keeps its data, state, cache, config and temp roots in
   * the process's own scratch folder, outside the real home. OpenCode computes them once, at import
   * (`@opencode/util/dist/global.js`), so before this every in-process test (bridge, driver, contract)
   * wrote its logs, `shell/` outputs and locks into the real `~/.local/share/opencode`, and named the real
   * `%TEMP%\opencode` to the model (found 2026-11-11). Red: PRELOAD-ROOTS (the redirect removed). */
  test("OpenCode's global roots in a test process are outside the real home", async () => {
    const { Path } = await import("@opencode/util/global");
    const home = realHome().toLowerCase();
    const roots = { data: Path.data, state: Path.state, cache: Path.cache, config: Path.config, tmp: Path.tmp };
    const inside = Object.entries(roots).filter(([, p]) => p.toLowerCase().startsWith(home + "\\") || p.toLowerCase().startsWith(home + "/")).map(([k]) => k);
    expect(inside).toEqual([]);
  });
});
