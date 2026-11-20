/** Loaded by `bunfig.toml` before every `bun test` file. See `tests/credentials.ts`.
 *
 * It also gives the test process its own OpenCode roots. OpenCode 2.2.7 computes its data, state, cache,
 * config and temp folders once, when `@opencode/util` is first imported, from XDG_* and the temp folder
 * (`@opencode/util/dist/global.js`), so every in-process host a test starts (bridge, driver, contract)
 * wrote its logs, `shell/` outputs and locks into the real `~/.local/share/opencode`, and named the real
 * `%TEMP%\opencode` to the model as approved for external access (found 2026-11-11). The scratch folder
 * lies outside the real home, as sandbox roots do, and is removed after the last test. HOME and
 * USERPROFILE stay: the sandbox and port tests read the real `~/.claude` to copy it. */
import { afterAll } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { stripCredentials } from "./credentials";

stripCredentials(process.env);

const parent = process.platform === "win32" ? "C:\\recall-sandboxes" : tmpdir();
mkdirSync(parent, { recursive: true });
const scratch = mkdtempSync(join(parent, "bun-test-"));
const roots = { XDG_DATA_HOME: "data", XDG_STATE_HOME: "state", XDG_CACHE_HOME: "cache", XDG_CONFIG_HOME: "config", TEMP: "tmp", TMP: "tmp" };
for (const [name, dir] of Object.entries(roots)) {
  mkdirSync(join(scratch, dir), { recursive: true });
  process.env[name] = join(scratch, dir);
}
// A preload's afterAll runs once, after every file; under `bun test` no `exit` or `beforeExit` handler runs
// (measured 2026-11-11, Bun 1.4.4).
afterAll(() => rmSync(scratch, { recursive: true, force: true, maxRetries: 3 }));
