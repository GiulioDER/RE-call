/** A throwaway home for a test that starts OpenCode: every variable through which OpenCode finds a home or
 * its data, state, cache and config roots points inside it, and no credential reaches it.
 *
 * The root lies outside the real home, because OpenCode reads `.claude`, `.agents` and `.opencode` in
 * every ancestor of its directory, and on Windows the temp folder sits inside the real home. */

import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, parse } from "node:path";
import { stripCredentials } from "./credentials";

export type IsolatedHome = { home: string; env: Record<string, string>; dispose(): void };

/** The parent of every isolated home: `<drive>\recall-sandboxes` on Windows, the temp folder elsewhere. */
export const SANDBOX_PARENT = process.platform === "win32" ? join(parse(tmpdir()).root, "recall-sandboxes") : tmpdir();

export function isolatedHome(prefix = "home-"): IsolatedHome {
  mkdirSync(SANDBOX_PARENT, { recursive: true });
  const root = mkdtempSync(join(SANDBOX_PARENT, prefix));
  const home = join(root, "home");
  const own: Record<string, string> = {
    USERPROFILE: home,
    HOME: home,
    APPDATA: join(home, "AppData", "Roaming"),
    LOCALAPPDATA: join(home, "AppData", "Local"),
    TEMP: join(home, "AppData", "Local", "Temp"),
    TMP: join(home, "AppData", "Local", "Temp"),
    XDG_CONFIG_HOME: join(home, ".config"),
    XDG_DATA_HOME: join(home, ".local", "share"),
    XDG_STATE_HOME: join(home, ".local", "state"),
    XDG_CACHE_HOME: join(home, ".cache"),
  };
  for (const dir of new Set(Object.values(own))) mkdirSync(dir, { recursive: true });
  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) if (v !== undefined && !["PWD", "OLDPWD"].includes(k)) env[k] = v;
  stripCredentials(env);
  Object.assign(env, own);
  return { home, env, dispose: () => rmSync(root, { recursive: true, force: true, maxRetries: 5 }) };
}
