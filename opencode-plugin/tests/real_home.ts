/** The user's real home, for the checks that must never mistake a redirected home for it: the sandbox
 * builder's "a root inside the real home is refused" and the hook bridge's off switch.
 *
 * Read at call time, never at import: `os.homedir()` and `os.userInfo().homedir` both follow USERPROFILE
 * and HOME (measured, Bun 1.4.4), so a value taken when the module loads is whatever home the process had
 * then. In one `bun test` process a test that redirects USERPROFILE made a sandbox builder, loaded at
 * that moment, take the decoy for the real home and stop refusing roots under it (2026-11-11). On Windows
 * the logon profile comes from HKCU\Volatile Environment, which a process's environment cannot move.
 *
 * When the registry gives no answer it throws, and never answers with the process's own home: under a
 * sandbox that is the sandbox, so the builder would wave through a root under the real home and the bridge
 * would take a sandbox for the real home (one bridge in the full-stack suite of 2026-11-11 logged "off:
 * home is the real home" that way). */

import { spawnSync } from "node:child_process";
import { userInfo } from "node:os";
import { join } from "node:path";

const ATTEMPTS = 3;

export function realHome(): string {
  if (process.platform !== "win32") return userInfo().homedir;
  // By absolute path: a sandbox puts its own bin first on PATH, where a `reg` could name any home.
  const reg = join(process.env.SystemRoot || "C:\\Windows", "System32", "reg.exe");
  let detail = "";
  for (let attempt = 0; attempt < ATTEMPTS; attempt++) {
    const out = spawnSync(reg, ["query", "HKCU\\Volatile Environment", "/v", "USERPROFILE"], { encoding: "utf8", windowsHide: true, timeout: 10_000 });
    const match = /USERPROFILE\s+REG_\w+\s+(.+)/.exec(out.stdout ?? "");
    if (match) return match[1]!.trim();
    detail = out.error ? String(out.error) : `exit ${out.status}: ${(out.stderr ?? "").trim()}`;
  }
  throw new Error(`cannot read the logon profile from HKCU\\Volatile Environment with ${reg} (${detail})`);
}
