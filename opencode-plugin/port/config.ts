/** Write an OpenCode 2.2.7 `opencode.json` (schema `Config.Info` in `@opencode/schema`).
 *
 * In `instructions`, OpenCode expands a leading `~/` against its home, reads an absolute path as
 * given, and finds a relative name by walking up from the session's directory, so
 * `["~/.claude/CLAUDE.md", "CLAUDE.md", "CLAUDE.local.md"]` names the global layer and the project's
 * two (read in `@opencode/core`, `ConfigInstructionPlugin.match`). */

import * as fs from "node:fs";
import { join } from "node:path";

export function writeConfig(dest: string, opts: { instructions: string[]; plugins: object[]; mcp: object }): void {
  fs.mkdirSync(dest, { recursive: true });
  const config = { instructions: opts.instructions, plugins: opts.plugins, mcp: opts.mcp };
  fs.writeFileSync(join(dest, "opencode.json"), JSON.stringify(config, null, 2) + "\n", "utf-8");
}
