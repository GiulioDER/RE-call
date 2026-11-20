/** Install the ported stack into an OpenCode config directory: the ported commands and agents, and an
 * `opencode.json` naming the three CLAUDE.md layers and the RE-call memory plugin. The plugin is named by
 * its directory in this checkout, since it imports `@opencode/plugin` from this folder's `node_modules`.
 * Skills need nothing: OpenCode reads `~/.claude/skills` and `~/.agents/skills` itself. The memory plugin
 * reports itself inactive until its store exists, and logs the command that creates it.
 *
 * Nothing is overwritten: an existing `opencode.json`, `commands/` or `agents/` stops the install before
 * anything is written. Tests run `install` against a temporary home only. Against your own home:
 *
 *     bun port/install.ts --real-home [--recall-command <path to recall-mcp>] */

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { portAgents } from "./agents";
import { portCommands } from "./commands";
import { writeConfig } from "./config";

const ROOT = resolve(import.meta.dir, "..");

export const INSTRUCTIONS = ["~/.claude/CLAUDE.md", "CLAUDE.md", "CLAUDE.local.md"];

export type InstallOptions = {
  /** The home whose `.claude` is read and whose OpenCode config is written. */
  home: string;
  /** `XDG_CONFIG_HOME`, which OpenCode reads ahead of `<home>/.config`. */
  xdgConfigHome?: string;
  /** The memory plugin's `recall_command`; without it the plugin runs `recall-mcp` from PATH. */
  recallCommand?: string;
};

export function install(opts: InstallOptions): { configDir: string; written: string[] } {
  const claude = join(opts.home, ".claude");
  const configDir = join(opts.xdgConfigHome || join(opts.home, ".config"), "opencode");
  // Every name OpenCode 2.2.7 loads from its global config folder: a config under any of them would be
  // merged with or shadowed by the installed one without a word (audit J DEPLOY-001).
  const clash = ["opencode.json", "opencode.jsonc", "command", "commands", "agent", "agents", "mode", "modes"]
    .map((n) => join(configDir, n))
    .filter((p) => existsSync(p));
  if (clash.length > 0) throw new Error(`install: refusing to overwrite ${clash.join(", ")}`);
  // Both sources are checked before anything is written: a missing one would fail midway and leave a
  // partial config that every later run refuses (audit J DEPLOY-002).
  const missing = ["commands", "agents"].map((n) => join(claude, n)).filter((p) => !existsSync(p));
  if (missing.length > 0) throw new Error(`install: ${missing.join(", ")} does not exist; nothing was written`);
  const plugins = [
    { package: join(ROOT, "src"), options: opts.recallCommand ? { recall_command: opts.recallCommand } : {} },
  ];
  const written = [
    ...portCommands(join(claude, "commands"), join(configDir, "commands")).written,
    ...portAgents(join(claude, "agents"), join(configDir, "agents")).written,
  ];
  writeConfig(configDir, { instructions: INSTRUCTIONS, plugins, mcp: {} });
  written.push(join(configDir, "opencode.json"));
  return { configDir, written };
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  const at = args.indexOf("--recall-command");
  const recallCommand = at >= 0 ? args[at + 1] : undefined;
  if (!args.includes("--real-home") || (at >= 0 && !recallCommand)) {
    console.error("usage: bun port/install.ts --real-home [--recall-command <path>]   (writes into the real ~/.config/opencode)");
    process.exit(2);
  }
  if (recallCommand && !existsSync(recallCommand)) {
    console.error(`install: --recall-command ${recallCommand} does not exist; nothing was written`);
    process.exit(2);
  }
  const result = install({ home: homedir(), xdgConfigHome: process.env.XDG_CONFIG_HOME, recallCommand });
  console.log(`installed into ${result.configDir}:\n${result.written.join("\n")}`);
}
