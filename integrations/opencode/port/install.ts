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

import {
  constants, copyFileSync, existsSync, lstatSync, mkdirSync, mkdtempSync, rmdirSync, rmSync, unlinkSync,
} from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { configHome } from "../src/config";
import { portAgents } from "./agents";
import { portCommands, type PortReport } from "./commands";
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

function refuseClashes(configDir: string): void {
  // Every name OpenCode 2.2.7 loads from its global config folder: a config under any of them would be
  // merged with or shadowed by the installed one without a word (audit J DEPLOY-001).
  const clash = ["opencode.json", "opencode.jsonc", "command", "commands", "agent", "agents", "mode", "modes"]
    .map((n) => join(configDir, n))
    .filter((p) => lstatSync(p, { throwIfNoEntry: false }) !== undefined);
  if (clash.length > 0) throw new Error(`install: refusing to overwrite ${clash.join(", ")}`);
}

export function install(opts: InstallOptions): Pick<PortReport, "written" | "unportable"> & { configDir: string } {
  const claude = join(opts.home, ".claude");
  const configDir = join(configHome(opts.xdgConfigHome, opts.home), "opencode");
  refuseClashes(configDir);
  // Both sources are checked before anything is written: a missing one would fail midway and leave a
  // partial config that every later run refuses (audit J DEPLOY-002).
  const missing = ["commands", "agents"].map((n) => join(claude, n)).filter((p) => !existsSync(p));
  if (missing.length > 0) throw new Error(`install: ${missing.join(", ")} does not exist; nothing was written`);
  const plugins = [
    { package: join(ROOT, "src"), options: opts.recallCommand ? { recall_command: opts.recallCommand } : {} },
  ];
  const stage = mkdtempSync(join(tmpdir(), "re-call-opencode-port-"));
  try {
    const commands = portCommands(join(claude, "commands"), join(stage, "commands"));
    const agents = portAgents(join(claude, "agents"), join(stage, "agents"));
    writeConfig(stage, { instructions: INSTRUCTIONS, plugins, mcp: {} });
    const staged = [...new Set([...commands.written, ...agents.written, join(stage, "opencode.json")])];
    const written = publish(stage, configDir, staged);
    return { configDir, written, unportable: [...commands.unportable, ...agents.unportable] };
  } finally {
    try {
      rmSync(stage, { recursive: true, force: true, maxRetries: 3 });
    } catch (error) {
      console.warn(`install: could not remove staging directory ${stage}`, error);
    }
  }
}

function publish(stage: string, configDir: string, staged: string[]): string[] {
  refuseClashes(configDir);
  const rollback: (() => void)[] = [];
  try {
    mkdirSync(dirname(configDir), { recursive: true });
    const mkdir = (path: string) => {
      mkdirSync(path);
      const created = lstatSync(path, { bigint: true });
      rollback.push(() => {
        const current = lstatSync(path, { bigint: true, throwIfNoEntry: false });
        if (current?.dev === created.dev && current.ino === created.ino) rmdirSync(path);
      });
    };
    if (!existsSync(configDir)) {
      mkdir(configDir);
    }
    // Both porters produce these directories even when they find nothing to write.
    for (const folder of ["commands", "agents"]) mkdir(join(configDir, folder));
    const written: string[] = [];
    for (const source of staged) {
      const target = join(configDir, relative(stage, source));
      copyFileSync(source, target, constants.COPYFILE_EXCL);
      const created = lstatSync(target, { bigint: true });
      rollback.push(() => {
        const current = lstatSync(target, { bigint: true, throwIfNoEntry: false });
        if (current?.dev === created.dev && current.ino === created.ino && current.ctimeNs === created.ctimeNs) {
          unlinkSync(target);
        }
      });
      written.push(target);
    }
    return written;
  } catch (error) {
    for (const undo of rollback.reverse()) {
      try {
        undo();
      } catch (cleanupError) {
        // rmdir is deliberately nonrecursive so files introduced by a user are kept.
        console.warn("install: incomplete rollback", cleanupError);
      }
    }
    throw error;
  }
}

if (import.meta.main) {
  const args = process.argv.slice(2);
  const at = args.indexOf("--recall-command");
  // Absolute, because the plugin takes a `recall_command` path only when it is absolute.
  const recallCommand = at >= 0 && args[at + 1] ? resolve(args[at + 1]!) : undefined;
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
  for (const { file, reason } of result.unportable) console.warn(`${file}: ${reason}`);
}
