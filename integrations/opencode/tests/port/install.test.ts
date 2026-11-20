/** `port/install.ts`, run against isolated homes only, never the real one. Red proof rows:
 * `tests/mutations.integration.ts`, ids INSTALL-*. The isolated home gets copies of the fixture
 * `commands` and `agents` in `tests/fixtures/claude`, which is what the install reads. */

import { afterAll, describe, expect, setDefaultTimeout, test } from "bun:test";
import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { INSTRUCTIONS, install } from "../../port/install";
import { type IsolatedHome, isolatedHome } from "../isolated_home";
import { realHome } from "../real_home";

setDefaultTimeout(120_000);

const ROOT = resolve(import.meta.dir, "..", "..");
const FIXTURE = join(import.meta.dir, "..", "fixtures", "claude");
// The real home's, named outright: the test preload points XDG_CONFIG_HOME at its own scratch folder.
const REAL_CONFIG = join(realHome(), ".config", "opencode");
const homes: IsolatedHome[] = [];
afterAll(() => {
  for (const h of homes) h.dispose();
});

function files(root: string): string[] {
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else out.push(relative(root, p).replaceAll("\\", "/"));
    }
  };
  if (existsSync(root)) walk(root);
  return out.sort();
}

function snapshot(root: string, skip?: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const f of files(root)) {
    const p = join(root, f);
    if (skip && !relative(skip, p).startsWith("..")) continue;
    out[f] = `${createHash("sha256").update(readFileSync(p)).digest("hex")} ${statSync(p).mtimeMs}`;
  }
  return out;
}

function sandbox(): { sb: IsolatedHome; xdg: string } {
  const sb = isolatedHome("install-");
  homes.push(sb);
  for (const d of ["commands", "agents"]) cpSync(join(FIXTURE, d), join(sb.home, ".claude", d), { recursive: true });
  // Somewhere other than <home>/.config, so a script that ignored XDG_CONFIG_HOME would write elsewhere.
  const xdg = join(sb.home, "..", "xdg");
  return { sb, xdg };
}

describe("install", () => {
  /** Invariant: the install writes exactly the ported commands, the ported agents and an `opencode.json`
   * into `<XDG_CONFIG_HOME>/opencode`, nothing anywhere else in the home and nothing in the real config;
   * the config names the three CLAUDE.md layers and the memory plugin with its command. Red: INSTALL-XDG,
   * INSTALL-INSTR, INSTALL-AGENTS, INSTALL-RECALL. */
  test("install writes the ported commands, agents and config into the given config folder only", async () => {
    const { sb, xdg } = sandbox();
    const realBefore = snapshot(REAL_CONFIG);
    const homeBefore = snapshot(sb.home);
    const { configDir, written } = install({ home: sb.home, xdgConfigHome: xdg, recallCommand: "C:/x/recall-mcp.exe" });

    expect(configDir).toBe(join(xdg, "opencode"));
    const commands = readdirSync(join(FIXTURE, "commands")).filter((n) => n.endsWith(".md")).map((n) => `commands/${n}`);
    const agents = readdirSync(join(FIXTURE, "agents")).filter((n) => n.endsWith(".md")).length;
    const got = files(configDir);
    expect({
      commands: got.filter((f) => f.startsWith("commands/")),
      agents: got.filter((f) => f.startsWith("agents/")).length,
      rest: got.filter((f) => !/^(commands|agents)\//.test(f)),
      written: written.map((w) => relative(configDir, w).replaceAll("\\", "/")).sort(),
    }).toEqual({ commands: commands.sort(), agents, rest: ["opencode.json"], written: got });

    expect(JSON.parse(readFileSync(join(configDir, "opencode.json"), "utf8"))).toEqual({
      instructions: ["~/.claude/CLAUDE.md", "CLAUDE.md", "CLAUDE.local.md"],
      plugins: [
        { package: join(ROOT, "src"), options: { recall_command: "C:/x/recall-mcp.exe" } },
      ],
      mcp: {},
    });
    expect(INSTRUCTIONS).toHaveLength(3);
    expect(snapshot(sb.home)).toEqual(homeBefore);
    expect(snapshot(REAL_CONFIG)).toEqual(realBefore);
  });

  /** Invariant: an existing config under any name OpenCode loads from its global folder (`opencode.json`,
   * `opencode.jsonc`, `command(s)/`, `agent(s)/`, `mode(s)/`) stops the install before it writes anything,
   * so a second run cannot overwrite or silently merge with the owner's config. Red: INSTALL-CLASH, and
   * INSTALL-CLASH-NAMES (the list back to its first three names; `opencode.jsonc` is not refused), audit J
   * DEPLOY-001, 2026-11-13. */
  test("install refuses to overwrite an existing config and writes nothing", async () => {
    // One home for every name (install reads only its home), a fresh config folder per name.
    const { sb, xdg: base } = sandbox();
    for (const [i, existing] of ["opencode.json", "opencode.jsonc", "command", "commands", "agent", "agents", "mode", "modes"].entries()) {
      const xdg = `${base}-${i}`;
      const configDir = join(xdg, "opencode");
      mkdirSync(configDir, { recursive: true });
      if (existing.startsWith("opencode.")) await Bun.write(join(configDir, existing), '{"owner": true}\n');
      else mkdirSync(join(configDir, existing, "mine"), { recursive: true });
      const before = snapshot(xdg);
      let error = "";
      try {
        install({ home: sb.home, xdgConfigHome: xdg });
      } catch (e) {
        error = String(e);
      }
      expect({ existing, refused: error.includes("refusing to overwrite"), after: snapshot(xdg) }).toEqual({ existing, refused: true, after: before });
    }
  });

  /** Invariant: a home missing `~/.claude/agents` (or `commands`) is refused before the install writes
   * anything, so no partial config is left for every later run to refuse. Red: INSTALL-SOURCES (the check
   * removed; the commands are written, then the missing agents folder throws), audit J DEPLOY-002, 2026-11-13. */
  test("install refuses a home without both source folders and writes nothing", async () => {
    const { sb, xdg } = sandbox();
    rmSync(join(sb.home, ".claude", "agents"), { recursive: true, force: true });
    let error = "";
    try {
      install({ home: sb.home, xdgConfigHome: xdg });
    } catch (e) {
      error = String(e);
    }
    expect({ written: files(xdg), refused: error.includes("does not exist; nothing was written") }).toEqual({ written: [], refused: true });
  });
});
