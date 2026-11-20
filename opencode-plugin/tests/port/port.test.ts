/** The port scripts. Every behaviour test here has its red proof in `tests/mutations.port.ts` (rows
 * `PORT-*`). The sources are temp copies of the fixture Claude home in `tests/fixtures/claude`; nothing
 * here writes to a source, and the OpenCode runs use an isolated home outside the real one. */

import { afterAll, describe, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import { cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, parse, relative } from "node:path";
import { portAgents } from "../../port/agents";
import { mcpName, OPENCODE_TOOL, portCommands } from "../../port/commands";
import { writeConfig } from "../../port/config";
import { auditSkills } from "../../port/skills_audit";
import { isolatedHome } from "../isolated_home";

// An OpenCode start takes 7 to 10 s here.
setDefaultTimeout(120_000);

const CLAUDE = join(import.meta.dir, "..", "fixtures", "claude");
const TMP = tmpdir();
const temps: string[] = [];

function temp(prefix: string, parent = TMP): string {
  mkdirSync(parent, { recursive: true });
  const dir = mkdtempSync(join(parent, prefix));
  temps.push(dir);
  return dir;
}

afterAll(() => {
  for (const dir of temps) rmSync(dir, { recursive: true, force: true, maxRetries: 5 });
});

/** A temp copy of a fixture source directory. */
function copyOf(real: string): string {
  const dir = join(temp("port-src-"), parse(real).base);
  cpSync(real, dir, { recursive: true });
  return dir;
}

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

function snapshot(root: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const f of files(root)) {
    const p = join(root, f);
    out[f] = `${createHash("sha256").update(readFileSync(p)).digest("hex")} ${statSync(p).mtimeMs}`;
  }
  return out;
}

/** The number of mapped substitutions that turn `source` into `port`, or the offset of the first
 * difference that is not one: a Claude tool name in the bridge's table replaced by its OpenCode
 * name, or an `mcp__<server>__<tool>` name replaced by `<server>_<tool>`. */
function mappedOnly(source: string, port: string): { substitutions: number } | { mismatchAt: number } {
  const tokens = source.match(/mcp__[\w-]+?__\w+|\w+|[^\w]/g) ?? [];
  let at = 0;
  let substitutions = 0;
  for (const token of tokens) {
    if (port.startsWith(token, at)) {
      at += token.length;
      continue;
    }
    const mapped = token.startsWith("mcp__") ? mcpName(token) : OPENCODE_TOOL[token];
    if (mapped === undefined || !port.startsWith(mapped, at)) return { mismatchAt: at };
    at += mapped.length;
    substitutions++;
  }
  return at === port.length ? { substitutions } : { mismatchAt: at };
}

/** OpenCode's own rule evaluation (`evaluate3` in `@opencode/core`): the last matching rule wins. */
function effect(rules: { action: string; resource: string; effect: string }[], action: string): string {
  const hit = [...rules].reverse().find((r) => (r.action === "*" || r.action === action) && r.resource === "*");
  return hit?.effect ?? "ask";
}

/** `opencode <args> --standalone` in an isolated home, parsed as JSON. One retry: 1 of 9 measured starts
 * listed no configured agent at all (2026-11-11). */
async function opencodeJson(args: string[], ready: (out: any) => boolean, configure: (configDir: string) => void) {
  const sb = isolatedHome("port-");
  configure(join(sb.env.XDG_CONFIG_HOME!, "opencode"));
  let out: any;
  try {
    const opencode = Bun.which("opencode", { PATH: sb.env.PATH });
    // The CLI is not a dependency of this package: `npm install -g @opencode/cli@2.2.7`.
    expect({ opencode: opencode !== null }).toEqual({ opencode: true });
    for (let attempt = 0; attempt < 2; attempt++) {
      const p = Bun.spawnSync([opencode!, ...args, "--standalone"],
        { env: sb.env, cwd: sb.home, stdin: "ignore", stdout: "pipe", stderr: "pipe", timeout: 90_000 });
      expect(p.exitCode).toBe(0);
      out = JSON.parse(p.stdout.toString());
      if (ready(out)) break;
    }
  } finally {
    sb.dispose();
  }
  return out;
}

const COMMAND = `---
description: "Fixture command"
---

Write the plan, then Read this twice.
Launch the auditors in a SINGLE message, as parallel Agent calls.
#### Agent 7: High-Stakes
Register the \`Agent\` matcher in settings.json.
Apply each fix via Edit, and run checks with the Bash tool.
Call \`Bash\` or \`mcp__recall-memory__recall_search\` directly.
`;

describe("commands", () => {
  test("each ~/.claude/commands file is written with its tool names mapped and nothing else changed", () => {
    const src = copyOf(join(CLAUDE, "commands"));
    const dest = temp("port-dest-");
    const report = portCommands(src, dest);
    const sources = readdirSync(src).filter((f) => f.endsWith(".md")).sort();
    expect(sources.length).toBeGreaterThan(3);
    expect(report.written.map((p) => parse(p).base).sort()).toEqual(sources);
    let total = 0;
    for (const name of sources) {
      const result = mappedOnly(readFileSync(join(src, name), "utf-8"), readFileSync(join(dest, name), "utf-8"));
      expect({ name, ...result }).toEqual({ name, substitutions: expect.any(Number) });
      const n = (result as { substitutions: number }).substitutions;
      expect(report.changed.filter((c) => parse(c.file).base === name).length).toBe(n);
      total += n;
    }
    expect(total).toBeGreaterThan(0);
  });

  test("the ported commands load in OpenCode with their descriptions", async () => {
    const src = copyOf(join(CLAUDE, "commands"));
    const sb = isolatedHome("port-");
    const config = join(sb.env.XDG_CONFIG_HOME!, "opencode");
    portCommands(src, join(config, "commands"));
    const work = join(sb.home, "work");
    mkdirSync(work);
    const expected = readdirSync(src).filter((f) => f.endsWith(".md")).sort().map((f) => ({
      name: f.replace(/\.md$/, ""),
      description: JSON.parse(/^description:\s*(".*")\s*$/m.exec(readFileSync(join(src, f), "utf-8"))![1]!),
    }));
    const p = Bun.spawnSync(["bun", join(import.meta.dir, "list_commands.ts"), config, work, join(sb.home, "opencode.db"),
      expected.map((e) => e.name).join(",")], { env: sb.env, cwd: work, stdin: "ignore", stdout: "pipe", stderr: "pipe", timeout: 90_000 });
    sb.dispose();
    expect(p.exitCode).toBe(0);
    const listed: { name: string; description?: string }[] = JSON.parse(p.stdout.toString().trim().split("\n").at(-1)!);
    expect(listed.filter((c) => expected.some((e) => e.name === c.name)).map((c) => ({ name: c.name, description: c.description }))
      .sort((a, b) => a.name.localeCompare(b.name))).toEqual(expected.sort((a, b) => a.name.localeCompare(b.name)));
  });

  test("a tool name in prose or in a hook matcher is left alone", () => {
    const src = temp("port-src-");
    writeFileSync(join(src, "fixture.md"), COMMAND);
    const dest = temp("port-dest-");
    const report = portCommands(src, dest);
    expect(readFileSync(join(dest, "fixture.md"), "utf-8")).toBe(COMMAND
      .replace("parallel Agent calls", "parallel subagent calls")
      .replace("via Edit", "via edit")
      .replace("the Bash tool", "the shell tool")
      .replace("`Bash`", "`shell`")
      .replace("`mcp__recall-memory__recall_search`", "`recall-memory_recall_search`"));
    expect(report.changed.map((c) => `${c.from} => ${c.to}`).sort()).toEqual([
      "Agent calls => subagent calls",
      "Bash tool => shell tool",
      "`Bash` => `shell`",
      "mcp__recall-memory__recall_search => recall-memory_recall_search",
      "via Edit => via edit",
    ]);
  });

  test("a step that needs a Claude-only feature is listed as unportable, not dropped", () => {
    const src = temp("port-src-");
    const text = COMMAND + "subagent_type: fp-check\nTrack the steps with `TodoWrite`.\nWith no CLAUDE_PID the script only reports.\n";
    writeFileSync(join(src, "fixture.md"), text);
    const dest = temp("port-dest-");
    const report = portCommands(src, dest);
    expect(report.unportable.map((u) => u.reason.split(":")[0]).sort()).toEqual([
      "claude-env (line 13)",
      "claude-only-tool (line 12)",
      "parallel-subagents (line 6)",
      "subagent_type (line 11)",
    ]);
    const port = readFileSync(join(dest, "fixture.md"), "utf-8");
    for (const line of ["subagent_type: fp-check", "Track the steps with `TodoWrite`.", "With no CLAUDE_PID the script only reports.",
      "Launch the auditors in a SINGLE message, as parallel subagent calls."]) expect(port).toContain(line);
  });
});

const AGENT = `---
name: probe-auditor
description: Probe auditor. Finds probes.
tools: Read, Grep
model: inherit
---

You audit with the Read tool and the Bash tool.
`;

describe("agents", () => {
  /** Invariant: an agent whose `name:` is not a plain file name (a relative path, quotes) is reported as
   * unportable and written nowhere, so it cannot land outside `dest`. Red: PORT-AGENT-SAFE-NAME (the check
   * removed; the agent is written two folders up), audit J SEC-003, 2026-11-13. */
  test("an agent whose name is not a plain file name is reported and written nowhere", () => {
    const src = temp("port-src-");
    writeFileSync(join(src, "escape.md"), AGENT.replace("name: probe-auditor", "name: ../../escaped"));
    const dest = join(temp("port-dest-"), "a", "agents");
    const report = portAgents(src, dest);
    expect({ written: report.written, outside: existsSync(join(dest, "..", "..", "escaped.md")) }).toEqual({ written: [], outside: false });
    expect(report.unportable.map((u) => u.reason)).toContain('name "../../escaped" is not a plain file name, not written');
  });

  test("an agent with tools: Read, Grep becomes permission rules that deny every other tool", () => {
    const src = temp("port-src-");
    writeFileSync(join(src, "cca-probe-auditor.md"), AGENT);
    const dest = temp("port-dest-");
    const report = portAgents(src, dest);
    expect(report.written).toEqual([join(dest, "probe-auditor.md")]);
    expect(readFileSync(join(dest, "probe-auditor.md"), "utf-8")).toBe(`---
description: Probe auditor. Finds probes.
mode: subagent
permissions:
  - action: "*"
    resource: "*"
    effect: deny
  - action: read
    resource: "*"
    effect: allow
  - action: grep
    resource: "*"
    effect: allow
---

You audit with the Read tool and the shell tool.
`);
    expect(report.unportable).toEqual([]);
  });

  test("a Claude-only step in an agent body is listed as unportable and kept", () => {
    const src = temp("port-src-");
    writeFileSync(join(src, "cca-probe-auditor.md"), AGENT + "Fan out as parallel Agent calls.\n");
    const dest = temp("port-dest-");
    const report = portAgents(src, dest);
    expect(report.unportable.map((u) => u.reason.split(":")[0])).toEqual(["parallel-subagents (line 9)"]);
    expect(readFileSync(join(dest, "probe-auditor.md"), "utf-8")).toEndWith("Fan out as parallel subagent calls.\n");
  });

  test("the ported CCA agents load in OpenCode as subagents that can use only their listed tools", async () => {
    const src = copyOf(join(CLAUDE, "agents"));
    const names = readdirSync(src).filter((f) => f.endsWith(".md")).map((f) => {
      const m = /^name:\s*(\S+)/m.exec(readFileSync(join(src, f), "utf-8"));
      return m![1]!;
    });
    expect(names.length).toBeGreaterThan(1);
    const agents: any[] = await opencodeJson(["debug", "agents"], (a) => names.every((n) => a.some((x: any) => x.id === n)),
      (config) => portAgents(src, join(config, "agents")));
    for (const name of names) {
      const agent = agents.find((a) => a.id === name);
      expect({ name, mode: agent?.mode }).toEqual({ name, mode: "subagent" });
      const got: Record<string, string> = Object.fromEntries(["read", "grep", "glob", "shell", "edit", "websearch", "webfetch", "subagent", "skill", "recall_search"]
        .map((action) => [action, effect(agent.permissions, action)]));
      const row: Record<string, string> = { name, ...got };
      expect(row).toEqual({ name, read: "allow", grep: "allow", glob: "allow", shell: "allow", edit: "deny",
        websearch: "deny", webfetch: "deny", subagent: "deny", skill: "deny", recall_search: "deny" });
    }
  });
});

describe("skills and config", () => {
  test("auditSkills lists each skill that names a Claude-only tool", () => {
    const a = temp("port-skills-");
    const b = temp("port-skills-");
    mkdirSync(join(a, "alpha"));
    writeFileSync(join(a, "alpha", "SKILL.md"), "Call `mcp__recall-memory__recall_search` first.\nThen Read the file.\n");
    mkdirSync(join(a, "beta", "references"), { recursive: true });
    writeFileSync(join(a, "beta", "SKILL.md"), "---\nname: beta\n---\nUse the Agent tool.\n");
    writeFileSync(join(a, "beta", "references", "more.md"), "Then call `TodoWrite`.\nTask 3 is prose; Task(subagent_type=x) is a call.\n");
    mkdirSync(join(b, "gamma"));
    writeFileSync(join(b, "gamma", "SKILL.md"), "Read the docs and Write a summary, then Edit it.\n");
    const findings = auditSkills([a, b]);
    expect(findings.map((f) => `${f.skill} ${relative(join(f.file, ".."), f.file)}:${f.line} ${f.tool} -> ${f.opencode}`).sort()).toEqual([
      "alpha SKILL.md:1 mcp__recall-memory__recall_search -> recall-memory_recall_search",
      "beta SKILL.md:4 Agent -> subagent",
      "beta more.md:1 TodoWrite -> null",
      "beta more.md:2 Task -> null",
    ]);
  });

  test("writeConfig names the three CLAUDE.md files in instructions", async () => {
    const instructions = ["~/.claude/CLAUDE.md", "CLAUDE.md", "CLAUDE.local.md"];
    const plugins = [{ package: "C:/plugins/re-call", options: { k: 3 } }];
    const mcp = { servers: {} };
    const dest = temp("port-config-");
    writeConfig(dest, { instructions, plugins, mcp });
    expect(JSON.parse(readFileSync(join(dest, "opencode.json"), "utf-8"))).toEqual({ instructions, plugins, mcp });
    // OpenCode reads it as a config document with those instructions.
    const sources: any[] = await opencodeJson(["debug", "config"], (s) => s.some((x: any) => x.type === "document"),
      (config) => writeConfig(config, { instructions, plugins: [], mcp }));
    const doc = sources.find((s) => s.type === "document");
    expect(doc?.info?.instructions).toEqual(instructions);
  });
});

test("the source dirs are opened read-only (spy the fs)", () => {
  const commands = copyOf(join(CLAUDE, "commands"));
  const agents = copyOf(join(CLAUDE, "agents"));
  const skills = copyOf(join(CLAUDE, "skills"));
  const before = [commands, agents, skills].map(snapshot);
  const writers = ["writeFileSync", "mkdirSync", "rmSync", "cpSync", "copyFileSync", "renameSync", "appendFileSync", "unlinkSync", "rmdirSync", "utimesSync"] as const;
  const spies = writers.map((name) => spyOn(fs, name));
  const dest = temp("port-dest-");
  let calls: string[][] = [];
  try {
    portCommands(commands, join(dest, "commands"));
    portAgents(agents, join(dest, "agents"));
    auditSkills([skills]);
    calls = spies.map((s) => s.mock.calls.map((c) => String(c[0])));
  } finally {
    for (const s of spies) s.mockRestore();
  }
  const touched = calls.flatMap((paths, i) => paths.map((p) => `${writers[i]} ${p}`))
    .filter((call) => [commands, agents, skills].some((src) => !relative(src, call.split(" ").slice(1).join(" ")).startsWith("..")));
  expect(touched).toEqual([]);
  // The spies saw the ports write, into dest: a check that cannot pass by observing nothing.
  expect(calls[0]!.filter((p) => !relative(dest, p).startsWith("..")).length).toBeGreaterThan(3);
  expect([commands, agents, skills].map(snapshot)).toEqual(before);
});
