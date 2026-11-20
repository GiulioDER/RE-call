/** Port Claude Code slash commands to OpenCode commands (Task 4 of
 * `docs/superpowers/plans/2026-11-11-coding-round.md`). Also home to the text rules the agent port
 * and the skill audit share.
 *
 * A Claude tool name is mapped only where the text uses it as a tool: `mcp__<server>__<tool>`
 * anywhere, and the bridge's four names (`Bash`, `Write`, `Edit`, `Agent`) in backticks, before
 * "tool" or "call", after "via", or before "(". Elsewhere they are English ("Write the plan", "Agent
 * 7"), and a backticked name before "matcher" names a Claude hook matcher, which the bridge keeps. */

import * as fs from "node:fs";
import { join } from "node:path";

export type PortReport = {
  written: string[];
  changed: { file: string; from: string; to: string }[];
  unportable: { file: string; reason: string }[];
};

/** OpenCode's name for each Claude tool the bridge maps: `toClaudeTool` in reverse. */
export const OPENCODE_TOOL: Record<string, string> = { Bash: "shell", Write: "write", Edit: "edit", Agent: "subagent" };

/** Claude Code tools OpenCode 2.2.7 has under no name (its default tools, `docs/V2-FACTS.md`).
 * `Task` is Claude Code's former name for the Agent tool. */
export const CLAUDE_ONLY = ["Task", "MultiEdit", "NotebookEdit", "TodoWrite", "ExitPlanMode", "EnterPlanMode", "AskUserQuestion", "SendMessage",
  "ScheduleWakeup", "CronCreate", "Monitor", "EnterWorktree", "ExitWorktree", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet"];

/** `mcp__<server>__<tool>` becomes `<server>_<tool>`, as OpenCode names an MCP tool. */
export const mcpName = (claude: string) => claude.replace(/^mcp__(.+?)__/, "$1_");

const MCP = /\bmcp__[\w-]+?__\w+/g;

/** Each regex captures the tool name as `name`, with the context around it. */
function contexts(names: string[]): RegExp[] {
  const n = `(?<name>${names.join("|")})`;
  return [
    new RegExp(`\`${n}\`(?! matcher)`, "g"),
    new RegExp(`\\b${n} (?:tools?|calls?)\\b`, "g"),
    new RegExp(`\\bvia ${n}\\b`, "g"),
    new RegExp(`\\b${n}\\(`, "g"),
  ];
}

const MAPPED = contexts(Object.keys(OPENCODE_TOOL));

/** Every use of a Claude tool name as a tool in `text`: the name, its line and its context. */
export function toolMentions(text: string, names: string[]): { name: string; line: number; context: string }[] {
  const found: { name: string; line: number; context: string; at: number }[] = [];
  const lineOf = (at: number) => text.slice(0, at).split("\n").length;
  for (const m of text.matchAll(MCP)) found.push({ name: m[0], line: lineOf(m.index!), context: m[0], at: m.index! });
  for (const re of contexts(names)) {
    for (const m of text.matchAll(re)) found.push({ name: m.groups!.name!, line: lineOf(m.index!), context: m[0], at: m.index! });
  }
  return found.sort((a, b) => a.at - b.at).map(({ at: _at, ...rest }) => rest);
}

/** `text` with its tool names mapped, and each substitution as it appears in context. */
export function mapText(text: string): { text: string; changed: { from: string; to: string }[] } {
  const changed: { from: string; to: string }[] = [];
  let out = text.replace(MCP, (m) => {
    changed.push({ from: m, to: mcpName(m) });
    return mcpName(m);
  });
  for (const re of MAPPED) {
    out = out.replace(re, (...args) => {
      const m = args[0] as string;
      const name = (args.at(-1) as { name: string }).name;
      const to = m.replace(name, OPENCODE_TOOL[name]!);
      changed.push({ from: m, to });
      return to;
    });
  }
  return { text: out, changed };
}

const has = (re: RegExp) => (line: string) => new RegExp(re.source, re.flags.replace("g", "")).test(line);

/** A step that needs a Claude-only feature: an id, the test that finds it, and why it does not port. */
const FEATURES: [string, (line: string) => boolean, string][] = [
  ["parallel-subagents", has(/\bparallel Agent calls\b|\bin a SINGLE message\b/i),
    "launches several subagents in one message to run at once, as Claude Code runs parallel Agent calls; OpenCode 2.2.7's subagent tool runs a call to completion unless given background: true, and several calls in one turn are not measured"],
  ["subagent_type", has(/\bsubagent_type\b/),
    "names the Claude Agent tool's subagent_type parameter; OpenCode's subagent tool takes agent"],
  ["claude-env", has(/\bCLAUDE_PID\b|\bCLAUDE_CODE_[A-Z_]+\b/),
    "relies on an environment variable only Claude Code sets"],
  ["claude-only-tool", (line) => contexts(CLAUDE_ONLY).some((re) => has(re)(line)),
    "names a Claude Code tool OpenCode 2.2.7 does not have"],
];

/** The Claude-only steps in `text`, one entry per feature with the lines that need it. */
export function unportableSteps(text: string): string[] {
  const lines = text.split("\n");
  const out: string[] = [];
  for (const [id, found, why] of FEATURES) {
    const hits = lines.flatMap((l, i) => (found(l) ? [i + 1] : []));
    if (hits.length) out.push(`${id} (line${hits.length > 1 ? "s" : ""} ${hits.join(", ")}): ${why}`);
  }
  return out;
}

export function portCommands(src: string, dest: string): PortReport {
  const report: PortReport = { written: [], changed: [], unportable: [] };
  fs.mkdirSync(dest, { recursive: true });
  for (const name of fs.readdirSync(src).filter((f) => f.endsWith(".md")).sort()) {
    const file = join(src, name);
    const source = fs.readFileSync(file, "utf-8");
    const { text, changed } = mapText(source);
    const out = join(dest, name);
    fs.writeFileSync(out, text, "utf-8");
    report.written.push(out);
    report.changed.push(...changed.map((c) => ({ file, ...c })));
    report.unportable.push(...unportableSteps(source).map((reason) => ({ file, reason })));
  }
  return report;
}
