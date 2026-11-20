/** Port Claude Code subagents (`~/.claude/agents/*.md`) to OpenCode 2.2.7 agents.
 *
 * OpenCode names an agent by its file, so `name: bug-auditor` becomes `bug-auditor.md`, the name the
 * CCA commands pass as `subagent_type`. `tools:` becomes OpenCode's own idiom for a read-only
 * subagent (its built-in `explore`): deny every action, then allow the listed ones; the last matching
 * rule wins. `model: inherit` is dropped, since an OpenCode subagent without a model uses the
 * session's. The body keeps its text, with tool names mapped as in `commands.ts`. */

import * as fs from "node:fs";
import { join, parse } from "node:path";
import { mapText, type PortReport, unportableSteps } from "./commands";

/** OpenCode's permission action for each Claude tool a `tools:` line can name. Write and Edit share
 * `edit`, the action OpenCode's write tool asks for. */
export const PERMISSION: Record<string, string> = {
  Read: "read", Grep: "grep", Glob: "glob", Bash: "shell", Write: "edit", Edit: "edit", MultiEdit: "edit",
  Agent: "subagent", WebFetch: "webfetch", WebSearch: "websearch", Skill: "skill",
};

const rule = (action: string, effect: string) =>
  `  - action: ${action === "*" ? '"*"' : action}\n    resource: "*"\n    effect: ${effect}\n`;

export function portAgents(src: string, dest: string): PortReport {
  const report: PortReport = { written: [], changed: [], unportable: [] };
  fs.mkdirSync(dest, { recursive: true });
  for (const name of fs.readdirSync(src).filter((f) => f.endsWith(".md")).sort()) {
    const file = join(src, name);
    const source = fs.readFileSync(file, "utf-8");
    const front = /^---\r?\n([\s\S]*?)\r?\n---\r?\n/.exec(source);
    if (!front) {
      report.unportable.push({ file, reason: "no frontmatter: not an agent definition, not written" });
      continue;
    }
    let agent = parse(name).name;
    const head: string[] = [];
    let permissions = "";
    for (const line of front[1]!.split(/\r?\n/)) {
      const [, key, value = ""] = /^([\w-]+):\s*(.*)$/.exec(line) ?? [];
      if (key === "name") {
        agent = value;
        report.changed.push({ file, from: line, to: `the file name ${agent}.md` });
      } else if (key === "tools") {
        const tools = value.split(",").map((t) => t.trim()).filter(Boolean);
        const unknown = tools.filter((t) => !(t in PERMISSION));
        if (unknown.length) report.unportable.push({ file, reason: `tools ${unknown.join(", ")}: no OpenCode permission action, so denied` });
        const actions = [...new Set(tools.filter((t) => t in PERMISSION).map((t) => PERMISSION[t]!))];
        permissions = "permissions:\n" + rule("*", "deny") + actions.map((a) => rule(a, "allow")).join("");
        report.changed.push({ file, from: line, to: `permissions: deny every action, allow ${actions.join(", ")}` });
      } else if (key === "model" && value === "inherit") {
        report.changed.push({ file, from: line, to: "" });
      } else {
        head.push(line);
        if (key !== "description") report.unportable.push({ file, reason: `frontmatter "${line}" kept as written; OpenCode has no such agent field` });
      }
    }
    // The name becomes the file name, so one that is not a plain file name (a path, quotes) could write
    // outside `dest` or fail midway: it is reported, not written (audit J SEC-003).
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(agent)) {
      report.unportable.push({ file, reason: `name ${JSON.stringify(agent)} is not a plain file name, not written` });
      continue;
    }
    report.changed.push({ file, from: "", to: "mode: subagent" });
    const body = mapText(source.slice(front[0].length));
    report.changed.push(...body.changed.map((c) => ({ file, ...c })));
    // The source, not the mapped body: "parallel Agent calls" no longer reads so once mapped.
    report.unportable.push(...unportableSteps(source).map((reason) => ({ file, reason })));
    const out = join(dest, `${agent}.md`);
    fs.writeFileSync(out, `---\n${head.join("\n")}\nmode: subagent\n${permissions}---\n${body.text}`, "utf-8");
    report.written.push(out);
  }
  return report;
}
