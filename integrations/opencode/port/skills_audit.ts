/** List each skill that names a Claude tool OpenCode does not have under that name. It edits no
 * skill: OpenCode already loads `~/.claude/skills` and `~/.agents/skills`, and a skill that needs a
 * change gets an OpenCode copy, by hand, never an edit of the Claude original.
 *
 * A finding is a tool used as a tool (the contexts in `commands.ts`): an `mcp__` name, one of the
 * bridge's four names, or a Claude-only tool. Names that differ only in case (`Read` and `read`,
 * `Skill` and `skill`) are not findings. */

import * as fs from "node:fs";
import { join } from "node:path";
import { CLAUDE_ONLY, mcpName, OPENCODE_TOOL, toolMentions } from "./commands";

export type SkillFinding = { skill: string; file: string; line: number; tool: string; opencode: string | null };

export function auditSkills(dirs: string[]): SkillFinding[] {
  const findings: SkillFinding[] = [];
  const names = [...Object.keys(OPENCODE_TOOL), ...CLAUDE_ONLY];
  for (const dir of dirs.filter((d) => fs.existsSync(d))) {
    for (const skill of fs.readdirSync(dir).sort()) {
      const root = join(dir, skill);
      if (!fs.statSync(root).isDirectory()) continue;
      const docs = (fs.readdirSync(root, { recursive: true }) as string[]).filter((f) => f.endsWith(".md")).sort();
      for (const doc of docs) {
        const file = join(root, doc);
        const seen = new Set<string>();
        for (const m of toolMentions(fs.readFileSync(file, "utf-8"), names)) {
          if (seen.has(`${m.line} ${m.name}`)) continue;
          seen.add(`${m.line} ${m.name}`);
          const opencode = m.name.startsWith("mcp__") ? mcpName(m.name) : OPENCODE_TOOL[m.name] ?? null;
          findings.push({ skill, file, line: m.line, tool: m.name, opencode });
        }
      }
    }
  }
  return findings;
}
