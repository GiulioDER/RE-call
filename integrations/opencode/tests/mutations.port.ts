/** Red proof rows for the port scripts, in the shape of `ROWS` in
 * `tests/mutations.ts`, which replays them: apply the mutation (the `find` text must occur exactly
 * once), run the named test, require it to fail at an `expect`, restore, run it green. Each row names
 * its test by file and full name.
 */

import type { Row } from "./mutation_row";

const CMD = "port/commands.ts";
const AGT = "port/agents.ts";
const SKL = "port/skills_audit.ts";
const CFG = "port/config.ts";
const PORT_TEST = "tests/port/port.test.ts";
const CMD_REAL = "each ~/.claude/commands file is written with its tool names mapped and nothing else changed";
const PROSE = "a tool name in prose or in a hook matcher is left alone";
const CMD_LOADS = "the ported commands load in OpenCode with their descriptions";
const UNPORT = "a step that needs a Claude-only feature is listed as unportable, not dropped";
const AGENT = "an agent with tools: Read, Grep becomes permission rules that deny every other tool";
const AGENT_UNPORT = "a Claude-only step in an agent body is listed as unportable and kept";
const AGENT_LOADS = "the ported CCA agents load in OpenCode as subagents that can use only their listed tools";
const SKILLS = "auditSkills lists each skill that names a Claude-only tool";
const CONFIG = "writeConfig names the three CLAUDE.md files in instructions";
const READONLY = "the source dirs are opened read-only (spy the fs)";

const port = (id: string, file: string, find: string, replace: string, name: string, assertion: string): Row =>
  ({ id, file, find, replace, test: PORT_TEST, name, assertion });

export const ROWS: Row[] = [
  // -- Task 4: the port scripts ------------------------------------------------------------------
  port("PORT-CMD-MAPPED", CMD, "const to = m.replace(name, OPENCODE_TOOL[name]!);", "const to = m.replace(name, OPENCODE_TOOL[name]!.toUpperCase());", CMD_REAL, "expect({ name, ...result }).toEqual({ name, substitutions: expect.any(Number) });"),
  port("PORT-CMD-RECORD", CMD, "      changed.push({ from: m, to });\n", "", CMD_REAL, "expect(report.changed.filter((c) => parse(c.file).base === name).length).toBe(n);"),
  port("PORT-PROSE-MATCHER", CMD, "new RegExp(`\\`${n}\\`(?! matcher)`, \"g\"),", "new RegExp(`\\`${n}\\``, \"g\"),", PROSE, "expect(readFileSync(join(dest, \"fixture.md\"), \"utf-8\")).toBe(COMMAND"),
  port("PORT-PROSE-CONTEXT", CMD, "new RegExp(`\\\\b${n} (?:tools?|calls?)\\\\b`, \"g\"),", "new RegExp(`\\\\b${n}\\\\b`, \"g\"),", PROSE, "expect(readFileSync(join(dest, \"fixture.md\"), \"utf-8\")).toBe(COMMAND"),
  port("PORT-MCP", CMD, "    return mcpName(m);\n", "    return m;\n", PROSE, "expect(readFileSync(join(dest, \"fixture.md\"), \"utf-8\")).toBe(COMMAND"),
  port("PORT-UNPORT-RECORD", CMD, "    report.unportable.push(...unportableSteps(source).map((reason) => ({ file, reason })));\n", "", UNPORT, "expect(report.unportable.map((u) => u.reason.split(\":\")[0]).sort()).toEqual(["),
  port("PORT-UNPORT-ONLY", CMD, "(line) => contexts(CLAUDE_ONLY).some((re) => has(re)(line)),", "(line) => line === \"\\0\",", UNPORT, "expect(report.unportable.map((u) => u.reason.split(\":\")[0]).sort()).toEqual(["),
  port("PORT-UNPORT-ENV", CMD, "has(/\\bCLAUDE_PID\\b|\\bCLAUDE_CODE_[A-Z_]+\\b/)", "has(/\\bCLAUDE_CODE_[A-Z_]+\\b/)", UNPORT, "expect(report.unportable.map((u) => u.reason.split(\":\")[0]).sort()).toEqual(["),
  port("PORT-CMD-LOADS", CMD, "    const out = join(dest, name);\n", "    const out = join(dest, name + \".txt\");\n", CMD_LOADS, ".sort((a, b) => a.name.localeCompare(b.name))).toEqual(expected.sort((a, b) => a.name.localeCompare(b.name)));"),
  port("PORT-READONLY", CMD, "    const out = join(dest, name);\n", "    const out = join(src, name);\n", READONLY, "expect(touched).toEqual([]);"),
  port("PORT-AGENT-DENY", AGT, 'rule("*", "deny") + ', "", AGENT, "expect(readFileSync(join(dest, \"probe-auditor.md\"), \"utf-8\")).toBe(`---"),
  port("PORT-AGENT-MODE", AGT, "\\nmode: subagent\\n${permissions}---", "\\n${permissions}---", AGENT, "expect(readFileSync(join(dest, \"probe-auditor.md\"), \"utf-8\")).toBe(`---"),
  port("PORT-AGENT-NAME", AGT, "const out = join(dest, `${agent}.md`);", "const out = join(dest, name);", AGENT, "expect(report.written).toEqual([join(dest, \"probe-auditor.md\")]);"),
  port("PORT-AGENT-UNPORT", AGT, "unportableSteps(source).map(", "unportableSteps(body.text).map(", AGENT_UNPORT, "expect(report.unportable.map((u) => u.reason.split(\":\")[0])).toEqual([\"parallel-subagents (line 9)\"]);"),
  port("PORT-AGENT-SHELL", AGT, 'Bash: "shell",', 'Bash: "bash",', AGENT_LOADS, "expect(row).toEqual({ name, read: \"allow\", grep: \"allow\", glob: \"allow\", shell: \"allow\", edit: \"deny\","),
  port("PORT-SKILL-MCP", SKL, 'm.name.startsWith("mcp__") ? mcpName(m.name) : ', "", SKILLS, "expect(findings.map((f) => `${f.skill} ${relative(join(f.file, \"..\"), f.file)}:${f.line} ${f.tool} -> ${f.opencode}`).sort()).toEqual(["),
  port("PORT-SKILL-ONLY", SKL, "const names = [...Object.keys(OPENCODE_TOOL), ...CLAUDE_ONLY];", "const names = [...Object.keys(OPENCODE_TOOL)];", SKILLS, "expect(findings.map((f) => `${f.skill} ${relative(join(f.file, \"..\"), f.file)}:${f.line} ${f.tool} -> ${f.opencode}`).sort()).toEqual(["),
  port("PORT-CONFIG-INSTR", CFG, "instructions: opts.instructions,", "instructions: opts.instructions.slice(0, 1),", CONFIG, "expect(JSON.parse(readFileSync(join(dest, \"opencode.json\"), \"utf-8\"))).toEqual({ instructions, plugins, mcp });"),
];
