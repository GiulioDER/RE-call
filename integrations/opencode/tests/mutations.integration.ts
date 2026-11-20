/** Red proof rows for the install script, the test preload, the typecheck's scope and one port row, in the
 * shape of `ROWS` in `tests/mutations.ts`, which replays them. Each row mutates the production line its
 * test protects; the test must fail at an `expect`. */

import type { Row } from "./mutation_row";

const INSTALL = "port/install.ts";
const INSTALL_TEST = "tests/port/install.test.ts";
const WRITES = "install writes the ported commands, agents and config into the given config folder only";
const CLASH = "install refuses to overwrite an existing config and writes nothing";

const row = (id: string, file: string, test: string, name: string, find: string, replace: string, assertion: string): Row =>
  ({ id, file, find, replace, test, name, assertion });

export const ROWS: Row[] = [
  row("PRELOAD-ROOTS", "tests/preload.ts", "tests/unit/preload.test.ts", "OpenCode's global roots in a test process are outside the real home",
    "  process.env[name] = join(scratch, dir);\n", "", "expect(inside).toEqual([]);"),
  // -- the install script --------------------------------------------------------------------
  row("INSTALL-XDG", INSTALL, INSTALL_TEST, WRITES, 'join(opts.xdgConfigHome || join(opts.home, ".config"), "opencode")', 'join(opts.home, ".config", "opencode")', "expect(configDir).toBe(join(xdg, \"opencode\"));"),
  row("INSTALL-INSTR", INSTALL, INSTALL_TEST, WRITES, 'export const INSTRUCTIONS = ["~/.claude/CLAUDE.md", "CLAUDE.md", "CLAUDE.local.md"];', 'export const INSTRUCTIONS = ["~/.claude/CLAUDE.md", "AGENTS.md", "CLAUDE.local.md"];', "expect(JSON.parse(readFileSync(join(configDir, \"opencode.json\"), \"utf8\"))).toEqual({"),
  row("INSTALL-AGENTS", INSTALL, INSTALL_TEST, WRITES, '    ...portAgents(join(claude, "agents"), join(configDir, "agents")).written,\n', "", "}).toEqual({ commands: commands.sort(), agents, rest: [\"opencode.json\"], written: got });"),
  row("INSTALL-RECALL", INSTALL, INSTALL_TEST, WRITES, "options: opts.recallCommand ? { recall_command: opts.recallCommand } : {} },", "options: {} },", "expect(JSON.parse(readFileSync(join(configDir, \"opencode.json\"), \"utf8\"))).toEqual({"),
  row("INSTALL-CLASH", INSTALL, INSTALL_TEST, CLASH, '  if (clash.length > 0) throw new Error(`install: refusing to overwrite ${clash.join(", ")}`);\n', "", "expect({ existing, refused: error.includes(\"refusing to overwrite\"), after: snapshot(xdg) }).toEqual({ existing, refused: true, after: before });"),
  row("INSTALL-CLASH-NAMES", INSTALL, INSTALL_TEST, CLASH,
    '["opencode.json", "opencode.jsonc", "command", "commands", "agent", "agents", "mode", "modes"]', '["opencode.json", "commands", "agents"]',
    "expect({ existing, refused: error.includes(\"refusing to overwrite\"), after: snapshot(xdg) }).toEqual({ existing, refused: true, after: before });"),
  row("INSTALL-SOURCES", INSTALL, INSTALL_TEST, "install refuses a home without both source folders and writes nothing",
    "  if (missing.length > 0) throw new Error(`install: ${missing.join(\", \")} does not exist; nothing was written`);\n", "",
    'expect({ written: files(xdg), refused: error.includes("does not exist; nothing was written") }).toEqual({ written: [], refused: true });'),
  row("TSC-INCLUDE", "tsconfig.json", "tests/unit/typecheck_scope.test.ts", "the typecheck includes every top-level directory that holds TypeScript",
    '"include": ["src", "harness", "tests", "port"]', '"include": ["src", "harness", "tests"]',
    "expect(dirs.filter((d) => !include.includes(d))).toEqual([]);"),
  row("PORT-AGENT-SAFE-NAME", "port/agents.ts", "tests/port/port.test.ts", "an agent whose name is not a plain file name is reported and written nowhere",
    "    if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(agent)) {", "    if (false) {",
    'expect({ written: report.written, outside: existsSync(join(dest, "..", "..", "escaped.md")) }).toEqual({ written: [], outside: false });'),
];
