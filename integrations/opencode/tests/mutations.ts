/** Red proof, replayed: every behaviour test must fail, in its own `expect`, against a deliberate
 * mutation of the production line it protects.
 *
 *     bun run mutations            # every row
 *     bun run mutations MCP-LOCK   # some rows, by id
 *
 * For each row: run the test green, apply the mutation (the `find` text must occur exactly once),
 * run the test and require it to fail at the row's `assertion`, restore the file, and run it green
 * again. A failure at another `expect` (a precondition, say) or of another kind (an exception thrown
 * outside `expect`, a syntax error, a missing module) is not red proof and counts as a survivor. Exit 1
 * on any survivor. Other sessions add their own rows below; each row names its test by file and full
 * name.
 *
 * `assertion` is the failing expect's source line as bun's code frame prints it, trimmed (the line above
 * the `^`). Every row needs one (owner, 2026-11-11), and `Row` requires it except on a live row, whose
 * line only a live run can capture. A row that has none yet is never RED: it is listed as UNKNOWN with
 * the line it failed at, which `out/mutation-captured.json` also records.
 *
 * A row with `live` names a test that needs that environment variable set to "1" (a real server):
 * without it the row is not run and is listed as UNKNOWN, never as RED. */

import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { ROWS as PLUGIN_ROWS } from "./mutations.plugin";
import { ROWS as PORT_ROWS } from "./mutations.port";
import { ROWS as INTEGRATION_ROWS } from "./mutations.integration";

import type { Row } from "./mutation_row";

const ROOT = resolve(import.meta.dir, "..");

export const ROWS: Row[] = [
  // -- Task 2: MCP client, child process, config ------------------------------------------------
  {
    id: "MCP-JSON",
    file: "src/mcp_client.ts",
    find: "return { ok: true, json: JSON.parse(text) };",
    replace: "return { ok: true, json: { result: JSON.parse(text) } };",
    test: "tests/unit/mcp_client.test.ts",
    name: "handshake then call returns the JSON text unchanged",
    assertion: "expect(result).toEqual({ ok: true, json: SEARCH_ANSWER });",
  },
  {
    id: "MCP-LIST",
    file: "src/mcp_client.ts",
    find: 'tools.push({ name: t.name, description: t.description ?? "", inputSchema: t.inputSchema ?? {} });',
    replace: 'tools.push({ name: t.name, description: t.description ?? "", inputSchema: { type: "object" } });',
    test: "tests/unit/mcp_client.test.ts",
    name: "listTools returns the server's own tool definitions",
    assertion: "expect(await client.listTools(10_000)).toEqual(listed.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })));",
  },
  {
    id: "CHILD-ENV-PASSTHROUGH",
    file: "src/child.ts",
    find: '"HF_HOME", "HF_HUB_OFFLINE", "FASTEMBED_CACHE_PATH",',
    replace: '"HF_HOME", "FASTEMBED_CACHE_PATH",',
    test: "tests/unit/child.test.ts",
    name: "the child starts in dataDir with only the allowed variables",
    assertion: "expect(seen).toEqual([...new Set(expected.map((k) => k.toUpperCase()))].sort());",
  },
  {
    // recall-hermes 8134e23 passes the policy `recall setup` applied (recall-claw audit ENV-003), 2026-11-15.
    id: "CHILD-ENV-POLICY",
    file: "src/child.ts",
    find: ' "RECALL_SOURCE_POLICY_FILE",',
    replace: "",
    test: "tests/unit/child.test.ts",
    name: "the source policy file reaches the child",
    assertion: "expect(env.RECALL_SOURCE_POLICY_FILE).toBe(policy);",
  },
  {
    id: "CHILD-ENV-PYTHON",
    file: "src/child.ts",
    find: '    PYTHONSAFEPATH: "1",\n',
    replace: "",
    test: "tests/unit/child.test.ts",
    name: "the child starts in dataDir with only the allowed variables",
    assertion: "expect(seen).toEqual([...new Set(expected.map((k) => k.toUpperCase()))].sort());",
  },
  {
    id: "MCP-TOOL-ERROR",
    file: "src/mcp_client.ts",
    find: "if (result.isError) return",
    replace: "if (false) return",
    test: "tests/unit/mcp_client.test.ts",
    name: "a tool error comes back as ok false with the server's text",
    assertion: "expect(result).toEqual({ ok: false, error: \"store is not calibrated\" });",
  },
  {
    id: "MCP-LOCK",
    file: "src/mcp_client.ts",
    find: "const run = this.queue.then(fn, fn);",
    replace: "const run = fn();",
    test: "tests/unit/mcp_client.test.ts",
    name: "calls are serialized: two concurrent calls reach the server in order, never interleaved",
    assertion: "expect(settled).toEqual([\"first\", \"second\"]);",
  },
  {
    id: "MCP-TIMEOUT",
    file: "src/mcp_client.ts",
    find: "const bounded = timeoutMs === undefined ? answer : withTimeout(answer, Math.max(0, timeoutMs), method);",
    replace: "const bounded = answer;",
    test: "tests/unit/mcp_client.test.ts",
    name: "a call past its timeout returns ok false and the client stays usable",
    assertion: "expect(late.ok).toBe(false);",
  },
  {
    id: "MCP-RESTART",
    file: "src/mcp_client.ts",
    find: "if (this.now() - this.lastStart < interval) throw",
    replace: "if (false) throw",
    test: "tests/unit/mcp_client.test.ts",
    name: "a dead child restarts on the next call, and not again within 60 s",
    assertion: "expect(refused).toEqual({ ok: false, error: \"the RE-call server is not running\" });",
  },
  {
    id: "MCP-RESTART-NONE",
    file: "src/mcp_client.ts",
    find: "      await this.stop();\n      await this.start();",
    replace: '      throw new McpError("the RE-call server is not running");',
    test: "tests/unit/mcp_client.test.ts",
    name: "a dead child restarts on the next call, and not again within 60 s",
    assertion: "expect(await client.call(\"recall_evidence\", { query: \"x\" }, 30_000)).toEqual({ ok: true, json: EVIDENCE_ANSWER });",
  },
  {
    id: "MCP-READY",
    file: "src/mcp_client.ts",
    find: 'await withTimeout(this.handshake, timeoutMs, "initialize");',
    replace: "await this.handshake;",
    test: "tests/unit/mcp_client.test.ts",
    name: "ready waits for a slow initialize and gives up at the bound",
    assertion: "expect(outcome).toBe(\"initialize got no answer within 0.5s\");",
  },
  {
    id: "MCP-CLOSE",
    file: "src/mcp_client.ts",
    find: "await proc.stdin.end();",
    replace: "void 0;",
    test: "tests/unit/mcp_client.test.ts",
    name: "close closes stdin and the child exits",
    assertion: "expect(fake.events().map((e) => e.event)).toEqual([\"start\", \"exit\"]);",
  },
  {
    id: "CHILD-ENV",
    file: "src/child.ts",
    find: "const env: Record<string, string> = {};",
    replace: "const env: Record<string, string> = { ...(parent as Record<string, string>) };",
    test: "tests/unit/child.test.ts",
    name: "the child starts in dataDir with only the allowed variables",
    assertion: "expect(seen).toEqual([...new Set(expected.map((k) => k.toUpperCase()))].sort());",
  },
  {
    id: "CHILD-DSN",
    file: "src/config.ts",
    find: 'return LITE_PREFIX + file.replaceAll("\\\\", "/");',
    replace: 'return LITE_PREFIX + encodeURI(file.replaceAll("\\\\", "/"));',
    test: "tests/unit/child.test.ts",
    name: "a store path with spaces and non-ASCII characters gives a sqlite:/// DSN the child accepts",
    assertion: "expect(existsSync(storePath(received))).toBe(true);",
  },
  {
    id: "CHILD-RESOLVE",
    file: "src/child.ts",
    find: "for (const entry of pathEnv.split(delimiter)) {",
    replace: "for (const entry of [cwd, ...pathEnv.split(delimiter)]) {",
    test: "tests/unit/child.test.ts",
    name: "resolveExecutable refuses a recall-mcp found only in cwd",
    assertion: "expect(() => resolveExecutable(\"recall-mcp\", [\".\", empty].join(delimiter), project)).toThrow(",
  },
  {
    id: "CONFIG-DEFAULTS",
    file: "src/config.ts",
    find: "const k = options.k ?? 5;",
    replace: "const k = options.k ?? 10;",
    test: "tests/unit/config.test.ts",
    name: "defaults are k 5, every switch on, and a Lite store under dataDir",
    assertion: "expect(loadConfig({}, dataDir)).toEqual({",
  },
  {
    id: "CONFIG-RELATIVE",
    file: "src/config.ts",
    find: "return isAbsolute(path) ? resolve(path) : resolve(base, path);",
    replace: "return resolve(path);",
    test: "tests/unit/config.test.ts",
    name: "relative paths resolve against dataDir",
    assertion: "expect(config.dsn).toBe(\"sqlite:///\" + join(dataDir, \"store\", \"m.db\").replaceAll(\"\\\\\", \"/\"));",
  },
  {
    id: "CONFIG-UNKNOWN",
    file: "src/config.ts",
    find: "if (!known.has(key)) throw",
    replace: "if (false) throw",
    test: "tests/unit/config.test.ts",
    name: "an unknown or mistyped option is refused",
    assertion: "expect(() => loadConfig({ write_tool: false }, \"/d\")).toThrow('unknown option \"write_tool\"');",
  },
  // The shared per-user store (owner, 2026-11-13, "One shared Lite store"), session G.
  ...[
    ["CONFIG-SHARED-CHOSEN", "if (!sharedHasStore && existsSync(", "if (existsSync(", "the shared store is chosen when present"],
    ["CONFIG-OLD-KEPT", "} else storeDir = sharedDir;", "} storeDir = sharedDir;", "an older OpenCode store is kept and the move is logged in one line"],
    ["CONFIG-SHARED-DEFAULT", "if (!sharedHasStore && existsSync(join(dataDir, \"memory.db\"))) {", "if (!sharedHasStore) {", "the shared default applies when neither store exists"],
    ["CONFIG-DSN-WINS", "if (!dsn && sharedDir !== undefined) {", "if (sharedDir !== undefined) {", "an explicit dsn wins"],
    // The memo root follows the chosen store: a split would index one folder and write to another.
    ["CONFIG-MEMOS-FOLLOW-STORE", 'memo ?? join(storeDir, "memos"),', 'memo ?? join(dataDir, "memos"),', "the shared store is chosen when present"],
  ].map(([id, find, replace, name]) => ({
    id: id!,
    file: "src/config.ts",
    find: find!,
    replace: replace!,
    test: "tests/unit/config.test.ts",
    name: name!,
    assertion: "expect({ dsn: config.dsn, memoFolder: config.memoFolder, notice: config.notice }).toEqual({",
  })),
  {
    id: "CONFIG-SHARED-DIR",
    file: "src/config.ts",
    find: 'return join(configHome(env.XDG_CONFIG_HOME), "re-call");',
    replace: 'return join(configHome(env.XDG_CONFIG_HOME), "opencode", "re-call");',
    test: "tests/unit/config.test.ts",
    name: "the shared folder is re-call under XDG_CONFIG_HOME",
    assertion: "expect(sharedStoreDir({ XDG_CONFIG_HOME: xdg })).toBe(join(xdg, \"re-call\"));",
  },
  // XDG_CONFIG_HOME read as recall-hermes `shared_dir` reads it (recall-claw audit DAT-003, ENV-002), 2026-11-15.
  // The ROOTED and UNC rows are red on Windows only, where `pythonIsAbs` has those branches.
  ...[
    ["CONFIG-XDG-ANY", "return pythonIsAbs(xdg) ? xdg :", "return xdg ? xdg :"],
    ["CONFIG-XDG-ROOTED", String.raw`return head.startsWith(":\\", 1) || head.startsWith("\\\\");`, String.raw`return head.startsWith(":\\", 1) || head.startsWith("\\");`],
    ["CONFIG-XDG-UNC", String.raw`return head.startsWith(":\\", 1) || head.startsWith("\\\\");`, String.raw`return head.startsWith(":\\", 1);`],
  ].map(([id, find, replace]) => ({
    id: id!,
    file: "src/config.ts",
    find: find!,
    replace: replace!,
    test: "tests/unit/config.test.ts",
    name: "XDG_CONFIG_HOME counts only when Hermes would call it absolute",
    assertion: "expect(cases.map((c) => sharedStoreDir({ XDG_CONFIG_HOME: c.xdg }))).toEqual(cases.map(expected));",
  })),
  // -- Task 3: renderer and prefetch block ------------------------------------------------------
  ...[
    ["RENDER-CITABLE", "&& (ids === null || ids.has(pyStr(hit.chunk_id)))", "&& true", "expect(cases.map((c) => ({ name: c.name, output: renderSearch(c.input) }))).toEqual("],
    ["RENDER-RETIRED", "if (RETIRED_VERDICTS.has(verdict)) {", "if (false) {", "expect(cases.map((c) => ({ name: c.name, output: renderSearch(c.input) }))).toEqual("],
    ["RENDER-ABSTAIN", '!abstained && verdict === "ok"', 'verdict === "ok"', "expect(cases.map((c) => ({ name: c.name, output: renderSearch(c.input) }))).toEqual("],
    ["RENDER-CLIP", "points.slice(0, MAX_HIT_CHARS - 3)", "points.slice(0, MAX_HIT_CHARS)", "expect(cases.map((c) => ({ name: c.name, output: renderSearch(c.input) }))).toEqual("],
    ["RENDER-CLIP-BOUND", "points.length <= MAX_HIT_CHARS ?", "points.length < MAX_HIT_CHARS ?", "expect(cases.map((c) => ({ name: c.name, output: renderSearch(c.input) }))).toEqual("],
    ["RENDER-CODEPOINT", "const points = Array.from(flat);", 'const points = flat.split("");', "expect(cases.map((c) => ({ name: c.name, output: renderSearch(c.input) }))).toEqual("],
    ["RENDER-PYWS", "\\\\x1c-\\\\x20\\\\x85", "\\\\x20", "expect(cases.map((c) => ({ name: c.name, output: renderSearch(c.input) }))).toEqual("],
    ["RENDER-TRUTHY", "if (isMapping(v)) return Object.keys(v).length > 0;", "", "expect(cases.map((c) => ({ name: c.name, output: renderSearch(c.input) }))).toEqual("],
    ["RENDER-PYSTR", 'if (v === true) return "True";', 'if (v === true) return "true";', "expect(cases.map((c) => ({ name: c.name, output: renderSearch(c.input) }))).toEqual("],
    ["RENDER-VALIDITY", 'parts.join(", ")', 'parts.join("; ")', "expect(cases.map((c) => ({ name: c.name, output: renderSearch(c.input) }))).toEqual("],
  ].map(([id, find, replace, assertion]) => ({
    id: id!, assertion: assertion!, file: "src/render.ts", find: find!, replace: replace!,
    test: "tests/unit/render.test.ts", name: "renderSearch: every fixture renders byte-identically",
  })),
  {
    id: "RENDER-UNAVAILABLE",
    file: "src/render.ts",
    find: "could not answer this turn: ${clip(reason)}",
    replace: "could not answer this turn: ${reason}",
    test: "tests/unit/render.test.ts",
    name: "renderUnavailable: every fixture renders byte-identically",
    assertion: "expect(cases.map((c) => renderUnavailable(c.input))).toEqual(cases.map((c) => c.output));",
  },
  ...[
    ["PREFETCH-TRIVIAL", '{ i: "\\u0130\\u0131", k: "\\u212a", s: "\\u017f" }', "{}", "expect(cases.map((c) => ({ input: c.input, output: isTrivialPrompt(c.input) }))).toEqual(cases);"],
    ["PREFETCH-TRIVIAL-WORD", '"lgtm", "k",', '"lgtm",', "expect(cases.map((c) => ({ input: c.input, output: isTrivialPrompt(c.input) }))).toEqual(cases);"],
    ["PREFETCH-SLASH", ' || stripped.startsWith("/")', "", "expect(cases.map((c) => ({ input: c.input, output: isTrivialPrompt(c.input) }))).toEqual(cases);"],
  ].map(([id, find, replace, assertion]) => ({
    id: id!, assertion: assertion!, file: "src/prefetch.ts", find: find!, replace: replace!,
    test: "tests/unit/prefetch.test.ts", name: "isTrivialPrompt: every fixture agrees with Hermes",
  })),
  ...[
    ["PREFETCH-BLOCK", "Treat as authoritative reference data \\u2014 ", "Treat as authoritative reference data - ", "expect(cases.map((c) => ({ input: c.input, output: buildMemoryContextBlock(c.input) }))).toEqual(cases);"],
    ["PREFETCH-SANITIZE", "[INTERNAL_CONTEXT_RE, INTERNAL_NOTE_RE, FENCE_TAG_RE]", "[FENCE_TAG_RE]", "expect(cases.map((c) => ({ input: c.input, output: buildMemoryContextBlock(c.input) }))).toEqual(cases);"],
    ["PREFETCH-DEDUPE", "if (stripped && !isBullet) seen.clear();", "", "expect(cases.map((c) => ({ input: c.input, output: buildMemoryContextBlock(c.input) }))).toEqual(cases);"],
  ].map(([id, find, replace, assertion]) => ({
    id: id!, assertion: assertion!, file: "src/prefetch.ts", find: find!, replace: replace!,
    test: "tests/unit/prefetch.test.ts", name: "buildMemoryContextBlock: every fixture is byte-identical",
  })),
  {
    id: "PREFETCH-COMPOSE",
    file: "src/prefetch.ts",
    find: 'return injection ? userContent + "\\n\\n" + injection : userContent;',
    replace: 'return injection ? userContent + "\\n" + injection : userContent;',
    test: "tests/unit/prefetch.test.ts",
    name: "compose: every fixture is byte-identical",
    assertion: "expect(cases.map((c) => compose(c.input.user, c.input.injection))).toEqual(cases.map((c) => c.output));",
  },
  {
    id: "PREFETCH-FLOW",
    file: "src/prefetch.ts",
    find: 'if (!isMapping(result.json)) return { injection: "", answer: result.json };',
    replace: "",
    test: "tests/unit/prefetch.test.ts",
    name: "prefetchBlock: every fixture is byte-identical",
    assertion: "expect(played).toEqual(cases.map((c) => ({ name: c.name, ...c.output })));",
  },
  {
    id: "PREFETCH-ERROR",
    file: "src/prefetch.ts",
    find: "if (!result.ok) return { injection: buildMemoryContextBlock(renderUnavailable(result.error).text), answer: null };",
    replace: 'if (!result.ok) return { injection: "", answer: null };',
    test: "tests/unit/prefetch.test.ts",
    name: "prefetchBlock on a search error gives the unavailable line, wrapped",
    assertion: "expect(injection).toBe(c.output.injection);",
  },
  {
    id: "PREFETCH-USER-FENCE",
    file: "src/prefetch.ts",
    find: "const result = await search(userTurn);",
    replace: "const result = await search(sanitizeContext(userTurn));",
    test: "tests/unit/prefetch.test.ts",
    name: "a user turn holding a fence is searched and passed through verbatim",
    assertion: "expect(played.searched).toEqual([c.input.turn]);",
  },
  {
    id: "PREFETCH-USER-FENCE-COMPOSE",
    file: "src/prefetch.ts",
    find: 'return injection ? userContent + "\\n\\n" + injection : userContent;',
    replace: 'return injection ? sanitizeContext(userContent) + "\\n\\n" + injection : userContent;',
    test: "tests/unit/prefetch.test.ts",
    name: "a user turn holding a fence is searched and passed through verbatim",
    assertion: "expect(played.composed.startsWith(c.input.turn + \"\\n\\n<memory-context>\\n\")).toBe(true);",
  },
  {
    id: "PREFETCH-SENTENCE",
    file: "src/prefetch.ts",
    find: '"calibrated trust check, flags superseded ones, and says when it abstained."',
    replace: '"calibrated trust check, flags superseded ones, says when it abstained."',
    test: "tests/unit/prefetch.test.ts",
    name: "the system sentence is the provider's",
    assertion: "expect(SYSTEM_SENTENCE).toBe(fixture<{ output: string }>(\"system_prompt_block.json\").output);",
  },
  // -- Task 4: tool surface ---------------------------------------------------------------------
  {
    id: "SURFACE-VENDORED",
    file: "src/surface.ts",
    find: '"Delete one memo from long-term memory by memo_id.',
    replace: '"Delete a memo from long-term memory by memo_id.',
    test: "tests/unit/surface.test.ts",
    name: "the six are byte-identical to RE-call's assistant surface contract",
    assertion: "expect(ours).toEqual(theirs);",
  },
  {
    id: "SURFACE-VENDORED-SCHEMA",
    file: "src/surface.ts",
    find: '"title": "Memo Id",',
    replace: '"title": "Memo ID",',
    test: "tests/unit/surface.test.ts",
    name: "the six are byte-identical to RE-call's assistant surface contract",
    assertion: "expect(ours).toEqual(theirs);",
  },
  {
    id: "SURFACE-NOT-LITE",
    file: "src/surface.ts",
    find: "!writes(name) || (isLite && config.writeTools)",
    replace: "!writes(name) || config.writeTools",
    test: "tests/unit/surface.test.ts",
    name: "write tools are absent when not Lite",
    assertion: "expect(offered.filter((n) => WRITES.includes(n))).toEqual([]);",
  },
  {
    id: "SURFACE-EXTRA-OFF",
    file: "src/surface.ts",
    find: "...(config.extraTools ? EXTRA.map((e) => e.name) : [])",
    replace: "...EXTRA.map((e) => e.name)",
    test: "tests/unit/surface.test.ts",
    name: "extras are absent when extraTools is false",
    assertion: "expect(names(selectTools(SERVER, config({ extra_tools: false }), true))).toEqual(names(TRAINED));",
  },
  {
    id: "SURFACE-UNLISTED",
    file: "src/surface.ts",
    find: "allowed.has(e.name) && served.has(e.name) ? [served.get(e.name)!] : []",
    replace: "allowed.has(e.name) ? [served.get(e.name) ?? { name: e.name, description: e.reason, inputSchema: {} }] : []",
    test: "tests/unit/surface.test.ts",
    name: "an extra the server does not list is not offered",
    assertion: "expect(names(offered)).not.toContain(\"recall_withdraw\");",
  },
  {
    id: "SURFACE-OPERATOR",
    file: "src/surface.ts",
    find: "return [...TRAINED.filter((t) => allowed.has(t.name)), ...extras];",
    replace: "return [...TRAINED.filter((t) => allowed.has(t.name)), ...serverTools.filter((t) => !TRAINED.some((x) => x.name === t.name))];",
    test: "tests/unit/surface.test.ts",
    name: "no operator tool is ever selected",
    assertion: "expect(names(selectTools(SERVER, config(options), lite)).filter((n) => dropped.includes(n))).toEqual([]);",
  },
  {
    id: "SURFACE-CHILD-TOOLS",
    file: "src/child.ts",
    find: 'RECALL_MCP_TOOLS: offeredNames(config, isLiteDsn(config.dsn)).join(","),',
    replace: 'RECALL_MCP_TOOLS: "assistant",',
    test: "tests/unit/surface.test.ts",
    name: "the child is told to serve exactly the offered tools",
    assertion: "expect(childEnv(c, {}).RECALL_MCP_TOOLS!.split(\",\")).toEqual(names(selectTools(SERVER, c, true)));",
  },
  {
    id: "SURFACE-CLEANDOC",
    file: "src/surface.ts",
    find: "...lines.slice(1).map((l) => l.slice(margin))]",
    replace: "...lines.slice(1)]",
    test: "tests/unit/surface.test.ts",
    name: "cleandoc matches inspect.cleandoc",
    assertion: "expect(cleandoc(\"First line.\\n\\n    Indented body\\n        deeper\\n    back\\n\\n\")).toBe(\"First line.\\n\\nIndented body\\n    deeper\\nback\");",
  },
  {
    id: "LIVE-TRAINED",
    file: "src/surface.ts",
    find: '"Delete one memo from long-term memory by memo_id.',
    replace: '"Delete a memo from long-term memory by memo_id.',
    test: "tests/contract/surface_live.test.ts",
    name: "the six served are the vendored six, and every extra is served",
    live: "RECALL_OPENCODE_SURFACE",
    assertion: "expect(JSON.stringify(normalizeTool(served.get(trained.name)!))).toBe(JSON.stringify(normalizeTool(trained)));",
  },
  {
    id: "LIVE-CLASSIFIED",
    file: "src/extra_tools.json",
    find: '    {"name": "recall_stats", "reason": "Operator: counters and operational state, kept off by the spec."},\n',
    replace: "",
    test: "tests/contract/surface_live.test.ts",
    name: "every tool Lite serves is classified, so a changed list fails",
    live: "RECALL_OPENCODE_SURFACE",
    assertion: "expect(tools.map((t) => t.name).sort()).toEqual(classified.filter((n) => !NOT_ON_LITE.includes(n)).sort());",
  },
  // -- core-fixes: review of src/mcp_client.ts, 2026-11-10 ----------------------------------------
  ...[
    ["MCP-CLOSED", "if (this.closed) throw", "if (false) throw", "after close, a call starts no child", "expect(await client.call(\"recall_evidence\", { query: \"x\" }, 10_000)).toEqual({ ok: false, error: \"the RE-call client was closed\" });"],
    ["MCP-CLOSED-RACE", "if (this.closed) throw", "if (false) throw", "a close racing a restart leaves no child running", "expect((await restarting).ok).toBe(false);"],
    ["MCP-WEDGED-READY", "if (error instanceof TimeoutError && proc && this.proc === proc) proc.kill();", "void proc;",
      "a child that never answers the handshake is killed when ready gives up", "expect(fake.events().filter((e) => e.event === \"start\").length).toBe(2);"],
    ["MCP-WEDGED-CALLS", "export const WEDGED_AFTER_TIMEOUTS = 2;", "export const WEDGED_AFTER_TIMEOUTS = 1;",
      "two consecutive call timeouts kill a child that stopped reading, one does not", "expect(running(first)).toBe(true); // one timeout left the child alone"],
    ["MCP-WEDGED-KILL", "if (this.timeouts >= WEDGED_AFTER_TIMEOUTS && proc && this.proc === proc) proc.kill();", "void proc;",
      "two consecutive call timeouts kill a child that stopped reading, one does not", "expect(starts()).toBe(2); // the second timeout killed it, so this call started a new one"],
    ["MCP-FLOOR", "if (left < SEND_FLOOR_MS) return", "if (false) return", "a call is not sent with its budget almost spent", "expect(await request).toEqual({"],
    ["MCP-CANCEL", 'this.send({ jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId: id, reason } });', "void reason;",
      "a call that times out is cancelled on the server", "expect(events).toEqual([\"call\", \"cancelled\", \"call\"]);"],
    ["MCP-INIT-ERROR", 'const init = this.requestResult("initialize", {', 'const init = this.request("initialize", {',
      "an initialize answered with an error is not ready", "expect(await client.ready(10_000).then(() => \"ready\", (e: Error) => e.message)).toBe(\"initialize: unsupported protocol version\");"],
  ].map(([id, find, replace, name, assertion]) => ({
    id: id!, assertion: assertion!, file: "src/mcp_client.ts", find: find!, replace: replace!, test: "tests/unit/mcp_client.test.ts", name: name!,
  })),
  // -- audit (Task 9) -------------------------------------------------------------------------------
  {
    id: "PREFETCH-SANITIZE-FIXPOINT",
    file: "src/prefetch.ts",
    find: 'for (let before = ""; before !== text; ) {',
    replace: 'for (let before = ""; before === ""; ) {',
    test: "tests/unit/prefetch.test.ts",
    name: "a memo holding a nested fence tag cannot close the block",
    assertion: "expect(injection.split(\"</memory-context>\").length - 1).toBe(1);",
  },
  ...[
    ["MCP-QUEUE-BUDGET", "return await Promise.race([run, gaveUp]);", "return await run;",
      "a call queued behind a slow one comes back at its own budget, unsent", "expect(elapsed).toBeLessThan(2_000);"],
    ["MCP-QUEUE-LEFT", "const left = deadline - this.now();", "const left = timeoutMs;",
      "a call that waited for its turn is bounded by what is left of its budget", "expect(queued.ok).toBe(false);"],
    ["MCP-LATE-ANSWER", "else this.timeouts = 0;", "", "late answers do not get a slow child killed as wedged", "expect(await client.call(\"recall_search\", { query: \"x\" }, 5_000)).toEqual({ ok: true, json: SEARCH_ANSWER });"],
    ["MCP-START-BOUND", "      if (this.proc === proc) proc.kill();\n    });", "      void proc;\n    });",
      "a child that refuses the handshake is replaced on a later call", "expect(starts()).toBe(2);"],
    ["MCP-ABORT", "waiter.reject(new McpError(`${method} was aborted`));", "void waiter;",
      "an aborted call returns at once and is cancelled on the server", "expect(await pending).toEqual({ ok: false, error: \"tools/call was aborted\" });"],
  ].map(([id, find, replace, name, assertion]) => ({
    id: id!, assertion: assertion!, file: "src/mcp_client.ts", find: find!, replace: replace!, test: "tests/unit/mcp_client.test.ts", name: name!,
  })),
  {
    id: "FAKE-ENV-MINIMAL",
    file: "tests/unit/fake.ts",
    find: "env: { ...(extra.env ?? base), ...fake.env },",
    replace: "env: { ...(extra.env ?? (process.env as Record<string, string>)), ...fake.env },",
    test: "tests/unit/mcp_client.test.ts",
    name: "a client over the fake does not hand it the parent's environment",
    assertion: "expect(Object.keys(env)).not.toContain(\"RECALL_OC_PROBE_TOKEN\");",
  },
  {
    id: "MUT-RECOVER",
    file: "tests/mutations.ts",
    find: "  restoreMutation(root, file, original);\n  return file;",
    replace: "  return file;",
    test: "tests/unit/mutations.test.ts",
    name: "a row an interrupted run left applied is undone by the next run",
    assertion: "expect(readFileSync(file, \"utf-8\")).toBe(\"original\\n\");",
  },
  {
    id: "MUT-RECOVER-CLEAN",
    file: "tests/mutations.ts",
    find: "if (now === original) {\n",
    replace: 'if (now === "never") {\n',
    test: "tests/unit/mutations.test.ts",
    name: "a journal whose file is already the original is cleared",
    assertion: "expect(refused).toBeUndefined();",
  },
  {
    id: "MUT-RECOVER-CHANGED",
    file: "tests/mutations.ts",
    find: "if (now !== mutated) {\n",
    replace: 'if (now === "never") {\n',
    test: "tests/unit/mutations.test.ts",
    name: "a file changed since the interruption is not overwritten",
    assertion: "expect(() => recoverInterrupted(dir)).toThrow(\"src/x.ts changed after an interrupted mutation run\");",
  },
  // -- audit (deferred pass, CCA run 20261111T143416Z) ----------------------------------------------
  {
    id: "CHILD-ABSOLUTE-PATHEXT",
    file: "src/child.ts",
    find: "if (found) return resolve(found);",
    replace: "void found;",
    test: "tests/unit/child.test.ts",
    name: "an absolute recall_command without .exe finds the .exe on Windows",
    assertion: "expect(got).toBe(norm(join(bin, \"recall-mcp.exe\")));",
  },
  {
    id: "MCP-READ-STREAM",
    file: "src/mcp_client.ts",
    find: "decoder.decode(chunk, { stream: true })",
    replace: "decoder.decode(chunk)",
    test: "tests/unit/mcp_client.test.ts",
    name: "an answer split inside a UTF-8 character comes back whole",
    assertion: "expect(result).toEqual({ ok: true, json: { q: \"café ☕ 東京\" } });",
  },
  {
    id: "MCP-LOG-CAP",
    file: "src/mcp_client.ts",
    find: 'renameSync(log, log + ".1");',
    replace: "void log;",
    test: "tests/unit/mcp_client.test.ts",
    name: "a server log at its cap is rotated at the next start",
    assertion: "expect(existsSync(rotated)).toBe(true);",
  },
  // -- the intended assertion (owner, 2026-11-11) ---------------------------------------------------
  {
    id: "MUT-ASSERT-FRAME",
    file: "tests/mutations.ts",
    find: "found.push(lines[i - 2]!.replace(/^\\s*\\d+ \\| /, \"\").trim());\n",
    replace: "found.push(lines[i - 3]!.replace(/^\\s*\\d+ \\| /, \"\").trim());\n",
    test: "tests/unit/mutations.test.ts",
    name: "the failing assertion is read from bun's code frame",
    assertion: "expect(failedAssertions(out)).toEqual([SECOND]);",
  },
  {
    id: "MUT-ASSERT-PIN",
    file: "tests/mutations.ts",
    find: "? \"expect\" : \"elsewhere\";\n}",
    replace: "? \"expect\" : \"expect\";\n}",
    test: "tests/unit/mutations.test.ts",
    name: "a failure at another expect is not red",
    assertion: "expect(verdict(early, \"fixture fails at its first expect\", SECOND)).toBe(\"elsewhere\");",
  },
  {
    id: "MUT-ASSERT-UNPINNED",
    file: "tests/mutations.ts",
    find: "  if (assertion === undefined) return \"unpinned\";\n",
    replace: "",
    test: "tests/unit/mutations.test.ts",
    name: "a row without an assertion is never red",
    assertion: "expect(verdict(late, \"fixture fails at its second expect\")).toBe(\"unpinned\");",
  },
];

/** The row in flight, kept on disk: a run killed before its restore (a kill skips `finally`) leaves the
 * mutation in the very `src/` OpenCode loads, and the next run undoes it from here. */
export const JOURNAL = "out/mutation-in-progress.json";
/** For each row run without an `assertion` that went red at expects only, the lines it failed at. */
export const CAPTURED = "out/mutation-captured.json";

export function applyMutation(root: string, file: string, original: string, mutated: string): void {
  const journal = resolve(root, JOURNAL);
  mkdirSync(dirname(journal), { recursive: true });
  writeFileSync(journal, JSON.stringify({ file, original, mutated }), "utf-8");
  writeFileSync(resolve(root, file), mutated, "utf-8");
}

export function restoreMutation(root: string, file: string, original: string): void {
  writeFileSync(resolve(root, file), original, "utf-8");
  rmSync(resolve(root, JOURNAL), { force: true });
}

/** Undo the row an interrupted run left applied, and name its file; refuse when the file changed since,
 * so an edit made after the interruption is never overwritten. */
export function recoverInterrupted(root: string): string | undefined {
  const journal = resolve(root, JOURNAL);
  if (!existsSync(journal)) return undefined;
  const { file, original, mutated } = JSON.parse(readFileSync(journal, "utf-8")) as { file: string; original: string; mutated: string };
  const now = readFileSync(resolve(root, file), "utf-8");
  // Killed between the journal and the file write, or between the restore and the journal's removal.
  if (now === original) {
    rmSync(journal, { force: true });
    return undefined;
  }
  if (now !== mutated) {
    throw new Error(`${file} changed after an interrupted mutation run; its original is in ${journal}`);
  }
  restoreMutation(root, file, original);
  return file;
}

const escapeRegex = (s: string) => s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");

function runTest(row: Row): { code: number; out: string } {
  const proc = Bun.spawnSync(["bun", "test", row.test, "-t", escapeRegex(row.name)], { cwd: ROOT, stdout: "pipe", stderr: "pipe" });
  return { code: proc.exitCode ?? 1, out: proc.stdout.toString() + proc.stderr.toString() };
}

/** The source line of each failing `expect`, as bun's code frame prints it: the frame line just above
 * the `^` under it, without its line number, trimmed (bun 1.4.4; a call into a helper shows the call,
 * a multi-line expect the line the caret points at). */
export function failedAssertions(out: string): string[] {
  const lines = out.split(/\r?\n/);
  const found: string[] = [];
  lines.forEach((line, i) => {
    if (!line.trim().startsWith("error: expect(") || i < 2 || lines[i - 1]!.trim() !== "^") return;
    found.push(lines[i - 2]!.replace(/^\s*\d+ \| /, "").trim());
  });
  return found;
}

/** Green means the named test ran and passed; red (`expect`) means it ran and failed at `assertion`,
 * and nothing else went wrong. A failure at other expects is `elsewhere`; with no assertion given, a
 * failure only at expects is `unpinned`. */
export function verdict(out: string, name: string, assertion?: string): "pass" | "expect" | "elsewhere" | "unpinned" | "other" {
  // Bun names only failing tests when its output is not a terminal; the filter must select exactly one.
  if (!/Ran 1 test /.test(out)) return "other";
  if (/ 1 pass/.test(out) && / 0 fail/.test(out)) return "pass";
  if (!out.split(/\r?\n/).some((l) => l.startsWith("(fail)") && l.includes(name))) return "other";
  const errors = out.split(/\r?\n/).filter((l) => /^(\w*Error\b|error:)/.test(l.trim()));
  if (errors.length === 0 || !errors.every((l) => l.trim().startsWith("error: expect("))) return "other";
  if (assertion === undefined) return "unpinned";
  const failed = failedAssertions(out);
  return failed.length > 0 && failed.every((text) => text === assertion) ? "expect" : "elsewhere";
}

function main(): number {
  const recovered = recoverInterrupted(ROOT);
  if (recovered) console.log(`restored ${recovered}, left mutated by an interrupted run`);
  // Ctrl+C reaches the test run too; with a handler this process lives on to restore the file.
  // Nonbehavioral control: no test sends this process a signal.
  let interrupted = false;
  process.on("SIGINT", () => (interrupted = true));
  const wanted = new Set(process.argv.slice(2));
  const all: Row[] = [...ROWS, ...PLUGIN_ROWS, ...PORT_ROWS, ...INTEGRATION_ROWS];
  const rows = wanted.size ? all.filter((r) => wanted.has(r.id)) : all;
  const results: { id: string; outcome: string; detail: string }[] = [];
  const captured: Record<string, string[]> = {};
  for (const row of rows) {
    if (interrupted) {
      results.push({ id: row.id, outcome: "UNKNOWN", detail: "not run: the run was interrupted" });
      continue;
    }
    if (row.live && process.env[row.live] !== "1") {
      results.push({ id: row.id, outcome: "UNKNOWN", detail: `not run: needs ${row.live}=1 and its server` });
      continue;
    }
    const path = resolve(ROOT, row.file);
    const original = readFileSync(path, "utf-8");
    const count = original.split(row.find).length - 1;
    if (count !== 1) {
      results.push({ id: row.id, outcome: "SURVIVED", detail: `find text occurs ${count} times in ${row.file}` });
      continue;
    }
    const before = runTest(row);
    if (verdict(before.out, row.name) !== "pass") {
      results.push({ id: row.id, outcome: "SURVIVED", detail: "not green before the mutation" });
      continue;
    }
    let red: { code: number; out: string };
    try {
      applyMutation(ROOT, row.file, original, original.replace(row.find, row.replace));
      red = runTest(row);
    } finally {
      restoreMutation(ROOT, row.file, original);
    }
    const kind = verdict(red.out, row.name, row.assertion);
    const after = verdict(runTest(row).out, row.name);
    const failed = red.out.split(/\r?\n/).find((l) => l.trim().startsWith("error: expect(")) ?? "";
    const at = red.out.split(/\r?\n/).find((l) => /^\s*at .*\.test\.ts:\d+:\d+/.test(l))?.trim() ?? "";
    const failedAt = failedAssertions(red.out);
    const redAndBack = red.code !== 0 && after === "pass";
    if (kind === "expect" && redAndBack) {
      results.push({ id: row.id, outcome: "RED", detail: `${failed.trim()} ${at.replace(/^at .*[\\/]/, "at ")}` });
    } else if (kind === "unpinned" && redAndBack) {
      captured[row.id] = failedAt;
      results.push({ id: row.id, outcome: "UNKNOWN", detail: `no assertion recorded; it failed at ${JSON.stringify(failedAt)}` });
    } else {
      const why =
        kind === "pass" ? "the test passed against the mutation"
        : after !== "pass" ? "not green after restore"
        : kind === "elsewhere" ? `failed at ${JSON.stringify(failedAt)}, not at ${JSON.stringify(row.assertion)}`
        : `failed outside expect (exit ${red.code})`;
      results.push({ id: row.id, outcome: "SURVIVED", detail: why });
    }
  }
  if (Object.keys(captured).length > 0) {
    mkdirSync(resolve(ROOT, "out"), { recursive: true });
    writeFileSync(resolve(ROOT, CAPTURED), JSON.stringify(captured, null, 1) + "\n", "utf-8");
    console.log(`assertions of rows without one: ${CAPTURED}`);
  }
  for (const r of results) console.log(`${r.outcome.padEnd(8)} ${r.id.padEnd(22)} ${r.detail}`);
  const survivors = results.filter((r) => r.outcome === "SURVIVED");
  const unknown = results.filter((r) => r.outcome === "UNKNOWN").length;
  console.log(`\n${results.filter((r) => r.outcome === "RED").length} RED, ${survivors.length} survived, ${unknown} UNKNOWN (not run, or no assertion recorded)`);
  if (interrupted) return 130; // a partial run is never a pass
  return survivors.length ? 1 : 0;
}

if (import.meta.main) process.exit(main());
