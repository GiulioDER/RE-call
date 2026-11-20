// The plugin loaded into a real in-process OpenCode 2.2.7 host, with the scripted model and
// tests/fake_mcp_server.py as its `recall-mcp` child. What is asserted is what the model's endpoint
// received (the driver's records, or the fake model's own capture) and what the child was asked.
import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { OpenCode } from "@opencode/sdk";

import { HOST_LOG, runScenario, type ModelRequest, type TurnRecord } from "../../harness/driver";
import { startFakeModel, type ScriptedTurn } from "../../harness/fake_model";
import type { Scenario } from "../../harness/scenario";
import type { Plugin } from "@opencode/plugin";
import { setup } from "../../src/plugin";
import { SYSTEM_SENTENCE } from "../../src/prefetch";
import { EXTRA, TRAINED } from "../../src/surface";
import { FAKE_SERVER, fakeScenario, PYTHON, tempDir, type Fake } from "../unit/fake";
import { fixture } from "../unit/fixtures";
import { assertPluginActive, recallBin } from "../e2e/apparatus";

const PLUGIN = resolve(import.meta.dir, "..", "..", "src");
const TIMEOUT = 90_000;

// One answer and the block the Python originals render for it (tests/fixtures/render/prefetch.json,
// written by scripts/gen_fixtures.py from vendor/ft4/ft4_hermes.py), so the expected bytes do not come
// from the TypeScript port under test.
const CASE = fixture<{ name: string; input: { result: { json: unknown } }; output: { injection: string } }[]>("prefetch.json").find(
  (c) => c.name === "a non-trivial turn gets the block",
)!;
const ANSWER = CASE.input.result.json;
const BLOCK = CASE.output.injection;

/** What the fake `recall-mcp` serves for `tools/list`: the six, every extra, and one operator tool. */
const SERVED = [
  ...TRAINED,
  ...EXTRA.map((e) => ({ name: e.name, description: `server description of ${e.name}`, inputSchema: { type: "object", properties: { x: { type: "string" } } } })),
  { name: "recall_index", description: "operator", inputSchema: { type: "object", properties: {} } },
];

const dirs: string[] = [];
let savedXdg: string | undefined;
let dataRoot: string;
beforeEach(() => {
  savedXdg = process.env.XDG_CONFIG_HOME;
  dataRoot = tempDir("recall-oc-xdg-");
  dirs.push(dataRoot);
  process.env.XDG_CONFIG_HOME = dataRoot; // the plugin's data dir is <XDG_CONFIG_HOME>/opencode/re-call
});
afterEach(() => {
  if (savedXdg === undefined) delete process.env.XDG_CONFIG_HOME;
  else process.env.XDG_CONFIG_HOME = savedXdg;
  for (const d of dirs.splice(0)) {
    try {
      rmSync(d, { recursive: true, force: true, maxRetries: 5 });
    } catch (error) {
      // A child left running (the failure PLUGIN-DISPOSE provokes) keeps its working directory busy on
      // Windows; that must not hide the test's own failure. The temp dir is left behind.
      if ((error as { code?: string }).code !== "EBUSY") throw error;
    }
  }
});

type Recall = { fake: Fake; memos: string; store: string; options: Record<string, unknown> };

/** A store, a memo folder, and a `recall_command` that runs the fake. The child gets only the plugin's
 * minimal environment, so a wrapper script sets the fake's own variables. */
function fakeRecall(scenario: object, { withStore = true } = {}): Recall {
  const dir = tempDir("recall-oc-contract-");
  dirs.push(dir);
  const fake = fakeScenario({ tools_list: SERVED, ...scenario }, dir);
  const store = join(dir, "store", "memory.db");
  const memos = join(dir, "memos");
  if (withStore) {
    mkdirSync(join(dir, "store"));
    writeFileSync(store, "");
  }
  let command: string;
  if (process.platform === "win32") {
    command = join(dir, "recall-mcp.cmd");
    writeFileSync(command, `@echo off\r\nset "FAKE_SCENARIO=${fake.scenario}"\r\nset "FAKE_LOG=${fake.log}"\r\n"${PYTHON}" -I "${FAKE_SERVER}"\r\n`);
  } else {
    command = join(dir, "recall-mcp");
    writeFileSync(command, `#!/bin/sh\nFAKE_SCENARIO='${fake.scenario}' FAKE_LOG='${fake.log}' exec '${PYTHON}' -I '${FAKE_SERVER}'\n`, { mode: 0o755 });
  }
  return { fake, memos, store, options: { recall_command: command, dsn: `sqlite:///${store.replaceAll("\\", "/")}`, memo_folder: memos } };
}

function readRecords(path: string): TurnRecord[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l));
}

async function drive(recall: Recall, script: ScriptedTurn[], sessions: string[][]): Promise<TurnRecord[]> {
  const out = join(tempDir("recall-oc-out-"), "out.jsonl");
  dirs.push(resolve(out, ".."));
  const s: Scenario = { name: "contract", model: { kind: "scripted", script }, sessions: sessions.map((turns) => ({ turns })) };
  await runScenario(s, { plugins: [{ path: PLUGIN, options: recall.options }], outPath: out, configHome: dataRoot });
  return readRecords(out);
}

type Tool = { type: string; function: { name: string; description: string; parameters: unknown; strict?: boolean } };
const toolsOf = (r: ModelRequest) => r.tools as Tool[];
const users = (r: ModelRequest) => r.messages.filter((m) => (m as { role: string }).role === "user").map((m) => (m as { content: unknown }).content);
const searches = (fake: Fake) => fake.events().filter((e) => e.event === "call" && e.tool === "recall_search");

/**
 * Invariant: the model's request lists the six trained tools exactly as vendored and each extra as the
 * server describes it, under their own names, outside Code Mode, beside OpenCode's built-ins; no
 * operator tool the server serves is offered. v2.2.7 adds `strict: false` to every entry and changes
 * nothing else (docs/V2-FACTS.md step 2), so that is the expected shape, not a difference hidden.
 * Red proof, row PLUGIN-TOOLS-SCHEMA: `input: structuredClone(tool.inputSchema)` replaced by
 * `input: { type: "object", properties: {} }` in src/plugin.ts; fails at `expect(recall_).toEqual(expected)`.
 * Row PLUGIN-CODEMODE: `options: { codemode: false }` replaced by `options: {}` (the tools go behind Code
 * Mode's `execute`); fails at the same `expect`.
 */
test(
  "the request lists the six tools with the vendored schemas and the committed extras, beside the built-ins",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER] } });
    const [record] = await drive(recall, [{ text: "done" }], [["Where do I work?"]]);
    const tools = toolsOf(record!.request!);
    const recall_ = tools.filter((t) => t.function.name.startsWith("recall_")).sort((a, b) => a.function.name.localeCompare(b.function.name));
    const expected = [...TRAINED, ...SERVED.filter((t) => EXTRA.some((e) => e.name === t.name))]
      .map((t) => ({ type: "function", function: { name: t.name, description: t.description, parameters: t.inputSchema, strict: false } }))
      .sort((a, b) => a.function.name.localeCompare(b.function.name));
    expect<unknown>(recall_).toEqual(expected);
    expect(tools.map((t) => t.function.name)).toEqual(expect.arrayContaining(["read", "edit", "shell", "execute"]));
  },
  TIMEOUT,
);

/**
 * Invariant: the provider's sentence is one of the system parts, so it ends the system message OpenCode
 * builds from them; OpenCode joins parts with one newline (FT-4 joins its parts with two, a recorded
 * difference of the host). The child starts in the plugin's data dir, `<memoFolder>/opencode` exists
 * before it starts, and the setup logs how long the server took to be ready.
 * Red proof, row PLUGIN-SYSTEM: `event.system.push(...)` removed from src/plugin.ts; fails at
 * `expect(system.endsWith("\n" + SYSTEM_SENTENCE)).toBe(true)`.
 * Row PLUGIN-MEMO-DIR: the `mkdirSync(env.RECALL_ASSISTANT_MEMOS, ...)` line removed; fails at
 * `expect(existsSync(join(recall.memos, "opencode"))).toBe(true)`.
 * Row PLUGIN-READY-LOG: the ready log line removed; fails at `expect(ready.length).toBe(1)`.
 * Row PLUGIN-READY-WAIT: `await client.ready(READY_TIMEOUT_MS);` removed (the log no longer waits for the
 * fake's 1 s `init_delay`); fails at `expect(seconds).toBeGreaterThanOrEqual(1)`.
 * Row PLUGIN-CWD: `cwd: dataDir,` replaced by `cwd: process.cwd(),`; fails at
 * `expect(resolve(start.cwd!)).toBe(...)`.
 */
test(
  "the system sentence is in the system parts",
  async () => {
    const info = spyOn(console, "info");
    try {
      const recall = fakeRecall({ tools: { recall_search: [ANSWER] }, init_delay: 1 });
      const [record] = await drive(recall, [{ text: "done" }], [["Where do I work?"]]);
      const system = record!.request!.system.join("\n") as string;
      expect(system.endsWith("\n" + SYSTEM_SENTENCE)).toBe(true);
      expect(existsSync(join(recall.memos, "opencode"))).toBe(true);
      const start = recall.fake.events().find((e) => e.event === "start")!;
      expect(resolve(start.cwd!)).toBe(resolve(dataRoot, "opencode", "re-call"));
      const ready = info.mock.calls.map((c) => String(c[0])).filter((l) => /^re-call: server ready after \d+\.\d s$/.test(l));
      expect(ready.length).toBe(1);
      const seconds = Number(/after (\S+) s/.exec(ready[0]!)![1]);
      expect(seconds).toBeGreaterThanOrEqual(1);
    } finally {
      info.mockRestore();
    }
  },
  TIMEOUT,
);

/**
 * Invariant: the block is appended to the user message exactly as `compose` does (the bytes the Python
 * originals produce), and a past turn keeps its block in every later request, as Hermes v0.23.0
 * replays its `api_content` sidecar (docs/V2-FACTS.md step 5). The search is `{query: <turn>, k: 5}`.
 * Red proof, row PLUGIN-COMPOSE: `event.prompt.text = compose(event.prompt.text, injection)` replaced by
 * `event.prompt.text = injection + event.prompt.text`; fails at
 * `expect(users(first.request!)).toEqual([prompt + "\n\n" + BLOCK])`.
 * Row PLUGIN-K: `{ query: q, k: config.k }` replaced by `{ query: q }`; fails at
 * `expect(searches(recall.fake).map((e) => e.arguments)).toEqual(...)`.
 */
test(
  "the block is appended to the user message exactly as compose does, and present in later requests",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER] } });
    const prompt = "Where do I work?";
    const second = "And what do I do there?";
    const [first, next] = await drive(recall, [{ text: "a1" }, { text: "a2" }], [[prompt, second]]);
    expect<unknown>(users(first!.request!)).toEqual([prompt + "\n\n" + BLOCK]);
    expect<unknown>(users(next!.request!)).toEqual([prompt + "\n\n" + BLOCK, second + "\n\n" + BLOCK]);
    // The driver's record carries the same block, split from the stored user text.
    expect([first!.prefetch, next!.prefetch]).toEqual([BLOCK, BLOCK]);
    expect<unknown>(searches(recall.fake).map((e) => e.arguments)).toEqual([
      { query: prompt, k: 5 },
      { query: second, k: 5 },
    ]);
  },
  TIMEOUT,
);

/**
 * Invariant: a trivial prompt (Hermes's `is_trivial_prompt`) sends no search and gets no block.
 * Red proof, row PLUGIN-TRIVIAL: the prompt hook's `prefetchBlock(..., event.prompt.text)` given
 * `event.prompt.text + " please"` instead (never trivial); fails at `expect(searches(recall.fake).length).toBe(0)`.
 */
test(
  "a trivial prompt gets no block and no search",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER] } });
    const [record] = await drive(recall, [{ text: "fine" }], [["ok"]]);
    expect(searches(recall.fake).length).toBe(0);
    expect<unknown>(users(record!.request!)).toEqual(["ok"]);
    expect(record!.prefetch).toBeNull();
  },
  TIMEOUT,
);

/**
 * Invariant: the prefetch runs once per user turn, not once per model call: a tool call and its
 * continuation do not search again.
 * Red proof, row PLUGIN-ONCE: the prefetch registered on the `context` hook instead of the `prompt` hook
 * (`hook("prompt", async (event) => {` replaced by a `context` hook that takes the last message's text
 * as `event.prompt.text`), so it searches on every model call; fails at
 * `expect(searches(recall.fake).length).toBe(1)`.
 */
test(
  "a tool continuation does not search again",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER], recall_current_facts: [{ facts: [] }] } });
    const [record] = await drive(recall, [{ toolCalls: [{ name: "recall_current_facts", arguments: {} }] }, { text: "done" }], [["Where do I work?"]]);
    expect(record!.continuations.length).toBe(1);
    expect(searches(recall.fake).length).toBe(1);
    expect(recall.fake.events().filter((e) => e.event === "call").map((e) => e.tool)).toEqual(["recall_search", "recall_current_facts"]);
  },
  TIMEOUT,
);

/**
 * Invariant: a model's tool call reaches the child under its own name with the model's arguments, less
 * those that are null or "" (recall-hermes `handle_tool_call`), and the model reads the server's answer
 * as Python's `json.dumps` writes it, which is what Hermes returns. The expected string was written by
 * `python -c "import json; print(json.dumps({'memo_id': 'm1.md', 'note': 'café ✓'}))"`.
 * Red proof, row PLUGIN-ARGS: `.filter(([, v]) => v !== null && v !== undefined && v !== "")` replaced
 * by `.filter(() => true)`; fails at `expect(call.arguments).toEqual({ text: "I work at Borealis." })`.
 * Row PLUGIN-DUMPS: `content: pyJsonDumps(...)` replaced by `content: JSON.stringify(...)`; fails at
 * `expect(record!.toolCalls[0]!.result).toBe(DUMPED)`.
 */
test(
  "a scripted call to recall_remember reaches the child with the model's arguments",
  async () => {
    const DUMPED = '{"memo_id": "m1.md", "note": "caf\\u00e9 \\u2713"}';
    const recall = fakeRecall({ tools: { recall_search: [ANSWER], recall_remember: [{ memo_id: "m1.md", note: "café ✓" }] } });
    const [record] = await drive(
      recall,
      [{ toolCalls: [{ name: "recall_remember", arguments: { text: "I work at Borealis.", supersedes: "", kind: null } }] }, { text: "stored" }],
      [["Remember that I work at Borealis."]],
    );
    const call = recall.fake.events().find((e) => e.event === "call" && e.tool === "recall_remember")!;
    expect<unknown>(call.arguments).toEqual({ text: "I work at Borealis." });
    expect(record!.toolCalls[0]!.result).toBe(DUMPED);
  },
  TIMEOUT,
);

/**
 * Invariant: with no store at the configured path the plugin registers nothing, starts no child, and
 * logs one line naming the setup command; OpenCode runs as if it were absent.
 * Red proof, row PLUGIN-NO-STORE: `if (lite && !existsSync(storePath(config.dsn)))` replaced by
 * `if (false)`; fails at `expect(recallTools).toEqual([])`.
 */
test(
  "with no store the plugin registers nothing and logs the setup command",
  async () => {
    const warn = spyOn(console, "warn");
    try {
      const recall = fakeRecall({ tools: { recall_search: [ANSWER] } }, { withStore: false });
      const [record] = await drive(recall, [{ text: "plain" }], [["Where do I work?"]]);
      const recallTools = toolsOf(record!.request!).filter((t) => t.function.name.startsWith("recall_"));
      expect<unknown>(recallTools).toEqual([]);
      expect<unknown>(users(record!.request!)).toEqual(["Where do I work?"]);
      expect(record!.request!.system.join("\n")).not.toContain(SYSTEM_SENTENCE);
      expect(recall.fake.events()).toEqual([]);
      const lines = warn.mock.calls.map((c) => String(c[0])).filter((l) => l.startsWith("re-call:"));
      expect(lines).toEqual([
        `re-call: inactive: no store at ${recall.store}; create it once with: recall --embedder fastembed setup --lite --memory "${recall.memos}" --store "${recall.store}" --no-register`,
      ]);
    } finally {
      warn.mockRestore();
    }
  },
  TIMEOUT,
);

/** A variable the fake child was started with ("" when no child started); Windows upper-cases the key. */
function childVar(fake: Fake, name: string): string {
  const env = fake.events().find((e) => e.event === "start")?.env ?? {};
  return Object.entries(env).find(([k]) => k.toUpperCase() === name)?.[1] ?? "";
}
const childDsn = (fake: Fake) => childVar(fake, "RECALL_SERVING_DSN");
/** Where the child indexes memos and where it writes its own: they must follow the store. */
const childMemos = (fake: Fake) => [childVar(fake, "RECALL_INDEX_ROOT"), childVar(fake, "RECALL_ASSISTANT_MEMOS")];

/** A store file in `dir`, and options with no `dsn` and no `memo_folder`, so the defaults apply. */
function defaultStore(recall: Recall, dir: string): Recall {
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "memory.db"), "");
  return { ...recall, options: { recall_command: recall.options.recall_command } };
}

const lite = (file: string) => "sqlite:///" + file.replaceAll("\\", "/");

/**
 * Invariant: with no `dsn`, the child gets the shared per-user store (owner, 2026-11-13, "One shared Lite
 * store"), while its working directory and `server.log` stay in OpenCode's own folder, so two agents on
 * one store never write one log. The child indexes the whole shared memo root and writes only to its
 * `opencode/` subfolder, so every agent sees the others' memos.
 * Red proof (2026-11-14, local Windows), row PLUGIN-MEMOS-FOLLOW-STORE: in src/config.ts, `: join(storeDir,
 * "memos"),` replaced by `: join(dataDir, "memos"),`; fails at `expect(childMemos(recall.fake)).toEqual(`.
 * Row PLUGIN-SHARED-STORE: `loadConfig(ctx.options, dataDir,
 * sharedStoreDir())` replaced by `loadConfig(ctx.options, dataDir)`; fails at
 * `expect(childDsn(recall.fake)).toBe(lite(join(shared, "memory.db")));`. Row PLUGIN-LOG-PER-AGENT:
 * `stderrPath: join(dataDir, "server.log")` replaced by `stderrPath: join(storePath(config.dsn), "..",
 * "server.log")`; fails at `expect([existsSync(join(own, "server.log")), existsSync(join(shared,
 * "server.log"))]).toEqual([true, false]);`.
 */
test(
  "with no dsn the child gets the shared store and server.log stays in OpenCode's folder",
  async () => {
    const shared = join(dataRoot, "re-call");
    const own = join(dataRoot, "opencode", "re-call");
    const recall = defaultStore(fakeRecall({ tools: { recall_search: [ANSWER] } }, { withStore: false }), shared);
    await drive(recall, [{ text: "done" }], [["Where do I work?"]]);
    expect(childDsn(recall.fake)).toBe(lite(join(shared, "memory.db")));
    expect(childMemos(recall.fake)).toEqual([join(shared, "memos"), join(shared, "memos", "opencode")]);
    expect([existsSync(join(own, "server.log")), existsSync(join(shared, "server.log"))]).toEqual([true, false]);
  },
  TIMEOUT,
);

/**
 * Invariant: with no `dsn` and a store only in OpenCode's folder (the default before 2026-11-13), the
 * child keeps that store, and the plugin logs the one line `loadConfig` composes for the move.
 * Red proof (2026-11-14, local Windows), row PLUGIN-OLD-STORE-NOTICE: `if (config.notice)
 * console.warn(config.notice);` removed from src/plugin.ts; fails at `expect(lines).toEqual([`.
 */
test(
  "an older OpenCode store is kept and the plugin logs the move once",
  async () => {
    const warn = spyOn(console, "warn");
    try {
      const shared = join(dataRoot, "re-call");
      const own = join(dataRoot, "opencode", "re-call");
      const recall = defaultStore(fakeRecall({ tools: { recall_search: [ANSWER] } }, { withStore: false }), own);
      await drive(recall, [{ text: "done" }], [["Where do I work?"]]);
      expect(childDsn(recall.fake)).toBe(lite(join(own, "memory.db")));
      expect(childMemos(recall.fake)).toEqual([join(own, "memos"), join(own, "memos", "opencode")]);
      const lines = warn.mock.calls.map((c) => String(c[0])).filter((l) => l.startsWith("re-call:"));
      expect(lines).toEqual([
        `re-call: using the older store ${join(own, "memory.db")}, not the shared one ${join(shared, "memory.db")}; ` +
          `to share it, close every agent, then move memory.db, memory.db-wal, memory.db-shm and the memos folder from ${own} to ${shared}`,
      ]);
    } finally {
      warn.mockRestore();
    }
  },
  TIMEOUT,
);

/**
 * Invariant: an expected failure never makes the plugin's setup throw. OpenCode 2.2.7 skips a plugin whose
 * setup throws and the run still succeeds, so a throw would look like a run without memory, with only an
 * `opencode warn: failed to load plugin` line to show for it. Each case must instead take the plugin's
 * own logged path (`re-call: inactive: ...`) and offer no RE-call tool.
 * Red proof: against the implementation at commit 7d1890b (before this test), the unknown option case
 * failed at `expect(loadFailures).toEqual([])` (`loadConfig` threw). Row PLUGIN-SETUP-NO-THROW: the
 * server's start failure rethrown instead of logged (`await release();` followed by `throw error;`);
 * fails at the same `expect`. Row PLUGIN-CONFIG-NO-THROW: `loadConfig`'s error rethrown instead of logged;
 * fails at the same `expect`.
 */
test(
  "an expected failure is logged and never fails the plugin's setup",
  async () => {
    const error = spyOn(console, "error");
    const warn = spyOn(console, "warn");
    try {
      const exiting = fakeRecall({ tools: { recall_search: [ANSWER] } });
      const quits = join(dataRoot, process.platform === "win32" ? "quits.cmd" : "quits.sh");
      writeFileSync(quits, process.platform === "win32" ? "@exit /b 3\r\n" : "#!/bin/sh\nexit 3\n", { mode: 0o755 });
      const cases: [string, Recall][] = [
        ["no store", fakeRecall({}, { withStore: false })],
        ["missing recall-mcp", { ...exiting, options: { ...exiting.options, recall_command: join(dataRoot, "absent", "recall-mcp.exe") } }],
        ["a child that exits at once", { ...exiting, options: { ...exiting.options, recall_command: quits } }],
        ["an unknown option", { ...exiting, options: { ...exiting.options, write_tool: false } }],
      ];
      for (const [label, recall] of cases) {
        error.mockClear();
        warn.mockClear();
        const [record] = await drive(recall, [{ text: "plain" }], [["Where do I work?"]]);
        const loadFailures = error.mock.calls.map((c) => String(c[0])).filter((l) => /^opencode \w+: .*failed to load plugin/.test(l));
        expect([label, loadFailures]).toEqual([label, []]);
        expect([label, warn.mock.calls.some((c) => String(c[0]).startsWith("re-call: inactive: "))]).toEqual([label, true]);
        expect([label, toolsOf(record!.request!).filter((t) => t.function.name.startsWith("recall_"))]).toEqual([label, []]);
      }
    } finally {
      error.mockRestore();
      warn.mockRestore();
    }
  },
  TIMEOUT,
);

/**
 * Invariant: the end to end tests' precondition (`tests/e2e/apparatus.ts` `assertPluginActive`) passes when the
 * plugin was active in every request, and fails as an apparatus failure ("plugin inactive"), not at a block
 * check, when it was not: here the plugin is deactivated by giving it no store.
 * Red proof, row APPARATUS-PLUGIN-ACTIVE: `if (problems.length > 0) {` replaced by `if (false) {` in
 * tests/e2e/apparatus.ts (the check never fires); fails at
 * `expect(() => assertPluginActive(inactive)).toThrow("apparatus failure: plugin inactive")`.
 */
test(
  "an inactive plugin is caught as an apparatus failure before any block check",
  async () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    try {
      const active = await drive(fakeRecall({ tools: { recall_search: [ANSWER] } }), [{ text: "a" }], [["Where do I work?"]]);
      const inactive = await drive(fakeRecall({}, { withStore: false }), [{ text: "b" }], [["Where do I work?"]]);
      expect(() => assertPluginActive(active)).not.toThrow();
      expect(() => assertPluginActive(inactive)).toThrow("apparatus failure: plugin inactive");
      expect(() => assertPluginActive([])).toThrow("apparatus failure: plugin inactive");
    } finally {
      warn.mockRestore();
    }
  },
  TIMEOUT,
);

/**
 * Invariant (Review Focus 1): the child dying mid-session never hangs the session: the call that
 * killed it comes back as an error the model reads, the next turn's block says memory could not
 * answer, and the session goes on to its answer.
 * Red proof, row PLUGIN-DEAD-CHILD: the prompt hook's search made to treat a failed call as no answer
 * (`.then((r) => (r.ok ? r : { ok: true, json: null }))`, Hermes's rule for a transport failure); fails at
 * `expect(users(next!.request!).at(-1)).toContain("RE-call memory could not answer this turn:")`.
 */
test(
  "the child dying mid-session gives the unavailable block and the session continues",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER], recall_remember: [{ memo_id: "m.md" }] }, exit_on: "recall_remember" });
    const [first, next] = await drive(
      recall,
      [{ toolCalls: [{ name: "recall_remember", arguments: { text: "x" } }] }, { text: "a1" }, { text: "a2" }],
      [["Remember x.", "Where do I work?"]],
    );
    expect(JSON.parse(first!.toolCalls[0]!.result as string)).toHaveProperty("error");
    expect(first!.answer).toBe("a1");
    expect(users(next!.request!).at(-1)).toContain("RE-call memory could not answer this turn:");
    expect(next!.answer).toBe("a2");
    expect(next!.error).toBeUndefined();
  },
  TIMEOUT,
);

/**
 * Invariant: a prefetch search is abandoned at Hermes's deadline (recall-hermes `PREFETCH_DEADLINE_S`,
 * 6 s), not the 60 s a tool call gets, so a slow server costs a turn the unavailable line, not a minute.
 * Red proof, row PLUGIN-PREFETCH-DEADLINE: `{ query: q, k: config.k }, PREFETCH_DEADLINE_MS)` replaced by
 * `{ query: q, k: config.k })` (the 60 s default); fails at
 * `expect(users(record!.request!)[0]).toContain("RE-call memory could not answer this turn:")`.
 */
test(
  "a slow prefetch search gives the unavailable block at Hermes's deadline",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER] }, delay: { recall_search: 9 } });
    const [record] = await drive(recall, [{ text: "done" }], [["Where do I work?"]]);
    expect(users(record!.request!)[0]).toContain("RE-call memory could not answer this turn:");
    expect(record!.answer).toBe("done");
  },
  TIMEOUT,
);

/**
 * Invariant: closing OpenCode closes the child: its input is closed and it exits.
 * Red proof, row PLUGIN-DISPOSE: `return release;` replaced by `return;`; fails at
 * `expect(recall.fake.events().at(-1)!.event).toBe("exit")`.
 */
test(
  "disposing the plugin closes the child",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER] } });
    await drive(recall, [{ text: "done" }], [["Where do I work?"]]);
    await Bun.sleep(500); // the fake writes its exit line after its input closes
    expect(recall.fake.events().at(-1)!.event).toBe("exit");
  },
  TIMEOUT,
);

// -- cases the driver cannot set up: an untitled session, two sessions at once, a subagent -------------

type Host = Awaited<ReturnType<typeof OpenCode.create>>;

async function host(recall: Recall, script: ScriptedTurn[], extra: Record<string, unknown> = {}) {
  const work = tempDir("recall-oc-host-");
  dirs.push(work);
  const project = join(work, "project");
  mkdirSync(project);
  const model = await startFakeModel(script);
  const config = {
    model: "harness/scripted",
    share: "disabled",
    update: "disable",
    plugins: [{ package: PLUGIN, options: recall.options }],
    providers: {
      harness: {
        name: "Harness",
        package: "@opencode/ai/providers/openai-compatible",
        settings: { baseURL: model.baseURL },
        models: { scripted: { name: "scripted", capabilities: { tools: true, input: ["text"], output: ["text"] }, limit: { context: 65536, output: 4096 } } },
      },
    },
    ...extra,
  };
  const oc = await OpenCode.create({
    database: { path: join(work, "opencode.db") },
    config: { content: JSON.stringify(config), project: false, claude: false, directory: join(work, "config") },
    models: { fetch: false },
    log: HOST_LOG,
  });
  return { oc, model, project, close: async () => (await oc.close(), await model.stop()) };
}

async function finish(oc: Host, sessionID: string, messageID: string): Promise<void> {
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    await oc.experimental.session.wait({ sessionID });
    const context = await oc.sessions.context({ sessionID });
    const at = context.findIndex((m) => m.id === messageID);
    if (at >= 0 && context.slice(at + 1).some((m) => m.type === "idle")) return;
    await Bun.sleep(50);
  }
  throw new Error(`turn ${messageID} did not finish`);
}

type Body = { messages: { role: string; content: unknown }[]; tools?: Tool[] };
const userTexts = (b: Body) => b.messages.filter((m) => m.role === "user").map((m) => (typeof m.content === "string" ? m.content : JSON.stringify(m.content)));

/**
 * Invariant: the prompt edit reaches OpenCode's title request too (V2-FACTS step 3); the `title` hook
 * takes the block out, so no memory text reaches a title model, while the turn itself keeps it.
 * Red proof, row PLUGIN-TITLE: `text: stripBlocks(part.text)` replaced by `text: part.text`; fails at
 * `expect(JSON.stringify(title)).not.toContain("<memory-context>")`.
 */
test(
  "the title request carries no memory-context block",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER] } });
    const h = await host(recall, [{ text: "Work question" }, { text: "done" }, { text: "spare" }]);
    try {
      const session = await h.oc.sessions.create({ location: { directory: h.project } });
      const user = await h.oc.sessions.prompt({ sessionID: session.id, text: "Where do I work?" });
      await finish(h.oc, session.id, user.id);
      const bodies = h.model.requests as Body[];
      const title = bodies.filter((b) => !b.tools?.length);
      const turn = bodies.filter((b) => b.tools?.length);
      expect(title.length).toBe(1);
      expect(JSON.stringify(title)).toContain("Where do I work?");
      expect(JSON.stringify(title)).not.toContain("<memory-context>");
      expect(userTexts(turn[0]!)).toEqual(["Where do I work?\n\n" + BLOCK]);
    } finally {
      await h.close();
    }
  },
  TIMEOUT,
);

/**
 * Invariant (Review Focus 3): two sessions prompted at once share one child; their searches are
 * serialized, and each session's block renders the answer to its own prompt.
 * Red proof, row PLUGIN-CONCURRENT: the prompt hook composes from a block kept in one shared slot
 * (`globalThis.__last = injection`, then a 1 s wait, then `globalThis.__last`), so a session can take the
 * other session's block; fails at `expect(userTexts(body)[0]).toContain(...)` or the `not.toContain`
 * after it.
 */
test(
  "two sessions at once each get the block for their own prompt",
  async () => {
    const answer = (who: string) => ({ abstained: false, evidence: { citable: ["c1"] }, hits: [{ memo_id: `${who}.md`, chunk_id: "c1", verdict: "ok", text: `About ${who}` }] });
    const recall = fakeRecall({ tools: { recall_search: [answer("first"), answer("second")] }, delay: { recall_search: 0.5 } });
    const h = await host(recall, [{ text: "x" }, { text: "y" }]);
    try {
      const a = await h.oc.sessions.create({ title: "a", location: { directory: h.project } });
      const b = await h.oc.sessions.create({ title: "b", location: { directory: h.project } });
      const [ua, ub] = await Promise.all([
        h.oc.sessions.prompt({ sessionID: a.id, text: "Question from session A" }),
        h.oc.sessions.prompt({ sessionID: b.id, text: "Question from session B" }),
      ]);
      await Promise.all([finish(h.oc, a.id, ua.id), finish(h.oc, b.id, ub.id)]);
      const searched = searches(recall.fake).map((e) => e.arguments!.query as string);
      expect(searched.sort()).toEqual(["Question from session A", "Question from session B"]);
      // The fake answers in arrival order: the first query to arrive gets the "first" memo.
      const order = searches(recall.fake).map((e) => e.arguments!.query as string);
      const bodies = h.model.requests as Body[];
      for (const [i, query] of order.entries()) {
        const body = bodies.find((x) => userTexts(x)[0]!.startsWith(query))!;
        expect(userTexts(body)[0]).toContain(`memo ${i === 0 ? "first" : "second"}.md: About ${i === 0 ? "first" : "second"}`);
        expect(userTexts(body)[0]).not.toContain(`memo ${i === 0 ? "second" : "first"}.md`);
      }
      const ends = recall.fake.events().filter((e) => e.event === "call");
      expect(ends.length).toBe(2);
    } finally {
      await h.close();
    }
  },
  TIMEOUT,
);

/**
 * Invariant: a subagent's request offers no write tool (Hermes offers them to the primary agent only)
 * while the primary agent's request keeps them; a write the subagent makes anyway never reaches the child.
 * Red proof, row PLUGIN-SUBAGENT: `if (!primary) for` replaced by `if (false) for`; fails at
 * `expect(subWrites).toEqual([])`.
 */
test(
  "a subagent's request has no write tools",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER], recall_remember: [{ memo_id: "m.md" }] } });
    const h = await host(
      recall,
      [
        { toolCalls: [{ name: "subagent", arguments: { agent: "general", description: "child", prompt: "CHILD_TASK" } }] },
        { toolCalls: [{ name: "recall_remember", arguments: { text: "child tried" } }] },
        { text: "child done" },
        { text: "parent done" },
      ],
      { permissions: [{ action: "*", resource: "*", effect: "allow" }] },
    );
    try {
      const session = await h.oc.sessions.create({ title: "sub", location: { directory: h.project } });
      const user = await h.oc.sessions.prompt({ sessionID: session.id, text: "Spawn a subagent please" });
      await finish(h.oc, session.id, user.id);
      const bodies = h.model.requests as Body[];
      const names = (b: Body) => (b.tools ?? []).map((t) => t.function.name);
      const writes = ["recall_apply_fact", "recall_remember", "recall_forget_memo", ...EXTRA.filter((e) => e.write).map((e) => e.name)];
      const primary = bodies[0]!;
      const sub = bodies.filter((b) => userTexts(b).some((t) => t.includes("CHILD_TASK")));
      expect(names(primary)).toEqual(expect.arrayContaining(writes));
      expect(sub.length).toBeGreaterThan(0);
      const subWrites = sub.flatMap(names).filter((n) => writes.includes(n));
      expect(subWrites).toEqual([]);
      expect(names(sub[0]!)).toContain("recall_search");
      expect(recall.fake.events().filter((e) => e.tool === "recall_remember")).toEqual([]);
    } finally {
      await h.close();
    }
  },
  TIMEOUT,
);

// -- audit (Task 9) -------------------------------------------------------------------------------

const WRITES = ["recall_apply_fact", "recall_remember", "recall_forget_memo", ...EXTRA.filter((e) => e.write).map((e) => e.name)];
const toolNames = (b: Body) => (b.tools ?? []).map((t) => t.function.name);

async function untilTrue(done: () => boolean, ms = 60_000): Promise<void> {
  for (const end = Date.now() + ms; !done() && Date.now() < end; ) await Bun.sleep(50);
}

/**
 * Invariant (audit H-01, H-17): with `prefetch: false` the plugin searches nothing, appends no block, and
 * the system prompt does not get the sentence that promises a block before each turn; the tools stay.
 * Red proof, row PLUGIN-SYSTEM-PREFETCH-OFF: `if (config.prefetch) event.system.push(` replaced by
 * `event.system.push(` (the sentence with no block); fails at
 * `expect(record!.request!.system.join("\n")).not.toContain(SYSTEM_SENTENCE)`. Row PLUGIN-PREFETCH-OFF:
 * `  if (config.prefetch) {` replaced by `  if (true) {` (the option ignored); fails at
 * `expect(searches(recall.fake)).toEqual([])`.
 */
test(
  "with prefetch off there is no search, no block and no system sentence",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER] } });
    const [record] = await drive({ ...recall, options: { ...recall.options, prefetch: false } }, [{ text: "done" }], [["Where do I work?"]]);
    expect(searches(recall.fake)).toEqual([]);
    expect<unknown>(users(record!.request!)).toEqual(["Where do I work?"]);
    expect(record!.request!.system.join("\n")).not.toContain(SYSTEM_SENTENCE);
    expect(toolsOf(record!.request!).map((t) => t.function.name)).toContain("recall_search");
  },
  TIMEOUT,
);

/**
 * Invariant (audit H-36): a session a `subagent: true` command creates is a delegated run, so its
 * requests have no write tool even though its agent is the primary `build` (OpenCode 2.2.7 runs such a
 * command with the parent's agent in a child session, skill-5vv535z9.js).
 * Red proof, row PLUGIN-CHILD-SESSION: `!session.parentID && ` removed from the gate in src/plugin.ts;
 * fails at `expect(child().flatMap(toolNames).filter((n) => WRITES.includes(n))).toEqual([])`.
 */
test(
  "a subagent command's session has no write tools, whatever its agent",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER], recall_remember: [{ memo_id: "m.md" }] } });
    const h = await host(recall, [{ text: "child done" }, { text: "spare" }], {
      permissions: [{ action: "*", resource: "*", effect: "allow" }],
      commands: { delegate: { template: "CMD_CHILD", subagent: true } },
    });
    try {
      const session = await h.oc.sessions.create({ title: "cmd", location: { directory: h.project } });
      await h.oc.sessions.command({ sessionID: session.id, name: "delegate", text: "" });
      const child = () => (h.model.requests as Body[]).filter((b) => userTexts(b).some((t) => t.includes("CMD_CHILD")));
      await untilTrue(() => child().length > 0);
      expect(child().length).toBeGreaterThan(0);
      expect(toolNames(child()[0]!)).toContain("recall_search");
      expect(child().flatMap(toolNames).filter((n) => WRITES.includes(n))).toEqual([]);
    } finally {
      await h.close();
    }
  },
  TIMEOUT,
);

/**
 * Invariant (audit H-37, owner item K2): an agent of mode `all` driving a root session is the primary
 * run, so its request keeps the write tools, as Hermes gives them to `agent_context == "primary"`.
 * Red proof, row PLUGIN-MODE-ALL: `(mode === "primary" || mode === "all")` replaced by
 * `mode === "primary"` in src/plugin.ts; fails at `expect(toolNames(turn[0]!)).toEqual(expect.arrayContaining(WRITES))`.
 */
test(
  "an agent of mode all driving a root session keeps the write tools",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER], recall_remember: [{ memo_id: "m.md" }] } });
    const h = await host(recall, [{ text: "done" }, { text: "spare" }], { agents: { coder: { mode: "all", description: "A coding agent" } } });
    try {
      const session = await h.oc.sessions.create({ title: "all", agent: "coder", location: { directory: h.project } });
      const user = await h.oc.sessions.prompt({ sessionID: session.id, text: "Where do I work?" });
      await finish(h.oc, session.id, user.id);
      const turn = (h.model.requests as Body[]).filter((b) => b.tools?.length);
      expect(turn.length).toBeGreaterThan(0);
      expect(toolNames(turn[0]!)).toEqual(expect.arrayContaining(WRITES));
    } finally {
      await h.close();
    }
  },
  TIMEOUT,
);

/**
 * Invariant (audit H-49): interrupting a turn while a RE-call tool call is in flight aborts it, and the
 * server is told to cancel it (the write must not run on unannounced, holding the client).
 * Red proof, row PLUGIN-ABORT: `CALL_TIMEOUT_MS, context.signal)` replaced by `CALL_TIMEOUT_MS)` in
 * src/plugin.ts (the signal ignored); fails at `expect(cancelled().length).toBe(1)`.
 */
test(
  "interrupting a turn cancels its RE-call call on the server",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER], recall_remember: [{ memo_id: "m.md" }] }, delay: { recall_remember: 3 } });
    const h = await host(recall, [{ toolCalls: [{ name: "recall_remember", arguments: { text: "slow write" } }] }, { text: "after" }, { text: "spare" }], {
      permissions: [{ action: "*", resource: "*", effect: "allow" }],
    });
    try {
      const session = await h.oc.sessions.create({ title: "abort", location: { directory: h.project } });
      await h.oc.sessions.prompt({ sessionID: session.id, text: "Remember that I like tea" });
      await untilTrue(() => recall.fake.events().some((e) => e.event === "call" && e.tool === "recall_remember"));
      await h.oc.sessions.interrupt({ sessionID: session.id });
      const cancelled = () => recall.fake.events().filter((e) => e.event === "cancelled");
      await untilTrue(() => cancelled().length > 0, 10_000);
      expect(cancelled().length).toBe(1);
    } finally {
      await h.close();
    }
  },
  TIMEOUT,
);

// -- audit (deferred pass) ------------------------------------------------------------------------

/** The plugin's `context` hook, run once on a request offering every write and `recall_search`, with
 * `agent.get` and `session.get` answered by the given functions; the names the hook left. OpenCode cannot
 * be made to fail its own lookups, so the context here is a stand-in that records the hooks. */
async function gateWith(agentGet: () => Promise<unknown>, sessionGet: () => Promise<unknown>): Promise<string[]> {
  const recall = fakeRecall({ tools: { recall_search: [ANSWER] } });
  const hooks: Record<string, (event: unknown) => unknown> = {};
  const ctx = {
    options: recall.options,
    tool: { transform: async () => {} },
    session: { hook: async (name: string, fn: (event: unknown) => unknown) => void (hooks[name] = fn), get: sessionGet },
    agent: { get: agentGet },
  } as unknown as Plugin.Context;
  const dispose = await setup(ctx);
  try {
    const event = { agent: "build", sessionID: "ses_root", system: [] as unknown[], tools: Object.fromEntries([...WRITES, "recall_search"].map((n) => [n, {}])) };
    await hooks.context!(event);
    return Object.keys(event.tools);
  } finally {
    if (dispose) await dispose();
  }
}

/**
 * Invariant (audit H-38): a lookup the gate cannot complete counts as not primary, the safe side for a
 * write: when `agent.get` or `session.get` fails, the request keeps `recall_search` and loses every
 * write. The control (both lookups answer a primary agent at a root session) keeps the writes, so the
 * stand-in context drives the real gate.
 * Red proof, row PLUGIN-AGENT-LOOKUP: in src/plugin.ts, `.then((a) => a.data.mode, () => undefined)`
 * replaced by `.then((a) => a.data.mode, () => "primary")`; fails at
 * `expect(agentFails.filter((n) => WRITES.includes(n))).toEqual([])`. Row PLUGIN-SESSION-LOOKUP:
 * `.then((s) => s, () => undefined)` replaced by `.then((s) => s, () => ({}))`; fails at
 * `expect(sessionFails.filter((n) => WRITES.includes(n))).toEqual([])`.
 */
test(
  "a failed agent or session lookup removes the write tools",
  async () => {
    const primary = async () => ({ data: { mode: "primary" } });
    const root = async () => ({ id: "ses_root" });
    const failed = async () => {
      throw new Error("lookup failed");
    };
    const control = await gateWith(primary, root);
    expect(control).toEqual(expect.arrayContaining([...WRITES, "recall_search"]));
    const agentFails = await gateWith(failed, root);
    expect(agentFails).toContain("recall_search");
    expect(agentFails.filter((n) => WRITES.includes(n))).toEqual([]);
    const sessionFails = await gateWith(primary, failed);
    expect(sessionFails).toContain("recall_search");
    expect(sessionFails.filter((n) => WRITES.includes(n))).toEqual([]);
  },
  TIMEOUT,
);

/**
 * Invariant (audit H-43): a live run (the real end to end, the live surface test) refuses an unset or
 * empty RECALL_OPENCODE_BIN instead of taking PATH's `recall`; a set one is used as given. Both live
 * files take BIN from `recallBin` when they are on, before any setup.
 * Red proof, row E2E-BIN-REQUIRED: in tests/e2e/apparatus.ts `recallBin`, `if (!bin) throw` replaced by
 * `if (false) throw`; fails at `expect(refused({})).toContain("RECALL_OPENCODE_BIN is unset")`.
 */
test("a live run refuses an unset RECALL_OPENCODE_BIN", () => {
  const refused = (env: Record<string, string | undefined>) => {
    try {
      return recallBin(env);
    } catch (error) {
      return String(error); // caught, so a missing refusal fails this expect, not a thrown error
    }
  };
  expect(refused({})).toContain("RECALL_OPENCODE_BIN is unset");
  expect(refused({ RECALL_OPENCODE_BIN: "" })).toContain("RECALL_OPENCODE_BIN is unset");
  expect(recallBin({ RECALL_OPENCODE_BIN: "C:/recall/Scripts" })).toBe("C:/recall/Scripts");
});

/**
 * Invariant (audit DEF-39 / H-39): compaction and generate requests, which OpenCode 2.2.7 builds through
 * their own hooks rather than `context`, get the same write gate: in a session whose agent is a subagent
 * they keep `recall_search` and offer no write tool, while a primary agent's root session keeps the writes
 * in its compaction request (the control). Before the fix both listed every write (probe 2026-11-11).
 * Red proof, row PLUGIN-GATE-COMPACTION: `await ctx.session.hook("compaction", gate);` removed from
 * src/plugin.ts; fails at `expect(writes(compaction(sub))).toEqual([])`. Row PLUGIN-GATE-GENERATE:
 * `await ctx.session.hook("generate", gate);` removed; fails at `expect(writes(generated)).toEqual([])`.
 */
test(
  "compaction and generate requests get the write gate too",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER], recall_remember: [{ memo_id: "m.md" }] } });
    const h = await host(recall, Array.from({ length: 12 }, (_, i) => ({ text: `answer ${i}` })), {
      agents: { helper: { mode: "subagent", description: "A helper" } },
    });
    const requests = () => h.model.requests as Body[];
    const writes = (bodies: Body[]) => bodies.flatMap(toolNames).filter((n) => WRITES.includes(n));
    const compactionOf = async (agent?: string) => {
      const session = await h.oc.sessions.create({ title: "gate", ...(agent ? { agent } : {}), location: { directory: h.project } });
      const user = await h.oc.sessions.prompt({ sessionID: session.id, text: "Where do I work?" });
      await finish(h.oc, session.id, user.id);
      const before = requests().length;
      await h.oc.sessions.compact({ sessionID: session.id });
      await untilTrue(() => requests().length > before, 20_000);
      await h.oc.experimental.session.wait({ sessionID: session.id });
      return { session, bodies: requests().slice(before) };
    };
    try {
      const sub = await compactionOf("helper");
      const compaction = (c: typeof sub) => c.bodies;
      expect(compaction(sub).length).toBeGreaterThan(0);
      expect(compaction(sub).flatMap(toolNames)).toContain("recall_search");
      expect(writes(compaction(sub))).toEqual([]);
      const before = requests().length;
      await h.oc.sessions.generate({ sessionID: sub.session.id, prompt: "Say hi" });
      const generated = requests().slice(before);
      expect(generated.length).toBeGreaterThan(0);
      expect(generated.flatMap(toolNames)).toContain("recall_search");
      expect(writes(generated)).toEqual([]);
      const root = await compactionOf();
      expect(writes(compaction(root))).toEqual(expect.arrayContaining(WRITES));
    } finally {
      await h.close();
    }
  },
  TIMEOUT,
);

/**
 * Invariant (audit DEF-40 / H-40, owner 2026-11-11: share one child): a host serving two directories sets
 * the plugin up twice (V2-FACTS fact 7), and both instances use one `recall-mcp` child: one start, and each
 * directory's turn still gets its search.
 * Red proof, row PLUGIN-SHARED-CHILD: in src/plugin.ts `setup`, `shared = children.get(key);` replaced by
 * `shared = undefined;` (every instance starts its own); fails at `expect(starts().length).toBe(1)`.
 */
test(
  "sessions in two directories of one host share one child",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER] } });
    const h = await host(recall, [{ text: "a" }, { text: "b" }, { text: "spare" }, { text: "spare" }]);
    const starts = () => recall.fake.events().filter((e) => e.event === "start");
    try {
      const other = join(h.project, "..", "other");
      mkdirSync(other);
      for (const directory of [h.project, other]) {
        const session = await h.oc.sessions.create({ title: "loc", location: { directory } });
        const user = await h.oc.sessions.prompt({ sessionID: session.id, text: "Where do I work?" });
        await finish(h.oc, session.id, user.id);
      }
      expect(searches(recall.fake).length).toBe(2);
      expect(starts().length).toBe(1);
    } finally {
      await h.close();
    }
  },
  TIMEOUT,
);

/** The plugin set up on a stand-in context with these options; its dispose. */
async function setUp(options: Record<string, unknown>): Promise<() => Promise<void>> {
  const ctx = {
    options,
    tool: { transform: async () => {} },
    session: { hook: async () => {}, get: async () => ({ id: "ses_root" }) },
    agent: { get: async () => ({ data: { mode: "primary" } }) },
  } as unknown as Plugin.Context;
  const dispose = await setup(ctx);
  if (!dispose) throw new Error("apparatus failure: the plugin did not activate");
  return dispose;
}

/**
 * Invariant (audit DEF-40): instances with the same options share the child, which stays up until the last
 * of them is disposed; instances whose options give the child another launch (here `write_tools: false`,
 * which changes RECALL_MCP_TOOLS) get their own.
 * Red proof, row PLUGIN-SHARED-RELEASE: in src/plugin.ts `release`, `if (child.users > 0) return;` replaced
 * by `void child.users;` (the first dispose closes the shared child); fails at
 * `expect(exits()).toBe(0)`.
 */
test(
  "a shared child closes when its last plugin instance is disposed",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER] } });
    const events = (name: string) => recall.fake.events().filter((e) => e.event === name).length;
    const exits = () => events("exit");
    const first = await setUp(recall.options);
    const second = await setUp(recall.options);
    const narrow = await setUp({ ...recall.options, write_tools: false });
    expect(events("start")).toBe(2);
    await first();
    await Bun.sleep(500); // the fake writes its exit line after its input closes
    expect(exits()).toBe(0);
    await narrow();
    await Bun.sleep(500);
    expect(exits()).toBe(1);
    await second();
    await Bun.sleep(500);
    expect(exits()).toBe(2);
  },
  TIMEOUT,
);

/**
 * Invariant (audit DEF-40, planning review 2026-11-11): only an identical launch shares a child. Two plugin
 * instances with the same `recall_command` on two different stores (dsn and memo folder) get two children,
 * each started on its own store, so neither reads or writes the other's memory. OpenCode 2.2.7 gives every
 * directory of one host the same plugin options, so the two configurations are set up on stand-in contexts.
 * Red proof, row PLUGIN-SHARED-KEY: in src/plugin.ts `setup`, `key = JSON.stringify(options);` replaced by
 * `key = JSON.stringify(options.argv);` (the executable alone); fails at `expect(dsns()).toEqual(...)`.
 */
test(
  "plugin instances on two different stores never share a child",
  async () => {
    const recall = fakeRecall({ tools: { recall_search: [ANSWER] } });
    const otherDir = tempDir("recall-oc-contract-other-");
    dirs.push(otherDir);
    mkdirSync(join(otherDir, "store"));
    const otherStore = join(otherDir, "store", "memory.db");
    writeFileSync(otherStore, "");
    const dsnOf = (store: string) => `sqlite:///${store.replaceAll("\\", "/")}`;
    const other = { ...recall.options, dsn: dsnOf(otherStore), memo_folder: join(otherDir, "memos") };
    const dsns = () =>
      recall.fake
        .events()
        .filter((e) => e.event === "start")
        .map((e) => Object.entries(e.env ?? {}).find(([k]) => k.toUpperCase() === "RECALL_SERVING_DSN")?.[1]);
    const first = await setUp(recall.options);
    const second = await setUp(other);
    try {
      expect(dsns()).toEqual([dsnOf(recall.store), dsnOf(otherStore)]);
    } finally {
      await first();
      await second();
    }
  },
  TIMEOUT,
);
