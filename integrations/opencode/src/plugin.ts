/** The OpenCode V2 plugin: one `recall-mcp` child, RE-call's tools under their own names, and the Hermes
 * prefetch block on every non-trivial user turn. A port of recall-hermes 86fdc528 `provider.py`, placed the
 * way docs/V2-FACTS.md measured OpenCode 2.2.7 (steps 2 to 5). */

import { existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

import type { Plugin } from "@opencode/plugin";

import { childEnv, resolveExecutable } from "./child";
import { type Config, isLiteDsn, loadConfig, sharedStoreDir, storePath } from "./config";
import { CALL_TIMEOUT_MS, McpClient, READY_TIMEOUT_MS, type ToolDef } from "./mcp_client";
import { compose, prefetchBlock, SYSTEM_SENTENCE } from "./prefetch";
import { EXTRA, selectTools, TRAINED_WRITES } from "./surface";

/** recall-hermes `PREFETCH_DEADLINE_S`: a prefetch search past it gives the turn the unavailable line. */
export const PREFETCH_DEADLINE_MS = 6_000;

/** OpenCode's own folder, beside its global config (`@opencode/util` `global-roots.js`:
 * `$XDG_CONFIG_HOME/opencode`, else `~/.config/opencode`): the child's working directory and its
 * `server.log`, which are per agent, so two agents never write one log. The store is shared
 * (`sharedStoreDir`); a store made here before 2026-11-13 is kept until it is moved. */
export function defaultDataDir(env: Record<string, string | undefined> = process.env): string {
  return join(env.XDG_CONFIG_HOME || join(homedir(), ".config"), "opencode", "re-call");
}

export function setupCommand(config: Config): string {
  return `recall --embedder fastembed setup --lite --memory "${config.memoFolder}" --store "${resolve(storePath(config.dsn))}" --no-register`;
}

/** Python's `json.dumps` with its defaults (`ensure_ascii`, separators `", "` and `": "`), which is what
 * Hermes's `handle_tool_call` returns, so the model reads a tool result in the bytes Hermes sends. A float
 * that JSON wrote with a fractional zero comes out without it (see src/render.ts), and an exponent keeps
 * JavaScript's spelling (`1e-7`, where Python writes `1e-07`). Three more differences come from
 * `JSON.parse`, before this function sees the value, so they cannot be fixed here (measured 2026-11-11
 * against `python -I -c "import json; print(json.dumps(...))"`):
 * - JavaScript writes a number in exponent form from 1e21 up and below 1e-6; Python writes a float so
 *   from 1e16 up and below 1e-4, so `1e+16` comes out as `10000000000000000`.
 * - An object's integer-like keys come first, in ascending order (`{"2": 2, "b": 1}` where Python keeps
 *   `{"b": 1, "2": 2}`).
 * - An integer beyond 2^53 is already rounded (`9007199254740993` comes out as `9007199254740992`). */
export function pyJsonDumps(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : value > 0 ? "Infinity" : value < 0 ? "-Infinity" : "NaN";
  if (typeof value === "string") {
    let out = '"';
    for (let i = 0; i < value.length; i += 1) {
      const ch = value[i]!;
      const code = value.charCodeAt(i);
      const named: Record<string, string> = { '"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f" };
      if (named[ch] !== undefined) out += named[ch];
      // `ensure_ascii` escapes everything outside 0x20 to 0x7e, DEL (0x7f) included.
      else if (code < 0x20 || code >= 0x7f) out += "\\u" + code.toString(16).padStart(4, "0");
      else out += ch;
    }
    return out + '"';
  }
  if (Array.isArray(value)) return "[" + value.map(pyJsonDumps).join(", ") + "]";
  return "{" + Object.entries(value as Record<string, unknown>).map(([k, v]) => `${pyJsonDumps(k)}: ${pyJsonDumps(v)}`).join(", ") + "}";
}

const BLOCK_OPEN = "\n\n<memory-context>\n[System note: The following is recalled memory context, NOT new user input.";
const BLOCK_CLOSE = "\n</memory-context>";

/** The text without any block `compose` appended. The title request wraps the stored user text in its own
 * markup, so the block is not at the end there; `sanitize_context` already took every fence tag out of the
 * block's body, so its first closing tag is its own. The search starts from the last opening marker,
 * which is the block's: one in the user's own text comes before it. A marker with no closing tag after
 * it is the user's, so the search goes on before it rather than stopping with a block still in. */
export function stripBlocks(text: string): string {
  for (let from = text.length; from >= 0; ) {
    const at = text.lastIndexOf(BLOCK_OPEN, from);
    if (at < 0) return text;
    const end = text.indexOf(BLOCK_CLOSE, at + BLOCK_OPEN.length);
    if (end < 0) {
      from = at - 1;
      continue;
    }
    text = text.slice(0, at) + text.slice(end + BLOCK_CLOSE.length);
    from = at;
  }
  return text;
}

/** recall-hermes `handle_tool_call`: arguments that are None or "" are dropped, and a search gets `k`. */
function callArguments(name: string, input: unknown, k: number): Record<string, unknown> {
  const args = Object.fromEntries(
    Object.entries(input && typeof input === "object" ? (input as Record<string, unknown>) : {}).filter(([, v]) => v !== null && v !== undefined && v !== ""),
  );
  if (name === "recall_search" && !("k" in args)) args.k = k;
  return args;
}

const isWrite = (name: string) => TRAINED_WRITES.has(name) || EXTRA.some((e) => e.name === name && e.write);

/** One `recall-mcp` child per process for one launch (executable, environment, data dir). OpenCode 2.2.7
 * sets the plugin up once per directory a host serves (V2-FACTS fact 7), and each instance would otherwise
 * start its own child, embedding model included, on the one per-user store. Owner, 2026-11-11: share it. */
type SharedChild = { client: McpClient; users: number; tools: Promise<ToolDef[]> };
// On the process, not the module: OpenCode loads the module once per directory too, so a module-level map
// would not be shared (measured 2026-11-11: with a module-level map two directories still started two
// children).
const children: Map<string, SharedChild> = ((globalThis as Record<symbol, unknown>)[Symbol.for("re-call-opencode.children")] ??=
  new Map<string, SharedChild>()) as Map<string, SharedChild>;

const inactive = (error: unknown) => console.warn(`re-call: inactive: ${error instanceof Error ? error.message : String(error)}`);

/** Every expected failure (a bad option, no store, no server, a server that does not start) is logged and
 * leaves the plugin inactive. OpenCode 2.2.7 skips a plugin whose setup throws and runs on, so a throw would
 * only look like a session without memory. */
export async function setup(ctx: Plugin.Context): Promise<(() => Promise<void>) | void> {
  const dataDir = defaultDataDir();
  let config: Config;
  try {
    config = loadConfig(ctx.options, dataDir, sharedStoreDir());
  } catch (error) {
    inactive(error);
    return;
  }
  if (config.notice) console.warn(config.notice);
  const lite = isLiteDsn(config.dsn);
  if (lite && !existsSync(storePath(config.dsn))) {
    console.warn(`re-call: inactive: no store at ${resolve(storePath(config.dsn))}; create it once with: ${setupCommand(config)}`);
    return;
  }

  let key = "";
  let shared: SharedChild | undefined;
  try {
    mkdirSync(dataDir, { recursive: true });
    // RECALL_ASSISTANT_MEMOS must exist before the child starts (Hermes: `assistant_folder.mkdir`).
    const env = childEnv(config, process.env);
    if (env.RECALL_ASSISTANT_MEMOS) mkdirSync(env.RECALL_ASSISTANT_MEMOS, { recursive: true });
    const options = {
      argv: [resolveExecutable(config.recallCommand, process.env.PATH ?? process.env.Path ?? "", process.cwd())],
      env,
      cwd: dataDir, // no project .env is in reach of the child
      stderrPath: join(dataDir, "server.log"),
    };
    key = JSON.stringify(options);
    shared = children.get(key);
    if (!shared) {
      const client = new McpClient(options);
      const started = Date.now();
      const tools = (async () => {
        await client.start();
        // The first turn must find a ready server (Hermes measured a lost first-turn memory without this wait).
        await client.ready(READY_TIMEOUT_MS);
        console.info(`re-call: server ready after ${((Date.now() - started) / 1000).toFixed(1)} s`);
        return client.listTools();
      })();
      shared = { client, users: 0, tools };
      children.set(key, shared);
    }
  } catch (error) {
    inactive(error);
    return;
  }
  const child = shared;
  child.users += 1;
  // The last instance to go closes the child; a failed start is forgotten, so a later instance starts afresh.
  const release = async () => {
    child.users -= 1;
    if (child.users > 0) return;
    if (children.get(key) === child) children.delete(key);
    await child.client.close();
  };
  const client = child.client;
  let tools;
  try {
    tools = selectTools(await child.tools, config, lite);
  } catch (error) {
    if (children.get(key) === child) children.delete(key);
    await release();
    console.warn(`re-call: inactive: the RE-call server did not start: ${error instanceof Error ? error.message : String(error)} (see ${join(dataDir, "server.log")})`);
    return;
  }

  await ctx.tool.transform((editor) => {
    for (const tool of tools) {
      editor.add({
        name: tool.name,
        description: tool.description,
        input: structuredClone(tool.inputSchema) as never,
        options: { codemode: false },
        execute: async (input, context) => {
          // An interrupted turn aborts the call: unsent, it is dropped; sent, the server is told to cancel it.
          const result = await client.call(tool.name, callArguments(tool.name, input, config.k), CALL_TIMEOUT_MS, context.signal);
          return { content: pyJsonDumps(result.ok ? result.json : { error: result.error }) };
        },
      });
    }
  });

  if (config.prefetch) {
    // The edit is the stored user input, so later requests replay the block as Hermes's `api_content`
    // sidecar does (V2-FACTS steps 3 and 5).
    await ctx.session.hook("prompt", async (event) => {
      const { injection } = await prefetchBlock((q) => client.call("recall_search", { query: q, k: config.k }, PREFETCH_DEADLINE_MS), event.prompt.text);
      event.prompt.text = compose(event.prompt.text, injection);
    });
    // The prompt edit also reaches OpenCode's title request; memory text never goes to a title model.
    await ctx.session.hook("title", (event) => {
      for (const message of event.messages) {
        if (message.role !== "user") continue;
        const content = message.content as unknown as { type: string; text?: string }[];
        content.forEach((part, i) => {
          if (part.type === "text" && typeof part.text === "string") content[i] = { ...part, text: stripBlocks(part.text) };
        });
      }
    });
  }

  // Hermes offers the write tools to `agent_context == "primary"` only: the run the user drives, not a
  // delegated one. Here that is a root session whose agent is not a subagent. A subagent's or a subtask
  // command's session has a parent, whatever its agent's mode; a fork has none (OpenCode 2.2.7 stores it
  // with `parent_id: null`). A lookup that fails is treated as not primary: the safe side for a write.
  const gate = async (event: Parameters<Parameters<typeof ctx.session.hook<"context">>[1]>[0]) => {
    const [mode, session] = await Promise.all([
      ctx.agent.get({ agentID: event.agent }).then((a) => a.data.mode, () => undefined),
      ctx.session.get({ sessionID: event.sessionID }).then((s) => s, () => undefined),
    ]);
    const primary = session !== undefined && !session.parentID && (mode === "primary" || mode === "all");
    if (!primary) for (const name of Object.keys(event.tools)) if (isWrite(name)) delete event.tools[name];
  };
  await ctx.session.hook("context", async (event) => {
    // The sentence promises a block before each turn, so it goes only where the block does (FT-4 adds both
    // under its prefetch switch).
    if (config.prefetch) event.system.push({ type: "text", text: SYSTEM_SENTENCE });
    await gate(event);
  });
  // Compaction and generate requests carry the session's tools through their own hooks, not `context`
  // (measured 2026-11-11, audit DEF-39: both listed every write in a subagent session).
  await ctx.session.hook("compaction", gate);
  await ctx.session.hook("generate", gate);

  return release;
}
