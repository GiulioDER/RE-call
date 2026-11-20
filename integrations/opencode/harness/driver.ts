// The headless driver: runs a scenario's sessions through an in-process OpenCode 2.2.7 host and
// writes one JSONL record per user turn. A record's `request` is the body the model received, as
// the model's HTTP endpoint captured it (the scripted model, or a recording proxy in front of
// Ollama); nothing in it is rebuilt from OpenCode's own state. The driver never scores.
import { appendFileSync, cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { OpenCode } from "@opencode/sdk";
import { stripCredentials } from "../tests/credentials";
import { startFakeModel } from "./fake_model";
import { splitInjection } from "../src/prefetch";
import { loadScenario, type Scenario } from "./scenario";

/** A plugin directory, or a directory with the options OpenCode passes to its `setup`. */
export type PluginSpec = string | { path: string; options?: Record<string, unknown> };

export type ModelRequest = {
  /** Contents of the leading system messages, in order. */
  system: unknown[];
  /** Every message after the leading system ones, verbatim. */
  messages: unknown[];
  tools: unknown;
  /** The rest of the body (model, stream, token limit and so on), verbatim. */
  params: Record<string, unknown>;
};

export type ToolCallRecord = {
  id: string;
  name: string;
  /** The input OpenCode ran the tool with. */
  arguments: unknown;
  status: string;
  /** The tool message content the model received in a later request; null if it never received one. */
  result: unknown;
  resultBytes: number | null;
};

export type TurnRecord = {
  scenario: string;
  session: number;
  turn: number;
  sessionID: string;
  messageID: string;
  prompt: string;
  /** The first model request of the turn; null when the turn made none. */
  request: ModelRequest | null;
  /** The requests that followed within the same turn, after tool calls. */
  continuations: ModelRequest[];
  /** The turn's `<memory-context>` block: from the prefetch log when one is given, else split off the
   * stored user text; null when there is none. */
  prefetch: string | null;
  /** Prefetch log lines that did not parse, when there were any. */
  prefetchLogErrors?: string[];
  toolCalls: ToolCallRecord[];
  answer: string | null;
  /** Set when the turn did not end in `succeeded`. */
  error?: string;
  /** OpenCode's error for a turn that did not succeed, from its `idle` entry, when it gives one: the outcome
   * word alone could not say why a turn failed (2026-11-14). */
  errorDetail?: { type: string; message: string; status?: number; response?: { body: string } };
};

const PROVIDER = "harness";
const TURN_TIMEOUT_MS = 10 * 60_000;
/** How long the driver waits for OpenCode to stop a turn that reached the cap. */
const INTERRUPT_TIMEOUT_MS = 10_000;
const STORE_TOKEN = "{store}";
/** The window OpenCode is told a model has, unless an ollama scenario names its own. */
const DEFAULT_CONTEXT = 65536;

type Endpoint = { baseURL: string; requests: object[]; stop(): Promise<void> };

/** The OpenAI-compatible root the recording proxy forwards to: RECALL_OPENCODE_OLLAMA_URL, or the local
 * Ollama when it is unset or empty. Refused unless it ends in `/v1`, since the proxy puts the request
 * path after it and Ollama's native root answers that path 404. */
export function ollamaUpstream(value: string | undefined): string {
  const url = value || "http://127.0.0.1:11434/v1";
  if (!url.endsWith("/v1")) throw new Error(`RECALL_OPENCODE_OLLAMA_URL must end in /v1 (got ${url})`);
  return url;
}

/** OpenCode's own warnings and errors, to stderr, for every host this repository creates. The host drops
 * them unless it is given a writer, so a run that failed inside OpenCode left nothing on stderr to say
 * why. One line per entry, so a multi-line cause cannot pass for a test's own error output. */
export const HOST_LOG: NonNullable<NonNullable<Parameters<typeof OpenCode.create>[0]>["log"]> = {
  level: "warn",
  emit: (e) =>
    console.error(`opencode ${e.level}: ${e.message}${e.cause === undefined ? "" : ` (cause: ${String(e.cause)})`}`.replace(/\s*\n\s*/g, " ")),
};

// For a real model: forward every request unchanged and keep a parsed copy of each body the model
// received. A failed request is answered with `x-should-retry: false`: OpenCode would otherwise retry
// it (up to ten times) and every retry would land in the record as one more body. The turn fails
// instead, and its record says so in `error`.
async function startRecordingProxy(upstream: string): Promise<Endpoint> {
  const requests: object[] = [];
  const server = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    // Ollama sends nothing while the model writes a tool call (up to 332 s seen), and Bun closes a connection
    // silent for 10 s by default, which cut those answers (2026-11-14). OpenCode's own chunk timeout still applies.
    idleTimeout: 0,
    async fetch(req) {
      const url = new URL(req.url);
      const body = req.method === "POST" ? await req.text() : undefined;
      const headers = new Headers(req.headers);
      headers.delete("host");
      let res: Response;
      try {
        res = await fetch(upstream + url.pathname.replace(/^\/v1/, "") + url.search, { method: req.method, headers, body });
      } catch (error) {
        return new Response(`recording proxy: ${upstream} unreachable: ${String(error)}`, {
          status: 502,
          headers: { "x-should-retry": "false" },
        });
      }
      if (body !== undefined && url.pathname.endsWith("/chat/completions")) requests.push(JSON.parse(body));
      const out = new Headers(res.headers);
      if (!res.ok) out.set("x-should-retry", "false");
      return new Response(res.body, { status: res.status, headers: out });
    },
  });
  return { baseURL: `http://127.0.0.1:${server.port}/v1`, requests, stop: () => server.stop(true) };
}

function splitRequest(body: object): ModelRequest {
  const { messages, tools, ...params } = body as { messages: { role: string; content: unknown }[]; tools?: unknown };
  let k = 0;
  while (k < messages.length && messages[k]!.role === "system") k += 1;
  return { system: messages.slice(0, k).map((m) => m.content), messages: messages.slice(k), tools, params };
}

function pluginEntry(spec: PluginSpec, storeDir: string | undefined) {
  const path = resolve(typeof spec === "string" ? spec : spec.path);
  // OpenCode 2.2.7 skips a configured plugin file with only a logged warning; fail here instead.
  if (!existsSync(path) || !statSync(path).isDirectory()) {
    throw new Error(`plugin ${path}: OpenCode loads a configured plugin only from a directory`);
  }
  const options: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(typeof spec === "string" ? {} : (spec.options ?? {}))) {
    if (typeof value === "string" && value.includes(STORE_TOKEN)) {
      if (storeDir === undefined) throw new Error(`plugin option ${key} uses ${STORE_TOKEN} but the scenario has no store`);
      options[key] = value.replaceAll(STORE_TOKEN, storeDir);
    } else {
      options[key] = value;
    }
  }
  return { package: path, options };
}

// The turn's prefetch block. With no log, it is the block a plugin's prompt hook appended to the
// stored user text. With a log, it is the logged block for this user message; a line that does not
// parse (a write torn by a crash) is skipped and named in `prefetchLogErrors`, so it costs that
// line, not the turn's record.
function prefetchFor(
  log: string | undefined,
  sessionID: string,
  messageID: string,
  userText: string,
): { prefetch: string | null; prefetchLogErrors?: string[] } {
  if (log === undefined) {
    const [, block] = splitInjection(userText);
    return { prefetch: block === "" ? null : block };
  }
  if (!existsSync(log)) return { prefetch: null };
  let prefetch: string | null = null;
  const errors: string[] = [];
  readFileSync(log, "utf8")
    .split("\n")
    .forEach((line, i) => {
      if (line.trim() === "") return;
      let entry: { sessionID?: unknown; messageID?: unknown; block?: unknown } | null;
      try {
        entry = JSON.parse(line);
      } catch (error) {
        errors.push(`line ${i + 1}: ${String(error)}`);
        return;
      }
      if (prefetch === null && entry?.sessionID === sessionID && entry.messageID === messageID && typeof entry.block === "string") {
        prefetch = entry.block;
      }
    });
  return errors.length > 0 ? { prefetch, prefetchLogErrors: errors } : { prefetch };
}

function byteLength(content: unknown): number {
  return Buffer.byteLength(typeof content === "string" ? content : JSON.stringify(content), "utf8");
}

function resultFor(requests: object[], callID: string): { result: unknown; resultBytes: number | null } {
  for (const body of requests) {
    for (const m of (body as { messages: { role: string; tool_call_id?: string; content: unknown }[] }).messages) {
      if (m.role === "tool" && m.tool_call_id === callID) return { result: m.content, resultBytes: byteLength(m.content) };
    }
  }
  return { result: null, resultBytes: null };
}

async function withTimeout<T>(p: Promise<T>, ms: number, what: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${what} timed out after ${ms} ms`)), ms);
  });
  try {
    return await Promise.race([p, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

export async function runScenario(
  s: Scenario,
  /** `configHome`: the XDG_CONFIG_HOME the run's plugins see; the run's own temp dir when absent. */
  opts: {
    plugins: PluginSpec[];
    outPath: string;
    prefetchLog?: string;
    configHome?: string;
    /** A config directory (its `opencode.json`, `commands/`, `agents/`) in place of an empty one, and
     * whether OpenCode reads the home's `.claude` (skills). The full stack of Task 7 uses both. */
    config?: { directory?: string; claude?: boolean };
  },
): Promise<void> {
  let work: string | undefined;
  let backend: Endpoint | undefined;
  let oc: Host | undefined;
  let failure: { error: unknown } | undefined;
  let savedConfigHome: { value: string | undefined } | undefined;
  try {
    work = mkdtempSync(join(tmpdir(), "recall-opencode-driver-"));
    let storeDir: string | undefined;
    if (s.store !== undefined) {
      storeDir = join(work, "store");
      cpSync(resolve(s.store), storeDir, { recursive: true });
    }
    // Checked before the model starts, so a bad plugin path starts nothing.
    const plugins = opts.plugins.map((p) => pluginEntry(p, storeDir));
    const project = join(work, "project");
    mkdirSync(project);
    mkdirSync(dirname(resolve(opts.outPath)), { recursive: true });
    writeFileSync(opts.outPath, "");

    backend = s.model.kind === "scripted" ? await startFakeModel(s.model.script) : await startRecordingProxy(ollamaUpstream(process.env.RECALL_OPENCODE_OLLAMA_URL));
    const modelID = s.model.kind === "scripted" ? "scripted" : s.model.id;
    const config = {
      model: `${PROVIDER}/${modelID}`,
      share: "disabled",
      update: "disable",
      plugins,
      providers: {
        [PROVIDER]: {
          name: "Harness",
          package: "@opencode/ai/providers/openai-compatible",
          settings: { baseURL: backend.baseURL },
          models: {
            [modelID]: {
              name: modelID,
              capabilities: { tools: true, input: ["text"], output: ["text"] },
              limit: { context: s.model.kind === "ollama" ? (s.model.context ?? DEFAULT_CONTEXT) : DEFAULT_CONTEXT, output: 4096 },
            },
          },
        },
      },
    };
    // A plugin runs in this process and may keep per-user state under XDG_CONFIG_HOME (the RE-call
    // plugin's default store, memos and server.log), so the run gets its own, never the operator's.
    savedConfigHome = { value: process.env.XDG_CONFIG_HOME };
    process.env.XDG_CONFIG_HOME = opts.configHome ?? join(work, "xdg");
    oc = await OpenCode.create({
      database: { path: join(work, "opencode.db") },
      config: { content: JSON.stringify(config), project: false, claude: opts.config?.claude ?? false, directory: opts.config?.directory ?? join(work, "config") },
      models: { fetch: false },
      log: HOST_LOG,
    });
    for (const [si, session] of s.sessions.entries()) {
      // A titled session skips OpenCode's title request, which would otherwise reach the model too.
      const info = await oc.sessions.create({ title: `${s.name} ${si}`, location: { directory: project } });
      for (const [ti, prompt] of session.turns.entries()) {
        const before = backend.requests.length;
        const user = await oc.sessions.prompt({ sessionID: info.id, text: prompt });
        const turn = await waitForTurn(oc, info.id, user.id);
        const requests = backend.requests.slice(before);
        const assistants = turn.messages.filter((m) => m.type === "assistant");
        const toolCalls: ToolCallRecord[] = assistants.flatMap((m) =>
          m.content.flatMap((c) =>
            c.type === "tool"
              ? [{ id: c.id, name: c.name, arguments: c.state.input, status: c.state.status, ...resultFor(requests, c.id) }]
              : [],
          ),
        );
        const last = assistants.at(-1);
        const record: TurnRecord = {
          scenario: s.name,
          session: si,
          turn: ti,
          sessionID: info.id,
          messageID: user.id,
          prompt,
          request: requests[0] === undefined ? null : splitRequest(requests[0]),
          continuations: requests.slice(1).map(splitRequest),
          ...prefetchFor(opts.prefetchLog, info.id, user.id, turn.userText),
          toolCalls,
          answer: last === undefined ? null : last.content.flatMap((c) => (c.type === "text" ? [c.text] : [])).join(""),
        };
        if (turn.outcome !== "succeeded") record.error = turn.outcome;
        if (turn.outcome !== "succeeded" && turn.error !== undefined) record.errorDetail = turn.error;
        appendFileSync(opts.outPath, JSON.stringify(record) + "\n");
      }
    }
  } catch (error) {
    failure = { error };
  }
  // Every step runs whatever the one before it did, and the run's own error is the one thrown.
  const errors: unknown[] = [];
  for (const step of [() => oc?.close(), () => backend?.stop()]) {
    try {
      await step();
    } catch (error) {
      errors.push(error);
    }
  }
  if (savedConfigHome !== undefined) {
    if (savedConfigHome.value === undefined) delete process.env.XDG_CONFIG_HOME;
    else process.env.XDG_CONFIG_HOME = savedConfigHome.value;
  }
  if (work !== undefined) {
    try {
      rmSync(work, { recursive: true, force: true, maxRetries: 5 });
    } catch (error) {
      // A temp dir Windows still holds open neither fails a run that wrote its records nor
      // replaces the error of one that did not; it is reported and left.
      console.error(`driver: could not remove ${work}: ${String(error)}`);
    }
  }
  if (failure !== undefined) throw failure.error;
  if (errors.length > 0) throw new AggregateError(errors, "driver cleanup failed");
}

type Host = Awaited<ReturnType<typeof OpenCode.create>>;
type Message = Awaited<ReturnType<Host["sessions"]["context"]>>[number];

// The turn is over when an `idle` entry follows its user message. `wait` can return before the
// prompt is picked up, so it is re-checked against the context until that entry is there.
/** The turn cap: TURN_TIMEOUT_MS, or RECALL_OPENCODE_TURN_TIMEOUT_MS (a test seam). */
const turnTimeoutMs = () => Number(process.env.RECALL_OPENCODE_TURN_TIMEOUT_MS) || TURN_TIMEOUT_MS;

// A turn that reaches the cap ends with outcome "timeout" and the messages it had by then, so a capped run
// leaves a record of what it did (its tool calls, its requests) instead of only a crash (2026-11-12).
async function waitForTurn(oc: Host, sessionID: string, messageID: string) {
  const deadline = Date.now() + turnTimeoutMs();
  for (;;) {
    try {
      await withTimeout(oc.experimental.session.wait({ sessionID }), Math.max(1, deadline - Date.now()), "turn");
    } catch (error) {
      if (Date.now() < deadline) throw error;
    }
    const context = await oc.sessions.context({ sessionID });
    const start = context.findIndex((m) => m.id === messageID);
    const user = start >= 0 ? context[start]! : undefined;
    if (start >= 0) {
      const after = context.slice(start + 1);
      const end = after.findIndex((m) => m.type === "idle" || m.type === "user");
      const idle = end >= 0 ? after[end] : undefined;
      if (idle !== undefined && idle.type === "idle" && user!.type === "user") {
        return { userText: user!.text, messages: after.slice(0, end) as Message[], outcome: idle.outcome as string, error: idle.error };
      }
      if (Date.now() >= deadline && user!.type === "user") {
        // A capped turn keeps running inside OpenCode unless stopped: its later requests and tool calls would
        // land in the next turn's record (audit J BUG-001). A stop that fails or hangs is reported, not fatal.
        await withTimeout(oc.sessions.interrupt({ sessionID }), INTERRUPT_TIMEOUT_MS, "interrupt").catch((error) =>
          console.warn(`driver: could not interrupt capped turn ${messageID}: ${error instanceof Error ? error.message : String(error)}`),
        );
        return { userText: user!.text, messages: after.slice(0, end >= 0 ? end : after.length) as Message[], outcome: "timeout", error: undefined };
      }
    }
    if (Date.now() >= deadline) throw new Error(`turn ${messageID} did not finish within ${turnTimeoutMs()} ms`);
    await Bun.sleep(50);
  }
}

// bun harness/driver.ts <scenario.json> --out <path> [--plugin <dir> [--options <json>]]... [--prefetch-log <path>]
if (import.meta.main) {
  // Outside `bun test` no preload strips the environment, and the in-process host configures a provider
  // from any key it finds there, so no provider key or token may reach it (tests/credentials.ts).
  stripCredentials(process.env);
  const args = process.argv.slice(2);
  const plugins: { path: string; options?: Record<string, unknown> }[] = [];
  let scenarioPath: string | undefined;
  let outPath: string | undefined;
  let prefetchLog: string | undefined;
  let configHome: string | undefined;
  const config: { directory?: string; claude?: boolean } = {};
  for (let i = 0; i < args.length; i += 1) {
    const a = args[i]!;
    const value = () => {
      const v = args[++i];
      if (v === undefined) throw new Error(`${a} needs a value`);
      return v;
    };
    if (a === "--plugin") plugins.push({ path: value() });
    else if (a === "--options") {
      const last = plugins.at(-1);
      if (last === undefined) throw new Error("--options must follow a --plugin");
      last.options = JSON.parse(value());
    } else if (a === "--out") outPath = value();
    else if (a === "--prefetch-log") prefetchLog = value();
    else if (a === "--config-dir") config.directory = value();
    else if (a === "--claude") config.claude = true;
    else if (a === "--config-home") configHome = value();
    else if (scenarioPath === undefined) scenarioPath = a;
    else throw new Error(`unexpected argument ${a}`);
  }
  if (scenarioPath === undefined || outPath === undefined) {
    throw new Error(
      "usage: bun harness/driver.ts <scenario.json> --out <path> [--plugin <dir> [--options <json>]]... [--prefetch-log <path>] [--config-dir <dir>] [--claude] [--config-home <dir>]",
    );
  }
  await runScenario(loadScenario(scenarioPath), { plugins, outPath, prefetchLog, config, configHome });
}
