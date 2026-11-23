/** A minimal MCP client over stdio for one child `recall-mcp`, a port of recall-hermes `mcp_client.py`.
 *
 * MCP's stdio transport is newline-delimited JSON-RPC 2.0. Answers are matched to requests by id, so
 * a request that timed out cannot be answered by the next one's response, and the server is told to
 * cancel it. One request runs at a time; a dead child is started again on the next call, at most once
 * per restart interval, and a child that stops answering is killed so that it can be. After `close()`
 * the client starts nothing. */

import { closeSync, mkdirSync, openSync, renameSync, statSync } from "node:fs";
import { dirname } from "node:path";

import type { Subprocess } from "bun";

export const PROTOCOL_VERSION = "2025-06-18";
const CLIENT_INFO = { name: "re-call-opencode", version: "0.0.0" };
export const READY_TIMEOUT_MS = 120_000; // the first start loads the embedding model and refreshes the store
export const CALL_TIMEOUT_MS = 60_000;
export const RESTART_INTERVAL_MS = 60_000;
const CLOSE_WAIT_MS = 5_000;
/** A request is not sent with less than this left of its budget: it would be reported failed while the
 * server still ran it (a write landing after the caller was told it failed). */
export const SEND_FLOOR_MS = 250;
/** Consecutive requests that get no answer before the child is taken as wedged and killed. One is a slow
 * answer (a cold search); two in a row, with nothing answered between, is a server that stopped reading. */
export const WEDGED_AFTER_TIMEOUTS = 2;
/** At a start, a server log this large is moved to `<log>.1` (replacing an older one) and a new log is
 * begun, so the disk holds the current log plus one of about this size. Measured 2026-11-11
 * (docs/preregistrations/2026-11-11-server-log-cap.md): 0.2 KB per start and 0.7 KB per failed search,
 * none per successful call; 8 MiB holds ten starts of 1,000 failed searches. The child keeps the file
 * open while it runs, so a log is rotated only between runs. */
export const SERVER_LOG_CAP_BYTES = 8 * 1024 * 1024;

export type ToolDef = { name: string; description: string; inputSchema: object };
export type CallResult = { ok: true; json: unknown } | { ok: false; error: string };

export type McpClientOptions = {
  argv: string[];
  env: Record<string, string>;
  cwd: string;
  /** The child's stderr is appended here; without it, it is discarded. */
  stderrPath?: string;
  restartIntervalMs?: number;
  /** A monotonic clock for elapsed time, overridable by tests. */
  now?: () => number;
};

type Message = { id?: unknown; method?: unknown; result?: unknown; error?: unknown };
type Waiter = { resolve: (m: Message) => void; reject: (e: Error) => void };

class McpError extends Error {}
class TimeoutError extends McpError {}

function withTimeout<T>(promise: Promise<T>, ms: number, what: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const bound = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new TimeoutError(`${what} got no answer within ${(ms / 1000).toFixed(1)}s`)), ms);
  });
  return Promise.race([promise, bound]).finally(() => clearTimeout(timer));
}

export class McpClient {
  private readonly opts: McpClientOptions;
  private readonly now: () => number;
  private proc: Subprocess<"pipe", "pipe", number | "ignore"> | null = null;
  private stderrFd: number | null = null;
  private alive = false;
  private nextId = 1;
  private waiters = new Map<number, Waiter>();
  private handshake: Promise<void> | null = null;
  private lastStart = -Infinity;
  private queue: Promise<unknown> = Promise.resolve();
  private closed = false;
  private timeouts = 0;

  constructor(opts: McpClientOptions) {
    this.opts = opts;
    this.now = opts.now ?? (() => performance.now());
  }

  /** Launch the child and send the handshake; `ready()` waits for its answer. */
  async start(): Promise<void> {
    // close() is final: a restart racing a dispose must not leave a child nothing will close (Hermes
    // `_closed`, vendor/hermes/provider.py:209 and :229).
    if (this.closed) throw new McpError("the RE-call client was closed");
    this.lastStart = this.now();
    this.timeouts = 0;
    let stderr: number | "ignore" = "ignore";
    if (this.opts.stderrPath) {
      mkdirSync(dirname(this.opts.stderrPath), { recursive: true });
      const log = this.opts.stderrPath;
      if ((statSync(log, { throwIfNoEntry: false })?.size ?? 0) >= SERVER_LOG_CAP_BYTES) {
        try {
          renameSync(log, log + ".1");
        } catch {
          // A rename that fails (another process holding the log, a read-only folder) leaves the log to
          // grow: a start must not fail over its log.
        }
      }
      stderr = this.stderrFd = openSync(log, "a");
    }
    const proc = Bun.spawn(this.opts.argv, {
      cwd: this.opts.cwd, // never the project, so no project .env is in reach of the child
      env: this.opts.env,
      stdin: "pipe",
      stdout: "pipe",
      stderr,
    });
    this.proc = proc;
    this.alive = true;
    // A read error ends the reader: the requests it would have answered fail now, not at their bound.
    const reading = this.readLoop(proc).catch((error) => {
      if (this.proc === proc) this.failAll(new McpError(`reading the RE-call server failed: ${error}`));
    });
    void proc.exited.then(async () => {
      if (this.proc !== proc) return;
      this.alive = false;
      // An answer written just before the exit may still be in the pipe: let the reader take it.
      await Promise.race([reading, Bun.sleep(500)]);
      if (this.proc === proc) this.failAll(new McpError("the RE-call server exited"));
    });
    const init = this.requestResult("initialize", {
      protocolVersion: PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: CLIENT_INFO,
    }).then(() => this.send({ jsonrpc: "2.0", method: "notifications/initialized" }));
    init.catch(() => {}); // observed by ready(); a start nobody waits on must not crash the host
    // Every start is bounded, a restart's as well as setup's (Hermes START_TIMEOUT_S): a child that refuses
    // the handshake, or never completes it, is killed, so a later call can start another.
    withTimeout(init, READY_TIMEOUT_MS, "initialize").catch(() => {
      if (this.proc === proc) proc.kill();
    });
    this.handshake = init;
  }

  /** Resolves once the handshake has completed; rejects at the bound or when the child exits. A child
   * that has not answered the handshake by the bound is killed, so the next call can start a new one. */
  async ready(timeoutMs: number = READY_TIMEOUT_MS): Promise<void> {
    const proc = this.proc;
    try {
      await this.handshakeWithin(timeoutMs);
    } catch (error) {
      if (error instanceof TimeoutError && proc && this.proc === proc) proc.kill();
      throw error;
    }
  }

  async listTools(timeoutMs: number = CALL_TIMEOUT_MS): Promise<ToolDef[]> {
    return this.locked(async () => {
      const deadline = this.now() + timeoutMs;
      await this.ensureRunning(deadline);
      const tools: ToolDef[] = [];
      let cursor: string | undefined;
      do {
        const result = await this.requestResult("tools/list", cursor ? { cursor } : {}, deadline - this.now());
        for (const t of Array.isArray(result.tools) ? result.tools : []) {
          tools.push({ name: t.name, description: t.description ?? "", inputSchema: t.inputSchema ?? {} });
        }
        cursor = typeof result.nextCursor === "string" && result.nextCursor ? result.nextCursor : undefined;
      } while (cursor);
      return tools;
    });
  }

  /** The tool's answer parsed from its JSON text; a tool error, a timeout, an abort or a dead child is
   * `ok: false`. The budget covers the wait behind other calls too, so a prefetch never waits out another
   * call's budget: a call whose budget runs out, or whose `signal` aborts, before it is sent is never sent. */
  async call(name: string, args: object, timeoutMs: number = CALL_TIMEOUT_MS, signal?: AbortSignal): Promise<CallResult> {
    const deadline = this.now() + timeoutMs;
    let sent = false;
    let unsent: CallResult | undefined;
    let settle: (result: CallResult) => void = () => {};
    const gaveUp = new Promise<CallResult>((resolve) => (settle = resolve));
    const giveUp = (why: string) => {
      if (sent || unsent) return;
      unsent = { ok: false, error: `${name} was not sent: ${why}` };
      settle(unsent);
    };
    const timer = setTimeout(() => giveUp(`its ${(timeoutMs / 1000).toFixed(1)}s budget ran out first`), timeoutMs);
    const onAbort = () => giveUp("the call was aborted");
    signal?.addEventListener("abort", onAbort);
    if (signal?.aborted) onAbort();
    const run = this.locked(async (): Promise<CallResult> => {
      try {
        if (unsent) return unsent;
        await this.ensureRunning(deadline);
        if (unsent) return unsent;
        const left = deadline - this.now();
        if (left < SEND_FLOOR_MS) return { ok: false, error: `${name} was not sent: ${Math.max(0, left)} ms of its budget left` };
        sent = true;
        clearTimeout(timer);
        const result = await this.requestResult("tools/call", { name, arguments: args }, left, signal);
        const content = Array.isArray(result.content) ? result.content : [];
        const text = content
          .filter((p: { type?: unknown }) => p && p.type === "text")
          .map((p: { text?: unknown }) => String(p.text ?? ""))
          .join("");
        if (result.isError) return { ok: false, error: text || `${name} failed` };
        try {
          return { ok: true, json: JSON.parse(text) };
        } catch {
          return { ok: false, error: `${name} answered with text that is not JSON: ${JSON.stringify(text.slice(0, 200))}` };
        }
      } catch (error) {
        return { ok: false, error: error instanceof Error ? error.message : String(error) };
      }
    });
    try {
      return await Promise.race([run, gaveUp]);
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    }
  }

  /** Close the child's input, wait for it to exit, and kill it if it does not. Final: afterwards the
   * client starts no child, and a call answers `ok: false`. */
  async close(): Promise<void> {
    this.closed = true;
    await this.stop();
  }

  // -- internals ------------------------------------------------------------------------------

  private async stop(): Promise<void> {
    const proc = this.proc;
    this.proc = null;
    this.alive = false;
    this.handshake = null;
    this.failAll(new McpError("the RE-call client was closed"));
    if (proc) {
      try {
        await proc.stdin.end();
      } catch {
        // the child closed its input first
      }
      const exited = await Promise.race([proc.exited.then(() => true), Bun.sleep(CLOSE_WAIT_MS).then(() => false)]);
      if (!exited) {
        proc.kill();
        await proc.exited;
      }
    }
    if (this.stderrFd !== null) {
      closeSync(this.stderrFd);
      this.stderrFd = null;
    }
  }

  private async handshakeWithin(timeoutMs: number): Promise<void> {
    if (!this.handshake) throw new McpError(this.closed ? "the RE-call client was closed" : "the RE-call server was not started");
    await withTimeout(this.handshake, timeoutMs, "initialize");
  }

  private locked<T>(fn: () => Promise<T>): Promise<T> {
    const run = this.queue.then(fn, fn);
    this.queue = run.catch(() => {});
    return run;
  }

  /** A dead child starts again here, at most once per restart interval. */
  private async ensureRunning(deadline: number): Promise<void> {
    if (!this.alive) {
      const interval = this.opts.restartIntervalMs ?? RESTART_INTERVAL_MS;
      if (this.now() - this.lastStart < interval) throw new McpError("the RE-call server is not running");
      await this.stop();
      await this.start();
    }
    // A caller's own budget may be shorter than a cold start: waiting here kills nothing.
    await this.handshakeWithin(Math.max(0, deadline - this.now()));
  }

  private send(message: object): void {
    const proc = this.proc;
    if (!proc || !this.alive) throw new McpError("the RE-call server is not running");
    proc.stdin.write(JSON.stringify(message) + "\n");
    const flushed = proc.stdin.flush();
    // A failed flush means the child stopped reading: the requests in flight are lost, so they fail now.
    // Nonbehavioral control, with the reader's catch in start(): no test makes either reject. Measured
    // 2026-11-10, Bun 1.4.4 on Windows: flush() after a 10 MB write to a child then killed returned no
    // promise, and a write after its exit returned true. They are observed so a rejection is never lost.
    if (flushed instanceof Promise) {
      flushed.catch((error) => {
        if (this.proc === proc) this.failAll(new McpError(`writing to the RE-call server failed: ${error}`));
      });
    }
  }

  /** The answer to one request. A caller that gives up drops its id, so a late answer is discarded. An
   * aborted `signal` gives up as a timeout does, but says nothing about whether the child still reads. */
  private request(method: string, params: object, timeoutMs?: number, signal?: AbortSignal): Promise<Message> {
    const id = this.nextId++;
    const answer = new Promise<Message>((resolve, reject) => this.waiters.set(id, { resolve, reject }));
    try {
      this.send({ jsonrpc: "2.0", id, method, params });
    } catch (error) {
      this.waiters.delete(id);
      return Promise.reject(error);
    }
    const onAbort = () => {
      const waiter = this.waiters.get(id);
      if (!waiter) return;
      this.cancel(id, `${method} was aborted`);
      waiter.reject(new McpError(`${method} was aborted`));
    };
    signal?.addEventListener("abort", onAbort);
    if (signal?.aborted) onAbort();
    const bounded = timeoutMs === undefined ? answer : withTimeout(answer, Math.max(0, timeoutMs), method);
    return bounded
      .then(
        (message) => {
          this.timeouts = 0;
          return message;
        },
        (error) => {
          if (error instanceof TimeoutError) this.gaveUp(id, error.message);
          throw error;
        },
      )
      .finally(() => {
        this.waiters.delete(id);
        signal?.removeEventListener("abort", onAbort);
      });
  }

  /** Tell the server to drop a request its caller gave up on. */
  private cancel(id: number, reason: string): void {
    try {
      this.send({ jsonrpc: "2.0", method: "notifications/cancelled", params: { requestId: id, reason } });
    } catch {
      // the child is gone, so there is nothing left to cancel
    }
  }

  /** Cancel a request that timed out, and kill a child that stopped answering. */
  private gaveUp(id: number, reason: string): void {
    const proc = this.proc;
    this.cancel(id, reason);
    this.timeouts += 1;
    if (this.timeouts >= WEDGED_AFTER_TIMEOUTS && proc && this.proc === proc) proc.kill();
  }

  private async requestResult(method: string, params: object, timeoutMs?: number, signal?: AbortSignal): Promise<Record<string, any>> {
    const message = await this.request(method, params, timeoutMs, signal);
    if (message.error !== undefined) {
      const error = message.error as { message?: unknown };
      throw new McpError(`${method}: ${typeof error === "object" && error && "message" in error ? error.message : JSON.stringify(error)}`);
    }
    if (!message.result || typeof message.result !== "object") {
      throw new McpError(`${method} answered with a result that is not an object`);
    }
    return message.result as Record<string, any>;
  }

  private failAll(error: Error): void {
    for (const waiter of this.waiters.values()) waiter.reject(error);
    this.waiters.clear();
  }

  private async readLoop(proc: Subprocess<"pipe", "pipe", number | "ignore">): Promise<void> {
    const decoder = new TextDecoder();
    let buffer = "";
    for await (const chunk of proc.stdout) {
      buffer += decoder.decode(chunk, { stream: true });
      let newline: number;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        let message: Message;
        try {
          message = JSON.parse(line);
        } catch {
          continue; // not JSON-RPC
        }
        // Notifications and server requests carry no answer to us; only answers have our ids.
        if (message && typeof message.id === "number" && message.method === undefined) {
          const waiter = this.waiters.get(message.id);
          // A late answer to a request its caller gave up on still shows a child that reads and answers.
          if (waiter) waiter.resolve(message);
          else this.timeouts = 0;
        }
      }
    }
  }
}
