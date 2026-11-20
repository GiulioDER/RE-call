import { afterEach, describe, expect, test } from "bun:test";
import { existsSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { type McpClient, SERVER_LOG_CAP_BYTES } from "../../src/mcp_client";
import { fakeClient, fakeScenario, tempDir } from "./fake";

const SEARCH_ANSWER = { abstained: false, hits: [{ memo_id: "a", chunk_id: "c1", verdict: "ok", text: "Works at Acme" }], evidence: { citable: ["c1"] } };
const EVIDENCE_ANSWER = { decision: "abstain", items: [] };

const open: McpClient[] = [];
function track(client: McpClient): McpClient {
  open.push(client);
  return client;
}
// A child that never reads its input is killed only after close's 5 s wait.
afterEach(async () => {
  await Promise.all(open.splice(0).map((c) => c.close()));
}, 15_000);

/** Wait until `done` holds, polling; false at the deadline. A child's start and exit take as long as the
 * machine makes them, so a test waits for them to be seen instead of sleeping a guess. */
async function until(done: () => boolean | Promise<boolean>, ms = 15_000): Promise<boolean> {
  for (const end = Date.now() + ms; Date.now() < end; await Bun.sleep(50)) if (await done()) return true;
  return false;
}

/** True once a call finds no running child: the kill has reached the exit handler. */
const notRunning = (client: McpClient) => async () => {
  const r = await client.call("recall_evidence", { query: "x" }, 200);
  return !r.ok && r.error === "the RE-call server is not running";
};

describe("McpClient", () => {
  /** Invariant: a tool's JSON text comes back parsed and otherwise unchanged.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > handshake then call returns the JSON text
   * unchanged`. Red proof (2026-11-10, local Windows, `bun run mutations MCP-JSON`): in
   * src/mcp_client.ts:160, `return { ok: true, json: JSON.parse(text) };` became `return { ok: true,
   * json: { result: JSON.parse(text) } };`; the test failed at `expect(result).toEqual({ ok: true,
   * json: SEARCH_ANSWER })`. Restored, green. */
  test("handshake then call returns the JSON text unchanged", async () => {
    const fake = fakeScenario({ tools: { recall_search: [SEARCH_ANSWER] } });
    const client = track(fakeClient(fake));
    await client.start();
    await client.ready(30_000);
    const result = await client.call("recall_search", { query: "where do I work", k: 5 });
    expect(result).toEqual({ ok: true, json: SEARCH_ANSWER });
    expect(fake.events().filter((e) => e.event === "call")).toEqual([
      { event: "call", tool: "recall_search", arguments: { query: "where do I work", k: 5 } },
    ]);
  });

  /** Invariant (audit H-35): an answer that arrives in two reads, cut inside a line and inside a
   * multibyte UTF-8 character (the fake's `split_utf8`), is framed and decoded whole.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > an answer split inside a UTF-8 character comes
   * back whole`. Red proof, row MCP-READ-STREAM: in src/mcp_client.ts `readLoop`,
   * `decoder.decode(chunk, { stream: true })` replaced by `decoder.decode(chunk)`; fails at
   * `expect(result).toEqual({ ok: true, json: { q: "café ☕ 東京" } })` (the cut character decodes as
   * U+FFFD). */
  test("an answer split inside a UTF-8 character comes back whole", async () => {
    const answer = { q: "café ☕ 東京" };
    const fake = fakeScenario({
      split_utf8: true,
      tools: { recall_search: [{ raw_result: { content: [{ type: "text", text: JSON.stringify(answer) }], isError: false } }] },
    });
    const client = track(fakeClient(fake));
    await client.start();
    await client.ready(30_000);
    const result = await client.call("recall_search", { query: "x" });
    expect(result).toEqual({ ok: true, json: { q: "café ☕ 東京" } });
  });

  /** Invariant (audit H-26): at a start, a server log at or past SERVER_LOG_CAP_BYTES is moved whole to
   * `server.log.1` and a new log is begun; a log under the cap is appended to, untouched.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > a server log at its cap is rotated at the next
   * start`. Red proof, row MCP-LOG-CAP: in src/mcp_client.ts `start`, `renameSync(log, log + ".1");`
   * replaced by `void log;`; fails at `expect(existsSync(rotated)).toBe(true)`. */
  test("a server log at its cap is rotated at the next start", async () => {
    const dir = tempDir();
    const small = join(dir, "small.log");
    writeFileSync(small, "kept;");
    const full = join(dir, "server.log");
    const rotated = full + ".1";
    writeFileSync(full, "x".repeat(SERVER_LOG_CAP_BYTES));
    for (const stderrPath of [small, full]) {
      const client = track(fakeClient(fakeScenario({ tools: {} }), { stderrPath }));
      await client.start();
      await client.ready(30_000);
      await client.close();
    }
    expect(readFileSync(small, "utf-8").startsWith("kept;")).toBe(true);
    expect(existsSync(small + ".1")).toBe(false);
    expect(existsSync(rotated)).toBe(true);
    expect(statSync(rotated).size).toBe(SERVER_LOG_CAP_BYTES);
    expect(statSync(full).size).toBeLessThan(SERVER_LOG_CAP_BYTES);
  });

  /** Invariant: `listTools` gives each tool's name, description and input schema as the server sent
   * them (the extras are offered with the server's own definitions).
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > listTools returns the server's own tool
   * definitions`. Red proof (2026-11-10, local Windows, `bun run mutations MCP-LIST`): in
   * src/mcp_client.ts:136, `inputSchema: t.inputSchema ?? {}` became `inputSchema: { type: "object" }`;
   * the test failed at `expect(await client.listTools(10_000)).toEqual(...)`. Restored, green. */
  test("listTools returns the server's own tool definitions", async () => {
    const listed = [
      { name: "recall_related", description: "Retrieve related evidence.", inputSchema: { type: "object", properties: { chunk_id: { type: "string" } }, title: "t" } },
      { name: "recall_withdraw", description: "Withdraw a memo.", inputSchema: { type: "object" }, annotations: { title: "Withdraw" } },
    ];
    const fake = fakeScenario({ tools: {}, tools_list: listed });
    const client = track(fakeClient(fake));
    await client.start();
    expect(await client.listTools(10_000)).toEqual(listed.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })));
    expect(fake.events().filter((e) => e.event === "list").length).toBe(1);
  }, 20_000);

  /** Invariant: a tool error is `ok: false` carrying the server's own text.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > a tool error comes back as ok false with the
   * server's text`. Red proof (2026-11-10, local Windows, `bun run mutations MCP-TOOL-ERROR`): in
   * src/mcp_client.ts:158, `if (result.isError) return` became `if (false) return`; the test failed at
   * `expect(result).toEqual({ ok: false, error: "store is not calibrated" })`. Restored, green. */
  test("a tool error comes back as ok false with the server's text", async () => {
    const fake = fakeScenario({ tools: { recall_search: [{ isError: true, text: "store is not calibrated" }] } });
    const client = track(fakeClient(fake));
    await client.start();
    const result = await client.call("recall_search", { query: "x" });
    expect(result).toEqual({ ok: false, error: "store is not calibrated" });
  });

  /** Invariant: one request at a time; a second call is not sent until the first has settled.
   * The first call is one the fake never answers, so without the lock the second is answered first.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > calls are serialized: two concurrent calls reach
   * the server in order, never interleaved`. Red proof (2026-11-10, local Windows, `bun run mutations
   * MCP-LOCK`): in src/mcp_client.ts:209, `const run = this.queue.then(fn, fn);` became `const run =
   * fn();`; the test failed at `expect(settled).toEqual(["first", "second"])`. Restored, green. */
  test("calls are serialized: two concurrent calls reach the server in order, never interleaved", async () => {
    const fake = fakeScenario({ tools: { recall_remember: [{ memo_id: "m1" }], recall_search: [SEARCH_ANSWER] }, no_answer_text: ["lost"] });
    const client = track(fakeClient(fake));
    await client.start();
    await client.ready(30_000);
    const settled: string[] = [];
    const first = client.call("recall_remember", { text: "lost" }, 1_500).then((r) => {
      settled.push("first");
      return r;
    });
    const second = client.call("recall_search", { query: "x" }, 10_000).then((r) => {
      settled.push("second");
      return r;
    });
    const [a, b] = await Promise.all([first, second]);
    expect(settled).toEqual(["first", "second"]);
    expect(a.ok).toBe(false);
    expect(b).toEqual({ ok: true, json: SEARCH_ANSWER });
    expect(fake.events().filter((e) => e.event === "call").map((e) => e.tool)).toEqual(["recall_remember", "recall_search"]);
  }, 20_000);

  /** Invariant: a call past its timeout returns `ok: false`, and its late answer is never taken as the
   * next call's.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > a call past its timeout returns ok false and the
   * client stays usable`. Red proof (2026-11-10, local Windows, `bun run mutations MCP-TIMEOUT`): in
   * src/mcp_client.ts:252, `const bounded = timeoutMs === undefined ? answer : withTimeout(...)` became
   * `const bounded = answer;`; the test failed at `expect(late.ok).toBe(false)`. Restored, green. */
  test("a call past its timeout returns ok false and the client stays usable", async () => {
    const fake = fakeScenario({ tools: { recall_search: [SEARCH_ANSWER], recall_evidence: [EVIDENCE_ANSWER] }, delay: { recall_search: 1.5 } });
    const client = track(fakeClient(fake));
    await client.start();
    await client.ready(30_000);
    const late = await client.call("recall_search", { query: "x" }, 300);
    expect(late.ok).toBe(false);
    expect(late.ok ? "" : late.error).toContain("no answer within");
    const next = await client.call("recall_evidence", { query: "x" }, 10_000);
    expect(next).toEqual({ ok: true, json: EVIDENCE_ANSWER });
  }, 20_000);

  /** Invariant (Review Focus 1): a dead child starts again on the next call, and not again within the
   * restart interval, measured on the client's clock.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > a dead child restarts on the next call, and not
   * again within 60 s`. Red proof (2026-11-10, local Windows, `bun run mutations MCP-RESTART`): in
   * src/mcp_client.ts:218, `if (this.now() - this.lastStart < interval) throw` became `if (false)
   * throw`; the test failed at `expect(refused).toEqual({ ok: false, error: "the RE-call server is not
   * running" })`. Restored, green. Red proof (2026-11-10, local Windows, `bun run mutations
   * MCP-RESTART-NONE`): in src/mcp_client.ts:219-220, `await this.stop(); await this.start();` became
   * `throw new McpError("the RE-call server is not running");`; the test failed at `expect(await
   * client.call("recall_evidence", ...)).toEqual({ ok: true, json: EVIDENCE_ANSWER })`. Restored,
   * green. */
  test("a dead child restarts on the next call, and not again within 60 s", async () => {
    const fake = fakeScenario({ tools: { recall_evidence: [EVIDENCE_ANSWER] }, exit_on: "recall_search" });
    let clock = 1_000_000;
    const client = track(fakeClient(fake, { now: () => clock }));
    await client.start();
    await client.ready(30_000);
    const starts = () => fake.events().filter((e) => e.event === "start").length;

    expect((await client.call("recall_search", { query: "x" }, 10_000)).ok).toBe(false); // the child exits
    clock += 61_000;
    expect(await client.call("recall_evidence", { query: "x" }, 30_000)).toEqual({ ok: true, json: EVIDENCE_ANSWER });
    expect(starts()).toBe(2);
    expect((await client.call("recall_search", { query: "x" }, 10_000)).ok).toBe(false); // exits again
    clock += 10_000;
    const refused = await client.call("recall_evidence", { query: "x" }, 10_000);
    expect(refused).toEqual({ ok: false, error: "the RE-call server is not running" });
    expect(starts()).toBe(2);
  }, 30_000);

  /** Invariant: `ready` waits for a slow handshake, and gives up at its bound.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > ready waits for a slow initialize and gives up
   * at the bound`. Red proof (2026-11-10, local Windows, `bun run mutations MCP-READY`): in
   * src/mcp_client.ts:205, `await withTimeout(this.handshake, timeoutMs, "initialize");` became `await
   * this.handshake;`; the test failed at `expect(outcome).toBe("initialize got no answer within
   * 0.5s")`. Restored, green. */
  test("ready waits for a slow initialize and gives up at the bound", async () => {
    const slow = track(fakeClient(fakeScenario({ tools: {}, init_delay: 1.0 })));
    const started = Date.now();
    await slow.start();
    await slow.ready(10_000);
    expect(Date.now() - started).toBeGreaterThanOrEqual(900);

    const stuck = track(fakeClient(fakeScenario({ tools: {}, init_delay: 4.0 })));
    await stuck.start();
    const outcome = await stuck.ready(500).then(
      () => "ready",
      (e: Error) => e.message,
    );
    expect(outcome).toBe("initialize got no answer within 0.5s");
  }, 20_000);

  /** Invariant: close ends the child by closing its input (the fake logs `exit` only at end of input).
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > close closes stdin and the child exits`. Red
   * proof (2026-11-10, local Windows, `bun run mutations MCP-CLOSE`): in src/mcp_client.ts:187, `await
   * proc.stdin.end();` became `void 0;`; the test failed at `expect(fake.events().map((e) =>
   * e.event)).toEqual(["start", "exit"])`. Restored, green. */
  test("close closes stdin and the child exits", async () => {
    const fake = fakeScenario({ tools: {} });
    const client = fakeClient(fake);
    await client.start();
    await client.ready(30_000);
    await client.close();
    expect(fake.events().map((e) => e.event)).toEqual(["start", "exit"]);
  }, 20_000);

  /** Invariant (review P2, item 1): `close()` is final; a call after it starts no child, even once
   * the restart interval has passed.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > after close, a call starts no child`. Red proof
   * (2026-11-10, local Windows, `bun run mutations MCP-CLOSED`): in src/mcp_client.ts:78, `if
   * (this.closed) throw` became `if (false) throw`; the test failed at `expect(await
   * client.call(...)).toEqual({ ok: false, error: "the RE-call client was closed" })`. Restored, green. */
  test("after close, a call starts no child", async () => {
    const fake = fakeScenario({ tools: { recall_evidence: [EVIDENCE_ANSWER] } });
    let clock = 1_000_000;
    const client = track(fakeClient(fake, { now: () => clock }));
    await client.start();
    await client.ready(30_000);
    await client.close();
    clock += 61_000;
    expect(await client.call("recall_evidence", { query: "x" }, 10_000)).toEqual({ ok: false, error: "the RE-call client was closed" });
    expect(fake.events().filter((e) => e.event === "start").length).toBe(1);
  }, 20_000);

  /** Invariant (review P2, item 1): a close that lands while a call is restarting a dead child leaves
   * no child running.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > a close racing a restart leaves no child
   * running`. Red proof (2026-11-10, local Windows, `bun run mutations MCP-CLOSED-RACE`): in
   * src/mcp_client.ts:78, `if (this.closed) throw` became `if (false) throw`; the test failed at
   * `expect((await restarting).ok).toBe(false)`. Restored, green. */
  test("a close racing a restart leaves no child running", async () => {
    const fake = fakeScenario({ tools: { recall_evidence: [EVIDENCE_ANSWER] }, exit_on: "recall_search" });
    let clock = 1_000_000;
    const client = track(fakeClient(fake, { now: () => clock }));
    await client.start();
    await client.ready(30_000);
    expect((await client.call("recall_search", { query: "x" }, 10_000)).ok).toBe(false); // the child exits
    clock += 61_000;
    const restarting = client.call("recall_evidence", { query: "x" }, 10_000);
    await client.close(); // lands before the queued restart reaches start()
    expect((await restarting).ok).toBe(false);
    await Bun.sleep(500);
    expect(fake.events().filter((e) => e.event === "start").length).toBe(1);
  }, 20_000);

  /** Invariant (review P2, item 2): a child that never answers the handshake is killed when `ready`
   * gives up, so the next call (after the restart interval) starts a new one instead of waiting on it.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > a child that never answers the handshake is
   * killed when ready gives up`. Red proof (2026-11-10, local Windows, `bun run mutations
   * MCP-WEDGED-READY`): in src/mcp_client.ts:122, `if (error instanceof TimeoutError && proc &&
   * this.proc === proc) proc.kill();` became `void proc;`; the test failed at
   * `expect(fake.events().filter((e) => e.event === "start").length).toBe(2)`. Restored, green.
   *
   * Made deterministic in the audit (H-15, owner item K3): it raced Python's cold start (0.7 to 1.3 s
   * measured) against the 0.5 s `ready`, which could kill the first child before it logged its start,
   * against a fixed 1 s sleep for the exit, and against the 2 s call for the second start. It now waits
   * for each of the three to be seen; the assertions are unchanged. Re-proved red with the same row. */
  test("a child that never answers the handshake is killed when ready gives up", async () => {
    const fake = fakeScenario({ tools: {}, stall_stdin: true });
    let clock = 1_000_000;
    const client = track(fakeClient(fake, { now: () => clock }));
    const starts = () => fake.events().filter((e) => e.event === "start").length;
    await client.start();
    await until(() => starts() === 1); // the first child runs before ready gives up on it
    expect(await client.ready(500).then(() => "ready", (e: Error) => e.message)).toBe("initialize got no answer within 0.5s");
    await until(notRunning(client)); // the kill reaches the exit handler
    clock += 61_000;
    expect((await client.call("recall_evidence", { query: "x" }, 2_000)).ok).toBe(false); // the new child stalls too
    await until(() => starts() === 2);
    expect(fake.events().filter((e) => e.event === "start").length).toBe(2);
  }, 60_000);

  /** Invariant (review P2, item 2): two consecutive requests with no answer kill a child that stopped
   * reading (WEDGED_AFTER_TIMEOUTS = 2); one does not.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > two consecutive call timeouts kill a child that
   * stopped reading, one does not`. Red proof (2026-11-10, local Windows, `bun run mutations
   * MCP-WEDGED-CALLS`): in src/mcp_client.ts:25, `export const WEDGED_AFTER_TIMEOUTS = 2;` became
   * `export const WEDGED_AFTER_TIMEOUTS = 1;`; the test failed at `expect(starts()).toBe(1) after the
   * second call`. Restored, green. Red proof (2026-11-10, local Windows, `bun run mutations
   * MCP-WEDGED-KILL`): in src/mcp_client.ts:276, `if (this.timeouts >= WEDGED_AFTER_TIMEOUTS && proc &&
   * this.proc === proc) proc.kill();` became `void proc;`; the test failed at `expect(starts()).toBe(2)
   * after the third call`. Restored, green.
   *
   * Audit H-15: the start count alone could not see a restart within the 0.4 s call, since the new
   * child logs its start only after Python's cold start, so MCP-WEDGED-CALLS survived a replay
   * (2026-11-11). The first child's own process is now checked after the first timeout as well; with
   * MCP-WEDGED-CALLS the test now fails at `expect(running(first)).toBe(true)`. Re-proved red with both
   * rows. */
  test("two consecutive call timeouts kill a child that stopped reading, one does not", async () => {
    const fake = fakeScenario({ tools: {}, stall_after_init: true });
    let clock = 1_000_000;
    const client = track(fakeClient(fake, { now: () => clock }));
    await client.start();
    await client.ready(30_000);
    const starts = () => fake.events().filter((e) => e.event === "start").length;
    const first = (fake.events().find((e) => e.event === "start") as { pid?: number }).pid!;
    const running = (pid: number) => {
      try {
        process.kill(pid, 0);
        return true;
      } catch {
        return false;
      }
    };
    expect((await client.call("recall_search", { query: "x" }, 400)).ok).toBe(false);
    await Bun.sleep(1_000);
    expect(running(first)).toBe(true); // one timeout left the child alone
    clock += 61_000;
    expect((await client.call("recall_search", { query: "x" }, 400)).ok).toBe(false);
    expect(starts()).toBe(1); // one timeout left the child alone, so this call did not restart it
    await Bun.sleep(1_000);
    expect((await client.call("recall_search", { query: "x" }, 3_000)).ok).toBe(false);
    await until(() => starts() === 2); // the new child logs its start in its own time (audit H-15)
    expect(starts()).toBe(2); // the second timeout killed it, so this call started a new one
  }, 45_000);

  /** Invariant (review P3, item 3): a request is not sent when less than SEND_FLOOR_MS of its budget
   * is left. The client's clock advances 400 ms per reading, so 100 ms are left of 500.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > a call is not sent with its budget almost
   * spent`. Red proof (2026-11-10, local Windows, `bun run mutations MCP-FLOOR`): in
   * src/mcp_client.ts:151, `if (left < SEND_FLOOR_MS) return` became `if (false) return`; the test
   * failed at `expect(await client.call("recall_search", ..., 500)).toEqual({ ok: false, error:
   * "recall_search was not sent: 100 ms of its budget left" })`. Restored, green. */
  test("a call is not sent with its budget almost spent", async () => {
    const fake = fakeScenario({ tools: { recall_search: [SEARCH_ANSWER] } });
    let clock = 1_000_000;
    const client = track(fakeClient(fake, { now: () => (clock += 400) }));
    await client.start();
    await client.ready(30_000);
    expect(await client.call("recall_search", { query: "x" }, 500)).toEqual({
      ok: false,
      error: "recall_search was not sent: 100 ms of its budget left",
    });
    expect(fake.events().filter((e) => e.event === "call")).toEqual([]);
  }, 20_000);

  /** Invariant (review P3, item 3): a request its caller gave up on is cancelled on the server
   * (`notifications/cancelled` with its id), so a late write is not left running unannounced.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > a call that times out is cancelled on the
   * server`. Red proof (2026-11-10, local Windows, `bun run mutations MCP-CANCEL`): in
   * src/mcp_client.ts:271, `this.send({ jsonrpc: "2.0", method: "notifications/cancelled", params: {
   * requestId: id, reason } });` became `void reason;`; the test failed at
   * `expect(events).toEqual(["call", "cancelled", "call"])`. Restored, green. */
  test("a call that times out is cancelled on the server", async () => {
    const fake = fakeScenario({ tools: { recall_search: [SEARCH_ANSWER], recall_evidence: [EVIDENCE_ANSWER] }, delay: { recall_search: 1.5 } });
    const client = track(fakeClient(fake));
    await client.start();
    await client.ready(30_000);
    expect((await client.call("recall_search", { query: "x" }, 300)).ok).toBe(false);
    expect(await client.call("recall_evidence", { query: "x" }, 10_000)).toEqual({ ok: true, json: EVIDENCE_ANSWER });
    const events = fake.events().filter((e) => e.event !== "start").map((e) => e.event);
    expect(events).toEqual(["call", "cancelled", "call"]);
    expect(typeof (fake.events().find((e) => e.event === "cancelled") as { requestId?: unknown }).requestId).toBe("number");
  }, 20_000);

  /** Invariant (review P3, item 4): an `initialize` answered with a JSON-RPC error is not a ready
   * server.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > an initialize answered with an error is not
   * ready`. Red proof (2026-11-10, local Windows, `bun run mutations MCP-INIT-ERROR`): in
   * src/mcp_client.ts:106, `const init = this.requestResult("initialize", {` became `const init =
   * this.request("initialize", {`; the test failed at `expect(await
   * client.ready(10_000).then(...)).toBe("initialize: unsupported protocol version")`. Restored, green. */
  test("an initialize answered with an error is not ready", async () => {
    const fake = fakeScenario({ tools: {}, init_error: { code: -32602, message: "unsupported protocol version" } });
    const client = track(fakeClient(fake));
    await client.start();
    expect(await client.ready(10_000).then(() => "ready", (e: Error) => e.message)).toBe("initialize: unsupported protocol version");
  }, 20_000);

  // -- audit (Task 9) -----------------------------------------------------------------------------

  /** Invariant (audit H-02): a call's budget covers its wait behind another call, so a 0.5 s prefetch
   * queued behind a 3 s tool call comes back at its own bound, unsent, instead of waiting the other out.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > a call queued behind a slow one comes back at its
   * own budget, unsent`. Red proof (2026-11-11, local Windows, `bun run mutations MCP-QUEUE-BUDGET`): in
   * src/mcp_client.ts `call`, `return await Promise.race([run, gaveUp]);` became `return await run;`;
   * the test failed at `expect(elapsed).toBeLessThan(2_000)`. Restored, green. */
  test("a call queued behind a slow one comes back at its own budget, unsent", async () => {
    const fake = fakeScenario({ tools: { recall_search: [SEARCH_ANSWER], recall_evidence: [EVIDENCE_ANSWER] }, delay: { recall_evidence: 3 } });
    const client = track(fakeClient(fake));
    await client.start();
    await client.ready(30_000);
    const slow = client.call("recall_evidence", { query: "slow" }, 10_000);
    const started = Date.now();
    const queued = await client.call("recall_search", { query: "queued" }, 500);
    const elapsed = Date.now() - started;
    expect(queued).toEqual({ ok: false, error: "recall_search was not sent: its 0.5s budget ran out first" });
    expect(elapsed).toBeLessThan(2_000);
    expect(await slow).toEqual({ ok: true, json: EVIDENCE_ANSWER });
    expect(fake.events().filter((e) => e.event === "call").map((e) => e.tool)).toEqual(["recall_evidence"]);
  }, 30_000);

  /** Invariant (audit H-02, differential review): a call sent after waiting behind another gets only what
   * is left of its budget, so the whole call, wait included, ends at its bound: a 3 s search that waited
   * 1 s and is answered 2.5 s after it is sent times out at 3 s, not at 4 s.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > a call that waited for its turn is bounded by
   * what is left of its budget`. Red proof (2026-11-11, local Windows, `bun run mutations
   * MCP-QUEUE-LEFT`): in src/mcp_client.ts `call`, `const left = timeoutMs - (this.now() - started);`
   * became `const left = timeoutMs;`; the test failed at `expect(queued.ok).toBe(false)`. Restored,
   * green. */
  test("a call that waited for its turn is bounded by what is left of its budget", async () => {
    const fake = fakeScenario({ tools: { recall_search: [SEARCH_ANSWER], recall_evidence: [EVIDENCE_ANSWER] }, delay: { recall_evidence: 1, recall_search: 2.5 } });
    const client = track(fakeClient(fake));
    await client.start();
    await client.ready(30_000);
    const slow = client.call("recall_evidence", { query: "slow" }, 10_000);
    const started = Date.now();
    const queued = await client.call("recall_search", { query: "queued" }, 3_000);
    const elapsed = Date.now() - started;
    expect(queued.ok).toBe(false);
    expect(elapsed).toBeLessThan(3_500);
    expect(await slow).toEqual({ ok: true, json: EVIDENCE_ANSWER });
  }, 30_000);

  /** Invariant (audit H-03): a late answer to a request that timed out shows a child that still reads,
   * so slow answers do not add up to a wedged child: two searches answered after their budget leave the
   * child running, and the next search is answered by it.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > late answers do not get a slow child killed as
   * wedged`. Red proof (2026-11-11, local Windows, `bun run mutations MCP-LATE-ANSWER`): in
   * src/mcp_client.ts `readLoop`, `else this.timeouts = 0;` was removed; the test failed at
   * `expect(await client.call("recall_search", { query: "x" }, 5_000)).toEqual(...)`. Restored, green. */
  test("late answers do not get a slow child killed as wedged", async () => {
    const fake = fakeScenario({ tools: { recall_search: [SEARCH_ANSWER] }, delay: { recall_search: 0.8 } });
    const client = track(fakeClient(fake));
    await client.start();
    await client.ready(30_000);
    // The fake reads a cancel only after it has sent the answer it was busy with: the late answer is out.
    const cancelled = () => fake.events().filter((e) => e.event === "cancelled").length;
    expect((await client.call("recall_search", { query: "x" }, 500)).ok).toBe(false);
    await until(() => cancelled() === 1);
    await Bun.sleep(200);
    expect((await client.call("recall_search", { query: "x" }, 500)).ok).toBe(false);
    await until(() => cancelled() === 2);
    await Bun.sleep(200);
    expect(await client.call("recall_search", { query: "x" }, 5_000)).toEqual({ ok: true, json: SEARCH_ANSWER });
    expect(fake.events().filter((e) => e.event === "start").length).toBe(1);
  }, 45_000);

  /** Invariant (audit H-48): every start is bounded, a restart's included: a child that refuses the
   * handshake is killed, so a call past the restart interval starts a new one instead of failing on the
   * refused handshake until the host restarts.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > a child that refuses the handshake is replaced on
   * a later call`. Red proof (2026-11-11, local Windows, `bun run mutations MCP-START-BOUND`): in
   * src/mcp_client.ts `start`, `if (this.proc === proc) proc.kill();` became `void proc;`; the test
   * failed at `expect(starts()).toBe(2)`. Restored, green. */
  test("a child that refuses the handshake is replaced on a later call", async () => {
    const fake = fakeScenario({ tools: {}, init_error: { code: -32602, message: "unsupported protocol version" } });
    let clock = 1_000_000;
    const client = track(fakeClient(fake, { now: () => clock }));
    const starts = () => fake.events().filter((e) => e.event === "start").length;
    await client.start();
    expect(await client.ready(10_000).then(() => "ready", (e: Error) => e.message)).toBe("initialize: unsupported protocol version");
    await until(notRunning(client));
    clock += 61_000;
    expect((await client.call("recall_evidence", { query: "x" }, 2_000)).ok).toBe(false); // the new child refuses too
    await until(() => starts() === 2);
    expect(starts()).toBe(2);
  }, 60_000);

  /** Invariant (audit H-49): an aborted call stops waiting at once, its request is cancelled on the
   * server, and the client is free for the next call.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > an aborted call returns at once and is cancelled
   * on the server`. Red proof (2026-11-11, local Windows, `bun run mutations MCP-ABORT`): in
   * src/mcp_client.ts `request`, the abort's `waiter.reject(...)` became `void waiter;`; the test failed
   * at `expect(await pending).toEqual({ ok: false, error: "tools/call was aborted" })`. Restored, green. */
  test("an aborted call returns at once and is cancelled on the server", async () => {
    const fake = fakeScenario({ tools: { recall_remember: [{ memo_id: "m.md" }], recall_evidence: [EVIDENCE_ANSWER] }, delay: { recall_remember: 3 } });
    const client = track(fakeClient(fake));
    await client.start();
    await client.ready(30_000);
    const abort = new AbortController();
    const pending = client.call("recall_remember", { text: "t" }, 10_000, abort.signal);
    await until(() => fake.events().some((e) => e.event === "call")); // the request reached the server
    const abortedAt = Date.now();
    abort.abort();
    expect(await pending).toEqual({ ok: false, error: "tools/call was aborted" });
    expect(Date.now() - abortedAt).toBeLessThan(1_000);
    await until(() => fake.events().some((e) => e.event === "cancelled"));
    expect(fake.events().filter((e) => e.event === "cancelled").length).toBe(1);
    expect(await client.call("recall_evidence", { query: "x" }, 10_000)).toEqual({ ok: true, json: EVIDENCE_ANSWER });
  }, 30_000);

  /** Invariant (audit H-47): a client over the fake hands it only the variables a Python child needs to
   * start, not the parent's environment, because the fake logs its environment to a file in a temp dir.
   *
   * Node: `tests/unit/mcp_client.test.ts > McpClient > a client over the fake does not hand it the
   * parent's environment`. Red proof (2026-11-11, local Windows, `bun run mutations FAKE-ENV-MINIMAL`):
   * in tests/unit/fake.ts `fakeClient`, `...(extra.env ?? base)` became `...(extra.env ?? (process.env
   * as Record<string, string>))`; the test failed at
   * `expect(Object.keys(env)).not.toContain("RECALL_OC_PROBE_TOKEN")`. Restored, green. */
  test("a client over the fake does not hand it the parent's environment", async () => {
    process.env.RECALL_OC_PROBE_TOKEN = "not-for-the-log";
    try {
      const fake = fakeScenario({ tools: {} });
      const client = track(fakeClient(fake));
      await client.start();
      await client.ready(30_000);
      const env = fake.events().find((e) => e.event === "start")!.env!;
      expect(Object.keys(env)).not.toContain("RECALL_OC_PROBE_TOKEN");
      expect(env.FAKE_LOG).toBe(fake.log);
    } finally {
      delete process.env.RECALL_OC_PROBE_TOKEN;
    }
  }, 20_000);
});
