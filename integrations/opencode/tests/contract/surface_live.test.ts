/** The tool surface against a real `recall-mcp` on a throwaway Lite store (last run 2026-11-20 on 0.16.0).
 *
 *     RECALL_OPENCODE_SURFACE=1 RECALL_OPENCODE_EMBEDDER=hashing RECALL_OPENCODE_BIN=<dir holding recall and recall-mcp> bun test tests/contract/surface_live.test.ts
 *
 * The server is started exactly as the plugin starts it (`childEnv`), except that RECALL_EMBEDDER is
 * RECALL_OPENCODE_EMBEDDER (default `fastembed`, the plugin's own, whose default model is
 * `BAAI/bge-small-en-v1.5`). `hashing` (RE-call's HashingEmbedder: MD5 token hashing) loads no model,
 * and the tool list does not depend on the embedder. Run 2026-11-20 with `hashing` against the
 * published recall-rag 0.16.0: 2 pass, and rows LIVE-TRAINED and LIVE-CLASSIFIED RED. Everything lives
 * in a temp dir, removed afterwards. */

import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { childEnv, PASSTHROUGH_ENV, resolveExecutable } from "../../src/child";
import { loadConfig } from "../../src/config";
import { McpClient, type ToolDef } from "../../src/mcp_client";
import { DROPPED, EXTRA, normalizeTool, offeredNames, TRAINED } from "../../src/surface";
import { recallBin } from "../e2e/apparatus";

const ON = process.env.RECALL_OPENCODE_SURFACE === "1";
const BIN = ON ? recallBin(process.env) : "";
const EMBEDDER = process.env.RECALL_OPENCODE_EMBEDDER || "fastembed";
const EXE = process.platform === "win32" ? ".exe" : "";
/** Tools a Lite server never registers (RE-call 0.15.0 `LITE_UNSERVED_TOOLS`). */
const NOT_ON_LITE = ["recall_calibration_publish"];

let dataDir = "";

/** The server's tools/list, started exactly as the plugin starts it; `tools` overrides RECALL_MCP_TOOLS. */
async function listed(tools?: string): Promise<ToolDef[]> {
  const config = loadConfig({ recall_command: join(BIN, "recall-mcp" + EXE) }, dataDir);
  const env = childEnv(config, process.env);
  env.RECALL_EMBEDDER = EMBEDDER;
  if (tools) env.RECALL_MCP_TOOLS = tools;
  const client = new McpClient({
    argv: [resolveExecutable(config.recallCommand, process.env.PATH ?? "", dataDir)],
    env,
    cwd: dataDir,
    stderrPath: join(dataDir, "server.log"),
  });
  try {
    await client.start();
    await client.ready(120_000);
    return await client.listTools(60_000);
  } finally {
    await client.close();
  }
}

describe.skipIf(!ON)("the surface against a real recall-mcp", () => {
  beforeAll(() => {
    dataDir = mkdtempSync(join(tmpdir(), "recall-oc-surface-"));
    mkdirSync(join(dataDir, "memos", "opencode"), { recursive: true });
    writeFileSync(join(dataDir, "memos", "note.md"), "# Office\n\nThe office is in Oslo.\n");
    // The variables the server gets too (audit H-23), so setup and server find the same model cache
    // and the same proxy.
    const env = Object.fromEntries(PASSTHROUGH_ENV.filter((k) => process.env[k] !== undefined).map((k) => [k, process.env[k]!]));
    const setup = Bun.spawnSync(
      [join(BIN, "recall" + EXE), "--embedder", EMBEDDER, "setup", "--lite", "--memory", join(dataDir, "memos"),
        "--store", join(dataDir, "memory.db"), "--no-register"],
      // spawnSync blocks the event loop, so only its own timeout can stop a hung setup (audit H-25); 15
      // minutes, as the end to end allows its setup, inside the hook's bound.
      { cwd: dataDir, env: { ...env, PYTHONSAFEPATH: "1" }, stdout: "pipe", stderr: "pipe", timeout: 900_000 },
    );
    if (setup.exitCode !== 0) throw new Error(`recall setup failed: ${setup.stderr.toString().slice(-2000)}`);
  }, 1_200_000);

  afterAll(() => {
    if (dataDir) rmSync(dataDir, { recursive: true, force: true });
  });

  /** Invariant (spec anti-goal): the six the server serves are, compared as RE-call's surface test
   * compares them, exactly the vendored six; and every committed extra is served.
   *
   * Node: `tests/contract/surface_live.test.ts > the surface against a real recall-mcp > the six
   * served are the vendored six, and every extra is served`. Red proof (2026-11-10, local Windows,
   * hashing embedder, `RECALL_OPENCODE_SURFACE=1 bun run mutations LIVE-TRAINED`): in
   * src/surface.ts:260, `"Delete one memo` became `"Delete a memo`; the test failed at
   * `expect(JSON.stringify(normalizeTool(served.get(trained.name)!))).toBe(...)`. Restored, green. */
  test("the six served are the vendored six, and every extra is served", async () => {
    const tools = await listed();
    const served = new Map(tools.map((t) => [t.name, t]));
    for (const trained of TRAINED) {
      expect(served.has(trained.name)).toBe(true);
      expect(JSON.stringify(normalizeTool(served.get(trained.name)!))).toBe(JSON.stringify(normalizeTool(trained)));
    }
    expect(EXTRA.map((e) => e.name).filter((n) => !served.has(n))).toEqual([]);
    expect(tools.map((t) => t.name).sort()).toEqual(offeredNames(loadConfig({}, dataDir), true).sort());
  }, 300_000);

  /** Invariant: every tool a Lite server serves is classified (trained, kept with a reason, or dropped
   * with a reason), so a change in the server's list fails here.
   *
   * Node: `tests/contract/surface_live.test.ts > the surface against a real recall-mcp > every tool
   * Lite serves is classified, so a changed list fails`. Red proof (2026-11-10, local Windows, hashing
   * embedder, `RECALL_OPENCODE_SURFACE=1 bun run mutations LIVE-CLASSIFIED`): in src/extra_tools.json,
   * the dropped `recall_stats` entry was removed; the test failed at
   * `expect(tools.map((t) => t.name).sort()).toEqual(classified...)`. Restored, green. */
  test("every tool Lite serves is classified, so a changed list fails", async () => {
    const tools = await listed("all");
    const classified = [...TRAINED.map((t) => t.name), ...EXTRA.map((e) => e.name), ...DROPPED.map((d) => d.name)];
    expect(tools.map((t) => t.name).sort()).toEqual(classified.filter((n) => !NOT_ON_LITE.includes(n)).sort());
  }, 300_000);
});
