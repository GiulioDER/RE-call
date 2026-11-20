/** Spawning `tests/fake_mcp_server.py` (a scripted `recall-mcp`) and reading its event log. */

import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { McpClient, type McpClientOptions } from "../../src/mcp_client";

export const FAKE_SERVER = resolve(import.meta.dir, "..", "fake_mcp_server.py");
const PYTHON_NAME = process.env.PYTHON ?? (process.platform === "win32" ? "python" : "python3");
export const PYTHON = Bun.which(PYTHON_NAME) ?? PYTHON_NAME;

export type FakeEvent = { event: string; tool?: string; arguments?: Record<string, unknown>; cwd?: string; env?: Record<string, string> };

export function tempDir(prefix = "recall-oc-"): string {
  return mkdtempSync(join(tmpdir(), prefix));
}

export type Fake = { scenario: string; log: string; env: Record<string, string>; events: () => FakeEvent[] };

export function fakeScenario(scenario: object, dir = tempDir()): Fake {
  const path = join(dir, "scenario.json");
  const log = join(dir, "fake.log");
  writeFileSync(path, JSON.stringify(scenario), "utf-8");
  writeFileSync(log, "", "utf-8");
  const events = () =>
    readFileSync(log, "utf-8")
      .split("\n")
      .filter(Boolean)
      .map((line) => JSON.parse(line) as FakeEvent);
  return { scenario: path, log, env: { FAKE_SCENARIO: path, FAKE_LOG: log }, events };
}

/** The parent's variables a Python child needs to start. The fake logs its whole environment to a file
 * in a temp dir, so the rest of the parent's (its tokens included) must not reach it. */
const START_VARIABLES = ["PATH", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "HOME", "LANG"];

/** A client over the fake, with only START_VARIABLES of the parent's environment unless `env` is given. */
export function fakeClient(fake: Fake, extra: Partial<McpClientOptions> = {}): McpClient {
  const base = Object.fromEntries(START_VARIABLES.flatMap((n) => (process.env[n] === undefined ? [] : [[n, process.env[n]!]])));
  return new McpClient({
    argv: [PYTHON, FAKE_SERVER],
    cwd: tempDir(),
    ...extra,
    env: { ...(extra.env ?? base), ...fake.env },
  });
}
