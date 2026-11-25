import { afterEach, describe, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, realpathSync, writeFileSync } from "node:fs";
import { delimiter, join, resolve } from "node:path";

import { childEnv, resolveExecutable } from "../../src/child";
import { loadConfig, storePath } from "../../src/config";
import type { McpClient } from "../../src/mcp_client";
import { fakeClient, fakeScenario, tempDir } from "./fake";

const open: McpClient[] = [];
afterEach(async () => {
  await Promise.all(open.splice(0).map((c) => c.close()));
});

const RECALL_KEYS = [
  "RECALL_ASSISTANT_MEMOS", "RECALL_EMBED_CACHE", "RECALL_EMBEDDER", "RECALL_ENV", "RECALL_INDEX_MODE",
  "RECALL_INDEX_ROOT", "RECALL_MCP_TOOLS", "RECALL_SERVING_DSN", "RECALL_TENANT", "RECALL_TRANSPORT",
  "RECALL_TRUST_MODE",
];
/** recall-hermes e60b0f9 `re_call_hermes/config.py` `PASSTHROUGH_ENV` (unchanged since 8134e23), copied here
 * rather than imported, so a variable added to or dropped from the plugin's list fails this test. */
const HERMES_PASSTHROUGH = [
  "PATH", "SYSTEMROOT", "SystemRoot", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "TMPDIR",
  "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "LANG", "LC_ALL",
  "HF_HOME", "HF_HUB_OFFLINE", "FASTEMBED_CACHE_PATH",
  "RECALL_MODEL_CACHE", "RECALL_MODEL_SHA256", "RECALL_EMBED_PROFILE", "RECALL_SOURCE_POLICY_FILE",
  "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
  "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
];
/** libuv, under Bun's spawn on Windows, copies these from the parent into any child environment that
 * lacks them (measured 2026-11-10, Bun 1.4.4): none is a secret, and none can be withheld. */
const LIBUV_WINDOWS = ["HOMEDRIVE", "HOMEPATH", "LOGONSERVER", "SYSTEMDRIVE", "USERDOMAIN", "USERNAME", "USERPROFILE", "WINDIR"];

function parentEnv(): Record<string, string | undefined> {
  // HF_HUB_OFFLINE stands for a passthrough variable beyond the OS ones; the other two must not reach it.
  return { ...process.env, HF_HUB_OFFLINE: "1", OPENAI_API_KEY: "sk-must-not-leak", NODE_OPTIONS: "--x" };
}

/** Start the fake as the child the plugin would start: in `dataDir`, with `childEnv` only (plus the
 * fake's own two variables, which say where its script and log are). */
async function startChild(dataDir: string) {
  const config = loadConfig({}, dataDir);
  const fake = fakeScenario({ tools: {} });
  const client = fakeClient(fake, { cwd: dataDir, env: childEnv(config, parentEnv()) });
  open.push(client);
  await client.start();
  await client.ready(30_000);
  const start = fake.events().find((e) => e.event === "start")!;
  return { config, start };
}

const norm = (p: string) => (process.platform === "win32" ? realpathSync(p).toLowerCase() : realpathSync(p));

describe("the child process", () => {
  /** Invariant: the child runs in the plugin's data folder and sees only the RE-call settings and the
   * PASSTHROUGH_ENV variables the parent has, never the rest of OpenCode's environment.
   *
   * Node: `tests/unit/child.test.ts > the child process > the child starts in dataDir with only the
   * allowed variables`. Red proof (2026-11-10, local Windows, `bun run mutations CHILD-ENV`): in
   * src/child.ts:75, `const env: Record<string, string> = {};` became `const env: Record<string,
   * string> = { ...(parent as Record<string, string>) };`; the test failed at
   * `expect(seen).toEqual(expected...)`. Restored, green. The same holds for `bun run mutations
   * CHILD-ENV-PASSTHROUGH` (src/child.ts:17, `"HF_HUB_OFFLINE", ` dropped from PASSTHROUGH_ENV) and
   * `CHILD-ENV-PYTHON` (src/child.ts:88, the `PYTHONSAFEPATH: "1",` line removed): each failed at the
   * same `expect(seen).toEqual(...)`, and the test was green again after each restore. */
  test("the child starts in dataDir with only the allowed variables", async () => {
    const dataDir = tempDir();
    const { start } = await startChild(dataDir);
    expect(norm(start.cwd!)).toBe(norm(dataDir));
    const seen = Object.keys(start.env!)
      .filter((k) => !k.startsWith("FAKE_"))
      .map((k) => k.toUpperCase())
      .sort();
    const parent = parentEnv();
    const win = process.platform === "win32";
    const inParent = (k: string) => Object.keys(parent).some((p) => (win ? p.toUpperCase() === k.toUpperCase() : p === k));
    const added = win ? LIBUV_WINDOWS : [];
    const expected = [...RECALL_KEYS, "PYTHONIOENCODING", "PYTHONSAFEPATH", ...HERMES_PASSTHROUGH.filter(inParent), ...added.filter(inParent)];
    expect(seen).toEqual([...new Set(expected.map((k) => k.toUpperCase()))].sort());
    expect(seen).toContain("HF_HUB_OFFLINE");
  }, 20_000);

  /** Invariant: the option `settle_successors` hands the server RE-call's
   * `RECALL_ASSISTANT_SETTLE_SUCCESSORS=1` on a Lite store, and nothing sets it otherwise, so the plugin's
   * `settle` parameter can work where the user turned it on (CCA ENV-003, run 20261125T132233Z).
   *
   * Red proof (2026-11-25, local Windows): against src/child.ts before this change, which never set the
   * variable, the first `expect` failed with `undefined` received. Restored, green. */
  test("settle_successors reaches the server only when turned on", () => {
    const dataDir = tempDir();
    expect(childEnv(loadConfig({ settle_successors: true }, dataDir), {}).RECALL_ASSISTANT_SETTLE_SUCCESSORS).toBe("1");
    expect(childEnv(loadConfig({}, dataDir), { RECALL_ASSISTANT_SETTLE_SUCCESSORS: "1" }).RECALL_ASSISTANT_SETTLE_SUCCESSORS).toBeUndefined();
    expect(() => loadConfig({ settle_successors: "yes" }, dataDir)).toThrow("settle_successors must be true or false");
  });

  /** Invariant (Review Focus 4): a store under a folder with spaces and non-ASCII characters gets a
   * `sqlite:///` DSN that, read the way RE-call reads it (`Path(dsn[len("sqlite:///"):])`), names the
   * store file, after the trip through the child's environment.
   *
   * Node: `tests/unit/child.test.ts > the child process > a store path with spaces and non-ASCII
   * characters gives a sqlite:/// DSN the child accepts`. Red proof (2026-11-10, local Windows, `bun
   * run mutations CHILD-DSN`): in src/config.ts:35, `return LITE_PREFIX + file.replaceAll(...);` became
   * `return LITE_PREFIX + encodeURI(file.replaceAll(...));`; the test failed at
   * `expect(existsSync(storePath(received))).toBe(true)`. Restored, green. */
  test("a store path with spaces and non-ASCII characters gives a sqlite:/// DSN the child accepts", async () => {
    const dataDir = join(tempDir(), "dati di prova è ü 数据");
    mkdirSync(dataDir, { recursive: true });
    writeFileSync(join(dataDir, "memory.db"), "");
    const { start } = await startChild(dataDir);
    const received = Object.entries(start.env!).find(([k]) => k.toUpperCase() === "RECALL_SERVING_DSN")![1];
    expect(received.startsWith("sqlite:///")).toBe(true);
    expect(existsSync(storePath(received))).toBe(true);
    expect(norm(resolve(storePath(received)))).toBe(norm(join(dataDir, "memory.db")));
    expect(norm(start.cwd!)).toBe(norm(dataDir));
  }, 20_000);

  /** Invariant: the source policy `recall setup` applied (`RECALL_SOURCE_POLICY_FILE`) reaches the child
   * unchanged, as recall-hermes 8134e23 passes it, so it is in force on the store every agent shares.
   *
   * Node: `tests/unit/child.test.ts > the child process > the source policy file reaches the child`. Red
   * proof (2026-11-15, local Windows, `bun run mutations CHILD-ENV-POLICY`): in src/child.ts
   * `PASSTHROUGH_ENV`, ` "RECALL_SOURCE_POLICY_FILE",` dropped; the test failed at
   * `expect(env.RECALL_SOURCE_POLICY_FILE).toBe(policy);`. Restored, green. */
  test("the source policy file reaches the child", () => {
    const policy = join(tempDir(), "source policy.toml");
    const env = childEnv(loadConfig({}, tempDir()), { RECALL_SOURCE_POLICY_FILE: policy });
    expect(env.RECALL_SOURCE_POLICY_FILE).toBe(policy);
  });

  /** Invariant: the executable is never taken from the working directory (Hermes SEC-001), and one
   * on PATH comes back absolute.
   *
   * Node: `tests/unit/child.test.ts > the child process > resolveExecutable refuses a recall-mcp found
   * only in cwd`. Red proof (2026-11-10, local Windows, `bun run mutations CHILD-RESOLVE`): in
   * src/child.ts:56, `for (const entry of pathEnv.split(delimiter)) {` became `for (const entry of
   * [cwd, ...pathEnv.split(delimiter)]) {`; the test failed at `expect(() =>
   * resolveExecutable(...)).toThrow("found only in the working directory")`. Restored, green. */
  test("resolveExecutable refuses a recall-mcp found only in cwd", () => {
    const project = tempDir();
    const bin = tempDir();
    const name = process.platform === "win32" ? "recall-mcp.exe" : "recall-mcp";
    writeFileSync(join(project, name), "");
    chmodSync(join(project, name), 0o755);
    const empty = tempDir();
    expect(() => resolveExecutable("recall-mcp", [".", empty].join(delimiter), project)).toThrow(
      "found only in the working directory",
    );
    writeFileSync(join(bin, name), "");
    chmodSync(join(bin, name), 0o755);
    const found = resolveExecutable("recall-mcp", [".", bin].join(delimiter), project);
    expect(norm(found)).toBe(norm(join(bin, name)));
  });

  /** Invariant (audit H-32): on Windows an absolute `recall_command` given without its extension (the
   * install's `Scripts\recall-mcp`) finds `recall-mcp.exe` there, as a PATH entry does; it is not refused.
   * Windows only: elsewhere the bare name is the file.
   *
   * Node: `tests/unit/child.test.ts > the child process > an absolute recall_command without .exe finds
   * the .exe on Windows`. Red proof, row CHILD-ABSOLUTE-PATHEXT: in src/child.ts `resolveExecutable`,
   * `if (found) return resolve(found);` replaced by `void found;`; fails at
   * `expect(got).toBe(norm(join(bin, "recall-mcp.exe")))`. */
  test.skipIf(process.platform !== "win32")("an absolute recall_command without .exe finds the .exe on Windows", () => {
    const bin = tempDir();
    writeFileSync(join(bin, "recall-mcp.exe"), "");
    let got: string;
    try {
      got = norm(resolveExecutable(join(bin, "recall-mcp"), "", bin));
    } catch (error) {
      got = String(error); // caught, so the failure is this expect's, not a thrown error
    }
    expect(got).toBe(norm(join(bin, "recall-mcp.exe")));
  });
});
