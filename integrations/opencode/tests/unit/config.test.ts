import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { loadConfig, sharedStoreDir } from "../../src/config";
import { defaultDataDir } from "../../src/plugin";
import { tempDir } from "./fake";

const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

const lite = (file: string) => "sqlite:///" + file.replaceAll("\\", "/");

/** OpenCode's own folder and the shared one, laid out as under one XDG_CONFIG_HOME, with a store file
 * in each folder named. */
function folders(withStore: ("own" | "shared")[]): { own: string; shared: string } {
  const root = tempDir("recall-oc-config-");
  roots.push(root);
  const dirs = { own: join(root, "opencode", "re-call"), shared: join(root, "re-call") };
  for (const which of withStore) {
    mkdirSync(dirs[which], { recursive: true });
    writeFileSync(join(dirs[which], "memory.db"), "");
  }
  return dirs;
}

describe("loadConfig", () => {
  /** Invariant: no options give Hermes's defaults: k 5, every switch on, the store and the memo folder
   * under the data folder.
   *
   * Node: `tests/unit/config.test.ts > loadConfig > defaults are k 5, every switch on, and a Lite store
   * under dataDir`. Red proof (2026-11-10, local Windows, `bun run mutations CONFIG-DEFAULTS`): in
   * src/config.ts:54, `const k = options.k ?? 5;` became `const k = options.k ?? 10;`; the test failed
   * at `expect(loadConfig({}, dataDir)).toEqual({...})`. Restored, green. */
  test("defaults are k 5, every switch on, and a Lite store under dataDir", () => {
    const dataDir = join(process.platform === "win32" ? "C:\\" : "/", "data dir");
    expect(loadConfig({}, dataDir)).toEqual({
      recallCommand: "recall-mcp",
      dsn: "sqlite:///" + join(dataDir, "memory.db").replaceAll("\\", "/"),
      memoFolder: join(dataDir, "memos"),
      k: 5,
      prefetch: true,
      writeTools: true,
      extraTools: true,
    });
  });

  /** Invariant: relative paths are taken against the data folder, never OpenCode's working directory.
   *
   * Node: `tests/unit/config.test.ts > loadConfig > relative paths resolve against dataDir`. Red proof
   * (2026-11-10, local Windows, `bun run mutations CONFIG-RELATIVE`): in src/config.ts:30, `return
   * isAbsolute(path) ? resolve(path) : resolve(base, path);` became `return resolve(path);`; the test
   * failed at `expect(config.dsn).toBe(...)`. Restored, green. */
  test("relative paths resolve against dataDir", () => {
    const dataDir = join(process.platform === "win32" ? "C:\\" : "/", "data");
    const config = loadConfig({ dsn: "sqlite:///store/m.db", memo_folder: "notes", k: 50, write_tools: false }, dataDir);
    expect(config.dsn).toBe("sqlite:///" + join(dataDir, "store", "m.db").replaceAll("\\", "/"));
    expect(config.memoFolder).toBe(join(dataDir, "notes"));
    expect(config.k).toBe(20);
    expect(config.writeTools).toBe(false);
  });

  /** Invariant: a misspelt or mistyped option is refused rather than ignored.
   *
   * Node: `tests/unit/config.test.ts > loadConfig > an unknown or mistyped option is refused`. Red
   * proof (2026-11-10, local Windows, `bun run mutations CONFIG-UNKNOWN`): in src/config.ts:42, `if
   * (!known.has(key)) throw` became `if (false) throw`; the test failed at `expect(() => loadConfig({
   * write_tool: false }, "/d")).toThrow(...)`. Restored, green. */
  test("an unknown or mistyped option is refused", () => {
    expect(() => loadConfig({ write_tool: false }, "/d")).toThrow('unknown option "write_tool"');
    expect(() => loadConfig({ prefetch: "no" }, "/d")).toThrow("option prefetch must be true or false");
  });
});

/** The shared default store (owner, 2026-11-13, "One shared Lite store"), and the OpenCode only store
 * made before it, which is kept rather than stranded. */
describe("the default store", () => {
  /** Invariant: the shared folder is `<XDG_CONFIG_HOME, else ~/.config>/re-call`, with no `opencode`
   * segment, the folder every agent uses.
   *
   * Node: `tests/unit/config.test.ts > the default store > the shared folder is re-call under
   * XDG_CONFIG_HOME`. Red proof (2026-11-14, local Windows, `bun run mutations CONFIG-SHARED-DIR`): in
   * src/config.ts `sharedStoreDir`, `".config"), "re-call");` became `".config"), "opencode", "re-call");`;
   * the test failed at `expect(sharedStoreDir({ XDG_CONFIG_HOME: xdg })).toBe(join(xdg, "re-call"));`.
   * Restored, green. */
  test("the shared folder is re-call under XDG_CONFIG_HOME", () => {
    const xdg = join(process.platform === "win32" ? "C:\\" : "/", "xdg home");
    expect(sharedStoreDir({ XDG_CONFIG_HOME: xdg })).toBe(join(xdg, "re-call"));
  });

  /** Invariant: XDG_CONFIG_HOME counts only when recall-hermes `shared_dir` would use it (`os.path.isabs`),
   * else `~/.config` is used, so OpenCode and Hermes resolve one shared folder, not two (recall-claw audit
   * DAT-003 and ENV-002, 2026-11-15). `windows` is Python 3.13.0 and 3.14.6 `ntpath.isabs` on each value,
   * measured 2026-11-15 on each value with `py -3.13 -c "import ntpath; print([ntpath.isabs(v) for v in
   * (r'\rooted', '/rooted', 'C:foo', r'C:\abs', r'\\server\share')])"` and the rest of the list; `posix` is a
   * leading `/`, as `posixpath.isabs` reads it. Python 3.11 and 3.12 differ on `\rooted` and `/rooted`,
   * which they call absolute.
   *
   * Node: `tests/unit/config.test.ts > the default store > XDG_CONFIG_HOME counts only when Hermes would
   * call it absolute`. Red proof (2026-11-15, local Windows), each failing at `expect(cases.map((c) =>
   * sharedStoreDir({ XDG_CONFIG_HOME: c.xdg }))).toEqual(cases.map(expected));` and green once restored:
   * row CONFIG-XDG-ANY (`pythonIsAbs(xdg) ? xdg :` became `xdg ? xdg :`, the value taken in any form),
   * CONFIG-XDG-ROOTED (Python 3.12's rule, a leading separator counted) and CONFIG-XDG-UNC (two leading
   * separators no longer counted). */
  test("XDG_CONFIG_HOME counts only when Hermes would call it absolute", () => {
    const cases = [
      { xdg: "", windows: false, posix: false },
      { xdg: "rel", windows: false, posix: false },
      { xdg: ".\\rel", windows: false, posix: false },
      { xdg: "\\rooted", windows: false, posix: false },
      { xdg: "/rooted", windows: false, posix: true },
      { xdg: "C:foo", windows: false, posix: false },
      { xdg: "C:", windows: false, posix: false },
      { xdg: "C:\\abs", windows: true, posix: false },
      { xdg: "c:/abs", windows: true, posix: false },
      { xdg: "\\\\server\\share", windows: true, posix: false },
      { xdg: "//server/share", windows: true, posix: true },
      { xdg: "\\\\?\\C:\\abs", windows: true, posix: false },
    ];
    const win = process.platform === "win32";
    const expected = (c: (typeof cases)[number]) => join((win ? c.windows : c.posix) ? c.xdg : join(homedir(), ".config"), "re-call");
    expect(cases.map((c) => sharedStoreDir({ XDG_CONFIG_HOME: c.xdg }))).toEqual(cases.map(expected));
    expect(sharedStoreDir({})).toBe(join(homedir(), ".config", "re-call"));
  });

  test("the private data directory ignores relative XDG_CONFIG_HOME", () => {
    const fallback = join(homedir(), ".config", "opencode", "re-call");
    expect(defaultDataDir({ XDG_CONFIG_HOME: "relative/config" })).toBe(fallback);
    if (process.platform === "win32") {
      expect(defaultDataDir({ XDG_CONFIG_HOME: "\\rooted" })).toBe(fallback);
      expect(defaultDataDir({ XDG_CONFIG_HOME: "C:relative" })).toBe(fallback);
    }
    const absolute = join(process.platform === "win32" ? "C:\\" : "/", "xdg home");
    expect(defaultDataDir({ XDG_CONFIG_HOME: absolute })).toBe(join(absolute, "opencode", "re-call"));
  });

  /** Invariant: with no `dsn`, a store in the shared folder is used, with its memos, even when an older
   * store is still in OpenCode's folder, and nothing is logged.
   *
   * Node: `tests/unit/config.test.ts > the default store > the shared store is chosen when present`. Red
   * proof (2026-11-14, local Windows, `bun run mutations CONFIG-SHARED-CHOSEN`): in src/config.ts
   * `loadConfig`, `if (!sharedHasStore && existsSync(` became `if (existsSync(` (the older store
   * preferred); the test failed at `expect({ dsn: config.dsn, memoFolder: config.memoFolder, notice:
   * config.notice }).toEqual({`. Restored, green. Row CONFIG-MEMOS-FOLLOW-STORE: `: join(storeDir,
   * "memos"),` became `: join(dataDir, "memos"),` (the memo root left behind, so the index root and the
   * assistant's memos would split from the store); fails at the same `expect`. */
  test("the shared store is chosen when present", () => {
    const { own, shared } = folders(["own", "shared"]);
    const config = loadConfig({}, own, shared);
    expect({ dsn: config.dsn, memoFolder: config.memoFolder, notice: config.notice }).toEqual({
      dsn: lite(join(shared, "memory.db")),
      memoFolder: join(shared, "memos"),
      notice: undefined,
    });
  });

  /** Invariant: with no `dsn` and a store only in OpenCode's folder, that store and its memos are kept,
   * and one line names both paths and how to move it: every agent closed, the store file with its two
   * sidecars, and the memos folder.
   *
   * Node: `tests/unit/config.test.ts > the default store > an older OpenCode store is kept and the move
   * is logged in one line`. Red proof (2026-11-14, local Windows, `bun run mutations CONFIG-OLD-KEPT`):
   * in src/config.ts `loadConfig`, `} else storeDir = sharedDir;` became `} storeDir = sharedDir;`
   * (the older store dropped for the shared path); the test failed at `expect({ dsn: config.dsn,
   * memoFolder: config.memoFolder, notice: config.notice }).toEqual({`. Restored, green. */
  test("an older OpenCode store is kept and the move is logged in one line", () => {
    const { own, shared } = folders(["own"]);
    const config = loadConfig({}, own, shared);
    expect({ dsn: config.dsn, memoFolder: config.memoFolder, notice: config.notice }).toEqual({
      dsn: lite(join(own, "memory.db")),
      memoFolder: join(own, "memos"),
      notice:
        `re-call: using the older store ${join(own, "memory.db")}, not the shared one ${join(shared, "memory.db")}; ` +
        `to share it, close every agent, then move memory.db, memory.db-wal, memory.db-shm and the memos folder from ${own} to ${shared}`,
    });
    expect(config.notice).not.toContain("\n");
  });

  /** Invariant: with no `dsn` and no store anywhere, the default is the shared folder, so the "no store"
   * line (src/plugin.ts `setupCommand`) names the shared store and memos.
   *
   * Node: `tests/unit/config.test.ts > the default store > the shared default applies when neither store
   * exists`. Red proof (2026-11-14, local Windows, `bun run mutations CONFIG-SHARED-DEFAULT`): in
   * src/config.ts `loadConfig`, `if (!sharedHasStore && existsSync(join(dataDir, "memory.db"))) {` became
   * `if (!sharedHasStore) {` (OpenCode's folder kept whenever the shared one is empty); the test failed
   * at `expect({ dsn: config.dsn, memoFolder: config.memoFolder, notice: config.notice }).toEqual({`.
   * Restored, green. */
  test("the shared default applies when neither store exists", () => {
    const { own, shared } = folders([]);
    const config = loadConfig({}, own, shared);
    expect({ dsn: config.dsn, memoFolder: config.memoFolder, notice: config.notice }).toEqual({
      dsn: lite(join(shared, "memory.db")),
      memoFolder: join(shared, "memos"),
      notice: undefined,
    });
  });

  /** Invariant: an explicit `dsn` wins over both folders, and nothing else moves with it: the memo
   * default stays where it was before the shared store, and nothing is logged.
   *
   * Node: `tests/unit/config.test.ts > the default store > an explicit dsn wins`. Red proof (2026-11-14,
   * local Windows, `bun run mutations CONFIG-DSN-WINS`): in src/config.ts `loadConfig`, `if (!dsn &&
   * sharedDir !== undefined) {` became `if (sharedDir !== undefined) {`; the test failed at `expect({
   * dsn: config.dsn, memoFolder: config.memoFolder, notice: config.notice }).toEqual({`. Restored, green. */
  test("an explicit dsn wins", () => {
    const { own, shared } = folders(["own"]);
    const elsewhere = join(own, "..", "elsewhere.db");
    const config = loadConfig({ dsn: lite(elsewhere) }, own, shared);
    expect({ dsn: config.dsn, memoFolder: config.memoFolder, notice: config.notice }).toEqual({
      dsn: lite(join(own, "..", "elsewhere.db")),
      memoFolder: join(own, "memos"),
      notice: undefined,
    });
  });

  test("an explicit shared Lite DSN takes its shared memo root", () => {
    const { own, shared } = folders(["own", "shared"]);
    const config = loadConfig({ dsn: lite(join(shared, "memory.db")) }, own, shared);
    expect(config.memoFolder).toBe(join(shared, "memos"));
  });

  test("a memo override selects the matching default Lite store", () => {
    const { own, shared } = folders(["own", "shared"]);
    const config = loadConfig({ memo_folder: "private-memos" }, own, shared);
    expect([config.dsn, config.memoFolder]).toEqual([lite(join(own, "memory.db")), join(own, "private-memos")]);
    const { own: old, shared: empty } = folders(["own"]);
    expect(loadConfig({ memo_folder: join(empty, "memos") }, old, empty).dsn).toBe(lite(join(empty, "memory.db")));
  });

  test("an aliased shared memo path keeps the same Lite store before and after creation", () => {
    const root = tempDir("recall-oc-config-");
    roots.push(root);
    const actual = join(root, "actual");
    const alias = join(root, "alias");
    mkdirSync(actual);
    symlinkSync(actual, alias, process.platform === "win32" ? "junction" : "dir");
    const own = join(root, "opencode", "re-call");
    const shared = join(actual, "re-call");
    const options = { memo_folder: join(alias, "re-call", "memos") };
    const before = loadConfig(options, own, shared).dsn;
    mkdirSync(join(shared, "memos"), { recursive: true });
    const after = loadConfig(options, own, shared).dsn;
    const expected = lite(join(shared, "memory.db"));
    expect([before, after]).toEqual([expected, expected]);
  });

  // FA-2-9 red at ced240de0: loadConfig chose the private DSN only before target creation.
  test("a dangling memo alias keeps the shared Lite store before and after its target exists", () => {
    const root = tempDir("recall-oc-config-");
    roots.push(root);
    const actual = join(root, "actual");
    const alias = join(root, "alias");
    symlinkSync(actual, alias, process.platform === "win32" ? "junction" : "dir");
    const own = join(root, "opencode", "re-call");
    const shared = join(actual, "re-call");
    const options = { memo_folder: join(alias, "re-call", "memos") };
    const before = loadConfig(options, own, shared).dsn;
    mkdirSync(join(shared, "memos"), { recursive: true });
    const after = loadConfig(options, own, shared).dsn;
    const expected = lite(join(shared, "memory.db"));
    expect([before, after]).toEqual([expected, expected]);
  });

  test("explicit Lite pairs cannot mix shared and private paths", () => {
    const { own, shared } = folders(["own", "shared"]);
    const sharedDsn = lite(join(shared, "memory.db"));
    const privateDsn = lite(join(own, "memory.db"));
    const sharedMemo = join(shared, "memos");
    for (const options of [
      { dsn: sharedDsn, memo_folder: "private-memos" },
      { dsn: privateDsn, memo_folder: sharedMemo },
    ]) {
      expect(() => loadConfig(options, own, shared)).toThrow("the shared Lite store and its memo folder must be used together");
    }
    expect(loadConfig({ dsn: privateDsn, memo_folder: "private-memos" }, own, shared).memoFolder).toBe(join(own, "private-memos"));
    expect(loadConfig({ dsn: "postgresql://localhost/recall", memo_folder: sharedMemo }, own, shared).memoFolder).toBe(sharedMemo);
  });
});
