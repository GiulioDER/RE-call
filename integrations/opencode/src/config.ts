/** The plugin's options from `opencode.json`, validated once, with Hermes's defaults. */

import { existsSync, readlinkSync, realpathSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";

export type Config = {
  recallCommand: string;
  dsn: string;
  memoFolder: string;
  k: number;
  prefetch: boolean;
  writeTools: boolean;
  extraTools: boolean;
  /** One line to log when an older OpenCode only store is kept instead of the shared one. */
  notice?: string;
};

const LITE_PREFIX = "sqlite:///";
const STRINGS = { recall_command: "recallCommand", dsn: "dsn", memo_folder: "memoFolder" } as const;
const BOOLEANS = { prefetch: "prefetch", write_tools: "writeTools", extra_tools: "extraTools" } as const;

export function isLiteDsn(dsn: string): boolean {
  return dsn.startsWith(LITE_PREFIX);
}

/** The SQLite file a lite DSN names; RE-call reads it as `Path(dsn[len("sqlite:///"):])`. */
export function storePath(dsn: string): string {
  return dsn.slice(LITE_PREFIX.length);
}

/** Python 3.13's `os.path.isabs`, which recall-hermes `shared_dir` applies to XDG_CONFIG_HOME: on Windows
 * a drive with a root (any one character before the colon) or two leading separators (UNC, `\\?\`), so
 * `\rooted` and `C:foo` are not absolute; elsewhere a leading `/`. */
function pythonIsAbs(path: string): boolean {
  if (process.platform !== "win32") return isAbsolute(path);
  const head = path.slice(0, 3).replaceAll("/", "\\");
  return head.startsWith(":\\", 1) || head.startsWith("\\\\");
}

/** The per-user store every agent uses by default (owner, 2026-11-13: "One shared Lite store").
 * XDG_CONFIG_HOME counts only when absolute, as Hermes reads it, so both resolve the same folder. */
export function sharedStoreDir(env: Record<string, string | undefined> = process.env): string {
  return join(configHome(env.XDG_CONFIG_HOME), "re-call");
}

export function configHome(xdg = "", home = homedir()): string {
  return pythonIsAbs(xdg) ? xdg : join(home, ".config");
}

/** A user path made absolute against the data folder, which is the child's cwd too (as Hermes does). */
function absolute(path: string, base: string): string {
  return isAbsolute(path) ? resolve(path) : resolve(base, path);
}

function pathIdentity(path: string): string {
  const absolutePath = resolve(path);
  let ancestor = absolutePath;
  const suffix: string[] = [];
  let links = 0;
  let resolved = absolutePath;
  // Resolve the existing prefix and any dangling links before appending missing components.
  while (!existsSync(ancestor)) {
    const parent = dirname(ancestor);
    if (parent === ancestor) break;
    let target: string | undefined;
    try {
      target = readlinkSync(ancestor);
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code !== "ENOENT" && code !== "ENOTDIR" && code !== "EINVAL" && code !== "ELOOP") throw error;
    }
    if (target !== undefined) {
      if (++links > 40) throw new Error(`re-call: too many symbolic links resolving ${absolutePath}`);
      ancestor = isAbsolute(target) ? target : resolve(realpathSync(parent), target);
      continue;
    }
    suffix.push(basename(ancestor));
    ancestor = parent;
  }
  if (existsSync(ancestor)) resolved = join(realpathSync(ancestor), ...suffix.reverse());
  return process.platform === "win32" ? resolved.toLowerCase() : resolved;
}

function samePath(left: string, right: string): boolean {
  return pathIdentity(left) === pathIdentity(right);
}

/** RE-call's documented form: `sqlite:///C:/x/memory.db` on Windows, `sqlite:////home/x/memory.db` elsewhere. */
function liteDsn(file: string): string {
  return LITE_PREFIX + file.replaceAll("\\", "/");
}

/** `dataDir` is OpenCode's own folder: relative paths resolve against it, and without `sharedDir` the
 * default store is there. A custom memo folder uses the OpenCode store unless a DSN is supplied.
 * With neither option set, an older OpenCode store is kept if the shared store is absent. */
export function loadConfig(options: Record<string, unknown>, dataDir: string, sharedDir?: string): Config {
  const known = new Set<string>([...Object.keys(STRINGS), ...Object.keys(BOOLEANS), "k"]);
  for (const key of Object.keys(options)) {
    // A misspelt option silently ignored would, for `write_tools`, offer what the user turned off.
    if (!known.has(key)) throw new Error(`re-call: unknown option ${JSON.stringify(key)}`);
  }
  const text = (key: keyof typeof STRINGS): string => {
    const value = options[key] ?? "";
    if (typeof value !== "string") throw new Error(`re-call: option ${key} must be a string`);
    return value.trim();
  };
  const flag = (key: keyof typeof BOOLEANS): boolean => {
    const value = options[key] ?? true;
    if (typeof value !== "boolean") throw new Error(`re-call: option ${key} must be true or false`);
    return value;
  };
  const k = options.k ?? 5;
  if (typeof k !== "number" || !Number.isInteger(k)) throw new Error("re-call: option k must be an integer");

  const memoOption = text("memo_folder");
  const memo = memoOption ? absolute(memoOption, dataDir) : undefined;
  let dsn = text("dsn");
  const sharedMemo = sharedDir === undefined ? undefined : join(sharedDir, "memos");
  const usesSharedMemo = (!dsn || isLiteDsn(dsn)) && memo !== undefined && sharedMemo !== undefined && samePath(memo, sharedMemo);
  let storeDir = dataDir;
  let notice: string | undefined;
  if (!dsn && sharedDir !== undefined) {
    const sharedHasStore = existsSync(join(sharedDir, "memory.db"));
    if (usesSharedMemo) storeDir = sharedDir;
    else if (memo !== undefined) storeDir = dataDir;
    else if (!sharedHasStore && existsSync(join(dataDir, "memory.db"))) {
      // RE-call's docs/LITE.md: the -wal and -shm sidecars go with the store file.
      notice = `re-call: using the older store ${join(dataDir, "memory.db")}, not the shared one ${join(sharedDir, "memory.db")}; to share it, close every agent, then move memory.db, memory.db-wal, memory.db-shm and the memos folder from ${dataDir} to ${sharedDir}`;
    } else storeDir = sharedDir;
  }
  if (!dsn) dsn = liteDsn(join(storeDir, "memory.db"));
  else if (isLiteDsn(dsn)) dsn = liteDsn(absolute(storePath(dsn), dataDir));
  if (isLiteDsn(dsn) && sharedDir !== undefined) {
    const usesSharedStore = samePath(storePath(dsn), join(sharedDir, "memory.db"));
    if (memo !== undefined && usesSharedStore !== usesSharedMemo) {
      throw new Error("re-call: the shared Lite store and its memo folder must be used together; set dsn and memo_folder to a private pair or use both shared paths");
    }
    if (usesSharedStore) storeDir = sharedDir;
  }
  return {
    recallCommand: text("recall_command") || "recall-mcp",
    dsn,
    memoFolder: memo ?? join(storeDir, "memos"),
    k: Math.max(1, Math.min(k, 20)), // Hermes's bounds for the same setting
    prefetch: flag("prefetch"),
    writeTools: flag("write_tools"),
    extraTools: flag("extra_tools"),
    ...(notice ? { notice } : {}),
  };
}
