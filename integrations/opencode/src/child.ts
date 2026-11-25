/** How the `recall-mcp` child is found and what environment it gets. */

import { accessSync, constants, statSync } from "node:fs";
import { basename, delimiter, dirname, extname, isAbsolute, join, resolve } from "node:path";

import { type Config, isLiteDsn, storePath } from "./config";
import { offeredNames } from "./surface";

/** The only variables of OpenCode's environment the child sees, each copied when present: recall-hermes
 * e60b0f9 `PASSTHROUGH_ENV` (`re_call_hermes/config.py`, as of 8134e23), verbatim. The OS and cache variables
 * docs/LITE.md lists, PATH, TMPDIR and the locale; the model cache, model settings and source policy
 * `recall setup` used, so the server finds the model cached and applies the policy; and the proxy and CA
 * variables, so a first download works behind a proxy. Nothing else, provider keys included, reaches it. */
export const PASSTHROUGH_ENV = [
  "PATH", "SYSTEMROOT", "SystemRoot", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "TMPDIR",
  "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "LANG", "LC_ALL",
  "HF_HOME", "HF_HUB_OFFLINE", "FASTEMBED_CACHE_PATH",
  "RECALL_MODEL_CACHE", "RECALL_MODEL_SHA256", "RECALL_EMBED_PROFILE", "RECALL_SOURCE_POLICY_FILE",
  "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
  "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
] as const;
/** The assistant's memos go in a subfolder of the memo root, as Hermes keeps its own in `hermes/`. */
export const ASSISTANT_SUBFOLDER = "opencode";

function isExecutableFile(path: string): boolean {
  try {
    if (!statSync(path).isFile()) return false;
    if (process.platform !== "win32") accessSync(path, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

function candidates(dir: string, name: string): string[] {
  if (process.platform !== "win32") return [join(dir, name)];
  const exts = (process.env.PATHEXT || ".COM;.EXE;.BAT;.CMD").split(";").filter(Boolean);
  // `python3.12` still needs `.exe`: only a known extension counts as one.
  if (exts.some((e) => e.toUpperCase() === extname(name).toUpperCase())) return [join(dir, name)];
  return exts.map((e) => join(dir, name + e));
}

/** `cmd` as an absolute path, searched in PATH's absolute entries only. Windows' own search, and any
 * relative PATH entry, would try the working directory first, so a project could plant the server
 * (Hermes SEC-001). */
export function resolveExecutable(cmd: string, pathEnv: string, cwd: string): string {
  const name = cmd.trim();
  if (!name) throw new Error("re-call: recall_command is empty");
  if (isAbsolute(name)) {
    if (isExecutableFile(name)) return resolve(name);
    // An absolute path without its extension gets PATHEXT on Windows, as a PATH entry does.
    const found = candidates(dirname(name), basename(name)).find(isExecutableFile);
    if (found) return resolve(found);
    throw new Error(`re-call: ${name} is not an executable file`);
  }
  if (name.includes("/") || name.includes("\\")) {
    throw new Error(`re-call: recall_command ${name} is a relative path; give an absolute one`);
  }
  for (const entry of pathEnv.split(delimiter)) {
    if (!entry || !isAbsolute(entry)) continue;
    const found = candidates(entry, name).find(isExecutableFile);
    if (found) return found;
  }
  if (candidates(cwd, name).some(isExecutableFile)) {
    throw new Error(`re-call: ${name} was found only in the working directory ${cwd}, which is refused`);
  }
  throw new Error(`re-call: ${name} was not found on PATH; install recall-rag[fastembed,mcp]`);
}

/** Case-insensitive on Windows, where the parent may spell PATH as `Path`. */
function lookup(parent: Record<string, string | undefined>, key: string): string | undefined {
  if (parent[key] !== undefined || process.platform !== "win32") return parent[key];
  const found = Object.keys(parent).find((k) => k.toUpperCase() === key.toUpperCase());
  return found === undefined ? undefined : parent[found];
}

export function childEnv(config: Config, parent: Record<string, string | undefined>): Record<string, string> {
  const env: Record<string, string> = {};
  const copied = new Set<string>();
  for (const key of PASSTHROUGH_ENV) {
    // Windows names are case-insensitive: SYSTEMROOT and SystemRoot are one variable, copied once.
    const name = process.platform === "win32" ? key.toUpperCase() : key;
    const value = lookup(parent, key);
    if (value === undefined || copied.has(name)) continue;
    copied.add(name);
    env[key] = value;
  }
  // As Hermes sets them: no working directory on sys.path, and UTF-8 on the child's stdio; then the
  // settings docs/LITE.md lists for a stdio client, at RE-call v0.15.0.
  Object.assign(env, {
    PYTHONSAFEPATH: "1",
    PYTHONIOENCODING: "utf-8",
    RECALL_SERVING_DSN: config.dsn,
    RECALL_EMBEDDER: "fastembed",
    RECALL_TENANT: "default",
    RECALL_INDEX_MODE: "legacy",
    RECALL_ENV: "development",
    RECALL_TRUST_MODE: "strict",
    RECALL_TRANSPORT: "stdio",
    RECALL_MCP_TOOLS: offeredNames(config, isLiteDsn(config.dsn)).join(","), // exactly what the plugin may offer
  });
  if (isLiteDsn(config.dsn)) {
    env.RECALL_INDEX_ROOT = config.memoFolder;
    env.RECALL_ASSISTANT_MEMOS = join(config.memoFolder, ASSISTANT_SUBFOLDER);
    // The server's half of `recall_remember(settle=true)`: without it every settle is refused.
    if (config.settleSuccessors) env.RECALL_ASSISTANT_SETTLE_SUCCESSORS = "1";
    // It holds vectors of memo text, so it stays beside the store, as Hermes keeps it with the
    // profile's memory rather than under the platform cache directory.
    env.RECALL_EMBED_CACHE = join(dirname(storePath(config.dsn)), "embeddings.sqlite");
  }
  return env;
}
