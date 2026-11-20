// A scenario: the sessions to run through OpenCode, each a list of user turns, the model that
// answers them, and optionally a store directory to start from. The file form is the same object
// as JSON; `loadScenario` checks its shape once, here, so the driver can trust it.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import type { ScriptedTurn } from "./fake_model";

export type Scenario = {
  name: string;
  store?: string;
  /** `context`: the window the Ollama model is served with (its `num_ctx`), for OpenCode's `limit.context`; default 65536. */
  model: { kind: "scripted"; script: ScriptedTurn[] } | { kind: "ollama"; id: string; context?: number };
  sessions: { turns: string[] }[];
};

function fail(where: string, what: string): never {
  throw new Error(`scenario ${where}: ${what}`);
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/** A misspelt key would otherwise be ignored: a run without its store, or at the default context. */
function onlyKeys(o: Record<string, unknown>, allowed: string[], where: string): void {
  const unknown = Object.keys(o).filter((k) => !allowed.includes(k));
  if (unknown.length > 0) fail(where, `unknown key ${unknown.join(", ")}`);
}

function checkTurn(t: unknown, where: string): ScriptedTurn {
  if (!isObject(t)) fail(where, "a scripted turn must be an object");
  onlyKeys(t, ["text", "toolCalls"], where);
  if (t.text !== undefined && typeof t.text !== "string") fail(where, "text must be a string");
  if (t.toolCalls !== undefined) {
    if (!Array.isArray(t.toolCalls)) fail(where, "toolCalls must be an array");
    t.toolCalls.forEach((c, i) => {
      if (!isObject(c) || typeof c.name !== "string" || !isObject(c.arguments)) {
        fail(`${where}.toolCalls[${i}]`, "a tool call needs a string name and an object arguments");
      }
    });
  }
  if (t.text === undefined && t.toolCalls === undefined) fail(where, "a scripted turn needs text or toolCalls");
  return t as ScriptedTurn;
}

export function parseScenario(raw: unknown): Scenario {
  if (!isObject(raw)) fail("", "must be an object");
  onlyKeys(raw, ["name", "store", "model", "sessions"], "");
  if (typeof raw.name !== "string" || raw.name === "") fail("name", "must be a non-empty string");
  if (raw.store !== undefined && typeof raw.store !== "string") fail("store", "must be a path string");
  const model = raw.model;
  if (!isObject(model)) fail("model", "must be an object");
  if (model.kind === "scripted") {
    onlyKeys(model, ["kind", "script"], "model");
    if (!Array.isArray(model.script)) fail("model.script", "must be an array");
    model.script.forEach((t, i) => checkTurn(t, `model.script[${i}]`));
  } else if (model.kind === "ollama") {
    onlyKeys(model, ["kind", "id", "context"], "model");
    if (typeof model.id !== "string" || model.id === "") fail("model.id", "must be a non-empty string");
    if (model.context !== undefined && !(Number.isInteger(model.context) && (model.context as number) > 0)) {
      fail("model.context", "must be a positive integer");
    }
  } else {
    fail("model.kind", 'must be "scripted" or "ollama"');
  }
  if (!Array.isArray(raw.sessions) || raw.sessions.length === 0) fail("sessions", "must be a non-empty array");
  raw.sessions.forEach((s, i) => {
    if (!isObject(s) || !Array.isArray(s.turns) || s.turns.length === 0) {
      fail(`sessions[${i}]`, "needs a non-empty turns array");
    }
    onlyKeys(s, ["turns"], `sessions[${i}]`);
    s.turns.forEach((t, j) => {
      if (typeof t !== "string") fail(`sessions[${i}].turns[${j}]`, "must be a string");
    });
  });
  return raw as Scenario;
}

/** Reads a scenario file; a relative `store` in it is relative to the file's own directory. */
export function loadScenario(path: string): Scenario {
  const s = parseScenario(JSON.parse(readFileSync(path, "utf8")));
  if (s.store !== undefined) s.store = resolve(dirname(path), s.store);
  return s;
}
