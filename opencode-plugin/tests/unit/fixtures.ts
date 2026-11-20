import { readFileSync } from "node:fs";
import { resolve } from "node:path";

/** Fixtures written by `python -I scripts/gen_fixtures.py` from the vendored Python originals. */
export function fixture<T = any>(name: string): T {
  return JSON.parse(readFileSync(resolve(import.meta.dir, "..", "fixtures", "render", name), "utf-8"));
}
