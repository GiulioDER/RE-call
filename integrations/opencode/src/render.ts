/** recall-hermes 86fdc528 `re_call_hermes/render.py`, ported line for line.
 *
 * Turn RE-call's compact `recall_search` answer into the block injected before a turn. Three outcomes,
 * and the model must be able to tell them apart: citable hits (`evidence.citable`), retired hits
 * (superseded or expired, flagged so the model does not state them as current), and abstention.
 *
 * The Python helpers below reproduce what the original gets from Python itself, where JavaScript differs:
 * truthiness (`[]` and `{}` are false), `str()` of scalars, `str.split()` and `str.strip()` whitespace, and
 * lengths and slices counted in code points. Not reproduced, all for values that are not strings:
 * - `str()` of a JSON object or array;
 * - `str()` of a float that JSON wrote with a fractional zero (`1.0`), which JavaScript cannot tell from
 *   `1` once parsed;
 * - `str()` of a float Python writes in exponent form: Python prints `1e+16` and `1e-05` where
 *   JavaScript prints `10000000000000000` and `0.00001`;
 * - a number or a boolean as `hits` or as `evidence.citable`: Python raises TypeError iterating it, here it
 *   counts as no hits (abstention) or no citable ids;
 * - `str()` of an integer JavaScript cannot hold exactly (beyond 2^53, rounded when the answer is parsed),
 *   which can also make a numeric `chunk_id` match a citable id Python would not match;
 * - `str()` of an integer from 1e21 up, which JavaScript prints in exponent form (`1e+21`).
 * RE-call's compact view sends strings for every field the renderer prints. */

export const HEADER = "## RE-call memory";
export const RETIRED_VERDICTS = new Set(["superseded", "expired"]);
export const MAX_HIT_CHARS = 600;

export const ABSTAIN_LINE =
  "RE-call abstained: nothing in memory passed its calibrated trust check for this message. " +
  "Do not answer questions about the present from memory.";
export const CITABLE_LEAD =
  "These memories passed RE-call's calibrated trust check for this message. " + "Name the memo id when you rely on one.";
export const RETIRED_LEAD = "Retired memories (superseded or expired): do not state these as current.";

/** The injected text and how many citable memories it holds. */
export type Rendered = { text: string; citable: number };

// -- Python semantics --------------------------------------------------------------------------

/** What `str.split()`, `str.strip()` and `str.isspace()` count as whitespace (and `\s` in a `str`
 * pattern): Unicode White_Space less U+180E, plus U+001C to U+001F. JavaScript's `\s` differs: it
 * counts U+FEFF and not U+001C to U+001F or U+0085. A regex character class body. */
export const PY_WS = "\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const WS_RUN = new RegExp(`[${PY_WS}]+`);
const WS_EDGES = new RegExp(`^[${PY_WS}]+|[${PY_WS}]+$`, "g");
const WS_ONE = new RegExp(`^[${PY_WS}]$`);

export const pyStrip = (s: string): string => s.replace(WS_EDGES, "");
export const pyIsSpace = (ch: string | undefined): boolean => ch !== undefined && WS_ONE.test(ch);

export function isMapping(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

export function pyTruthy(v: unknown): boolean {
  if (Array.isArray(v)) return v.length > 0;
  if (isMapping(v)) return Object.keys(v).length > 0;
  return Boolean(v);
}

export function pyStr(v: unknown): string {
  if (v === null || v === undefined) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  if (typeof v === "string") return v;
  if (typeof v === "number") return String(v);
  return JSON.stringify(v);
}

/** Iterating a value as Python does: a list's items, a string's code points, a dict's keys. */
function pyIter(v: unknown): unknown[] {
  if (Array.isArray(v)) return v;
  if (typeof v === "string") return Array.from(v);
  if (isMapping(v)) return Object.keys(v);
  return [];
}

// -- render.py ---------------------------------------------------------------------------------

function clip(text: unknown): string {
  const flat = (pyTruthy(text) ? pyStr(text) : "").split(WS_RUN).filter(Boolean).join(" ");
  const points = Array.from(flat);
  return points.length <= MAX_HIT_CHARS ? flat : points.slice(0, MAX_HIT_CHARS - 3).join("") + "...";
}

function validity(hit: Record<string, unknown>): string {
  const parts: string[] = [];
  if (pyTruthy(hit.valid_from)) parts.push(`valid from ${pyStr(hit.valid_from)}`);
  if (pyTruthy(hit.valid_until)) parts.push(`valid until ${pyStr(hit.valid_until)}`);
  return parts.length ? ` (${parts.join(", ")})` : "";
}

function citableIds(answer: Record<string, unknown>): Set<string> | null {
  const evidence = answer.evidence;
  if (!isMapping(evidence)) return null; // a server run with RECALL_SEARCH_EVIDENCE=off sends none
  return new Set(pyIter(pyTruthy(evidence.citable) ? evidence.citable : []).map(pyStr));
}

const textOf = (h: Record<string, unknown>): unknown => ("text" in h ? h.text : "");

/** The prefetch block for one compact `recall_search` answer. */
export function renderSearch(answer: unknown): Rendered {
  const a = isMapping(answer) ? answer : {};
  const hits = pyIter(pyTruthy(a.hits) ? a.hits : []).filter(isMapping);
  const ids = citableIds(a);
  const abstained = pyTruthy(a.abstained);

  const citable: Record<string, unknown>[] = [];
  const retired: Record<string, unknown>[] = [];
  for (const hit of hits) {
    const verdict = pyTruthy(hit.verdict) ? pyStr(hit.verdict) : "";
    if (RETIRED_VERDICTS.has(verdict)) {
      retired.push(hit);
    } else if (!abstained && verdict === "ok" && (ids === null || ids.has(pyStr(hit.chunk_id)))) {
      citable.push(hit);
    }
  }

  const lines = [HEADER];
  if (citable.length) {
    lines.push(CITABLE_LEAD);
    for (const h of citable) lines.push(`- [ok] memo ${pyStr(h.memo_id)}${validity(h)}: ${clip(textOf(h))}`);
  } else {
    lines.push(ABSTAIN_LINE);
  }
  if (retired.length) {
    lines.push(RETIRED_LEAD);
    for (const h of retired) {
      const replaced = pyTruthy(h.superseded_by) ? `, replaced by ${pyStr(h.superseded_by)}` : "";
      lines.push(`- [${pyStr(h.verdict)}${replaced}] memo ${pyStr(h.memo_id)}${validity(h)}: ${clip(textOf(h))}`);
    }
  }
  return { text: lines.join("\n"), citable: citable.length };
}

/** One line when the server refused the search (for example, a store not yet calibrated). */
export function renderUnavailable(reason: string): Rendered {
  return { text: `${HEADER}\nRE-call memory could not answer this turn: ${clip(reason)}`, citable: 0 };
}
