"""Minimal, dependency-free `.env` loader for local dev.

Reads ``KEY=VALUE`` lines from a `.env` file into ``os.environ`` WITHOUT overriding variables that
are already set. The `.env` file is gitignored — it is for local secrets (e.g. VOYAGE_API_KEY),
never committed. Entry points call ``load_dotenv()`` so those keys are picked up.
"""
from __future__ import annotations

import os
from pathlib import Path


#: What `recall.setup._quote_env` writes, mapped back. Keep the two in step: a value this cannot
#: undo is a value the writer must not produce.
_ESCAPES = {"\\": "\\", '"': '"', "n": "\n", "r": "\r"}


def _unquote(raw: str) -> str:
    """Undo the quoting `_quote_env` applies, and nothing else.

    This used to be `val.strip().strip('"').strip("'")`, which is wrong in two ways at once on a
    value the writer escaped. It leaves the backslashes in place, so `my model \\"quoted\\"` comes
    back carrying them; and `str.strip` removes EVERY trailing quote it finds, including the one
    that belonged to an escape, so the value also loses a character off the end. A model id or a
    base URL containing a quote therefore did not survive the trip it had just made.

    An unbalanced or unquoted value is returned as it stands. Guessing at a repair for a
    hand-written line would be worse than handing back what is actually there.
    """
    val = raw.strip()
    if len(val) < 2 or val[0] != val[-1] or val[0] not in "\"'":
        return val
    inner = val[1:-1]
    if val[0] == "'":
        # `_quote_env` never emits single quotes, so there is nothing to unescape inside them.
        return inner
    out: list[str] = []
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == "\\" and i + 1 < len(inner):
            nxt = inner[i + 1]
            # An unrecognised escape keeps both characters rather than silently eating the
            # backslash: this parser's job is fidelity, not interpretation.
            out.append(_ESCAPES.get(nxt, "\\" + nxt))
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def load_dotenv(path: str | Path = ".env") -> None:
    """Apply a `.env` ALL-OR-NOTHING: parse the whole file, then set the variables.

    This used to assign into `os.environ` inside the parse loop, one key at a time. A malformed
    line therefore left every earlier key applied and every later one dropped, and because the
    caller treats a failure as "no .env", that half-configured state was silent. Measured
    consequence: a file whose second line held a NUL byte set `VOYAGE_API_KEY` (live, billing)
    while `RECALL_SERVING_DSN` never arrived, so the DSN fell back to the local default, passed
    the insecure-DSN guard because it is local, and indexed into the wrong database.

    A NUL is the realistic trigger, from a `.env` truncated by a crash mid-write: it is valid
    UTF-8, so `read_text` succeeds and `os.environ.__setitem__` is what raises.

    Two things a first version of this rewrite got wrong, found by audit, both from checking
    a key's fate too late:

    * A key already in `os.environ` was never going to be applied, so it must not be able to
      fail the whole file. The first version ran the NUL check on every parsed key regardless,
      so a corrupt byte on a line that would have been SKIPPED anyway still discarded a file
      that previously loaded completely.
    * A key repeated in the file used to resolve to its FIRST occurrence — the incremental
      version applied line 1, then skipped line 2 because the key was already set. Collecting
      into a dict and assigning `pending[key] = val` on every line silently flipped that to
      LAST-wins, with no test covering a repeated key to catch it. `recall/setup.py` appends
      its block to the end of an existing `.env`, so this is not a hypothetical: a hand-written
      line above the block used to win and would have started losing to it.

    Both are fixed the same way: decide whether a line matters (not already exported, not
    already claimed by an earlier line in this file) BEFORE parsing its content for validity.
    """
    p = Path(path)
    if not p.exists():
        return
    pending: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = _unquote(val)
        if not key or key in os.environ or key in pending:
            # An exported variable always wins over the file, and so does the file's OWN
            # first occurrence of a key — checked before validity, so a line that was never
            # going to apply cannot fail one that already has.
            continue
        if "\x00" in key or "\x00" in val:
            raise ValueError(f"{p}: embedded null character in {key!r}")
        pending[key] = val
    for key, val in pending.items():
        os.environ[key] = val
