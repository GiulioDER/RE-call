"""Minimal frontmatter for validity metadata — no YAML dependency.

A document may begin with a ``---`` line, followed by ``key: value`` lines, closed by ``---``.
Only VALIDITY_KEYS are meaningful to recall; unknown keys are ignored and the returned body
always excludes the block. Dates are ISO ``YYYY-MM-DD``, interpreted in UTC: ``valid_from``
starts at 00:00:00 (inclusive), ``valid_until`` ends at 23:59:59.999999 (inclusive end of day).
"""
from __future__ import annotations

from datetime import datetime, time, timezone

VALIDITY_KEYS = ("valid_from", "valid_until", "supersedes")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a document into (recognized frontmatter keys, body without the block).

    A document without a leading ``---`` line — or with an unclosed block — is returned
    unchanged as ``({}, text)``.
    """
    lines = text.split("\n")
    # tolerate a UTF-8 BOM before the opening fence — Windows editors add one, and a BOM
    # that silently disabled frontmatter would mean validity metadata lost without a signal
    if not lines or lines[0].lstrip("\ufeff").strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return meta, "\n".join(lines[i + 1 :]).lstrip("\n")
        if ":" in line:
            key, _, value = line.partition(":")
            if key.strip() in VALIDITY_KEYS:
                value = value.strip()
                # strip one layer of matching quotes: YAML-habit `supersedes: "v1.md"` must
                # match the unquoted file name, not silently never apply
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1].strip()
                meta[key.strip()] = value
    return {}, text  # unclosed block: treat the whole text as body


# --- writing the block back, on bytes ------------------------------------------------------------
#
# These live HERE, beside the parser, rather than in either module that writes memos. Both writers
# — `recall/fix.py`, declaring an edge a memo already states in prose, and `recall/rewrite.py`,
# writing back a reviewed promotion — edit the block this file defines, and every byte-level defect
# either of them has had came from disagreeing with `parse_frontmatter` about what a line is or
# where a BOM ends. Sharing the definitions is the only version of that fix which stays fixed.
#
# `recall/fix.py` is reached from `recall lint --fix`, which `recall/cli.py` documents as a "pure
# filesystem check — no embedder, no DB". This module imports nothing but `datetime`, so it can be
# the shared home; `recall/rewrite.py` cannot, because it pulls in `recall.trust` and with it the
# whole retrieval stack.

BOM = b"\xef\xbb\xbf"


def split_bom(raw: bytes) -> tuple[bytes, bytes]:
    """`(leading BOMs, the rest)`, counting them the way `parse_frontmatter` does.

    `parse_frontmatter` uses ``lstrip("\\ufeff")``, which removes ANY number. A writer that removed
    exactly one saw the second BOM sitting ahead of the fence, concluded the file had no
    frontmatter and prepended a fresh block — orphaning the authored `valid_until` into the body,
    where the trust layer can no longer read it, so an expired memo silently became live again.
    """
    end = 0
    while raw.startswith(BOM, end):
        end += len(BOM)
    return raw[:end], raw[end:]


def split_lines(data: bytes) -> list[bytes]:
    """Split on CRLF, LF or a lone CR, keeping terminators. Universal newlines, done on bytes.

    The boundary has to match how the corpus is READ, which is not the same as how
    `parse_frontmatter` splits the string it is handed. Every reader in the package —
    `lint.py`, `index.py`, `check.py`, `semantic_lint.py` and `fix.py`'s own `propose_fixes` —
    calls `Path.read_text` with the default ``newline=None``, so Python translates a lone CR to LF
    *before* `parse_frontmatter` ever runs. A CR-terminated memo therefore does have frontmatter
    as far as recall is concerned.

    An LF-only split disagreed with that: it read such a file as one long line, found no block,
    and prepended a SECOND one — orphaning the authored `valid_until` into the body, where the
    trust layer cannot see it, so a memo expired since 2020 silently went live again.

    `bytes.splitlines` is exactly that boundary and no more: unlike `str.splitlines` it does NOT
    break on VT, FF, the ASCII separators or U+0085, which is what makes it safe here — universal
    newlines does not translate those either, so the two agree. A hand-rolled loop lived here
    briefly on the mistaken belief that it had to; it was verified equivalent to this call across
    every candidate separator and 20 000 random byte strings, and a hand-rolled parser nobody
    needs is just somewhere else for a bug to live.

    `keepends` is the other half: rejoining with ``b"".join`` reproduces the input exactly.
    """
    return data.splitlines(keepends=True)


def dominant_newline(raw: bytes) -> bytes:
    """The file's line terminator, taken from its first line rather than from the platform."""
    for index, char in enumerate(raw):
        if char == 0x0D:
            return b"\r\n" if raw[index + 1 : index + 2] == b"\n" else b"\r"
        if char == 0x0A:
            return b"\n"
    return b"\n"


def line_terminator(line: bytes, default: bytes) -> bytes:
    """A line's own ending, so an insertion beside it matches it exactly."""
    if line.endswith(b"\r\n"):
        return b"\r\n"
    if line.endswith(b"\n"):
        return b"\n"
    if line.endswith(b"\r"):
        return b"\r"
    return default  # the final line of a file with no terminator


def is_fence(line: bytes) -> bool:
    """Whether `line` is a `---` frontmatter fence, judged the way `parse_frontmatter` judges it.

    Compared as TEXT, because `parse_frontmatter` uses `str.strip`, which removes NBSP,
    U+3000 and the rest of Unicode's whitespace, while `bytes.strip` removes ASCII only. A memo
    whose fence picked up a trailing NBSP — which is what pasting out of a browser or a word
    processor does — was seen as a block by the parser and as prose by the byte writer, which then
    prepended a second block and orphaned the real one.
    """
    return line.decode("utf-8", "replace").strip() == "---"


def insert_frontmatter_line(raw: bytes, key: str, value: str) -> bytes:
    """`key: value` into the frontmatter block, adding a block if the file has none.

    Bytes in, bytes out. Decoding to `str`, splitting on ``"\\n"`` and re-encoding is how a memo
    loses its BOM and has every line ending in it rewritten — both invisible when the result is
    read back through `read_text`, and both present in the user's next diff. The BOM prefix is
    carried across untouched and the inserted line borrows the closing fence's own terminator, so
    a CRLF memo gains a CRLF line.

    The search for the closing fence is unbounded, and deliberately so. A memo whose body contains
    a `---` thematic break within what looks like a block gets the key written next to prose,
    which is startling — but `parse_frontmatter` makes exactly the same reading, and a writer that
    disagreed with the parser would put the key somewhere the trust layer cannot see it and
    rewrite it again on every run. The parser defines what the block is; this follows it. Tighten
    both together or neither.
    """
    bom, body = split_bom(raw)
    newline = dominant_newline(body)
    entry = f"{key}: {value}".encode("utf-8")
    lines = split_lines(body)
    if lines and is_fence(lines[0]):
        for index, line in enumerate(lines[1:], start=1):
            if is_fence(line):
                lines.insert(index, entry + line_terminator(line, newline))
                return bom + b"".join(lines)
        # unclosed block — treat as no frontmatter rather than corrupt it further
    return bom + b"---" + newline + entry + newline + b"---" + newline + body


def _parse_date(value: str, key: str) -> datetime:
    try:
        d = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"bad {key} date {value!r} (expected YYYY-MM-DD)") from exc
    return d.replace(tzinfo=timezone.utc)


def validity_bounds(meta: dict) -> tuple[datetime | None, datetime | None]:
    """Interpret a chunk's validity metadata as tz-aware UTC (start, end) bounds.

    Either bound is None when its key is absent. Raises ValueError on a malformed date.
    """
    start = end = None
    if v := meta.get("valid_from"):
        start = _parse_date(str(v), "valid_from")
    if v := meta.get("valid_until"):
        end = datetime.combine(_parse_date(str(v), "valid_until").date(), time.max, timezone.utc)
    return start, end


def supersedes_key(value: str) -> str:
    """Normalise a ``supersedes:`` target to the key both the linter and the store match on.

    The reference is authored by a human, and on a real 792-memo corpus **every** declared edge
    failed to resolve because of how it was written — not because the target was missing:

    - ``supersedes: [project_lrp_maker_2026-06-24]`` — wikilink brackets, kept verbatim
    - ``supersedes: project-recall-abstention-...-2026-07-18`` — no ``.md``, while the corpus
      matched on full basenames

    Both targets existed. `recall lint` reported "does not exist in the corpus", which was
    actively misleading. A convention that the corpus's own author cannot follow is a defect in
    the convention: strip the wrapping and compare on the STEM, so `name`, `name.md`, `[name]`
    and `[[name]]` all mean the same document.

    Ambiguity handling is unchanged — two files sharing a stem are still refused rather than
    guessed at.
    """
    v = value.strip()
    while len(v) >= 2 and v[0] == "[" and v[-1] == "]":
        v = v[1:-1].strip()  # handles both [name] and [[name]]
    if v.lower().endswith(".md"):
        v = v[:-3]
    return v.rsplit("/", 1)[-1].strip()


def encodable_name(name: str) -> str:
    """``name`` itself when it encodes as UTF-8, and a readable stand-in when it does not.

    A POSIX filename is bytes, not text. One that is not valid UTF-8 arrives as a lone
    surrogate through `Path.glob`'s surrogateescape, and everything downstream of a corpus name
    eventually encodes it: `canonical_sha256` hashes it into a `reasoning_graph` node id,
    `canonical_json` serialises it, an HTTP client would send it. Each of those raised
    `UnicodeEncodeError`, and each was patched where it raised — a cache key here, a `print`
    there — while the next consumer kept inheriting the same landmine. `recall rewrite plan`
    was the third, and it did not degrade: one such file in a corpus emptied the whole review
    queue. This is the boundary those consumers share, so the guard belongs here.

    Two properties, and the second is the one that is easy to lose:

    **A name that needs no stand-in gets none.** The output is hashed into the proposal ids a
    reviewer types into `recall rewrite apply`, so rewriting every name would renumber every
    queue that already exists. `caffè.md` is good UTF-8 and is returned untouched.

    **INJECTIVE**, or the guard against a crash quietly costs a document. The stand-in is the
    backslash escape a reader already sees, because `recall`'s streams print with
    ``errors="backslashreplace"`` and a mangled name beats no name. But a file may legitimately
    be NAMED ``bad\\udcff.md``, and mapping the surrogate onto it would give two documents one
    key in the caller's dict and one node id in the graph — the same collision the
    corpus-relative key exists to prevent. So a name already carrying that escape's own
    characters is diverted too, and its backslashes are doubled: a diverted output always
    contains ``\\u``, a passed-through one never does, and doubling makes the escape reversible.
    That is the single case where an ordinary name's id changes, and it is a name that cannot
    be told apart from an escape.
    """
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        pass
    else:
        if "\\u" not in name:
            return name
    return name.replace("\\", "\\\\").encode("utf-8", "backslashreplace").decode("utf-8")
