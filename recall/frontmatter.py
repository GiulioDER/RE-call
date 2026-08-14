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


def has_line_break(text: str) -> bool:
    """True when `text` holds anything ``str.splitlines()`` treats as the end of a line.

    Deliberately NOT a hand-listed subset, and deliberately not ``"\\n" in text``. ``\\n`` and
    ``\\r`` are the notion of a line break `parse_frontmatter` uses, and that notion is too narrow
    to protect the file: ``str.splitlines()`` honours eight more characters (``\\x0b \\x0c \\x1c
    \\x1d \\x1e \\x85`` and U+2028, U+2029), so a value carrying one of those is invisible to the
    parser while still splitting the written line for every reader that uses ``splitlines()``.
    `recall/context.py:document_title` is one such reader, and it decides a memo's indexed title.

    Phrasing it as "the text must survive ``splitlines()`` unchanged" keeps it correct for any
    reader on that boundary rather than for a list someone has to remember to extend. Comparing
    the JOINED result also catches a TRAILING separator, which a ``len(...) > 1`` test misses.
    """
    return text != "".join(text.splitlines())


def insert_frontmatter_line(raw: bytes, key: str, value: str) -> bytes:
    """`key: value` into the frontmatter block, adding a block if the file has none.

    Refuses a `value` carrying a line break rather than writing a line that splits. The value
    reaches here from a FILENAME (see `recall/fix.py`), so it is attacker-controlled wherever a
    corpus is not entirely hand-authored, and a separator in it writes a second key that the
    parser cannot see but `splitlines()` readers can. `propose_fixes` reports that case before it
    gets this far; this is the backstop for any other caller that builds a `Proposal` itself.

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
    if has_line_break(value):
        raise ValueError(f"{key} value {value!r} contains a line break and would split the block")
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


#: Prefix for the stand-in. NUL cannot occur in a path on either platform — Python refuses to
#: build a `Path` from one — so no real corpus name can be mistaken for a stand-in, and no name
#: that needs no stand-in ever gets one. `recall/truth_extraction/_cache.py` marks its own for
#: the same reason; the two markers differ because that one stands in for file CONTENT too.
NAME_STAND_IN_MARK = "\x00name:"


def encodable_name(name: str) -> str:
    """``name`` itself when it encodes as UTF-8, and a marked stand-in when it does not.

    A POSIX filename is bytes, not text. One that is not valid UTF-8 arrives as a lone
    surrogate through `Path.glob`'s surrogateescape, and everything downstream of a corpus name
    eventually encodes it: `canonical_sha256` hashes it into a `reasoning_graph` node id,
    `canonical_json` serialises it, an HTTP client would send it. Each of those raised
    `UnicodeEncodeError`, and each was patched where it raised — a cache key here, a `print`
    there — while the next consumer kept inheriting the same landmine. `recall rewrite plan`
    was the third, and it did not degrade: one such file in a corpus emptied the whole review
    queue. This is the boundary those consumers share, so the guard belongs here.

    Two properties, and the second is the one that is easy to lose:

    **A name that encodes gets no stand-in, with no exceptions.** The output is hashed into the
    proposal ids a reviewer types into `recall rewrite apply`, so rewriting names that do not
    need it would renumber queues that already exist. `caffè.md` is good UTF-8 and is returned
    untouched, and so is the POSIX-legal `policy\\update.md`, whose backslash makes it look like
    an escape without making it one.

    **INJECTIVE**, or the guard against a crash quietly costs a document. Two documents sharing
    one key in the caller's dict and one node id in the graph is the collision the
    corpus-relative key exists to prevent, and an earlier version of this function reintroduced
    it: the stand-in was the bare backslash escape, which a file may legitimately be NAMED. The
    marker is what separates the two ranges, and it holds a NUL precisely because no filename
    can. Backslashes are doubled INSIDE the stand-in as well, so the escape stays reversible
    and the map is one to one over arbitrary strings rather than only over real paths.

    The stand-in stays readable after the marker, spelled the way `recall`'s streams already
    print such a name — they reconfigure with ``errors="backslashreplace"`` — because a mangled
    name beats no name. It is a name for the queue and never one for the corpus: `rewrite`
    refuses to write it into a memo, where no other reader would resolve it.

    **Per path SEGMENT**, because `name` is often a corpus-relative path and `supersedes_key`
    reduces one to the stem of its last segment. A marker on the front of the whole path is
    thrown away by that reduction: `sub/bad<surrogate>.md` was a marked reference and an
    unmarked file, `_resolve` could never match the two, and `claim_key` merged it with the
    file literally NAMED `sub/bad\\udcff.md` — the collision the marker exists to prevent,
    re-entering through the normaliser rather than through this map. Marking the segment that
    needs it survives the reduction, and stays one to one: the escape never emits ``/``, so
    splitting and rejoining is a bijection over the segments it is applied to.

    ⚠️ The marker's NUL is invisible to SQL. SQLite's string functions treat TEXT as NUL
    terminated, so a stand-in stored in a column round trips through parameter binding intact
    while `length()` reads 0 and `LIKE` never matches it. Nothing in `recall` queries such a
    column — the extraction cache stores `file` for a human to SELECT — but an operator
    inspecting one by hand will find these rows blank rather than absent.
    """
    return "/".join(_encodable_segment(segment) for segment in name.split("/"))


def _encodable_segment(segment: str) -> str:
    try:
        segment.encode("utf-8")
    except UnicodeEncodeError:
        pass
    else:
        if NAME_STAND_IN_MARK not in segment:
            return segment
    escaped = segment.replace("\\", "\\\\").encode("utf-8", "backslashreplace").decode("utf-8")
    return NAME_STAND_IN_MARK + escaped


def writable_reference(value: str) -> str:
    """The value as a memo can carry it, with the segments no reader looks at dropped.

    `supersedes_key` reduces a reference to the stem of its LAST segment, and `rewrite._resolve`,
    `lint`, `check`, `fix` and the store all compare through it, so `legal/old.md` and `old.md`
    name the same document to every one of them. That makes a directory whose name is not valid
    UTF-8 a different case from a FILE whose name is not: the basename is still a reference every
    reader resolves, and refusing it invented a restriction this package does not have, about a
    file whose own name was never the problem. Verified by asking a reader rather than by
    reasoning: `rewrite verify` resolves the edge this writes.

    Only when the marker survives into the last segment is there nothing writable left. Saying so
    is left to the CALLER, because the two writers of the user's memos refuse in different
    vocabularies — `rewrite` raises `RewriteRefused`, `fix` records an `Unfixable` its dry run
    prints — while "is any of this writable" has one answer, and this module is where both
    writers already share one. Each names that surviving segment when it refuses: the file's own
    name is what has to change. Ambiguity is unaffected either way, because the stem is what was
    being compared already, so dropping the directory cannot make two documents collide that did
    not collide before.

    A value carrying no marker is returned untouched, directories and all. Trimming those too
    would read the same to every resolver and still be wrong: a derived block's dedup recognises
    the line it wrote by comparing the value, so a spelling that changes between runs is an entry
    appended forever.

    The invariant is CHECKED rather than assumed, by asking `supersedes_key` whether the trim
    changed anything. A bare corpus name is not the only shape a value arrives in: `[[name]]` is
    what the corpus's own author writes and what a provider can hand over, and splitting that on
    the last `/` cuts inside the brackets, taking the marker with the discarded half and leaving
    `old.md]]`, which resolves to nothing. Refusing it whole is the honest outcome. Rewriting it
    into `[[old.md]]` would be this module editing a reference it was handed, which is where a
    writer starts guessing at what a human meant.
    """
    if NAME_STAND_IN_MARK not in value:
        return value
    trimmed = value.rsplit("/", 1)[-1]
    return trimmed if supersedes_key(trimmed) == supersedes_key(value) else value
