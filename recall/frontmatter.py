"""Minimal frontmatter for validity metadata — no YAML dependency.

A document may begin with a ``---`` line, followed by ``key: value`` lines, closed by ``---``.
Only VALIDITY_KEYS and the namespaced ``recall_graph`` JSON object are meaningful to recall;
the graph object may additionally carry an authored ``authority`` tier and exact source
``depends_on`` references. Unknown keys are ignored and the returned body always excludes the block. Dates are ISO
``YYYY-MM-DD``, interpreted in UTC: ``valid_from`` starts at 00:00:00 (inclusive),
``valid_until`` ends at 23:59:59.999999 (inclusive end of day). ``recall_graph`` must be a
single line of JSON so this deliberately small parser does not pretend to be a YAML parser.

``---`` is also markdown's thematic break, so an opening fence is not on its own enough to
declare a block. `frontmatter_span` decides the pairing and is the single definition of it:
every consumer calls it rather than scanning for the next ``---`` itself.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, time, timezone
from collections.abc import Mapping, Sequence
from typing import Any, Literal

VALIDITY_KEYS = ("valid_from", "valid_until", "supersedes")
GRAPH_KEY = "recall_graph"
Authority = Literal[
    "policy",
    "user_confirmed_decision",
    "tool_observation",
    "model_inference",
    "unknown",
]
AUTHORITY_VALUES: tuple[Authority, ...] = (
    "policy",
    "user_confirmed_decision",
    "tool_observation",
    "model_inference",
    "unknown",
)


def _recall_graph_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = metadata.get(GRAPH_KEY)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("recall_graph must be a JSON object")
    if "__parse_error__" in value:
        raise ValueError(str(value["__parse_error__"]))
    return value


def authority_from_metadata(metadata: Mapping[str, Any]) -> Authority:
    """Return the closed authored authority tier, or ``unknown`` when it is absent."""
    graph = _recall_graph_metadata(metadata)
    if graph is None or "authority" not in graph:
        return "unknown"
    value = graph["authority"]
    if not isinstance(value, str) or value not in AUTHORITY_VALUES[:-1]:
        raise ValueError(
            "recall_graph.authority must be one of: " + ", ".join(AUTHORITY_VALUES[:-1])
        )
    return value


def dependencies_from_metadata(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Return sorted exact source dependencies from authored graph metadata."""
    graph = _recall_graph_metadata(metadata)
    if graph is None or "depends_on" not in graph:
        return ()
    value = graph["depends_on"]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("recall_graph.depends_on must be a JSON array of source strings")
    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("recall_graph.depends_on entries must be non-empty strings")
        values.append(item.strip())
    return tuple(sorted(set(values)))

#: A mapping key: a quoted key, or a bare one, then a colon. A bare key may contain spaces
#: (``date created:`` is ordinary Obsidian frontmatter) and may lead with a digit or a non-ASCII
#: letter. What it may NOT lead with is any character markdown uses to open a line: ``#`` ``-``
#: ``*`` ``+`` ``>`` ``|`` `````` and ``[``. None of those is a plausible unquoted YAML key, and
#: excluding them is what stops ``**Warning**: text``, ``[spec]: https://x`` and ``> quoted: x``
#: from reading as keys. That matters more than it looks: see `frontmatter_span` on how one key
#: unlocks the rest of a block.
#: ``-`` and ``+`` are readmitted when the next character is not a space, because a markdown
#: bullet REQUIRES the space: ``-k: x`` is a mapping and ``- k`` is a list item. ``*`` gets no
#: such reprieve, since ``*emphasis*: text`` is ordinary markdown.
#:
#: The dichotomy is not complete, and the gap is paid for knowingly. An unfenced diff paste
#: (``-old_key: value`` / ``+new_key: value``) is neither a bullet nor a mapping, and it now
#: reads as a key, so a section led by one is paired. That is what the old rule did too, while
#: REFUSING ``-k:`` frontmatter would lose its metadata and dump the block into the body. The
#: asymmetry decides it: the reprieve trades a missed improvement for avoiding a regression.
_KEY_LINE = re.compile(r"""(?x)
    (?: ["'] [^"']* ["'] | (?: [-+](?=\S) | [^\s:\#\-*+`\[>|] ) [^:]* )
    \s* :
""")

#: YAML explicit key syntax, ``? key`` on one line and ``: value`` on the next. Neither opens a
#: line in markdown, so both are safe to read as keys wherever they appear.
_EXPLICIT_KEY = re.compile(r"[?:](\s|$)")

#: A YAML block sequence item at column 0. Identical in text to a markdown bullet, so it counts
#: only AFTER a key has been seen: a sequence belongs to the key that opened it, and a bullet
#: list following a bare rule has no key to belong to.
_SEQUENCE_ITEM = re.compile(r"-(\s|$)")

#: The closing bracket of a flow collection written across several lines, at column 0. Same
#: ordering argument as `_SEQUENCE_ITEM`: it belongs to the key that opened the collection.
_FLOW_CLOSER = re.compile(r"[\]}],?\s*$")

#: The line separators `str.splitlines` honours and ``split("\n")`` does not. `document_title`
#: used to split with the former while the span is counted over the latter, so a document
#: carrying one of these is a document where the old and new scans could address different
#: lines. A lone ``\r`` is absent: every production reader uses universal newlines, which
#: translates it before this code sees it.
_EXOTIC_BREAKS = ("\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")


def frontmatter_span(text: str) -> int | None:
    """Index of the line closing a real frontmatter block, or None when there is no block.

    ``---`` opens a frontmatter block AND draws a horizontal rule. Testing only "line 0 is
    ``---`` and some later line is ``---``" pairs two rules that happen to sit either side of a
    section, and everything between them is then deleted from the body with no signal. On a memo
    opening with a rule that is its whole first section, in exchange for an empty ``meta``.

    A block is recognised only when every line before the closing fence is a plausible member,
    AND at least one of them is a key. That last clause is what keeps two adjacent rules apart:
    a block declaring nothing has no metadata to contribute, so pairing it could only ever
    remove body. A member is a blank line, an indented line, a key, or, once a key has been
    seen, a comment or a column 0 block sequence item.

    ORDER carries the weight, because ``- archive`` and ``# Notes`` are a YAML sequence item and
    a comment AND a markdown bullet and heading, with nothing in the text to tell them apart. A
    sequence belongs to the key that opened it, so it counts only after one::

        ---                     ---
        tags:                   <blank>
        - archive               - first point
        valid_until: 2020-01-01 - second point
        ---                     ---
        frontmatter             a rule, a list, another rule

    Refusing is NOT the safe default, and the asymmetry is the whole reason this rule is as
    permissive as it is. Pairing a block the old rule also paired is, at worst, no worse than
    before. REFUSING a block the old rule accepted is strictly worse: the validity metadata is
    lost AND the raw block is handed to the chunker as prose. So every refusal here has to earn
    its place by naming the markdown shape it protects, and digit-leading, non-ASCII, quoted,
    space-containing and explicit keys are all keys.

    What this does NOT fix, stated plainly because the obvious reading of the rule above is more
    generous than the truth:

    - **One key unlocks the rest of the block.** After any key, every comment and sequence item
      is accepted, so a prose section led by a key shaped line and followed by a heading and a
      bullet list is still paired and still deleted. Identical to the old behaviour, so
      `legacy_pairing_differs` is correctly False and nothing is re-indexed.
    - **The bar for "key shaped" is low, and that is the price of the line above it.** Spaces
      are allowed inside a key so that ``date created:`` parses, which means ANY sentence with
      a colon anywhere in it is a key. So is a bare ``http://example.com``, and so is a line
      opening ``:`` or ``?``, both of which render as ordinary paragraphs.

    What IS fixed, then, is narrower than it first looks: a section whose first non-blank line
    is a heading, a bullet, a blockquote, a link reference definition, a table row, or a
    sentence with **no colon in it**. That is the reported defect and the common shape.
    - A ``#`` comment BEFORE the first key is refused: at that position it cannot be told apart
      from a markdown heading, and a heading right after a rule is the commoner document.
    - An unquoted key containing a space is fine, but ``%YAML 1.2`` and a key whose colon is on
      a later line are not, and refuse the block.
    """
    lines = text.split("\n")
    # tolerate a UTF-8 BOM before the opening fence: Windows editors add one, and a BOM
    # that silently disabled frontmatter would mean validity metadata lost without a signal
    if not lines or lines[0].lstrip("\ufeff").strip() != "---":
        return None
    seen_key = False
    for i, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if stripped == "---":
            return i if seen_key else None
        if not stripped:
            continue
        if line[:1].isspace():
            # A continuation or sub-object member. It still counts as a key when it IS one, so a
            # block whose every line is indented is not refused over its indentation alone.
            seen_key = seen_key or bool(_KEY_LINE.match(stripped))
            continue
        if _KEY_LINE.match(line) or _EXPLICIT_KEY.match(line):
            seen_key = True
            continue
        if seen_key and (
            stripped.startswith("#")
            or _SEQUENCE_ITEM.match(line)
            or _FLOW_CLOSER.match(line)
        ):
            continue  # a comment, a sequence item, or a closer belonging to the key above it
        return None  # prose: the opening ``---`` was a thematic break
    return None  # unclosed block: treat the whole text as body


def legacy_pairing_differs(text: str) -> bool:
    """True when the pre-2026-08-11 rule paired a block that `frontmatter_span` now refuses.

    Load bearing for index freshness, NOT dead code. `recall.index` fingerprints a file on its
    raw bytes, so a corpus whose files have not changed is skipped on the next run and would go
    on serving bodies with a section missing. This is the narrowest possible trigger for a
    re-index: it names exactly the files whose body moved, so every other corpus fingerprints
    bit-identically to before and nothing is re-embedded needlessly.

    It stays permanently. Deleting it would revert those files' fingerprints and charge them a
    second re-index, so it is not a migration shim and must not be labelled one.
    """
    head, _, rest = text.partition("\n")
    # The fence is tested BOTH ways the two scans split. Testing only `partition("\n")` lets an
    # exotic break sitting on the first physical line hide the very divergence the next check
    # exists for: that split sees no fence and returns, while the old title scan split with
    # `splitlines`, saw one, and read the block.
    # `head.splitlines()`, not `text.splitlines()`: this runs for every file on every index run,
    # and splitting the whole text allocates a second copy of it before the early return below.
    # The answers are identical, because `splitlines` splits at least as finely as `split("\n")`,
    # so the first line of the finer split always lies inside the first line of the coarser one.
    physical = (head.splitlines() or [""])[0]
    if not any(c.lstrip(chr(0xFEFF)).strip() == "---" for c in (head, physical)):
        return False  # no opening fence, so the old rule did not pair it either
    if any(ch in text for ch in _EXOTIC_BREAKS):
        # The old title scan split on these and the span does not, so the two could address
        # different lines. Flagged on the separator alone rather than on which key moved,
        # because the cheap test is exact enough and the expensive one is not obviously so.
        return True
    if frontmatter_span(text) is not None:
        return False  # still paired, so nothing this file contributes has moved
    lines = rest.split("\n")
    if any(line.strip() == "---" for line in lines):
        return True  # the old rule paired it and the new one does not: the body moved
    # An UNCLOSED block moves no body, because neither rule ever paired it. It can still move
    # the TITLE: `recall.context.document_title` used to scan an unclosed block to its end and
    # take a `title:` out of it, and the title is embedded into every passage in `section` and
    # `neighbor` mode. Without this an existing index would pin the old title permanently.
    return any(
        line[:1].strip() and line.partition(":")[0].strip().lower() == "title" for line in lines
    )


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Split a document into (recognized frontmatter keys, body without the block).

    A document without a leading ``---`` line, with an unclosed block, or whose leading ``---``
    is a thematic break rather than a fence, is returned unchanged as ``({}, text)``.
    `frontmatter_span` is what separates that last case from a real block.
    """
    span = frontmatter_span(text)
    if span is None:
        return {}, text
    lines = text.split("\n")
    meta: dict[str, object] = {}
    for line in lines[1:span]:
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            if key in VALIDITY_KEYS:
                value = value.strip()
                # strip one layer of matching quotes: YAML-habit `supersedes: "v1.md"` must
                # match the unquoted file name, not silently never apply
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1].strip()
                meta[key] = value
            elif key == GRAPH_KEY:
                value = value.strip()
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    meta[key] = {"__parse_error__": "recall_graph must be one-line JSON"}
                else:
                    meta[key] = parsed
    return meta, "\n".join(lines[span + 1 :]).lstrip("\n")


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

    Whether there IS a block is `frontmatter_span`'s call, not a raw fence scan. Scanning forward
    for the next `---` treats a leading horizontal rule as an open block and writes the key into
    the author's prose, which then parses as frontmatter on the next read and deletes the first
    section from retrieval. A document whose opening `---` is a rule therefore takes the same path
    as a document with no frontmatter at all: a real block is added above it, with the rule left
    standing as the first line of the body.
    """
    if has_line_break(value):
        raise ValueError(f"{key} value {value!r} contains a line break and would split the block")
    bom, body = split_bom(raw)
    newline = dominant_newline(body)
    entry = f"{key}: {value}".encode("utf-8")
    lines = split_lines(body)
    # Match the text boundary every reader reaches `frontmatter_span` through: universal newlines
    # have already translated CRLF and a lone CR to LF by the time the parser sees the text.
    text = body.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    span = frontmatter_span(text)
    if span is not None:
        closing = lines[span] if span < len(lines) else b""
        lines.insert(span, entry + line_terminator(closing, newline))
        return bom + b"".join(lines)
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
