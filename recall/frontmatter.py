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


#: A UTF-8 BOM. `parse_frontmatter` above tolerates one before the opening fence because Windows
#: editors add it; a writer that reads `utf-8-sig` and writes plain `utf-8` deletes the very thing
#: that tolerance exists for.
_BOM = b"\xef\xbb\xbf"


def _newline(raw: bytes) -> bytes:
    """The file's line terminator, taken from its first line rather than from the platform."""
    index = raw.find(b"\n")
    if index == -1:
        return b"\n"
    return b"\r\n" if raw[index - 1 : index] == b"\r" else b"\n"


def _terminator(line: bytes, default: bytes) -> bytes:
    """A line's own ending, so an insertion beside it matches it exactly."""
    if line.endswith(b"\r\n"):
        return b"\r\n"
    if line.endswith(b"\n"):
        return b"\n"
    return default  # the final line of a file with no trailing newline


def _is_fence(line: bytes, *, allow_bom: bool = False) -> bool:
    """Whether `line` is a ``---`` fence, judged exactly as `parse_frontmatter` judges it.

    The comparison is done on `str` rather than `bytes` on purpose. `bytes.strip()` removes only
    ASCII whitespace, while `str.strip()` removes Unicode whitespace, so a fence behind a
    non-breaking space is a fence to the reader and not to a bytes-only writer. A writer that
    disagrees with the reader about where the block starts inserts a SECOND block above the first,
    and every key in the original then sits below the reader's stopping point: still in the file,
    no longer metadata. A silently dropped `valid_until` is a memo that never expires.
    """
    text = line.decode("utf-8", "replace")
    if allow_bom:
        text = text.lstrip("﻿")
    return text.strip() == "---"


def insert_frontmatter_line(raw: bytes, key: str, value: str) -> bytes:
    """`key: value` into the frontmatter block, adding one if the file has none.

    Bytes in, bytes out, and that signature is the guard. Every byte-level defect this replaces
    came from a writer that decoded to `str` first: `utf-8-sig` silently ate a Windows memo's BOM,
    splitting and rejoining on ``"\\n"`` normalised every line ending in the document, and a
    text-mode write then translated the result back to the *platform's* endings, so the same input
    produced different files depending on where the tool ran. None of the three is visible in an
    editor; all three are visible in every subsequent diff, which is how a one-line edge
    declaration arrives for review as a total rewrite.

    So: the BOM is carried across untouched rather than decoded away, and the inserted line borrows
    the closing fence's own terminator, so a CRLF memo gains a CRLF line and an LF memo does not.

    This is the one implementation. `recall/fix.py` declares a `supersedes:` edge a memo already
    states in prose; a second writer of the user's own memos must import this rather than carry its
    own copy, because a second copy is how one of them ends up back on `split("\\n")`.
    """
    if any(c in key or c in value for c in ("\n", "\r")):
        # The value reaches here from `fix.py`'s passive branch as a file's relative PATH, verbatim,
        # and a POSIX filename may contain a newline. One would write arbitrary keys, or a second
        # `---`, into a memo the user never edited. Refusing here closes it for every writer at
        # once; refusing per caller is how the next caller gets it wrong.
        raise ValueError(f"a frontmatter line may not contain a line break: {key!r}: {value!r}")
    bom = _BOM if raw.startswith(_BOM) else b""
    body = raw[len(bom) :]
    newline = _newline(body)
    entry = f"{key}: {value}".encode("utf-8")
    lines = body.splitlines(keepends=True)
    if lines and _is_fence(lines[0], allow_bom=True):
        for index, line in enumerate(lines[1:], start=1):
            if _is_fence(line):
                lines.insert(index, entry + _terminator(line, newline))
                return bom + b"".join(lines)
        # unclosed block: treat as no frontmatter rather than corrupt it further
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
