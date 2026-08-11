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


def has_unclosed_frontmatter(text: str) -> bool:
    """True when the document opens a ``---`` block and never closes it.

    `parse_frontmatter` deliberately returns ``({}, text)`` for this case, which makes it
    indistinguishable from "no frontmatter at all" — and the two need OPPOSITE treatment from
    anything that writes: an absent block should be created, a broken one must not be papered
    over with a second one. A writer that cannot tell them apart prepends a fresh block above the
    damaged one, and the file then declares two different predecessors while retrieval acts on
    the newer.

    Lives here, beside the parser whose ambiguity creates the need, so that every writer shares
    one answer. It previously lived in `rewrite.py`, which guarded `recall rewrite` and left
    `recall lint --fix` writing the same files with the opposite behaviour.
    """
    lines = text.split("\n")
    if not lines or lines[0].lstrip("﻿").strip() != "---":
        return False
    return not any(line.strip() == "---" for line in lines[1:])


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
