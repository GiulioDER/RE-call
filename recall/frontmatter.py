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
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return meta, "\n".join(lines[i + 1 :]).lstrip("\n")
        if ":" in line:
            key, _, value = line.partition(":")
            if key.strip() in VALIDITY_KEYS:
                meta[key.strip()] = value.strip()
    return {}, text  # unclosed block: treat the whole text as body


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
