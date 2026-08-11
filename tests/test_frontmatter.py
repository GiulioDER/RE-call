from __future__ import annotations

from datetime import datetime, timezone

import pytest

from recall.frontmatter import insert_frontmatter_line, parse_frontmatter, validity_bounds

DOC = """---
valid_from: 2026-06-01
valid_until: 2026-06-30
supersedes: old_policy.md
color: blue
---
# Body title

Body paragraph.
"""


def test_parse_frontmatter_extracts_keys_and_strips_block():
    meta, body = parse_frontmatter(DOC)
    assert meta == {
        "valid_from": "2026-06-01",
        "valid_until": "2026-06-30",
        "supersedes": "old_policy.md",
    }
    assert body.startswith("# Body title")
    assert "---" not in body
    assert "color" not in meta  # unknown keys ignored


def test_no_frontmatter_returns_empty_meta_and_full_text():
    text = "# Just a doc\n\nNo block here."
    assert parse_frontmatter(text) == ({}, text)


def test_unclosed_block_treated_as_body():
    text = "---\nvalid_until: 2026-01-01\nno closing fence"
    assert parse_frontmatter(text) == ({}, text)


def test_validity_bounds_inclusive_utc_day_bounds():
    start, end = validity_bounds({"valid_from": "2026-06-01", "valid_until": "2026-06-30"})
    assert start == datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert end is not None
    assert end.tzinfo is not None
    assert end.date().isoformat() == "2026-06-30"
    assert end.hour == 23 and end.minute == 59 and end.second == 59
    # a moment inside the last day is still valid
    assert start <= datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc) <= end


def test_validity_bounds_absent_keys_are_none():
    assert validity_bounds({}) == (None, None)
    start, end = validity_bounds({"supersedes": "x.md"})
    assert start is None and end is None


def test_validity_bounds_malformed_date_raises():
    with pytest.raises(ValueError, match="valid_until"):
        validity_bounds({"valid_until": "June 30th"})


def test_bom_does_not_disable_frontmatter():
    meta, body = parse_frontmatter("﻿---\nsupersedes: old.md\n---\nbody text")
    assert meta == {"supersedes": "old.md"}
    assert body == "body text"


def test_quoted_values_are_unquoted():
    # YAML-habit quoting must match unquoted file names, not silently never apply
    meta, _ = parse_frontmatter("---\nsupersedes: \"v1.md\"\nvalid_from: '2026-01-01'\n---\nx")
    assert meta["supersedes"] == "v1.md"
    assert meta["valid_from"] == "2026-01-01"


# --- insert_frontmatter_line: the writer must agree with the reader ----------------------------
#
# `parse_frontmatter` decides what a fence IS. Any writer that decides differently will insert a
# second block above a block the reader can see, and every key in the original is then below the
# reader's stopping point: still in the file, no longer metadata. A lost `valid_until` is a memo
# that never expires.


@pytest.mark.parametrize(
    "prefix, label",
    [
        (" ", "a non-breaking space"),
        ("　", "an ideographic space"),
        ("﻿﻿", "a doubled BOM"),
    ],
)
def test_a_fence_the_reader_accepts_is_a_fence_to_the_writer(prefix, label):
    """`str.strip()` drops Unicode whitespace; `bytes.strip()` drops only ASCII.

    The reader has always used the former, so it accepts these as an opening fence. A writer using
    the latter does not, and prepends a second block instead of inserting into the first.
    """
    text = f"{prefix}---\nvalid_until: 2026-01-01\n---\nbody\n"
    assert parse_frontmatter(text)[0] == {"valid_until": "2026-01-01"}, f"premise: {label}"

    out = insert_frontmatter_line(text.encode("utf-8"), "supersedes", "new_2026").decode("utf-8")
    meta, _ = parse_frontmatter(out)

    assert meta["supersedes"] == "new_2026"
    assert meta["valid_until"] == "2026-01-01", f"the existing key was orphaned behind {label}"
    assert out.count("---") == text.count("---"), "no second block may be created"


@pytest.mark.parametrize("bad", ["a\nvalid_until: 1999-01-01\nb", "e.md\n---\n\nHIJACKED", "a\rb"])
def test_a_value_carrying_a_line_break_is_refused(bad):
    """A newline in the value writes arbitrary keys, or a fake fence, into someone else's memo.

    `fix.py`'s passive branch sets the value to the *evidence file's relative path*, verbatim. A
    filename may contain a line separator, and `pathlib`'s glob still collects it, so this is
    reachable without the user writing anything unusual in prose. Refusing at the shared helper
    closes it for every writer at once rather than once per caller.
    """
    with pytest.raises(ValueError, match="line break"):
        insert_frontmatter_line(b"---\nvalid_until: 2030-01-01\n---\nbody\n", "supersedes", bad)


#: Every separator `str.splitlines()` honours beyond ``\n`` and ``\r``. `parse_frontmatter` splits
#: on ``"\n"`` and does not care, but `context.document_title` uses `splitlines()`, so to that
#: reader each of these DOES end the line.
_SPLITLINES_ONLY = ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", chr(0x2028), chr(0x2029)]


@pytest.mark.parametrize("sep", _SPLITLINES_ONLY)
def test_every_separator_splitlines_honours_is_refused(sep):
    """Guarding only ``\\n`` and ``\\r`` was too narrow for the readers this package already has.

    `context.document_title` scans the same block with `str.splitlines()`, which treats eight more
    characters as line ends. A value carrying one passed the first version of this guard and split
    the written line for that scanner, letting a crafted filename set the indexed title. U+2028 is
    a legal filename character on NTFS as well as POSIX, so this is not a Unix-only concern.

    The check is now "the text must survive `splitlines()` unchanged", which is the same notion of
    a line break the reader uses rather than a hand-listed subset of it.
    """
    with pytest.raises(ValueError, match="line break"):
        insert_frontmatter_line(
            b"---\nvalid_until: 2030-01-01\n---\nbody\n", "supersedes", f"note{sep}title: INJECTED"
        )


def test_an_ordinary_value_is_still_accepted():
    """The widened refusal must not swallow the values this tool actually writes."""
    out = insert_frontmatter_line(
        b"---\nvalid_until: 2030-01-01\n---\nbody\n", "supersedes", "sub dir/old_plan_2026-01-01.md"
    )
    assert parse_frontmatter(out.decode("utf-8"))[0]["supersedes"] == \
        "sub dir/old_plan_2026-01-01.md"
