from __future__ import annotations

from datetime import datetime, timezone

import pytest

from recall.frontmatter import (
    has_line_break,
    insert_frontmatter_line,
    parse_frontmatter,
    validity_bounds,
)

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


# --- a value that would split the line it is written into --------------------------------------
#
# `insert_frontmatter_line` builds `f"{key}: {value}"`, and `recall/fix.py` derives that value
# from a FILENAME. Nothing between the two validates it, so a name carrying a line separator
# writes a line that some readers see as two. The separators below are spelled with `chr()` on
# purpose: a literal one pasted into source does not survive every editor and paste buffer, it
# degrades to a space, and the test then asserts the opposite of what it claims while staying
# green.

#: Everything `str.splitlines()` treats as a line ending. The last eight are the reason this
#: guard is not a `"\n" in value` check: `parse_frontmatter` splits on `"\n"` alone and cannot
#: see them, while `recall/context.py:document_title` calls `splitlines()` and can.
_SEPARATORS = [
    "\n", "\r", "\r\n",
    chr(0x0B), chr(0x0C), chr(0x1C), chr(0x1D), chr(0x1E),
    chr(0x85), chr(0x2028), chr(0x2029),
]

#: Ids built from the WHOLE separator, not its first character. `"\r"` and `"\r\n"` both start
#: 0xd, and pytest silently disambiguates a collision by appending an index, giving `0xd0` and
#: `0xd1` — which read as U+00D0 and U+00D1, so `-k "not 0xd0"` deselects the lone-CR case while
#: appearing to name a Latin-1 letter.
_SEPARATOR_IDS = ["_".join(hex(ord(c)) for c in s) for s in _SEPARATORS]


@pytest.mark.parametrize("sep", _SEPARATORS, ids=_SEPARATOR_IDS)
def test_has_line_break_covers_every_separator_splitlines_honours(sep):
    assert has_line_break(f"evil{sep}injected: yes") is True


def test_has_line_break_is_false_for_an_ordinary_name():
    assert has_line_break("decisions/plan_2026.md") is False


def test_has_line_break_catches_a_separator_at_the_very_end():
    """A `len(text.splitlines()) > 1` test misses this: a trailing separator yields one element.

    It still matters, because the written line ends early and whatever followed it in the block
    becomes a line of its own to a `splitlines()` reader.
    """
    assert has_line_break("plan.md" + chr(0x2028)) is True


@pytest.mark.parametrize("sep", _SEPARATORS, ids=_SEPARATOR_IDS)
def test_insert_frontmatter_line_refuses_a_value_that_would_split(sep):
    """The writer refuses rather than corrupting the block.

    `propose_fixes` reports this case before it gets here, so this is the backstop for any other
    caller that builds a `Proposal` itself.
    """
    with pytest.raises(ValueError, match="line break"):
        insert_frontmatter_line(b"---\nx: 1\n---\nbody\n", "supersedes", f"a{sep}b")


def test_the_two_readers_disagree_about_the_value_the_writer_now_refuses():
    """Why the refusal belongs in the writer, asserted against the READERS themselves.

    This is the whole justification for the guard, so it is checked by calling
    `parse_frontmatter` and `document_title` rather than by restating their line boundaries with
    `str.split` and `str.splitlines`. An earlier version of this test did the latter, and it
    could not have gone red if either reader's notion of a line moved — which is precisely the
    event the guard would need to hear about.

    `parse_frontmatter` splits on `"\\n"` and sees ONE key whose value happens to contain an odd
    character. `document_title` uses `splitlines()`, which honours U+2028, and reads the injected
    text as a `title:` line of its own.
    """
    from recall.context import document_title

    sep = chr(0x2028)
    raw = f"---\nsupersedes: evil{sep}title: pwned\nvalid_from: 2026-01-01\n---\n# real heading\n"
    meta, body = parse_frontmatter(raw)

    assert set(meta) == {"supersedes", "valid_from"}, "the parser sees one key, not two"
    assert meta["supersedes"] == f"evil{sep}title: pwned"

    assert document_title(raw, body, "memo.md") == "pwned", (
        "the injected line is what the other reader takes as the memo's indexed title, which is "
        "the damage this guard exists to prevent"
    )
