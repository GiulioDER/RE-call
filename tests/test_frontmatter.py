from __future__ import annotations

from datetime import datetime, timezone

import pytest

from recall.frontmatter import encodable_name, parse_frontmatter, validity_bounds

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


@pytest.mark.parametrize(
    "name",
    ["old_2026-01-01.md", "legal/policy.md", "caffè.md", "a\\b.md"],
)
def test_encodable_name_returns_an_ordinary_name_unchanged(name):
    """Every name a real corpus holds, byte for byte.

    The output is hashed into `reasoning_graph` node ids and therefore into the proposal ids a
    reviewer types into `recall rewrite apply`. A stand-in applied to names that do not need
    one would renumber every queue that already exists, which reads as "the tool forgot".
    Non-ASCII is unchanged too: `caffè.md` is perfectly good UTF-8 and needs no escaping, and
    so is a backslash that no escape could be confused with.
    """
    assert encodable_name(name) == name


def test_encodable_name_is_encodable_for_a_name_that_is_not_valid_utf8():
    assert encodable_name("bad\udcff.md").encode("utf-8") == b"bad\\udcff.md"


def test_encodable_name_does_not_collapse_two_names_onto_one():
    """INJECTIVE, or a corpus loses a file rather than a hash.

    The stand-in is the escape a reader already sees on stderr, and a file may legitimately be
    NAMED with that escape's own characters. Mapping `bad<surrogate>.md` onto the literal
    `bad\\udcff.md` would give two different documents one dict key in `corpus_documents` and
    one node id in the graph: the second silently replaces the first, which is the collision
    the corpus-relative key exists to prevent, reintroduced by the guard against the crash.
    """
    surrogate = "bad\udcff.md"
    impostor = "bad\\udcff.md"
    assert surrogate != impostor
    assert encodable_name(surrogate) != encodable_name(impostor)
