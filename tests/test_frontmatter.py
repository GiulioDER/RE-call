from __future__ import annotations

from datetime import datetime, timezone

import pytest

from recall.frontmatter import (
    encodable_name,
    has_line_break,
    insert_frontmatter_line,
    legacy_pairing_differs,
    parse_frontmatter,
    supersedes_key,
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

@pytest.mark.parametrize(
    "name",
    ["old_2026-01-01.md", "legal/policy.md", "caffè.md", "a\\b.md", "policy\\update.md"],
)
def test_encodable_name_returns_a_name_that_encodes_unchanged(name):
    """Every name a real corpus holds, byte for byte, with no exceptions.

    The output is hashed into `reasoning_graph` node ids and therefore into the proposal ids a
    reviewer types into `recall rewrite apply`. A stand-in applied to a name that does not need
    one would renumber every queue that already exists, which reads as "the tool forgot".
    `caffè.md` is perfectly good UTF-8, and so is `policy\\update.md`: a backslash is legal in a
    POSIX filename, and an earlier version of this function diverted anything that LOOKED like
    an escape, which cost that file its id and its ability to be named by an edge.
    """
    assert encodable_name(name) == name


def test_encodable_name_is_encodable_for_a_name_that_is_not_valid_utf8():
    stand_in = encodable_name("bad\udcff.md")
    assert stand_in.encode("utf-8") == b"\x00name:bad\\udcff.md"


def test_encodable_name_does_not_collapse_two_names_onto_one():
    """INJECTIVE, or a corpus loses a file rather than a hash.

    The stand-in reads as the escape a reader already sees on stderr, and a file may
    legitimately be NAMED with that escape's own characters. Mapping `bad<surrogate>.md` onto
    the literal `bad\\udcff.md` would give two different documents one dict key in
    `corpus_documents` and one node id in the graph: the second silently replaces the first,
    which is the collision the corpus-relative key exists to prevent, reintroduced by the guard
    against the crash. The marker is what keeps the two ranges apart.
    """
    surrogate = "bad\udcff.md"
    impostor = "bad\\udcff.md"
    assert surrogate != impostor
    assert encodable_name(surrogate) != encodable_name(impostor)


def test_a_stand_in_is_still_a_stand_in_after_the_name_normaliser():
    """`supersedes_key` reduces a name to the stem of its LAST path segment.

    A marker prepended to the whole relative path is thrown away by that reduction, so a
    nested memo was `\\x00name:sub/bad\\udcff` as a reference and `\\x00name:bad\\udcff` as a
    file, the two could never meet, and `_resolve` refused to write about a file sitting right
    there. The marker travels with the segment that needs it, which is the only part of a path
    the normaliser keeps.
    """
    assert supersedes_key(encodable_name("sub/bad\udcff.md")) == supersedes_key(
        encodable_name("bad\udcff.md")
    )


def test_the_normaliser_does_not_collapse_a_nested_stand_in_onto_its_impostor():
    """Injectivity has to survive normalisation too, or `claim_key` re-merges what it separated.

    Both names reduce through `supersedes_key` before they are hashed into a claim, so a
    stripped marker meant a reviewer's rejection of one silently suppressed the other.
    """
    surrogate = encodable_name("sub/bad\udcff.md")
    impostor = encodable_name("sub/bad\\udcff.md")
    assert supersedes_key(surrogate) != supersedes_key(impostor)


def test_encodable_name_survives_a_name_wearing_the_marker():
    """The marker cannot occur in a real path, and the map is one to one anyway.

    `_hashable` in the extraction cache shipped without this, reasoned that its marker could
    not occur, and was right about filenames and wrong about file CONTENT, which collided two
    documents onto one cache key. The cost of not relying on the argument is one comparison.
    """
    from recall.frontmatter import NAME_STAND_IN_MARK

    worn = NAME_STAND_IN_MARK + "bad\\udcff.md"
    assert encodable_name(worn) != encodable_name("bad\udcff.md")


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
# --- a leading `---` is also markdown's thematic break -----------------------------------

RULE_THEN_PROSE = (
    "---\n"
    "\n"
    "# Release notes\n"
    "\n"
    "This release supersedes archive_policy_2026-01-05.md.\n"
    "\n"
    "---\n"
    "\n"
    "Contact ops.\n"
)


def test_leading_thematic_break_is_not_frontmatter():
    # Pairing the opening rule with the closing one deleted the whole first section from the
    # body, and returned an empty meta, so nothing was gained in exchange for the loss.
    assert parse_frontmatter(RULE_THEN_PROSE) == ({}, RULE_THEN_PROSE)


def test_rule_then_bullet_list_then_rule_keeps_body():
    text = "---\n\n- first point\n- second point\n\n---\n\nTail.\n"
    assert parse_frontmatter(text) == ({}, text)


def test_empty_block_is_not_frontmatter():
    # Two adjacent rules. A block with no key line declares nothing, so pairing it could only
    # ever remove body.
    text = "---\n---\n\nBody.\n"
    assert parse_frontmatter(text) == ({}, text)


def test_blank_only_block_is_not_frontmatter():
    text = "---\n\n---\n\nBody.\n"
    assert parse_frontmatter(text) == ({}, text)


def test_nested_block_is_still_frontmatter():
    # `recall.context` documents that an indented key belongs to a sub-object, so nested blocks
    # are a real shape here and must not be refused into the body.
    text = "---\nsupersedes: old.md\nnested:\n  key: value\n---\nBody.\n"
    meta, body = parse_frontmatter(text)
    assert meta == {"supersedes": "old.md"}
    assert body == "Body.\n"


def test_key_shaped_prose_line_is_still_paired():
    # Documented residual, asserted so the boundary is explicit rather than assumed away: a one
    # line prose block whose first word is followed by a colon cannot be told apart from YAML
    # without a real parser.
    assert parse_frontmatter("---\nNote: something\n---\nBody.\n") == ({}, "Body.\n")


def test_bare_url_line_is_still_paired():
    # Documented residual, same cause: `http` reads as a key.
    assert parse_frontmatter("---\nhttp://example.com\n---\nBody.\n") == ({}, "Body.\n")


# --- shapes of real YAML that must keep parsing -------------------------------------------
#
# Refusing to pair is not free. Every one of these declares validity metadata, and a rule that
# refused them would drop that metadata AND hand the raw block to the chunker as prose, which is
# worse than the thematic-break defect it is meant to fix. A block sequence and a comment are
# told apart from a markdown bullet list and a heading by ORDER: a sequence item belongs to the
# key that opened it, so it counts only once a key has been seen.


def test_a_column_zero_block_sequence_belongs_to_the_key_above_it():
    text = "---\ntags:\n- archive\n- policy\nvalid_until: 2020-01-01\n---\nbody\n"
    meta, body = parse_frontmatter(text)
    assert meta == {"valid_until": "2020-01-01"}
    assert body == "body\n"


def test_a_comment_after_a_key_does_not_refuse_the_block():
    text = "---\nvalid_until: 2020-01-01\n# provisional\n---\nbody\n"
    meta, body = parse_frontmatter(text)
    assert meta == {"valid_until": "2020-01-01"}
    assert body == "body\n"


def test_a_digit_leading_key_is_a_key():
    text = "---\n2026: plan\nvalid_until: 2020-01-01\n---\nbody\n"
    assert parse_frontmatter(text)[0] == {"valid_until": "2020-01-01"}


def test_a_non_ascii_key_is_a_key():
    text = "---\ntítulo: x\nvalid_until: 2020-01-01\n---\nbody\n"
    assert parse_frontmatter(text)[0] == {"valid_until": "2020-01-01"}


def test_a_quoted_key_is_a_key():
    text = "---\n\"my key\": x\nvalid_until: 2020-01-01\n---\nbody\n"
    assert parse_frontmatter(text)[0] == {"valid_until": "2020-01-01"}


def test_a_key_leading_with_a_dash_or_plus_is_a_key():
    # `-k:` and `+k:` load as mappings. A markdown bullet needs a SPACE after its marker, so the
    # two ARE distinguishable, and refusing these costs the whole block for nothing.
    for block in ("-k: x", "+k: x"):
        text = f"---\n{block}\nvalid_until: 2020-01-01\n---\nbody\n"
        assert parse_frontmatter(text)[0] == {"valid_until": "2020-01-01"}, block


def test_a_wholly_indented_block_is_still_frontmatter():
    # `parse_frontmatter` has always read an indented validity key as top level. Whether that is
    # right is a separate question; refusing the whole block over it is not.
    text = "---\n  valid_until: 2020-01-01\n---\nbody\n"
    meta, body = parse_frontmatter(text)
    assert meta == {"valid_until": "2020-01-01"}
    assert body == "body\n"


def test_a_bullet_list_before_any_key_is_still_prose():
    # The counterpart of the block-sequence test: with no key to belong to, a column 0 `-` is a
    # markdown bullet and the opening rule is a thematic break.
    text = "---\n\n- first point\n- second point\n\n---\n\nTail.\n"
    assert parse_frontmatter(text) == ({}, text)


def test_a_heading_before_any_key_is_still_prose():
    text = "---\n\n# Release notes\n\nProse.\n\n---\n\nTail.\n"
    assert parse_frontmatter(text) == ({}, text)


# These two pin the ORDER rule specifically. The pair above them is refused for having no key at
# all, so they pass whether or not order is enforced; mutation testing caught that and these are
# the cases that actually bite. In each, a later line supplies the key, so without the ordering
# rule the bullets or the heading above it would be accepted and the section deleted.


def test_a_flow_collection_closed_at_column_zero_is_still_frontmatter():
    text = "---\ntags: [\n  archive,\n  policy,\n]\nvalid_until: 2020-01-01\n---\nbody\n"
    assert parse_frontmatter(text)[0] == {"valid_until": "2020-01-01"}


def test_explicit_key_syntax_is_still_frontmatter():
    text = "---\n? complex key\n: value\nvalid_until: 2020-01-01\n---\nbody\n"
    assert parse_frontmatter(text)[0] == {"valid_until": "2020-01-01"}


def test_an_unquoted_key_containing_a_space_is_a_key():
    # `date created:` and `project name:` are ordinary Obsidian and Jekyll frontmatter. Refusing
    # them costs the whole block, which is strictly worse than what the old rule did.
    text = "---\ndate created: 2026-01-01\nvalid_until: 2020-01-01\n---\nbody\n"
    assert parse_frontmatter(text)[0] == {"valid_until": "2020-01-01"}


# A key shaped line unlocks the block: after one, comments and sequence items are accepted. That
# makes the LEAD-IN of a prose section load bearing, so markdown's own lead-ins must not read as
# keys. None of these is a plausible unquoted YAML key.


def test_a_bold_lead_in_does_not_unlock_a_prose_section():
    text = (
        "---\n\n**Warning**: the rotation steps changed in June.\n\n# New procedure\n\n"
        "- Rotate the key.\n- Restart the collector.\n\n---\n\nContact ops.\n"
    )
    assert parse_frontmatter(text) == ({}, text)


def test_a_link_reference_definition_does_not_unlock_a_prose_section():
    text = "---\n\n[spec]: https://example.com\n\n# Procedure\n\n- Step one.\n\n---\n\nTail.\n"
    assert parse_frontmatter(text) == ({}, text)


def test_an_inline_code_lead_in_does_not_unlock_a_prose_section():
    text = "---\n\n`config`: the new shape\n\n# Procedure\n\n- Step one.\n\n---\n\nTail.\n"
    assert parse_frontmatter(text) == ({}, text)


def test_a_blockquote_lead_in_does_not_unlock_a_prose_section():
    text = "---\n\n> quoted: an aside\n\n# Procedure\n\n- Step one.\n\n---\n\nTail.\n"
    assert parse_frontmatter(text) == ({}, text)


def test_a_key_shaped_lead_in_still_unlocks_a_whole_prose_section():
    """The honest limit of this fix, asserted so it cannot quietly be claimed away.

    One key shaped line unlocks the block: every comment and sequence item after it is accepted.
    A section led by ``Note:`` and followed by a heading and a bullet list is therefore still
    paired, and still deleted. That is exactly what the old rule did, so it is not a regression
    and `legacy_pairing_differs` is correctly False. It is simply not fixed, and telling a
    markdown bullet from a YAML sequence item at that position needs a real YAML parser.
    """
    text = (
        "---\n\nNote: the rotation steps changed.\n\n# New procedure\n\n"
        "- Rotate the key.\n\n---\n\nContact ops.\n"
    )
    assert parse_frontmatter(text) == ({}, "Contact ops.\n")
    assert legacy_pairing_differs(text) is False


def test_a_sentence_containing_a_colon_is_a_key_and_unlocks_the_block():
    """The price of allowing spaces inside a key, pinned rather than left to be discovered.

    Spaces are what buy ``date created:``, and they also make any sentence with a colon in it a
    key. Equal to the old behaviour, so not a regression and correctly not flagged, but it means
    the fix protects a section led by a colon-free sentence, not one led by any sentence.
    """
    text = (
        "---\n\nIn June we changed this: rotate quarterly.\n\n# Procedure\n\n"
        "- step\n\n---\n\nTail.\n"
    )
    assert parse_frontmatter(text) == ({}, "Tail.\n")
    assert legacy_pairing_differs(text) is False


def test_a_sentence_without_a_colon_is_protected():
    # The counterpart, and the shape the reported defect actually had.
    text = "---\n\nIn June we changed the rotation.\n\n# Procedure\n\n- step\n\n---\n\nTail.\n"
    assert parse_frontmatter(text) == ({}, text)
    assert legacy_pairing_differs(text) is True


def test_a_table_row_is_protected():
    # The cells carry a colon deliberately. Without one, this document is protected by the
    # no-colon rule and the assertion says nothing about the `|` exclusion it is named for.
    # The alignment row is what a real table with alignment always has.
    text = "---\n\n| key: value | b |\n| :-- | --: |\n\n---\n\nTail.\n"
    assert parse_frontmatter(text) == ({}, text)


def test_a_heading_containing_a_colon_is_protected():
    # A memo heading like `# Release notes: June` is common, and it is the only thing standing
    # between the `#` exclusion and a deleted section, since the colon makes the rest of the
    # line key shaped.
    text = "---\n\n# Release notes: June\n\n- one\n- two\n\n---\n\nTail.\n"
    assert parse_frontmatter(text) == ({}, text)


def test_a_bullet_list_is_prose_even_when_a_later_line_looks_like_a_key():
    text = "---\n\n- ship the migration\n- tell the team\n\nDeadline: Friday\n\n---\n\nTail.\n"
    assert parse_frontmatter(text) == ({}, text)


def test_a_heading_is_prose_even_when_a_later_line_looks_like_a_key():
    text = "---\n\n# Release notes\n\nDeadline: Friday\n\n---\n\nTail.\n"
    assert parse_frontmatter(text) == ({}, text)


# --- which files an existing index has to rebuild ------------------------------------------
#
# `recall.index` fingerprints a file on its raw bytes, so a corpus whose files have not changed
# is skipped and would go on serving bodies with a section missing. Both directions matter: the
# True cases are files that MUST be re-indexed, and every False case is a file that must NOT be,
# because a spurious True re-embeds a corpus for nothing.


def test_legacy_pairing_differs_on_a_document_whose_body_moved():
    assert legacy_pairing_differs(RULE_THEN_PROSE) is True


def test_legacy_pairing_differs_on_an_empty_block():
    assert legacy_pairing_differs("---\n---\n\nBody.\n") is True


def test_legacy_pairing_is_unchanged_for_real_frontmatter():
    assert legacy_pairing_differs(DOC) is False


def test_legacy_pairing_is_unchanged_without_an_opening_fence():
    assert legacy_pairing_differs("# Just a doc\n\nNo block here.") is False


def test_legacy_pairing_is_unchanged_for_an_unclosed_block():
    # The old rule did not pair this either, so nothing moved and it must not be re-indexed.
    assert legacy_pairing_differs("---\nvalid_until: 2026-01-01\nno closing fence") is False


def test_legacy_pairing_is_unchanged_for_a_lone_opening_rule():
    # A rule with no second rule anywhere: unpaired before, unpaired now.
    assert legacy_pairing_differs("---\n\n# Heading\n\nProse with no second rule.\n") is False


def test_legacy_pairing_differs_on_an_exotic_line_separator():
    # `document_title` split with `splitlines()`, which breaks on U+2028, a form feed, a vertical
    # tab and NEL; the span is counted over `split("\n")`. The two scans can therefore address
    # different lines, moving the title while the body stays byte-identical. Flagged on the
    # separator alone, independent of which key moved.
    assert legacy_pairing_differs("---\ntitle: A\u2028title: B\n---\nbody") is True
    assert legacy_pairing_differs("---\nz: 1\x0ctitle: C") is True


def test_legacy_pairing_differs_when_the_separator_is_on_the_first_line():
    """The separator guard must not be skippable by the separator it exists for.

    Its fence precondition splits on ``\\n``. When the exotic break falls on the FIRST physical
    line, that precondition sees no fence and returns before the separator test runs, while the
    old title scan (which split with `splitlines`) saw a fence and read the block.
    """
    assert legacy_pairing_differs("---\x0ctitle: X\n\nbody\n") is True
    assert legacy_pairing_differs("---\u2028title: X\u2028---\u2028body") is True


def test_legacy_pairing_is_unchanged_for_a_residual_shape():
    # Still paired under both rules, so its body did not move.
    assert legacy_pairing_differs("---\nNote: something\n---\nBody.\n") is False
