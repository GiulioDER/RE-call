"""`INSTRUMENT_STATUS.md` must not silently fall behind `FINDINGS.md`.

`results/INSTRUMENT_STATUS.md` is a hand-maintained inventory of which abstention claims in
`results/FINDINGS.md` are still checkable. `FINDINGS.md` is grown continuously by other sessions,
and the inventory has gone stale three times in one day: once when a merged artifact falsified a
row, twice when a new section appeared the inventory did not know existed. This file is what
turns the second failure mode into a loud, named test failure instead of a silent gap.

Two kinds of test live here. The unit tests exercise `recall.eval.claims`'s pieces against small
synthetic documents, so a change to the parsing/classification MECHANISM is caught precisely and
fast. The one integration test at the bottom runs the real, mechanically-derived claim list
against the real, hand-maintained inventory -- that is the actual guard this cycle exists to add.
"""
from __future__ import annotations

import pytest

from recall.eval import claims
from recall.eval.claims import (
    Section,
    abstention_claiming_sections,
    load_missing_sections,
    missing_sections,
    parse_covered_ids,
    parse_findings_sections,
    section_mentions_abstention,
)

# ---------------------------------------------------------------------------------------------
# parse_findings_sections
# ---------------------------------------------------------------------------------------------


def test_parses_top_level_and_lettered_headings() -> None:
    text = (
        "preamble text, not a section\n\n"
        "## 1. First\n\nbody one\n\n"
        "### 1b. First sub\n\nbody one-b\n\n"
        "## 2. Second\n\nbody two\n"
    )
    ids = [s.id for s in parse_findings_sections(text)]
    assert ids == ["1", "1b", "2"]


def test_section_body_excludes_the_heading_line() -> None:
    text = "## 1. A Title mentioning abstention\n\nbody without the keyword\n"
    [section] = parse_findings_sections(text)
    assert "Title" not in section.body
    assert section.title == "A Title mentioning abstention"


def test_unnumbered_subheading_does_not_split_a_section() -> None:
    """`#### Why quoting one depth was a mistake` (FINDINGS.md L623) sits inside `### 9a`."""
    text = (
        "## 9. Parent\n\n"
        "### 9a. Child\n\nfirst half\n\n"
        "#### An unnumbered sub-heading\n\nsecond half\n\n"
        "### 9b. Next child\n\nother body\n"
    )
    sections = {s.id: s for s in parse_findings_sections(text)}
    assert sections["9a"].body.count("first half") == 1
    assert sections["9a"].body.count("second half") == 1
    assert "other body" not in sections["9a"].body


def test_unnumbered_top_level_heading_is_never_a_section() -> None:
    """e.g. "## What this document establishes" -- no digit after "## ", so no id to assign."""
    text = "## What this document establishes\n\nsome abstention text\n\n## 1. Real\n\nbody\n"
    ids = [s.id for s in parse_findings_sections(text)]
    assert ids == ["1"]


def test_five_letter_and_bare_ids_both_parse_at_either_heading_depth() -> None:
    """"5b" is a "##" heading in the real document, not "###" -- depth must not gate the id."""
    text = "## 5. Bare\n\nx\n\n## 5b. Lettered at the SAME depth\n\ny\n"
    ids = [s.id for s in parse_findings_sections(text)]
    assert ids == ["5", "5b"]


# ---------------------------------------------------------------------------------------------
# section_mentions_abstention / ABSTENTION_KEYWORDS
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("keyword", claims.ABSTENTION_KEYWORDS)
def test_each_declared_keyword_is_individually_detected(keyword: str) -> None:
    section = Section(id="1", title="t", body=f"some prose containing {keyword} in the middle")
    assert section_mentions_abstention(section)


def test_no_match_when_no_keyword_present() -> None:
    section = Section(id="1", title="t", body="plain retrieval prose about hit@5 and MRR")
    assert not section_mentions_abstention(section)


def test_matching_is_case_insensitive() -> None:
    section = Section(id="1", title="t", body="ABSTENTION accuracy was measured")
    assert section_mentions_abstention(section)


def test_abstain_and_abstention_are_independent_keywords() -> None:
    """"abstention" (noun) is not a substring of "abstain" (verb) or vice versa.

    Regression guard: an earlier draft of this classifier assumed "abstain" would also catch
    "abstention" and nearly shipped with only one of the two keywords. A body using only the noun
    form must still match.
    """
    assert "abstain" not in "abstention"
    assert "abstention" not in "abstain"
    noun_only = Section(id="1", title="t", body="Abstention (category 5) is measured here")
    verb_only = Section(id="2", title="t", body="the system abstains on every query")
    assert section_mentions_abstention(noun_only)
    assert section_mentions_abstention(verb_only)


def test_title_alone_does_not_count_as_a_match() -> None:
    """Classification reads body text "not just its title" (task spec) -- title-only must miss."""
    section = Section(id="1", title="An abstention-heavy title", body="unrelated retrieval prose")
    assert not section_mentions_abstention(section)


# ---------------------------------------------------------------------------------------------
# abstention_claiming_sections / EXCLUDED_SECTIONS
# ---------------------------------------------------------------------------------------------


def test_excluded_section_is_dropped_even_though_it_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claims, "EXCLUDED_SECTIONS", {"1": "passing mention, see section 2"})
    sections = [
        Section(id="1", title="t", body="mentions abstention only in passing"),
        Section(id="2", title="t", body="abstention accuracy 1.00"),
    ]
    assert abstention_claiming_sections(sections) == ["2"]


def test_excluded_sections_naming_an_absent_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale exclusion (its heading renamed/removed/renumbered) must fail loud, not silently."""
    monkeypatch.setattr(claims, "EXCLUDED_SECTIONS", {"99z": "no longer exists"})
    sections = [Section(id="1", title="t", body="abstention accuracy 1.00")]
    with pytest.raises(ValueError, match="99z"):
        abstention_claiming_sections(sections)


def test_real_excluded_sections_all_name_ids_present_in_the_real_document() -> None:
    """Same failure mode as above, pinned against the live FINDINGS.md rather than a fixture."""
    findings_text = claims.FINDINGS_PATH.read_text(encoding="utf-8")
    sections = parse_findings_sections(findings_text)
    abstention_claiming_sections(sections)  # raises ValueError if any exclusion id is stale


def test_claiming_sections_are_sorted_numerically_then_alphabetically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claims, "EXCLUDED_SECTIONS", {})  # isolate from the real document's list
    sections = [
        Section(id="10", title="t", body="abstention"),
        Section(id="2", title="t", body="abstention"),
        Section(id="9b", title="t", body="abstention"),
        Section(id="9a", title="t", body="no keyword here"),
    ]
    assert abstention_claiming_sections(sections) == ["2", "9b", "10"]


# ---------------------------------------------------------------------------------------------
# parse_covered_ids
# ---------------------------------------------------------------------------------------------

_STATUS_HEADER = "| claim | status | artifact | notes |\n|---|---|---|---|\n"


def test_reads_bare_id_from_the_claim_column() -> None:
    text = _STATUS_HEADER + "| §2 fixed gap threshold | current | x.json | n/a |\n"
    assert parse_covered_ids(text, known_ids=["2"]) == {"2"}


def test_ignores_ids_mentioned_only_in_the_notes_column() -> None:
    """A cross-reference like "also quoted at FINDINGS.md section 10c" is not a coverage claim."""
    text = _STATUS_HEADER + "| §2 fixed gap threshold | current | x.json | also see §10c |\n"
    assert parse_covered_ids(text, known_ids=["2", "10c"]) == {"2"}


def test_ignores_header_and_separator_rows() -> None:
    text = _STATUS_HEADER
    assert parse_covered_ids(text, known_ids=["2"]) == set()


def test_all_rows_marker_expands_to_lettered_subsections_present_in_findings() -> None:
    text = _STATUS_HEADER + "| §10 LongMemEval, all rows | unfalsifiable | - | discarded |\n"
    known = ["10", "10b", "10c", "10d", "11"]
    assert parse_covered_ids(text, known_ids=known) == {"10", "10b", "10c", "10d"}


def test_all_rows_marker_does_not_leak_into_an_unrelated_numeric_prefix() -> None:
    """Base id "1" + "all rows" must not swallow "10" or "11" (they start with "1" too)."""
    text = _STATUS_HEADER + "| §1 all rows | current | - | n/a |\n"
    assert parse_covered_ids(text, known_ids=["1", "10", "11"]) == {"1"}


def test_without_the_all_rows_marker_only_the_literal_id_is_covered() -> None:
    text = _STATUS_HEADER + "| §10 LongMemEval retrieval only | current | x.json | n/a |\n"
    assert parse_covered_ids(text, known_ids=["10", "10b"]) == {"10"}


def test_row_with_no_section_id_is_skipped_without_error() -> None:
    """Mirrors the real "every row above" meta-row, which asserts nothing about a section id."""
    text = _STATUS_HEADER + "| every row above | no row count on any artifact | - | n/a |\n"
    assert parse_covered_ids(text, known_ids=["2"]) == set()


def test_a_second_table_elsewhere_in_the_document_is_not_scanned() -> None:
    """Only the block whose header cell reads "claim" counts -- not every "|"-led line."""
    text = (
        "| topic | owner |\n|---|---|\n| §2 unrelated table | someone |\n\n"
        + _STATUS_HEADER
        + "| §5 real claim | current | - | n/a |\n"
    )
    assert parse_covered_ids(text, known_ids=["2", "5"]) == {"5"}


# ---------------------------------------------------------------------------------------------
# missing_sections — the end-to-end diff, on synthetic documents
# ---------------------------------------------------------------------------------------------


def test_missing_sections_empty_when_every_claim_is_covered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claims, "EXCLUDED_SECTIONS", {})  # isolate from the real document's list
    findings = "## 1. Claim\n\nabstention accuracy 1.00\n\n## 2. No claim\n\nplain prose\n"
    status = _STATUS_HEADER + "| §1 claim | current | - | n/a |\n"
    assert missing_sections(findings, status) == []


def test_missing_sections_names_the_uncovered_claiming_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claims, "EXCLUDED_SECTIONS", {})  # isolate from the real document's list
    findings = "## 1. Covered\n\nabstention accuracy 1.00\n\n## 2. Not covered\n\nFCR 0.00\n"
    status = _STATUS_HEADER + "| §1 covered | current | - | n/a |\n"
    assert missing_sections(findings, status) == ["2"]


def test_guard_is_not_vacuous_removing_a_covered_row_reintroduces_its_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct proof that the check can fail: dropping a real row surfaces its section again.

    Synthetic counterpart to the live-document proof in the verification report (an editor
    temporarily removes a real INSTRUMENT_STATUS.md row and reruns the test) -- this version is
    permanent and fast, so a future change to the diff logic itself cannot quietly go vacuous.
    """
    monkeypatch.setattr(claims, "EXCLUDED_SECTIONS", {})  # isolate from the real document's list
    findings = "## 1. Claim\n\nabstention accuracy 1.00\n"
    covered_status = _STATUS_HEADER + "| §1 claim | current | - | n/a |\n"
    empty_status = _STATUS_HEADER  # the "§1" row deleted, exactly as an editor would delete it

    assert missing_sections(findings, covered_status) == []
    assert missing_sections(findings, empty_status) == ["1"]


# ---------------------------------------------------------------------------------------------
# The real guard: FINDINGS.md vs INSTRUMENT_STATUS.md
# ---------------------------------------------------------------------------------------------


def test_every_abstention_claiming_finding_is_in_the_instrument_status_inventory() -> None:
    """The guard `recall/eval/claims.py` exists for.

    Failing here means `results/FINDINGS.md` gained a section that keyword-matches
    `ABSTENTION_KEYWORDS` (and is not in `EXCLUDED_SECTIONS`) with no corresponding row in
    `results/INSTRUMENT_STATUS.md`'s table. Fix: read the named section(s), decide an honest
    status, add the row(s) -- or, if a mention truly is in passing, add a commented entry to
    `claims.EXCLUDED_SECTIONS` explaining why it is not a claim.
    """
    missing = load_missing_sections()
    assert missing == [], (
        "results/INSTRUMENT_STATUS.md has no row for FINDINGS.md section(s): "
        + ", ".join(f"\xa7{section_id}" for section_id in missing)
    )
