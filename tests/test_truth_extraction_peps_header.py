"""Parsing PEP RFC822 headers and detecting prose restatements of a header edge."""
from __future__ import annotations

from benchmarks.labelling.truth_extraction.peps_header import (
    Edge,
    edges_from_fields,
    header_fields,
    pep_refs,
    restates,
    sentences,
    split_header,
)

HEAD = """PEP: 216
Title: Docstring Format
Status: Superseded
Superseded-By: 287

Abstract
========
Body text here.
"""


def test_split_header_cuts_at_first_blank_line():
    head, body = split_header(HEAD)
    assert "Superseded-By: 287" in head
    assert "Abstract" in body
    assert "Superseded-By" not in body


def test_split_header_without_blank_line_yields_empty_body():
    head, body = split_header("PEP: 1\nTitle: X")
    assert head == "PEP: 1\nTitle: X"
    assert body == ""


def test_header_fields_parses_rfc822_continuations():
    fields = header_fields("Replaces: 245,\n  246\nTitle: T")
    assert fields["Replaces"] == "245, 246"
    assert fields["Title"] == "T"


def test_pep_refs_accepts_all_three_citation_forms():
    assert pep_refs("see :pep:`287` and PEP 292 and pep-0435") == {
        "pep-0287", "pep-0292", "pep-0435",
    }


def test_pep_refs_zero_pads_so_pep_5_and_pep_0005_are_one_document():
    assert pep_refs("PEP 5") == pep_refs("PEP 0005") == {"pep-0005"}


def test_edges_from_superseded_by_points_away_from_this_pep():
    assert edges_from_fields("pep-0216", {"Superseded-By": "287"}) == {
        Edge(superseded="pep-0216", successor="pep-0287")
    }


def test_edges_from_replaces_points_toward_this_pep():
    # `Replaces:` is active voice — the edge's SUCCESSOR is the document declaring it.
    # Inverting this would demote the live PEP beneath the one it replaced.
    assert edges_from_fields("pep-0440", {"Replaces": "386"}) == {
        Edge(superseded="pep-0386", successor="pep-0440")
    }


def test_edges_from_multivalued_replaces_yields_one_edge_each():
    assert edges_from_fields("pep-3124", {"Replaces": "245, 246"}) == {
        Edge(superseded="pep-0245", successor="pep-3124"),
        Edge(superseded="pep-0246", successor="pep-3124"),
    }


def test_sentences_joins_hard_wrapped_lines_before_splitting():
    # RST hard-wraps prose. Splitting on newlines would cut this restatement in half.
    got = sentences("It has been\nsuperseded by :pep:`287`. Next one.")
    assert got[0].strip() == "It has been superseded by :pep:`287`."


def test_sentences_never_glues_across_a_blank_line():
    # A paragraph with no terminal punctuation must not run into the next section. Gluing here
    # is whole-body co-occurrence arriving through the back door.
    got = sentences("Heading\n\nThis is deprecated\n\nSee :pep:`287` for formatting.")
    assert not any("deprecated" in s and "287" in s for s in got)


def test_sentences_keeps_a_trailing_fragment_with_no_terminator():
    # pep-0634's real restatement ends in a colon. Dropping it loses a true positive.
    got = sentences("It replaces :pep:`622`, which is hereby split in three parts:")
    assert any(":pep:`622`" in s for s in got)


def test_restates_finds_an_unterminated_restatement():
    body = "It replaces :pep:`622`, which is hereby split in three parts:"
    assert restates(body, "pep-0622") is not None


def test_restates_returns_the_evidence_sentence():
    assert restates("It has been superseded by :pep:`287`.", "pep-0287") == (
        "It has been superseded by :pep:`287`."
    )


def test_restates_requires_marker_and_partner_in_the_SAME_sentence():
    # Marker in one sentence, reference in another: whole-body co-occurrence, which is the
    # loose matching fix.py records as producing garbage. Must not count.
    body = "This is deprecated. Unrelatedly, see :pep:`287` for formatting."
    assert restates(body, "pep-0287") is None


def test_restates_returns_none_when_partner_absent():
    assert restates("It has been superseded by :pep:`999`.", "pep-0287") is None
