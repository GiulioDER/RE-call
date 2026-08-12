"""The report is the whole output of an audit run. Its counts and its quotes are what a reader
acts on.

`render` is told what actually ran (`documents`, `revisions`, `questions`) separately from what
was found (`drifts`, `stale`). Every test here that renders a non-empty finding also asserts the
coverage line, because a report that gets the finding right and the coverage wrong still lies
about what it did.
"""
from __future__ import annotations

from recall_consistency.findings import ClaimDrift, StaleAnswer
from recall_consistency.report import render

DRIFT = ClaimDrift(
    path="docs/notes.md",
    old_sha="aaa1111",
    new_sha="bbb2222",
    old_date="2026-01-01",
    new_date="2026-02-01",
    old_line="The SPLADE leg buys +0.0303 over the lexical leg.",
    new_line="The SPLADE leg buys +0.0512 over the lexical leg.",
)

STALE = StaleAnswer(
    question="what does the SPLADE leg buy?",
    plain_top_file="docs_notes__2026-01-01__aaa1111.md",
    plain_top_superseded_by="docs_notes__2026-02-01__bbb2222.md",
    trusted_verdict="ok",
    trusted_abstained=False,
)

ABSTENTION = StaleAnswer(
    question="what is the rate limit?",
    plain_top_file="docs_limits__2026-01-01__ccc3333.md",
    plain_top_superseded_by="",
    trusted_verdict="low_confidence",
    trusted_abstained=True,
)


def test_the_report_names_the_corpus_and_quotes_both_versions_of_a_drift() -> None:
    out = render(
        "acme-memory", [DRIFT], [], documents=1, revisions=2, questions=0
    )

    assert "acme-memory" in out
    assert "1 restated claim" in out
    assert "docs/notes.md" in out
    assert "+0.0303" in out
    assert "+0.0512" in out
    assert "2026-01-01" in out


def test_the_report_names_the_questions_answered_from_replaced_text() -> None:
    out = render(
        "acme-memory", [], [STALE], documents=1, revisions=2, questions=1
    )

    assert "1 question" in out
    assert "what does the SPLADE leg buy?" in out
    assert "docs_notes__2026-01-01__aaa1111.md" in out
    assert "docs_notes__2026-02-01__bbb2222.md" in out


def test_a_clean_corpus_says_so_instead_of_rendering_empty_sections() -> None:
    out = render("acme-memory", [], [], documents=3, revisions=9, questions=5)

    assert "No restated claims found in 9 revisions of 3 documents." in out
    assert "None of the 5 questions supplied resolved to replaced text." in out


def test_counts_are_pluralised() -> None:
    out = render(
        "acme-memory", [DRIFT, DRIFT], [STALE, STALE], documents=1, revisions=2, questions=2
    )

    assert "2 restated claims" in out
    assert "2 questions" in out


def test_the_coverage_line_names_documents_and_revisions_and_pluralises_at_one() -> None:
    plural = render("acme-memory", [], [], documents=36, revisions=178, questions=0)
    assert "Read 178 revisions of 36 tracked markdown documents." in plural

    singular = render("acme-memory", [], [], documents=1, revisions=1, questions=0)
    assert "Read 1 revision of 1 tracked markdown document." in singular


def test_zero_documents_is_an_empty_result_not_a_clean_one() -> None:
    """This is the report a caller whose docs are `.rst`, gitignored, or outside the glob
    sees. It must not read like a clean bill of health from a scan that never happened.
    """
    out = render("acme-memory", [], [], documents=0, revisions=0, questions=0)

    assert "No tracked markdown documents matched, so nothing was audited." in out
    assert "check the glob" in out
    # None of the sections that imply something was scanned, indexed, or asked may appear.
    assert "## Claims the corpus restated" not in out
    assert "## Questions answered from replaced text" not in out
    assert "## Method" not in out
    assert "No restated claims" not in out
    assert "question" not in out.lower()


def test_zero_documents_still_carries_the_coverage_line() -> None:
    out = render("acme-memory", [], [], documents=0, revisions=0, questions=0)
    assert "Read 0 revisions of 0 tracked markdown documents." in out


def test_zero_documents_with_a_completed_but_clean_probe_is_not_reported_as_nothing_audited() -> (
    None
):
    """`documents == 0` and `stale == []` are not the same fact as `questions == 0`. A probe
    that ran against 12 real questions and legitimately found none stale still ran, and must
    say so rather than being folded into the "nothing was audited" empty-scan wording.
    """
    out = render("acme-memory", [], [], documents=0, revisions=0, questions=12)

    assert "nothing was audited" not in out
    assert "None of the 12 questions supplied resolved to replaced text." in out


def test_zero_documents_does_not_discard_a_completed_probes_real_findings() -> None:
    """`stale` comes from an independent probe against an already-indexed corpus, not from this
    run's glob match. A run whose glob missed every markdown file but which still supplied and
    ran real questions must not report "nothing was audited": that would silently drop findings
    a probe actually produced, which is itself the under-report this tool exists to refuse.
    """
    out = render("acme-memory", [], [STALE], documents=0, revisions=0, questions=1)

    assert STALE.question in out
    assert f"Replaced by: `{STALE.plain_top_superseded_by}`" in out
    assert "## Questions answered from replaced text" in out
    # The wholesale "nothing was audited" claim must not appear: something WAS audited.
    assert "nothing was audited" not in out


def test_no_questions_supplied_says_that_half_did_not_run() -> None:
    out = render("acme-memory", [], [], documents=2, revisions=4, questions=0)

    assert (
        "No questions were supplied, so this half of the audit did not run." in out
    )
    assert "ten to twenty questions" in out
    # Must not claim a negative result for a probe that never ran.
    assert "No questions were answered from replaced text" not in out
    assert "None of the 0 questions" not in out


def test_a_superseded_stale_answer_and_an_abstention_are_two_separate_groups() -> None:
    out = render(
        "acme-memory",
        [],
        [STALE, ABSTENTION],
        documents=1,
        revisions=2,
        questions=2,
    )

    # The superseded group keeps its "answered from replaced text" framing and count.
    assert "1 question" in out
    assert "text the corpus has already replaced" in out
    assert STALE.question in out
    assert f"Replaced by: `{STALE.plain_top_superseded_by}`" in out

    # The abstention group is never described as "answered from replaced text": its nearest
    # match was current, and the false claim is exactly what Important 7 flagged.
    assert "abstained even though the nearest match was current" in out
    assert ABSTENTION.question in out
    assert "RE-call abstained rather than answering from it." in out
    assert f"Replaced by: `{ABSTENTION.plain_top_superseded_by}`" not in out


def test_the_stale_baseline_sentence_describes_a_fused_pool_not_a_dense_top_k() -> None:
    out = render(
        "acme-memory", [], [STALE], documents=1, revisions=2, questions=1
    )

    assert "highest-cosine hit among the results this retrieval returned" in out
    assert "hybrid and may be reranked" in out
    # The old, inaccurate claim must be gone.
    assert "plain top-k retriever returns" not in out


def test_the_method_section_names_all_six_claim_matching_skips() -> None:
    out = render("acme-memory", [DRIFT], [], documents=1, revisions=2, questions=0)

    assert "## Method" in out
    assert "a claim that was rewritten rather than restated" in out
    assert "a sentence shape that occurs twice in one document" in out
    assert "more than one number moved" in out
    assert "an ordered list item renumbered by an insertion above it" in out
    assert "a line whose numeric count differs from its counterpart" in out
    assert "too little text around its numbers to identify a claim" in out
    # The old undercount must be gone.
    assert "Three things are deliberately skipped" not in out


def test_the_method_section_is_absent_when_nothing_was_read() -> None:
    out = render("acme-memory", [], [], documents=0, revisions=0, questions=0)
    assert "## Method" not in out
