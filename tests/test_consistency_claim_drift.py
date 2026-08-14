"""A restated claim with a changed number is the contradiction an audit is looking for."""
from __future__ import annotations

from recall_consistency.claim_drift import claim_skeleton, drifts
from recall_consistency.history_corpus import Revision


def _rev(sha: str, date: str, body: str) -> Revision:
    return Revision(path="docs/notes.md", sha=sha, date=date, body=body)


def test_a_restated_claim_with_a_changed_number_is_a_drift() -> None:
    revs = [
        _rev("aaa1111", "2026-01-01", "The SPLADE leg buys +0.0303 over the lexical leg.\n"),
        _rev("bbb2222", "2026-02-01", "The SPLADE leg buys +0.0512 over the lexical leg.\n"),
    ]

    found = drifts(revs)

    assert len(found) == 1
    assert found[0].old_line == "The SPLADE leg buys +0.0303 over the lexical leg."
    assert found[0].new_line == "The SPLADE leg buys +0.0512 over the lexical leg."
    assert (found[0].old_sha, found[0].new_sha) == ("aaa1111", "bbb2222")
    assert (found[0].old_date, found[0].new_date) == ("2026-01-01", "2026-02-01")
    assert found[0].path == "docs/notes.md"


def test_an_unchanged_claim_beside_edited_prose_is_not_a_drift() -> None:
    revs = [
        _rev("aaa1111", "2026-01-01", "The SPLADE leg buys +0.0303 over the lexical leg.\nprose\n"),
        _rev("bbb2222", "2026-02-01", "The SPLADE leg buys +0.0303 over the lexical leg.\nmore\n"),
    ]

    assert drifts(revs) == []


def test_bare_table_rows_carry_too_little_context_to_match_on() -> None:
    """`| 1 | 2 |` against `| 1 | 3 |` is a coincidence, not a restated claim."""
    revs = [
        _rev("aaa1111", "2026-01-01", "| 1 | 2 |\n"),
        _rev("bbb2222", "2026-02-01", "| 1 | 3 |\n"),
    ]

    assert drifts(revs) == []


def test_a_single_revision_has_nothing_to_compare_against() -> None:
    assert drifts([_rev("aaa1111", "2026-01-01", "recall@5 is 0.92\n")]) == []


def test_reordering_a_document_does_not_invent_a_restated_claim() -> None:
    """Nothing changed here but the order of two lines. A reported drift would be fabricated.

    Both lines reduce to `recall@# is # on known-item queries`, so one skeleton has two
    candidates. Keeping the first would pair revision A's line 1 with revision B's line 1, which
    are different claims, and quote them as one claim restated. Revert `_claim_lines` to
    `setdefault` with no ambiguity set and this test goes red with exactly that fabrication.
    """
    first = "recall@5 is 0.92 on known-item queries"
    second = "recall@10 is 0.88 on known-item queries"
    revs = [
        _rev("aaa1111", "2026-01-01", f"{first}\n{second}\n"),
        _rev("bbb2222", "2026-02-01", f"{second}\n{first}\n"),
    ]

    assert drifts(revs) == []


def test_a_repeated_identical_line_is_not_ambiguous() -> None:
    """One sentence written twice has one answer, so refusing it buys nothing.

    Drop the `seen[skeleton] != line` condition and both revisions lose the claim entirely, so
    this test goes red.
    """
    revs = [
        _rev("aaa1111", "2026-01-01", "recall@5 is 0.92 on known-item queries\n" * 2),
        _rev("bbb2222", "2026-02-01", "recall@5 is 0.945 on known-item queries\n" * 2),
    ]

    found = drifts(revs)

    assert len(found) == 1
    assert found[0].new_line == "recall@5 is 0.945 on known-item queries"


def test_a_range_keeps_its_separator_in_the_skeleton() -> None:
    """A range is one claim, not two adjacent numbers, and the skeleton has to show that.

    This asserts on `claim_skeleton` directly because the matching rule is only observable
    through `drifts` once two skeletons collide, and the correct behaviour here is that they do
    not. Revert either character of the `NUMBER` lookbehind and one of these goes red.
    """
    assert claim_skeleton("the window is 2020-2024 inclusive") == "the window is #-# inclusive"
    assert claim_skeleton("deflection was 45%-50% overall") == "deflection was #-# overall"
    assert claim_skeleton("latency dropped 45ms-50ms today") == "latency dropped #ms-#ms today"
    assert claim_skeleton("the window is 2020 2024 inclusive") == "the window is # # inclusive"
    assert claim_skeleton("the lever is +0.0512 over lexical") == "the lever is # over lexical"


def test_a_digit_run_after_a_letter_is_still_a_number() -> None:
    """`p95`, `v2` and `batch_32` are ordinary benchmark prose and must stay readable.

    Move the lookbehind off the sign and onto the whole match, as in
    `(?<![\\w.%])[-+]?\\d+...`, and every assertion here goes red: the digits stop matching at
    all, both revisions skip the line, and a real restatement is never flagged.
    """
    assert claim_skeleton("p95 latency was 45ms") == "p# latency was #ms"
    assert claim_skeleton("the v2 model beat v1") == "the v# model beat v#"
    assert claim_skeleton("batch_32 throughput held") == "batch_# throughput held"


def test_two_claims_about_different_subjects_are_not_paired() -> None:
    """`gpt3 scored 88` and `gpt4 scored 91` share a skeleton and are not one restated claim.

    The subject's digit and the score both changed. One restated claim changes one number, so
    two changed numbers means the subject moved. Delete the `len(changed) != 1` check and this
    goes red, quoting two different models as one retraction.
    """
    revs = [
        _rev("aaa1111", "2026-01-01", "gpt3 scored 88 on the internal set\n"),
        _rev("bbb2222", "2026-02-01", "gpt4 scored 91 on the internal set\n"),
    ]

    assert drifts(revs) == []


def test_a_model_size_change_is_not_a_restated_score() -> None:
    """`llama2-7b` against `llama2-13b` is the same failure wearing a different name.

    Two numbers changed, the size and the score, so this is a different subject. Delete the
    `len(changed) != 1` check and it goes red.
    """
    revs = [
        _rev("aaa1111", "2026-01-01", "llama2-7b reached 0.80 on the held-out split\n"),
        _rev("bbb2222", "2026-02-01", "llama2-13b reached 0.85 on the held-out split\n"),
    ]

    assert drifts(revs) == []


def test_a_literal_hash_in_the_text_does_not_fake_a_single_change() -> None:
    """The skeleton placeholder is `#`, so a literal `#` fills a slot a number could fill.

    One line holds one number and a literal `#`; the other holds two numbers. Their skeletons
    match, their counts do not, and comparing them anyway lets `zip` truncate so a two-number
    change reads as one. Delete the length check and this goes red.
    """
    revs = [
        _rev("aaa1111", "2026-01-01", "count is 5 # done in the table\n"),
        _rev("bbb2222", "2026-02-01", "count is 7 9 done in the table\n"),
    ]

    assert drifts(revs) == []


def test_an_ordered_list_renumbering_is_not_a_restated_claim() -> None:
    """Taken from this repository's own docs/WRITEUP.md, where an item was inserted above.

    The list marker moved from 2 to 3 and the sentence after it is byte-identical. Delete the
    `LIST_MARKER` check and this goes red, reporting a renumber as a retraction.
    """
    claim = "**Confident retrieval on a gap.** The agent asks something the memory misses"
    revs = [
        _rev("aaa1111", "2026-01-01", f"2. {claim}\n"),
        _rev("bbb2222", "2026-02-01", f"3. {claim}\n"),
    ]

    assert drifts(revs) == []
