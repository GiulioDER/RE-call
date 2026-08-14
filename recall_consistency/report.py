"""Render audit findings as the corpus consistency report a caller reads.

The rule that holds this file to what it can defend: it never asserts a result from a scan that
did not run. `render` is told exactly what happened, how many documents, how many revisions,
how many questions, and every section says so before it says anything else. A report that
overstates its own coverage is worse than no report at all.
"""
from __future__ import annotations

from recall_consistency.findings import ClaimDrift, StaleAnswer

#: The Method section, written from `recall_consistency/claim_drift.py` and `history_corpus.py`
#: directly. Six guards live in `claim_drift.py`; keep this list and that file in sync.
_METHOD = (
    "Every revision of every tracked markdown document that git could read under its current "
    "name was read from history. Renames start a fresh history, deleted files stop at their "
    "last revision, and a shallow clone only carries what it fetched, so coverage is bounded by "
    "what the repository still holds.\n"
    "\n"
    "Restated claims are lines whose words held steady while exactly one number moved. Six "
    "things are deliberately skipped rather than guessed at: a claim that was rewritten rather "
    "than restated, a sentence shape that occurs twice in one document, a line where more than "
    "one number moved (usually the subject changed rather than the measurement), an ordered "
    "list item renumbered by an insertion above it, a line whose numeric count differs from its "
    "counterpart, and a line with too little text around its numbers to identify a claim. So "
    "this is a floor rather than a complete count.\n"
    "\n"
    "One limit worth knowing before quoting any single finding: lines are matched by their own "
    "text alone, with no surrounding context. A document restructured so that the same sentence "
    "shape now sits under a different subject can produce a pairing that reads as a retraction "
    "but is not one. Check the two dates and what surrounds each line before you quote it."
)

#: The lead sentence for stale answers whose nearest match was genuinely superseded.
_SUPERSEDED_BASELINE = (
    "The comparison is the highest-cosine hit among the results this retrieval returned, "
    "against what the trust layer served instead. The pool is hybrid and may be reranked, so "
    "this is a statement about the results in hand and not a measurement against a dense-only "
    "baseline."
)


def _plural(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _drift_section(drifts: list[ClaimDrift], *, documents: int, revisions: int) -> list[str]:
    if not drifts:
        return [
            f"No restated claims found in {_plural(revisions, 'revision', 'revisions')} of "
            f"{_plural(documents, 'document', 'documents')}."
        ]
    lines = [
        f"Found {_plural(len(drifts), 'restated claim', 'restated claims')}: the same sentence, "
        "a different number. Both versions are in the corpus, so both are retrievable.",
        "",
    ]
    for d in drifts:
        lines += [
            f"### `{d.path}`",
            "",
            f"- **{d.old_date}** (`{d.old_sha}`): {d.old_line}",
            f"- **{d.new_date}** (`{d.new_sha}`): {d.new_line}",
            "",
        ]
    return lines


def _stale_group(items: list[StaleAnswer], lead: str) -> list[str]:
    lines = [lead, ""]
    for s in items:
        lines += [f"### {s.question}", ""]
        lines.append(f"- Nearest match: `{s.plain_top_file}`")
        if s.plain_top_superseded_by:
            lines.append(f"- Replaced by: `{s.plain_top_superseded_by}`")
        if s.trusted_abstained:
            lines.append("- RE-call abstained rather than answering from it.")
        else:
            lines.append(f"- RE-call served a hit with verdict `{s.trusted_verdict}` instead.")
        lines.append("")
    return lines


def _stale_section(stale: list[StaleAnswer], *, questions: int) -> list[str]:
    if questions == 0:
        return [
            "No questions were supplied, so this half of the audit did not run. It needs ten "
            "to twenty questions you actually ask your agent."
        ]
    if not stale:
        return [f"None of the {questions} questions supplied resolved to replaced text."]

    # Only the first group is "answered from replaced text": its nearest match was superseded.
    # The second group is an abstention over a match that was NOT superseded, which is a
    # different finding and must not be counted under the same claim.
    superseded = [s for s in stale if s.plain_top_superseded_by]
    abstained = [s for s in stale if not s.plain_top_superseded_by]

    lines: list[str] = []
    if superseded:
        lead = (
            f"{_plural(len(superseded), 'question', 'questions')} where the nearest match is "
            f"text the corpus has already replaced. {_SUPERSEDED_BASELINE}"
        )
        lines += _stale_group(superseded, lead)
    if abstained:
        if lines:
            lines.append("")
        lead = (
            f"{_plural(len(abstained), 'question', 'questions')} where the trust layer "
            "abstained even though the nearest match was current, not replaced, text: a plain "
            "retriever would have answered confidently from it."
        )
        lines += _stale_group(abstained, lead)
    return lines


def render(
    corpus_name: str,
    drifts: list[ClaimDrift],
    stale: list[StaleAnswer],
    *,
    documents: int,
    revisions: int,
    questions: int,
) -> str:
    """The full audit report for one corpus.

    `documents`, `revisions` and `questions` are facts about what actually ran, not the sizes of
    `drifts` and `stale`: this is what lets the report tell "looked and found nothing" apart
    from "did not look".
    """
    lines = [
        f"# Memory audit: {corpus_name}",
        "",
        f"Read {_plural(revisions, 'revision', 'revisions')} of "
        f"{_plural(documents, 'tracked markdown document', 'tracked markdown documents')}.",
        "",
    ]
    if documents == 0 and questions == 0:
        lines += [
            "No tracked markdown documents matched, so nothing was audited. This is not a "
            "clean result, it is an empty one: check the glob and that the path is a git "
            "repository.",
            "",
        ]
        return "\n".join(lines)

    if documents == 0:
        # `stale` comes from an independent probe against an already-indexed corpus, not from
        # this run's glob match, so a completed probe's real findings must never be dropped just
        # because the claim-drift scan found no markdown here. Silently discarding them would be
        # exactly the kind of under-report this tool exists to refuse to produce.
        lines += [
            "No tracked markdown documents matched, so the restated-claims scan found nothing "
            "to read: check the glob and that the path is a git repository. Questions were "
            "still supplied and run against the already-indexed corpus, and are reported below.",
            "",
            "## Questions answered from replaced text",
            "",
        ]
        lines += _stale_section(stale, questions=questions)
        lines.append("")
        return "\n".join(lines)

    lines += [
        "This report reads the full history of the corpus, not its current state. Agent memory "
        "is append-only, so a claim and its correction are both retrievable, and the correction "
        "is not always the nearer match.",
        "",
        "## Claims the corpus restated",
        "",
    ]
    lines += _drift_section(drifts, documents=documents, revisions=revisions)
    lines += ["", "## Questions answered from replaced text", ""]
    lines += _stale_section(stale, questions=questions)
    lines += ["", "## Method", "", _METHOD, ""]
    return "\n".join(lines)
