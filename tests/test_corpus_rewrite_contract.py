"""`recall.rewrite`: the only path that writes a REVIEWED proposal back into a corpus file.

`promotion.py` has held a complete `proposed → reviewed → accepted → promoted` machine since
session 3 with zero callers outside its own tests. This is its first one, and being the first
caller is the whole risk: every invariant that machine enforces is enforced *for* something now,
and a write path that quietly accepts an unreviewed proposal would make all of it decorative.

The properties this file pins, one test each:

1.  **Human prose is not touched.** Compared on ``read_bytes()``, never on parsed text — a
    comparison of parsed text cannot see a lost BOM or a converted line ending, which are exactly
    the two defects the existing writer has. Carries a mutate-one-character positive control, so
    the comparison is provably able to fail.
2.  **A UTF-8 BOM survives.** `parse_frontmatter` tolerates a BOM *precisely because* Windows
    editors add one (`frontmatter.py:24`); a writer that drops it corrupts the file it was asked
    to annotate.
3.  **CRLF accounting.** A CRLF file gains a CRLF line, not an LF one. Byte-counted, and the
    file must contain no lone LF afterwards.
4.  **Permission bits survive** the atomic swap.
5.  **Atomicity.** With ``os.replace`` made to fail, the original survives byte-for-byte and no
    temp file is left behind.
6.  **Dry run is the default and writes nothing** — matching `cli.py:954`, whose reason applies
    with more force here: this rewrites the user's own memory.
7.  **An unreviewed fact is refused** — one rejection path per missing review field, each with
    its own ``pytest.raises(match=<field>)``.
8.  **The write site actually calls the validator**, rather than merely having one available.
9.  **A rejected proposal does not reappear** on the next run. `fix.py:62` names this failure
    directly: "a proposal tool whose output must itself be filtered has not saved anyone any
    work."
10. **An existing `supersedes:` is never silently overwritten** (`fix.py:264`).
11. **Fixed point.** A non-empty first pass — asserted non-empty *first*, because a fixed-point
    test over an empty set is vacuous, and an empty set is precisely the state `fix.py` is in on
    the real 792-memo corpus — then apply, then re-extract yields nothing new, while files that
    were not edited still produce output.
"""
from __future__ import annotations

import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from recall.fix import propose_fixes
from recall.promotion import (
    PromotedFact,
    accept_reviewed_proposal,
    promote_accepted_proposal,
    review_proposal,
)
from recall.reasoning_proposals import InferenceProposal
from recall.rewrite import (
    DERIVED_BEGIN,
    DERIVED_END,
    RejectionLedger,
    apply_rewrite,
    human_body,
    plan_rewrite,
    promoted_prose_edge,
    prose_edge_proposal_id,
)

REVIEWED_AT = datetime(2026, 8, 11, 9, 0, tzinfo=timezone.utc)
PROMOTED_AT = datetime(2026, 8, 11, 10, 0, tzinfo=timezone.utc)


def _proposal(
    subject_id: str,
    object_id: str,
    *,
    relation: str = "supersedes",
    evidence: tuple[str, ...] = ("ev-1",),
) -> InferenceProposal:
    """A raw proposal. Its ``id`` is supplied directly: the content-hash recomputation in
    `_providers.py:86` guards the *provider* boundary, and this path never crosses it."""
    return InferenceProposal(
        id=f"ip_{subject_id}_{relation}_{object_id}",
        source_evidence_ids=evidence,
        proposed_relation=relation,  # type: ignore[arg-type]
        subject_id=subject_id,
        object_id=object_id,
        explanation="prose stated the relation",
        model_id="test-model",
        pipeline_id="test-pipeline",
        provider_id="test-provider",
        provider_revision="rev-1",
        confidence=0.9,
        uncertainty=(),
        generation_id="gen-1",
    )


def _promoted(subject_id: str, object_id: str, *, relation: str = "supersedes") -> PromotedFact:
    """Drive the real promotion machine end to end, rather than hand-building a PromotedFact.

    Hand-building one would let this file's fixtures drift away from what `promotion.py` actually
    produces, and the point of the exercise is that the write path consumes the real output.
    """
    review = review_proposal(
        _proposal(subject_id, object_id, relation=relation),
        reviewer_id="giulio",
        reviewed_at=REVIEWED_AT,
        audit_note="checked the prose against both memos",
    )
    return promote_accepted_proposal(accept_reviewed_proposal(review), promoted_at=PROMOTED_AT)


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def _body_after_frontmatter(raw: bytes) -> bytes:
    """Everything after the closing ``---`` fence, as raw bytes.

    Deliberately byte-level and fence-counting rather than reusing `parse_frontmatter`: this is
    the oracle for "the human's prose is untouched", and an oracle built from the parser under
    test would agree with the writer about any transformation both of them perform.
    """
    body = raw.lstrip(b"\xef\xbb\xbf")
    if not body.lstrip().startswith(b"---"):
        return raw
    parts = body.split(b"---", 2)
    return parts[2] if len(parts) > 2 else raw


# --- 1. human prose is not touched ------------------------------------------------------------


def test_applying_a_fact_leaves_the_human_prose_byte_identical(tmp_path):
    memo = tmp_path / "new.md"
    original = b"---\ntitle: keep me\n---\n\nprecious body, every byte of it\n"
    _write(memo, original)
    _write(tmp_path / "old.md", b"# old\n")

    apply_rewrite(tmp_path, _promoted("new.md", "old.md"), apply=True)

    after = memo.read_bytes()
    assert _body_after_frontmatter(after) == _body_after_frontmatter(original)
    assert b"supersedes: old.md" in after, "the edge must actually have been declared"


def test_the_prose_comparison_can_fail(tmp_path):
    """Positive control for the guard above.

    Without this, a `_body_after_frontmatter` that returned a constant would make every prose
    assertion in this file pass forever. Mutating one character must be visible to the oracle.
    """
    original = b"---\ntitle: keep me\n---\n\nprecious body, every byte of it\n"
    mutated = original.replace(b"precious", b"preciouS", 1)
    assert mutated != original
    assert _body_after_frontmatter(mutated) != _body_after_frontmatter(original)


# --- 2. BOM ------------------------------------------------------------------------------------


def test_a_utf8_bom_survives_the_rewrite(tmp_path):
    """`fix.apply_proposal` reads ``utf-8-sig`` and writes plain ``utf-8``, so a Windows-authored
    memo silently loses its BOM. `parse_frontmatter` tolerates a BOM on purpose; a writer that
    strips it is undoing a deliberate accommodation."""
    memo = tmp_path / "new.md"
    _write(memo, b"\xef\xbb\xbf---\ntitle: t\n---\n\nbody\n")
    _write(tmp_path / "old.md", b"# old\n")

    apply_rewrite(tmp_path, _promoted("new.md", "old.md"), apply=True)

    after = memo.read_bytes()
    assert after.startswith(b"\xef\xbb\xbf"), "the BOM was dropped"
    assert after.count(b"\xef\xbb\xbf") == 1, "the BOM was duplicated"
    assert b"supersedes: old.md" in after


def test_a_file_without_a_bom_does_not_gain_one(tmp_path):
    """The non-over-rejection half: preserving a BOM must not mean inventing one."""
    memo = tmp_path / "new.md"
    _write(memo, b"---\ntitle: t\n---\n\nbody\n")
    _write(tmp_path / "old.md", b"# old\n")

    apply_rewrite(tmp_path, _promoted("new.md", "old.md"), apply=True)

    assert not memo.read_bytes().startswith(b"\xef\xbb\xbf")


# --- 3. CRLF accounting ------------------------------------------------------------------------


def test_a_crlf_memo_gains_a_crlf_line_not_an_lf_one(tmp_path):
    """`fix.apply_proposal` splits and rejoins on ``"\\n"``, so a CRLF file gains one LF-only
    line — a mixed-ending file that every diff tool will report as wholly rewritten."""
    memo = tmp_path / "new.md"
    original = b"---\r\ntitle: t\r\n---\r\n\r\nbody line one\r\nbody line two\r\n"
    _write(memo, original)
    _write(tmp_path / "old.md", b"# old\n")

    apply_rewrite(tmp_path, _promoted("new.md", "old.md"), apply=True)

    after = memo.read_bytes()
    assert after.count(b"\r\n") == original.count(b"\r\n") + 1, "exactly one CRLF line added"
    assert after.replace(b"\r\n", b"").count(b"\n") == 0, "a lone LF was introduced"
    assert b"supersedes: old.md\r\n" in after


def test_an_lf_memo_does_not_gain_a_crlf_line(tmp_path):
    """Non-over-rejection: honouring CRLF must not mean imposing it."""
    memo = tmp_path / "new.md"
    _write(memo, b"---\ntitle: t\n---\n\nbody\n")
    _write(tmp_path / "old.md", b"# old\n")

    apply_rewrite(tmp_path, _promoted("new.md", "old.md"), apply=True)

    assert b"\r\n" not in memo.read_bytes()


# --- 4. permission bits ------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows st_mode carries only the read-only bit, and a read-only destination cannot "
           "be os.replace'd, so there is no mode this assertion could distinguish. The guard is "
           "falsifiable on POSIX only; asserting equality of two identical 0o666s here would be "
           "a test that cannot fail.",
)
def test_permission_bits_survive_the_atomic_swap(tmp_path):
    """`mkstemp` creates the staging file 0o600. Without an explicit `copymode` the swap silently
    tightens the user's memo — or, running as root, re-owns it."""
    memo = tmp_path / "new.md"
    _write(memo, b"---\ntitle: t\n---\n\nbody\n")
    _write(tmp_path / "old.md", b"# old\n")
    os.chmod(memo, 0o640)
    before = stat.S_IMODE(memo.stat().st_mode)
    assert before == 0o640, "the fixture itself must take effect for this test to mean anything"

    apply_rewrite(tmp_path, _promoted("new.md", "old.md"), apply=True)

    assert stat.S_IMODE(memo.stat().st_mode) == before


# --- 5. atomicity ------------------------------------------------------------------------------


def test_a_failed_rename_leaves_the_memo_untouched(tmp_path, monkeypatch):
    """The new writer gets its own atomicity proof rather than inheriting `test_fix.py`'s.

    That test pins `fix.apply_proposal`; nothing about it constrains this module, and the shared
    `recall.atomic_write` implementation is exactly the kind of code that gets a fast path added
    to it later by someone who only ran the other file's tests.
    """
    memo = tmp_path / "new.md"
    original = b"---\ntitle: t\n---\n\nprecious body\n"
    _write(memo, original)
    _write(tmp_path / "old.md", b"# old\n")

    real_replace = os.replace

    def failing_replace(src, dst, *a, **k):
        if os.fspath(dst).replace("\\", "/").endswith("/new.md"):
            raise OSError(28, "simulated ENOSPC on atomic rename")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError):
        apply_rewrite(tmp_path, _promoted("new.md", "old.md"), apply=True)

    assert memo.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["new.md", "old.md"], "temp litter left"


# --- 6. dry run is the default -----------------------------------------------------------------


def test_dry_run_is_the_default_and_writes_nothing(tmp_path):
    memo = tmp_path / "new.md"
    original = b"---\ntitle: t\n---\n\nbody\n"
    _write(memo, original)
    _write(tmp_path / "old.md", b"# old\n")

    plan = apply_rewrite(tmp_path, _promoted("new.md", "old.md"))

    assert memo.read_bytes() == original, "the default call wrote to disk"
    assert plan.planned, "a dry run must still report what it would have done"
    assert plan.written is False


# --- 7. an unreviewed fact is refused ----------------------------------------------------------


def test_a_fact_without_a_reviewer_is_refused(tmp_path):
    from dataclasses import replace

    fact = replace(_promoted("new.md", "old.md"), reviewer_id="")
    with pytest.raises(ValueError, match="reviewer_id"):
        apply_rewrite(tmp_path, fact, apply=True)


def test_a_fact_without_an_audit_note_is_refused(tmp_path):
    from dataclasses import replace

    fact = replace(_promoted("new.md", "old.md"), audit_note="   ")
    with pytest.raises(ValueError, match="audit_note"):
        apply_rewrite(tmp_path, fact, apply=True)


def test_a_fact_without_evidence_ids_is_refused(tmp_path):
    from dataclasses import replace

    fact = replace(_promoted("new.md", "old.md"), proposal_evidence_ids=())
    with pytest.raises(ValueError, match="proposal_evidence_ids"):
        apply_rewrite(tmp_path, fact, apply=True)


def test_a_fact_without_provider_identity_is_refused(tmp_path):
    from dataclasses import replace

    fact = replace(_promoted("new.md", "old.md"), source_model_revision="")
    with pytest.raises(ValueError, match="source_model_revision"):
        apply_rewrite(tmp_path, fact, apply=True)


def test_a_fact_that_is_not_promoted_is_refused(tmp_path):
    from dataclasses import replace

    fact = replace(_promoted("new.md", "old.md"), state="accepted")
    with pytest.raises(ValueError, match="state"):
        apply_rewrite(tmp_path, fact, apply=True)


def test_a_fully_reviewed_fact_is_accepted(tmp_path):
    """Non-over-rejection. Five rejection paths above are worthless if the sixth case, the valid
    one, is also refused."""
    _write(tmp_path / "new.md", b"---\ntitle: t\n---\n\nbody\n")
    _write(tmp_path / "old.md", b"# old\n")
    result = apply_rewrite(tmp_path, _promoted("new.md", "old.md"), apply=True)
    assert result.written is True


# --- 8. the write site calls the validator -----------------------------------------------------


def test_the_write_site_calls_metadata_is_trusted(tmp_path, monkeypatch):
    """Having a validator and calling it are different claims, and only the second one protects
    anything. Replacing `metadata_is_trusted` with a refusal must stop the write."""
    import recall.rewrite as rewrite_mod

    _write(tmp_path / "new.md", b"---\ntitle: t\n---\n\nbody\n")
    _write(tmp_path / "old.md", b"# old\n")
    calls: list[object] = []

    def refusing(value: object) -> bool:
        calls.append(value)
        return False

    monkeypatch.setattr(rewrite_mod, "metadata_is_trusted", refusing)

    with pytest.raises(ValueError):
        apply_rewrite(tmp_path, _promoted("new.md", "old.md"), apply=True)
    assert calls, "apply_rewrite never consulted metadata_is_trusted"


# --- 9. a rejected proposal does not reappear --------------------------------------------------


def test_a_rejected_proposal_does_not_reappear(tmp_path):
    fact = _promoted("new.md", "old.md")
    _write(tmp_path / "new.md", b"---\ntitle: t\n---\n\nbody\n")
    _write(tmp_path / "old.md", b"# old\n")
    ledger = RejectionLedger(tmp_path / ".recall" / "rejections.sqlite3")

    first = apply_rewrite(tmp_path, fact, ledger=ledger)
    assert first.planned, "the first pass must offer it, or the rejection means nothing"

    ledger.reject(fact, reviewer_id="giulio", note="augments, does not replace")

    second = apply_rewrite(tmp_path, fact, ledger=ledger)
    assert second.planned is False
    assert second.skipped_reason and "rejected" in second.skipped_reason


def test_the_rejection_ledger_survives_a_reopen(tmp_path):
    """Durable is the whole requirement. An in-memory set would pass the test above."""
    fact = _promoted("new.md", "old.md")
    path = tmp_path / ".recall" / "rejections.sqlite3"
    ledger = RejectionLedger(path)
    ledger.reject(fact, reviewer_id="giulio", note="augments")
    ledger.close()

    assert RejectionLedger(path).is_rejected(fact) is True


def test_an_unrelated_proposal_is_not_swept_up_by_a_rejection(tmp_path):
    """Non-over-rejection: the ledger is keyed on proposal identity, not on the file."""
    rejected = _promoted("new.md", "old.md")
    other = _promoted("new.md", "third.md")
    ledger = RejectionLedger(tmp_path / "rejections.sqlite3")
    ledger.reject(rejected, reviewer_id="giulio", note="no")

    assert ledger.is_rejected(rejected) is True
    assert ledger.is_rejected(other) is False


# --- 10. an existing supersedes: is never overwritten ------------------------------------------


def test_an_existing_supersedes_is_never_silently_overwritten(tmp_path):
    """`fix.py:264` refuses this case; a second writer that does not would make that refusal
    pointless, since either tool can be the one that runs."""
    memo = tmp_path / "new.md"
    original = b"---\nsupersedes: already_declared.md\n---\n\nbody\n"
    _write(memo, original)
    _write(tmp_path / "old.md", b"# old\n")

    result = apply_rewrite(tmp_path, _promoted("new.md", "old.md"), apply=True)

    assert memo.read_bytes() == original
    assert result.written is False
    assert result.skipped_reason and "already declares" in result.skipped_reason


# --- destinations: frontmatter vs the derived block --------------------------------------------


def test_contradicts_goes_to_the_derived_block_not_the_frontmatter(tmp_path):
    """The schema recognises exactly three keys (`frontmatter.py:12`). A `contradicts:` line in
    the frontmatter would be silently dropped by `parse_frontmatter` and read by a human as
    declared — the worst combination."""
    memo = tmp_path / "new.md"
    _write(memo, b"---\ntitle: t\n---\n\nbody\n")
    _write(tmp_path / "old.md", b"# old\n")

    apply_rewrite(tmp_path, _promoted("new.md", "old.md", relation="contradicts"), apply=True)

    text = memo.read_text(encoding="utf-8")
    head, _, tail = text.partition(DERIVED_BEGIN)
    assert "contradicts" not in head, "a derived relation leaked into the frontmatter"
    assert "contradicts: old.md" in tail
    assert DERIVED_END in tail


def test_a_relation_with_no_defined_destination_is_refused(tmp_path):
    _write(tmp_path / "new.md", b"---\ntitle: t\n---\n\nbody\n")
    _write(tmp_path / "old.md", b"# old\n")
    with pytest.raises(ValueError, match="relation"):
        apply_rewrite(tmp_path, _promoted("new.md", "old.md", relation="references"), apply=True)


def test_human_body_excludes_the_frontmatter_and_the_derived_block(tmp_path):
    """`human_body` is the extractor's prompt input (and its cache key). If the derived block
    leaked into it, this system's own writes would change the model's input, and a re-run after
    an apply would re-invoke the model on text it had already judged."""
    memo = tmp_path / "new.md"
    _write(memo, b"---\ntitle: t\n---\n\nreal prose\n")
    _write(tmp_path / "old.md", b"# old\n")
    before = human_body(memo.read_text(encoding="utf-8"))

    apply_rewrite(tmp_path, _promoted("new.md", "old.md", relation="contradicts"), apply=True)

    assert human_body(memo.read_text(encoding="utf-8")) == before
    assert before == "real prose"


# --- defects found by the bug audit of d8e236d ------------------------------------------------


def test_a_memo_quoting_the_end_marker_does_not_get_prose_spliced(tmp_path):
    """`_insert_derived_line` required only that both markers EXIST, not that BEGIN came first.

    A memo that merely quotes the closing marker in its prose therefore had machine-written text
    inserted into the middle of the author's sentence. Worse than a cosmetic corruption: it
    changes `human_body`, which changes the claim cache key, which re-invokes the model on prose
    no human edited. That is the exact re-invocation the extractor's design exists to prevent.
    """
    memo = tmp_path / "new.md"
    prose = f"# memo\n\nthe closer is {DERIVED_END} in prose\n"
    _write(memo, prose.encode("utf-8"))
    _write(tmp_path / "old.md", b"# old\n")
    _write(tmp_path / "third.md", b"# third\n")

    apply_rewrite(tmp_path, _promoted("new.md", "old.md", relation="contradicts"), apply=True)
    apply_rewrite(tmp_path, _promoted("new.md", "third.md", relation="same_entity"), apply=True)

    text = memo.read_text(encoding="utf-8")
    assert "the closer is <!-- /recall:derived --> in prose" in text, (
        "the author's sentence was spliced"
    )
    assert human_body(memo.read_text(encoding="utf-8")).startswith("# memo")


def test_prose_that_merely_mentions_a_relation_does_not_suppress_a_real_write(tmp_path):
    """The duplicate check was a substring test over the WHOLE FILE, so an author quoting
    `status: something.md` anywhere in their prose silently suppressed a reviewed edge and told
    the operator it was already recorded. A refusal that reports a falsehood is worse than none.
    """
    memo = tmp_path / "new.md"
    _write(memo, b"---\ntitle: t\n---\n\nA reviewer once wrote: status: old.md was the label.\n")
    _write(tmp_path / "old.md", b"# old\n")

    result = apply_rewrite(tmp_path, _promoted("new.md", "old.md", relation="status"), apply=True)

    assert result.written is True, f"a real write was suppressed: {result.skipped_reason}"
    text = memo.read_text(encoding="utf-8")
    assert DERIVED_BEGIN in text and "status: old.md" in text.split(DERIVED_BEGIN)[1]


def test_a_genuine_duplicate_in_the_derived_block_is_still_skipped(tmp_path):
    """Non-over-rejection for the fix above: narrowing the check to the block must not lose it."""
    memo = tmp_path / "new.md"
    _write(memo, b"---\ntitle: t\n---\n\nbody\n")
    _write(tmp_path / "old.md", b"# old\n")

    first = apply_rewrite(tmp_path, _promoted("new.md", "old.md", relation="contradicts"),
                          apply=True)
    second = apply_rewrite(tmp_path, _promoted("new.md", "old.md", relation="contradicts"),
                           apply=True)

    assert first.written is True
    assert second.written is False
    assert memo.read_text(encoding="utf-8").count("contradicts: old.md") == 1


def test_an_unclosed_frontmatter_block_is_refused_not_double_declared(tmp_path):
    """`parse_frontmatter` returns {} for an unclosed block, so the "already declares" guard
    could not fire and the writer prepended a SECOND block. The file then visibly stated two
    different predecessors, retrieval acted on the new one, and the result reported success."""
    memo = tmp_path / "new.md"
    original = b"---\nsupersedes: already_declared.md\ntitle: t\n"
    _write(memo, original)
    _write(tmp_path / "old.md", b"# old\n")

    result = apply_rewrite(tmp_path, _promoted("new.md", "old.md"), apply=True)

    assert memo.read_bytes() == original, "an unclosed block was rewritten"
    assert result.written is False
    assert result.skipped_reason and "unclosed" in result.skipped_reason


def test_a_file_with_no_frontmatter_at_all_still_gains_one(tmp_path):
    """Non-over-rejection: refusing an UNCLOSED block must not mean refusing an ABSENT one."""
    memo = tmp_path / "new.md"
    _write(memo, b"# new\n\njust prose, no block\n")
    _write(tmp_path / "old.md", b"# old\n")

    result = apply_rewrite(tmp_path, _promoted("new.md", "old.md"), apply=True)

    assert result.written is True
    assert memo.read_bytes().startswith(b"---\nsupersedes: old.md\n---\n")


@pytest.mark.parametrize("marker_name", ["begin", "end"])
def test_writing_a_derived_record_never_changes_human_body(tmp_path, marker_name):
    """The other half of the spliced-marker defect, and the one with the running cost.

    Fixing the WRITER stopped the bytes being corrupted, but `human_body` still took the first
    BEGIN and the first END independently. For a memo whose prose contains either marker that
    picked the wrong span: five words of real prose vanished from the extractor's input in one
    direction, and the machine-written block was INCLUDED in it in the other. Either way the
    sha256 that keys the claim cache moves, so applying an accepted proposal makes the next
    `recall extract` pay for a re-call of prose no human edited. The module docstring promises
    exactly the opposite.
    """
    marker = DERIVED_BEGIN if marker_name == "begin" else DERIVED_END
    memo = tmp_path / "new.md"
    memo.write_text(f"# memo\n\nI use {marker} as a marker in prose\n", encoding="utf-8",
                    newline="")
    _write(tmp_path / "old.md", b"# old\n")
    before = human_body(memo.read_text(encoding="utf-8"))

    apply_rewrite(tmp_path, _promoted("new.md", "old.md", relation="contradicts"), apply=True)

    after = human_body(memo.read_text(encoding="utf-8"))
    assert after == before, "the extractor's input, and so its cache key, moved"
    assert marker in after, "the author's own use of the marker was eaten"


def test_a_record_in_a_second_derived_block_is_not_written_again(tmp_path):
    """`_derived_records` inspected only the first block, so a record already present in a later
    one was reported as a fresh write and duplicated."""
    memo = tmp_path / "new.md"
    memo.write_text(
        f"# memo\n\nprose\n\n{DERIVED_BEGIN}\n{DERIVED_END}\n\n"
        f"{DERIVED_BEGIN}\ncontradicts: old.md\n{DERIVED_END}\n",
        encoding="utf-8", newline="",
    )
    _write(tmp_path / "old.md", b"# old\n")

    result = apply_rewrite(tmp_path, _promoted("new.md", "old.md", relation="contradicts"),
                           apply=True)

    assert result.written is False
    assert memo.read_text(encoding="utf-8").count("contradicts: old.md") == 1


def test_the_other_writer_also_refuses_an_unclosed_frontmatter_block(tmp_path):
    """`recall lint --fix` and `recall rewrite` write the same files, so a guard on one of them
    is a guard on neither. `fix.apply_proposal` still prepended a second block, producing the
    double declaration this project calls corruption, and it is the unguarded one that runs
    unattended."""
    from recall.fix import Proposal, apply_proposal

    memo = tmp_path / "new.md"
    original = b"---\nsupersedes: already_declared.md\ntitle: t\n"
    _write(memo, original)

    with pytest.raises(ValueError, match="unclosed"):
        apply_proposal(tmp_path, Proposal(
            edit_file="new.md", target="old.md",
            evidence_file="other.md", evidence="supersedes [[new]]",
        ))
    assert memo.read_bytes() == original


def test_the_other_writers_unattended_loop_survives_the_new_refusal(tmp_path, capsys):
    """Adding a refusal to `apply_proposal` gave `recall lint --fix --apply` an exception it
    never had, in a bare loop that writes the user's memos unattended.

    Left alone it would abort part-way through a corpus, having already written some files, and
    never reach its own `wrote N edge(s)` line — so the operator gets a traceback and no record
    of what was changed. A guard that turns silent corruption into a half-finished run with no
    report has moved the problem, not fixed it.
    """
    from recall.cli import main

    _write(tmp_path / "broken.md", b"---\nsupersedes: pinned.md\ntitle: t\n")
    _write(tmp_path / "old_a_2026.md", b"# a\n\nthe original\n")
    _write(tmp_path / "good.md", b"# good\n\nThis supersedes [[old_a_2026]].\n")
    _write(tmp_path / "old_b_2026.md", b"# b\n\nSuperseded by [[broken]].\n")

    main(["lint", str(tmp_path), "--fix", "--apply"])

    out = capsys.readouterr().out
    assert b"supersedes: old_a_2026" in (tmp_path / "good.md").read_bytes(), (
        "the run aborted before writing the memo it could write"
    )
    assert "SKIP" in out and "unclosed" in out
    assert "wrote 1 edge(s)" in out, f"the count did not match what was written: {out}"


def test_the_other_writer_still_writes_a_well_formed_memo(tmp_path):
    """Non-over-rejection for the guard above."""
    from recall.fix import Proposal, apply_proposal

    memo = tmp_path / "new.md"
    _write(memo, b"---\ntitle: t\n---\n\nbody\n")

    apply_proposal(tmp_path, Proposal(
        edit_file="new.md", target="old.md",
        evidence_file="other.md", evidence="supersedes [[new]]",
    ))
    assert b"supersedes: old.md" in memo.read_bytes()


# --- 11. fixed point ---------------------------------------------------------------------------


def test_applying_the_proposals_reaches_a_fixed_point(tmp_path):
    """First pass non-empty, asserted BEFORE anything is applied.

    A fixed-point test that never had anything to apply proves only that nothing happened, and
    "nothing happened" is the state `fix.py` is measurably in on the real corpus — so the vacuous
    version of this test would pass there too, while telling us nothing.
    """
    _write(tmp_path / "old_thing_2026.md", b"# old\n\nthe original\n")
    _write(tmp_path / "new.md", b"# new\n\nThis supersedes [[old_thing_2026]].\n")
    _write(tmp_path / "untouched.md", b"# other\n\nThis supersedes [[old_thing_2026]].\n")

    first, _ = propose_fixes(tmp_path)
    assert first, "the first pass must be non-empty or this test is vacuous"
    edited = {p.edit_file for p in first}
    assert "new.md" in edited

    for p in first:
        if p.edit_file == "new.md":
            apply_rewrite(tmp_path, _promoted("new.md", p.target), apply=True)

    second, _ = propose_fixes(tmp_path)
    assert "new.md" not in {p.edit_file for p in second}, "an applied edge was re-proposed"
    assert "untouched.md" in {p.edit_file for p in second}, (
        "the re-extraction went silent everywhere, which would make the fixed point meaningless"
    )


# --- the bridge from prose edges into the review machine ---------------------------------------


def test_a_prose_edge_cannot_be_promoted_without_a_reviewer():
    """`--reviewer` being a required flag protects the CLI, and only the CLI.

    A surviving mutant found this gap: substituting a placeholder reviewer inside
    `promoted_prose_edge` broke nothing, because every test reached that function through
    argparse, which had already rejected the empty case. The next caller will not be argparse.
    """
    with pytest.raises(ValueError, match="reviewer"):
        promoted_prose_edge(
            edit_file="new.md", target="old.md", evidence_file="new.md",
            evidence="supersedes [[old]]", reviewer_id="   ",
            audit_note="checked it", at=PROMOTED_AT,
        )


def test_a_prose_edge_cannot_be_promoted_without_an_audit_note():
    with pytest.raises(ValueError, match="audit note"):
        promoted_prose_edge(
            edit_file="new.md", target="old.md", evidence_file="new.md",
            evidence="supersedes [[old]]", reviewer_id="giulio",
            audit_note="  ", at=PROMOTED_AT,
        )


def test_the_same_prose_edge_hashes_to_the_same_proposal_id():
    """The rejection ledger is keyed on this id and outlives the process that made it. An id
    that varied per run would let a reviewer's refusal expire silently."""
    args = ("new.md", "old.md", "new.md", "This supersedes [[old]].")
    assert prose_edge_proposal_id(*args) == prose_edge_proposal_id(*args)
    assert prose_edge_proposal_id(*args) != prose_edge_proposal_id("other.md", *args[1:])


def test_plan_rewrite_is_pure_and_names_the_file_it_would_edit(tmp_path):
    _write(tmp_path / "new.md", b"---\ntitle: t\n---\n\nbody\n")
    _write(tmp_path / "old.md", b"# old\n")
    before = (tmp_path / "new.md").read_bytes()

    plan = plan_rewrite(tmp_path, _promoted("new.md", "old.md"))

    assert plan.edit_file == "new.md"
    assert plan.key == "supersedes"
    assert plan.value == "old.md"
    assert plan.destination == "frontmatter"
    assert (tmp_path / "new.md").read_bytes() == before
