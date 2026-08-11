"""The reviewed write path's contract, enforced rather than described.

`recall/promotion.py` holds a complete `proposed → reviewed → accepted → promoted` machine and,
until now, no caller: a gate nothing passes through protects nothing. `recall/rewrite.py` is its
first caller, and it is the only code in this package besides `recall/fix.py` that edits a memo
the user wrote. That makes byte fidelity a correctness property, not a nicety — a rewrite that
"works" while quietly re-encoding the file has damaged the thing it was asked to annotate.

One test per property, each written so a plausible wrong implementation fails it:

* **direction** — `supersedes` is written on the SUPERSEDING memo (`object_id`), naming the
  superseded one. Inverting it demotes the live memo beneath the one it replaced;
* **human prose survives byte for byte**, with a mutate-one-character positive control so the
  guard is falsifiable rather than vacuously true;
* a **UTF-8 BOM** survives — `parse_frontmatter` tolerates one precisely because Windows editors
  add it, and `fix.apply_proposal` reads `utf-8-sig` and writes plain `utf-8`, so it eats one;
* **CRLF accounting** — a CRLF memo gains CRLF lines and no lone LF. `fix.apply_proposal` splits
  and rejoins on `"\\n"`, so its inserted line is LF-only inside a CRLF file;
* the file's **permission bits** survive the atomic swap;
* the original survives an **injected `os.replace` failure**;
* **dry run is the default** and writes nothing;
* metadata that is not a fully reviewed promotion is **refused before any write** — missing
  reviewer id, missing audit note;
* a **rejected claim does not reappear**, and its ledger key is generation independent, because a
  proposal id is not: `_proposal_id` mixes in `generation_id`, so a ledger keyed on it would
  forget every rejection at the next re-index;
* an existing `supersedes:` is **never silently overwritten**;
* `contradicts` / `same_entity` / `status` land in the derived block and **no new frontmatter key**
  is ever introduced;
* the **fixed point**: a non-empty first pass, then apply, then re-extract yields nothing new for
  the edited memo while an untouched one still produces output. The non-emptiness is asserted
  FIRST — a fixed-point test over an empty proposal set is the vacuous case, and it is exactly the
  state `recall/fix.py` is in on the real 792-memo corpus.
"""
from __future__ import annotations

import inspect
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest

from recall.frontmatter import VALIDITY_KEYS, parse_frontmatter
from recall.promotion import (
    PromotedFact,
    accept_reviewed_proposal,
    promote_accepted_proposal,
    review_proposal,
)
from recall.reasoning_proposals import InferenceProposal
from recall.rewrite import (
    DERIVED_CLOSE,
    DERIVED_KEYS,
    DERIVED_OPEN,
    RejectionLedger,
    RewriteRefused,
    apply_rewrite,
    claim_key,
    default_ledger_path,
    destination,
)
from recall.trust import metadata_is_trusted

_WHEN = datetime(2026, 8, 11, tzinfo=timezone.utc)


# --- fixtures -----------------------------------------------------------------------------------


def _proposal(
    *,
    relation: str = "supersedes",
    subject_id: str = "old_decision_2026-01-01.md",
    object_id: str = "new_decision_2026-06-01.md",
    generation_id: str = "gen-a",
) -> InferenceProposal:
    return InferenceProposal(
        id=f"ip_{relation}_{subject_id}_{object_id}_{generation_id}",
        source_evidence_ids=("node-a", "node-b"),
        proposed_relation=relation,  # type: ignore[arg-type]
        subject_id=subject_id,
        object_id=object_id,
        explanation="the body states the relation the frontmatter does not declare",
        model_id="rules",
        pipeline_id="pipe-a",
        provider_id="recall.deterministic",
        provider_revision="rev-a",
        confidence=0.9,
        uncertainty=(),
        generation_id=generation_id,
    )


def _fact(**kwargs: object) -> PromotedFact:
    """A PromotedFact that has actually been through the review machine."""
    reviewed = review_proposal(
        _proposal(**kwargs),  # type: ignore[arg-type]
        reviewer_id="reviewer@example.com",
        reviewed_at=_WHEN,
        audit_note="Read both memos; the supersession is stated and unhedged.",
    )
    return promote_accepted_proposal(accept_reviewed_proposal(reviewed), promoted_at=_WHEN)


def _memo(root: Path, name: str, raw: bytes) -> Path:
    path = root / name
    path.write_bytes(raw)
    return path


_OLD = b"# old decision\n\nThe original call, made in January.\n"
_NEW = (
    b"# new decision\n"
    b"\n"
    b"This supersedes [[old_decision_2026-01-01]] after the July review.\n"
    b"\n"
    b"Rationale follows in the usual place.\n"
)


def _corpus(root: Path) -> None:
    _memo(root, "old_decision_2026-01-01.md", _OLD)
    _memo(root, "new_decision_2026-06-01.md", _NEW)


# --- direction ----------------------------------------------------------------------------------


def test_supersedes_is_written_on_the_superseding_memo_naming_the_superseded_one(
    tmp_path: Path,
) -> None:
    """`object_id` is the memo that gains the key; `subject_id` is what it names.

    The graph builds `authored_supersedes` with ``from_file=target`` (superseded) and
    ``to_file=superseding``, and the deterministic proposer mirrors it: `subject_id` is the OLD
    document. The schema has no `superseded_by`, so the edge must land on the NEW memo. Writing it
    on the old one declares the live memo stale and demotes it beneath the memo it replaced — the
    exact failure the trust layer exists to prevent, caused by the tool meant to fix it.
    """
    _corpus(tmp_path)

    result = apply_rewrite(tmp_path, _fact(), apply=True)

    assert result.written is True
    assert result.plan is not None
    assert result.plan.edit_file == "new_decision_2026-06-01.md"
    assert result.plan.value == "old_decision_2026-01-01.md"

    new_meta, _ = parse_frontmatter(
        (tmp_path / "new_decision_2026-06-01.md").read_text(encoding="utf-8-sig")
    )
    old_meta, _ = parse_frontmatter(
        (tmp_path / "old_decision_2026-01-01.md").read_text(encoding="utf-8-sig")
    )
    assert new_meta["supersedes"] == "old_decision_2026-01-01.md"
    assert "supersedes" not in old_meta, "the superseded memo must not declare the edge"


# --- byte fidelity ------------------------------------------------------------------------------


def test_human_prose_survives_the_rewrite_byte_for_byte(tmp_path: Path) -> None:
    _corpus(tmp_path)

    apply_rewrite(tmp_path, _fact(), apply=True)

    raw = (tmp_path / "new_decision_2026-06-01.md").read_bytes()
    assert raw.endswith(_NEW), "every byte the human wrote must still be there, in order"


def test_the_prose_guard_fails_when_one_character_moves(tmp_path: Path) -> None:
    """Positive control for the test above.

    `endswith` over prose that the writer never touches passes for a writer that does nothing at
    all, and would keep passing if the comparison were against the wrong bytes. Mutating a single
    character of the expected prose must turn the guard red; if it does not, the guard is
    measuring nothing.
    """
    _corpus(tmp_path)

    apply_rewrite(tmp_path, _fact(), apply=True)

    raw = (tmp_path / "new_decision_2026-06-01.md").read_bytes()
    mutated = _NEW.replace(b"Rationale", b"Ratlonale")
    assert mutated != _NEW
    assert not raw.endswith(mutated), "the prose comparison is not actually comparing prose"


def test_a_utf8_bom_survives_the_rewrite(tmp_path: Path) -> None:
    """`parse_frontmatter` tolerates a BOM because Windows editors add one.

    `fix.apply_proposal` reads `utf-8-sig` (which strips it) and writes plain `utf-8` (which does
    not put it back), so it silently re-encodes a Windows-authored memo. Round-tripping a file
    through a tool must not change what the user's editor sees.
    """
    _memo(tmp_path, "old_decision_2026-01-01.md", _OLD)
    _memo(tmp_path, "new_decision_2026-06-01.md", b"\xef\xbb\xbf" + _NEW)

    apply_rewrite(tmp_path, _fact(), apply=True)

    raw = (tmp_path / "new_decision_2026-06-01.md").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "the BOM was eaten"
    assert raw.count(b"\xef\xbb\xbf") == 1, "the BOM was duplicated"
    meta, _ = parse_frontmatter(raw.decode("utf-8-sig"))
    assert meta["supersedes"] == "old_decision_2026-01-01.md"


def test_a_crlf_memo_gains_only_crlf_lines(tmp_path: Path) -> None:
    """CRLF accounting, counted rather than eyeballed.

    `fix.apply_proposal` splits and rejoins on `"\\n"`, so its inserted line ends LF inside an
    otherwise-CRLF file — one mixed line, invisible in an editor and visible in every diff. The
    count is asserted both ways: no lone LF at all, and exactly the number of CRLFs the original
    had plus the lines this rewrite added.
    """
    original = _NEW.replace(b"\n", b"\r\n")
    _memo(tmp_path, "old_decision_2026-01-01.md", _OLD)
    _memo(tmp_path, "new_decision_2026-06-01.md", original)

    apply_rewrite(tmp_path, _fact(), apply=True)

    raw = (tmp_path / "new_decision_2026-06-01.md").read_bytes()
    assert raw.replace(b"\r\n", b"") .count(b"\n") == 0, "a lone LF was introduced"
    # three added lines: the opening fence, the `supersedes:` line, the closing fence
    assert raw.count(b"\r\n") == original.count(b"\r\n") + 3
    assert raw.endswith(original), "the CRLF prose must be untouched"


def test_the_staged_file_takes_the_originals_mode_before_the_swap(
    tmp_path: Path, monkeypatch
) -> None:
    """`mkstemp` creates 0600. Without `copymode` the swap silently tightens the user's memo.

    This asserts the MECHANISM rather than the outcome, because the outcome is not observable on
    every platform this is developed on: Windows encodes only the read-only bit in `st_mode`, so a
    memo at 0640 and `mkstemp`'s 0600 both read back as 0o666 and an outcome comparison there
    compares 0o666 to itself — a guard that cannot fail. Ordering is part of the assertion:
    copying the mode AFTER the rename would apply it to a path that no longer exists.
    """
    calls: list[str] = []
    real_copymode, real_replace = shutil.copymode, os.replace

    def spy_copymode(src: object, dst: object, **k: object) -> None:
        calls.append(f"copymode from {Path(os.fspath(src)).name}")
        real_copymode(src, dst, **k)  # type: ignore[arg-type]

    def spy_replace(src: object, dst: object, *a: object, **k: object) -> None:
        calls.append(f"replace onto {Path(os.fspath(dst)).name}")
        real_replace(src, dst, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(shutil, "copymode", spy_copymode)
    monkeypatch.setattr(os, "replace", spy_replace)
    _corpus(tmp_path)

    apply_rewrite(tmp_path, _fact(), apply=True)

    assert calls == [
        "copymode from new_decision_2026-06-01.md",
        "replace onto new_decision_2026-06-01.md",
    ]


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows st_mode carries only the read-only bit, so 0640 and mkstemp's 0600 both read "
    "back as 0o666 and this comparison cannot fail; the mechanism is covered portably above",
)
def test_the_file_permission_bits_survive_the_atomic_swap(tmp_path: Path) -> None:
    """The outcome the mechanism above exists to produce, on a platform that can observe it."""
    _corpus(tmp_path)
    target = tmp_path / "new_decision_2026-06-01.md"
    os.chmod(target, 0o640)
    before = stat.S_IMODE(target.stat().st_mode)

    apply_rewrite(tmp_path, _fact(), apply=True)

    assert stat.S_IMODE(target.stat().st_mode) == before


def test_the_memo_survives_an_injected_replace_failure(tmp_path: Path, monkeypatch) -> None:
    """A crash / disk-full mid-write must leave the original intact and no temp litter behind."""
    _corpus(tmp_path)
    target = tmp_path / "new_decision_2026-06-01.md"
    real_replace = os.replace

    def failing_replace(src: object, dst: object, *a: object, **k: object) -> None:
        if os.fspath(dst).replace("\\", "/").endswith("/new_decision_2026-06-01.md"):
            raise OSError(28, "No space left on device")
        real_replace(src, dst, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError):
        apply_rewrite(tmp_path, _fact(), apply=True)

    assert target.read_bytes() == _NEW
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


# --- the review gate ----------------------------------------------------------------------------


def test_dry_run_is_the_default_and_writes_nothing(tmp_path: Path) -> None:
    """A tool that rewrites your memory the first time you try it has earned distrust."""
    _corpus(tmp_path)
    before = (tmp_path / "new_decision_2026-06-01.md").read_bytes()

    result = apply_rewrite(tmp_path, _fact())

    assert result.written is False
    assert result.plan is not None, "a dry run still says exactly what it would do"
    assert (tmp_path / "new_decision_2026-06-01.md").read_bytes() == before


def test_a_fact_without_a_reviewer_id_is_refused_before_any_write(tmp_path: Path) -> None:
    _corpus(tmp_path)
    before = (tmp_path / "new_decision_2026-06-01.md").read_bytes()
    unreviewed = PromotedFact(**{**vars(_fact()), "reviewer_id": ""})
    assert metadata_is_trusted(unreviewed) is False

    with pytest.raises(RewriteRefused, match="reviewer_id"):
        apply_rewrite(tmp_path, unreviewed, apply=True)

    assert (tmp_path / "new_decision_2026-06-01.md").read_bytes() == before


def test_a_fact_without_an_audit_note_is_refused_before_any_write(tmp_path: Path) -> None:
    _corpus(tmp_path)
    before = (tmp_path / "new_decision_2026-06-01.md").read_bytes()
    unnoted = PromotedFact(**{**vars(_fact()), "audit_note": "   "})
    assert metadata_is_trusted(unnoted) is False

    with pytest.raises(RewriteRefused, match="audit_note"):
        apply_rewrite(tmp_path, unnoted, apply=True)

    assert (tmp_path / "new_decision_2026-06-01.md").read_bytes() == before


def test_apply_rewrite_is_typed_to_take_a_promoted_fact_not_a_proposal() -> None:
    """The type system, not a convention, is what makes an unreviewed write unexpressible.

    A reviewer reading `apply_rewrite(root, proposal)` cannot tell whether the argument passed
    review. Annotating the parameter `PromotedFact` — a type only `promote_accepted_proposal` can
    construct legitimately — moves that from something you have to remember to something mypy
    refuses. The runtime check backs it up for callers that are not type-checked.
    """
    annotation = inspect.signature(apply_rewrite).parameters["fact"].annotation
    assert annotation in (PromotedFact, "PromotedFact")

    with pytest.raises(RewriteRefused, match="PromotedFact"):
        apply_rewrite(Path("."), _proposal())  # type: ignore[arg-type]


def test_an_existing_supersedes_is_never_silently_overwritten(tmp_path: Path) -> None:
    """Refusing is not enough: the refusal has to be reported, or it is indistinguishable from
    a successful write to anything reading the return value."""
    _memo(tmp_path, "old_decision_2026-01-01.md", _OLD)
    declared = b"---\nsupersedes: something_else_2025-12-01.md\n---\n" + _NEW
    _memo(tmp_path, "new_decision_2026-06-01.md", declared)

    result = apply_rewrite(tmp_path, _fact(), apply=True)

    assert result.written is False
    assert result.refusal is not None and "already declares" in result.refusal
    assert (tmp_path / "new_decision_2026-06-01.md").read_bytes() == declared


# --- what a memo's own bytes are allowed to be ----------------------------------------------------


def test_a_value_spanning_lines_cannot_inject_a_second_frontmatter_key(tmp_path: Path) -> None:
    """The one that makes "no new frontmatter keys" a property rather than a wish.

    `value` is written verbatim, and both `subject_id` and `object_id` originate outside this
    package — `_providers.py` builds them with a bare `str()` over a provider's raw JSON, and the
    deterministic rules take them from a wikilink parsed out of prose. A newline in one of them
    writes a SECOND `key: value` line into the block. `valid_until: 1999-01-01` is the version of
    that which matters: it does not corrupt the file, it silently expires a live memo, and the
    trust layer then does exactly what it was told.
    """
    _corpus(tmp_path)
    before = (tmp_path / "new_decision_2026-06-01.md").read_bytes()
    smuggled = _fact(subject_id="old_decision_2026-01-01.md\nvalid_until: 1999-01-01")

    with pytest.raises(RewriteRefused, match="single line"):
        apply_rewrite(tmp_path, smuggled, apply=True)

    assert (tmp_path / "new_decision_2026-06-01.md").read_bytes() == before


def test_an_empty_supersedes_is_a_declaration_and_not_an_invitation(tmp_path: Path) -> None:
    """`supersedes:` with nothing after it is a human writing "this supersedes nothing".

    Testing truthiness rather than presence reads that as absence and appends a second
    `supersedes:` line — two copies of a single-valued key in one block, which is not the
    overwrite the refusal was written to prevent but something worse.
    """
    _memo(tmp_path, "old_decision_2026-01-01.md", _OLD)
    declared = b"---\nsupersedes:\ntitle: x\n---\n" + _NEW
    _memo(tmp_path, "new_decision_2026-06-01.md", declared)

    result = apply_rewrite(tmp_path, _fact(), apply=True)

    assert result.written is False
    assert result.refusal is not None and "already declares" in result.refusal
    assert (tmp_path / "new_decision_2026-06-01.md").read_bytes() == declared


def test_a_cr_terminated_memo_is_not_spliced_where_the_parser_cannot_look(tmp_path: Path) -> None:
    """The byte writer's idea of a line must be `parse_frontmatter`'s idea of a line.

    `parse_frontmatter` splits on `"\\n"` alone, so a CR-terminated file has no frontmatter as far
    as recall is concerned. `bytes.splitlines` also splits on a lone CR, and a writer using it
    finds an opening `---`, splices the key into a block the parser cannot see, and — because the
    parser still reports nothing declared — does it again on every subsequent run.
    """
    _memo(tmp_path, "old_decision_2026-01-01.md", _OLD)
    cr_only = b"---\rtitle: x\r---\rbody\r"
    _memo(tmp_path, "new_decision_2026-06-01.md", cr_only)

    apply_rewrite(tmp_path, _fact(), apply=True)
    once = (tmp_path / "new_decision_2026-06-01.md").read_bytes()
    apply_rewrite(tmp_path, _fact(), apply=True)
    twice = (tmp_path / "new_decision_2026-06-01.md").read_bytes()

    assert once.endswith(cr_only), "the CR-terminated prose must survive whole"
    meta, _ = parse_frontmatter(once.decode("utf-8"))
    assert meta["supersedes"] == "old_decision_2026-01-01.md", (
        "the key was written somewhere the parser cannot read it"
    )
    assert twice == once, "a second run appended a duplicate, so the first was invisible"


def test_a_memo_that_is_not_utf8_text_is_refused_rather_than_appended_to(tmp_path: Path) -> None:
    """A UTF-16 memo is what Windows Notepad's "Unicode" option produces.

    The derived path never decodes, so it would append UTF-8 bytes to a UTF-16 file and turn its
    tail into mojibake; the frontmatter path would raise a bare `UnicodeDecodeError`, which is a
    `ValueError` but not a `RewriteRefused`, so a batch caller catching the documented type aborts
    the whole run on one odd memo.
    """
    _memo(tmp_path, "old_decision_2026-01-01.md", _OLD)
    utf16 = "# new decision\n\nbody\n".encode("utf-16")
    _memo(tmp_path, "new_decision_2026-06-01.md", utf16)

    with pytest.raises(RewriteRefused, match="UTF-8"):
        apply_rewrite(tmp_path, _fact(), apply=True)
    with pytest.raises(RewriteRefused, match="UTF-8"):
        apply_rewrite(
            tmp_path,
            _fact(
                relation="contradicts",
                subject_id="new_decision_2026-06-01.md",
                object_id="old_decision_2026-01-01.md",
            ),
            apply=True,
        )

    assert (tmp_path / "new_decision_2026-06-01.md").read_bytes() == utf16


def test_a_bom_less_utf16_memo_is_refused_even_though_it_decodes(tmp_path: Path) -> None:
    """The case a decode check alone cannot catch, and the reason NUL is checked separately.

    UTF-16LE ASCII is every character followed by a NUL, and NUL is a perfectly valid UTF-8
    codepoint — so `raw.decode("utf-8")` succeeds and reports a string full of them. Without a
    separate check the derived path appends UTF-8 bytes to it and the tail becomes mojibake.
    """
    _memo(tmp_path, "old_decision_2026-01-01.md", _OLD)
    utf16le = "# new decision\n\nbody\n".encode("utf-16-le")
    assert utf16le.decode("utf-8"), "this fixture only bites if it DOES decode as UTF-8"
    _memo(tmp_path, "new_decision_2026-06-01.md", utf16le)

    with pytest.raises(RewriteRefused, match="NUL"):
        apply_rewrite(
            tmp_path,
            _fact(
                relation="contradicts",
                subject_id="new_decision_2026-06-01.md",
                object_id="old_decision_2026-01-01.md",
            ),
            apply=True,
        )

    assert (tmp_path / "new_decision_2026-06-01.md").read_bytes() == utf16le


def test_a_half_open_derived_block_is_refused_not_appended_past(tmp_path: Path) -> None:
    """One block per file, or the machine's annotations can never be stripped back off again.

    A user who deleted the closing marker has left a region this module cannot reason about.
    Appending a second opening marker past it produces exactly the ambiguity the single-block
    rule exists to prevent, and duplicates the entry the dedup was supposed to catch.
    """
    _corpus(tmp_path)
    half_open = _NEW + b"\n" + DERIVED_OPEN.encode("utf-8") + b"\ncontradicts: old.md\n"
    _memo(tmp_path, "new_decision_2026-06-01.md", half_open)
    fact = _fact(
        relation="contradicts",
        subject_id="new_decision_2026-06-01.md",
        object_id="old_decision_2026-01-01.md",
    )

    with pytest.raises(RewriteRefused, match="unclosed"):
        apply_rewrite(tmp_path, fact, apply=True)

    assert (tmp_path / "new_decision_2026-06-01.md").read_bytes() == half_open


def test_the_staging_descriptor_is_closed_when_the_stage_itself_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """`mkstemp` hands back a raw fd, and `os.fdopen` only takes ownership if it succeeds.

    The condition that makes `fdopen` fail is running out of descriptors, which is the exact
    condition a descriptor leak compounds. On Windows the leaked handle then makes the cleanup
    `unlink` fail too, so the temp file is left sitting beside the user's memo.
    """
    _corpus(tmp_path)
    real_fdopen = os.fdopen
    captured: list[int] = []

    def failing_fdopen(fd: int, *a: object, **k: object) -> object:
        captured.append(fd)
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(os, "fdopen", failing_fdopen)
    with pytest.raises(OSError):
        apply_rewrite(tmp_path, _fact(), apply=True)
    monkeypatch.setattr(os, "fdopen", real_fdopen)

    assert captured, "the staging descriptor was never opened"
    with pytest.raises(OSError):
        os.fstat(captured[0])  # closed: fstat on a live fd would succeed
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []
    assert (tmp_path / "new_decision_2026-06-01.md").read_bytes() == _NEW


def test_the_parent_directory_is_flushed_after_the_swap(tmp_path: Path, monkeypatch) -> None:
    """`fsync` on the file puts its bytes on the platter; only `fsync` on the directory puts the
    RENAME there. Without it the memo can revert to its pre-edit content after a power loss while
    the caller has already been told the edit was applied.

    Asserted as "attempted", because a directory cannot be opened for fsync on Windows — the call
    is made and the OSError swallowed there, and this still goes red if the call is removed.
    """
    _corpus(tmp_path)
    opened: list[str] = []
    real_open = os.open

    def spy_open(path: object, flags: int, *a: object, **k: object) -> int:
        if Path(os.fspath(path)).is_dir():
            opened.append(Path(os.fspath(path)).name)
        return real_open(path, flags, *a, **k)

    monkeypatch.setattr(os, "open", spy_open)

    apply_rewrite(tmp_path, _fact(), apply=True)

    assert opened == [tmp_path.name]


# --- the rejection ledger -----------------------------------------------------------------------


def test_a_rejected_claim_does_not_reappear(tmp_path: Path) -> None:
    """A proposal tool whose output must itself be filtered has not saved anyone any work."""
    _corpus(tmp_path)
    fact = _fact()
    with RejectionLedger(tmp_path / "ledger.sqlite3") as ledger:
        ledger.reject(
            claim_key(fact.relation, fact.subject_id, fact.object_id),
            reviewer_id="reviewer@example.com",
            reason="augments rather than supersedes; asked the author",
            rejected_at=_WHEN,
        )
        result = apply_rewrite(tmp_path, fact, ledger=ledger, apply=True)

    assert result.written is False
    assert result.refusal is not None and "rejected" in result.refusal
    assert (tmp_path / "new_decision_2026-06-01.md").read_bytes() == _NEW


def test_the_ledger_survives_the_process_that_wrote_it(tmp_path: Path) -> None:
    """Durable, not in-memory: the rejection has to outlive the run that recorded it."""
    key = claim_key("supersedes", "old_decision_2026-01-01.md", "new_decision_2026-06-01.md")
    with RejectionLedger(tmp_path / "ledger.sqlite3") as ledger:
        ledger.reject(key, reviewer_id="r", reason="no", rejected_at=_WHEN)

    with RejectionLedger(tmp_path / "ledger.sqlite3") as reopened:
        assert reopened.is_rejected(key) is True


def test_a_rejection_without_a_named_human_and_a_reason_is_refused(tmp_path: Path) -> None:
    """The same rule as promotion: an unattributed entry silences a proposal forever and nobody
    can later find out who decided that, or why."""
    with RejectionLedger(tmp_path / "ledger.sqlite3") as ledger:
        with pytest.raises(RewriteRefused, match="reviewer_id"):
            ledger.reject("claim_x", reviewer_id="  ", reason="no", rejected_at=_WHEN)
        with pytest.raises(RewriteRefused, match="reason"):
            ledger.reject("claim_x", reviewer_id="r", reason="", rejected_at=_WHEN)
        assert ledger.is_rejected("claim_x") is False


def test_the_ledger_lives_beside_the_corpus_not_inside_it(tmp_path: Path) -> None:
    """A growing binary under the corpus glob would eventually be indexed as a memo."""
    path = default_ledger_path(tmp_path)
    assert path.parent.name.startswith(".")
    assert path.suffix != ".md"


def test_a_ledger_that_cannot_open_reports_itself_and_leaks_no_connection(tmp_path: Path) -> None:
    """A constructor that raises after connecting leaves a handle no `__exit__` can ever close.

    `with RejectionLedger(path) as ledger:` never reaches `__enter__` if `__init__` raises, so the
    connection opened one line earlier survives on whatever frame the traceback holds. The failure
    is also translated: `sqlite3.DatabaseError` escaping `apply_rewrite` is a type its docstring
    does not promise and no caller of this module handles.
    """
    import sqlite3

    not_a_database = tmp_path / "ledger.sqlite3"
    not_a_database.write_bytes(b"this is a memo, not a database\n" * 64)
    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def spy_connect(*a: object, **k: object) -> sqlite3.Connection:
        conn = real_connect(*a, **k)  # type: ignore[arg-type]
        opened.append(conn)
        return conn

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sqlite3, "connect", spy_connect)
        with pytest.raises(RewriteRefused, match="ledger"):
            RejectionLedger(not_a_database)

    assert opened, "no connection was ever opened, so this proves nothing about closing one"
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")


def test_the_ledger_key_is_generation_independent(tmp_path: Path) -> None:
    """Keying rejections on `proposal.id` forgets them at the next re-index.

    `_proposal_id` hashes `generation_id` and `pipeline_id` into the identity, by design — the
    schema has been generation scoped since migration 0008. A claim is not: the same prose stating
    the same relation between the same two memos is the same human decision whatever generation
    observed it. So the ledger keys on the claim, and this test is what stops someone "fixing" it
    to use the id that is right there on the object.
    """
    first = _fact(generation_id="gen-a")
    second = _fact(generation_id="gen-b")
    assert first.proposal_id != second.proposal_id

    assert claim_key(first.relation, first.subject_id, first.object_id) == claim_key(
        second.relation, second.subject_id, second.object_id
    )

    _corpus(tmp_path)
    with RejectionLedger(tmp_path / "ledger.sqlite3") as ledger:
        ledger.reject(
            claim_key(first.relation, first.subject_id, first.object_id),
            reviewer_id="r",
            reason="reviewed and declined",
            rejected_at=_WHEN,
        )
        result = apply_rewrite(tmp_path, second, ledger=ledger, apply=True)

    assert result.written is False, "a re-index resurrected a rejected claim"


def test_a_different_claim_is_not_swept_up_by_the_rejection(tmp_path: Path) -> None:
    """The ledger must not be a blanket mute: only the rejected claim stays rejected."""
    _corpus(tmp_path)
    _memo(tmp_path, "third_decision_2026-07-01.md", b"# third\n\nA separate decision.\n")
    with RejectionLedger(tmp_path / "ledger.sqlite3") as ledger:
        ledger.reject(
            claim_key("supersedes", "third_decision_2026-07-01.md", "new_decision_2026-06-01.md"),
            reviewer_id="r",
            reason="wrong pair",
            rejected_at=_WHEN,
        )
        result = apply_rewrite(tmp_path, _fact(), ledger=ledger, apply=True)

    assert result.written is True


# --- routing: no new frontmatter keys -------------------------------------------------------------


def test_no_derived_key_is_ever_written_into_frontmatter(tmp_path: Path) -> None:
    """Three keys are recognised, and `recall/frontmatter.py` is the one place that says which.

    Deriving the frontmatter set FROM `VALIDITY_KEYS` rather than restating it is what makes
    "no new frontmatter keys" a property instead of a promise: adding `contradicts` to the derived
    set cannot leak it into the block the trust layer reads.
    """
    assert set(DERIVED_KEYS).isdisjoint(VALIDITY_KEYS)
    for key in VALIDITY_KEYS:
        assert destination(key) == "frontmatter"
    for key in DERIVED_KEYS:
        assert destination(key) == "derived"
    with pytest.raises(RewriteRefused, match="unknown_key"):
        destination("superseded_by")


def test_contradicts_lands_in_the_derived_block_and_not_in_frontmatter(tmp_path: Path) -> None:
    _corpus(tmp_path)
    fact = _fact(
        relation="contradicts",
        subject_id="new_decision_2026-06-01.md",
        object_id="old_decision_2026-01-01.md",
    )

    apply_rewrite(tmp_path, fact, apply=True)

    raw = (tmp_path / "new_decision_2026-06-01.md").read_bytes()
    meta, _ = parse_frontmatter(raw.decode("utf-8"))
    assert meta == {}, "a derived relation must not create a frontmatter block"
    text = raw.decode("utf-8")
    assert DERIVED_OPEN in text and DERIVED_CLOSE in text
    assert "contradicts: old_decision_2026-01-01.md" in text
    assert raw.startswith(_NEW), "the derived block is appended; prose is not rewritten"


def test_a_second_derived_relation_reuses_the_one_block(tmp_path: Path) -> None:
    """Two blocks would make the file's derived state ambiguous and unremovable."""
    _corpus(tmp_path)
    apply_rewrite(
        tmp_path,
        _fact(
            relation="contradicts",
            subject_id="new_decision_2026-06-01.md",
            object_id="old_decision_2026-01-01.md",
        ),
        apply=True,
    )
    apply_rewrite(
        tmp_path,
        _fact(
            relation="same_entity",
            subject_id="new_decision_2026-06-01.md",
            object_id="old_decision_2026-01-01.md",
        ),
        apply=True,
    )

    text = (tmp_path / "new_decision_2026-06-01.md").read_bytes().decode("utf-8")
    assert text.count(DERIVED_OPEN) == 1
    assert text.count(DERIVED_CLOSE) == 1
    assert "contradicts: old_decision_2026-01-01.md" in text
    assert "same_entity: old_decision_2026-01-01.md" in text


# --- the fixed point ------------------------------------------------------------------------------


def test_applying_a_pass_reaches_a_fixed_point_without_silencing_the_rest(tmp_path: Path) -> None:
    """Extract, apply, re-extract: the edited memo drops out, the untouched one does not.

    The order of assertions is the point. A fixed-point test whose first pass is empty passes
    trivially, and an empty first pass is exactly the state `recall/fix.py` reports on the real
    792-memo corpus — so non-emptiness is asserted BEFORE anything is applied. The second half is
    the other failure mode: a writer that reached the fixed point by suppressing everything would
    satisfy "nothing new" and be useless, so an untouched pair must still produce output.
    """
    from recall.fix import propose_fixes

    _corpus(tmp_path)
    _memo(tmp_path, "left_decision_2026-02-02.md", b"# left\n\nAn earlier call.\n")
    _memo(
        tmp_path,
        "right_decision_2026-08-08.md",
        b"# right\n\nThis replaces [[left_decision_2026-02-02]].\n",
    )

    first, _ = propose_fixes(tmp_path)
    edited = {p.edit_file for p in first}
    assert edited == {"new_decision_2026-06-01.md", "right_decision_2026-08-08.md"}, (
        "the first pass must be non-empty, or the fixed point below is vacuous"
    )

    result = apply_rewrite(tmp_path, _fact(), apply=True)
    assert result.written is True

    second, unfixable = propose_fixes(tmp_path)

    assert "new_decision_2026-06-01.md" not in {p.edit_file for p in second}, (
        "the applied edge came back as a fresh proposal"
    )
    assert any("already declares" in u.reason for u in unfixable), (
        "re-extraction must report the declared edge, not silently drop it"
    )
    assert "right_decision_2026-08-08.md" in {p.edit_file for p in second}, (
        "the untouched pair stopped producing output — the fixed point is a mute, not a fix"
    )
