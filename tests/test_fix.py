"""`recall lint --fix`: propose the frontmatter edge a memo already states in prose.

The dangerous part is DIRECTION. The schema has no `superseded_by`, so "A is superseded by B"
must be written as `supersedes: A` on **B**. Getting it backwards would declare the live memo
stale and demote it beneath the one it replaced — the exact failure the trust layer exists to
prevent, caused by the tool meant to fix it. Hence a pure, string-only test for the rule.

The second rule is refusal: a fix is proposed only when the named target resolves to exactly one
file. A bare "DEPRECATED" with no target is reported, never guessed.
"""
from __future__ import annotations

import os

import pytest

from recall.fix import (
    Proposal,
    UnreadableMemo,
    apply_proposal,
    extract_edges,
    propose_fixes,
)
from recall.frontmatter import parse_frontmatter, supersedes_key


def _write(d, name, text):
    (d / name).write_text(text, encoding="utf-8")


# --- direction, on strings alone --------------------------------------------------------------


def test_active_voice_means_this_memo_supersedes_the_target():
    active, passive = extract_edges("This decision supersedes [[old_plan_2026-01-01]].")
    assert active == ["old_plan_2026-01-01"]
    assert passive == []


def test_passive_voice_means_the_target_supersedes_this_memo():
    active, passive = extract_edges("Superseded by [[new_plan_2026-02-02]] after the review.")
    assert passive == ["new_plan_2026-02-02"]
    assert active == []


def test_replaced_by_is_passive_and_replaces_is_active():
    assert extract_edges("replaced by [[b_memo_x]]")[1] == ["b_memo_x"]
    assert extract_edges("replaces [[a_memo_x]]")[0] == ["a_memo_x"]


def test_reference_forms_are_all_recognised():
    for ref in ("[[old_plan_2026-01-01]]", "[old_plan_2026-01-01]",
                "`old_plan_2026-01-01`", "old_plan_2026-01-01.md", "old_plan_2026-01-01"):
        active, _ = extract_edges(f"This supersedes {ref}.")
        assert active, f"did not recognise {ref!r}"


def test_a_marker_with_no_target_yields_nothing():
    """A bare closure marker is exactly the case that must NOT be guessed at."""
    assert extract_edges("This approach is DEPRECATED and no longer used.") == ([], [])


# --- proposals against a corpus ---------------------------------------------------------------


def test_passive_marker_writes_the_edge_on_the_other_file(tmp_path):
    _write(tmp_path, "old.md", "# old\n\nSuperseded by [[new_decision_2026]].")
    _write(tmp_path, "new_decision_2026.md", "# new\n\nthe current decision")
    proposals, _ = propose_fixes(tmp_path)
    assert len(proposals) == 1
    p = proposals[0]
    assert p.edit_file == "new_decision_2026.md", "edge must go on the SUCCESSOR"
    assert p.target == "old.md"


def test_active_marker_writes_the_edge_on_this_file(tmp_path):
    _write(tmp_path, "old_thing_2026.md", "# old\n\nthe original")
    _write(tmp_path, "new.md", "# new\n\nThis supersedes [[old_thing_2026]].")
    proposals, _ = propose_fixes(tmp_path)
    assert len(proposals) == 1
    assert proposals[0].edit_file == "new.md"
    assert proposals[0].target == "old_thing_2026"


def test_an_unresolvable_target_is_reported_not_guessed(tmp_path):
    _write(tmp_path, "a.md", "# a\n\nSuperseded by [[something_never_written]].")
    proposals, unfixable = propose_fixes(tmp_path)
    assert proposals == []
    assert unfixable and "not a file in the corpus" in unfixable[0].reason


def test_an_ambiguous_target_is_reported_not_guessed(tmp_path):
    for sub in ("x", "y"):
        (tmp_path / sub).mkdir()
        _write(tmp_path / sub, "dup_memo_2026.md", "# dup\n\nbody")
    _write(tmp_path, "new.md", "# new\n\nThis supersedes [[dup_memo_2026]].")
    proposals, unfixable = propose_fixes(tmp_path)
    assert proposals == []
    assert unfixable and "matches 2 files" in unfixable[0].reason


def test_an_existing_edge_is_never_overwritten(tmp_path):
    _write(tmp_path, "old_thing_2026.md", "# old\n\nbody")
    _write(tmp_path, "other_thing_2026.md", "# other\n\nbody")
    _write(tmp_path, "new.md",
           "---\nsupersedes: other_thing_2026.md\n---\n# new\n\nThis supersedes [[old_thing_2026]].")
    proposals, unfixable = propose_fixes(tmp_path)
    assert proposals == []
    assert unfixable and "refusing to overwrite" in unfixable[0].reason


def test_a_memo_that_already_declares_the_edge_produces_no_proposal(tmp_path):
    _write(tmp_path, "old_thing_2026.md", "# old\n\nbody")
    _write(tmp_path, "new.md",
           "---\nsupersedes: old_thing_2026\n---\n# new\n\nThis supersedes [[old_thing_2026]].")
    proposals, _ = propose_fixes(tmp_path)
    assert proposals == []


# --- writing ----------------------------------------------------------------------------------


def test_apply_adds_frontmatter_to_a_file_without_any(tmp_path):
    _write(tmp_path, "old_thing_2026.md", "# old\n\nbody")
    _write(tmp_path, "new.md", "# new\n\nThis supersedes [[old_thing_2026]].")
    proposals, _ = propose_fixes(tmp_path)
    apply_proposal(tmp_path, proposals[0])

    meta, body = parse_frontmatter((tmp_path / "new.md").read_text(encoding="utf-8"))
    assert meta["supersedes"] == "old_thing_2026"
    assert body.startswith("# new"), "the body must survive untouched"


def test_apply_preserves_existing_frontmatter_keys_and_body(tmp_path):
    _write(tmp_path, "old_thing_2026.md", "# old\n\nbody")
    _write(tmp_path, "new.md",
           "---\nvalid_until: 2030-01-01\n---\n# new\n\nThis supersedes [[old_thing_2026]].\n\ntail")
    proposals, _ = propose_fixes(tmp_path)
    apply_proposal(tmp_path, proposals[0])

    text = (tmp_path / "new.md").read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    assert meta["valid_until"] == "2030-01-01"
    assert meta["supersedes"] == "old_thing_2026"
    assert body.rstrip().endswith("tail")


def test_applying_makes_the_edge_real_end_to_end(tmp_path):
    """The point of the feature: after the fix, the corpus lints clean and the edge exists."""
    from recall.lint import lint_corpus
    from recall.store import resolve_supersession

    _write(tmp_path, "old.md", "# old\n\nSuperseded by [[new_decision_2026]].")
    _write(tmp_path, "new_decision_2026.md", "# new\n\nthe current decision")
    assert any(i.code == "closure-marker-unlinked" for i in lint_corpus(tmp_path))

    proposals, _ = propose_fixes(tmp_path)
    for p in proposals:
        apply_proposal(tmp_path, p)

    assert not any(i.code == "closure-marker-unlinked" for i in lint_corpus(tmp_path))
    rows = []
    for f in sorted(tmp_path.glob("*.md")):
        meta, _ = parse_frontmatter(f.read_text(encoding="utf-8"))
        rows.append((f.name, meta.get("supersedes")))
    edges, unresolved = resolve_supersession(rows)
    assert edges == {"old.md": "new_decision_2026.md"}
    assert unresolved == frozenset()


# --- rejecting what real memos are actually full of --------------------------------------------


def test_markdown_checkboxes_are_not_document_references():
    """`[x]` and `[ ]` are everywhere in real notes; a single-bracket pattern matched them."""
    assert extract_edges("- [x] superseded by the new plan\n- [ ] replaces nothing")[0] == []
    assert extract_edges("- [x] superseded by the new plan")[1] == []


def test_inline_code_is_not_a_document_reference():
    """Backticks mean code in these memos. One real match captured
    `curate_wallets.wallet_weight = clamp(...)` as a filename."""
    body = "This supersedes `curate_wallets.wallet_weight = clamp(shrunk_EV/REF_EV)` behaviour."
    assert extract_edges(body) == ([], [])


def test_a_long_prose_aside_is_not_a_document_reference():
    """A real bracket match ran 600 characters into the next paragraph."""
    body = "superseded by [a long editorial aside that rambles on " + "and on " * 40 + "]"
    assert extract_edges(body)[1] == []


def test_a_bare_stem_needs_a_year_to_count():
    """Without a date a bare token is indistinguishable from ordinary prose."""
    assert extract_edges("This supersedes the old_rate_policy entirely.")[0] == []
    assert extract_edges("This supersedes old_rate_policy_2026-03-01.")[0] == \
        ["old_rate_policy_2026-03-01"]


def test_an_index_file_never_proposes_an_edge(tmp_path):
    """An index ENUMERATES closed decisions; it does not supersede them.

    On the real corpus, `closed_hypotheses_index.md` listing an archived memo was read as
    "the archive supersedes the index" — syntactically valid, semantically backwards.
    """
    _write(tmp_path, "old_thing_2026-01-01.md", "# old\n\nbody")
    _write(tmp_path, "closed_hypotheses_index.md",
           "# closed\n\n- replaces old_thing_2026-01-01")
    proposals, _ = propose_fixes(tmp_path)
    assert proposals == []


# --- refusals learned from reviewing real proposals --------------------------------------------
#
# Reviewing 4 proposals against the real corpus: 1 correct, 2 partial, 1 flat wrong. Each
# sentence below is quoted verbatim from the memo that produced the bad proposal.


def test_reported_speech_is_refused():
    """The subject of "supersedes" is ANOTHER document, so the sentence reports a relation
    rather than declaring this memo's own.

    Verbatim from project-docs-rag-trust-layer-deployed-2026-07-17.md. Attributing this to the
    narrating memo invented a second, false claimant for an edge that another memo already
    declares correctly — the worst kind of false positive, because it looks authoritative.
    """
    body = ("First annotations: LRP closure memo supersedes "
            "`project_lrp_maker_2026-06-24`; queue-position falsified")
    assert extract_edges(body) == ([], [])


def test_superseding_a_claim_inside_a_document_is_refused():
    """Verbatim from project_gabigol_maker_onchain_proof_btc_pivot_2026-06-09.md.

    It supersedes one CLAIM in the predecessor, not the predecessor. Declaring `supersedes:`
    would demote the whole document and lose everything else it holds.
    """
    body = ('Live HOLD, gate #1387 (06-13). Supersedes the *inferred* "maker" claim in '
            "[[project_gabigol_vs_us_btc_execution_2026-06-08]] with **direct on-chain proof**.")
    assert extract_edges(body) == ([], [])


def test_superseding_the_scope_of_a_document_is_refused():
    """Verbatim from project_vps3_drift_reconcile_5files_2026-06-16.md — the same shape."""
    body = ("md5 census on 2026-06-16 shows most flagged files now MATCH master. "
            "Supersedes the scope in [[project_vps3_manual_drift_live_subsystems_2026-06-15]].")
    assert extract_edges(body) == ([], [])


def test_the_last_surviving_proposal_was_also_wrong():
    """This test previously asserted the OPPOSITE, and that assertion was mine, not the author's.

    Verbatim from project_ci_pipeline_optimization_2026-07-05.md. I judged it the one genuine
    edge of four and wrote a test pinning that it must survive the refusals. Asked directly, the
    author said it **augments** — the hedge in "Supersedes/augments" was the answer all along.
    A test encoding a reviewer's guess is worth less than one question to whoever wrote the memo.

    Final tally on the real corpus: 60 prose closure markers, **zero** safely auto-declarable.
    """
    body = "Supersedes/augments [[feedback_ci_green_constraints_2026-06-22]]."
    assert extract_edges(body) == ([], [])


def test_an_ordinary_subject_is_not_mistaken_for_reported_speech():
    """"This decision supersedes X" is the memo's own claim — the refusal must not overreach."""
    assert extract_edges("This decision supersedes [[old_plan_2026-01-01]].")[0] == \
        ["old_plan_2026-01-01"]


def test_a_hedged_marker_is_refused():
    """Verbatim from project_ci_pipeline_optimization_2026-07-05.md — the last surviving
    proposal, and the author's answer when asked was **augments**.

    The slash was doing real work. An augmenting memo does not replace its predecessor, and
    declaring the edge would demote a memo that is still current. A hedge is the author saying
    they are not sure; resolving it for them is the confident wrong answer this project exists
    to avoid.
    """
    body = "Supersedes/augments [[feedback_ci_green_constraints_2026-06-22]]."
    assert extract_edges(body) == ([], [])


def test_other_hedges_are_refused_too():
    for body in (
        "This partially supersedes [[old_plan_2026-01-01]].",
        "Largely supersedes [[old_plan_2026-01-01]].",
        "Supersedes or augments [[old_plan_2026-01-01]].",
    ):
        assert extract_edges(body) == ([], []), body


def test_an_unhedged_claim_is_still_accepted():
    """The refusals must not swallow a plain, committed statement."""
    assert extract_edges("This decision supersedes [[old_plan_2026-01-01]].")[0] == \
        ["old_plan_2026-01-01"]
    assert extract_edges("Superseded by [[new_plan_2026-02-02]].")[1] == ["new_plan_2026-02-02"]


def test_apply_proposal_preserves_the_memo_when_the_write_fails(tmp_path, monkeypatch):
    """A crash / disk-full mid-write must not corrupt the user's memo (DAT-001).

    ``apply_proposal`` is the one path in the package that rewrites a user's own document in
    place. The pre-fix code used ``Path.write_text``, which opens mode ``'w'`` and truncates the
    file to zero bytes *before* the first byte of new content is written — there is no staging
    step, so an interruption commits corruption and nothing can roll it back. The fix stages the
    new content in a sibling temp file and swaps it in with a single ``os.replace``; if that
    rename fails, the original is left untouched.

    Injecting a failure at ``os.replace`` therefore separates the two: the fixed code raises and
    leaves the memo byte-for-byte intact, while the pre-fix code has no ``os.replace`` in its
    path at all — it writes in place and returns success, so this test's ``pytest.raises`` fails
    with *DID NOT RAISE*. That absence of an atomic stage is exactly the defect.
    """
    memo = tmp_path / "note.md"
    original = "---\ntitle: keep me\n---\n\nprecious body\n"
    memo.write_text(original, encoding="utf-8")
    proposal = Proposal(
        edit_file="note.md",
        target="new_decision_2026",
        evidence_file="other.md",
        evidence="This supersedes [[note]].",
    )

    real_replace = os.replace

    def failing_replace(src, dst, *a, **k):
        if os.fspath(dst).replace("\\", "/").endswith("/note.md"):
            raise OSError(28, "simulated ENOSPC on atomic rename")  # errno 28 == ENOSPC
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError):
        apply_proposal(tmp_path, proposal)

    # The original memo must survive the failed write byte-for-byte, and no temp litter is left.
    assert memo.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [memo], "no partial temp file should be left behind"


# --- bytes, not text --------------------------------------------------------------------------
#
# Every assertion above reads back through `read_text`, whose universal-newline decoding and
# `utf-8` codec hide exactly the two defects below: a memo can lose its BOM and have every line
# ending rewritten and still compare equal. These read `read_bytes`.


def _write_bytes(d, name, raw):
    (d / name).write_bytes(raw)


def test_apply_proposal_preserves_a_utf8_bom(tmp_path):
    """`parse_frontmatter` tolerates a leading BOM precisely because Windows editors add one.

    Reading `utf-8-sig` (which strips it) and writing `utf-8` (which does not put it back) means
    declaring an edge silently re-encodes the user's memo. Nothing warns, and the next time they
    open it their editor sees a different file than the one it wrote.
    """
    _write_bytes(tmp_path, "old_thing_2026.md", b"# old\n\nbody\n")
    original = "\ufeff# new\n\nThis supersedes [[old_thing_2026]].\n".encode("utf-8")
    _write_bytes(tmp_path, "new.md", original)

    proposals, _ = propose_fixes(tmp_path)
    apply_proposal(tmp_path, proposals[0])

    raw = (tmp_path / "new.md").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "the BOM was eaten"
    assert raw.count(b"\xef\xbb\xbf") == 1, "the BOM was duplicated"
    meta, _ = parse_frontmatter(raw.decode("utf-8-sig"))
    assert meta["supersedes"] == "old_thing_2026"


def test_a_created_frontmatter_block_uses_the_memos_own_line_endings(tmp_path):
    """The other branch, and the common one for `lint --fix`: a memo with no block yet.

    The insert-into-an-existing-block path borrows the closing fence's terminator, so it is CRLF
    safe almost by accident. The path that CREATES a block writes three lines of its own, and
    nothing forces those to match the file unless it is asked to — a CRLF memo would open with
    three LF-only lines above prose that is CRLF throughout.
    """
    _write_bytes(tmp_path, "old_thing_2026.md", b"# old\r\n\r\nbody\r\n")
    original = b"# new\r\n\r\nThis supersedes [[old_thing_2026]].\r\n"
    _write_bytes(tmp_path, "new.md", original)

    proposals, _ = propose_fixes(tmp_path)
    apply_proposal(tmp_path, proposals[0])

    raw = (tmp_path / "new.md").read_bytes()
    assert raw.replace(b"\r\n", b"").count(b"\n") == 0, "a lone LF was introduced"
    assert raw.count(b"\r\n") == original.count(b"\r\n") + 3, "the opened block is not CRLF"
    assert raw.endswith(original), "the prose must be untouched"


def test_apply_proposal_preserves_crlf_line_endings(tmp_path):
    """Splitting and rejoining on `"\n"` normalises every line ending in the file.

    The visible damage is one LF-only line inside a CRLF memo, which is invisible in an editor and
    present in every diff; the full damage is that the whole file is rewritten. The count is
    asserted both ways: no lone LF anywhere, and exactly the original CRLFs plus the line added.
    """
    _write_bytes(tmp_path, "old_thing_2026.md", b"# old\r\n\r\nbody\r\n")
    original = b"---\r\nvalid_until: 2030-01-01\r\n---\r\n# new\r\n\r\nThis supersedes [[old_thing_2026]].\r\n"
    _write_bytes(tmp_path, "new.md", original)

    proposals, _ = propose_fixes(tmp_path)
    apply_proposal(tmp_path, proposals[0])

    raw = (tmp_path / "new.md").read_bytes()
    assert raw.replace(b"\r\n", b"").count(b"\n") == 0, "a lone LF was introduced"
    assert raw.count(b"\r\n") == original.count(b"\r\n") + 1
    meta, _ = parse_frontmatter(raw.decode("utf-8"))
    assert meta["valid_until"] == "2030-01-01"
    assert meta["supersedes"] == "old_thing_2026"


def test_apply_proposal_refuses_a_memo_it_cannot_decode(tmp_path):
    """A file `propose_fixes` could not read is exactly the file the overwrite guard cannot see.

    `propose_fixes` skips an undecodable memo, so it never enters `existing` and its authored
    `supersedes:` is invisible to the "refusing to overwrite" refusal — yet it can still be the
    TARGET of a passive-voice marker in another memo, and so still be chosen as `edit_file`.
    Writing then replaces a declared edge, because `parse_frontmatter` is last-wins, and appends
    another line every run since the file still will not decode. The refusal names the file.
    """
    _write_bytes(tmp_path, "evidence_2026.md",
                 b"# evidence\n\nThis is superseded by [[replacement_2026]].\n")
    latin1 = "---\nsupersedes: authored_target_2024\n---\ncaf\u00e9 na\u00efve\n".encode("latin-1")
    _write_bytes(tmp_path, "replacement_2026.md", latin1)

    proposals, _ = propose_fixes(tmp_path)
    assert [p.edit_file for p in proposals] == ["replacement_2026.md"], (
        "the undecodable file must still be the one selected, or this proves nothing"
    )

    with pytest.raises(UnreadableMemo, match="not valid UTF-8"):
        apply_proposal(tmp_path, proposals[0])

    assert (tmp_path / "replacement_2026.md").read_bytes() == latin1


def test_apply_proposal_rechecks_the_target_before_overwriting(tmp_path):
    """The corpus-wide scan's view of what is declared can be stale or incomplete.

    `propose_fixes` builds `existing` from the files it managed to read, at the time it read them.
    The re-check is against the file about to be written, immediately before writing it, which is
    the only view that cannot have gone out of date.
    """
    _write_bytes(tmp_path, "old_thing_2026.md", b"# old\n\nbody\n")
    _write_bytes(tmp_path, "new.md", b"# new\n\nThis supersedes [[old_thing_2026]].\n")
    proposals, _ = propose_fixes(tmp_path)
    assert proposals, "no proposal to apply"

    # something else declared the edge between the scan and the write
    (tmp_path / "new.md").write_bytes(
        b"---\nsupersedes: someone_elses_choice\n---\n# new\n\nThis supersedes [[old_thing_2026]].\n"
    )
    before = (tmp_path / "new.md").read_bytes()

    with pytest.raises(UnreadableMemo, match="already declares"):
        apply_proposal(tmp_path, proposals[0])

    assert (tmp_path / "new.md").read_bytes() == before


def test_a_cr_terminated_memo_keeps_its_authored_frontmatter(tmp_path):
    """Every reader hands `parse_frontmatter` text from `read_text(newline=None)`, which has
    already turned a lone CR into LF — so a CR-terminated memo has frontmatter, and treating the
    file as one long line prepends a second block and orphans the authored `valid_until`."""
    _write_bytes(tmp_path, "old_thing_2026.md", b"# old\r\rbody\r")
    original = b"---\rvalid_until: 2020-01-01\r---\r# new\r\rThis supersedes [[old_thing_2026]].\r"
    _write_bytes(tmp_path, "new.md", original)

    proposals, _ = propose_fixes(tmp_path)
    apply_proposal(tmp_path, proposals[0])

    raw = (tmp_path / "new.md").read_bytes()
    assert raw.count(b"---") == 2, "a second frontmatter block was prepended"
    assert b"supersedes: old_thing_2026\r" in raw, "the inserted line is not CR-terminated"
    meta, _ = parse_frontmatter((tmp_path / "new.md").read_text(encoding="utf-8-sig"))
    assert meta["valid_until"] == "2020-01-01", "the authored validity bound was orphaned"
    assert meta["supersedes"] == "old_thing_2026"


# --- a filename that would inject a second line ------------------------------------------------


def _crafted_name(tmp_path, template):
    """A name carrying U+2028, PROVEN to exist on this filesystem under exactly that name.

    Three things this does that a bare `try: write except OSError: skip` did not. It asserts the
    created entry round-trips, because a mount that MANGLES the character rather than rejecting
    it would otherwise fail the caller with "not a file in the corpus", a reason that has nothing
    to do with the guard. It catches `ValueError` as well, since a non-UTF-8
    `sys.getfilesystemencoding()` raises `UnicodeEncodeError`, which is not an `OSError`. And it
    refuses to skip on POSIX, where the character is always legal, so the CI runner cannot
    quietly lose this coverage — a guard whose only test skips is a guard with no test.

    U+2028 is written `chr(0x2028)` because a literal one does not survive a paste: it degrades
    to a space, and every assertion here would then pass while testing nothing.
    """
    name = template.format(sep=chr(0x2028))
    probe = tmp_path / f"{name}.probe"
    try:
        probe.write_text("x", encoding="utf-8")
    except (OSError, ValueError) as exc:  # pragma: no cover - platform dependent
        if os.name == "nt":
            pytest.skip(f"this filesystem will not create a name containing U+2028: {exc}")
        raise
    assert probe.name in {p.name for p in tmp_path.iterdir()}, (
        "the filesystem mangled the crafted name, so this test would measure the wrong refusal"
    )
    probe.unlink()
    return name


def test_a_target_name_carrying_a_separator_is_reported_not_written(tmp_path):
    """A crafted FILENAME must not become a frontmatter line that splits.

    `propose_fixes` derives the written value from the file's own path, so the name is attacker
    controlled wherever memos are not all hand-authored. U+2028 is legal on NTFS and POSIX, and
    is spelled with `chr()` here because a literal one does not survive every paste.

    The refusal has to happen at PROPOSE time, reported like every other `Unfixable`, rather than
    only as the writer's own `ValueError`: `recall lint --fix` prints the plan before `--apply`
    writes anything, and a refusal the dry run cannot show is one the operator meets halfway
    through a corpus rewrite.
    """
    evil = _crafted_name(tmp_path, "evil{sep}injected_2026")
    _write(tmp_path, f"{evil}.md", "# evil\n\nthe superseded one")
    # ACTIVE voice: the referenced NAME becomes the written value, and here the reference and the
    # file share a name, so the crafted text reaches `insert_frontmatter_line` as the value while
    # `new.md` is the memo edited. The passive direction is covered below and is the worse of the
    # two, so neither stands in for the other.
    _write(tmp_path, "new.md", f"# new\n\nThis supersedes [[{evil}]].")

    proposals, unfixable = propose_fixes(tmp_path)

    assert proposals == [], "a name that would split the line must never reach the writer"
    assert any("line break" in u.reason for u in unfixable), (
        f"the refusal must be reported so the dry run can show it, got {unfixable!r}"
    )


def test_the_passive_direction_blames_the_memo_that_can_actually_be_fixed(tmp_path):
    """The crafted name reaches an INNOCENT memo, and the report must say whose fault it is.

    Passive voice makes the source memo's own path the written value (`value = name`), so the
    injected line lands in a memo whose author wrote nothing unusual. This is the wider blast
    radius of the two directions: in the active case the crafted text is at least named in the
    edited memo's own prose.

    Reporting `writer` here would send the operator to the victim, which is the one file they
    cannot fix — the offending name belongs to the source, and renaming that is the remedy.
    """
    evil = _crafted_name(tmp_path, "evil{sep}src_2026")
    _write(tmp_path, "victim_2026.md", "# victim\n\nbody")
    _write(tmp_path, f"{evil}.md", "# evil\n\nThis is superseded by [[victim_2026]].")

    proposals, unfixable = propose_fixes(tmp_path)

    assert proposals == [], "the innocent memo must not be written into"
    refusals = [u for u in unfixable if "line break" in u.reason]
    assert [u.file for u in refusals] == [f"{evil}.md"], (
        f"the refusal must name the memo whose path carries the separator, got {refusals!r}"
    )
    assert "victim_2026.md" in refusals[0].reason, "and must say where it would have been written"


def test_one_edge_spelled_twice_is_refused_once_and_never_also_proposed(tmp_path):
    """The refusal is judged against every edge the run could declare, not the ones before it.

    `supersedes_key` compares on the stem, so `sub/alpha_2026` and `alpha_2026` are ONE edge. If
    one spelling is clean the edge is declarable and the crafted spelling must not be reported at
    all; reporting it prints SKIP beside the proposal that declares the very same edge. `bodies`
    is iterated in sorted order, so which spelling comes first is stable but arbitrary, and
    deciding inline gets it right only for one of the two orders.

    ⚠️ Asserted as `unfixable == []`, flatly. An earlier version of this test compared the writers
    in `proposals` against the files named in `unfixable`, and that comparison CANNOT FAIL: this
    commit made the refusal name the SOURCE memo while proposals name the WRITER, so in the
    passive direction the two sets are disjoint whatever the code does. Deleting the suppression
    this test exists to protect left it green.
    """
    sub = _crafted_name(tmp_path, "0{sep}dir")
    _write(tmp_path, "target_2026.md", "# t\n\nbody")
    (tmp_path / sub).mkdir()
    _write(tmp_path / sub, "alpha_2026.md", "# a\n\nThis is superseded by [[target_2026]].")
    _write(tmp_path, "alpha_2026.md", "# a\n\nThis is superseded by [[target_2026]].")

    # The premise, asserted rather than assumed: if these two spellings did not collide the
    # fixture would be testing nothing and the test would pass for the wrong reason.
    assert supersedes_key(f"{sub}/alpha_2026.md") == supersedes_key("alpha_2026.md")

    proposals, unfixable = propose_fixes(tmp_path)

    assert [p.edit_file for p in proposals] == ["target_2026.md"], (
        f"the edge is declarable from the clean spelling, got {proposals!r}"
    )
    assert unfixable == [], (
        f"one declarable edge needs no human, so nothing may be reported, got {unfixable!r}"
    )
