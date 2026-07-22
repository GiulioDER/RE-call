"""`recall lint --fix`: propose the frontmatter edge a memo already states in prose.

The dangerous part is DIRECTION. The schema has no `superseded_by`, so "A is superseded by B"
must be written as `supersedes: A` on **B**. Getting it backwards would declare the live memo
stale and demote it beneath the one it replaced — the exact failure the trust layer exists to
prevent, caused by the tool meant to fix it. Hence a pure, string-only test for the rule.

The second rule is refusal: a fix is proposed only when the named target resolves to exactly one
file. A bare "DEPRECATED" with no target is reported, never guessed.
"""
from __future__ import annotations

from recall.fix import apply_proposal, extract_edges, propose_fixes
from recall.frontmatter import parse_frontmatter


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
