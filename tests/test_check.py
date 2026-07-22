"""Write-time gate: ask for the edge while the author still knows the answer.

`--fix` measured 60 prose closure markers in a real corpus and could safely declare ZERO after
the fact. This runs at the other end, and the disposition INVERTS: `--fix` refuses everything it
cannot prove because it writes unattended; this surfaces every candidate because a human is
right there to pick. A false candidate costs a glance; a missing one costs the edge.
"""
from __future__ import annotations

from recall.check import check_file, corpus_names, format_prompt


def _w(d, name, text):
    (d / name).write_text(text, encoding="utf-8")
    return d / name


def test_a_closure_marker_without_an_edge_needs_attention(tmp_path):
    f = _w(tmp_path, "new.md", "# new\n\nThis approach is DEPRECATED now.")
    assert check_file(f).needs_attention


def test_an_already_declared_edge_is_silent(tmp_path):
    f = _w(tmp_path, "new.md",
           "---\nsupersedes: old_plan_2026-01-01\n---\n# new\n\nDEPRECATED predecessor.")
    assert not check_file(f).needs_attention


def test_a_validity_window_also_counts_as_declared(tmp_path):
    """`valid_until:` is the other way to make a closure act on retrieval."""
    f = _w(tmp_path, "new.md", "---\nvalid_until: 2030-01-01\n---\n# new\n\nDEPRECATED soon.")
    assert not check_file(f).needs_attention


def test_a_memo_with_no_closure_marker_is_silent(tmp_path):
    f = _w(tmp_path, "new.md", "# new\n\nAn ordinary memo about an ordinary decision.")
    assert not check_file(f).needs_attention


def test_candidates_are_surfaced_liberally(tmp_path):
    """No marker-proximity requirement, unlike --fix: the author picks, so a wide net wins."""
    _w(tmp_path, "old_plan_2026-01-01.md", "# old\n\nbody")
    _w(tmp_path, "other_plan_2026-02-02.md", "# other\n\nbody")
    f = _w(tmp_path, "new.md",
           "# new\n\nThis is DEPRECATED.\n\nSee [[old_plan_2026-01-01]] and "
           "[[other_plan_2026-02-02]] for background.")
    r = check_file(f, corpus_names(tmp_path))
    assert r.candidates == ["old_plan_2026-01-01", "other_plan_2026-02-02"]


def test_candidates_are_filtered_to_real_documents(tmp_path):
    """A name that is not in the corpus is not a choice the author can make."""
    _w(tmp_path, "old_plan_2026-01-01.md", "# old\n\nbody")
    f = _w(tmp_path, "new.md",
           "# new\n\nDEPRECATED. See [[old_plan_2026-01-01]] and [[imaginary_memo_2026-09-09]].")
    r = check_file(f, corpus_names(tmp_path))
    assert r.candidates == ["old_plan_2026-01-01"]


def test_a_memo_never_offers_itself_as_a_candidate(tmp_path):
    f = _w(tmp_path, "self_ref_2026-01-01.md",
           "# self\n\nDEPRECATED. See [[self_ref_2026-01-01]].")
    assert check_file(f, corpus_names(tmp_path)).candidates == []


def test_index_files_are_not_prompted(tmp_path):
    """An index enumerates closures; it does not make one — the same rule --fix learned."""
    f = _w(tmp_path, "closed_hypotheses_index.md", "# closed\n\n- something DEPRECATED")
    assert not check_file(f).needs_attention


def test_the_prompt_offers_the_exact_line_to_paste(tmp_path):
    _w(tmp_path, "old_plan_2026-01-01.md", "# old\n\nbody")
    f = _w(tmp_path, "new.md", "# new\n\nDEPRECATED, see [[old_plan_2026-01-01]].")
    text = format_prompt(check_file(f, corpus_names(tmp_path)))
    assert "supersedes: old_plan_2026-01-01" in text
    assert "augmenting memo is not a successor" in text, "must not push the author to over-declare"


def test_the_prompt_still_helps_when_no_candidate_was_found(tmp_path):
    f = _w(tmp_path, "new.md", "# new\n\nThis approach is DEPRECATED.")
    text = format_prompt(check_file(f))
    assert "supersedes: <name>" in text and "valid_until" in text


def test_an_unreadable_byte_does_not_crash_the_gate(tmp_path):
    """A commit hook that dies on a stray byte blocks the commit for the wrong reason."""
    (tmp_path / "new.md").write_bytes(b"# new\n\nDEPRECATED \xff\xfe thing")
    assert check_file(tmp_path / "new.md").needs_attention
