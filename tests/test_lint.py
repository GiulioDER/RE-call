"""`recall lint`: completeness checks on the supersession graph, before indexing.

The trust layer's residual failure mode is authorial: a new memo that REPLACES an old one but
never declares `supersedes:` leaves an orphan that looks valid forever. That failure is not
detectable at read-time (both memos look fine in isolation) — but it IS lintable at write-time.
"""
from __future__ import annotations

import pytest

from recall.lint import LintIssue, lint_corpus


def _write(tmp_path, name: str, text: str):
    (tmp_path / name).write_text(text, encoding="utf-8")


def _codes(issues: list[LintIssue]) -> set[str]:
    return {i.code for i in issues}


def test_clean_corpus_has_no_issues(tmp_path):
    _write(tmp_path, "a.md", "some settled decision")
    _write(tmp_path, "b_v1.md", "rate limit is 100 rps")
    _write(tmp_path, "b_v2.md", "---\nsupersedes: b_v1.md\n---\nrate limit is 20 rps")
    assert lint_corpus(tmp_path) == []


def test_dangling_supersedes_is_an_error(tmp_path):
    _write(tmp_path, "new.md", "---\nsupersedes: ghost.md\n---\nthe new decision")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"dangling-supersedes"}
    assert issues[0].level == "error"
    assert "ghost.md" in issues[0].message


def test_self_supersedes_is_an_error(tmp_path):
    _write(tmp_path, "loop.md", "---\nsupersedes: loop.md\n---\ntext")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"self-supersedes"}
    assert issues[0].level == "error"


def test_supersession_cycle_is_an_error(tmp_path):
    _write(tmp_path, "a.md", "---\nsupersedes: b.md\n---\nA")
    _write(tmp_path, "b.md", "---\nsupersedes: a.md\n---\nB")
    issues = lint_corpus(tmp_path)
    assert "supersession-cycle" in _codes(issues)
    assert all(i.level == "error" for i in issues if i.code == "supersession-cycle")


def test_malformed_validity_date_is_an_error(tmp_path):
    _write(tmp_path, "bad.md", "---\nvalid_until: June 2026\n---\ntext")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"invalid-date"}
    assert issues[0].level == "error"


@pytest.mark.parametrize("block", ["valid_until:\n  2020-01-01", 'valid_until: ""', "valid_from:"])
def test_an_empty_validity_value_is_an_error(tmp_path, block):
    """DEF-23 (STAKES-003): a validity key with no value must not lint clean.

    Red proof: against `recall/lint.py` before this fix (no empty-value check after
    `validity_bounds`), the issue set is empty, so the assertion fails; `validity_bounds` reads the
    empty value as no bound and the memo would never expire.
    """
    _write(tmp_path, "empty.md", f"---\n{block}\n---\ntext")
    issues = lint_corpus(tmp_path)
    assert "invalid-date" in _codes(issues)


def test_version_siblings_without_edge_is_a_warning(tmp_path):
    _write(tmp_path, "policy_v1.md", "old policy")
    _write(tmp_path, "policy_v2.md", "new policy, forgot the frontmatter")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"version-sibling-unlinked"}
    assert issues[0].level == "warning"
    assert "policy_v1.md" in issues[0].message


def test_version_siblings_with_edge_are_clean(tmp_path):
    _write(tmp_path, "policy_v1.md", "old policy")
    _write(tmp_path, "policy_v2.md", "---\nsupersedes: policy_v1.md\n---\nnew policy")
    assert lint_corpus(tmp_path) == []


def test_closure_prose_without_edge_is_a_warning(tmp_path):
    _write(tmp_path, "closed.md", "This lane is now superseded by [[new_approach]].")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"closure-marker-unlinked"}
    assert issues[0].level == "warning"


def test_closure_prose_with_edge_is_clean(tmp_path):
    _write(tmp_path, "old.md", "the old approach")
    _write(tmp_path, "closed.md",
           "---\nsupersedes: old.md\n---\nThis lane is now superseded by the new approach.")
    assert lint_corpus(tmp_path) == []


def test_a_marker_that_names_no_memo_is_ordinary_english(tmp_path):
    """Red proof, 2026-10-07: with `closure_marker_naming_a_memo` returning the first
    CLOSURE_MARKERS match whatever surrounds it (the rule before this change), this failed with
    the warning present. Ordinary uses measured on a real store: "an env block replaces the
    environment", "the proxy replaces both"."""
    _write(tmp_path, "env.md", "The env block replaces the environment; it does not extend it.")
    assert lint_corpus(tmp_path) == []


def test_a_marker_and_a_memo_in_different_sentences_is_silent(tmp_path):
    """Red proof, 2026-10-07: with `_sentence_around` returning the whole prose, this failed with
    the warning present. The link is related reading, not the replaced memo."""
    _write(tmp_path, "proxy.md", "The proxy replaces both models. See [[proxy-costs]] for the bill.")
    assert lint_corpus(tmp_path) == []


def test_a_marker_inside_a_memo_name_is_not_prose(tmp_path):
    """Red proof, 2026-10-07: with the inside-a-reference skip removed, this failed with the
    warning present, because the link's own name contains "replaces"."""
    _write(tmp_path, "see.md", "Related: [[an-mcp-env-block-replaces-the-environment]].")
    assert lint_corpus(tmp_path) == []


def test_a_marker_wrapped_onto_the_next_line_stays_in_its_sentence(tmp_path):
    """Red proof, 2026-10-07: with a single newline counted as a sentence end, this failed with
    no warning. Memos wrap prose at about 100 columns, so a line break is not a boundary."""
    _write(tmp_path, "new.md", "This decision\nsupersedes the one in\n[[old-decision]], which stood until May.")
    assert _codes(lint_corpus(tmp_path)) == {"closure-marker-unlinked"}


def test_cycle_still_reported_when_a_member_has_a_second_superseder(tmp_path):
    # regression (BUG-001): a single-valued edge map dropped one of two superseders of the
    # same target, and the declared a<->z cycle silently vanished from the report
    _write(tmp_path, "a.md", "---\nsupersedes: z.md\n---\nA")
    _write(tmp_path, "z.md", "---\nsupersedes: a.md\n---\nZ")
    _write(tmp_path, "m.md", "---\nsupersedes: z.md\n---\nM")
    issues = lint_corpus(tmp_path)
    assert "supersession-cycle" in _codes(issues)


def test_same_filename_in_two_subdirs_does_not_shadow_checks(tmp_path):
    # regression (BUG-004): dicts keyed by bare filename let sub2/x.md overwrite sub1/x.md,
    # losing sub1's closure-marker warning entirely
    (tmp_path / "sub1").mkdir()
    (tmp_path / "sub2").mkdir()
    _write(tmp_path / "sub1", "x.md", "This memo is deprecated and replaced by y.md.")
    _write(tmp_path / "sub2", "x.md", "a perfectly healthy unrelated note")
    issues = lint_corpus(tmp_path)
    assert "closure-marker-unlinked" in _codes(issues)


def test_ambiguous_supersedes_target_is_a_warning(tmp_path):
    # a supersedes: edge pointing at a basename that exists in TWO places cannot be resolved
    # unambiguously — surface it instead of silently picking one
    (tmp_path / "sub1").mkdir()
    (tmp_path / "sub2").mkdir()
    _write(tmp_path / "sub1", "old.md", "v1")
    _write(tmp_path / "sub2", "old.md", "also v1")
    _write(tmp_path, "new.md", "---\nsupersedes: old.md\n---\nv2")
    issues = lint_corpus(tmp_path)
    assert "ambiguous-supersedes-target" in _codes(issues)


def test_zero_padded_version_siblings_warn(tmp_path):
    # deferred NUM-003: reconstructing "stem_v{n+1}" missed zero-padded series entirely —
    # x_v01/x_v02 never triggered the warning (silent false negative in a completeness lint)
    _write(tmp_path, "policy_v01.md", "old policy")
    _write(tmp_path, "policy_v02.md", "new policy, forgot the frontmatter")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"version-sibling-unlinked"}
    assert "policy_v01.md" in issues[0].message


def test_non_contiguous_version_siblings_warn(tmp_path):
    _write(tmp_path, "policy_v1.md", "old policy")
    _write(tmp_path, "policy_v3.md", "new policy, v2 was deleted")
    issues = lint_corpus(tmp_path)
    assert _codes(issues) == {"version-sibling-unlinked"}


def test_missing_root_raises_cleanly_and_cli_exits_2(tmp_path):
    # deferred BUG-003: a nonexistent path fell into the single-file branch and died with a
    # raw FileNotFoundError traceback from read_text
    import pytest

    from recall.cli import main
    from recall.lint import lint_corpus as lc

    with pytest.raises(FileNotFoundError):
        lc(tmp_path / "does-not-exist")
    with pytest.raises(SystemExit) as exc:
        main(["lint", str(tmp_path / "does-not-exist")])
    assert exc.value.code == 2


def test_unreadable_file_is_reported_and_rest_of_corpus_still_linted(tmp_path):
    # deferred BUG-003: one non-UTF8 file aborted the whole lint with zero issues reported
    (tmp_path / "binary.md").write_bytes(b"\xff\xfe\x00broken")
    _write(tmp_path, "new.md", "---\nsupersedes: ghost.md\n---\ntext")
    issues = lint_corpus(tmp_path)
    codes = _codes(issues)
    assert "unreadable-file" in codes
    assert "dangling-supersedes" in codes  # healthy files still checked


def test_cli_lint_exits_nonzero_on_errors_and_zero_on_clean(tmp_path, capsys):
    from recall.cli import main

    _write(tmp_path, "new.md", "---\nsupersedes: ghost.md\n---\ntext")
    with pytest.raises(SystemExit) as exc:
        main(["lint", str(tmp_path)])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "dangling-supersedes" in out

    clean = tmp_path / "clean"
    clean.mkdir()
    _write(clean, "a.md", "fine")
    main(["lint", str(clean)])  # returns without raising
    assert "0 errors" in capsys.readouterr().out


def test_lint_accepts_the_supersedes_forms_people_actually_write(tmp_path):
    """On a real 792-memo corpus BOTH declared edges were reported as dangling — one written
    with wikilink brackets, one without the `.md`. Both targets existed. The convention has to
    accept what its own users write, or the linter's headline error is simply wrong."""
    (tmp_path / "old.md").write_text("# old decision\n\nbody", encoding="utf-8")
    for i, written in enumerate(("old.md", "old", "[old]", "[[old]]")):
        d = tmp_path / f"c{i}"
        d.mkdir()
        (d / "old.md").write_text("# old decision\n\nbody", encoding="utf-8")
        (d / "new.md").write_text(
            f"---\nsupersedes: {written}\n---\n# new decision\n\nbody", encoding="utf-8"
        )
        codes = [i.code for i in lint_corpus(d)]
        assert "dangling-supersedes" not in codes, f"{written!r} reported as dangling"


def test_lint_echoes_the_authored_form_in_a_dangling_error(tmp_path):
    """The diagnostic must quote what the user typed, not a normalised form they never wrote."""
    (tmp_path / "new.md").write_text(
        "---\nsupersedes: [[ghost]]\n---\n# new\n\nbody", encoding="utf-8"
    )
    dangling = [i for i in lint_corpus(tmp_path) if i.code == "dangling-supersedes"]
    assert dangling and "[[ghost]]" in dangling[0].message


# --- the cycle check every writer asks before it writes (supersession audit 2026-10-21) ----------

#: b supersedes c, c supersedes a; d stands outside every chain.
_CHAIN = {"a.md": [], "b.md": ["c.md"], "c.md": ["a.md"], "d.md": []}


def _declare(tmp_path, declared: dict[str, list[str]]) -> None:
    for name, targets in declared.items():
        head = f"---\nsupersedes: [{', '.join(targets)}]\n---\n" if targets else ""
        _write(tmp_path, name, f"{head}# {name}")


@pytest.mark.parametrize("superseder,target,expected", [
    ("a.md", "b", ("b", "c", "a")),   # b supersedes c supersedes a: a supersedes b closes it
    ("a.md", "c", ("c", "a")),
    ("a.md", "a", ("a",)),            # a self-edge is the shortest cycle
    ("c.md", "a", None),              # already declared, in the same direction: no new cycle
    ("d.md", "a", None),              # d is outside every chain
    ("b.md", "d", None),
])
def test_the_cycle_check_agrees_with_what_lint_reports_after_the_write(
    tmp_path, superseder, target, expected
):
    """`supersession_cycle_closed_by` names a chain exactly when writing the edge makes
    `lint_corpus` report a cycle (or a self-edge) it did not report before.

    Red: mutation of `recall.lint.supersession_cycle_closed_by` searching from the target instead
    of the superseder (`start, goal = goal, start` after the self-edge check):
    `tests/test_lint.py::test_the_cycle_check_agrees_with_what_lint_reports_after_the_write`
    failed for `("a.md", "b", ...)` with `None == ('b', 'c', 'a')`.
    """
    from recall.lint import supersession_cycle_closed_by, supersession_graph

    _declare(tmp_path, _CHAIN)
    assert lint_corpus(tmp_path) == []

    assert supersession_cycle_closed_by(supersession_graph(tmp_path), superseder, target) == expected

    after = {**_CHAIN, superseder: [*_CHAIN[superseder], f"{target}.md"]}
    _declare(tmp_path, after)
    codes = _codes(lint_corpus(tmp_path))
    assert ("supersession-cycle" in codes or "self-supersedes" in codes) == (expected is not None)


def test_describe_cycle_keys_its_names_and_says_when_the_chain_is_transitive():
    """A writer passes `describe_cycle` the names it gave the cycle check, and a chain of three or
    more is named as reached through the memos between, since the target declares no edge to the
    superseder itself.

    Red: baseline 8e10eea6a,
    `tests/test_lint.py::test_describe_cycle_keys_its_names_and_says_when_the_chain_is_transitive`
    failed at the first assertion with the names printed as given (`a.md`, `[[b]]`); target
    `recall.lint.describe_cycle` (CCA deferred findings CODE-006 and DOC-012). Red also by mutation
    dropping the transitive wording (`through = ""`): it failed at the second assertion, the
    message saying "already supersedes it (b -> c -> a)".
    """
    from recall.lint import describe_cycle

    message = describe_cycle("a.md", "[[b]]", ("b", "c", "a"))
    assert message.startswith("declaring that a supersedes b would close a supersession cycle")
    assert "because b already supersedes it through c (b -> c -> a)" in message
    direct = describe_cycle("a.md", "b.md", ("b", "a"))
    assert "because b already supersedes it (b -> a)" in direct
    assert describe_cycle("a.md", "a.md", ("a",)) == "a memo cannot supersede itself"
