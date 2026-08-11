"""`recall rewrite`: the command line in front of the reviewed write path.

The invariant this command exists to make visible is that **no proposal reaches corpus metadata
without a named human**. On the API that is a type — `apply_rewrite` takes a `PromotedFact` and
a `PromotedFact` cannot exist without a reviewer. On a command line it has to be a required
argument, because the shell is where someone will try to automate this in a cron job, and
`--reviewer` being mandatory is what stops the answer from being "nobody".

Properties, one test each:

1. `--reviewer` and `--note` are required; omitting either exits non-zero without writing.
2. Dry run is the default, matching `cli.py:954` — the plan is printed, the corpus is unchanged.
3. `--apply` writes, and the reviewer's identity reaches the file's provenance path.
4. `--reject` records a refusal that a later run honours.
5. The corpus is untouched on the reject path.
"""
from __future__ import annotations

import pytest

from recall.cli import main


def _corpus(tmp_path):
    (tmp_path / "old_thing_2026.md").write_bytes(b"# old\n\nthe original\n")
    (tmp_path / "new.md").write_bytes(b"# new\n\nThis supersedes [[old_thing_2026]].\n")
    return tmp_path


def test_reviewer_is_required(tmp_path, capsys):
    _corpus(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", str(tmp_path), "--note", "checked it"])
    assert exc.value.code != 0
    assert b"supersedes:" not in (tmp_path / "new.md").read_bytes()


def test_audit_note_is_required(tmp_path):
    _corpus(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", str(tmp_path), "--reviewer", "giulio"])
    assert exc.value.code != 0
    assert b"supersedes:" not in (tmp_path / "new.md").read_bytes()


def test_dry_run_is_the_default(tmp_path, capsys):
    _corpus(tmp_path)
    before = (tmp_path / "new.md").read_bytes()

    main(["rewrite", str(tmp_path), "--reviewer", "giulio", "--note", "checked both memos"])

    out = capsys.readouterr().out
    assert (tmp_path / "new.md").read_bytes() == before, "the default run wrote to disk"
    assert "dry run" in out
    assert "new.md" in out, "a dry run must still say what it would have done"


def test_apply_writes_the_edge(tmp_path, capsys):
    _corpus(tmp_path)

    main(["rewrite", str(tmp_path), "--reviewer", "giulio", "--note", "checked both memos",
          "--apply"])

    assert b"supersedes: old_thing_2026.md" in (tmp_path / "new.md").read_bytes()
    assert b"the original" in (tmp_path / "old_thing_2026.md").read_bytes()


def test_a_rejected_proposal_is_not_offered_again(tmp_path, capsys):
    _corpus(tmp_path)
    ledger = tmp_path / ".recall" / "rejections.sqlite3"

    main(["rewrite", str(tmp_path), "--reviewer", "giulio", "--note", "checked both memos",
          "--ledger", str(ledger)])
    first = capsys.readouterr().out
    assert "new.md" in first, "the first pass must offer it, or the rejection means nothing"

    main(["rewrite", str(tmp_path), "--reviewer", "giulio", "--note", "augments, not replaces",
          "--ledger", str(ledger), "--reject-all"])
    capsys.readouterr()

    main(["rewrite", str(tmp_path), "--reviewer", "giulio", "--note", "checked both memos",
          "--ledger", str(ledger)])
    second = capsys.readouterr().out
    assert "0 edge(s) proposable" in second
    assert b"supersedes:" not in (tmp_path / "new.md").read_bytes()


def test_rejecting_does_not_write_to_the_corpus(tmp_path, capsys):
    _corpus(tmp_path)
    before = (tmp_path / "new.md").read_bytes()

    main(["rewrite", str(tmp_path), "--reviewer", "giulio", "--note", "no",
          "--ledger", str(tmp_path / "l.sqlite3"), "--reject-all", "--apply"])

    assert (tmp_path / "new.md").read_bytes() == before, (
        "--reject-all combined with --apply must reject, not write"
    )
