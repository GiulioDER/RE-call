"""`recall rewrite` puts the named human gate at the argument parser.

`--reviewer` and `--note` are hard argparse requirements on `apply`, so the gate fires before
any code runs. `required=True` is satisfied by an empty string, though, so a second check
refuses whitespace: a gate a caller passes by typing nothing is a field, not a person.

Properties, one test each:

1. `apply` without `--reviewer` is refused by the parser.
2. `apply` without `--note` is refused by the parser.
3. An empty `--reviewer` is refused, which argparse alone would allow.
4. An empty `--note` is likewise refused.
5. `apply` is a DRY RUN by default and writes nothing.
6. `--apply` writes the edge, and writes it to the SUPERSEDING memo.
7. `apply` takes exactly one proposal.
8. `reject` requires a named human too, and needs a corpus to resolve the id against.
9. A rejected claim does not resurface in a later `plan`.
10. `plan` writes nothing.
11. `verify` reports a declared edge whose target does not resolve.
12. None of it opens the database.
"""
from pathlib import Path

import pytest

from recall.cli import main
from recall.rewrite import corpus_proposals

OLD = "old_2026-01-01.md"
NEW = "new_2026-02-01.md"


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setenv("RECALL_TRUTH_EXTRACTION", "1")
    for name in ("RECALL_TRUTH_EXTRACTION_ENGINE", "RECALL_EXTRACTION_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / OLD).write_text("# old\n\nThe original call.\n", encoding="utf-8", newline="\n")
    (tmp_path / NEW).write_text(
        f"# new\n\nThis memo supersedes {OLD} after review.\n", encoding="utf-8", newline="\n"
    )
    return tmp_path


@pytest.fixture
def proposal_id(corpus):
    found = [p for p in corpus_proposals(corpus) if p.proposed_relation == "supersedes"]
    assert found, "the corpus states a supersession the deterministic engine should find"
    return found[0].id


def test_apply_without_a_reviewer_is_refused_by_the_parser(corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "apply", str(corpus), "--proposal", "ip_x", "--note", "n"])
    assert exc.value.code == 2
    assert "--reviewer" in capsys.readouterr().err


def test_apply_without_a_note_is_refused_by_the_parser(corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "apply", str(corpus), "--proposal", "ip_x", "--reviewer", "gde"])
    assert exc.value.code == 2
    assert "--note" in capsys.readouterr().err


@pytest.mark.parametrize("field", ["reviewer", "note"])
def test_an_empty_named_human_is_refused_though_argparse_allows_it(corpus, field, capsys):
    """`required=True` accepts "". A gate satisfied by typing nothing is not a gate."""
    args = ["rewrite", "apply", str(corpus), "--proposal", "ip_x",
            "--reviewer", "gde", "--note", "checked it"]
    args[args.index(f"--{field}") + 1] = "   "
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert field in capsys.readouterr().err.lower()


def test_apply_is_a_dry_run_by_default(corpus, proposal_id, capsys):
    before = {p.name: p.read_bytes() for p in corpus.glob("*.md")}
    main(["rewrite", "apply", str(corpus), "--proposal", proposal_id,
          "--reviewer", "gde", "--note", "Read both memos."])
    assert {p.name: p.read_bytes() for p in corpus.glob("*.md")} == before
    assert "dry run" in capsys.readouterr().out.lower()


def test_apply_writes_the_edge_to_the_superseding_memo(corpus, proposal_id, capsys):
    main(["rewrite", "apply", str(corpus), "--proposal", proposal_id,
          "--reviewer", "gde", "--note", "Read both memos.", "--apply"])
    assert "supersedes:" in (corpus / NEW).read_text(encoding="utf-8")
    assert "supersedes:" not in (corpus / OLD).read_text(encoding="utf-8")


def test_apply_takes_exactly_one_proposal(corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "apply", str(corpus), "--reviewer", "gde", "--note", "n"])
    assert exc.value.code == 2
    assert "--proposal" in capsys.readouterr().err


def test_an_unknown_proposal_is_refused(corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "apply", str(corpus), "--proposal", "ip_nope",
              "--reviewer", "gde", "--note", "n", "--apply"])
    assert exc.value.code == 2
    assert "ip_nope" in capsys.readouterr().err


def test_reject_requires_a_named_human(corpus, proposal_id, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "reject", str(corpus), "--proposal", proposal_id])
    assert exc.value.code == 2


def test_a_rejected_claim_does_not_resurface(corpus, proposal_id, capsys):
    """A proposal a human declined must stay declined, or the output must be hand filtered
    every run and the tool has saved nobody any work."""
    main(["rewrite", "reject", str(corpus), "--proposal", proposal_id,
          "--reviewer", "gde", "--note", "augments, does not supersede"])
    capsys.readouterr()
    main(["rewrite", "plan", str(corpus)])
    out = capsys.readouterr().out
    assert "REJECTED" in out, "a declined claim came back unmarked"


def test_plan_writes_nothing(corpus, capsys):
    before = {p.name: p.read_bytes() for p in corpus.glob("*.md")}
    main(["rewrite", "plan", str(corpus)])
    assert {p.name: p.read_bytes() for p in corpus.glob("*.md")} == before
    assert "dry run" in capsys.readouterr().out.lower()


def test_verify_reports_an_edge_whose_target_is_missing(corpus, capsys):
    (corpus / NEW).write_text(
        "---\nsupersedes: a_memo_that_never_existed.md\n---\n# new\n\nBody.\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "verify", str(corpus)])
    assert exc.value.code == 1
    assert "a_memo_that_never_existed.md" in capsys.readouterr().out


def test_verify_passes_on_a_corpus_whose_edges_resolve(corpus, proposal_id, capsys):
    main(["rewrite", "apply", str(corpus), "--proposal", proposal_id,
          "--reviewer", "gde", "--note", "Read both memos.", "--apply"])
    capsys.readouterr()
    main(["rewrite", "verify", str(corpus)])
    assert "0 unresolved" in capsys.readouterr().out


def test_plan_writes_nothing_at_all_including_the_ledger(corpus, capsys):
    """Snapshots the WHOLE tree, not just *.md.

    `plan` opened the rejection ledger for write, creating `<root>/.recall/rejections.sqlite3`
    while printing "nothing written". A test that only watched the memos could not see it.
    """
    before = {p.relative_to(corpus).as_posix() for p in corpus.rglob("*")}
    main(["rewrite", "plan", str(corpus)])
    assert {p.relative_to(corpus).as_posix() for p in corpus.rglob("*")} == before


def test_plan_works_on_a_corpus_it_cannot_write_to(corpus, monkeypatch, capsys):
    """Creating the ledger made `plan` fail outright on a read-only corpus."""
    from recall.rewrite import RejectionLedger

    def _forbidden(*a, **k):
        raise AssertionError("plan opened the rejection ledger for write")

    monkeypatch.setattr(RejectionLedger, "__init__", _forbidden)
    main(["rewrite", "plan", str(corpus)])
    assert "proposal(s)" in capsys.readouterr().out


def test_a_proposal_id_does_not_depend_on_how_the_path_was_spelled(corpus, monkeypatch):
    """Node ids are hashed into the proposal id, so an absolute `source` made the id depend on
    the user's cwd: `plan <abs>` then `apply --proposal <id>` from a relative path failed."""
    import os

    absolute = [p.id for p in corpus_proposals(corpus)]
    monkeypatch.chdir(corpus.parent)
    relative = [p.id for p in corpus_proposals(Path(os.path.relpath(corpus)))]
    assert absolute == relative, "the proposal id changed with the path spelling"


def test_an_unreadable_memo_does_not_remove_its_neighbours_proposals(corpus, capsys):
    """corpus_names was built from the DECODED documents, so one UTF-16 memo removed itself
    from the corpus and its neighbour's supersession was refused as target_not_in_corpus."""
    (corpus / OLD).write_bytes("# old\n\nThe original call.\n".encode("utf-16"))
    found = [p for p in corpus_proposals(corpus) if p.proposed_relation == "supersedes"]
    assert found, "an undecodable neighbour deleted the proposal from the review queue"


def test_verify_refuses_an_ambiguous_target(tmp_path, capsys):
    """Two files sharing a basename resolve to two files and therefore to none.

    `lint` calls this ambiguous-supersedes-target and `_resolve` refuses to write it; verify
    keyed by bare basename and reported it as resolved.
    """
    for folder in ("legal", "eng"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "old_2026-01-01.md").write_text(
            "# old\n", encoding="utf-8", newline="\n"
        )
    (tmp_path / "policy_2026-02-01.md").write_text(
        "---\nsupersedes: old_2026-01-01.md\n---\n# policy\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "verify", str(tmp_path)])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "AMBIGUOUS" in out
    assert "1 unresolved" in out


def test_verify_of_one_memo_resolves_against_its_corpus(corpus, proposal_id, capsys):
    """Scoping the corpus to the single file reported every resolving edge as unresolved."""
    main(["rewrite", "apply", str(corpus), "--proposal", proposal_id,
          "--reviewer", "gde", "--note", "Read both memos.", "--apply"])
    capsys.readouterr()
    main(["rewrite", "verify", str(corpus / NEW)])
    assert "0 unresolved" in capsys.readouterr().out


def test_plan_on_one_memo_resolves_against_its_corpus(corpus, capsys):
    main(["rewrite", "plan", str(corpus / NEW)])
    assert "1 proposal(s)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("env", "value", "expected"),
    [
        ("RECALL_TRUTH_EXTRACTION", "maybe", "not a boolean"),
        ("RECALL_TRUTH_EXTRACTION_ENGINE", "gpt9", "not a known engine"),
    ],
)
def test_a_bad_setting_is_refused_not_a_traceback(corpus, monkeypatch, capsys, env, value, expected):
    monkeypatch.setenv(env, value)
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "plan", str(corpus)])
    assert exc.value.code == 2
    assert expected in capsys.readouterr().err


def test_a_bad_glob_is_refused_not_a_traceback(corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "plan", str(corpus), "--glob", ""])
    assert exc.value.code == 2
    assert "--glob" in capsys.readouterr().err


def test_the_dry_run_reports_an_already_rejected_claim(corpus, proposal_id, capsys):
    """The preview promised a write that --apply then refused."""
    main(["rewrite", "reject", str(corpus), "--proposal", proposal_id,
          "--reviewer", "gde", "--note", "augments, does not supersede"])
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "apply", str(corpus), "--proposal", proposal_id,
              "--reviewer", "gde", "--note", "changed my mind"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "already rejected" in out
    assert "Re-run with --apply" not in out


def test_apply_exits_non_zero_when_nothing_was_written(corpus, proposal_id, capsys):
    """A script could not tell a completed declaration from a refused one."""
    main(["rewrite", "reject", str(corpus), "--proposal", proposal_id,
          "--reviewer", "gde", "--note", "no"])
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "apply", str(corpus), "--proposal", proposal_id,
              "--reviewer", "gde", "--note", "n", "--apply"])
    assert exc.value.code == 1
    assert "supersedes:" not in (corpus / NEW).read_text(encoding="utf-8")


def test_an_already_declared_proposal_is_marked_in_plan(corpus, proposal_id, capsys):
    """Otherwise accepted work reappears every run, indistinguishable from unreviewed work."""
    main(["rewrite", "apply", str(corpus), "--proposal", proposal_id,
          "--reviewer", "gde", "--note", "Read both memos.", "--apply"])
    capsys.readouterr()
    main(["rewrite", "plan", str(corpus)])
    assert "DECLARED" in capsys.readouterr().out


def test_rewrite_refuses_when_extraction_is_off(corpus, monkeypatch, capsys):
    """Every verb but `verify` needs proposals, and proposals need the extractor.

    The autouse fixture turns extraction on for every other test here, so without this one the
    refusal was unreachable: deleting it left the whole file green.
    """
    monkeypatch.delenv("RECALL_TRUTH_EXTRACTION", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "plan", str(corpus)])
    assert exc.value.code == 2
    assert "RECALL_TRUTH_EXTRACTION" in capsys.readouterr().err


def test_verify_works_without_the_extractor(corpus, monkeypatch, capsys):
    """`verify` only reads declared frontmatter, so it must not need the model path at all."""
    monkeypatch.delenv("RECALL_TRUTH_EXTRACTION", raising=False)
    main(["rewrite", "verify", str(corpus)])
    assert "0 unresolved" in capsys.readouterr().out


def test_rewrite_never_opens_the_database(corpus, monkeypatch, capsys):
    """Fails psycopg.connect itself: monkeypatching RECALL_DSN is too late, since DEFAULT_DSN
    is bound at import and its fallback reaches a native Postgres on this machine."""
    import psycopg

    def _forbidden(*a, **k):
        raise AssertionError("recall rewrite opened a database connection")

    monkeypatch.setattr(psycopg, "connect", _forbidden)
    main(["rewrite", "plan", str(corpus)])
    assert "proposal(s)" in capsys.readouterr().out
