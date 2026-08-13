"""`recall extract` is a filesystem command: no DB, no embedder, OFF unless enabled.

Properties, one test each:

1. It refuses when extraction is off, naming the variable that turns it on.
2. It refuses an unknown engine rather than quietly running the deterministic one.
3. It lists each claim with the quote that proves it, because a claim a reviewer cannot check
   against the memo is one they have to take on faith.
4. It writes nothing, and says so.
5. `--limit` bounds how many files are read.
6. It never opens the database. Extraction is an ingest concern and must not need a DSN.
7. `extract show` reports one file, refusals included.
8. A file whose claims were all refused reports the refusal rather than silence.
"""
import pytest

from recall.cli import main

ENABLED = {"RECALL_TRUTH_EXTRACTION": "1"}


@pytest.fixture(autouse=True)
def _no_ambient_extraction(monkeypatch):
    """The engine reads the process environment; a stray value would silently steer these."""
    for name in (
        "RECALL_TRUTH_EXTRACTION",
        "RECALL_TRUTH_EXTRACTION_ENGINE",
        "RECALL_EXTRACTION_API_KEY",
        "RECALL_EXTRACTION_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "old_2026-01-01.md").write_text(
        "# old\n\nThe original call.\n", encoding="utf-8", newline="\n"
    )
    (tmp_path / "new_2026-02-01.md").write_text(
        "# new\n\nThis memo supersedes old_2026-01-01.md after review.\n",
        encoding="utf-8",
        newline="\n",
    )
    return tmp_path


def _enable(monkeypatch):
    for key, value in ENABLED.items():
        monkeypatch.setenv(key, value)


def test_extract_run_refuses_when_extraction_is_off(corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["extract", "run", str(corpus)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "RECALL_TRUTH_EXTRACTION" in err, "the refusal must name the variable to set"


def test_extract_run_refuses_an_unknown_engine(corpus, monkeypatch, capsys):
    """Quietly downgrading would make the audit record wrong about how a claim was produced."""
    _enable(monkeypatch)
    monkeypatch.setenv("RECALL_TRUTH_EXTRACTION_ENGINE", "gpt9")
    with pytest.raises(SystemExit) as exc:
        main(["extract", "run", str(corpus)])
    assert exc.value.code == 2
    assert "not a known engine" in capsys.readouterr().err


def test_extract_run_lists_each_claim_with_its_quote(corpus, monkeypatch, capsys):
    _enable(monkeypatch)
    main(["extract", "run", str(corpus)])
    out = capsys.readouterr().out
    assert "new_2026-02-01.md" in out
    assert "supersession" in out
    assert "This memo supersedes old_2026-01-01.md after review." in out


def test_extract_run_says_it_wrote_nothing(corpus, monkeypatch, capsys):
    _enable(monkeypatch)
    main(["extract", "run", str(corpus)])
    assert "nothing written" in capsys.readouterr().out.lower()


def test_extract_run_help_states_it_writes_nothing(capsys):
    with pytest.raises(SystemExit):
        main(["extract", "run", "--help"])
    assert "writes nothing" in capsys.readouterr().out.lower()


def test_extract_run_honours_limit(corpus, monkeypatch, capsys):
    _enable(monkeypatch)
    main(["extract", "run", str(corpus), "--limit", "1"])
    assert "1 file(s) read" in capsys.readouterr().out


def test_extract_never_opens_the_database(corpus, monkeypatch, capsys):
    """Extraction runs on the ingest path and must not need a DSN.

    Monkeypatching RECALL_DSN does NOT work here and an earlier version of this test did
    exactly that: `DEFAULT_DSN` is bound at import time, so the env change arrived too late and
    the fallback pointed at this machine's native Postgres on 5432. A mutant that opened a real
    connection passed in 2.18s. The guard has to make ANY connection attempt fail, whatever is
    listening locally, so it fails `psycopg.connect` itself.
    """
    import psycopg

    _enable(monkeypatch)

    def _forbidden(*a, **k):
        raise AssertionError("recall extract opened a database connection")

    monkeypatch.setattr(psycopg, "connect", _forbidden)
    main(["extract", "run", str(corpus)])
    assert "claim(s) for review" in capsys.readouterr().out


def test_limit_does_not_change_the_answers(corpus, monkeypatch, capsys):
    """`--limit` is a SAMPLING flag. Slicing the corpus too fabricated refusals.

    Measured before the fix: the same memo yielded a valid claim over the full corpus and
    "names 'old_2026-01-01.md', which is not a file in the corpus" under `--limit 1`, about a
    file sitting right beside it. A fabricated refusal reads exactly like a real one.
    """
    _enable(monkeypatch)
    main(["extract", "run", str(corpus), "--limit", "1"])
    out = capsys.readouterr().out
    assert "target_not_in_corpus" not in out, "limiting the read fabricated a refusal"
    assert "1 file(s) read" in out


def test_running_on_a_single_file_resolves_against_its_corpus(corpus, monkeypatch, capsys):
    _enable(monkeypatch)
    main(["extract", "run", str(corpus / "new_2026-02-01.md")])
    out = capsys.readouterr().out
    assert "target_not_in_corpus" not in out
    assert "1 claim(s) for review" in out


@pytest.mark.parametrize("bad", ["0", "-1", "abc"])
def test_a_nonsensical_limit_is_refused_at_parse_time(corpus, bad, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["extract", "run", str(corpus), "--limit", bad])
    assert exc.value.code == 2


def test_a_missing_path_is_refused_not_reported_as_an_empty_corpus(
    tmp_path, monkeypatch, capsys
):
    """`0 claim(s) for review` on a typo reads as "this corpus states nothing"."""
    _enable(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["extract", "run", str(tmp_path / "does_not_exist")])
    assert exc.value.code == 2
    assert "no such path" in capsys.readouterr().err


def test_show_refuses_a_directory(corpus, monkeypatch, capsys):
    _enable(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["extract", "show", str(corpus)])
    assert exc.value.code == 2
    assert "not a file" in capsys.readouterr().err


def test_one_unreadable_file_does_not_abort_the_run(corpus, monkeypatch, capsys):
    """UnicodeDecodeError is a ValueError, not an OSError. One bad memo discarded every other.

    `lint.py` and `fix.py` both catch the pair per file and keep going.
    """
    _enable(monkeypatch)
    (corpus / "latin_2026-04-01.md").write_bytes(b"# caf\xe9\n\nStatus: deprecated\n")
    main(["extract", "run", str(corpus)])
    out = capsys.readouterr().out
    assert "UNREADABLE latin_2026-04-01.md" in out
    assert "supersession" in out, "a decodable memo was discarded with the undecodable one"


def test_files_sharing_a_basename_are_both_read(tmp_path, monkeypatch, capsys):
    """The default glob is recursive, so basename keys collapsed distinct files onto one."""
    _enable(monkeypatch)
    for folder in ("legal", "eng"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "policy_2026-01-01.md").write_text(
            f"# {folder}\n\nThe {folder} policy.\n", encoding="utf-8", newline="\n"
        )
    main(["extract", "run", str(tmp_path)])
    assert "2 file(s) read" in capsys.readouterr().out


def test_an_invalid_glob_is_refused_not_a_traceback(corpus, monkeypatch, capsys):
    _enable(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["extract", "run", str(corpus), "--glob", ""])
    assert exc.value.code == 2
    assert "--glob" in capsys.readouterr().err


def test_recheck_without_cache_refuses_before_extracting(corpus, monkeypatch, capsys):
    """Deferring this meant paying for a whole corpus of model calls, then exiting 2."""
    _enable(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["extract", "run", str(corpus), "--recheck"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--recheck needs --cache" in captured.err
    assert "claim(s) for review" not in captured.out, "it extracted before refusing"


def test_recheck_actually_checks_the_files_it_extracted(corpus, monkeypatch, capsys):
    """Passing different corpus_names than the run made every lookup miss, silently."""
    _enable(monkeypatch)
    main(["extract", "run", str(corpus), "--cache", "--recheck"])
    out = capsys.readouterr().out
    assert "recheck: 2 checked" in out
    assert "not measured" not in out


def test_extract_show_reports_only_the_named_file(corpus, monkeypatch, capsys):
    _enable(monkeypatch)
    main(["extract", "show", str(corpus / "new_2026-02-01.md")])
    out = capsys.readouterr().out
    assert "1 file(s) read" in out
    assert "supersession" in out


def test_extract_show_resolves_targets_against_the_containing_corpus(
    corpus, monkeypatch, capsys
):
    """Resolving against the single file refuses every supersession it states.

    "which is not a file in the corpus" is true of a one file corpus and useless as an answer:
    the reviewer asked about this memo, not about a corpus of one.
    """
    _enable(monkeypatch)
    main(["extract", "show", str(corpus / "new_2026-02-01.md")])
    out = capsys.readouterr().out
    assert "target_not_in_corpus" not in out
    assert "1 claim(s) for review" in out


def test_a_refused_claim_is_reported_rather_than_hidden(tmp_path, monkeypatch, capsys):
    """A refusal nobody sees is a refusal nobody reviews."""
    _enable(monkeypatch)
    (tmp_path / "memo_2026-03-01.md").write_text(
        "# memo\n\nThis memo supersedes a_memo_that_does_not_exist.md.\n",
        encoding="utf-8",
        newline="\n",
    )
    main(["extract", "run", str(tmp_path)])
    out = capsys.readouterr().out
    assert "target_not_in_corpus" in out
    assert "0 claim(s) for review" in out
