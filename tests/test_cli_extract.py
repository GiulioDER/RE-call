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
from recall.truth_extraction._engine import DeterministicExtractionEngine

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


def _is_not_utf8(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


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


def test_recheck_actually_checks_the_files_it_extracted(corpus, tmp_path, monkeypatch, capsys):
    """Passing different corpus_names than the run made every lookup miss, silently."""
    _enable(monkeypatch)
    main(["extract", "run", str(corpus), "--cache", str(tmp_path / "tx.sqlite3"), "--recheck"])
    out = capsys.readouterr().out
    assert "recheck: 2 checked" in out
    assert "not measured" not in out


def test_the_cache_persists_between_runs(corpus, tmp_path, monkeypatch, capsys):
    """The reason `--cache` takes a PATH. It was briefly a boolean, when nothing persisted.

    Proven by ENGINE CALLS, not by the file existing: a cache that is written and never read
    would still leave a file on disk.
    """
    _enable(monkeypatch)
    cache_path = tmp_path / "tx.sqlite3"

    main(["extract", "run", str(corpus), "--cache", str(cache_path)])
    first = capsys.readouterr().out
    assert cache_path.exists(), "nothing was written to the path"
    assert "1 claim(s) for review" in first

    calls: list[str] = []
    real_run = DeterministicExtractionEngine.run

    def _counted(self, prompt):
        calls.append(prompt.file)
        return real_run(self, prompt)

    monkeypatch.setattr(DeterministicExtractionEngine, "run", _counted)
    main(["extract", "run", str(corpus), "--cache", str(cache_path)])
    second = capsys.readouterr().out

    assert calls == [], f"the engine was re-asked for {calls}; the cache did not persist"
    assert "1 claim(s) for review" in second, "the cached run lost the claim"


def test_the_cache_is_closed_and_reported_even_when_the_run_raises(
    corpus, tmp_path, monkeypatch, capsys
):
    """Closing after the last print meant any exception leaked the connection AND skipped the
    one line that makes a silently degraded cache visible."""
    import recall.truth_extraction.extract as extract_mod
    from recall.truth_extraction._sqlite_cache import SqliteExtractionCache

    _enable(monkeypatch)
    opened: list[SqliteExtractionCache] = []
    real_init = SqliteExtractionCache.__init__

    def _tracking_init(self, path):
        real_init(self, path)
        self.write_failures = 1  # force the report line so its absence is detectable
        opened.append(self)

    monkeypatch.setattr(SqliteExtractionCache, "__init__", _tracking_init)
    monkeypatch.setattr(
        extract_mod,
        "extract_corpus_claims_for_report",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        main(["extract", "run", str(corpus), "--cache", str(tmp_path / "tx.sqlite3")])

    assert opened, "no cache was opened, so this proves nothing"
    assert "write failure(s)" in capsys.readouterr().out, "the counters were never reported"
    with pytest.raises(Exception):
        opened[0]._conn.execute("SELECT 1")  # closed


def test_an_empty_cache_path_is_refused_rather_than_silently_ignored(
    corpus, monkeypatch, capsys
):
    """`--cache "$CACHE"` with CACHE unset ran a full uncached run and said nothing."""
    _enable(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["extract", "run", str(corpus), "--cache", ""])
    assert exc.value.code == 2


def test_a_cache_path_holding_another_database_is_refused(corpus, tmp_path, monkeypatch, capsys):
    """Refused at OPEN, before a memo is read, not halfway through somebody's corpus."""
    import sqlite3

    _enable(monkeypatch)
    foreign = tmp_path / "not-ours.sqlite3"
    with sqlite3.connect(foreign) as raw:
        raw.execute("CREATE TABLE extraction_entries (id INTEGER PRIMARY KEY, whatever TEXT)")
        raw.commit()

    with pytest.raises(SystemExit) as exc:
        main(["extract", "run", str(corpus), "--cache", str(foreign)])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "extraction_entries" in captured.err
    assert "claim(s) for review" not in captured.out, "it read the corpus before refusing"


def test_a_filename_that_is_not_valid_utf8_is_reported_not_fatal(corpus, monkeypatch, capsys):
    """The END of the path, which every earlier version of this test stopped short of.

    `main()` reconfigures stdout to UTF-8, and reconfiguring the encoding RESETS errors to
    strict, dropping the inherited surrogateescape. So a name that is not valid UTF-8 raised at
    the `print`, and `recall extract run` over a corpus holding one exited 1 with EMPTY stdout,
    throwing away a completed extraction at the REPORT step. The library layer was fixed first
    and this still crashed, which is why the property is asserted here, at the entry point a
    user actually invokes, rather than one frame inside it.

    Run through `main()` deliberately, since `main()` is where the reconfigure lives.
    """
    _enable(monkeypatch)
    (corpus / "bad\udcff.md").write_bytes(b"Status: deprecated\n")
    main(["extract", "run", str(corpus)])
    out = capsys.readouterr().out
    assert "3 file(s) read" in out, "the awkward name aborted the run"
    assert "claim(s) for review" in out
    assert "supersession" in out, "the other memos' results were discarded"
    # The awkward file's OWN result, escaped rather than dropped. Without this the test passes
    # against a regression that quietly turns that file into an `engine_error` refusal: the
    # count above comes from `len(documents)` and the claim above comes from another memo, so
    # neither notices. "Showing a mangled name beats showing nothing" is the stated property,
    # and it needs the name asserted to be a property rather than a hope.
    assert "bad\\udcff.md: status" in out, "the awkward file's own claim was lost"


def test_no_corpus_name_reaches_the_engine_as_a_lone_surrogate(corpus, monkeypatch, capsys):
    """The deterministic engine takes the prompt as Python strings and never encodes them.

    A model engine does: it sends the prompt as JSON over HTTP, and a `file` or `corpus_names`
    entry holding a lone surrogate raises `UnicodeEncodeError` inside the client, on the first
    file of the corpus. Guarding the cache key made the DEFAULT engine survive such a corpus
    while leaving the same value on its way to every other one, which is why the name is made
    encodable where the corpus is read rather than at each thing that encodes it.
    """
    _enable(monkeypatch)
    (corpus / "bad\udcff.md").write_bytes(b"Status: deprecated\n")
    inner = DeterministicExtractionEngine()
    seen = []

    class _Recording:
        engine_id = inner.engine_id
        model_id = inner.model_id
        revision = inner.revision

        def run(self, prompt):
            seen.append(prompt)
            return inner.run(prompt)

    monkeypatch.setattr("recall.truth_extraction.resolve_extraction_engine", lambda: _Recording())
    main(["extract", "run", str(corpus)])
    assert seen, "the engine was never called, so this test proves nothing"
    unencodable = [
        value
        for prompt in seen
        for value in (prompt.file, *prompt.corpus_names)
        if _is_not_utf8(value)
    ]
    assert not unencodable, f"{len(unencodable)} value(s) would break a model engine"


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


class _FinalAlways:
    """Answers `final` whatever it is asked, which is what the real model did on `python/peps`.

    `DeterministicExtractionEngine` cannot reproduce the defect: it skips any status outside the
    vocabulary it was handed, so under the shipped set it emits NO status claim rather than a
    refused one. The batch refusal this flag exists to remove needs an engine that answers a word
    it was not offered.
    """

    engine_id = "final-always"
    model_id = "final-always"
    revision = "r"

    def run(self, prompt) -> str:
        return (
            '{"claims": [{"kind": "status", "value": "final", '
            '"quote": "Status: Final"}]}'
        )


@pytest.fixture
def pep_corpus(tmp_path):
    """A document whose status word is outside the shipped memo set, plus a supersession."""
    (tmp_path / "pep-0345.md").write_text(
        "Status: Final\n\nThe original.\n", encoding="utf-8", newline="\n"
    )
    (tmp_path / "pep-0376.md").write_text(
        "Status: Final\n\nThis PEP supersedes pep-0345.md after review.\n",
        encoding="utf-8",
        newline="\n",
    )
    return tmp_path


def test_a_custom_vocabulary_admits_a_status_the_shipped_set_drops(
    pep_corpus, monkeypatch, capsys
):
    """The defect, end to end at the CLI, in the direction the shipped engine can show."""
    _enable(monkeypatch)
    main(["extract", "run", str(pep_corpus)])
    without = capsys.readouterr().out
    # File-qualified for symmetry with the assertion below: the echoed
    # "status vocabulary: ..." line contains the bare word "status", so pin the claim line
    # itself rather than the substring.
    assert "pep-0376.md: status" not in without, "the shipped vocabulary should not admit `Final`"

    main(["extract", "run", str(pep_corpus), "--status-vocabulary", "Final,Rejected,Deferred"])
    with_flag = capsys.readouterr().out
    # File-qualified, not a bare "status" substring: the echoed
    # "status vocabulary: Final, Rejected, Deferred" line itself contains the word "status" and
    # would satisfy a looser check whether or not the vocabulary ever reached extraction.
    assert "pep-0376.md: status" in with_flag, "the custom vocabulary did not reach the prompt"
    assert "supersession" in with_flag, "the supersession claim was lost"


def test_the_flag_saves_the_other_claims_from_a_batch_refusal(
    pep_corpus, monkeypatch, capsys
):
    """The measured failure: one out-of-vocabulary status refuses the file at a BATCH rung.

    Driven with an engine that answers `final` regardless, because that is what the real model
    did and what the shipped deterministic engine cannot do.
    """
    import recall.truth_extraction._engine as engine_mod

    _enable(monkeypatch)
    monkeypatch.setattr(engine_mod, "resolve_extraction_engine", lambda *a, **k: _FinalAlways())
    monkeypatch.setattr(
        "recall.truth_extraction.resolve_extraction_engine", lambda *a, **k: _FinalAlways()
    )

    main(["extract", "run", str(pep_corpus)])
    assert "REFUSED" in capsys.readouterr().out, "the batch refusal is the defect being fixed"

    main(["extract", "run", str(pep_corpus), "--status-vocabulary", "final,rejected"])
    assert "REFUSED" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Final,,Rejected", "blank once stripped"),
        (",", "blank once stripped"),
        ("", "blank once stripped"),
        ("Final,FINAL", "differ only by case"),
    ],
)
def test_a_degenerate_vocabulary_exits_2_rather_than_refusing_every_status(
    corpus, monkeypatch, capsys, value, expected
):
    """Refused as the argument error it is, not carried into the ladder.

    A comma split that strips and drops blanks — which is what the labelling runner does —
    swallows exactly these. `Final,,Rejected` would pass quietly and `,` would become an empty
    vocabulary that refuses every status claim at a BATCH rung: the original defect, re-entered
    through the flag added to remove it.
    """
    _enable(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["extract", "run", str(corpus), "--status-vocabulary", value])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "recall extract: --status-vocabulary: " in err
    assert expected in err


def test_a_one_word_vocabulary_is_a_list_not_a_string(corpus, monkeypatch, capsys):
    """`Sequence[str]` accepts `str`, so a bare word would render `['f','i','n','a','l']`.

    The split is what prevents it: `"final".split(",")` is `["final"]`. Accepted, not refused.
    """
    _enable(monkeypatch)
    main(["extract", "run", str(corpus), "--status-vocabulary", "final"])
    assert "final" in capsys.readouterr().out


def test_the_vocabulary_is_refused_before_any_file_is_read(tmp_path, monkeypatch, capsys):
    """Knowable at parse time, so it must not cost a corpus of model calls first.

    Pointed at a path that does not exist: a bad vocabulary must lose the race to the missing
    path, proving it is checked before the corpus is even looked at.
    """
    _enable(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["extract", "run", str(tmp_path / "nope"), "--status-vocabulary", ","])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--status-vocabulary" in err, f"the path was checked first: {err}"


def test_show_takes_the_flag_too(pep_corpus, monkeypatch, capsys):
    """`show` carries the defect identically; a flag on `run` alone leaves the single-file
    diagnostic unable to reproduce what `run` just did."""
    _enable(monkeypatch)
    main(
        [
            "extract",
            "show",
            str(pep_corpus / "pep-0376.md"),
            "--status-vocabulary",
            "Final,Rejected",
        ]
    )
    out = capsys.readouterr().out
    # File-qualified: this is the test's only assertion, and the echoed
    # "status vocabulary: Final, Rejected" line contains the bare word "status" by itself, so
    # a substring check alone would pass even if the vocabulary never reached extraction.
    assert "pep-0376.md: status" in out


def test_recheck_measures_against_the_same_vocabulary_it_warmed(
    pep_corpus, tmp_path, monkeypatch, capsys
):
    """Not forwarded, every key misses and the report reads `checked=0`.

    A determinism measurement that silently became a non-measurement, which is the failure mode
    `recheck_cached_extractions` documents for exactly this argument.
    """
    _enable(monkeypatch)
    db = str(tmp_path / "rc.sqlite3")
    vocab = ["--status-vocabulary", "Final,Rejected"]
    main(["extract", "run", str(pep_corpus), "--cache", db, *vocab])
    capsys.readouterr()
    main(["extract", "run", str(pep_corpus), "--cache", db, "--recheck", *vocab])
    out = capsys.readouterr().out
    assert "recheck: 0 checked" not in out, f"the recheck measured nothing: {out}"


def test_the_report_names_a_non_default_vocabulary(pep_corpus, monkeypatch, capsys):
    """Two runs whose outputs differ must say on screen why they differ."""
    _enable(monkeypatch)
    main(["extract", "run", str(pep_corpus), "--status-vocabulary", "Final,Rejected"])
    assert "status vocabulary: Final, Rejected" in capsys.readouterr().out

    main(["extract", "run", str(pep_corpus)])
    assert "status vocabulary:" not in capsys.readouterr().out
