
import pytest

from recall.cli import main

from tests.conftest import TEST_DSN, requires_db


@pytest.fixture(autouse=True)
def _cli_development_mode(monkeypatch):
    """These tests drive the CLI against throwaway tables with no published calibration.

    The CLI is strict by default like everything else, so without this it would refuse before
    reaching the index, forget and supersession behaviour under test. Setting the variable here
    is the same deliberate opt-in a developer makes locally. The strict default itself is covered
    separately in `tests/test_cli_trust_mode.py`.
    """
    monkeypatch.setenv("RECALL_TRUST_MODE", "development")


@requires_db
def test_cli_index_then_search(tmp_path, capsys, cli_table):
    # Runs against a uuid-named throwaway table via --table, NOT the default `chunks`.
    # Dropping `chunks` was how this suite destroyed a real memory index when RECALL_DSN
    # was exported; a per-test table also makes the run independent of any prior one —
    # notably the FastEmbed demo, which indexes at a different dimension (384 vs 64).
    (tmp_path / "note.md").write_text("the caching layer decision was adopted", encoding="utf-8")
    main(["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table,
          "index", str(tmp_path)])
    out = capsys.readouterr().out
    assert "indexed 1 chunks" in out

    main(["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table, "search", "caching"])
    out = capsys.readouterr().out
    assert "caching" in out.lower()


@requires_db
def test_cli_forget_without_yes_is_a_dry_run_that_deletes_nothing(tmp_path, capsys, cli_table):
    (tmp_path / "note.md").write_text("the caching layer decision was adopted", encoding="utf-8")
    main(["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table,
          "index", str(tmp_path)])
    capsys.readouterr()

    main(["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table,
          "forget", str((tmp_path / "note.md").resolve())])
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "nothing deleted" in out

    main(["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table, "search", "caching"])
    out = capsys.readouterr().out
    assert "caching" in out.lower()  # the dry run must not have removed it


@requires_db
def test_cli_forget_with_yes_deletes_and_reports_not_found_separately(tmp_path, capsys, cli_table):
    (tmp_path / "note.md").write_text("the caching layer decision was adopted", encoding="utf-8")
    main(["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table,
          "index", str(tmp_path)])
    capsys.readouterr()

    main(["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table,
          "forget", str((tmp_path / "note.md").resolve()), "typo-source", "--yes"])
    out = capsys.readouterr().out
    assert "forgot 1 chunk(s) from 1 source(s)" in out
    assert "not found" in out and "typo-source" in out

    main(["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table,
          "search", "caching", "-k", "5"])
    out = capsys.readouterr().out
    assert "ABSTAIN" in out  # actually gone this time — no hit survives, not even the query echo


@requires_db
def test_cli_demo_shows_supersession_redirect(capsys, cli_table):
    main(["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table, "demo"])
    out = capsys.readouterr().out
    assert "superseded" in out          # the stale rate-limit memory is flagged, not trusted
    assert "rate_limits_v2.md" in out   # the successor is surfaced


def test_cli_legacy_process_global_calibration_form_is_rejected(tmp_path):
    import json

    (tmp_path / "a.md").write_text("cats purr loudly", encoding="utf-8")
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            [
                {"id": "q1", "query": "cats purr", "answerable": True, "relevant_ids": ["a.md:0"]},
                {"id": "u1", "query": "zebra stripes", "answerable": False, "relevant_ids": []},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(["--embedder", "hashing", "calibrate", str(queries)])
    assert exc.value.code == 2
    assert not (tmp_path / "calibration.json").exists()


@requires_db
def test_cli_tenant_flag_scopes_forget_to_that_tenant(tmp_path, capsys, cli_table):
    """`forget` is the right-to-erasure path; without `--tenant` it could only reach `default`.

    An erasure request against any other tenant reported "not found (check for typos)" and
    deleted nothing — a silent no-op that reads exactly like success, while the data stayed
    indexed and retrievable.
    """
    (tmp_path / "note.md").write_text("the caching layer decision was adopted", encoding="utf-8")
    base = ["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table]
    main([*base, "--tenant", "acme", "index", str(tmp_path)])
    capsys.readouterr()

    source = str(tmp_path / "note.md")

    # Without the flag the store looks at `default`, where that source does not exist.
    main([*base, "forget", source, "--yes"])
    assert "not found" in capsys.readouterr().out

    # ...and the memory is still there under its own tenant.
    main([*base, "--tenant", "acme", "search", "caching"])
    assert "caching" in capsys.readouterr().out.lower()

    # With the flag it is actually erased.
    main([*base, "--tenant", "acme", "forget", source, "--yes"])
    out = capsys.readouterr().out
    assert "forgot" in out and "not found" not in out


@requires_db
def test_cli_index_reports_unchanged_and_pruned_counts(tmp_path, capsys, cli_table):
    """A silent destructive step is the thing to avoid here.

    `files` counts what was RE-indexed, so an unchanged re-run printed "indexed 0 chunks from 0
    files" — indistinguishable from an empty index. And pruning, the destructive half of
    `index`, was reported only through a log record the CLI never configured a handler for, so a
    deletion could happen with no output at all.
    """
    (tmp_path / "a.md").write_text("the caching layer decision", encoding="utf-8")
    (tmp_path / "b.md").write_text("the retry policy decision", encoding="utf-8")
    base = ["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table]
    main([*base, "index", str(tmp_path)])
    capsys.readouterr()

    main([*base, "index", str(tmp_path)])  # nothing changed
    assert "2 unchanged" in capsys.readouterr().out

    (tmp_path / "b.md").unlink()
    main([*base, "index", str(tmp_path)])
    assert "pruned 1 source" in capsys.readouterr().out


@requires_db
def test_cli_logging_goes_to_stderr_not_stdout(tmp_path, capsys, cli_table):
    """`main()` now configures logging, and stdout must stay clean.

    The CLI prints results on stdout and callers pipe them; the MCP server has the sharper
    version of the same constraint (stdout carries JSON-RPC). Log records belong on stderr.
    """
    (tmp_path / "a.md").write_text("the caching layer decision", encoding="utf-8")
    base = ["--embedder", "hashing", "--dsn", TEST_DSN, "--table", cli_table]
    main([*base, "index", str(tmp_path)])
    capsys.readouterr()

    (tmp_path / "a.md").unlink()
    main([*base, "index", str(tmp_path)])
    captured = capsys.readouterr()
    # Both halves matter. Asserting only the absence would pass vacuously against the old code,
    # where no handler was configured at all and the record went nowhere — which WAS the bug.
    assert "pruning" in captured.err, "the prune log record was emitted nowhere"
    assert "pruning" not in captured.out, "a log record leaked onto stdout"


def test_cli_db_commands_fail_closed_on_insecure_default_dsn(monkeypatch, tmp_path):
    from recall import cli as cli_mod

    monkeypatch.setattr(
        cli_mod,
        "DEFAULT_DSN",
        "postgresql://recall:recall@db.example.com:5432/recall",
    )

    def _blocked(_: str) -> None:
        raise PermissionError("blocked")

    monkeypatch.setattr(cli_mod, "require_secure_dsn", _blocked)

    with pytest.raises(PermissionError, match="blocked"):
        main(["search", "caching"])

    note = tmp_path / "note.md"
    note.write_text("the caching layer decision was adopted", encoding="utf-8")
    monkeypatch.setattr(cli_mod, "require_secure_dsn", _blocked)
    main(["check", str(note)])
