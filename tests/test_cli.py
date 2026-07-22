
from recall.cli import main

from tests.conftest import TEST_DSN, requires_db


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


@requires_db
def test_cli_calibrate_writes_calibration_file(tmp_path, capsys):
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
    out_path = tmp_path / "cal.json"
    main(
        ["--embedder", "hashing", "--dsn", TEST_DSN, "calibrate", str(queries),
         "--corpus", str(tmp_path), "--out", str(out_path)]
    )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["embedder"] == "hashing-64"
    assert "threshold" in data and "scale" in data
    printed = capsys.readouterr().out
    assert "threshold" in printed
