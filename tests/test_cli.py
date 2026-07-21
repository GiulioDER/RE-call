
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
