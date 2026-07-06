import psycopg

from recall.cli import main

from tests.conftest import TEST_DSN, requires_db


@requires_db
def test_cli_index_then_search(tmp_path, capsys):
    # The CLI uses the default `chunks` table (not the test fixture) for a
    # genuine end-to-end check. Drop it first so this test is independent of any
    # prior run — notably the FastEmbed demo below, which indexes the same
    # default table at a different embedding dimension (384 vs the hashing 64).
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks")

    (tmp_path / "note.md").write_text("the caching layer decision was adopted", encoding="utf-8")
    main(["--embedder", "hashing", "--dsn", TEST_DSN, "index", str(tmp_path)])
    out = capsys.readouterr().out
    assert "indexed 1 chunks" in out

    main(["--embedder", "hashing", "--dsn", TEST_DSN, "search", "caching"])
    out = capsys.readouterr().out
    assert "caching" in out.lower()
