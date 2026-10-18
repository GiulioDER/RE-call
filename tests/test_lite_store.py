"""`LiteStore` answers the store calls the indexer and the retriever make, with the Postgres rules.

No database server: each test gets its own SQLite file under `tmp_path`.

Invariants and the failure each one catches:
- S1 dense search is an exact cosine ranking, best first, with the true cosine as the score.
- S2 a `source=` filter matches the root-relative `file` or the absolute source, as Postgres does.
- S3 the keyword leg matches stemmed terms, ignores stop words, and puts numeric matches first.
- S4 a keyword search given `vec` scores its hits by dense cosine, keeping keyword order.
- S5 re-indexing a source keeps each chunk's `first_indexed_at` (the trust layer dates
  supersession by it) while moving `indexed_at`.
- S6 a file re-indexed from another root (same project, same relative file) replaces its old rows
  instead of landing beside them.
- S7 folder, facet and prefix scopes filter exactly like `Scope.predicate`.
- S8 one file holds one tenant and one embedding width; opening it otherwise is refused.
- S9 the real `Indexer` indexes, skips unchanged files and prunes vanished ones on a lite store.

Red proof, 2026-10-07, each mutation alone against `recall/lite/store.py`, failing in the named
assertion (JUnit XML), then restored byte for byte and green:
- M1 (S1) the matrix left unnormalised: the ranking came back `['a2', 'a1', 'a3']`.
- M2 (S2) `source=` compared to the absolute source only: `[] == ['b1']`.
- M3 (S3) stop words kept in the query: "a query of stop words matched something".
- M4 (S3) numeric matches not sorted first: "numeric matches were not ranked first".
- M5 (S4) keyword hits re-sorted by cosine: "the vector changed the keyword order".
- M6 (S5) the `first_indexed_at` restore skipped: "re-indexing moved the first write forward".
- M7 (S6) the project and file delete skipped: "the old root's rows stayed beside the new ones".
- M8 (S7) the folder arm without the separator: "a folder matched a sibling with the same prefix".
- M9 (S8) the tenant and width check skipped: `DID NOT RAISE LiteStoreError`.
- M10 (S9) `source_content_hashes` forgetting every hash: "an unchanged file was re-indexed".

Audit of PR 894, 2026-10-07. Invariants added (S7's facet rows moved from a `facet` key to
`FACET_METADATA_KEY`, `type`, because the old fixture pinned the defect; a `facet` row stays as a
control that must not match):
- S10 a delete whose version bump fails leaves the cache and the file agreeing.
- S11 an error after which SQLite already rolled back reaches the caller unmasked.
- S12 a newline in a file name changes nothing an authorization prefix admits.
- S13 `_matches` knows every `Scope` field, and the digest arm filters.
- S14 a rowid freed by a delete is never given to a new row, so a stale read cannot alias it.
- S15 NaN and infinite embeddings are refused, as pgvector refuses them.
- S16 two first opens with different tenants cannot both claim one file.
- S17 the (project, file) delete each indexer flush runs uses its index.
- S18 `k` must be positive, as in the Postgres store.
- S19 a table name other than `chunks` is refused rather than ignored.
- S20 the helpers copied from `recall.store` still match it.
- S21 `chunks_for_source` and `iter_chunks` return Postgres's order without building the matrix.
- S22 a failed COMMIT does not leave the transaction open.
- S23 `iter_chunks` sees one snapshot and every id, the empty one included.

Red proof, run on the remote test host: the pre-fix `recall/lite` (PR head `f9d08cf5`) under these
tests failed S7 "the facet arm did not read FACET_METADATA_KEY", S10 "the delete committed without
moving the corpus version", S11 "the rollback masked the real error", S12 "a newline changed what
an authorization prefix admits", S14 "a reused rowid resolved to a deleted chunk", S15 and S16
`DID NOT RAISE LiteStoreError`, S18 `DID NOT RAISE ValueError`, S19 `DID NOT RAISE
LiteStoreError`, S21 "not in ingestion order". Where the pre-fix code lacks the symbol or the test
is a pin, a mutation of the fixed code, each alone, failed in the named assertion:
- MT (S10) `delete_sources` back under the bare lock, no transaction, `_bump` kept as fixed: "the
  delete committed without moving the corpus version, so another store still serves it".
- MI (S17) `chunks_project_file` index removed: "every indexer flush scans the table".
- MR (S20) `%?` dropped from the lite numeric regex: the pattern assertion.
- MC (S20) `_chunk_identity` accepting a mixed-project batch: the batch assertion.
- MS (S13) an `owner` field added to `Scope`: "Scope gained a field that ... `_matches` does not evaluate".
- MD (S13) the digest arm skipped: `{'d1', 'd2'} == {'d1'}`.
- MW (S11) ROLLBACK issued unconditionally: "the rollback masked the real error".
- W2 (S22) COMMIT moved outside the guarded `try`: "the failed commit left the transaction open".
- I2 (S23) keyset pages under separate reads: "the iteration mixed two corpus states".
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from recall.lite import LiteStore, LiteStoreError
from recall.scope import Scope
from recall.types import Chunk

DIM = 8


def _unit(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def _vec(*head: float) -> list[float]:
    return [*head, *([0.0] * (DIM - len(head)))]


def _chunk(cid: str, text: str, *, source: str = "/m/a.md", **metadata: object) -> Chunk:
    return Chunk(id=cid, source=source, text=text, metadata={"file": source.rsplit("/", 1)[-1], **metadata})


@pytest.fixture
def store(tmp_path: Path) -> LiteStore:
    return LiteStore(tmp_path / "memory.db", dim=DIM)


def test_dense_is_an_exact_cosine_ranking(store: LiteStore) -> None:
    store.replace_sources(
        ["/m/a.md"],
        [_chunk("a1", "alpha"), _chunk("a2", "beta"), _chunk("a3", "gamma")],
        [_vec(1, 0), _vec(1, 1), _vec(0, 1)],
    )
    hits = store.query_dense(_vec(1, 0.2), k=3)
    assert [h.chunk.id for h in hits] == ["a1", "a2", "a3"]
    expected = sum(a * b for a, b in zip(_unit(_vec(1, 0.2)), _unit(_vec(1, 1)), strict=True))
    assert hits[1].score == pytest.approx(expected, abs=1e-5), "the score is not the cosine"
    assert store.top_cosine(_vec(0, 1)) == pytest.approx(1.0, abs=1e-6)
    assert store.cosines_for(["a3", "missing"], _vec(0, 1)) == {"a3": pytest.approx(1.0, abs=1e-6)}


def test_a_source_filter_matches_the_file_or_the_source(store: LiteStore) -> None:
    store.replace_sources(["/m/a.md"], [_chunk("a1", "alpha", source="/m/a.md")], [_vec(1, 0)])
    store.replace_sources(["/m/b.md"], [_chunk("b1", "beta", source="/m/b.md")], [_vec(1, 0)])
    assert [h.chunk.id for h in store.query_dense(_vec(1, 0), k=5, source="b.md")] == ["b1"]
    assert [h.chunk.id for h in store.query_dense(_vec(1, 0), k=5, source="/m/a.md")] == ["a1"]


def test_keywords_are_stemmed_stop_words_ignored_numbers_first(store: LiteStore) -> None:
    store.replace_sources(
        ["/m/a.md"],
        [
            _chunk("c1", "The cache is warmed every night.", numeric_values=["5"]),
            _chunk("c2", "Caching prices for 5 minutes keeps the caches warm and the caches fresh.", numeric_values=["5"]),
            _chunk("c3", "Many caches caches caches serve stale prices.", numeric_values=[]),
            _chunk("c4", "Deploy region is eu-west.", numeric_values=[]),
        ],
        [_vec(1, 0), _vec(0, 1), _vec(1, 1), _vec(0, 0, 1)],
    )
    hits = store.query_sparse("how long are caches kept, 5 minutes?", k=10)
    ids = [h.chunk.id for h in hits]
    assert "c4" not in ids
    assert set(ids) == {"c1", "c2", "c3"}, "the stemmed term did not match every form of it"
    assert ids.index("c2") < ids.index("c3") and ids.index("c1") < ids.index("c3"), "numeric matches were not ranked first"
    assert store.query_sparse("the and of", k=5) == [], "a query of stop words matched something"


def test_keywords_with_a_vector_score_by_cosine(store: LiteStore) -> None:
    store.replace_sources(["/m/a.md"], [_chunk("c1", "cache warm"), _chunk("c2", "cache cache cache")], [_vec(1, 0), _vec(0, 1)])
    hits = store.query_sparse("cache", k=5, vec=_vec(1, 0))
    assert [h.chunk.id for h in hits] == ["c2", "c1"], "the vector changed the keyword order"
    assert {h.chunk.id: round(h.score, 5) for h in hits} == {"c2": 0.0, "c1": 1.0}


def test_reindexing_keeps_first_indexed_at(store: LiteStore) -> None:
    store.replace_sources(["/m/a.md"], [_chunk("a1", "v1")], [_vec(1, 0)])
    first = store.query_dense(_vec(1, 0), k=1)[0]
    time.sleep(0.01)
    store.replace_sources(["/m/a.md"], [_chunk("a1", "v2")], [_vec(1, 0)])
    second = store.query_dense(_vec(1, 0), k=1)[0]
    assert second.chunk.text == "v2"
    assert second.first_indexed_at == first.first_indexed_at, "re-indexing moved the first write forward"
    assert second.indexed_at is not None and first.indexed_at is not None and second.indexed_at > first.indexed_at


def test_a_file_from_another_root_replaces_its_old_rows(store: LiteStore) -> None:
    old = Chunk(id="x1", source="/home/a/memory/note.md", text="old", metadata={"project": "acme", "file": "note.md"})
    new = Chunk(id="y1", source="C:/memstores/acme/note.md", text="new", metadata={"project": "acme", "file": "note.md"})
    store.replace_sources([old.source], [old], [_vec(1, 0)])
    store.replace_sources([new.source], [new], [_vec(1, 0)])
    assert [h.chunk.text for h in store.query_dense(_vec(1, 0), k=5)] == ["new"], "the old root's rows stayed beside the new ones"
    assert store.project_file_hashes("acme") == {"note.md": ""}


def test_scopes_filter_like_the_postgres_predicate(store: LiteStore) -> None:
    rows = [
        Chunk(id="r1", source="/m/top.md", text="t", metadata={"file": "top.md", "type": "Decision"}),
        Chunk(id="r2", source="/m/ops/run.md", text="t", metadata={"file": "ops/run.md", "type": "decision"}),
        Chunk(id="r3", source="/m/ops/deep/x.md", text="t", metadata={"file": "ops/deep/x.md"}),
        Chunk(id="r4", source="/m/opsx/y.md", text="t", metadata={"file": "opsx/y.md", "facet": "decision"}),
    ]
    store.upsert(rows, [_vec(1, 0)] * 4)

    def ids(scope: Scope) -> set[str]:
        return {h.chunk.id for h in store.query_dense(_vec(1, 0), k=10, scope=scope)}

    assert ids(Scope(folder="ops")) == {"r2", "r3"}, "a folder matched a sibling with the same prefix"
    assert ids(Scope(folder="/")) == {"r1"}
    assert ids(Scope(facet="DECISION")) == {"r1", "r2"}, "the facet arm did not read FACET_METADATA_KEY"
    assert ids(Scope(source_prefixes=("ops",))) == {"r2", "r3"}
    assert ids(Scope(source_prefixes=())) == set()


def test_one_file_holds_one_tenant_and_one_width(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    LiteStore(path, dim=DIM, tenant="memory").close()
    with pytest.raises(LiteStoreError, match="tenant"):
        LiteStore(path, dim=DIM, tenant="other")
    with pytest.raises(LiteStoreError, match="dim"):
        LiteStore(path, dim=DIM + 1, tenant="memory")
    reopened = LiteStore(path, dim=DIM, tenant="memory")
    with pytest.raises(LiteStoreError, match="wide"):
        reopened.replace_sources(["/m/a.md"], [_chunk("a1", "x")], [[1.0, 0.0]])


class HashEmbedder:
    """Deterministic bag-of-words vectors: same words, same direction; no model, no network."""

    dim = DIM
    name = "test-hash"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * DIM
            for word in text.lower().split():
                vec[int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIM] += 1.0
            out.append(vec if any(vec) else [1.0] + [0.0] * (DIM - 1))
        return out


def test_the_real_indexer_runs_on_a_lite_store(store: LiteStore, tmp_path: Path) -> None:
    from recall.index import Indexer

    root = tmp_path / "memos"
    root.mkdir()
    (root / "port.md").write_text("# Port\n\nThe API listens on port 9090.\n", encoding="utf-8")
    (root / "region.md").write_text("# Region\n\nThe service runs in eu-west.\n", encoding="utf-8")
    indexer = Indexer(store, HashEmbedder(), allow_prune=True, env={})
    first = indexer.index_path(root)
    assert first.files == 2 and store.count() >= 2
    again = indexer.index_path(root)
    assert again.files == 0 and again.skipped == 2, "an unchanged file was re-indexed"
    (root / "region.md").unlink()
    pruned = indexer.index_path(root)
    assert pruned.deleted == 1
    assert {h.chunk.metadata.get("file") for h in store.query_dense(HashEmbedder().embed(["port"])[0], k=10)} == {"port.md"}


# ---------------------------------------------------------------------- audit fixes, 2026-10-07


def _raw(path: Path, sql: str) -> None:
    """Run DDL through a second connection, as another process would."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(sql)
    finally:
        conn.close()


def test_a_failed_delete_leaves_cache_and_file_agreeing(store: LiteStore) -> None:
    store.replace_sources(["/m/a.md"], [_chunk("a1", "alpha")], [_vec(1, 0)])
    assert [h.chunk.id for h in store.query_dense(_vec(1, 0), k=5)] == ["a1"]
    # Another process with the corpus already cached: only the version tells it to reload.
    other = LiteStore(store.path, dim=DIM)
    assert [h.chunk.id for h in other.query_dense(_vec(1, 0), k=5)] == ["a1"]
    _raw(store.path, "CREATE TRIGGER fail_bump BEFORE UPDATE ON meta BEGIN SELECT RAISE(ROLLBACK, 'bump refused'); END;")
    with pytest.raises(sqlite3.Error, match="bump refused"):
        store.delete_sources(["/m/a.md"])
    cached = [h.chunk.id for h in store.query_dense(_vec(1, 0), k=5)]
    on_disk = [h.chunk.id for h in LiteStore(store.path, dim=DIM).query_dense(_vec(1, 0), k=5)]
    assert cached == on_disk, "the delete committed without moving the corpus version"
    assert [h.chunk.id for h in other.query_dense(_vec(1, 0), k=5)] == on_disk, (
        "the delete committed without moving the corpus version, so another store still serves it"
    )


def test_a_rollback_sqlite_already_made_surfaces_the_real_error(store: LiteStore) -> None:
    _raw(store.path, "CREATE TRIGGER fail_insert BEFORE INSERT ON chunks BEGIN SELECT RAISE(ROLLBACK, 'insert refused'); END;")
    with pytest.raises(sqlite3.Error) as caught:
        store.upsert([_chunk("a1", "alpha")], [_vec(1, 0)])
    assert "insert refused" in str(caught.value), "the rollback masked the real error"
    _raw(store.path, "DROP TRIGGER fail_insert;")
    assert store.upsert([_chunk("a1", "alpha")], [_vec(1, 0)]) == 1


def test_source_prefixes_match_as_the_postgres_regex_does(store: LiteStore) -> None:
    rows = [
        Chunk(id="n1", source="/m/n1", text="t", metadata={"file": "ops\n"}),
        Chunk(id="n2", source="/m/n2", text="t", metadata={"file": "ops/a\nb"}),
    ]
    store.upsert(rows, [_vec(1, 0)] * 2)
    hits = {h.chunk.id for h in store.query_dense(_vec(1, 0), k=5, scope=Scope(source_prefixes=("ops",)))}
    assert hits == {"n2"}, "a newline changed what an authorization prefix admits"


def test_matches_knows_every_scope_field(store: LiteStore) -> None:
    assert {f.name for f in dataclasses.fields(Scope)} == {
        "source",
        "folder",
        "facet",
        "source_prefixes",
        "security_policy_digest",
    }, "Scope gained a field that recall.lite.store._matches does not evaluate"
    rows = [
        Chunk(id="d1", source="/m/d1.md", text="t", metadata={"file": "d1.md", "security_policy_digest": "p1"}),
        Chunk(id="d2", source="/m/d2.md", text="t", metadata={"file": "d2.md", "security_policy_digest": "p2"}),
    ]
    store.upsert(rows, [_vec(1, 0)] * 2)
    hits = {h.chunk.id for h in store.query_dense(_vec(1, 0), k=5, scope=Scope(security_policy_digest="p1"))}
    assert hits == {"d1"}


def test_a_reused_rowid_never_aliases_a_deleted_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "memory.db"
    writer, reader = LiteStore(path, dim=DIM), LiteStore(path, dim=DIM)
    writer.replace_sources(["/m/a.md"], [_chunk("c1", "alpha", source="/m/a.md")], [_vec(1, 0)])
    writer.replace_sources(["/m/c.md"], [_chunk("c2", "charlie", source="/m/c.md")], [_vec(0, 1)])
    stale = reader._load()
    writer.delete_sources(["/m/c.md"])
    writer.replace_sources(["/m/d.md"], [_chunk("c3", "delta", source="/m/d.md")], [_vec(0, 1)])
    # The write lands between the reader's matrix read and its keyword read.
    monkeypatch.setattr(reader, "_load", lambda: stale)
    hits = reader.query_sparse("delta", k=5)
    assert all("delta" in h.chunk.text for h in hits), "a reused rowid resolved to a deleted chunk"


def test_nan_and_infinite_embeddings_are_refused(store: LiteStore) -> None:
    for bad in (float("nan"), float("inf")):
        with pytest.raises(LiteStoreError, match="NaN or infinite"):
            store.upsert([_chunk("a1", "alpha")], [_vec(bad)])
    assert store.count() == 0


def test_two_first_opens_cannot_both_claim_the_file(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    # Another process claims the file between this open's read of `meta` and its own insert.
    _raw(
        path,
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TRIGGER other_open BEFORE INSERT ON meta WHEN NEW.key = 'schema_version' BEGIN "
        "INSERT OR IGNORE INTO meta VALUES ('tenant', 'first'); "
        f"INSERT OR IGNORE INTO meta VALUES ('dim', '{DIM}'); END;",
    )
    with pytest.raises(LiteStoreError, match="tenant"):
        LiteStore(path, dim=DIM, tenant="second")


def test_the_project_file_delete_uses_its_index(store: LiteStore) -> None:
    from recall.lite.store import _DELETE_PROJECT_FILES

    plan = store._conn.execute("EXPLAIN QUERY PLAN " + _DELETE_PROJECT_FILES, ("acme", '["note.md"]')).fetchall()
    assert any("chunks_project_file" in str(row[-1]) for row in plan), f"every indexer flush scans the table: {plan}"


def test_k_must_be_positive(store: LiteStore) -> None:
    store.upsert([_chunk("a1", "alpha")], [_vec(1, 0)])
    with pytest.raises(ValueError, match="k must be a positive int"):
        store.query_dense(_vec(1, 0), k=0)
    with pytest.raises(ValueError, match="k must be a positive int"):
        store.query_sparse("alpha", k=0)


def test_a_table_other_than_chunks_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LiteStoreError, match="one table"):
        LiteStore(tmp_path / "memory.db", dim=DIM, table="shadow_chunks")


def test_copied_helpers_match_the_postgres_store() -> None:
    from recall import store as pg
    from recall.lite import store as lite

    assert lite._NUMERIC_TOKEN.pattern == pg._NUMERIC_TOKEN_RE.pattern
    for text in ["5", "5,5% of 12 and 5", "v1.2 -3 +4.0 3.50", "", "no numbers"]:
        assert lite._numeric_terms(text) == pg._numeric_query_terms(text), text
    batches = [
        [],
        [_chunk("a", "t", project="p"), _chunk("b", "t", source="/m/b.md", project="p")],
        [_chunk("a", "t", project="p"), _chunk("b", "t", project="q")],
        [_chunk("a", "t")],
        [_chunk("a", "t", project="p", file=""), _chunk("b", "t", project="p")],
    ]
    for batch in batches:
        assert lite._chunk_identity(batch) == pg._chunk_identity(batch), batch


def test_reads_come_back_in_postgres_order_without_the_matrix(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    writer = LiteStore(path, dim=DIM)
    writer.upsert([_chunk("b1", "beta", source="/m/s.md"), _chunk("a1", "alpha", source="/m/s.md")], [_vec(1, 0)] * 2)
    time.sleep(0.02)
    writer.upsert([_chunk("b1", "beta again", source="/m/s.md")], [_vec(1, 0)])
    assert [c.id for c in writer.chunks_for_source("/m/s.md")] == ["a1", "b1"], "not in ingestion order"
    reader = LiteStore(path, dim=DIM)
    assert [c.id for c in reader.iter_chunks(batch_size=1)] == ["a1", "b1"], "not in id order"
    assert reader._matrix is None, "iterating text built the whole embedding matrix"


class _CommitFailsOnce:
    """The store's connection, except that its next COMMIT fails, as a busy or full disk makes it."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.failed = False

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)

    def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
        if sql == "COMMIT" and not self.failed:
            self.failed = True
            raise sqlite3.OperationalError("database is locked")
        return self._conn.execute(sql, *args)


def test_a_failed_commit_does_not_leave_the_write_lock_held(store: LiteStore) -> None:
    real = store._conn
    store._conn = _CommitFailsOnce(real)  # type: ignore[assignment]
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        store.upsert([_chunk("a1", "alpha")], [_vec(1, 0)])
    store._conn = real
    assert not real.in_transaction, "the failed commit left the transaction open"
    assert LiteStore(store.path, dim=DIM).upsert([_chunk("b1", "beta")], [_vec(1, 0)]) == 1
    assert store.upsert([_chunk("a1", "alpha")], [_vec(1, 0)]) == 1


def test_iterating_sees_one_snapshot_and_every_id(store: LiteStore) -> None:
    store.upsert([_chunk("", "empty id"), _chunk("a1", "alpha"), _chunk("b1", "beta")], [_vec(1, 0)] * 3)
    stream = store.iter_chunks(batch_size=1)
    first = next(stream)
    store.delete_sources(["/m/a.md"])
    store.upsert([_chunk("c1", "charlie", source="/m/c.md")], [_vec(1, 0)])
    assert [first.id, *(c.id for c in stream)] == ["", "a1", "b1"], "the iteration mixed two corpus states"
    with pytest.raises(ValueError, match="batch_size"):
        next(store.iter_chunks(batch_size=0))
