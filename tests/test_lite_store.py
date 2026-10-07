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
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

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
        Chunk(id="r1", source="/m/top.md", text="t", metadata={"file": "top.md", "facet": "Decision"}),
        Chunk(id="r2", source="/m/ops/run.md", text="t", metadata={"file": "ops/run.md", "facet": "decision"}),
        Chunk(id="r3", source="/m/ops/deep/x.md", text="t", metadata={"file": "ops/deep/x.md"}),
        Chunk(id="r4", source="/m/opsx/y.md", text="t", metadata={"file": "opsx/y.md"}),
    ]
    store.upsert(rows, [_vec(1, 0)] * 4)

    def ids(scope: Scope) -> set[str]:
        return {h.chunk.id for h in store.query_dense(_vec(1, 0), k=10, scope=scope)}

    assert ids(Scope(folder="ops")) == {"r2", "r3"}, "a folder matched a sibling with the same prefix"
    assert ids(Scope(folder="/")) == {"r1"}
    assert ids(Scope(facet="DECISION")) == {"r1", "r2"}
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
