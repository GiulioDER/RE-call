"""Per-query work done once, not repeated: P4 of the core-module optimization.

Invariants:

1. One `trusted_search` embeds each distinct query text once, however many expansion searches
   it runs (`recall.trust._RequestQueryMemo`). Document, structural and successor expansion
   re-search with the same text, and each call used to embed it again.
2. `register_evidence_cards` writes on the serving store's own tenant-bound connection
   (`recall.provenance_cards._put_cards`), not on a new psycopg connection per call.
3. `trusted_related(relation="supersession")` on a store without a bounded path fetches the
   seed by id, and scans the corpus only when the seed has supersession edges at all.

Red proof, recorded 2026-09-23 against this branch's base (`763bad2e`, P3) by restoring the
pre-change `recall/trust.py`, `recall/provenance_cards.py`, `recall_mcp/provenance.py` and
`recall/related.py`:

* `test_a_search_with_document_expansion_embeds_its_query_once` failed with `assert 3 == 1`.
* `test_evidence_cards_are_written_on_the_store_connection` failed at `register_evidence_cards`
  with `AssertionError: a new database connection was opened for the evidence cards`.
* `test_supersession_without_edges_reads_no_corpus` failed with `assert 2 == 0`.
* `test_supersession_with_an_edge_scans_once_and_finds_the_successor` failed with
  `assert 2 == 1`.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime

from recall.calibration import Calibration
from recall.related import trusted_related
from recall.retriever import DocumentExpansionPolicy
from recall.trust import trusted_search
from recall.trust_policy import TrustPolicy
from recall.types import Chunk, ScoredChunk


# --- 1. one embedding per query text -----------------------------------------------------------


class _CountingEmbedder:
    dim = 2
    name = "counting"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


class _SearchStore:
    tenant = "acme"

    def __init__(self) -> None:
        self.dense_calls = 0

    def query_dense(self, vector, k, source=None, scope=None):  # type: ignore[no-untyped-def]
        self.dense_calls += 1
        return [
            ScoredChunk(Chunk(f"c{i}", f"s{i % 2}.md", f"text {i}", {"file": f"s{i % 2}.md"}), 0.9)
            for i in range(4)
        ][:k]

    def query_sparse(self, query, k, source=None, vec=None, scope=None):  # type: ignore[no-untyped-def]
        return []

    def newest_indexed_at(self):  # type: ignore[no-untyped-def]
        return None

    def supersession(self):  # type: ignore[no-untyped-def]
        return {}, frozenset()


def test_a_search_with_document_expansion_embeds_its_query_once() -> None:
    embedder = _CountingEmbedder()
    store = _SearchStore()

    trusted_search(
        store,  # type: ignore[arg-type]
        embedder,
        "what changed",
        k=4,
        calibration=Calibration(embedder="counting", threshold=0.1, scale=0.1),
        policy=TrustPolicy.development(),
        document_expansion=DocumentExpansionPolicy(
            enabled=True, max_sources=2, chunks_per_source=2, relational_query_only=False
        ),
    )

    assert store.dense_calls > 1  # the expansion searches really ran
    assert len(embedder.queries) == 1


# --- 2. evidence cards on the store's connection -----------------------------------------------


class _CardConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    @contextmanager
    def transaction(self):  # type: ignore[no-untyped-def]
        yield None

    def execute(self, sql, params=None):  # type: ignore[no-untyped-def]
        self.statements.append(str(sql))

        class _Row:
            def fetchone(self):  # type: ignore[no-untyped-def]
                return ("inserted",)

        return _Row()


def test_evidence_cards_are_written_on_the_store_connection(monkeypatch) -> None:
    import psycopg

    from recall.types import EvidenceCard
    from recall_mcp.provenance import register_evidence_cards

    def refuse(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("a new database connection was opened for the evidence cards")

    monkeypatch.setattr(psycopg, "connect", refuse)
    connection = _CardConnection()

    class _Store:
        tenant = "acme"
        dsn = "postgresql://unused/never-connected"

        def _with_retry(self, op):  # type: ignore[no-untyped-def]
            return op(connection)

    card = EvidenceCard(
        card_id="",
        chunk_id="chunk-1",
        source="notes.md",
        source_digest="digest",
        valid_from=None,
        valid_until=None,
        first_indexed_at=datetime(2026, 8, 31, tzinfo=UTC),
        indexed_at=datetime(2026, 9, 1, tzinfo=UTC),
        tenant_id="acme",
        generation_id="gen-1",
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_id="cal-1",
        calibration_status="certified",
        trust_state="trusted",
        verdict="ok",
        confidence=0.99,
        rank=1,
    )

    register_evidence_cards([card], store=_Store())  # type: ignore[arg-type]

    assert any("INSERT INTO recall_evidence_cards" in sql for sql in connection.statements)


# --- 3. supersession without a corpus scan -----------------------------------------------------


class _RelatedStore:
    tenant = "tenant-a"
    generation_id = "gen-1"

    def __init__(self, edges: dict[str, str]) -> None:
        self.chunks = [
            Chunk("seed", "old.md", "seed", {"file": "old.md", "ord": 1}),
            Chunk("successor", "new.md", "successor", {"file": "new.md", "ord": 1}),
            Chunk("other", "other.md", "other", {"file": "other.md", "ord": 1}),
        ]
        self.edges = edges
        self.scans = 0

    def iter_chunks(self, batch_size=1000):  # type: ignore[no-untyped-def]
        self.scans += 1
        return iter(self.chunks)

    def chunks_by_ids(self, chunk_ids):  # type: ignore[no-untyped-def]
        return {c.id: c for c in self.chunks if c.id in set(chunk_ids)}

    def supersession_all(self):  # type: ignore[no-untyped-def]
        return dict(self.edges), frozenset(), {}


def _related(store: _RelatedStore):  # type: ignore[no-untyped-def]
    return trusted_related(
        store,  # type: ignore[arg-type]
        "seed",
        relation="supersession",
        calibration=Calibration(embedder="fixture", threshold=0.5),
        policy=TrustPolicy.development(),
        now=datetime(2026, 8, 24, tzinfo=UTC),
    )


def test_supersession_without_edges_reads_no_corpus() -> None:
    store = _RelatedStore(edges={})

    result = _related(store)

    assert [item.chunk.id for item in result.items] == []
    assert store.scans == 0


def test_supersession_with_an_edge_scans_once_and_finds_the_successor() -> None:
    store = _RelatedStore(edges={"old.md": "new.md"})

    result = _related(store)

    assert [item.chunk.id for item in result.items] == ["successor"]
    assert store.scans == 1
