"""Forgetting a memo keeps retired what it retired; withdrawing one gives it back, at the document layer.

Owner decision, 2026-10-22. `recall_forget` deleted a superseding memo's chunks, and with them its
`supersedes:` claim, so the resolver served what it had replaced as current again: a privacy forget
of "moved to Rome" brought Paris back, and forgetting the middle of a chain left two current
versions. Every delete now leaves a `SupersessionRecord` (`recall.supersession_records`) that
stands in for the memo while no memo by its name is indexed: a forget keeps its claims as a ghost,
a withdrawal voids them and splices whatever superseded it onto what it superseded.

Invariants, each a test below, on lite, the legacy PostgreSQL table and a GenerationStore alike:
- F1 forgetting the head of a line keeps its predecessor retired.
- F2 forgetting the middle of a chain keeps the chain: the last version still hides the first.
- F3 with two successors, forgetting the one valid now keeps the predecessor retired.
- F4 forgetting a successor changes nothing at any instant, its window included.
- W1 withdrawing the head of a line makes its predecessor current again.
- W2 withdrawing the middle of a chain splices it, never two current versions.
- W3 withdrawing a memo already forgotten gives back what the forget kept retired.
- R1 a record stands in for its memo only while no memo by that name is indexed, and forgetting a
  re-indexed memo that claims nothing replaces the record; a generation keeps a forgotten memo out.
- P1 the indexer's prune counts as a forget, and a restored file decides again.
- G1 a memo left out of a promoted generation counts as a forget, a rollback restores it.
- C1 a reader sees a withdrawal another process wrote, though it deleted no row.
- S1 to S3 the resolver's own rules (pure).

Red proof: in each docstring. "Red on master" means the test run against `e5d87caba` with only
this file added, failing in the named assertion.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from recall.lite import LiteStore
from recall.types import Chunk
from recall_mcp.lifecycle import forget_memory
from tests.conftest import TEST_DSN, _db_available, db_unreachable_reason

DIM = 64
NOW = datetime(2026, 10, 23, 12, tzinfo=UTC)
LAST_YEAR_BUT_ONE = datetime(2020, 6, 1, tzinfo=UTC)
EXPIRED = "2020-12-31"

#: A memo as these tests write it: its name, and the frontmatter it carries.
Memo = dict[str, str]


class Corpus:
    """One backend's store, indexed and removed from the same way for every backend."""

    def __init__(self, backend: str, tmp_path: Path) -> None:
        self.backend = backend
        self.tmp_path = tmp_path
        self.tenant = "srec-" + uuid.uuid4().hex[:10]
        self.table = "srec_" + uuid.uuid4().hex[:10]
        self.manager: Any = None
        self.store: Any
        if backend == "lite":
            self.store = LiteStore(tmp_path / "memory.db", dim=DIM)
        elif backend == "postgres":
            from recall.store import PgVectorStore

            self.store = PgVectorStore(TEST_DSN, dim=DIM, table=self.table, tenant=self.tenant)
            self.store.ensure_schema()
        else:
            from recall.generation_store import GenerationStore
            from recall.generations import GenerationManager

            self.manager = GenerationManager(TEST_DSN, self.tenant, actor="pytest", environment="test")
            self.store = GenerationStore(TEST_DSN, DIM, tenant=self.tenant)
        self.versions = 0

    def index(self, memos: dict[str, Memo]) -> None:
        """Index `memos`: an upsert on lite and PostgreSQL, a promoted generation of exactly
        these memos on a generation store."""
        if self.backend == "generation":
            from tests.test_stated_facts_postgres import promote

            self.versions += 1
            promote(self.manager, {name: _body(name, front) for name, front in memos.items()},
                    f"v{self.versions}")
            return
        chunks = [
            Chunk(f"chunk-{name}", f"/notes/{name}", f"text of {name}", {"file": name, **front})
            for name, front in memos.items()
        ]
        self.store.upsert(chunks, [[1.0] + [0.0] * (DIM - 1) for _ in chunks])

    def second_store(self) -> Any:
        """Another instance on the same corpus, as a second process would hold it."""
        if self.backend == "postgres":
            from recall.store import PgVectorStore

            return PgVectorStore(TEST_DSN, dim=DIM, table=self.table, tenant=self.tenant)
        from recall.generation_store import GenerationStore

        return GenerationStore(TEST_DSN, DIM, tenant=self.tenant)

    def forget(self, name: str) -> None:
        forget_memory(self.store, [name])

    def withdraw(self, name: str) -> None:
        """The store's withdrawal, by name, of a memo that is indexed or that a record stands in for."""
        sources = self.store.sources_for_identifiers([name]).get(name)
        if not sources:
            sources = self.store.recorded_sources_for_identifiers([name]).get(name)
        assert sources, f"{name} names nothing to withdraw"
        self.store.delete_sources(sources, withdraw=True)

    def successor(self, name: str, now: datetime | None = NOW) -> str | None:
        return self.store.supersession_graph().successor(name, now=now)

    def close(self) -> None:
        if self.backend == "lite":
            return
        import psycopg

        self.store.close()
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (self.tenant,))
            if self.backend == "postgres":
                conn.execute(f"DROP TABLE IF EXISTS {self.table}")
            tables = ("recall_source_tombstones", "recall_audit_events", "recall_ingest_jobs",
                      "recall_tenant_state", "recall_generations")
            for table in tables:
                conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (self.tenant,))  # noqa: S608


def _body(name: str, front: Memo) -> bytes:
    head = "".join(f"{key}: {value}\n" for key, value in front.items())
    return (f"---\n{head}---\n" if head else "").encode() + f"# {name}\n\nThe text of {name}.\n".encode()


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


@pytest.fixture(params=["lite", "postgres", "generation"])
def corpus(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Corpus]:
    if request.param != "lite" and not _db_available():
        pytest.skip(db_unreachable_reason())
    built = Corpus(request.param, tmp_path)
    try:
        yield built
    finally:
        built.close()


def test_f1_forgetting_the_head_keeps_its_predecessor_retired(corpus: Corpus) -> None:
    """Node tests/test_supersession_records.py::test_f1_forgetting_the_head_keeps_its_predecessor_retired.

    A privacy forget of "moved to Rome" must not bring Paris back as current.
    Red on master, every backend: fails on "the forgotten head revived its predecessor".
    """
    corpus.index({"paris.md": {}, "rome.md": {"supersedes": "paris.md"}})
    assert corpus.successor("paris.md") == "rome.md"

    corpus.forget("rome.md")
    assert corpus.successor("paris.md") == "rome.md", "the forgotten head revived its predecessor"


def test_f2_forgetting_the_middle_keeps_the_chain(corpus: Corpus) -> None:
    """Node tests/test_supersession_records.py::test_f2_forgetting_the_middle_keeps_the_chain.

    A -> B -> C, forget B: A stays retired, and through B's record C still hides it, so the line
    keeps one current version and A's successor is the one a reader can open.
    Red on master, every backend: fails on "two current versions after forgetting the middle".
    """
    corpus.index({"a.md": {}, "b.md": {"supersedes": "a.md"}, "c.md": {"supersedes": "b.md"}})
    corpus.forget("b.md")
    assert corpus.successor("a.md") == "c.md", "two current versions after forgetting the middle"
    assert corpus.successor("c.md") is None


def test_f3_forgetting_one_of_two_successors_keeps_the_predecessor_retired(corpus: Corpus) -> None:
    """Node tests/test_supersession_records.py::test_f3_forgetting_one_of_two_successors_keeps_the_predecessor_retired.

    Fan-in: two memos supersede one, and only the forgotten one is valid now (the other expired in
    2020), so nothing else keeps the predecessor retired once the forgotten claim is gone.
    Red on master, every backend: fails on "forgetting one of two successors revived the predecessor".
    """
    corpus.index({
        "a.md": {},
        "b1.md": {"supersedes": "a.md"},
        "b2.md": {"supersedes": "a.md", "valid_until": EXPIRED},
    })
    corpus.forget("b1.md")
    assert corpus.successor("a.md") == "b1.md", "forgetting one of two successors revived the predecessor"


def test_f4_forgetting_a_successor_changes_nothing_at_any_instant(corpus: Corpus) -> None:
    """Node tests/test_supersession_records.py::test_f4_forgetting_a_successor_changes_nothing_at_any_instant.

    A successor that expired in 2020 hid its predecessor then and does not now. Forgetting it keeps
    exactly that: the record keeps the window, so forget never makes anything current or retired.
    Red on master, every backend: fails on "the 2020 supersession was lost". Red, mutation of
    `_resolve_rows` (recall/supersession.py): a forgotten record's rows read without their window
    (`windowed = False`): fails on "forget retired a predecessor that was current".
    """
    corpus.index({"a.md": {}, "b.md": {"supersedes": "a.md", "valid_until": EXPIRED}})
    assert corpus.successor("a.md") is None

    corpus.forget("b.md")
    assert corpus.successor("a.md") is None, "forget retired a predecessor that was current"
    assert corpus.successor("a.md", LAST_YEAR_BUT_ONE) == "b.md", "the 2020 supersession was lost"


def test_w1_withdrawing_the_head_brings_back_its_predecessor(corpus: Corpus) -> None:
    """Node tests/test_supersession_records.py::test_w1_withdrawing_the_head_brings_back_its_predecessor.

    "This memo was a mistake": what it replaced is current again.
    Red, mutation of `record_kind` (recall/supersession_records.py) answering FORGOTTEN_RECORD
    for a withdrawal: fails on "the withdrawn head still retires its predecessor".
    """
    corpus.index({"paris.md": {}, "rome.md": {"supersedes": "paris.md"}})
    corpus.withdraw("rome.md")
    assert corpus.successor("paris.md") is None, "the withdrawn head still retires its predecessor"
    assert not corpus.store.sources_for_identifiers(["rome.md"])


def test_w2_withdrawing_the_middle_splices_the_chain(corpus: Corpus) -> None:
    """Node tests/test_supersession_records.py::test_w2_withdrawing_the_middle_splices_the_chain.

    A -> B -> C, withdraw B: A stays retired and C becomes its replacement; never two current.
    Red, mutation of `_splice_withdrawn` (recall/supersession.py) returning at once: fails on
    "withdrawing the middle left two current versions".
    """
    corpus.index({"a.md": {}, "b.md": {"supersedes": "a.md"}, "c.md": {"supersedes": "b.md"}})
    corpus.withdraw("b.md")
    assert corpus.successor("a.md") == "c.md", "withdrawing the middle left two current versions"
    assert corpus.successor("c.md") is None
    assert corpus.store.supersession_graph().successor_path("a.md") == ("c.md",)


def test_w3_withdrawing_a_forgotten_memo_gives_back_what_it_retired(corpus: Corpus) -> None:
    """Node tests/test_supersession_records.py::test_w3_withdrawing_a_forgotten_memo_gives_back_what_it_retired.

    A memo forgotten (or pruned) earlier turns out to have been a mistake: withdrawing it by name
    takes over its record. Red, mutation of `pending_records` (recall/supersession_records.py)
    ignoring `standing` for a source with no rows: fails on "the forget record still stands".
    """
    corpus.index({"paris.md": {}, "rome.md": {"supersedes": "paris.md"}})
    corpus.forget("rome.md")
    corpus.withdraw("rome.md")
    assert corpus.successor("paris.md") is None, "the forget record still stands"


def test_r1_a_record_stands_only_while_no_memo_by_its_name_is_indexed(corpus: Corpus) -> None:
    """Node tests/test_supersession_records.py::test_r1_a_record_stands_only_while_no_memo_by_its_name_is_indexed.

    Forget B, re-index a memo named B that no longer claims A: the indexed memo decides, so A is
    current. Forget B again: its new record (no claim) replaces the old one, so A stays current. A
    generation store never re-indexes a forgotten name (its tombstone keeps it out of every build),
    so there A stays retired.
    Red on master, every backend: fails on "forget revived the predecessor". Red, mutation of
    `_resolve_rows`: every record standing whether or not its memo is indexed: fails on "a
    re-indexed memo did not decide". Red, mutation of `pending_records`: no record for a memo
    that claims nothing: fails on "the old record came back".
    """
    corpus.index({"a.md": {}, "b.md": {"supersedes": "a.md"}})
    corpus.forget("b.md")
    assert corpus.successor("a.md") == "b.md", "forget revived the predecessor"

    corpus.index({"a.md": {}, "b.md": {}})
    if corpus.backend == "generation":
        assert not corpus.store.sources_for_identifiers(["b.md"]), "a forgotten memo was rebuilt"
        assert corpus.successor("a.md") == "b.md"
        return
    assert corpus.successor("a.md") is None, "a re-indexed memo did not decide"
    corpus.forget("b.md")
    assert corpus.successor("a.md") is None, "the old record came back"


@pytest.mark.parametrize("backend", ["lite", "postgres"])
def test_p1_the_indexer_prune_counts_as_a_forget(backend: str, tmp_path: Path) -> None:
    """Node tests/test_supersession_records.py::test_p1_the_indexer_prune_counts_as_a_forget.

    A memo deleted from disk is pruned by the next index (owner decision, 2026-10-23: a forget,
    so deleting the Rome memo by hand keeps Paris retired). Restoring the file brings the
    memo back, and it decides again with its own claims.
    Red on master, both backends: fails on "the prune revived the predecessor".
    """
    from recall.embeddings import HashingEmbedder
    from recall.index import Indexer

    if backend == "postgres" and not _db_available():
        pytest.skip(db_unreachable_reason())
    corpus = Corpus(backend, tmp_path)
    try:
        notes = tmp_path / "notes"
        notes.mkdir()
        (notes / "paris.md").write_text("# Paris\n\nI live in Paris.\n", encoding="utf-8")
        rome = notes / "rome.md"
        rome.write_text("---\nsupersedes: paris.md\n---\n# Rome\n\nI moved to Rome.\n", encoding="utf-8")
        indexer = Indexer(corpus.store, HashingEmbedder(dim=DIM), env={})
        indexer.index_path(notes)
        assert corpus.successor("paris.md") == "rome.md"

        rome.unlink()
        indexer.index_path(notes)
        assert not corpus.store.sources_for_identifiers(["rome.md"])
        assert corpus.successor("paris.md") == "rome.md", "the prune revived the predecessor"

        rome.write_text("# Rome\n\nA trip to Rome, no move.\n", encoding="utf-8")
        indexer.index_path(notes)
        assert corpus.successor("paris.md") is None, "the restored memo did not decide"
    finally:
        corpus.close()


def test_g1_a_memo_left_out_of_a_promoted_generation_counts_as_a_forget(tmp_path: Path) -> None:
    """Node tests/test_supersession_records.py::test_g1_a_memo_left_out_of_a_promoted_generation_counts_as_a_forget.

    The production form of the prune: a memo deleted from disk is a source the next promoted
    generation lacks. A rollback restores the generation that holds it, and a later generation
    whose copy claims nothing decides with that.
    Red on master: fails on "a memo left out of the promoted generation revived its predecessor".
    """
    if not _db_available():
        pytest.skip(db_unreachable_reason())
    corpus = Corpus("generation", tmp_path)
    try:
        corpus.index({"paris.md": {}, "rome.md": {"supersedes": "paris.md"}})
        corpus.index({"paris.md": {}})
        assert corpus.successor("paris.md") == "rome.md", (
            "a memo left out of the promoted generation revived its predecessor"
        )
        assert corpus.manager.last_omitted_records == 1

        corpus.manager.rollback()
        assert corpus.store.sources_for_identifiers(["rome.md"])
        assert corpus.successor("paris.md") == "rome.md"

        corpus.index({"paris.md": {}, "rome.md": {}})
        assert corpus.successor("paris.md") is None
    finally:
        corpus.close()


@pytest.mark.parametrize("backend", ["postgres", "generation"])
def test_c1_a_reader_sees_a_withdrawal_written_elsewhere(backend: str, tmp_path: Path) -> None:
    """Node tests/test_supersession_records.py::test_c1_a_reader_sees_a_withdrawal_written_elsewhere.

    A long-lived server holds one store and caches its graph. Withdrawing a memo already forgotten
    writes a record and deletes no row, so a cache keyed on the chunks alone would keep the forget.
    Red, mutation of `PgVectorStore.supersession_graph` (recall/store.py) dropping the record
    count from its fingerprint, and of `GenerationStore.supersession_graph`
    (recall/generation_store.py) dropping it from its key: fails on "the reader kept the
    forget".
    """
    if not _db_available():
        pytest.skip(db_unreachable_reason())
    writer = Corpus(backend, tmp_path)
    try:
        writer.index({"paris.md": {}, "rome.md": {"supersedes": "paris.md"}})
        writer.forget("rome.md")
        reader = writer.second_store()
        try:
            assert reader.supersession_graph().successor("paris.md") == "rome.md"
            writer.withdraw("rome.md")
            assert reader.supersession_graph().successor("paris.md") is None, "the reader kept the forget"
        finally:
            reader.close()
    finally:
        writer.close()


# --------------------------------------------------------------------------- the resolver alone


def _rows(*memos: tuple[str, str | None]) -> list[tuple[object, ...]]:
    return [(file, target, None, f"/n/{file}", None, None) for file, target in memos]


def _record(kind: str, file: str, *targets: str, source: str | None = None) -> Any:
    # Imported here, not at module scope, so the store tests above run (and fail in their own
    # assertions) against a tree that predates these names.
    from recall.supersession import SupersessionRecord

    return SupersessionRecord(kind, file, source or f"/n/{file}", tuple((t, None, None, None) for t in targets))


def _resolve(rows: list[tuple[object, ...]], records: list[Any]) -> Any:
    from recall.supersession import resolve_supersession_graph

    return resolve_supersession_graph(rows, records)


FORGOTTEN_RECORD = "supersession_forgotten"
WITHDRAWN_RECORD = "supersession_withdrawn"


def test_s1_two_withdrawn_links_splice_through_both() -> None:
    """Node tests/test_supersession_records.py::test_s1_two_withdrawn_links_splice_through_both.

    A -> B -> C -> D with B and C withdrawn: D hides A. Red, mutation of `_splice_withdrawn`
    (recall/supersession.py) not following a withdrawn target (`continue` in place of extending
    the stack): fails on "the splice stopped at a withdrawn link".
    """
    graph = _resolve(
        _rows(("a.md", None), ("d.md", "c.md")),
        [_record(WITHDRAWN_RECORD, "b.md", "a.md"), _record(WITHDRAWN_RECORD, "c.md", "b.md")],
    )
    assert graph.successor("a.md") == "d.md", "the splice stopped at a withdrawn link"
    assert graph.successor("d.md") is None


def test_s2_an_indexed_memo_wins_a_reference_over_a_ghost() -> None:
    """Node tests/test_supersession_records.py::test_s2_an_indexed_memo_wins_a_reference_over_a_ghost.

    A memo moved from notes/ to archive/ leaves a record under its old path; `supersedes: b`
    from another memo must keep naming the moved file, not turn ambiguous (which fails a read
    closed). Red, mutation of `ReferenceIndex._indexed_first` (recall/supersession.py) returning
    its keys unchanged: fails on "a ghost made the reference ambiguous".
    """
    graph = _resolve(
        _rows(("a.md", None), ("archive/b.md", "a.md"), ("c.md", "b")),
        [_record(FORGOTTEN_RECORD, "notes/b.md", "a.md")],
    )
    assert not graph.unresolved, "a ghost made the reference ambiguous"
    assert graph.successor("a.md") == "c.md"


def test_s3_a_record_is_dormant_while_its_file_is_indexed_under_another_root() -> None:
    """Node tests/test_supersession_records.py::test_s3_a_record_is_dormant_while_its_file_is_indexed_under_another_root.

    The corpus re-indexed under a new root keeps every root-relative name: a record from the old
    root must not stand beside the memo it describes. Red, mutation of `_resolve_rows`
    (recall/supersession.py) checking only the record's source: fails on "a record stood beside
    its own re-indexed memo".
    """
    graph = _resolve(
        [("a.md", None, None, "/new/a.md", None, None), ("b.md", None, None, "/new/b.md", None, None)],
        [_record(FORGOTTEN_RECORD, "b.md", "a.md", source="/old/b.md")],
    )
    assert graph.successor("a.md") is None, "a record stood beside its own re-indexed memo"
