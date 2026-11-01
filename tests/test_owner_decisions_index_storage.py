"""Regression tests for the owner's 2026-11-01 decisions on the indexing and storage hunt residue.

Each test names the item it pins, the baseline or mutation it was proved red against, and the
failure that produced. The baseline is `d56661c71` (master before this change) unless a mutation
is named. The database tests are in tests/test_generations.py and tests/test_schema_migrations.py;
none here needs a database, a network or a model.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.semantic_graph import SemanticRelation, build_semantic_graph
from recall.semantic_graph._store import write_semantic_graph
from recall.types import Chunk


class _Store:
    def __init__(self) -> None:
        self.chunks: list[object] = []

    def source_content_hashes(self) -> dict[str, str]:
        return {}

    def replace_sources(self, sources, chunks, embeddings):  # type: ignore[no-untyped-def]
        self.chunks.extend(chunks)
        return len(chunks)

    def analyze_if_stale(self, modified):  # type: ignore[no-untyped-def]
        return True

    def delete_sources(self, sources):  # type: ignore[no-untyped-def]
        return 0


def test_a_hard_linked_source_is_indexed_with_a_warning_naming_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Item 2: a file with a second hard link is indexed, not refused, and the log names it.

    `a.md` and `b.md` are one file; `c.md` has one link and is the negative control. Red against
    `d56661c71` (no link count is read): node
    tests/test_owner_decisions_index_storage.py::test_a_hard_linked_source_is_indexed_with_a_warning_naming_it
    fails on "no warning named the hard-linked file".
    """
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("one file under two names", encoding="utf-8")
    try:
        os.link(root / "a.md", root / "b.md")
    except OSError:
        pytest.skip("this filesystem cannot create a hard link")
    (root / "c.md").write_text("a file with one name", encoding="utf-8")
    store = _Store()
    with caplog.at_level(logging.WARNING, logger="recall.index"):
        Indexer(store, HashingEmbedder(dim=64)).index_path(root)
    assert {Path(chunk.source).name for chunk in store.chunks} == {"a.md", "b.md", "c.md"}
    warned = [r.getMessage() for r in caplog.records if "hard link" in r.getMessage()]
    assert any("a.md" in m for m in warned) and any("b.md" in m for m in warned), (
        "no warning named the hard-linked file"
    )
    assert not any("c.md" in m for m in warned), "a file with one link was reported"


def test_a_relation_without_evidence_is_refused() -> None:
    """Item 3 (DEF-11): forget's broad orphan delete relies on no relation existing without
    evidence, so the model must refuse one.

    Red by mutation of `SemanticRelation.__post_init__` (the empty-evidence `raise` removed):
    node tests/test_owner_decisions_index_storage.py::test_a_relation_without_evidence_is_refused
    fails on "DID NOT RAISE".
    """
    with pytest.raises(ValueError, match="at least one evidence chunk"):
        SemanticRelation(
            id="rel", tenant_id="t", generation_id="g", subject_id="s", object_id="o",
            relation="supports", evidence_chunk_ids=(), extraction_method="authored",
            confidence=1.0,
        )


def test_every_written_relation_has_its_evidence_written_with_it() -> None:
    """Item 3 (DEF-11): the other half of the invariant, at the writer.

    Red by mutation of `write_semantic_graph` (the evidence rows built from
    `relation.evidence_chunk_ids[:0]`): node
    tests/test_owner_decisions_index_storage.py::test_every_written_relation_has_its_evidence_written_with_it
    fails on "a relation was written without evidence".
    """
    graph = build_semantic_graph(
        [
            Chunk(
                "c1", "decision.md", "",
                {
                    "file": "decision.md",
                    "recall_graph": {
                        "entities": [
                            {"name": "Rate Limits", "kind": "decision"},
                            {"name": "Gateway", "kind": "service"},
                        ],
                        "relations": [
                            {"relation": "supports", "subject": "Rate Limits", "object": "Gateway"}
                        ],
                    },
                },
            )
        ],
        tenant_id="t", generation_id="g", pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
    )
    assert graph.relations, "the fixture produced no relation, so nothing would be checked"
    inserts: dict[str, list[tuple[Any, ...]]] = {}

    class _Cursor:
        def __enter__(self) -> "_Cursor":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def executemany(self, query: str, rows: list[tuple[Any, ...]]) -> None:
            inserts[query.split()[2]] = list(rows)

    class _Conn:
        def execute(self, query: str, params: object = None) -> None:
            return None

        def cursor(self) -> _Cursor:
            return _Cursor()

    write_semantic_graph(_Conn(), graph)
    relations = {row[2] for row in inserts["recall_graph_relations_v1"]}
    evidenced = {row[2] for row in inserts["recall_graph_relation_evidence_v1"]}
    assert relations <= evidenced, "a relation was written without evidence"


@pytest.mark.parametrize(
    ("version", "refused"),
    [("0.7.4", True), ("0.5", True), ("0.8.0", False), ("0.8.4", False), ("0.10.0", False)],
)
def test_the_pgvector_floor_is_0_8_0(version: str, refused: bool) -> None:
    """Item 6: one rule, read by both the migration runner and `recall doctor`.

    Red by mutation of `PGVECTOR_MIN_VERSION` to `(0, 7, 0)`: node
    tests/test_owner_decisions_index_storage.py::test_the_pgvector_floor_is_0_8_0[0.7.4-True]
    fails on "0.7.4 was accepted".
    """
    from recall.schema import pgvector_version_problem

    problem = pgvector_version_problem(version)
    if refused:
        assert problem is not None and "0.8.0" in problem, f"{version} was accepted"
    else:
        assert problem is None, f"{version} was refused: {problem}"


class _VersionConn:
    """Answers the two queries `_database_checks` asks, and nothing else."""

    def __init__(self, extversion: str) -> None:
        self.extversion = extversion

    def execute(self, query: str, params: object = None) -> SimpleNamespace:
        if "version()" in query:
            return SimpleNamespace(fetchone=lambda: ("PostgreSQL 18.0, compiled",))
        if "pg_extension" in query:
            return SimpleNamespace(fetchone=lambda: (self.extversion,))
        raise AssertionError(f"unexpected query: {query}")


def test_doctor_fails_a_pgvector_below_the_floor() -> None:
    """Item 6: `recall doctor` reports `fail` for pgvector 0.7.4 with the runner's message, and
    `ok` for 0.8.0 (the negative control).

    Red by mutation of `_database_checks` (the floor branch's condition forced to None, which is
    the baseline's behaviour: any installed version is `ok`; the baseline itself lacks the shared
    helper, so it fails on import, which is not red proof): node
    tests/test_owner_decisions_index_storage.py::test_doctor_fails_a_pgvector_below_the_floor
    fails on "doctor passed pgvector 0.7.4".
    """
    from recall import doctor
    from recall.schema import pgvector_version_problem

    def pgvector(extversion: str) -> Any:
        checks = list(doctor._database_checks("postgresql://u@h/db", _VersionConn(extversion)))
        return next(check for check in checks if check.name == "pgvector")

    old = pgvector("0.7.4")
    assert old.status == "fail", "doctor passed pgvector 0.7.4"
    assert old.detail == pgvector_version_problem("0.7.4")
    assert old.fix
    assert pgvector("0.8.0").status == "ok"


def test_doctor_names_the_cause_of_a_checksum_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item 14: a checksum mismatch is a changed migration file or a hand-edited database, and
    another recall version is not the repair.

    Red against `d56661c71` (the repair said "upgrade or downgrade recall"): node
    tests/test_owner_decisions_index_storage.py::test_doctor_names_the_cause_of_a_checksum_mismatch
    fails on "the repair does not name the cause".
    """
    from recall import doctor, schema

    def mismatch(*args: object, **kwargs: object) -> None:
        raise schema.MigrationChecksumMismatch("applied migration 0001_v08_baseline.sql checksum drift")

    monkeypatch.setattr(schema, "schema_status", mismatch)
    check = doctor._migration_check("postgresql://u@h/db", table="chunks", dim=64)
    assert check.status == "fail"
    assert check.fix is not None
    assert "edited by hand" in check.fix and "migration file changed" in check.fix, (
        "the repair does not name the cause"
    )
    assert "upgrade or downgrade" not in check.fix


class _Item(SimpleNamespace):
    pass


def _openai_embedder(monkeypatch: pytest.MonkeyPatch, respond: Any) -> Any:
    """An `OpenAICompatEmbedder` over a fake SDK: `respond(batch)` returns the response items."""
    from recall.embeddings import OpenAICompatEmbedder

    class _FakeOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.embeddings = SimpleNamespace(
                create=lambda **request: SimpleNamespace(data=respond(list(request["input"])))
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    return OpenAICompatEmbedder(api_key="test", base_url="http://127.0.0.1:9/v1")


def _vector(text: str) -> list[float]:
    return [float(len(text)), 1.0]


def test_openai_compat_orders_vectors_by_their_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item 7 (DEF-5): a response in another order is put back in input order by `index`.

    Red against `d56661c71` (response order taken as input order): node
    tests/test_owner_decisions_index_storage.py::test_openai_compat_orders_vectors_by_their_index
    fails on "vectors were paired by response position".
    """
    embedder = _openai_embedder(
        monkeypatch,
        lambda batch: list(reversed(
            [_Item(index=i, embedding=_vector(text)) for i, text in enumerate(batch)]
        )),
    )
    texts = ["a", "bb", "ccc"]
    assert embedder.embed(texts) == [_vector(t) for t in texts], (
        "vectors were paired by response position"
    )


def test_openai_compat_keeps_response_order_without_an_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 7 (DEF-5): a provider that sends no `index` keeps the previous behaviour.

    A nonbehavioural control for the test above: it pins that the sort is conditional, and
    was green before the change too, so red proof does not apply to it.
    """
    embedder = _openai_embedder(
        monkeypatch, lambda batch: [_Item(embedding=_vector(text)) for text in batch]
    )
    assert embedder.embed(["a", "bb"]) == [_vector("a"), _vector("bb")]


def test_openai_compat_refuses_a_short_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item 7 (DEF-5): one vector missing from a batch is refused, even in the width probe.

    The width probe calls `_embed_one_batch` directly, outside `batched_embed`'s own count
    check, so the method itself must refuse. Red against `d56661c71`: node
    tests/test_owner_decisions_index_storage.py::test_openai_compat_refuses_a_short_batch
    fails on "DID NOT RAISE".
    """
    embedder = _openai_embedder(
        monkeypatch, lambda batch: [_Item(embedding=_vector(text)) for text in batch]
    )
    embedder._client.embeddings.create = lambda **request: SimpleNamespace(
        data=[_Item(embedding=_vector("x"))]
    )
    with pytest.raises(RuntimeError, match="1 embeddings for 2 texts"):
        embedder._embed_one_batch(["a", "bb"])


def test_openai_compat_refuses_indices_that_do_not_cover_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 7 (DEF-5): a repeated index cannot be sorted into input order, so it is refused.

    Red by mutation of `_embed_one_batch` (the permutation check removed, so the sort runs on
    `[0, 0]`): node
    tests/test_owner_decisions_index_storage.py::test_openai_compat_refuses_indices_that_do_not_cover_the_batch
    fails on "DID NOT RAISE".
    """
    embedder = _openai_embedder(
        monkeypatch, lambda batch: [_Item(index=0, embedding=_vector(text)) for text in batch]
    )
    with pytest.raises(RuntimeError, match="indices"):
        embedder._embed_one_batch(["a", "bb"])


class _MigrationConn:
    """The migration runner's connection, up to the first statement after the pgvector check.

    Any query past the two it is allowed to answer is recorded and ends the run, so a runner that
    goes on to take the lock or migrate is seen doing so.
    """

    def __init__(self, extversion: str | None, default_version: str | None) -> None:
        self.extversion = extversion
        self.default_version = default_version
        self.queries: list[str] = []

    def __enter__(self) -> "_MigrationConn":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, query: str, params: object = None) -> SimpleNamespace:
        self.queries.append(query)
        if "has_schema_privilege" in query:
            return SimpleNamespace(fetchone=lambda: (True,))
        if "pg_extension" in query:
            row = None if self.extversion is None else (self.extversion,)
            return SimpleNamespace(fetchone=lambda: row)
        if "pg_available_extensions" in query:
            row = None if self.default_version is None else (self.default_version,)
            return SimpleNamespace(fetchone=lambda: row)
        raise RuntimeError("the runner went past the pgvector check")


@pytest.mark.parametrize(
    ("extversion", "default_version"), [("0.7.4", None), (None, "0.7.4")],
    ids=["installed", "about-to-be-installed"],
)
def test_the_migration_runner_refuses_an_old_pgvector_before_migrating(
    monkeypatch: pytest.MonkeyPatch, extversion: str | None, default_version: str | None
) -> None:
    """Item 6: an installed pgvector 0.7.4, or one that migration 0001 would install, is refused
    before the lock is taken or any migration runs.

    Red against `d56661c71` (no version is read): both nodes of
    tests/test_owner_decisions_index_storage.py::test_the_migration_runner_refuses_an_old_pgvector_before_migrating
    fail on "the runner did not refuse at the pgvector check", the run having gone on to the
    migration lock.
    """
    from recall import schema

    conn = _MigrationConn(extversion, default_version)
    monkeypatch.setattr(schema, "_connect", lambda dsn: conn)
    with pytest.raises(Exception) as caught:
        schema.apply_migrations("postgresql://u@h/db", dim=64)
    assert isinstance(caught.value, schema.SchemaError) and "0.8.0" in str(caught.value), (
        f"the runner did not refuse at the pgvector check: {caught.value!r} after {conn.queries[-1]!r}"
    )
    assert not any("advisory" in query for query in conn.queries)
