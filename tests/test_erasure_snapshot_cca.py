"""FIX-007: a retained claim describes the row version public erasure actually removed.

The connections and cursors below execute against the isolated session database. Only the
ordering is controlled: a second public store commits a metadata update just before DELETE.
There are no sleeps, threads, fabricated scan results or replacement persistence helpers.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from typing import Any

import pytest

from recall.store import PgVectorStore
from recall.types import Chunk
from tests.conftest import TEST_DSN, requires_db

DIM = 64
PROFILE = "erasure-snapshot-control"


def _chunk(name: str, supersedes: str | None = None) -> Chunk:
    metadata: dict[str, Any] = {"file": f"{name}.md"}
    if supersedes is not None:
        metadata["supersedes"] = f"{supersedes}.md"
    return Chunk(name, f"/corpus/{name}.md", f"memo {name}", metadata)


class _BeforeDeleteConnection:
    """Delegate real database I/O, scheduling one commit before the selected chunk DELETE."""

    def __init__(self, connection: Any, table: str, before_delete: Callable[[], None]) -> None:
        self._connection = connection
        self._delete = re.compile(
            r'\bDELETE\s+FROM\s+(?:"?public"?\.)?"?' + re.escape(table) + r'"?(?=\s|$)',
            re.IGNORECASE,
        )
        self._before_delete = before_delete

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def execute(self, query: Any, *args: Any, **kwargs: Any) -> Any:
        text = query if isinstance(query, str) else query.as_string(self._connection)
        if self._delete.search(text):
            self._before_delete()
        return self._connection.execute(query, *args, **kwargs)


@requires_db
@pytest.mark.parametrize("operation", ["delete_sources", "delete_sources_across"])
def test_erasure_retains_the_claim_of_the_deleted_row_version(
    make_store: Callable[[int], PgVectorStore], monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    """Nodes: test_erasure_snapshot_cca.py::test_erasure_retains_the_claim_of_the_deleted_row_version
    [delete_sources] and [delete_sources_across].

    Target: PgVectorStore public erasure and its retained supersession record (DAT-001).
    Baseline: ab1a48ba427599bd174e9d99b5280182f37b2aea, the original PR987 head.
    B initially retires A. After the old record scan, a second connection commits B retiring C;
    DELETE then removes that updated version. C must remain retired and A must be current.
    Predicted RED: the assertion "erasure retained a claim from a row version it did not delete"
    fails because the baseline ghost still retires A and successor("c.md") is None.
    Original-head RED and repaired GREEN were observed; see the audit receipts.

    The rendezvous is immediately before the real DELETE, not inside the obsolete scan helper.
    A DELETE RETURNING repair therefore sees the updated metadata through its normal public path.
    """
    table = make_store(DIM).table
    tenant = "snapshot-" + uuid.uuid4().hex
    eraser = PgVectorStore(TEST_DSN, DIM, table=table, tenant=tenant)
    updater = PgVectorStore(TEST_DSN, DIM, table=table, tenant=tenant)
    embedding = [1.0] + [0.0] * (DIM - 1)
    try:
        eraser.upsert([_chunk("a"), _chunk("b", "a"), _chunk("c")], [embedding] * 3)
        eraser.upsert_sparse(PROFILE, {"a": {1: 1.0}, "b": {2: 1.0}, "c": {3: 1.0}})
        assert eraser.supersession_map() == {"a.md": "b.md"}
        assert eraser.sparse_row_count(PROFILE) == 3
        commits = 0

        def change_claim() -> None:
            nonlocal commits
            if commits:
                return
            updater.upsert([_chunk("b", "c")], [embedding])
            assert updater.supersession_map() == {"c.md": "b.md"}, "the competing update did not commit"
            commits += 1

        real_with_retry = eraser._with_retry

        def coordinated(operation_body: Callable[[Any], Any]) -> Any:
            return real_with_retry(
                lambda connection: operation_body(_BeforeDeleteConnection(connection, table, change_claim))
            )

        # The proxy delegates transactions, cursors and all results unchanged. Only this handle
        # is wrapped, so the updater uses its own real connection and commits before DELETE.
        with monkeypatch.context() as patch:
            patch.setattr(eraser, "_with_retry", coordinated)
            if operation == "delete_sources":
                removed = eraser.delete_sources(["/corpus/b.md"])
            else:
                removed = eraser.delete_sources_across([table], ["/corpus/b.md"])

        assert commits == 1, "the competing update was not scheduled at the real DELETE boundary"
        assert removed == 1, "the deletion receipt lost the removed chunk"
        assert eraser.count() == 2
        assert not eraser.sources_for_identifiers(["b.md"])
        assert eraser.sparse_row_count(PROFILE) == 2, "erasure did not scrub the deleted chunk's sparse sidecar"
        graph = eraser.supersession_graph()
        assert graph.successor("c.md") == "b.md", (
            "erasure retained a claim from a row version it did not delete"
        )
        assert graph.successor("a.md") is None, "erasure retained the obsolete claim instead of the deleted one"
    finally:
        updater.close()
        eraser.close()


@requires_db
def test_generation_manager_legacy_erasure_retains_the_deleted_row_version(
    make_store: Callable[[int], PgVectorStore], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Node: test_erasure_snapshot_cca.py::test_generation_manager_legacy_erasure_retains_the_deleted_row_version.

    Target: GenerationManager.forget(..., legacy_table=table), FIX-007 / DAT-001.
    Pre-rework baseline: HEAD ab1a48ba427599bd174e9d99b5280182f37b2aea plus the candidate
    repair tree 67d3c0362d7cc976099fec9d8c5984b54f54238b, revised by the L6 architect.
    At this baseline generations.py SHA256 is
    cbcf12275e341fd64dd51ca99f77ee68542c31a624757bacf13742822bd7932d.
    After the legacy scan but before DELETE, public legacy upsert commits B's claim from A to C.
    Predicted RED: "manager erasure retained a claim from a row version it did not delete"
    because its retained record still retires A, whereas the deleted B actually retired C.
    Intended assertion RED and matching GREEN were observed on the isolated testbench; receipts are recorded with the audit.

    Only the real DELETE is coordinated. The updater takes no generation source lock, and
    all reads, writes, returned rows and transaction boundaries otherwise remain real.
    """
    from collections.abc import Iterator
    from contextlib import contextmanager

    from recall.generations import GenerationManager

    table = make_store(DIM).table
    tenant = "manager-snapshot-" + uuid.uuid4().hex
    manager = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    eraser = PgVectorStore(TEST_DSN, DIM, table=table, tenant=tenant)
    updater = PgVectorStore(TEST_DSN, DIM, table=table, tenant=tenant)
    embedding = [1.0] + [0.0] * (DIM - 1)
    try:
        eraser.upsert([_chunk("a"), _chunk("b", "a"), _chunk("c")], [embedding] * 3)
        assert eraser.supersession_map() == {"a.md": "b.md"}
        commits = 0

        def change_claim() -> None:
            nonlocal commits
            if commits:
                return
            updater.upsert([_chunk("b", "c")], [embedding])
            assert updater.supersession_map() == {"c.md": "b.md"}, (
                "the competing public legacy update did not commit"
            )
            commits += 1

        real_connect = manager._connect

        @contextmanager
        def coordinated() -> Iterator[Any]:
            with real_connect() as connection:
                yield _BeforeDeleteConnection(connection, table, change_claim)

        with monkeypatch.context() as patch:
            patch.setattr(manager, "_connect", coordinated)
            receipt = manager.forget("/corpus/b.md", legacy_table=table)

        assert commits == 1, "the competing update missed the manager's legacy DELETE boundary"
        assert receipt.source_uri == "/corpus/b.md"
        assert receipt.chunks_removed == 1, "the manager's receipt lost the removed legacy chunk"
        assert receipt.event_id
        assert eraser.count() == 2
        assert not eraser.sources_for_identifiers(["b.md"])
        graph = eraser.supersession_graph()
        assert graph.successor("c.md") == "b.md", (
            "manager erasure retained a claim from a row version it did not delete"
        )
        assert graph.successor("a.md") is None, (
            "manager erasure retained the obsolete legacy claim instead of the deleted one"
        )
    finally:
        updater.close()
        eraser.close()


@requires_db
def test_generation_manager_legacy_erasure_preserves_quoted_table_identity(
    make_store: Callable[[int], PgVectorStore],
) -> None:
    """Node: test_erasure_snapshot_cca.py::test_generation_manager_legacy_erasure_preserves_quoted_table_identity.

    Target: GenerationManager.forget's legacy DELETE through pg_delete_removals, FIX-007.
    Baseline: the attempt-2 repair interpolates the accepted mixed-case table without quoting;
    supersession_records.py SHA256 is
    47492e64e79f34fe3bf0313ed22311c88f3d1bc56b1bb68ae9fa3fdc1fea4b02.
    The quoted target's B retires C, while the lowercase namesake's B retires A.
    Predicted RED: "legacy erasure targeted the lowercase namesake instead of the quoted table".
    Intended assertion RED and matching GREEN were observed on the isolated testbench; receipts are recorded with the audit.

    Fixture SQL establishes compatible historic tables without asking PgVectorStore's own
    unquoted write path to create them. Only this test's two explicitly created tables are dropped.
    """
    import psycopg
    from psycopg import sql
    from psycopg.types.json import Jsonb

    from recall.generations import GenerationManager

    template = make_store(DIM).table
    mixed = "CcaMixed_" + uuid.uuid4().hex[:12]
    lowercase = mixed.lower()
    tenant = "quoted-snapshot-" + uuid.uuid4().hex
    manager = GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    source = "/corpus/b.md"
    created: list[str] = []
    try:
        with psycopg.connect(TEST_DSN, autocommit=True) as connection:
            for table, predecessor, text in (
                (mixed, "c", "quoted target"), (lowercase, "a", "lowercase decoy")
            ):
                connection.execute(
                    sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING ALL)").format(
                        sql.Identifier(table), sql.Identifier(template)
                    )
                )
                created.append(table)
                connection.execute(
                    sql.SQL(
                        "INSERT INTO {} (tenant_id, id, source, text, metadata) "
                        "VALUES (%s, %s, %s, %s, %s)"
                    ).format(sql.Identifier(table)),
                    (tenant, "b", source, text, Jsonb(_chunk("b", predecessor).metadata)),
                )
            lower_before = connection.execute(
                sql.SQL("SELECT id, source, text, metadata FROM {} WHERE tenant_id = %s").format(
                    sql.Identifier(lowercase)
                ),
                (tenant,),
            ).fetchall()
            quoted_before = connection.execute(
                sql.SQL("SELECT id FROM {} WHERE tenant_id = %s").format(sql.Identifier(mixed)),
                (tenant,),
            ).fetchall()
            assert quoted_before == [("b",)] and len(lower_before) == 1

        receipt = manager.forget(source, legacy_table=mixed)
        assert receipt.chunks_removed == 1, "the receipt lost the quoted legacy chunk"
        with psycopg.connect(TEST_DSN, autocommit=True) as connection:
            connection.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            quoted_after = connection.execute(
                sql.SQL("SELECT id FROM {} WHERE tenant_id = %s").format(sql.Identifier(mixed)),
                (tenant,),
            ).fetchall()
            lower_after = connection.execute(
                sql.SQL("SELECT id, source, text, metadata FROM {} WHERE tenant_id = %s").format(
                    sql.Identifier(lowercase)
                ),
                (tenant,),
            ).fetchall()
            events = connection.execute(
                "SELECT payload FROM recall_audit_events WHERE tenant_id = %s "
                "AND source_uri = %s AND event_type = 'supersession_forgotten' "
                "AND payload->>'scope' = %s",
                (tenant, source, mixed),
            ).fetchall()
        assert quoted_after == [], (
            "legacy erasure targeted the lowercase namesake instead of the quoted table"
        )
        assert lower_after == lower_before, "legacy erasure modified the lowercase namesake"
        assert len(events) == 1, "the retained record lost the original mixed-case scope"
        payload = events[0][0]
        assert payload["scope"] == mixed and payload["file"] == "b.md"
        assert [claim[0] for claim in payload["claims"]] == ["c.md"], (
            "legacy erasure retained the lowercase namesake's claim instead of its deleted tuple"
        )
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as connection:
            for table in reversed(created):
                connection.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(table)))
