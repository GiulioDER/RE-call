"""CCA PR469 through460 regressions.

Red proof: these nodes run unchanged against pre-fix source at 3b3d875f8 on the
audit-owned PostgreSQL database. Each must fail its stated invariant assertion,
then pass with the production repair. See the private audit proof receipt for
the exact source hashes, node IDs and assertion messages. No paid APIs are used.
"""
from __future__ import annotations

import subprocess
import uuid
from contextvars import Context

import psycopg
from psycopg import sql
import pytest

from benchmarks.atm_answer_diagnosis import token_coverage
from recall import cli, quickstart
from recall.cli_commands import setup_wizard
from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.index_lock import ConcurrentIndex, single_writer
from recall.pool import SharedPool
from recall.store import PgVectorStore
from tests.conftest import TEST_DSN, requires_db


def test_existing_dsn_is_checked_before_initialization(monkeypatch):
    """SEC-001: external default credentials must refuse before embedder or DB work."""
    class BoundaryReached(BaseException):
        pass

    reached = []

    def boundary(name):
        reached.append(name)
        raise BoundaryReached

    monkeypatch.setattr(setup_wizard, "_make_embedder", boundary)
    monkeypatch.delenv("RECALL_ALLOW_INSECURE_DSN", raising=False)
    insecure = "postgresql://recall:recall@db.example.com/recall"
    for opt_out in (None, "0", "false"):
        if opt_out is None:
            monkeypatch.delenv("RECALL_ALLOW_INSECURE_DSN", raising=False)
        else:
            monkeypatch.setenv("RECALL_ALLOW_INSECURE_DSN", opt_out)
        try:
            cli.main(["--embedder", "hashing", "quickstart", "--existing-dsn", insecure])
        except (SystemExit, PermissionError, BoundaryReached):
            pass
        assert not reached, "insecure existing DSN reached initialization"

    # Positive controls exercise the same consumer boundary without any connection.
    for dsn, opt_out in (("postgresql://operator:private@db.example.com/db", "0"),
                         (insecure, "1"),
                         ("postgresql://recall:recall@127.0.0.1/db", "0")):
        monkeypatch.setenv("RECALL_ALLOW_INSECURE_DSN", opt_out)
        with pytest.raises(BoundaryReached):
            cli.main(["--embedder", "hashing", "quickstart", "--existing-dsn", dsn])
        assert reached.pop() == "hashing"


def test_teardown_failure_retains_retry_metadata(tmp_path, monkeypatch):
    """BUG-001: Docker failure cannot erase metadata or announce successful removal."""
    compose = tmp_path / quickstart.COMPOSE_NAME
    compose.write_text("services: {}\n", encoding="utf8")
    monkeypatch.setattr(
        quickstart.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="permission denied by Docker daemon"
        ),
    )
    failure = None
    try:
        quickstart.remove_stack(tmp_path)
    except (RuntimeError, SystemExit) as exc:
        failure = exc
    assert compose.exists(), "failed teardown destroyed retry metadata"
    assert failure is not None and "permission denied" in str(failure)
    monkeypatch.setattr(
        quickstart.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        ),
    )
    assert quickstart.remove_stack(tmp_path) is True
    assert quickstart.remove_stack(tmp_path) is False


def test_own_notes_guidance_keeps_the_demo_separate():
    """DOC-002: onboarding real notes must not target the fictional corpus."""
    lines = quickstart.next_steps("postgresql://example/db", provisioned=False, compose_path=None)
    assert not any("index <folder>" in line for line in lines), "own notes target the demo corpus"
    assert any("your own notes" in line and "recall setup" in line for line in lines)


@pytest.mark.parametrize("gold,evidence", [("42", "142"), ("red", "hundred")])
def test_coverage_requires_complete_tokens(gold, evidence):
    """NUM-001: a substring collision cannot increase content-token coverage."""
    tokenize = lambda text: text.lower().split()  # noqa: E731
    assert token_coverage(gold, evidence, tokenize, set()) == 0, "substring counted as a token"
    assert token_coverage(gold, gold.upper(), tokenize, set()) == 1
    assert token_coverage("the", evidence, tokenize, {"the"}) is None


@requires_db
@pytest.mark.parametrize("alias_kind", ["uppercase", "truncated", "utf8_truncated"])
def test_table_aliases_contend_for_one_writer_lock(alias_kind):
    """BUG-002: accepted names resolving to one relation must serialize writers."""
    prefix = "al_" + uuid.uuid4().hex[:8]
    if alias_kind == "uppercase":
        table, alias = prefix, prefix.upper()
    elif alias_kind == "truncated":
        table, alias = prefix + "a" * 64, prefix + "a" * 64 + "b"
    else:
        table, alias = prefix + "é" * 40, prefix + "é" * 40 + "x"
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE TABLE {} (value integer)").format(sql.Identifier(table)))
        physical = conn.execute("SELECT (parse_ident(%s))[1]::name::text", (table,)).fetchone()[0]
        first = PgVectorStore(TEST_DSN, dim=64, table=table)
        other = PgVectorStore(TEST_DSN, dim=64, table=alias)
        tenant_control = PgVectorStore(TEST_DSN, dim=64, table=table, tenant="other")
        try:
            identities = conn.execute(
                "SELECT to_regclass(%s)::oid, to_regclass(%s)::oid", (table, alias)
            ).fetchone()
            assert identities[0] is not None and identities[0] == identities[1]
            entered = False
            with single_writer(first):
                try:
                    with single_writer(other, wait_seconds=0):
                        entered = True
                except ConcurrentIndex:
                    pass
                assert not entered, "table alias bypassed the writer lock"
                with single_writer(tenant_control, wait_seconds=0) as taken:
                    assert taken
        finally:
            first.close()
            other.close()
            tenant_control.close()
            conn.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(physical)))


@requires_db
@pytest.mark.parametrize("mode", ["direct", "pooled", "shared"])
def test_indexing_refuses_after_its_lock_backend_dies(tmp_path, mode):
    """STAKES-001: losing the guard session must prevent subsequent corpus writes."""
    table = "dead_" + uuid.uuid4().hex[:8]
    shared = SharedPool(
        TEST_DSN, min_size=1, max_size=2, statement_timeout_ms=12345,
        connect_kwargs={"options": "-c lock_timeout=5432"},
    ) if mode == "shared" else None
    store = PgVectorStore(
        TEST_DSN, dim=64, table=table, shared_pool=shared,
        pool_size=1 if mode == "pooled" else None,
    )
    store.ensure_schema()
    other = PgVectorStore(TEST_DSN, dim=64, table=table)
    corpus = tmp_path / "notes"
    corpus.mkdir()
    (corpus / "note.md").write_text("a fresh memory about database operations", encoding="utf8")
    guards = []

    class LosingEmbedder(HashingEmbedder):
        def embed(self, texts):
            if shared is not None:
                settings = store._with_retry(lambda conn: conn.execute(
                    "SELECT current_setting('statement_timeout'), current_setting('lock_timeout')"
                ).fetchone())
                assert settings == ("12345ms", "5432ms"), "shared connection policy was lost"
                view = store.for_tenant("another-tenant")
                assert view._with_retry(lambda conn: conn.execute(
                    "SELECT current_setting('recall.tenant_id')"
                ).fetchone()[0]) == "another-tenant", "tenant view inherited the indexing pin"
            with psycopg.connect(TEST_DSN, autocommit=True) as conn:
                rows = conn.execute(
                    "SELECT pid FROM pg_stat_activity WHERE application_name LIKE %s",
                    (f"%recall index {table}/%",),
                ).fetchall()
                assert len(rows) == 1, "could not uniquely identify this indexer's guard"
                assert conn.execute("SELECT pg_terminate_backend(%s)", (rows[0][0],)).fetchone()[0]
            guard = single_writer(other, wait_seconds=0)
            context = Context()
            assert context.run(guard.__enter__), "the terminated session still held its lock"
            guards.append((context, guard))
            return super().embed(texts)

    try:
        failure = None
        try:
            Indexer(store, LosingEmbedder(dim=64)).index_path(corpus)
        except psycopg.OperationalError as exc:
            failure = exc
        assert failure is not None, "indexer wrote after its guard backend died"
        assert store.count() == 0, "failed index persisted data without owning its lock"
        for context, guard in guards:
            context.run(guard.__exit__, None, None, None)
        guards.clear()
        # Binding must reset after failure so this same store can index normally.
        assert Indexer(store, HashingEmbedder(dim=64)).index_path(corpus).chunks > 0
        assert store.count() > 0
    finally:
        for context, guard in guards:
            context.run(guard.__exit__, None, None, None)
        store.drop_table()
        store.close()
        other.close()
        if shared is not None:
            shared.close()
