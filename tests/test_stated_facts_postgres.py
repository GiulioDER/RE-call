"""Facts a memo states reach the fact ledger on PostgreSQL, opt-in, under the lite store's rules.

PR 934 records the facts a memo states in its `recall_graph` frontmatter on a lite store. Here the
same recording follows a promoted (or rolled back) generation and an index of the legacy table,
when `RECALL_STATED_FACT_SOURCES` names the prefixes whose memos may state facts; without it
PostgreSQL states nothing. What is current is decided at read time, as on lite: a stated card
lives while the SERVED generation holds its memo with the content it recorded.

Invariants and the failure each one catches: in each test's docstring, with its red proof (against
PR 934's head `30ef33a3a` unless a mutation is named).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from recall.store import PgVectorStore
from recall_mcp.provenance import current_facts_memory
from tests.test_absent_memo_facts import _corpus
from tests.test_lite_calibration import ContentWordEmbedder, _write_memos

DOOR = {"subject": "user", "predicate": "door_code", "object": "4512"}


def body(facts: list[dict[str, object]] | None = None, *, text: str = "A note about the door.", **front: str) -> bytes:
    """A memo's bytes: frontmatter `front`, `facts` as its recall_graph, then `text`."""
    lines = [f"{key}: {value}" for key, value in front.items()]
    if facts is not None:
        lines.append("recall_graph: " + json.dumps({"facts": facts}))
    head = "---\n" + "\n".join(lines) + "\n---\n" if lines else ""
    return f"{head}# Memo\n\n{text}\n".encode()


def promote(manager: Any, memos: dict[str, bytes], version: str) -> str:
    """Build, validate and promote a generation of `memos`; return its id."""
    from tests.test_generations import _Embedder, _pipeline

    manifest, reader = _corpus(manager.tenant_id, memos, version)
    generation = manager.create(manifest, _pipeline("model-a"))
    manager.build(generation.generation_id, reader, _Embedder(1), lambda text: [text])
    manager.validate(generation.generation_id)
    manager.promote(generation.generation_id, unsafe_development=True)
    return str(generation.generation_id)


def listed(store: Any) -> list[tuple[str, tuple[str, ...]]]:
    """What `recall_current_facts` lists under `facts`: (object, sources), sorted."""
    out = current_facts_memory(store)
    return sorted((str(item["fact"]["object"]), tuple(item["sources"])) for item in out["facts"])  # type: ignore[index, union-attr]


def contested(store: Any) -> list[list[tuple[str, tuple[str, ...]]]]:
    out = current_facts_memory(store)
    return sorted(
        sorted((str(v["fact"]["object"]), tuple(v["sources"])) for v in group["values"])  # type: ignore[index]
        for group in out["contested"]  # type: ignore[union-attr]
    )


def served(manager: Any) -> Any:
    from recall.generation_store import GenerationStore
    from tests.conftest import TEST_DSN

    return GenerationStore(TEST_DSN, 64, tenant=manager.tenant_id)


#: Two folders, so a generation's names keep their folder (`manifest_relative_paths` strips only the
#: directory every object shares).
MEMOS = {
    "memory/door.md": body([DOOR]),
    "docs/gate.md": body([{**DOOR, "predicate": "gate_code", "object": "7777"}]),
}


def test_p1_a_promoted_generation_records_the_facts_its_memos_state(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p1_a_promoted_generation_records_the_facts_its_memos_state.

    Spec test 1 on PostgreSQL, with the operator's prefix and an upload: only `memory/` states, an
    upload staged under `uploads/<tenant>/` never does. Red against 30ef33a3a: fails on "a promoted
    generation's stated fact is not listed: []".
    """
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "memory uploads")
    memos = {**MEMOS, f"uploads/{manager.tenant_id}/job1/u.md": body([{**DOOR, "predicate": "wifi", "object": "6666"}])}
    promote(manager, memos, "corpus-v1")
    with served(manager) as store:
        got = listed(store)
    assert got == [("4512", ("memory/door.md",))], f"a promoted generation's stated fact is not listed: {got}"
    assert getattr(manager, "last_stated_facts", None) == 1


def test_p2_postgres_states_nothing_by_default(manager: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p2_postgres_states_nothing_by_default.

    Spec 4.5: a PostgreSQL tenant can hold documents, so without RECALL_STATED_FACT_SOURCES no memo
    states. Red, mutation of `stating_rule` (recall/stated_facts.py): give a PostgreSQL store the
    prefix `memory` when the variable is unset. Fails on "PostgreSQL stated a fact by default".
    """
    monkeypatch.delenv("RECALL_STATED_FACT_SOURCES", raising=False)
    promote(manager, MEMOS, "corpus-v1")
    with served(manager) as store:
        assert listed(store) == [], f"PostgreSQL stated a fact by default: {listed(store)}"
    assert getattr(manager, "last_stated_facts", None) == 0


def test_p3_an_edited_memo_in_a_new_generation_and_a_rollback(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p3_an_edited_memo_in_a_new_generation_and_a_rollback.

    Spec test 4 on generations: a new generation whose memo states another value lists only that
    value (the old content is not in the served generation), and a rollback lists the old one again.
    Red against 30ef33a3a: fails on "the edited memo's value is not the one listed: []". Red,
    mutation of `served_sources` (recall/fact_retraction.py): pass `content_hashes=None` for a
    PostgreSQL store. Fails on the same line, with both values contested.
    """
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "memory")
    first = promote(manager, MEMOS, "corpus-v1")
    promote(manager, {**MEMOS, "memory/door.md": body([{**DOOR, "object": "9001"}])}, "corpus-v2")
    with served(manager) as store:
        got = listed(store)
    assert got == [("9001", ("memory/door.md",))], f"the edited memo's value is not the one listed: {got}"
    assert manager.rollback() == first
    with served(manager) as store:
        assert listed(store) == [("4512", ("memory/door.md",))], "the rollback did not restore the old value"


def test_p4_postgres_and_lite_list_the_same_facts_and_contests(
    manager: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p4_postgres_and_lite_list_the_same_facts_and_contests.

    Spec test 2: the same memos on a lite store and in a promoted PostgreSQL generation give the
    same `facts` and `contested` (two memos disagree on the door code, a third states a gate code).
    Red against 30ef33a3a: fails on "PostgreSQL and lite disagree" (PostgreSQL lists nothing).
    """
    from recall.lite import LiteStore
    from recall.lite.refresh import Outcome, refresh

    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "memory")
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    memos = {
        "memory/door-a.md": body([DOOR]),
        "memory/door-b.md": body([{**DOOR, "object": "9001"}]),
        "memory/gate.md": body([{**DOOR, "predicate": "gate_code", "object": "5150"}]),
        "docs/readme.md": body(text="Plain notes."),
    }
    root = tmp_path / "root"
    for name, data in memos.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    _write_memos(root / "background", 4)
    with LiteStore(tmp_path / ".recall" / "memory.db", dim=2048) as lite:
        report = refresh(lite, [root], ContentWordEmbedder(), force=True, key_root=root)  # type: ignore[arg-type]
        assert report.outcome is Outcome.REFRESHED, report.to_dict()
        lite_view = (listed(lite), contested(lite))
    promote(manager, memos, "corpus-v1")
    with served(manager) as store:
        pg_view = (listed(store), contested(store))
    assert lite_view[0] == [("5150", ("memory/gate.md",))], f"the lite side is not what the test assumes: {lite_view}"
    assert pg_view == lite_view, f"PostgreSQL and lite disagree: {pg_view} != {lite_view}"


def test_p5_a_refused_append_leaves_the_promote_standing_and_names_the_role(
    manager: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p5_a_refused_append_leaves_the_promote_standing_and_names_the_role.

    Spec test 11: a serving role without the controller role cannot append to the ledger. The
    refusal is injected at the ledger (creating roles is one of the test host's known
    environment failures). The promote still stands, `last_stated_facts` is -1, and the WARNING
    names RECALL_FACT_WRITE_DSN. Red against 30ef33a3a: fails on "the refused append was not
    reported: None". Red, mutation of `_reconcile` (recall/stated_facts.py): drop the
    RECALL_FACT_WRITE_DSN hint: fails on "the warning does not name RECALL_FACT_WRITE_DSN".
    """
    import psycopg

    from recall.fact_ledger import PostgresFactLedger

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise psycopg.errors.InsufficientPrivilege("permission denied for function recall_append_fact_ledger_event")

    monkeypatch.setattr(PostgresFactLedger, "apply_assertion", refuse)
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "memory")
    with caplog.at_level(logging.WARNING):
        generation = promote(manager, MEMOS, "corpus-v1")
    assert getattr(manager, "last_stated_facts", None) == -1, (
        f"the refused append was not reported: {getattr(manager, 'last_stated_facts', None)}"
    )
    assert manager.active_generation_id() == generation, "a refused append undid the promote"
    assert any("RECALL_FACT_WRITE_DSN" in r.getMessage() for r in caplog.records), (
        "the warning does not name RECALL_FACT_WRITE_DSN"
    )


def test_p6_an_index_of_the_legacy_table_records_stated_facts(
    make_store: Callable[[int], PgVectorStore], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p6_an_index_of_the_legacy_table_records_stated_facts.

    The legacy table's entry point (`recall_index` on a PgVectorStore) records the facts its memos
    state when the operator names a prefix. Red against 30ef33a3a: fails on "an index of the
    legacy table recorded nothing: []".
    """
    from recall_mcp.indexing import index_memory
    from tests.test_generations import _Embedder

    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "memory")
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    root = tmp_path / "root"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "door.md").write_bytes(body([DOOR]))
    (root / "docs").mkdir()
    (root / "docs" / "notes.md").write_bytes(body(text="Plain notes."))
    store = make_store(64)
    store._tenant = "stated-" + uuid.uuid4().hex[:8]  # a tenant of its own, so the ledger is empty
    index_memory(store, _Embedder(1), str(root), env={"RECALL_INDEX_ROOT": str(root)})  # type: ignore[arg-type]
    got = listed(store)
    assert got == [("4512", ("memory/door.md",))], f"an index of the legacy table recorded nothing: {got}"


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    yield


@pytest.fixture
def manager() -> Iterator[Any]:
    """A GenerationManager on a tenant of its own, cleaned up afterwards (as test_absent_memo_facts)."""
    import psycopg

    from recall.generations import GenerationManager
    from tests.conftest import TEST_DSN, _db_available, db_unreachable_reason

    if not _db_available():
        pytest.skip(db_unreachable_reason())
    tenant = "stated-" + uuid.uuid4().hex[:10]
    yield GenerationManager(TEST_DSN, tenant, actor="pytest", environment="test")
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        for table in ("recall_source_tombstones", "recall_audit_events", "recall_ingest_jobs",
                      "recall_tenant_state", "recall_generations"):
            conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (tenant,))  # noqa: S608
