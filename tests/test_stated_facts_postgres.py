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


def uri(manager: Any, folder: str) -> str:
    """The s3 URI prefix `_corpus` gives `folder` of this manager's tenant."""
    return f"s3://approved/corpora/{manager.tenant_id}/{folder}"


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

    Spec test 1 on PostgreSQL, with the operator's prefixes and an upload: only `memory/` states,
    and an upload staged under `uploads/<tenant>/` never does although a prefix names it (the
    structural check refuses it). On PostgreSQL a prefix names the SOURCE (here an s3 URI). Red
    against 30ef33a3a: fails on "a promoted generation's stated fact is not listed: []".
    """
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", f"{uri(manager, 'memory')} {uri(manager, 'uploads')}")
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
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", uri(manager, "memory"))
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

    # One value for both sides: lite matches `memory` against names, PostgreSQL the URI against sources.
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", f"memory {uri(manager, 'memory')}")
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
    names RECALL_FACT_WRITE_DSN. Red, mutation of `GenerationManager.record_stated_facts`
    (recall/generations.py): return `max(report.facts_stated, 0)`: fails on "the refused append
    was not reported: 0". Red, mutation of `_reconcile` (recall/stated_facts.py): drop the
    RECALL_FACT_WRITE_DSN hint: fails on "the warning does not name RECALL_FACT_WRITE_DSN".
    """
    import psycopg

    from recall.fact_ledger import PostgresFactLedger

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise psycopg.errors.InsufficientPrivilege("permission denied for function recall_append_fact_ledger_event")

    monkeypatch.setattr(PostgresFactLedger, "apply_assertion", refuse)
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", uri(manager, "memory"))
    with caplog.at_level(logging.WARNING):
        generation = promote(manager, MEMOS, "corpus-v1")
    assert getattr(manager, "last_stated_facts", None) == -1, (
        f"the refused append was not reported: {getattr(manager, 'last_stated_facts', None)}"
    )
    assert manager.active_generation_id() == generation
    assert any("RECALL_FACT_WRITE_DSN" in r.getMessage() for r in caplog.records), (
        "the warning does not name RECALL_FACT_WRITE_DSN"
    )


def test_p6_an_index_of_the_legacy_table_records_stated_facts(
    make_store: Callable[[int], PgVectorStore], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p6_an_index_of_the_legacy_table_records_stated_facts.

    The legacy table's entry point (`recall_index` on a PgVectorStore) records the facts its memos
    state when the operator names a prefix, relative to RECALL_INDEX_ROOT. Red against 30ef33a3a:
    fails on "an index of the legacy table recorded nothing: []".
    """
    from recall_mcp.indexing import index_memory
    from tests.test_generations import _Embedder

    root = tmp_path / "root"
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "memory")
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "door.md").write_bytes(body([DOOR]))
    (root / "docs").mkdir()
    (root / "docs" / "notes.md").write_bytes(body(text="Plain notes."))
    store = make_store(64)
    store._tenant = "stated-" + uuid.uuid4().hex[:8]  # an empty ledger of its own; for_tenant needs a shared pool
    index_memory(store, _Embedder(1), str(root), env={"RECALL_INDEX_ROOT": str(root)})  # type: ignore[arg-type]
    got = listed(store)
    assert got == [("4512", ("memory/door.md",))], f"an index of the legacy table recorded nothing: {got}"


def test_p7_on_postgres_a_prefix_names_the_source_not_the_indexed_path(
    make_store: Callable[[int], PgVectorStore], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p7_on_postgres_a_prefix_names_the_source_not_the_indexed_path.

    Review C1 (PR B): the legacy table names a file relative to the path a caller indexed, so
    indexing `vendor/repo` named its `memory/notes.md` exactly as the operator's `memory/` and a
    document outside the admitted folder stated facts. A prefix now names a place under
    RECALL_INDEX_ROOT and is matched against the absolute source. Red (2026-10-16), mutation of
    `stating_rule` (recall/stated_facts.py): give PostgreSQL the name-matched rule
    (`StatingRule(prefixes=...)` without `by_source`): fails on "a memo outside the admitted folder
    stated a fact: [('8888', ('memory/notes.md',))]".
    """
    from recall_mcp.indexing import index_memory
    from tests.test_generations import _Embedder

    root = tmp_path / "root"
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "memory")
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "door.md").write_bytes(body([DOOR]))
    (root / "vendor" / "repo" / "memory").mkdir(parents=True)
    (root / "vendor" / "repo" / "memory" / "notes.md").write_bytes(body([{**DOOR, "predicate": "gate", "object": "8888"}]))
    store = make_store(64)
    store._tenant = "stated-" + uuid.uuid4().hex[:8]
    index_memory(store, _Embedder(1), str(root / "vendor" / "repo"), env={"RECALL_INDEX_ROOT": str(root)})  # type: ignore[arg-type]
    assert listed(store) == [], f"a memo outside the admitted folder stated a fact: {listed(store)}"


def test_p7b_a_percent_encoded_upload_is_still_an_upload() -> None:
    """Node tests/test_stated_facts_postgres.py::test_p7b_a_percent_encoded_upload_is_still_an_upload.

    Review C1 (PR B): `generation_ingest` stores `as_uri()` URIs, which percent-encode a tenant such
    as `acme co`, so the structural upload check did not see `uploads/acme co/`. No database.
    Red against 0d753ce78: fails on "a percent-encoded upload was admitted".
    """
    from recall.stated_facts import StatingRule

    rule = StatingRule(prefixes=("memory",), tenant="acme co")
    source = "file:///srv/root/uploads/acme%20co/job/memory/x.md"
    assert not rule.admits("memory/x.md", source), "a percent-encoded upload was admitted"


def test_p8_a_memo_gc_removed_states_again_when_it_returns(manager: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p8_a_memo_gc_removed_states_again_when_it_returns.

    Review I4 (PR B): once gc removes the last generation holding a memo, PR 933 retracts its facts
    (SOURCE_REMOVED); the same memo coming back byte for byte kept its request id, which counted as
    already recorded, so its facts never returned. Forget stays final. Red (2026-10-16), mutation
    of `_next_request_id` (recall/stated_facts.py): never derive (`while False:`): fails on "a memo
    that returned after gc states nothing: []".
    """
    from datetime import UTC, datetime, timedelta

    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", uri(manager, "memory"))
    promote(manager, MEMOS, "corpus-v1")
    promote(manager, {"docs/gate.md": MEMOS["docs/gate.md"], "memory/other.md": body(text="x")}, "corpus-v2")
    promote(manager, {"docs/gate.md": MEMOS["docs/gate.md"], "memory/other.md": body(text="y")}, "corpus-v3")
    manager.gc(now=datetime.now(UTC) + timedelta(days=30), retention_days=0, retain_previous=0)
    assert manager.last_gc_facts_retracted >= 1, "the test needs gc to have retracted the fact"
    promote(manager, MEMOS, "corpus-v4")
    with served(manager) as store:
        got = listed(store)
    assert got == [("4512", ("memory/door.md",))], f"a memo that returned after gc states nothing: {got}"


def test_p9_turning_stating_on_is_backfilled_by_a_command(
    manager: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p9_turning_stating_on_is_backfilled_by_a_command.

    Review I5 (PR B): a corpus promoted before stating was on, or whose recording failed, waited
    for another activation that an unchanged corpus never brings. `recall generation stated-facts`
    records the active generation's facts on demand. Red (2026-10-16), mutation of the
    `stated-facts` branch (recall/cli_commands/generation_cmd.py): `count = 0` in place of the
    call: fails on "the command did not backfill the active generation: []".
    """
    import argparse

    from recall.cli_commands.generation_cmd import _cmd_generation
    from tests.conftest import TEST_DSN

    monkeypatch.delenv("RECALL_STATED_FACT_SOURCES", raising=False)
    promote(manager, MEMOS, "corpus-v1")
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", uri(manager, "memory"))
    try:
        _cmd_generation(argparse.Namespace(dsn=TEST_DSN, tenant=manager.tenant_id,
                                           generation_cmd="stated-facts", generation_id=None))
    except SystemExit:
        pass
    with served(manager) as store:
        got = listed(store)
    assert got == [("4512", ("memory/door.md",))], f"the command did not backfill the active generation: {got}"
    assert "stated facts: 1" in capsys.readouterr().out


def test_p10_unsetting_the_rule_hides_recorded_facts_on_postgres(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p10_unsetting_the_rule_hides_recorded_facts_on_postgres.

    PR A's read-time rule on PostgreSQL: the serving process that no longer admits a memo lists none
    of its stated facts. Red, mutation of `card_liveness` (recall/provenance_controller.py): delete
    the `admitted` check. Fails on "a fact the operator no longer admits is listed on PostgreSQL".
    """
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", uri(manager, "memory"))
    promote(manager, MEMOS, "corpus-v1")
    monkeypatch.delenv("RECALL_STATED_FACT_SOURCES")
    with served(manager) as store:
        assert listed(store) == [], f"a fact the operator no longer admits is listed on PostgreSQL: {listed(store)}"


def test_p11_only_the_cards_a_new_statement_cites_are_stored(manager: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p11_only_the_cards_a_new_statement_cites_are_stored.

    Review I2 (PR B): a card's id includes its generation, and every promote put a fresh card for
    every stating memo, unreferenced, into an append-only table forget cannot reach. Red
    (2026-10-16), mutation of `_reconcile` (recall/stated_facts.py): `card_store.put(cards)`: fails
    on "a promote stored cards no statement cites: 4".
    """
    import psycopg

    from recall.provenance_cards import EVIDENCE_CARD_TABLE
    from tests.conftest import TEST_DSN

    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", f"{uri(manager, 'memory')} {uri(manager, 'docs')}")
    promote(manager, MEMOS, "corpus-v1")
    promote(manager, {**MEMOS, "docs/gate.md": body([{**DOOR, "predicate": "gate_code", "object": "7778"}])}, "corpus-v2")
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (manager.tenant_id,))
        (count,) = conn.execute(
            f"SELECT count(*) FROM {EVIDENCE_CARD_TABLE} WHERE tenant_id = %s", (manager.tenant_id,)  # noqa: S608
        ).fetchone()
    assert count == 3, f"a promote stored cards no statement cites: {count}"


def test_p12_recall_index_writes_through_its_own_write_dsn(
    make_store: Callable[[int], PgVectorStore], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p12_recall_index_writes_through_its_own_write_dsn.

    Review I3 (PR B): `recall_apply_fact` and forget resolve RECALL_FACT_WRITE_DSN from the server's
    runtime environment (secrets included), the stated path from the process only, so a DSN that
    arrives as a secret was never used. Red (2026-10-16), mutation of `_reconcile`
    (recall/stated_facts.py): `_fact_stores(store)` without `write_env`: fails on "the stated path
    ignored the runtime write DSN: [None]".
    """
    from recall import fact_retraction
    from recall_mcp.indexing import index_memory
    from tests.conftest import TEST_DSN
    from tests.test_generations import _Embedder

    seen: list[object] = []
    real = fact_retraction.fact_ledger_for

    def spy(store: object, env: Any = None) -> Any:
        seen.append((env or {}).get("RECALL_FACT_WRITE_DSN"))
        return real(store, env)

    monkeypatch.setattr(fact_retraction, "fact_ledger_for", spy)
    root = tmp_path / "root"
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", "memory")
    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))
    monkeypatch.delenv("RECALL_FACT_WRITE_DSN", raising=False)
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "door.md").write_bytes(body([DOOR]))
    store = make_store(64)
    store._tenant = "stated-" + uuid.uuid4().hex[:8]
    env = {"RECALL_INDEX_ROOT": str(root), "RECALL_FACT_WRITE_DSN": TEST_DSN}
    index_memory(store, _Embedder(1), str(root), env=env)  # type: ignore[arg-type]
    assert TEST_DSN in seen, f"the stated path ignored the runtime write DSN: {seen}"


def test_p13_a_failure_opening_the_generation_leaves_the_promote_standing(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p13_a_failure_opening_the_generation_leaves_the_promote_standing.

    The hook's own except branch (review Minor, PR B): an error before the reconcile runs (opening
    or pinning the generation) is caught, counted -1, and never fails the promote. Red, mutation of
    `GenerationManager._record_stated_facts_after_activation` (recall/generations.py): call
    `record_stated_facts` without its try: fails with the injected error out of `promote`, caught
    here as "the stated-facts hook failed the promote".
    """
    from recall.generation_store import GenerationStore

    def broken(self: object, generation_id: str) -> None:
        raise RuntimeError("cannot pin the generation")

    monkeypatch.setattr(GenerationStore, "set_fixed_generation", broken)
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", uri(manager, "memory"))
    try:
        promote(manager, MEMOS, "corpus-v1")
    except RuntimeError as exc:
        pytest.fail(f"the stated-facts hook failed the promote: {exc}")
    assert manager.last_stated_facts == -1


def test_p14_a_failed_content_lookup_shows_no_stated_value(manager: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p14_a_failed_content_lookup_shows_no_stated_value.

    Review of PR 935, finding 2: when the served generation's hash lookup failed, `content_hashes`
    answered None, which `card_liveness` reads as "no content check", while admission, computed
    from the same cached names, still passed. So a timed-out lookup brought an edited memo's old
    value back beside its new one. A stated card whose content cannot be checked is now not live,
    as admission already failed closed. Red against b7f5d9c03 (master after PR 935), the unchanged
    `content_hashes` in `served_sources` (recall/fact_retraction.py): fails on "a stated value was
    shown with its content unchecked: ['4512', '9001']" (both values contested).
    """
    from recall.generation_store import GenerationStore

    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", uri(manager, "memory"))
    promote(manager, MEMOS, "corpus-v1")
    promote(manager, {**MEMOS, "memory/door.md": body([{**DOOR, "object": "9001"}])}, "corpus-v2")

    def timed_out(self: object) -> dict[str, str]:
        raise TimeoutError("canceling statement due to statement timeout")

    monkeypatch.setattr(GenerationStore, "source_raw_hashes", timed_out)
    with served(manager) as store:
        shown = sorted(
            [value for value, _sources in listed(store)]
            + [value for group in contested(store) for value, _sources in group]
        )
    assert "4512" not in shown, f"a stated value was shown with its content unchecked: {shown}"


def test_p15_a_namesake_in_a_retained_generation_does_not_block_a_supersession(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p15_a_namesake_in_a_retained_generation_does_not_block_a_supersession.

    Review of PR 935, finding 3: the read-time checks resolved memo names with
    `sources_for_identifiers`, the erasure resolver, which spans every generation. A generation
    names its memos relative to its manifest's common directory, so G0, built from `docs/` alone,
    names `docs/new.md` plain `new.md`, the name the served generation gives `memory/new.md`.
    Supersession counts a claimant only if every source its name resolves to is admitted, so G0's
    retained rows kept `new.md` from retiring `old.md`. Red against b7f5d9c03, the unchanged
    `resolved_names` in `served_sources` (recall/fact_retraction.py): fails on "a namesake in a
    retained generation kept a superseded value current: [('4512', ('old.md',))]".
    """
    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", uri(manager, "memory"))
    promote(manager, {"docs/new.md": body(text="A docs note that happens to share a name.")}, "corpus-v0")
    promote(
        manager,
        {"memory/old.md": body([DOOR]), "memory/new.md": body(text="The door code changed.", supersedes="old.md")},
        "corpus-v1",
    )
    with served(manager) as store:
        got = listed(store)
    assert got == [], f"a namesake in a retained generation kept a superseded value current: {got}"


def test_p16_a_promote_between_two_lookups_does_not_mix_generations(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p16_a_promote_between_two_lookups_does_not_mix_generations.

    Review of PR 946, finding 3: presence, names, content hashes and supersession each read the
    ACTIVE generation, so a promote between two of them answered with one generation's names and
    another's content. Here corpus-v2, which edits the door code, is promoted right after the
    first lookup of a read that began on corpus-v1; every answer must still be corpus-v1's. Red
    against cd93af5fc, the unpinned lookups in `served_sources` (recall/fact_retraction.py): fails
    on "a promote between two lookups mixed generations: []" (v1's card checked against v2's hash).
    """
    from recall.generation_store import GenerationStore

    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", uri(manager, "memory"))
    promote(manager, MEMOS, "corpus-v1")
    moved: list[str] = []

    def promote_after(name: str) -> None:
        real = getattr(GenerationStore, name)

        def first_lookup_then_promote(self: Any, identifiers: list[str]) -> Any:
            answer = real(self, identifiers)
            if not moved:
                edited = {**MEMOS, "memory/door.md": body([{**DOOR, "object": "9001"}])}
                moved.append(promote(manager, edited, "corpus-v2"))
            return answer

        monkeypatch.setattr(GenerationStore, name, first_lookup_then_promote)

    for name in ("served_identifiers", "served_sources_for_identifiers"):
        promote_after(name)
    with served(manager) as store:
        got = listed(store)
    assert moved, "no lookup ran, so no promote came between two"
    assert got == [("4512", ("memory/door.md",))], f"a promote between two lookups mixed generations: {got}"


def test_p16b_a_promote_after_the_first_lookup_keeps_the_supersession_of_the_read(
    manager: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p16b_a_promote_after_the_first_lookup_keeps_the_supersession_of_the_read.

    CCA deferred pass over PR 946, DAT-003: p16 pins names and content but its memos have no
    `supersedes` edge, so nothing pinned the supersession lookup. corpus-v1 has `new.md` retire
    `old.md`; corpus-v2, promoted right after the first name lookup of a read that began on v1,
    drops that edge. The read must answer v1's supersession: `old.md`'s value stays retired.
    `served_sources` is called directly, as a caller without `current_facts_memory`'s own outer
    snapshot (`recall_mcp.provenance._read_generation`) would, so the pin under test is the one
    inside `supersession()` and no outer pin masks it.
    Red by mutation of `served_sources` (recall/fact_retraction.py): `supersession()` without its
    `with pinned():` reads v2's graph and fails on "a promote mid-read brought a retired value
    back: [('4512', ('memory/old.md',))]". Run in CI (no PostgreSQL on the workstation during the
    AML window); the mutation run is recorded in the PR.
    """
    from datetime import UTC, datetime

    from recall.fact_retraction import served_sources
    from recall.generation_store import GenerationStore
    from recall.provenance_controller import current_fact_listing, listing_payload
    from recall_mcp.provenance import _fact_backends

    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", uri(manager, "memory"))
    v1 = {"memory/old.md": body([DOOR]), "memory/new.md": body(text="The door code changed.", supersedes="old.md")}
    promote(manager, v1, "corpus-v1")
    moved: list[str] = []
    real = GenerationStore.served_sources_for_identifiers

    def first_lookup_then_promote(self: Any, identifiers: list[str]) -> Any:
        answer = real(self, identifiers)
        if not moved:
            v2 = {**v1, "memory/new.md": body(text="The door code changed.")}
            moved.append(promote(manager, v2, "corpus-v2"))
        return answer

    monkeypatch.setattr(GenerationStore, "served_sources_for_identifiers", first_lookup_then_promote)
    with served(manager) as store:
        lookups = served_sources(store)
        events, evidence = current_fact_listing(
            _fact_backends(store)[0], store.tenant, datetime.now(UTC), lookups
        )
        out = listing_payload(events, evidence, None)
        got = sorted((str(item["fact"]["object"]), tuple(item["sources"])) for item in out["facts"])  # type: ignore[index, union-attr]
    assert moved, "no lookup ran, so no promote came between two"
    assert got == [], f"a promote mid-read brought a retired value back: {got}"


def test_p17_a_failed_lookup_is_reported_as_degraded(manager: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Node tests/test_stated_facts_postgres.py::test_p17_a_failed_lookup_is_reported_as_degraded.

    Review of PR 946, finding 2: since test_p14 a failed content lookup lists no stated fact, and
    `recall_current_facts` said nothing, so an outage read as a memory that holds no such fact. It
    now answers `served_state` `degraded` and names the lookup; a healthy read answers `ok`. Red
    against cd93af5fc, `current_facts_memory` (recall_mcp/provenance.py) with no marker: fails on
    "a failed content lookup was not reported: (None, None)".
    """
    from recall.generation_store import GenerationStore

    monkeypatch.setenv("RECALL_STATED_FACT_SOURCES", uri(manager, "memory"))
    promote(manager, MEMOS, "corpus-v1")
    with served(manager) as store:
        healthy = current_facts_memory(store)

    def timed_out(self: object) -> dict[str, str]:
        raise TimeoutError("canceling statement due to statement timeout")

    monkeypatch.setattr(GenerationStore, "source_raw_hashes", timed_out)
    with served(manager) as store:
        out = current_facts_memory(store)
    marker = (out.get("served_state"), out.get("served_lookups_failed"))
    assert marker == ("degraded", ["content"]), f"a failed content lookup was not reported: {marker}"
    assert (healthy.get("served_state"), healthy.get("served_lookups_failed")) == ("ok", [])
    assert out["facts"] == [] and healthy["facts"], "the stated fact was not hidden only by the failure"


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
