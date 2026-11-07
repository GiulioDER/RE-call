"""CCA hunt of `recall/fact_retraction.py` and `recall/fact_projection.py` (run 20261105T225204Z).

Invariants and the failure each one catches:
- ENV-001: a desktop upload records stated facts through the controller DSN the MCP server
  resolved (its runtime environment, secrets included), not through the process environment.
- SEC-001 (owner decision 2026-11-06: erasure decides per card): two roots hold `x.md`; forgetting
  one root's memo retracts and redacts a fact whose only card cites that root's chunk, although
  the other root still serves `x.md`. A fact the other root's own chunk also supports stays.

Red proofs are in each test's docstring.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from recall.lite import LiteStore
from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger
from recall.provenance_controller import (
    ControllerDecision,
    DecisionCode,
    FactApplicationRequest,
    ProvenanceController,
)
from recall.types import Chunk
from recall_mcp.lifecycle import forget_memory
from tests.fact_fixtures import NOW, evidence_card
from tests.test_fact_retraction import DIM, DOOR, current


@pytest.fixture
def store(tmp_path: Path) -> LiteStore:
    """Two roots, each holding a memo whose root-relative name is `x.md`."""
    opened = LiteStore(tmp_path / "memory.db", dim=DIM)
    for root in ("A", "B"):
        opened.upsert(
            [Chunk(f"chunk-{root}", f"/{root}/x.md", f"text of {root}/x.md", {"file": "x.md"})],
            [[1.0] + [0.0] * (DIM - 1)],
        )
    return opened


def _apply(store: LiteStore, root: str, request_id: str, source: str = "x.md") -> ControllerDecision:
    """Apply DOOR with one card naming `source` and citing root `root`'s chunk."""
    card = evidence_card(DOOR, tenant=store.tenant, generation="g1", chunk=f"chunk-{root}", source=source)
    cards = LiteEvidenceCardStore(store)
    cards.put((card,))
    return ProvenanceController(
        tenant_id=store.tenant, generation_id="g1", cards=cards, ledger=LiteFactLedger(store),
        source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="sec001-test",
    ).apply_fact(FactApplicationRequest(DOOR, (card.card_id,), request_id))


def _value_kept(store: LiteStore) -> bool:
    return any(event.fact is not None and event.fact.object == "4512" for event in LiteFactLedger(store).events)


def test_sec001_a_namesake_under_another_root_does_not_keep_a_forgotten_fact(store: LiteStore) -> None:
    """Node tests/test_fact_retraction_hunt_cca.py::test_sec001_a_namesake_under_another_root_does_not_keep_a_forgotten_fact.

    Red against base 032c205e4 (`prepare_fact_erasure` decided gone by card name, and /B still
    serves `x.md`): fails on "the forgotten root's fact is still current: ['4512']". Red, mutation
    of the redaction half (the `erased_chunk=chunk_gone` argument dropped from `redact` in
    recall/fact_retraction.py): fails on "the forgotten root's fact value was not erased".
    """
    assert _apply(store, "A", "r1").code == DecisionCode.APPLIED

    result = forget_memory(store, ["/A/x.md"])

    assert current(store) == [], f"the forgotten root's fact is still current: {current(store)}"
    assert result.facts_retracted == 1
    assert result.facts_redacted == 1 and not _value_kept(store), "the forgotten root's fact value was not erased"


def test_sec001_control_the_other_roots_own_card_keeps_its_fact(store: LiteStore) -> None:
    """Node tests/test_fact_retraction_hunt_cca.py::test_sec001_control_the_other_roots_own_card_keeps_its_fact.

    Control: the fact /B's own chunk also supports survives the forget of /A, value and all.
    Red, mutation of `chunk_gone` in recall/fact_retraction.py (returning True for every chunk
    id, so a card counts as gone by name touch alone): fails on "a fact the other root still
    supports was retracted".
    """
    assert _apply(store, "A", "r1").code == DecisionCode.APPLIED
    assert _apply(store, "B", "r2").code == DecisionCode.DUPLICATE

    result = forget_memory(store, ["/A/x.md"])

    assert current(store) == ["4512"], f"a fact the other root still supports was retracted: {result.message}"
    assert result.facts_retracted == 0
    assert _value_kept(store), "the value the other root still supports was erased"


@pytest.mark.parametrize("failing", ["retract_forgotten", "redact"])
def test_sec001_a_rerun_completes_a_failed_per_card_erasure(
    store: LiteStore, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    """Node tests/test_fact_retraction_hunt_cca.py::test_sec001_a_rerun_completes_a_failed_per_card_erasure[retract_forgotten|redact].

    CCA BUG-001 of run 20261106T121445Z: the erased chunk ids were read from the rows the first
    run deleted, so a re-run (which deletes nothing) could not finish a failed retraction or
    redaction of a card a namesake keeps named. Red against the first SEC-001 commit (no
    `forget_source_chunks` record): fails on "the re-run left the forgotten root's fact current"
    ([retract_forgotten]) and "the re-run left the forgotten root's value" ([redact]).
    """
    from recall.fact_redaction import LiteFactRedactor

    assert _apply(store, "A", "r1").code == DecisionCode.APPLIED
    owner = LiteFactLedger if failing == "retract_forgotten" else LiteFactRedactor

    def down(*args: object, **kwargs: object) -> object:
        raise RuntimeError("unavailable once")

    with monkeypatch.context() as patch:
        patch.setattr(owner, failing, down)
        first = forget_memory(store, ["/A/x.md"])
    assert -1 in (first.facts_retracted, first.facts_redacted), first.message

    forget_memory(store, ["/A/x.md"])

    assert current(store) == [], f"the re-run left the forgotten root's fact current: {current(store)}"
    assert not _value_kept(store), "the re-run left the forgotten root's value"


def test_sec001_a_card_whose_name_no_longer_resolves_is_retracted_by_its_chunk(store: LiteStore) -> None:
    """Node tests/test_fact_retraction_hunt_cca.py::test_sec001_a_card_whose_name_no_longer_resolves_is_retracted_by_its_chunk.

    CCA SEC-001 of run 20261106T121445Z: a card stamped `old/x.md` (its memo indexed under
    another root then) cites /A's chunk, whose memo is now named `x.md`. The chunk rule redacted
    it, but retraction still asked for its name, so the fact stayed current. Red against the
    first SEC-001 change (`touched` by name only): fails on "a card of the forgotten chunk left
    its fact unretracted". The listing alone cannot show it: a name no memo serves is hidden
    anyway, so the retraction is asserted on the ledger.
    """
    assert _apply(store, "A", "r1", source="old/x.md").code == DecisionCode.APPLIED

    result = forget_memory(store, ["/A/x.md"])

    retractions = [event for event in LiteFactLedger(store).events if event.event_type == "retracted"]
    assert result.facts_retracted == 1 and len(retractions) == 1, (
        f"a card of the forgotten chunk left its fact unretracted: {result.message}"
    )
    assert current(store) == []
    assert not _value_kept(store), "a card of the forgotten chunk kept its value"


class _Stop(Exception):
    pass


def test_env001_a_desktop_upload_hands_its_runtime_environment_to_the_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node tests/test_fact_retraction_hunt_cca.py::test_env001_a_desktop_upload_hands_its_runtime_environment_to_the_manager.

    Red against the pre-fix `recall_mcp/desktop_ingest.py` (base 032c205e4): the manager was
    built with only `serving_environment`, so the controller DSN resolved from a secret never
    reached the stated-fact write, and the test fails on "the upload's runtime environment did
    not reach the generation manager".
    """
    from recall_mcp import desktop_ingest

    seen: dict[str, Any] = {}

    class FakeManager:
        def __init__(self, *args: object, **kwargs: object) -> None:
            seen.update(kwargs)

        def tenant_ingest_lock(self) -> Any:
            raise _Stop

    monkeypatch.setattr(desktop_ingest, "GenerationManager", FakeManager)
    staged = tmp_path / "tenant" / "job"
    staged.mkdir(parents=True)
    (staged / "note.md").write_text("a note\n", encoding="utf-8")
    env = {"RECALL_FACT_WRITE_DSN": "postgresql://controller@db/recall", "RECALL_ENV": "test"}

    class FakeStore:
        _dsn = "postgresql://serving@db/recall"
        tenant = "t1"

    with pytest.raises(_Stop):
        desktop_ingest.generation_ingest(
            FakeStore(), object(), str(staged), "memory", env=env  # type: ignore[arg-type]
        )

    assert seen.get("fact_write_env") == env, "the upload's runtime environment did not reach the generation manager"


def test_env001_activation_records_stated_facts_through_the_managers_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Node tests/test_fact_retraction_hunt_cca.py::test_env001_activation_records_stated_facts_through_the_managers_environment.

    The manager passes its `fact_write_env` to `reconcile_stated_facts` as `write_env`, the
    environment `fact_ledger_for` reads RECALL_FACT_WRITE_DSN from. The pre-fix manager has no
    `fact_write_env`, so a run against base 032c205e4 is not proof (TypeError); red proof is the
    mutation of `GenerationManager._record_stated_facts_after_activation` (recall/generations.py)
    back to `self.record_stated_facts(generation_id)`, which fails on "stated facts were
    recorded through the process environment".
    """
    import recall.generation_store
    import recall.stated_facts
    from recall.generations import GenerationManager

    written: list[Mapping[str, str] | None] = []

    class FakeStore:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeStore:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def set_fixed_generation(self, generation_id: str) -> None:
            pass

    class Report:
        enabled = True
        facts_stated = 1

    def reconcile(store: object, *, env: object = None, write_env: Mapping[str, str] | None = None) -> Report:
        written.append(write_env)
        return Report()

    monkeypatch.setattr(recall.generation_store, "GenerationStore", FakeStore)
    monkeypatch.setattr(recall.stated_facts, "reconcile_stated_facts", reconcile)
    env = {"RECALL_FACT_WRITE_DSN": "postgresql://controller@db/recall"}
    manager = GenerationManager("postgresql://serving@db/recall", "t1", environment="test", fact_write_env=env)

    assert manager._record_stated_facts_after_activation("g1") == 1
    assert written == [env], "stated facts were recorded through the process environment"


def _prune(store: LiteStore, root: str) -> None:
    """What `recall.index.Indexer._prune_vanished` does to root `root`'s memo gone from disk: its
    chunk ids recorded (`prune_source_chunks`), then its rows deleted."""
    store.append_audit_event("prune_source_chunks", {"chunk_ids": [f"chunk-{root}"]}, source_uri=f"/{root}/x.md")
    store.delete_sources([f"/{root}/x.md"])


def test_bug001_forgetting_a_root_never_decides_its_pruned_namesake(store: LiteStore) -> None:
    """Node tests/test_fact_retraction_hunt_cca.py::test_bug001_forgetting_a_root_never_decides_its_pruned_namesake.

    CCA BUG-001 of run 20261106T212835Z: two roots hold `x.md` and state DOOR; /B's memo is
    pruned (a state); /A's is forgotten. The forget named only /A, so /B's card stays evidence:
    no forget record for /B, nothing retracted or redacted, and the fact is current once /B is
    back. Red against the first prune-records change (`decided` matched `touched_names`, which
    holds `x.md` because /A is served under it): fails on "the pruned namesake was recorded as
    forgotten". Red, mutation of `prepare_fact_erasure`'s redaction call (recall/fact_retraction.py,
    the `kept_chunk=` argument dropped): fails on "the pruned namesake's value was erased".
    """
    assert _apply(store, "A", "r1").code == DecisionCode.APPLIED
    assert _apply(store, "B", "r2").code == DecisionCode.DUPLICATE
    _prune(store, "B")

    result = forget_memory(store, ["/A/x.md"])

    assert store.latest_chunk_records(["chunk-B"]) == {"chunk-B": ("prune_source_chunks", "/B/x.md")}, (
        "the pruned namesake was recorded as forgotten"
    )
    assert result.facts_retracted == 0, result.message
    assert _value_kept(store), "the pruned namesake's value was erased"
    store.upsert([Chunk("chunk-B", "/B/x.md", "text of B/x.md", {"file": "x.md"})], [[1.0] + [0.0] * (DIM - 1)])
    assert current(store) == ["4512"], f"the restored namesake's fact is gone: {current(store)}"


def test_bug001_control_naming_the_pruned_memo_forgets_it(store: LiteStore) -> None:
    """Node tests/test_fact_retraction_hunt_cca.py::test_bug001_control_naming_the_pruned_memo_forgets_it.

    Control: the same setup, but the forget names both sources, so the pruned /B is a decision
    and the fact goes. Red, mutation of `prepare_fact_erasure` (recall/fact_retraction.py: the
    `decided` match on `record[1]` against the sources this call names dropped, leaving the
    root-relative name only): fails on "a pruned memo the forget named kept its fact".
    """
    assert _apply(store, "A", "r1").code == DecisionCode.APPLIED
    assert _apply(store, "B", "r2").code == DecisionCode.DUPLICATE
    _prune(store, "B")

    result = forget_memory(store, ["/A/x.md", "/B/x.md"])

    assert result.facts_retracted == 1 and current(store) == [], f"a pruned memo the forget named kept its fact: {result.message}"
    assert store.latest_chunk_records(["chunk-B"])["chunk-B"][0] == "forget_source_chunks"


def test_sec001_a_later_forget_wins_whatever_clock_stamped_it(store: LiteStore) -> None:
    """Node tests/test_fact_retraction_hunt_cca.py::test_sec001_a_later_forget_wins_whatever_clock_stamped_it.

    CCA SEC-001 of run 20261106T212835Z: a prune recorded at wall time, then a forget recorded by
    a server whose evaluation clock reads 2023. The forget came later, so it is the chunk's latest
    record. Red, mutation of `LiteStore.latest_chunk_records` (recall/lite/store.py: `ORDER BY
    audit_events.rowid` replaced by `ORDER BY audit_events.created_at`): fails on "an earlier
    clock made a later forget lose to a prune".
    """
    from datetime import UTC, datetime

    from recall import eval_clock

    store.append_audit_event("prune_source_chunks", {"chunk_ids": ["chunk-B"]}, source_uri="/B/x.md")
    eval_clock.install(lambda: datetime(2023, 5, 1, tzinfo=UTC))
    try:
        store.append_audit_event("forget_source_chunks", {"chunk_ids": ["chunk-B"]}, source_uri="/B/x.md")
    finally:
        eval_clock.uninstall()

    assert store.latest_chunk_records(["chunk-B"]) == {"chunk-B": ("forget_source_chunks", "/B/x.md")}, (
        "an earlier clock made a later forget lose to a prune"
    )
