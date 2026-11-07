"""Deferred pass of CCA run 20261106T121445Z over `recall/fact_retraction.py` (run 20261107T161614Z).

Invariants and the failure each one catches:
- PERF-001: a forget learns the chunk ids it erases, and which forgotten sources are still served,
  from one ids-only read per store (`chunk_ids_for_sources`), never one full-text
  `chunks_for_source` read per source, all under the exclusive erasure lock.
- PERF-002 and CODE-001: after deletion, which erased chunks are back is one ids-only read per
  store (`present_chunk_ids`), memoised by `_settled`, never one full-text `chunk_by_id` per cited
  chunk.
- Control: the erasure decides exactly as before, retracting and erasing the fact whose every card
  cites a forgotten chunk, while a namesake under another root keeps its own.

Red proofs are in each test's docstring.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from recall.lite import LiteStore
from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger
from recall.provenance_controller import DecisionCode, FactApplicationRequest, ProvenanceController
from recall.types import Chunk
from recall_mcp.lifecycle import forget_memory
from tests.fact_fixtures import NOW, evidence_card
from tests.test_fact_retraction import DIM, DOOR, current

NAMES = ("x.md", "y.md", "z.md")


@pytest.fixture
def store(tmp_path: Path) -> Iterator[LiteStore]:
    """Root A holds three memos, root B a namesake of the first; a fact cites each of A's chunks."""
    opened = LiteStore(tmp_path / "memory.db", dim=DIM)
    for root, names in (("A", NAMES), ("B", NAMES[:1])):
        for name in names:
            opened.upsert(
                [Chunk(f"chunk-{root}-{name}", f"/{root}/{name}", f"text of {root}/{name}", {"file": name})],
                [[1.0] + [0.0] * (DIM - 1)],
            )
    cards = LiteEvidenceCardStore(opened)
    for i, name in enumerate(NAMES):
        card = evidence_card(DOOR, tenant=opened.tenant, generation="g1", chunk=f"chunk-A-{name}", source=name)
        cards.put((card,))
        decision = ProvenanceController(
            tenant_id=opened.tenant, generation_id="g1", cards=cards, ledger=LiteFactLedger(opened),
            source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="deferred-test",
        ).apply_fact(FactApplicationRequest(DOOR, (card.card_id,), f"r{i}"))
        assert decision.code in (DecisionCode.APPLIED, DecisionCode.DUPLICATE)
    yield opened


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """How often each chunk read of `LiteStore` runs while the test does."""
    counts: dict[str, int] = {}
    for name in ("chunks_for_source", "chunk_by_id", "chunk_ids_for_sources"):
        real = getattr(LiteStore, name)

        def counted(self: LiteStore, *args: Any, _real: Any = real, _name: str = name, **kwargs: Any) -> Any:
            counts[_name] = counts.get(_name, 0) + 1
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(LiteStore, name, counted)
    return counts


def _value_kept(store: LiteStore) -> bool:
    return any(event.fact is not None and event.fact.object == "4512" for event in LiteFactLedger(store).events)


def test_perf001_a_forget_reads_chunk_ids_once_not_texts_per_source(store: LiteStore, reads: dict[str, int]) -> None:
    """Node tests/test_fact_retraction_deferred_cca.py::test_perf001_a_forget_reads_chunk_ids_once_not_texts_per_source.

    Red against the pre-fix `prepare_fact_erasure` (recall/fact_retraction.py: one
    `chunks_for_source` per gone source before deletion, and one per forgotten source in
    `redact`): fails on "the forget read whole chunks per source", with six calls for three
    sources.
    """
    result = forget_memory(store, [f"/A/{name}" for name in NAMES])

    assert reads.get("chunks_for_source", 0) == 0, f"the forget read whole chunks per source: {reads}"
    assert reads.get("chunk_ids_for_sources", 0) == 2, f"not one ids-only read before and one after: {reads}"
    assert result.facts_retracted == 1 and current(store) == []
    assert result.facts_redacted == 1 and not _value_kept(store)


def test_perf002_after_deletion_no_chunk_is_read_one_by_one(store: LiteStore, reads: dict[str, int]) -> None:
    """Node tests/test_fact_retraction_deferred_cca.py::test_perf002_after_deletion_no_chunk_is_read_one_by_one.

    Red against the pre-fix `chunk_gone` (recall/fact_retraction.py: `chunk_by_id` per target per
    cited erased chunk, cached by hand): fails on "a cited chunk was read one by one".
    """
    forget_memory(store, [f"/A/{name}" for name in NAMES])

    assert reads.get("chunk_by_id", 0) == 0, f"a cited chunk was read one by one: {reads}"


def test_control_a_namesake_keeps_its_own_fact(store: LiteStore) -> None:
    """Node tests/test_fact_retraction_deferred_cca.py::test_control_a_namesake_keeps_its_own_fact.

    Control for the batched `chunk_gone`: once a card citing B's own `x.md` chunk also states the
    fact, forgetting every memo of A retracts and erases nothing. Red, mutation of `chunk_gone`
    (recall/fact_retraction.py: its body replaced by `return True`, so every cited chunk counts
    as gone): fails on "a fact a served chunk supports was retracted".
    """
    cards = LiteEvidenceCardStore(store)
    card = evidence_card(DOOR, tenant=store.tenant, generation="g1", chunk="chunk-B-x.md", source="x.md")
    cards.put((card,))
    decision = ProvenanceController(
        tenant_id=store.tenant, generation_id="g1", cards=cards, ledger=LiteFactLedger(store),
        source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="deferred-test",
    ).apply_fact(FactApplicationRequest(DOOR, (card.card_id,), "rB"))
    assert decision.code == DecisionCode.DUPLICATE

    result = forget_memory(store, [f"/A/{name}" for name in NAMES])

    assert current(store) == ["4512"], f"a fact a served chunk supports was retracted: {result.message}"
    assert result.facts_retracted == 0 and _value_kept(store)


def test_present_chunk_ids_names_only_chunks_that_exist(store: LiteStore) -> None:
    """Node tests/test_fact_retraction_deferred_cca.py::test_present_chunk_ids_names_only_chunks_that_exist.

    `LiteStore.present_chunk_ids` answers which ids exist, nothing else. Red, mutation of
    `LiteStore.present_chunk_ids` (recall/lite/store.py: return `set(chunk_ids)` without the
    read): fails on "an absent chunk was reported present".
    """
    found = store.present_chunk_ids(["chunk-A-x.md", "chunk-B-x.md", "chunk-gone"])

    assert found == {"chunk-A-x.md", "chunk-B-x.md"}, f"an absent chunk was reported present: {found}"
    assert store.present_chunk_ids([]) == set()
