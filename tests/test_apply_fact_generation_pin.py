"""One `recall_apply_fact` call reads one generation, even when a promotion lands during it.

Invariant: `recall_mcp.provenance.apply_fact_memory` on a `GenerationStore` runs inside the
store's own `snapshot()`, so the controller's generation, the fresh search's generation, the
card revalidation reads and the served-sources lookup all see the one active pointer read at the
start of the call.

Failure mode caught: the controller was built from `store.generation_id`, a fresh read of the
active pointer, and nothing pinned it. The fresh-search recovery (`_retrieve_trusted`) opens its
own snapshot and read the pointer again, so a promotion between the two reads refused a valid
fresh card of the new generation as `GENERATION_MISMATCH` (the controller compares card and
controller generation by equality), or recorded a fact under one generation after checking its
content in another. Found in the CCA audit of PR 852, which made the binding live.

Red proof, 2026-10-28, against origin/master 7de7ddcea (controller built from an unpinned
`store.generation_id`):

* `tests/test_apply_fact_generation_pin.py::test_a_promotion_during_one_call_is_not_seen_by_it`
  failed at its first assertion: `decision_code` was `GENERATION_MISMATCH`, the controller having
  read `gen-A` and the fresh search `gen-B`.

The lite and legacy stores are not entered: a lite `snapshot()` is a read transaction rolled back
on exit, around fact-ledger writes on the same connection.
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import recall_mcp.provenance as provenance
import recall_mcp.retrieval as retrieval
from recall.generation_store import GenerationStore
from recall.provenance_controller import EvidenceCardStore, InMemoryFactLedger, source_digest
from recall.types import AtomicFact, Chunk, EvidenceCard

CLAIM = {"subject": "service:api", "predicate": "port", "object": 8080}
TEXT = "The api service listens on port 8080."


class _PromotingGenerationStore(GenerationStore):
    """A `GenerationStore` with no database whose active pointer moves after its first read."""

    def __init__(self) -> None:  # does not call super().__init__
        self._tenant = "acme"
        self._dsn = "postgresql://unused/never-connected"
        self._index_generation_id = "legacy"
        self._pinned_generation = ContextVar("pinned_generation", default=None)
        self._pinned_corpus = ContextVar("pinned_corpus", default=None)
        self._fixed_generation = None
        self.pointer_reads = 0
        #: Every generation a step of the call read, in order.
        self.seen: list[tuple[str, str]] = []

    def _active_generation_state(self) -> tuple[str, str]:
        self.pointer_reads += 1
        # A promotion lands right after the first read of the active pointer.
        return ("gen-A" if self.pointer_reads == 1 else "gen-B"), "c" * 64

    def active_generation_id(self) -> str:
        return self._active_generation_state()[0]

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        self.seen.append(("chunk_by_id", self.generation_id))
        return Chunk(
            id=chunk_id,
            source="owner.md",
            text=TEXT,
            metadata={"recall_graph": {"facts": [dict(CLAIM)]}},
        )


def _card(generation_id: str) -> EvidenceCard:
    now = datetime.now(UTC)
    return EvidenceCard(
        card_id="",
        chunk_id="owner.md#0",
        source="owner.md",
        source_digest=source_digest(TEXT),
        valid_from=None,
        valid_until=None,
        first_indexed_at=now - timedelta(days=1),
        indexed_at=now - timedelta(hours=1),
        tenant_id="acme",
        generation_id=generation_id,
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_id="cal-a",
        calibration_status="certified",
        trust_state="trusted",
        verdict="ok",
        confidence=0.99,
        rank=1,
        structured_facts=(AtomicFact.from_payload(dict(CLAIM)),),
    )


def test_a_promotion_during_one_call_is_not_seen_by_it(monkeypatch) -> None:
    store = _PromotingGenerationStore()
    ledger = InMemoryFactLedger()
    cards = EvidenceCardStore()

    def fake_retrieve_trusted(store_arg, *_args, **_kwargs):
        # What the real one does first: enter the store's snapshot and name its generation.
        with retrieval._searched_generation(store_arg) as generation:
            store.seen.append(("fresh_search", generation))
            return SimpleNamespace(result=generation)

    def served(store_arg):
        def lookup(names):
            store.seen.append(("served_sources", store_arg.generation_id))
            return names

        return lookup

    monkeypatch.setattr(provenance, "_fact_backends", lambda _store: (ledger, cards))
    monkeypatch.setattr(retrieval, "_retrieve_trusted", fake_retrieve_trusted)
    monkeypatch.setattr(provenance, "cards_from_trusted_result", lambda gen: (_card(gen),))
    monkeypatch.setattr(provenance, "register_evidence_cards", lambda found, store: cards.put(found))
    monkeypatch.setattr(provenance, "served_sources", served)

    result = provenance.apply_fact_memory(
        store,
        embedder=object(),  # type: ignore[arg-type]
        claim=CLAIM,
        evidence_card_ids=["card-the-caller-no-longer-has"],
        request_id="request-1",
        writer="agent",
    )

    assert result["decision_code"] != "GENERATION_MISMATCH", (result, store.seen)
    assert result["allowed"] is True, (result, store.seen)
    assert result["retried"] is True
    assert ("fresh_search", "gen-A") in store.seen
    assert {generation for _step, generation in store.seen} == {"gen-A"}, store.seen
    assert store.pointer_reads == 1
