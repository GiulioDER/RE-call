"""A generation store reports the generation it reads, and fact application binds to it.

Invariant: `GenerationStore.generation_id` is the generation the store is serving (the pinned
one inside `snapshot()` or `pin_generation`, else the fixed one, else the active one).

Failure mode caught: `PgVectorStore.__init__` defaults `generation_id="legacy"` and
`GenerationStore.__init__` never passes one, so the inherited property answered `"legacy"` on
every production store. `recall_mcp.provenance.apply_fact_memory` builds its
`ProvenanceController` from that property, while the evidence cards it checks carry the real
generation, so `recall_apply_fact` refused every card with `GENERATION_MISMATCH`
(`recall/provenance_controller.py`, `card.generation_id != self.generation_id`).

Red proof, 2026-10-01, against origin/master 5a8fcaba (no `generation_id` override on
`GenerationStore`):

* `test_fact_application_binds_the_controller_to_the_served_generation` failed at its final
  assertion with `'legacy' == 'gen-1'`.
* `test_the_property_follows_the_pin` failed at its first assertion with `'legacy' == 'gen-1'`.

The database form, against a promoted generation, is
`tests/test_generations.py::test_generation_store_reports_the_generation_it_reads`.
"""

from __future__ import annotations

from contextvars import ContextVar

import pytest

import recall_mcp.provenance as provenance
from recall.generation_store import GenerationStore


class _Refused(Exception):
    """Raised by the controller double once it has recorded how it was built."""


class _PinnedGenerationStore(GenerationStore):
    """A real `GenerationStore` with no database, pinned the way `snapshot()` pins it."""

    def __init__(self, pinned: str | None = "gen-1") -> None:  # does not call super().__init__
        self._tenant = "acme"
        self._dsn = "postgresql://unused/never-connected"
        # What `PgVectorStore.__init__` stores when `GenerationStore` calls it: the default.
        self._index_generation_id = "legacy"
        self._pinned_generation = ContextVar("pinned_generation", default=pinned)
        self._pinned_corpus = ContextVar("pinned_corpus", default=None)
        self._fixed_generation = None

    def active_generation_id(self) -> str:
        return "gen-active"


def test_the_property_follows_the_pin() -> None:
    assert _PinnedGenerationStore("gen-1").generation_id == "gen-1"
    assert _PinnedGenerationStore(None).generation_id == "gen-active"


def test_fact_application_binds_the_controller_to_the_served_generation(monkeypatch) -> None:
    built: dict[str, object] = {}

    class _Controller:
        def __init__(self, **kwargs: object) -> None:
            built.update(kwargs)

        def apply_fact(self, _request: object) -> object:
            raise _Refused

    monkeypatch.setattr(provenance, "ProvenanceController", _Controller)
    monkeypatch.setattr(provenance, "PostgresFactLedger", lambda *_a, **_k: object())
    monkeypatch.setattr(provenance, "PostgresEvidenceCardStore", lambda *_a, **_k: object())

    with pytest.raises(_Refused):
        provenance.apply_fact_memory(
            _PinnedGenerationStore("gen-1"),
            embedder=object(),  # type: ignore[arg-type]
            claim={"subject": "service", "predicate": "port", "object": 8080},
            evidence_card_ids=["card-1"],
            request_id="request-1",
            writer="agent",
        )

    assert built["generation_id"] == "gen-1"


class _NoActiveGenerationStore(_PinnedGenerationStore):
    def active_generation_id(self) -> str:
        from recall.generations import NoActiveGeneration

        raise NoActiveGeneration("tenant 'acme' has no active generation")


def test_current_facts_still_answers_before_any_generation_is_promoted(monkeypatch) -> None:
    """The fact ledger is tenant scoped, so it answers even when no generation is active.

    Invariant: `recall_current_facts` returns the ledger projection, with `generation_id` null
    when the tenant has no active generation, rather than raising or claiming `"legacy"`.
    Failure mode caught: once `GenerationStore.generation_id` stopped answering the `"legacy"`
    default, `current_facts_memory` raised `NoActiveGeneration` out of the tool on such a tenant.

    Red proof, 2026-10-01, against this branch before the `current_facts_memory` change: the call
    raised `recall.generations.NoActiveGeneration` instead of returning.
    """

    class _Ledger:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def current(self, **_k: object) -> list[object]:
            return []

    monkeypatch.setattr(provenance, "PostgresFactLedger", _Ledger)

    result = provenance.current_facts_memory(_NoActiveGenerationStore(None))

    assert result["generation_id"] is None
    assert result["facts"] == []
