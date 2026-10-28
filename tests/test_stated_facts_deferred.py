"""Deferred CCA fixes for PR 946 (stated-fact read checks), the parts that need no database.

Invariants and the failure each one catches:
- S1 `_settled` remembers an answer per call, keyword arguments included: two calls that differ only
  in a keyword are two calls, not one answer reused (CCA CODE-003).
- S2 `GenerationStore.pin_generation` inside a context already pinned to that generation asks the
  database nothing: one fact check enters it once per lookup, inside `recall_apply_fact`'s advisory
  lock, and the outer pin validated the generation on entry (CCA PERF-001).
- S3 `current_facts_memory` names the generation its stated-fact checks read: a promotion between
  the checks and the reply named the new generation beside the old one's facts (CCA DAT-001).

Node IDs, all in `tests/test_stated_facts_deferred.py`:
S1 `test_s1_a_keyword_argument_is_part_of_the_settled_call`,
S2 `test_s2_a_pin_already_held_asks_the_database_nothing`,
S3 `test_s3_current_facts_name_the_generation_their_checks_read`.

Red proof, 2026-10-28, on this workstation (no database), CCA deferred run 20261028T180444Z, by
`cca_checks tautology verify` against the pre-fix files; see that run's L5.6 record.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import recall_mcp.provenance as provenance
from recall.fact_retraction import _settled
from recall.generation_store import GenerationStore


def test_s1_a_keyword_argument_is_part_of_the_settled_call() -> None:
    settled = _settled(lambda x=0: x)
    assert settled(x=1) == 1
    assert settled(x=2) == 2, "a call differing only in a keyword argument reused another call's answer"


def _bare_store() -> GenerationStore:
    store = GenerationStore.__new__(GenerationStore)
    store._tenant = "t"
    store._fixed_generation = None
    store._pinned_generation = ContextVar("pinned_generation", default=None)
    store._pinned_corpus = ContextVar("pinned_corpus", default=None)
    return store


def test_s2_a_pin_already_held_asks_the_database_nothing() -> None:
    store = _bare_store()
    queries: list[object] = []

    def with_retry(fn: Any) -> Any:
        queries.append(fn)
        return ("fingerprint",)

    store._with_retry = with_retry  # type: ignore[method-assign]
    store._pinned_generation.set("gen_a")
    store._pinned_corpus.set(("gen_a", "fingerprint"))
    with store.pin_generation("gen_a") as generation_id:
        assert generation_id == "gen_a"
    assert queries == [], "a pin already held for this generation asked the database again"
    with store.pin_generation("gen_b") as generation_id:
        assert generation_id == "gen_b"
    assert len(queries) == 1, "a pin of another generation was not validated"


def test_s3_current_facts_name_the_generation_their_checks_read(monkeypatch) -> None:
    store = _bare_store()
    active = ["gen_v1"]
    store._active_generation_state = lambda: (active[0], "fp-" + active[0])  # type: ignore[method-assign]
    store.active_generation_id = lambda: active[0]  # type: ignore[method-assign]
    checked: list[str] = []

    def served_sources(target: Any, *, failures: list[str]) -> object:
        checked.append(target.generation_id)
        active[0] = "gen_v2"  # a promotion lands while the read is in progress
        return object()

    monkeypatch.setattr(provenance, "served_sources", served_sources)
    monkeypatch.setattr(provenance, "_fact_backends", lambda _store: (object(), object()))
    monkeypatch.setattr(provenance, "current_fact_listing", lambda *args: ([], []))
    monkeypatch.setattr(provenance, "listing_payload", lambda *args: {})
    result = provenance.current_facts_memory(store)
    assert checked == ["gen_v1"]
    named_checked = result["generation_id"] == "gen_v1"
    assert named_checked, "current facts named a generation their checks never read"
