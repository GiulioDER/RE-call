"""`recall_reasoning_query(mode="evidence_proof")` driven through the REGISTERED tool.

`tests/test_proof_obligations.py` exercises the proof core as a library. It cannot see the MCP
wrapper, which owns three guarantees the core deliberately leaves to its caller: the repair
retrieval keeps the caller's source scope, a repair that lands in another tenant or another index
generation is refused, and the receipt exposes decisions and identifiers but no provider text.
These tests await the tool coroutine `build_server()` registers, with retrieval stubbed at
`reasoning_query` and the proof provider scripted, so they need no database and no network.

Red proof receipts, each run against a deliberate mutation of `recall_mcp/server.py` and restored
before the green run:

* ``test_a_repair_in_another_tenant_is_refused`` and
  ``test_a_repair_in_another_generation_is_refused``: deleting the tenant and generation comparison
  inside ``execute_proof.repair`` makes both fail on ``decision == "abstain"``, because the foreign
  evidence is merged and the proof is accepted as sufficient.
* ``test_the_repair_keeps_the_callers_source_scope``: passing ``source=None`` in place of
  ``source=source`` inside ``execute_reasoning`` makes it fail on the recorded source scope.
* ``test_the_receipt_carries_no_provider_text``: adding ``"assessment": proof.final`` to
  ``_proof_payload`` makes it fail on the exact receipt key set.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import recall_mcp.server as server_module
from recall.evidence import EvidenceBundle, EvidenceItem
from recall_mcp.server import build_server
from recall_mcp.settings import Settings


_RECEIPT_KEYS = {
    "decision",
    "reason_code",
    "repair_attempted",
    "model_calls",
    "evidence_chunk_ids",
    "initial_missing_slot_ids",
    "final_missing_slot_ids",
}


def _item(chunk_id: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        text=text,
        source="alpha.md",
        ordinal=1,
        indexed_at=None,
        valid_from=None,
        valid_until=None,
        cosine=0.9,
        confidence=0.9,
    )


def _bundle(query: str, *items: EvidenceItem) -> EvidenceBundle:
    return EvidenceBundle(
        query=query,
        decision="answer",
        reason_code=None,
        decision_state="supported",
        calibrated=True,
        stale=False,
        embedding_profile="test-profile",
        retrieval_profile="test-retrieval",
        index_generation="generation-1",
        items=items,
        trust_state="trusted",
    )


class _Retriever:
    """Stands in for `reasoning_query`, recording every call and serving scripted evidence."""

    def __init__(self, *, repair_tenant: str = "acme", repair_generation: str = "gen-1") -> None:
        self.calls: list[dict[str, object]] = []
        self._repair_tenant = repair_tenant
        self._repair_generation = repair_generation

    def __call__(self, store, embedder, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        if len(self.calls) == 1:
            evidence = _bundle(query, _item("one", "Project alpha ships."))
            tenant, generation = "acme", "gen-1"
        else:
            evidence = _bundle(query, _item("two", "Release on 2026-10-15."))
            tenant, generation = self._repair_tenant, self._repair_generation
        payload = {"outcome": "retrieval_only", "answer": None, "tenant_id": tenant}
        return SimpleNamespace(
            trusted_evidence=evidence,
            tenant_id=tenant,
            generation_id=generation,
            to_dict=lambda: dict(payload),
        )


def _assessment(*, date_supported: bool, repair: object | None) -> str:
    support: list[dict[str, object]] = [
        {"slot_id": "project", "chunk_id": "one", "start": 0, "end": 13, "quote": "Project alpha"}
    ]
    if date_supported:
        support.append(
            {"slot_id": "release_date", "chunk_id": "two", "start": 11, "end": 21, "quote": "2026-10-15"}
        )
    return json.dumps(
        {
            "schema_version": 1,
            "slots": [
                {"slot_id": "project", "kind": "entity", "requirement": "project", "required": True},
                {"slot_id": "release_date", "kind": "time", "requirement": "date", "required": True},
            ],
            "support": support,
            "repair": repair,
        }
    )


class _Provider:
    """First call asks for one repair, second call supports every slot."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, system: str, user: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return _assessment(
                date_supported=False,
                repair={"query": "alpha release date", "missing_slot_ids": ["release_date"]},
            )
        return _assessment(date_supported=True, repair=None)


def _invoke(monkeypatch, retriever: _Retriever, provider: object | None, **kwargs) -> dict:
    monkeypatch.setattr(server_module, "get_access_token", lambda: None)
    monkeypatch.setattr(server_module, "reasoning_query", retriever)
    state = {
        "store": SimpleNamespace(tenant="acme"),
        "embedder": object(),
        "proof_provider": provider,
        "settings": Settings.from_env({}),
    }
    tools = {t.name: t for t in build_server()._tool_manager.list_tools()}
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))

    async def run() -> str:
        return await tools["recall_reasoning_query"].fn(
            ctx=ctx, query="When does project alpha release?", mode="evidence_proof", **kwargs
        )

    return json.loads(asyncio.run(run()))


def test_a_repaired_proof_is_sufficient_and_the_receipt_says_so(monkeypatch) -> None:
    retriever = _Retriever()
    provider = _Provider()

    result = _invoke(monkeypatch, retriever, provider)
    proof = result["proof_obligations"]

    assert proof["decision"] == "sufficient"
    assert proof["reason_code"] is None
    assert proof["repair_attempted"] is True
    assert proof["model_calls"] == 2 == provider.calls
    assert proof["evidence_chunk_ids"] == ["one", "two"]
    assert proof["initial_missing_slot_ids"] == ["release_date"]
    assert proof["final_missing_slot_ids"] == []
    assert [call["query"] for call in retriever.calls] == [
        "When does project alpha release?",
        "alpha release date",
    ]
    assert all(call["mode"] == "retrieval_only" for call in retriever.calls), (
        "proof mode must retrieve without invoking the answer provider"
    )


def test_the_repair_keeps_the_callers_source_scope(monkeypatch) -> None:
    retriever = _Retriever()

    _invoke(monkeypatch, retriever, _Provider(), source="alpha.md")

    assert [call["source"] for call in retriever.calls] == ["alpha.md", "alpha.md"]


def test_a_repair_in_another_tenant_is_refused(monkeypatch) -> None:
    retriever = _Retriever(repair_tenant="globex")

    proof = _invoke(monkeypatch, retriever, _Provider())["proof_obligations"]

    assert proof["decision"] == "abstain"
    assert proof["reason_code"] == "proof_repair_failure"
    assert proof["evidence_chunk_ids"] == ["one"]


def test_a_repair_in_another_generation_is_refused(monkeypatch) -> None:
    retriever = _Retriever(repair_generation="gen-2")

    proof = _invoke(monkeypatch, retriever, _Provider())["proof_obligations"]

    assert proof["decision"] == "abstain"
    assert proof["reason_code"] == "proof_repair_failure"
    assert proof["evidence_chunk_ids"] == ["one"]


def test_without_a_proof_provider_the_tool_abstains_without_a_model_call(monkeypatch) -> None:
    retriever = _Retriever()

    proof = _invoke(monkeypatch, retriever, None)["proof_obligations"]

    assert proof["decision"] == "abstain"
    assert proof["reason_code"] == "no_proof_provider"
    assert proof["model_calls"] == 0
    assert len(retriever.calls) == 1


def test_the_receipt_carries_no_provider_text(monkeypatch) -> None:
    result = _invoke(monkeypatch, _Retriever(), _Provider())

    assert set(result["proof_obligations"]) == _RECEIPT_KEYS
    assert result["answer"] is None
    # Slot requirements are provider-authored text. Evidence text is retrieval output and may
    # legitimately appear in the real response, so it is not asserted absent here.
    assert "requirement" not in json.dumps(result)
