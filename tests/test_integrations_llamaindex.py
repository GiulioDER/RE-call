"""RecallRetriever — the LlamaIndex adapter over the trust layer.

DB-less by construction: each test injects a fabricated ``TrustedResult`` (via ``search_fn`` or by
monkeypatching ``trusted_search``), so the mapping and the abstention behaviour are exercised
without Postgres. The behaviour unique to this adapter — abstention yields *no* nodes, not a
best-effort neighbour — is asserted directly.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("llama_index.core")

from recall.integrations.llamaindex import RecallRetriever  # noqa: E402
from recall.types import (  # noqa: E402
    Chunk,
    Provenance,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)

_INDEXED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _staleness() -> StalenessReport:
    return StalenessReport(stale=False, newest_indexed_at=None, age=None, max_age=timedelta(days=1))


def _hit(text: str, file: str, *, verdict: str = "ok") -> TrustedHit:
    return TrustedHit(
        chunk=Chunk(id=f"{file}#0", source="memory", text=text, metadata={"file": file, "topic": "limits"}),
        cosine=0.78,
        confidence=1.0,
        verdict=verdict,  # type: ignore[arg-type]
        provenance=Provenance(source="memory", file=file, ord=0, indexed_at=_INDEXED_AT),
        validity=Validity(valid_from=None, valid_until=None, superseded_by=None),
    )


def _result(hits: list[TrustedHit], *, abstained: bool = False, reason: str = "") -> TrustedResult:
    return TrustedResult(
        query="q",
        hits=hits,
        abstained=abstained,
        reason=reason,
        calibrated=True,
        gap_warning=abstained,
        staleness=_staleness(),
    )


def test_maps_ok_hits_to_nodes_with_trust_metadata() -> None:
    result = _result([_hit("rate limit is 500 rps", "rate_v2.md")])
    retriever = RecallRetriever(search_fn=lambda _q: result)

    nodes = retriever.retrieve("how many rps?")

    assert len(nodes) == 1
    nws = nodes[0]
    assert nws.node.get_content() == "rate limit is 500 rps"
    assert nws.score == 0.78  # cosine is the retriever score
    assert nws.node.metadata["recall_verdict"] == "ok"
    assert nws.node.metadata["recall_confidence"] == 1.0
    assert nws.node.metadata["recall_cosine"] == 0.78
    assert nws.node.metadata["file"] == "rate_v2.md"
    assert nws.node.metadata["indexed_at"] == _INDEXED_AT.isoformat()
    assert nws.node.metadata["topic"] == "limits"  # original chunk metadata preserved


def test_preserves_hit_order() -> None:
    result = _result([_hit("first", "a.md"), _hit("second", "b.md")])
    retriever = RecallRetriever(search_fn=lambda _q: result)

    assert [n.node.get_content() for n in retriever.retrieve("q")] == ["first", "second"]


def test_abstention_returns_no_nodes() -> None:
    result = _result([], abstained=True, reason="no hit above the calibrated threshold")
    retriever = RecallRetriever(search_fn=lambda _q: result)

    assert retriever.retrieve("how do we handle penguins on mars?") == []


def test_abstention_reason_surfaced_when_requested() -> None:
    result = _result([], abstained=True, reason="no hit above the calibrated threshold")
    retriever = RecallRetriever(search_fn=lambda _q: result, return_abstention_reason=True)

    nodes = retriever.retrieve("q")

    assert len(nodes) == 1
    assert nodes[0].node.get_content() == ""
    assert nodes[0].node.metadata["recall_abstained"] is True
    assert nodes[0].node.metadata["recall_reason"] == "no hit above the calibrated threshold"


def test_from_store_wires_trusted_search(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_trusted_search(store, embedder, query, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(store=store, embedder=embedder, query=query, **kwargs)
        return _result([_hit("hit", "f.md")])

    monkeypatch.setattr("recall.integrations.llamaindex.trusted_search", fake_trusted_search)

    retriever = RecallRetriever.from_store("STORE", "EMBEDDER", k=3, entailment="JUDGE")
    nodes = retriever.retrieve("how many rps?")

    assert captured["store"] == "STORE"
    assert captured["embedder"] == "EMBEDDER"
    assert captured["query"] == "how many rps?"
    assert captured["k"] == 3
    assert captured["entailment"] == "JUDGE"
    assert nodes[0].node.metadata["recall_verdict"] == "ok"
