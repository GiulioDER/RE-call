from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

from recall.retriever import DocumentExpansionPolicy, StructuralExpansionPolicy
from recall.types import (
    Chunk,
    Provenance,
    RetrievalDiagnostics,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)
from recall_mcp import service
from scripts.run_live_document_expansion_audit import _fact_covered, _score_arm, _summarize
from scripts.run_live_tty_graph_precision import _command


NOW = datetime(2026, 9, 13, tzinfo=UTC)


def _trusted(chunk_id: str, source: str, ordinal: int, text: str) -> TrustedHit:
    return TrustedHit(
        chunk=Chunk(chunk_id, source, text, {"file": source, "ord": ordinal}),
        cosine=0.8,
        confidence=0.9,
        verdict="ok",
        provenance=Provenance(source, source, ordinal, NOW),
        validity=Validity(None, None, None),
    )


def _trusted_result(*hits: TrustedHit) -> TrustedResult:
    return TrustedResult(
        query="question",
        hits=list(hits),
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(
            embedding_profile="voyage:voyage-4",
            retrieval_profile="fast",
            index_generation="generation-one",
        ),
        calibration_id="calibration-one",
        calibration_status="certified",
        tenant_id="memory",
        generation_id="generation-one",
    )


def test_document_expansion_audit_requires_generation_pin(monkeypatch) -> None:
    """The private treatment payload requires both its flag and a generation pin.

    Red proof node ``document-expansion-gate-01`` changes the conjunction to a disjunction. The
    flag-only assertion then fails at the intended boundary.
    """
    monkeypatch.delenv("RECALL_BENCHMARK_PIN", raising=False)
    monkeypatch.setenv("RECALL_BENCHMARK_DOCUMENT_EXPANSION_AUDIT", "1")
    assert service._document_expansion_benchmark_audit_enabled() is False
    monkeypatch.setenv("RECALL_BENCHMARK_PIN", "1")
    assert service._document_expansion_benchmark_audit_enabled() is True


def test_document_expansion_audit_uses_pinned_vector_and_existing_policies(monkeypatch) -> None:
    """Paired arms reuse the served vector and exercise source and structural policies.

    Red proof node ``document-expansion-policies-01`` restores adaptive query gating. The direct
    query treatment would then be skipped in production and this policy assertion fails.
    """
    doc = _trusted_result(
        _trusted("late", "recall/gold.md", 2, "later fact"),
        _trusted("other", "recall/other.md", 0, "other"),
        _trusted("early", "recall/gold.md", 0, "early fact"),
    )
    structural = _trusted_result(_trusted("neighbor", "recall/gold.md", 1, "neighbor fact"))
    seen: list[object] = []

    def fake_search(**kwargs):
        assert kwargs["embedder"].embed_query("question") == [0.2, 0.8]
        if "document_expansion" in kwargs:
            seen.append(kwargs["document_expansion"])
            return doc
        seen.append(kwargs["structural_expansion"])
        return structural

    class Inner:
        name = "voyage:voyage-4"
        dim = 2

    class Store:
        generation_id = "generation-one"

    monkeypatch.setattr(service, "trusted_search", fake_search)
    monkeypatch.setattr(service, "_build_reranker", lambda *_args, **_kwargs: None)
    payload = service._document_expansion_benchmark_audit_payload(
        Store(), Inner(), "question", [0.2, 0.8], None, 3, None, None, service.FAST_PROFILE
    )

    assert isinstance(seen[0], DocumentExpansionPolicy)
    assert seen[0].relational_query_only is False
    assert seen[0].max_sources == 2
    assert seen[0].chunks_per_source == 8
    assert isinstance(seen[1], StructuralExpansionPolicy)
    assert seen[1].relational_query_only is False
    assert [
        (item["source"], item["ordinal"])
        for item in payload["arms"]["document_bundle"]["items"]
    ] == [("recall/gold.md", 0), ("recall/gold.md", 2), ("recall/other.md", 0)]
    assert len(payload["diagnostic_pools"]["document"]) == 3


def test_essential_fact_coverage_requires_the_gold_source() -> None:
    """Matching words from an adjacent memo cannot satisfy an essential fact label.

    Red proof node ``essential-fact-source-01`` removes the source equality check. The first
    assertion then turns true and fails.
    """
    fact = {"terms": ["settlement", "never runs"], "min_matches": 2}
    wrong = [{"source": "recall/adjacent.md", "text": "settlement never runs"}]
    right = [{"source": "recall/gold.md", "text": "settlement never runs"}]

    assert _fact_covered(fact, wrong, "recall/gold.md") is False
    assert _fact_covered(fact, right, "recall/gold.md") is True


def _arm(decision: str, source: str | None, text: str = "") -> dict[str, object]:
    items = [] if source is None else [{"source": source, "text": text}]
    return {"decision": decision, "reason_code": None, "items": items, "retrieval_ms": 10.0}


def test_document_expansion_summary_separates_fact_gain_and_false_answers() -> None:
    """Treatment gains and unanswerable regressions are both visible in one summary.

    Red proof node ``document-expansion-summary-01`` counts any source hit as fact complete. The
    baseline complete query assertion then fails because its gold chunk lacks the essential fact.
    """
    label = {
        "source": "recall/gold.md",
        "facts": [{"name": "answer", "terms": ["right fact"], "min_matches": 1}],
    }
    answerable_arms = {
        "baseline": _arm("answer", "recall/gold.md", "wrong passage"),
        "document_retrieval": _arm("answer", "recall/gold.md", "right fact"),
        "document_bundle": _arm("answer", "recall/gold.md", "right fact"),
        "structural_bundle": _arm("answer", "recall/gold.md", "wrong passage"),
    }
    unanswerable_arms = {
        "baseline": _arm("abstain", None),
        "document_retrieval": _arm("answer", "recall/near.md", "near"),
        "document_bundle": _arm("answer", "recall/near.md", "near"),
        "structural_bundle": _arm("abstain", None),
    }
    rows = [
        {
            "label": label,
            "arms": answerable_arms,
            "scores": {
                name: _score_arm(arm, label) for name, arm in answerable_arms.items()
            },
            "pool_scores": {
                "document": _score_arm(answerable_arms["document_bundle"], label),
                "structural": _score_arm(answerable_arms["structural_bundle"], label),
            },
        },
        {
            "label": None,
            "arms": unanswerable_arms,
            "scores": {
                name: _score_arm(arm, None) for name, arm in unanswerable_arms.items()
            },
            "pool_scores": {
                "document": _score_arm(unanswerable_arms["document_bundle"], None),
                "structural": _score_arm(unanswerable_arms["structural_bundle"], None),
            },
        },
    ]

    summary = _summarize(rows)

    assert summary["arms"]["baseline"]["complete_queries"] == 0
    assert summary["arms"]["document_bundle"]["complete_queries"] == 1
    assert summary["arms"]["document_bundle"]["unanswerable_answers"] == 1
    assert summary["arms"]["structural_bundle"]["unanswerable_answers"] == 0


def test_document_expansion_command_is_private_and_uses_committed_checkout(monkeypatch) -> None:
    """Only the treatment runner enables the private expansion audit.

    Red proof node ``document-expansion-command-01`` omits the treatment environment assignment.
    The audited command assertion then fails while the ordinary command remains unchanged.
    """
    code_root = "/home/sentiment/recall-repos/document-expansion-audit"
    monkeypatch.setenv("RECALL_BENCHMARK_REMOTE_CODE_ROOT", code_root)
    ordinary = _command(
        "memory", "voyage:voyage-4", "/srv/memory", "fast", "combined", "none", 1, 32, 0.10
    )[-1]
    audited = _command(
        "memory",
        "voyage:voyage-4",
        "/srv/memory",
        "fast",
        "combined",
        "none",
        1,
        32,
        0.10,
        "generation-one",
        benchmark_document_expansion_audit=True,
    )[-1]

    assert "RECALL_BENCHMARK_DOCUMENT_EXPANSION_AUDIT" not in ordinary
    assert "RECALL_BENCHMARK_DOCUMENT_EXPANSION_AUDIT=1" in audited
    assert f"cd {code_root}" in audited


def test_essential_fact_labels_cover_exactly_the_source_gold_answers() -> None:
    """Every answerable query has reviewable essential facts bound to its gold source.

    Red proof node ``essential-gold-coverage-01`` removes one label. Exact id equality then fails.
    """
    root = Path(__file__).resolve().parents[1]
    queries = json.loads(
        (root / "docs/preregistrations/2026-09-13-memory-queries-source-gold.json").read_text(
            encoding="utf-8"
        )
    )
    labels = json.loads(
        (root / "docs/preregistrations/2026-09-13-memory-essential-facts.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        query["id"]: query["relevant_files"][0] for query in queries if query["answerable"]
    }
    actual = {label["query_id"]: label["source"] for label in labels}

    assert actual == expected
    assert sum(len(label["facts"]) for label in labels) == 25
    assert all(
        1 <= fact["min_matches"] <= len(fact["terms"])
        for label in labels
        for fact in label["facts"]
    )
