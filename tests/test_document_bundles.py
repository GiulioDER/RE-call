from datetime import datetime, timedelta, timezone

import pytest

from recall.calibration import Calibration
from recall.evidence import AnswerSlot, EvidencePolicy, build_evidence_bundle
from recall.retriever import (
    DocumentExpansionPolicy,
    StructuralExpansionPolicy,
    expand_retrieval_by_source,
    expand_retrieval_by_structure,
)
from recall.trust import trusted_search
from recall.trust_policy import TrustPolicy
from recall.types import (
    Chunk,
    Provenance,
    RetrievalDiagnostics,
    RetrievalResult,
    ScoredChunk,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)


NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _scored(cid: str, file: str, text: str, score: float, ordinal: int) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(cid, f"/corpus/{file}", text, {"file": file, "ord": ordinal}),
        score=score,
        indexed_at=NOW,
    )


def _retrieval(query: str, *hits: ScoredChunk) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        hits=list(hits),
        gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(stage_ms={"dense_retrieval": 1.0}),
    )


def _trusted(hit: ScoredChunk, verdict: str = "ok") -> TrustedHit:
    return TrustedHit(
        chunk=hit.chunk,
        cosine=hit.score,
        confidence=0.9,
        verdict=verdict,  # type: ignore[arg-type]
        provenance=Provenance(hit.chunk.source, hit.chunk.metadata["file"], hit.chunk.metadata["ord"], NOW),
        validity=Validity(None, None, None),
    )


def _trusted_result(*hits: TrustedHit) -> TrustedResult:
    return TrustedResult(
        query="why did the rollout change?",
        hits=list(hits),
        abstained=False,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(index_generation="gen-1"),
        calibration_id="cal-1",
        calibration_status="certified",
        tenant_id="tenant-1",
        generation_id="gen-1",
    )


def test_document_expansion_retrieves_distant_same_source_chunks() -> None:
    seed = _scored("a1", "decision.md", "The rollout owner was Ada.", 0.82, 1)
    other = _scored("b1", "other.md", "An unrelated rollout note.", 0.80, 0)
    distant = _scored("a4", "decision.md", "The rollout changed because the cache was retired.", 0.79, 4)
    scoped = {"decision.md": _retrieval("why did the rollout change?", seed, distant)}
    calls: list[str | None] = []

    def search(query: str, k: int, source: str | None = None) -> RetrievalResult:
        assert query == "why did the rollout change?"
        assert k == 3
        calls.append(source)
        return scoped[source or ""]

    expanded = expand_retrieval_by_source(
        _retrieval("why did the rollout change?", seed, other),
        search,
        DocumentExpansionPolicy(enabled=True, max_sources=1, chunks_per_source=3),
    )

    assert calls == ["decision.md"]
    assert [hit.chunk.id for hit in expanded.hits] == ["a1", "b1", "a4"]
    assert expanded.diagnostics.stage_ms["document_expansion_sources"] == 1.0


def test_document_expansion_is_adaptive_for_direct_questions() -> None:
    seed = _scored("a1", "decision.md", "Ada owns rollout.", 0.82, 1)
    calls: list[str | None] = []

    def search(_query: str, _k: int, source: str | None = None) -> RetrievalResult:
        calls.append(source)
        return _retrieval("who owns rollout?", seed)

    result = _retrieval("who owns rollout?", seed)
    assert expand_retrieval_by_source(
        result, search, DocumentExpansionPolicy(enabled=True)
    ) == result
    assert calls == []


def test_structural_expansion_keeps_neighbors_and_document_conclusion() -> None:
    seed = _scored("a1", "decision.md", "The rollout owner was Ada.", 0.82, 1)
    neighbor = _scored("a3", "decision.md", "The migration started after review.", 0.80, 3)
    conclusion = _scored(
        "a8", "decision.md", "Exception: the legacy client kept the old cache.", 0.78, 8
    )
    unrelated = _scored("a20", "decision.md", "Appendix metadata.", 0.77, 6)

    def search(query: str, k: int, source: str | None = None) -> RetrievalResult:
        assert query == "why did the rollout change?"
        assert source == "decision.md"
        assert k == 8
        return _retrieval(query, seed, neighbor, conclusion, unrelated)

    expanded = expand_retrieval_by_structure(
        _retrieval("why did the rollout change?", seed),
        search,
        StructuralExpansionPolicy(enabled=True, max_sources=1, chunks_per_source=8, radius=2),
    )

    assert [hit.chunk.id for hit in expanded.hits] == ["a1", "a3", "a8"]
    assert expanded.diagnostics.stage_ms["structural_expansion_sources"] == 1.0


def test_document_bundle_keeps_distant_sections_and_cross_document_coverage() -> None:
    a_late = _scored("a5", "decision.md", "Outcome: rollout succeeded.", 0.91, 5)
    b_first = _scored("b1", "comparison.md", "Comparison baseline.", 0.88, 1)
    a_first = _scored("a1", "decision.md", "Cause: cache was retired.", 0.87, 1)
    a_last = _scored("a9", "decision.md", "Appendix.", 0.86, 9)
    demoted = _scored("old", "old.md", "Old outcome.", 0.99, 0)
    result = _trusted_result(
        _trusted(a_late),
        _trusted(b_first),
        _trusted(a_first),
        _trusted(a_last),
        _trusted(demoted, "superseded"),
    )

    bundle = build_evidence_bundle(
        result,
        EvidencePolicy(max_items=3, bundle_mode="document", max_documents=2),
    )

    assert [item.chunk_id for item in bundle.items] == ["a1", "a5", "b1"]
    assert "old" not in {item.chunk_id for item in bundle.items}


def test_trusted_search_evaluates_expanded_chunks_before_evidence(monkeypatch) -> None:
    seed = _scored("a1", "decision.md", "The owner was Ada.", 0.82, 1)
    distant = _scored("a4", "decision.md", "The rollout changed because the cache was retired.", 0.79, 4)
    weak = _scored("a6", "decision.md", "An unrelated appendix.", 0.61, 6)

    class _Retriever:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def search(self, query: str, k: int = 5, source: str | None = None) -> RetrievalResult:
            assert query == "why did the rollout change?"
            if source is None:
                return _retrieval(query, seed)
            assert source == "decision.md"
            assert k == 3
            return _retrieval(query, seed, distant, weak)

    class _Store:
        tenant = "tenant-1"

        def generation_binding(self) -> dict[str, str]:
            return {
                "tenant_id": self.tenant,
                "generation_id": "gen-1",
                "pipeline_fingerprint": "p" * 64,
                "corpus_fingerprint": "c" * 64,
            }

        def supersession(self) -> tuple[dict[str, str], frozenset[str]]:
            return {}, frozenset()

    monkeypatch.setattr("recall.trust.HybridRetriever", _Retriever)
    result = trusted_search(
        _Store(),
        object(),
        "why did the rollout change?",
        k=1,
        calibration=Calibration(embedder="fixture", threshold=0.70, scale=0.05),
        policy=TrustPolicy.development(),
        document_expansion=DocumentExpansionPolicy(enabled=True, max_sources=1, chunks_per_source=3),
    )

    verdicts = {hit.chunk.id: hit.verdict for hit in result.hits}
    assert verdicts["a1"] == "ok"
    assert verdicts["a4"] == "ok"
    assert verdicts["a6"] == "low_confidence"
    assert result.diagnostics.stage_ms["document_expansion_sources"] == 1.0

    bundle = build_evidence_bundle(result, EvidencePolicy(bundle_mode="document", max_documents=1))
    assert "a4" in {item.chunk_id for item in bundle.items}
    assert "a6" not in {item.chunk_id for item in bundle.items}


def test_answer_slots_reject_partial_bundles_and_beam_covers_all_slots() -> None:
    misleading = _scored("near", "decision.md", "The rollout change was approved.", 0.95, 2)
    cause = _scored("cause", "decision.md", "The cache was retired after migration.", 0.84, 4)
    exception = _scored(
        "exception", "decision.md", "Exception: the legacy client kept the old cache.", 0.83, 8
    )
    result = _trusted_result(_trusted(misleading), _trusted(cause), _trusted(exception))
    slots = (
        AnswerSlot("cause", ("cache", "retired"), min_matches=2),
        AnswerSlot("exception", ("exception", "legacy client"), min_matches=2),
    )

    greedy = build_evidence_bundle(
        result,
        EvidencePolicy(
            max_items=2,
            bundle_mode="document",
            max_documents=1,
            answer_slots=slots,
        ),
    )
    beam = build_evidence_bundle(
        result,
        EvidencePolicy(
            max_items=2,
            bundle_mode="document",
            max_documents=1,
            answer_slots=slots,
            selection_mode="beam",
            beam_width=4,
        ),
    )

    assert greedy.decision == "answer"
    assert {item.chunk_id for item in greedy.items} == {"cause", "exception"}
    assert beam.decision == "answer"
    assert {item.chunk_id for item in beam.items} == {"cause", "exception"}

    partial = _trusted_result(_trusted(misleading), _trusted(cause))
    incomplete = build_evidence_bundle(partial, EvidencePolicy(answer_slots=slots))
    assert incomplete.decision == "abstain"
    assert incomplete.reason_code == "answer_slot_gap"
    assert incomplete.items == ()


@pytest.mark.parametrize("limit", ["items", "tokens-prefix", "tokens-beam"])
def test_required_slots_survive_the_final_bundle_limits(limit: str) -> None:
    """BUG-001: all emitted requirements must survive every final budget.

    Red proof: beb08a605, build_evidence_bundle returned an answer containing
    only the cause after the indicated limit removed the exception.
    """
    from recall.evidence import render_evidence_prompt

    cause = _trusted(_scored("cause", "cause.md", "Cause: retired cache.", 0.9, 1))
    exception = _trusted(_scored("exception", "exception.md", "Exception: legacy client.", 0.8, 2))
    result = _trusted_result(cause, exception)
    slots = (AnswerSlot("cause", ("retired",)), AnswerSlot("exception", ("legacy",)))

    class Characters:
        def count_tokens(self, text: str) -> int:
            return len(text)

    one = build_evidence_bundle(_trusted_result(cause), EvidencePolicy())
    token_budget = len(render_evidence_prompt(one)[1])
    policy = EvidencePolicy(
        answer_slots=slots,
        max_items=1 if limit == "items" else 2,
        max_tokens=token_budget if limit.startswith("tokens") else None,
        tokenizer=Characters() if limit.startswith("tokens") else None,
        selection_mode="beam" if limit == "tokens-beam" else "prefix",
    )
    bundle = build_evidence_bundle(result, policy)
    assert bundle.decision == "abstain", "a final budget removed a required answer component"
    assert bundle.reason_code == "answer_slot_gap"
    assert bundle.items == ()


@pytest.mark.parametrize("confidence", [0.5, 0.9, 1.0])
@pytest.mark.parametrize("cosine", [0.7, 1.0])
def test_slotless_beam_retains_a_supported_candidate(confidence: float, cosine: float) -> None:
    """NUM-001: a slotless beam must not penalize all evidence as irrelevant.

    Red proof: beb08a605, _beam_score charged three points for an empty slot
    specification, so every bounded candidate lost to the empty state.
    """
    from dataclasses import replace

    hit = replace(_trusted(_scored("one", "one.md", "Supported evidence.", cosine, 0)),
                  confidence=confidence)
    bundle = build_evidence_bundle(
        _trusted_result(hit), EvidencePolicy(max_items=1, selection_mode="beam")
    )
    assert bundle.decision == "answer", "empty slot requirements must not discard trusted evidence"
    assert [item.chunk_id for item in bundle.items] == ["one"]


def test_structural_expansion_collects_later_seeds_in_selected_sources() -> None:
    """BUG-003: a source cap limits searches, not seed ordinals in that source.

    Red proof: beb08a605 stopped after the first seed and omitted near-later.
    """
    first = _scored("first", "a.md", "First section.", 0.9, 1)
    other = _scored("other", "b.md", "Another source.", 0.85, 2)
    later = _scored("later", "a.md", "Later section.", 0.8, 10)
    neighbor = _scored("near-later", "a.md", "Neighbor.", 0.75, 9)
    far = _scored("far", "a.md", "Not adjacent.", 0.7, 30)
    terminal = _scored("terminal", "a.md", "Last retrieved candidate.", 0.6, 40)
    calls: list[str | None] = []

    def search(query: str, k: int, source: str | None = None) -> RetrievalResult:
        calls.append(source)
        return _retrieval(query, first, neighbor, far, terminal)

    expanded = expand_retrieval_by_structure(
        _retrieval("why did this change?", first, other, later), search,
        StructuralExpansionPolicy(enabled=True, max_sources=1, radius=1),
    )
    assert "near-later" in {hit.chunk.id for hit in expanded.hits}
    assert "far" not in {hit.chunk.id for hit in expanded.hits}
    assert calls == ["a.md"]
