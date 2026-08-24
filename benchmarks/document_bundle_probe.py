"""Preregistered offline probe for adaptive document expansion and bundle selection."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import cast

from recall.calibration import Calibration
from recall.evidence import EvidencePolicy, build_evidence_bundle
from recall.retriever import DocumentExpansionPolicy, expand_retrieval_by_source
from recall.trust import evaluate
from recall.types import (
    Chunk,
    RetrievalDiagnostics,
    RetrievalResult,
    ScoredChunk,
    StalenessReport,
    TrustedResult,
)

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)
CALIBRATION = Calibration(embedder="fixture", threshold=0.70)


def _hit(cid: str, file: str, text: str, score: float, ordinal: int) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(cid, f"/fixture/{file}", text, {"file": file, "ord": ordinal}),
        score=score,
        indexed_at=NOW,
    )


def _raw(query: str, hits: list[ScoredChunk]) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        hits=hits,
        gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(index_generation="fixture"),
    )


def _trusted(result: RetrievalResult) -> TrustedResult:
    return evaluate(result, {}, CALIBRATION, NOW)


def _run_case(
    query: str,
    initial: list[ScoredChunk],
    scoped: dict[str, list[ScoredChunk]],
    required: set[str],
) -> dict[str, object]:
    raw = _raw(query, initial)

    def search(_query: str, _k: int, source: str | None = None) -> RetrievalResult:
        return _raw(query, list(scoped.get(source or "", ())))

    started = time.perf_counter()
    baseline = build_evidence_bundle(
        _trusted(raw),
        EvidencePolicy(max_items=5, bundle_mode="retrieval"),
    )
    baseline_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    expanded = expand_retrieval_by_source(
        raw,
        search,
        DocumentExpansionPolicy(enabled=True, max_sources=2, chunks_per_source=3),
    )
    treatment = build_evidence_bundle(
        _trusted(expanded),
        EvidencePolicy(max_items=5, bundle_mode="document", max_documents=2),
    )
    treatment_ms = (time.perf_counter() - started) * 1000.0
    baseline_ids = {item.chunk_id for item in baseline.items}
    treatment_ids = {item.chunk_id for item in treatment.items}
    return {
        "query": query,
        "required": sorted(required),
        "baseline_ids": sorted(baseline_ids),
        "treatment_ids": sorted(treatment_ids),
        "baseline_complete": required <= baseline_ids,
        "treatment_complete": required <= treatment_ids,
        "baseline_answered": baseline.decision == "answer",
        "treatment_answered": treatment.decision == "answer",
        "treatment_trust_state": treatment.trust_state,
        "baseline_ms": baseline_ms,
        "treatment_ms": treatment_ms,
    }


def main() -> None:
    a1 = _hit("a1", "rollout.md", "The owner was Ada.", 0.91, 1)
    a4 = _hit("a4", "rollout.md", "The rollout changed because the cache was retired.", 0.86, 4)
    a7 = _hit("a7", "rollout.md", "The final result was a successful migration.", 0.84, 7)
    b1 = _hit("b1", "baseline.md", "The baseline used the old cache.", 0.88, 1)
    b4 = _hit("b4", "baseline.md", "The baseline failed during migration.", 0.83, 4)
    c1 = _hit("c1", "limits.md", "The limit is 100 requests per minute.", 0.90, 1)
    c4 = _hit("c4", "limits.md", "The limit applies after authentication.", 0.85, 4)
    near = _hit("near", "near-miss.md", "A similar but unrelated policy.", 0.71, 1)
    near2 = _hit("near2", "near-miss.md", "The unrelated policy is deprecated.", 0.69, 4)

    cases = [
        _run_case("why did the rollout change?", [a1], {"rollout.md": [a1, a4]}, {"a1", "a4"}),
        _run_case("how did the migration end?", [a1], {"rollout.md": [a1, a7]}, {"a1", "a7"}),
        _run_case("what was the rollout result?", [a1], {"rollout.md": [a1, a7]}, {"a1", "a7"}),
        _run_case("when is the limit active?", [c1], {"limits.md": [c1, c4]}, {"c1", "c4"}),
        _run_case("who owns rollout?", [a1], {"rollout.md": [a1, a4]}, {"a1"}),
        _run_case("what is the request limit?", [c1], {"limits.md": [c1, c4]}, {"c1"}),
        _run_case("what is the baseline?", [b1], {"baseline.md": [b1, b4]}, {"b1"}),
        _run_case("compare rollout and baseline", [a1, b1], {"rollout.md": [a1, a4], "baseline.md": [b1, b4]}, {"a1", "b1"}),
        _run_case("compare migration results", [a1, b1], {"rollout.md": [a1, a7], "baseline.md": [b1, b4]}, {"a7", "b4"}),
        _run_case("why did the unicorn policy change?", [near], {"near-miss.md": [near, near2]}, set()),
        _run_case("what about penguins on mars?", [], {}, set()),
        _run_case("why did the absent policy change?", [], {}, set()),
    ]
    distant = cases[:4]
    unanswerable = cases[9:]
    result = {
        "cases": len(cases),
        "distant_baseline_complete": sum(bool(case["baseline_complete"]) for case in distant),
        "distant_treatment_complete": sum(bool(case["treatment_complete"]) for case in distant),
        "overall_baseline_complete": sum(bool(case["baseline_complete"]) for case in cases),
        "overall_treatment_complete": sum(bool(case["treatment_complete"]) for case in cases),
        "baseline_unanswerable_false_positive": sum(bool(case["baseline_answered"]) for case in unanswerable),
        "treatment_unanswerable_false_positive": sum(bool(case["treatment_answered"]) for case in unanswerable),
        "all_treatment_trust_states_trusted": all(case["treatment_trust_state"] == "trusted" for case in cases),
        "baseline_ms_mean": sum(cast(float, case["baseline_ms"]) for case in cases) / len(cases),
        "treatment_ms_mean": sum(cast(float, case["treatment_ms"]) for case in cases) / len(cases),
        "details": cases,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
