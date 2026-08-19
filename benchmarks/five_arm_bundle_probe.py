"""Preregistered focused comparison for answer aware evidence selection."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from recall.calibration import Calibration
from recall.evidence import AnswerSlot, EvidencePolicy, build_evidence_bundle
from recall.retriever import (
    DocumentExpansionPolicy,
    StructuralExpansionPolicy,
    expand_retrieval_by_source,
    expand_retrieval_by_structure,
)
from recall.trust import evaluate
from typing import cast

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


@dataclass(frozen=True)
class Case:
    name: str
    query: str
    initial: tuple[ScoredChunk, ...]
    scoped: dict[str, tuple[ScoredChunk, ...]]
    required_ids: frozenset[str]
    slots: tuple[AnswerSlot, ...]
    forbidden_ids: frozenset[str] = frozenset()
    unanswerable: bool = False


def _hit(cid: str, file: str, text: str, score: float, ordinal: int) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(cid, f"/fixture/{file}", text, {"file": file, "ord": ordinal}),
        score=score,
        indexed_at=NOW,
    )


def _raw(query: str, hits: tuple[ScoredChunk, ...] | list[ScoredChunk]) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        hits=list(hits),
        gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(index_generation="fixture"),
    )


def _trusted(result: RetrievalResult) -> TrustedResult:
    return evaluate(result, {}, CALIBRATION, NOW)


def _slot_covered(slot: AnswerSlot, texts: list[str]) -> bool:
    return sum(term.casefold() in text.casefold() for text in texts for term in slot.terms) >= slot.min_matches


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def _run(case: Case, arm: str) -> dict[str, object]:
    raw = _raw(case.query, case.initial)

    def search(_query: str, _k: int, source: str | None = None) -> RetrievalResult:
        return _raw(case.query, case.scoped.get(source or "", ()))

    started = time.perf_counter()
    if arm == "structural_expansion":
        expanded = expand_retrieval_by_structure(
            raw,
            search,
            StructuralExpansionPolicy(enabled=True, max_sources=2, chunks_per_source=8, radius=2),
        )
        policy = EvidencePolicy(max_items=4, bundle_mode="document", max_documents=2)
    elif arm in {"answer_slots", "bundle_beam"}:
        expanded = expand_retrieval_by_source(
            raw,
            search,
            policy=DocumentExpansionPolicy(enabled=True, max_sources=2, chunks_per_source=8),
        )
        policy = EvidencePolicy(
            max_items=4,
            bundle_mode="document",
            max_documents=2,
            answer_slots=case.slots,
            selection_mode="beam" if arm == "bundle_beam" else "prefix",
            beam_width=8,
        )
    else:
        expanded = raw
        policy = EvidencePolicy(
            max_items=4,
            bundle_mode="document" if arm == "document_grouping" else "retrieval",
            max_documents=2,
        )
    bundle = build_evidence_bundle(_trusted(expanded), policy)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ids = {item.chunk_id for item in bundle.items}
    texts = [item.text for item in bundle.items]
    return {
        "case": case.name,
        "arm": arm,
        "decision": bundle.decision,
        "selected_ids": sorted(ids),
        "complete_ids": case.required_ids <= ids,
        "complete_slots": all(_slot_covered(slot, texts) for slot in case.slots),
        "forbidden_selected": len(case.forbidden_ids & ids),
        "false_positive": case.unanswerable and bundle.decision == "answer",
        "elapsed_ms": elapsed_ms,
    }


def _cases() -> tuple[Case, ...]:
    d1 = _hit("d1", "decision.md", "The rollout owner was Ada.", 0.92, 1)
    d4 = _hit("d4", "decision.md", "The cache was retired after migration.", 0.86, 4)
    m2 = _hit("m2", "misleading.md", "The rollout changed because approval was granted.", 0.95, 2)
    m4 = _hit("m4", "misleading.md", "The rollout changed because the cache was retired.", 0.84, 4)
    e1 = _hit("e1", "exceptions.md", "The standard rollout used the new cache.", 0.90, 1)
    e8 = _hit("e8", "exceptions.md", "Exception: the legacy client kept the old cache.", 0.83, 8)
    aa = _hit("aa", "alpha.md", "Alpha used the staged migration.", 0.90, 1)
    ao = _hit("ao", "alpha.md", "Alpha outcome: migration succeeded.", 0.84, 8)
    ba = _hit("ba", "beta.md", "Beta used the staged migration.", 0.89, 1)
    bo = _hit("bo", "beta.md", "Beta outcome: migration failed.", 0.84, 8)
    ac = _hit("ac", "alpha_compare.md", "Alpha result: succeeded.", 0.84, 5)
    bc = _hit("bc", "beta_compare.md", "Beta result: failed.", 0.83, 5)
    aci = _hit("aci", "alpha_compare.md", "Alpha was evaluated first.", 0.90, 1)
    bci = _hit("bci", "beta_compare.md", "Beta was evaluated first.", 0.89, 1)
    p1 = _hit("p1", "partial.md", "The policy has a ten minute limit.", 0.91, 1)
    return (
        Case("distant_two_sections", "why did the rollout change?", (d1,), {"decision.md": (d1, d4)}, frozenset({"d1", "d4"}), (AnswerSlot("owner", ("owner", "Ada"), 2), AnswerSlot("cause", ("cache", "retired"), 2))),
        Case("misleading_partial", "why did the rollout change?", (m2,), {"misleading.md": (m2, m4)}, frozenset({"m4"}), (AnswerSlot("cause", ("cache", "retired"), 2),), frozenset({"m2"})),
        Case("exception_at_conclusion", "what exception changed the outcome?", (e1,), {"exceptions.md": (e1, e8)}, frozenset({"e1", "e8"}), (AnswerSlot("standard", ("standard rollout",), 1), AnswerSlot("exception", ("exception", "legacy client"), 2))),
        Case("similar_entities_different_outcomes", "what happened to alpha and beta?", (aa, ba), {"alpha.md": (aa, ao), "beta.md": (ba, bo)}, frozenset({"ao", "bo"}), (AnswerSlot("alpha", ("alpha outcome",), 1), AnswerSlot("beta", ("beta outcome",), 1))),
        Case("comparison_one_section_per_entity", "compare alpha and beta outcomes", (aci, bci), {"alpha_compare.md": (aci, ac), "beta_compare.md": (bci, bc)}, frozenset({"ac", "bc"}), (AnswerSlot("alpha", ("alpha result",), 1), AnswerSlot("beta", ("beta result",), 1))),
        Case("partial_slot_unanswerable", "what is the policy limit and exception?", (p1,), {"partial.md": (p1,)}, frozenset(), (AnswerSlot("limit", ("ten minute",), 1), AnswerSlot("exception", ("exception",), 1)), unanswerable=True),
    )


def main() -> None:
    arms = ("current_retrieval", "document_grouping", "structural_expansion", "answer_slots", "bundle_beam")
    rows = [_run(case, arm) for case in _cases() for arm in arms]
    # `arms` is built in its own typed dict rather than through `report["arms"]`: a
    # `dict[str, object]` makes every lookup an `object`, which cannot be indexed or assigned into.
    # The emitted JSON is unchanged.
    arms_report: dict[str, dict[str, float]] = {}
    for arm in arms:
        subset = [row for row in rows if row["arm"] == arm]
        timings = [cast(float, row["elapsed_ms"]) for row in subset]
        arms_report[arm] = {
            "complete_id_recall": sum(bool(row["complete_ids"]) for row in subset),
            "complete_slot_recall": sum(bool(row["complete_slots"]) for row in subset),
            "forbidden_selected": sum(cast(int, row["forbidden_selected"]) for row in subset),
            "false_positives": sum(bool(row["false_positive"]) for row in subset),
            "mean_selection_ms": sum(timings) / len(timings),
            "p95_selection_ms": _p95(timings),
        }
    report: dict[str, object] = {"cases": len(_cases()), "arms": arms_report}
    report["details"] = rows
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
