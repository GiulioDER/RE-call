"""Session 1 reasoning baseline fixture and metrics.

This module is deliberately offline. It scores the frozen control observations in
``reasoning_session1.json`` so reasoning work can compare against the pre reasoning baseline
without opening a database, downloading a model, or changing retrieval behavior.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ReasoningOutcome = Literal["answer", "abstain", "needs_clarification", "needs_review"]
CaseCategory = Literal[
    "direct_answer",
    "multi_hop",
    "near_miss",
    "contradiction",
    "missing_supersession",
    "ambiguous_entity",
    "empty_corpus",
    "stale_corpus",
]

FIXTURE_PATH = Path(__file__).with_name("reasoning_session1.json")


@dataclass(frozen=True)
class BaselineRetrieval:
    retrieved_memory_ids: tuple[str, ...]
    abstained: bool
    reason: str


@dataclass(frozen=True)
class ReasoningCase:
    id: str
    category: CaseCategory
    question: str
    expected_outcome: ReasoningOutcome
    supporting_memory_ids: tuple[str, ...]
    baseline_retrieval: BaselineRetrieval
    expected_proposals: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class ReasoningFixture:
    version: str
    index_generation: str
    memories: tuple[dict[str, Any], ...]
    cases: tuple[ReasoningCase, ...]


def _require_bool(value: object, *, case_id: str, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{case_id}: {field} must be a JSON boolean")
    return value


def _unknown_memory_refs(
    item: dict[str, Any], baseline: dict[str, Any], memory_ids: set[str]
) -> list[str]:
    refs = set(item.get("supporting_memory_ids", ()))
    refs.update(baseline["retrieved_memory_ids"])
    for proposal in item.get("expected_proposals", ()):
        refs.add(proposal["from_memory_id"])
        refs.add(proposal["to_memory_id"])
    return sorted(ref for ref in refs if ref not in memory_ids)


def load_fixture(path: Path = FIXTURE_PATH) -> ReasoningFixture:
    raw = json.loads(path.read_text(encoding="utf-8"))
    memories = tuple(raw["memories"])
    memory_ids = {memory["id"] for memory in memories}
    cases = []
    for item in raw["cases"]:
        baseline = item["baseline_retrieval"]
        unknown = _unknown_memory_refs(item, baseline, memory_ids)
        if unknown:
            raise ValueError(f"{item['id']}: unknown memory id(s): {', '.join(unknown)}")
        cases.append(
            ReasoningCase(
                id=item["id"],
                category=item["category"],
                question=item["question"],
                expected_outcome=item["expected_outcome"],
                supporting_memory_ids=tuple(item.get("supporting_memory_ids", ())),
                baseline_retrieval=BaselineRetrieval(
                    retrieved_memory_ids=tuple(baseline["retrieved_memory_ids"]),
                    abstained=_require_bool(
                        baseline["abstained"], case_id=item["id"], field="baseline_retrieval.abstained"
                    ),
                    reason=baseline.get("reason", ""),
                ),
                expected_proposals=tuple(item.get("expected_proposals", ())),
            )
        )
    return ReasoningFixture(
        version=raw["version"],
        index_generation=raw["index_generation"],
        memories=memories,
        cases=tuple(cases),
    )


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def baseline_metrics(fixture: ReasoningFixture | None = None) -> dict[str, object]:
    """Return deterministic Session 1 baseline metrics.

    Direct hit rate and multi hop complete support rate are intentionally separate. A direct
    question is satisfied when any supporting memory is retrieved. A multi hop question is
    satisfied only when every supporting memory is retrieved.
    """
    fx = fixture or load_fixture()
    by_category: dict[str, list[ReasoningCase]] = {}
    for case in fx.cases:
        by_category.setdefault(case.category, []).append(case)

    direct = by_category.get("direct_answer", [])
    multi = by_category.get("multi_hop", [])
    near_miss = by_category.get("near_miss", [])
    missing_edges = by_category.get("missing_supersession", [])
    abstain_expected = [c for c in fx.cases if c.expected_outcome == "abstain"]

    direct_hits = sum(
        bool(set(c.supporting_memory_ids) & set(c.baseline_retrieval.retrieved_memory_ids))
        and not c.baseline_retrieval.abstained
        for c in direct
    )
    multi_complete = sum(
        set(c.supporting_memory_ids) <= set(c.baseline_retrieval.retrieved_memory_ids)
        and not c.baseline_retrieval.abstained
        for c in multi
    )
    near_miss_false_confident = sum(not c.baseline_retrieval.abstained for c in near_miss)
    abstention_hits = sum(c.baseline_retrieval.abstained for c in abstain_expected)
    missing_edge_unresolved = sum(
        bool(c.expected_proposals)
        and not c.baseline_retrieval.abstained
        and all(
            {
                proposal["from_memory_id"],
                proposal["to_memory_id"],
            } <= set(c.baseline_retrieval.retrieved_memory_ids)
            for proposal in c.expected_proposals
        )
        for c in missing_edges
    )

    return {
        "_provenance": {
            "generator": "recall.eval.reasoning_session1.baseline_metrics",
            "fixture": "recall/eval/reasoning_session1.json",
            "generation": "reasoning-session1-baseline",
            "status": "current",
            "backs": ["docs/REASONING_CONTRACT.md"],
            "note": (
                "Synthetic Session 1 control observations recorded before the reasoning layer "
                "is enabled. These are fixture-derived baseline metrics, not a model run."
            ),
        },
        "version": fx.version,
        "index_generation": fx.index_generation,
        "n_cases": len(fx.cases),
        "n_memories": len(fx.memories),
        "category_counts": {category: len(cases) for category, cases in sorted(by_category.items())},
        "direct_hit_rate": _rate(direct_hits, len(direct)),
        "multi_hop_complete_support_rate": _rate(multi_complete, len(multi)),
        "near_miss_false_confident_rate": _rate(
            near_miss_false_confident, len(near_miss)
        ),
        "expected_abstention_accuracy": _rate(abstention_hits, len(abstain_expected)),
        "missing_supersession_unresolved_rate": _rate(
            missing_edge_unresolved, len(missing_edges)
        ),
        "observations": [
            "Direct retrieval succeeds more often than complete multi hop support retrieval.",
            "The near miss case is answered confidently by the baseline observation.",
            "The missing supersession edge remains an unresolved proposal, not authored metadata.",
        ],
    }


def main() -> None:
    print(json.dumps(baseline_metrics(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
