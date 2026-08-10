"""Session 6 reasoning evaluation controls.

This module is an offline contract harness for the planned reasoning layer. It scores frozen
observations from ``reasoning_session6.json`` and enforces the evaluation controls before any
provider backed experiment is interpreted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

ReasoningTask = Literal[
    "direct_qa",
    "multi_hop",
    "temporal_reasoning",
    "supersession_recovery",
    "near_miss_abstention",
    "contradiction_detection",
    "entity_disambiguation",
    "missing_evidence_detection",
    "clarification_decision",
]
CorpusKind = Literal["synthetic", "real"]
ReasoningSystem = Literal[
    "current_retrieval",
    "retrieval_entailment",
    "retrieval_authored_graph",
    "retrieval_proposal_exploration",
    "retrieval_full_planner",
    "nearest_neighbor",
    "shuffled_edges_control",
    "removed_edges_control",
]
ExpectedDecision = Literal["answer", "abstain", "needs_clarification", "needs_review"]
MetricValue: TypeAlias = float | int | None
MetricMap: TypeAlias = dict[str, MetricValue]
EXPECTED_DECISIONS: tuple[ExpectedDecision, ...] = (
    "answer",
    "abstain",
    "needs_clarification",
    "needs_review",
)

FIXTURE_PATH = Path(__file__).with_name("reasoning_session6.json")
SYSTEMS: tuple[ReasoningSystem, ...] = (
    "current_retrieval",
    "retrieval_entailment",
    "retrieval_authored_graph",
    "retrieval_proposal_exploration",
    "retrieval_full_planner",
    "nearest_neighbor",
    "shuffled_edges_control",
    "removed_edges_control",
)
PRIMARY_SYSTEMS = SYSTEMS[:5]
CONTROL_SYSTEMS = SYSTEMS[5:]
TASKS: tuple[ReasoningTask, ...] = (
    "direct_qa",
    "multi_hop",
    "temporal_reasoning",
    "supersession_recovery",
    "near_miss_abstention",
    "contradiction_detection",
    "entity_disambiguation",
    "missing_evidence_detection",
    "clarification_decision",
)
RATE_METRICS = (
    "answer_accuracy",
    "citation_precision",
    "unsupported_claim_rate",
    "correct_abstention_rate",
    "false_abstention_rate",
    "proposal_precision",
    "proposal_recall",
    "contradiction_detection_precision",
)


@dataclass(frozen=True)
class ReasoningRunObservation:
    system: ReasoningSystem
    provider_id: str
    answer_correct: bool
    citations_precise: bool
    unsupported_claim: bool
    abstained: bool
    proposed_edges: tuple[tuple[str, str], ...]
    detected_contradiction: bool
    latency_ms: float
    model_calls: int
    tokens: int
    generation_id: str


@dataclass(frozen=True)
class ReasoningBenchmarkCase:
    id: str
    task: ReasoningTask
    corpus_kind: CorpusKind
    question: str
    expected_decision: ExpectedDecision
    expected_answer_facts: tuple[str, ...]
    supporting_memory_ids: tuple[str, ...]
    expected_proposals: tuple[tuple[str, str], ...]
    contradiction_memory_ids: tuple[str, ...]
    observations: tuple[ReasoningRunObservation, ...]


@dataclass(frozen=True)
class ReasoningSession6Fixture:
    version: str
    index_generation: str
    preregistered_thresholds: dict[str, Any]
    memories: tuple[dict[str, Any], ...]
    cases: tuple[ReasoningBenchmarkCase, ...]


def _as_bool(value: object, *, case_id: str, system: str, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{case_id}/{system}: {field} must be a JSON boolean")
    return value


def _as_float(value: object, *, case_id: str, system: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"{case_id}/{system}: {field} must be a non-negative JSON number")
    return float(value)


def _expect_literal(
    value: object, *, case_id: str, field: str, allowed: tuple[str, ...]
) -> str:
    if value not in allowed:
        raise ValueError(f"{case_id}: {field} must be one of {', '.join(allowed)}")
    return str(value)


def _as_non_negative_int(value: object, *, case_id: str, system: str, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{case_id}/{system}: {field} must be a non-negative JSON integer")
    return value


def _as_non_empty_string(value: object, *, case_id: str, system: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{case_id}/{system}: {field} must be a non-empty JSON string")
    return value


def _edge_tuple(raw: dict[str, str]) -> tuple[str, str]:
    return (raw["from_memory_id"], raw["to_memory_id"])


def _unknown_refs(case: dict[str, Any], memory_ids: set[str]) -> list[str]:
    refs = set(case.get("supporting_memory_ids", ()))
    refs.update(case.get("contradiction_memory_ids", ()))
    for proposal in case.get("expected_proposals", ()):
        refs.update(_edge_tuple(proposal))
    for obs in case["observations"].values():
        for proposal in obs.get("proposed_edges", ()):
            refs.update(_edge_tuple(proposal))
    return sorted(ref for ref in refs if ref not in memory_ids)


def _normalise_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _metadata_strings(memory: dict[str, Any]) -> list[str]:
    strings: list[str] = []
    for value in memory.get("metadata", {}).values():
        if isinstance(value, str):
            strings.append(_normalise_text(value))
        elif isinstance(value, list):
            strings.extend(_normalise_text(v) for v in value if isinstance(v, str))
    return strings


def _ensure_no_answer_leakage(
    case: dict[str, Any], *, memory_by_id: dict[str, dict[str, Any]]
) -> None:
    labels = [
        _normalise_text(fact)
        for fact in case.get("expected_answer_facts", ())
        if len(_normalise_text(fact)) >= 12
    ]
    if not labels:
        return
    for memory_id in case.get("supporting_memory_ids", ()):
        metadata = _metadata_strings(memory_by_id[memory_id])
        for label in labels:
            if any(label in value for value in metadata):
                raise ValueError(f"{case['id']}: answer text leaked into metadata for {memory_id}")


def load_fixture(path: Path = FIXTURE_PATH) -> ReasoningSession6Fixture:
    """Load a Session 6 fixture and validate references, labels, and observations.

    The loader rejects unknown memory references, missing system observations, answer text leaked
    into supporting metadata, and malformed provider, generation, latency, call, or token fields.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    memories = tuple(raw["memories"])
    memory_by_id = {memory["id"]: memory for memory in memories}
    memory_ids = set(memory_by_id)
    cases: list[ReasoningBenchmarkCase] = []

    for item in raw["cases"]:
        unknown = _unknown_refs(item, memory_ids)
        if unknown:
            raise ValueError(f"{item['id']}: unknown memory id(s): {', '.join(unknown)}")
        if set(item["observations"]) != set(SYSTEMS):
            missing = sorted(set(SYSTEMS) - set(item["observations"]))
            extra = sorted(set(item["observations"]) - set(SYSTEMS))
            raise ValueError(f"{item['id']}: observation systems mismatch: {missing=}, {extra=}")
        _ensure_no_answer_leakage(item, memory_by_id=memory_by_id)

        observations = []
        for system in SYSTEMS:
            obs = item["observations"][system]
            observations.append(
                ReasoningRunObservation(
                    system=system,
                    provider_id=_as_non_empty_string(
                        obs["provider_id"], case_id=item["id"], system=system, field="provider_id"
                    ),
                    answer_correct=_as_bool(
                        obs["answer_correct"], case_id=item["id"], system=system,
                        field="answer_correct"
                    ),
                    citations_precise=_as_bool(
                        obs["citations_precise"], case_id=item["id"], system=system,
                        field="citations_precise"
                    ),
                    unsupported_claim=_as_bool(
                        obs["unsupported_claim"], case_id=item["id"], system=system,
                        field="unsupported_claim"
                    ),
                    abstained=_as_bool(
                        obs["abstained"], case_id=item["id"], system=system, field="abstained"
                    ),
                    proposed_edges=tuple(_edge_tuple(edge) for edge in obs.get("proposed_edges", ())),
                    detected_contradiction=_as_bool(
                        obs["detected_contradiction"], case_id=item["id"], system=system,
                        field="detected_contradiction"
                    ),
                    latency_ms=_as_float(
                        obs["latency_ms"], case_id=item["id"], system=system, field="latency_ms"
                    ),
                    model_calls=_as_non_negative_int(
                        obs["model_calls"], case_id=item["id"], system=system, field="model_calls"
                    ),
                    tokens=_as_non_negative_int(
                        obs["tokens"], case_id=item["id"], system=system, field="tokens"
                    ),
                    generation_id=_as_non_empty_string(
                        obs["generation_id"], case_id=item["id"], system=system,
                        field="generation_id"
                    ),
                )
            )
        cases.append(
            ReasoningBenchmarkCase(
                id=item["id"],
                task=cast(
                    ReasoningTask,
                    _expect_literal(item["task"], case_id=item["id"], field="task", allowed=TASKS),
                ),
                corpus_kind=cast(
                    CorpusKind,
                    _expect_literal(
                        item["corpus_kind"],
                        case_id=item["id"],
                        field="corpus_kind",
                        allowed=("synthetic", "real"),
                    ),
                ),
                question=item["question"],
                expected_decision=cast(
                    ExpectedDecision,
                    _expect_literal(
                        item["expected_decision"],
                        case_id=item["id"],
                        field="expected_decision",
                        allowed=EXPECTED_DECISIONS,
                    ),
                ),
                expected_answer_facts=tuple(item.get("expected_answer_facts", ())),
                supporting_memory_ids=tuple(item.get("supporting_memory_ids", ())),
                expected_proposals=tuple(
                    _edge_tuple(proposal) for proposal in item.get("expected_proposals", ())
                ),
                contradiction_memory_ids=tuple(item.get("contradiction_memory_ids", ())),
                observations=tuple(observations),
            )
        )

    return ReasoningSession6Fixture(
        version=raw["version"],
        index_generation=raw["index_generation"],
        preregistered_thresholds=raw["preregistered_thresholds"],
        memories=memories,
        cases=tuple(cases),
    )


def _rate(flags: list[bool]) -> float | None:
    return sum(1 for flag in flags if flag) / len(flags) if flags else None


def _proposal_scores(
    cases: tuple[ReasoningBenchmarkCase, ...], system: ReasoningSystem
) -> tuple[float | None, float | None]:
    expected: set[tuple[str, tuple[str, str]]] = set()
    proposed: list[tuple[str, tuple[str, str]]] = []
    for case in cases:
        expected.update((case.id, edge) for edge in case.expected_proposals)
        obs = next(item for item in case.observations if item.system == system)
        proposed.extend((case.id, edge) for edge in obs.proposed_edges)
    true_positive = sum(edge in expected for edge in proposed)
    precision = true_positive / len(proposed) if proposed else None
    recall = sum(edge in proposed for edge in expected) / len(expected) if expected else None
    return precision, recall


def _system_metrics(
    cases: tuple[ReasoningBenchmarkCase, ...], system: ReasoningSystem
) -> MetricMap:
    observations = [next(item for item in case.observations if item.system == system) for case in cases]
    answerable = [
        obs
        for case, obs in zip(cases, observations, strict=True)
        if case.expected_decision == "answer"
    ]
    abstain_expected = [
        obs
        for case, obs in zip(cases, observations, strict=True)
        if case.expected_decision == "abstain"
    ]
    contradiction_labels = [
        bool(case.contradiction_memory_ids)
        for case, obs in zip(cases, observations, strict=True)
    ]
    contradiction_true_positive = sum(
        obs.detected_contradiction and expected
        for obs, expected in zip(observations, contradiction_labels, strict=True)
    )
    contradiction_positive = sum(obs.detected_contradiction for obs in observations)
    precision, recall = _proposal_scores(cases, system)
    return {
        "answer_accuracy": _rate([obs.answer_correct for obs in observations]),
        "citation_precision": _rate([obs.citations_precise for obs in observations]),
        "unsupported_claim_rate": _rate([obs.unsupported_claim for obs in observations]),
        "correct_abstention_rate": _rate([obs.abstained for obs in abstain_expected]),
        "false_abstention_rate": _rate([obs.abstained for obs in answerable]),
        "proposal_precision": precision,
        "proposal_recall": recall,
        "contradiction_detection_precision": (
            contradiction_true_positive / contradiction_positive
            if contradiction_positive else None
        ),
        "latency_ms_mean": round(sum(obs.latency_ms for obs in observations) / len(observations), 1),
        "model_calls": sum(obs.model_calls for obs in observations),
        "token_use": sum(obs.tokens for obs in observations),
        "cross_generation_reproducibility": None,
    }


def _by_task_metrics(
    cases: tuple[ReasoningBenchmarkCase, ...], system: ReasoningSystem
) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for task in TASKS:
        task_cases = tuple(case for case in cases if case.task == task)
        if not task_cases:
            continue
        observations = [
            next(item for item in case.observations if item.system == system) for case in task_cases
        ]
        out[task] = {
            "answer_accuracy": _rate([obs.answer_correct for obs in observations]),
            "unsupported_claim_rate": _rate([obs.unsupported_claim for obs in observations]),
        }
    return out


def _ablation_matrix() -> dict[str, dict[str, object]]:
    return {
        "current_retrieval": {
            "uses_entailment": False,
            "uses_authored_graph": False,
            "uses_proposals": False,
            "uses_planner": False,
            "control": False,
        },
        "retrieval_entailment": {
            "uses_entailment": True,
            "uses_authored_graph": False,
            "uses_proposals": False,
            "uses_planner": False,
            "control": False,
        },
        "retrieval_authored_graph": {
            "uses_entailment": False,
            "uses_authored_graph": True,
            "uses_proposals": False,
            "uses_planner": False,
            "control": False,
        },
        "retrieval_proposal_exploration": {
            "uses_entailment": True,
            "uses_authored_graph": True,
            "uses_proposals": True,
            "uses_planner": False,
            "control": False,
        },
        "retrieval_full_planner": {
            "uses_entailment": True,
            "uses_authored_graph": True,
            "uses_proposals": True,
            "uses_planner": True,
            "control": False,
        },
        "nearest_neighbor": {
            "uses_entailment": False,
            "uses_authored_graph": False,
            "uses_proposals": False,
            "uses_planner": False,
            "control": True,
        },
        "shuffled_edges_control": {
            "uses_entailment": True,
            "uses_authored_graph": "shuffled",
            "uses_proposals": True,
            "uses_planner": True,
            "control": True,
        },
        "removed_edges_control": {
            "uses_entailment": True,
            "uses_authored_graph": "removed",
            "uses_proposals": False,
            "uses_planner": True,
            "control": True,
        },
    }


def _error_tags(case: ReasoningBenchmarkCase, obs: ReasoningRunObservation) -> list[str]:
    tags: list[str] = []
    if not obs.answer_correct:
        tags.append(f"task_failure:{case.task}")
    if obs.unsupported_claim:
        tags.append("unsupported_claim")
    if not obs.citations_precise:
        tags.append("citation_imprecision")
    if case.expected_decision == "answer" and obs.abstained:
        tags.append("false_abstention")
    if case.expected_decision == "abstain" and not obs.abstained:
        tags.append("false_confidence")
    if case.expected_proposals and not set(case.expected_proposals) <= set(obs.proposed_edges):
        tags.append("missed_proposal")
    if case.contradiction_memory_ids and not obs.detected_contradiction:
        tags.append("missed_contradiction")
    return tags


def _per_query_error_taxonomy(
    cases: tuple[ReasoningBenchmarkCase, ...],
) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for case in cases:
        case_tags: dict[str, list[str]] = {}
        for obs in case.observations:
            tags = _error_tags(case, obs)
            if tags:
                case_tags[obs.system] = tags
        out[case.id] = case_tags
    return out


def _provider_generation_identities(
    cases: tuple[ReasoningBenchmarkCase, ...],
) -> dict[str, list[dict[str, str]]]:
    identities: dict[str, set[tuple[str, str]]] = {system: set() for system in SYSTEMS}
    for case in cases:
        for obs in case.observations:
            identities[obs.system].add((obs.provider_id, obs.generation_id))
    return {
        system: [
            {"provider_id": provider, "generation_id": generation}
            for provider, generation in sorted(values)
        ]
        for system, values in identities.items()
    }


def _threshold_status(
    metrics: dict[str, object], thresholds: dict[str, Any]
) -> dict[str, object]:
    full = cast(MetricMap, metrics["retrieval_full_planner"])
    retrieval = cast(MetricMap, metrics["current_retrieval"])
    strict = cast(MetricMap, metrics["retrieval_entailment"])
    proposal = cast(MetricMap, metrics["retrieval_proposal_exploration"])
    authored_graph = cast(MetricMap, metrics["retrieval_authored_graph"])
    nearest = cast(MetricMap, metrics["nearest_neighbor"])
    shuffled = cast(MetricMap, metrics["shuffled_edges_control"])
    removed = cast(MetricMap, metrics["removed_edges_control"])
    heldout = cast(MetricMap, metrics["heldout_split"])
    by_task = cast(dict[str, dict[str, dict[str, float | None]]], metrics["by_task"])
    unsupported_ceiling = thresholds["unsupported_claim_rate_max"]

    def required_number(values: MetricMap, metric: str) -> float:
        value = values[metric]
        if value is None:
            raise ValueError(f"{metric} is required for Session 6 threshold checks")
        return float(value)

    full_answer = required_number(full, "answer_accuracy")
    def required_task_number(system: str, task: str, metric: str) -> float:
        value = by_task[system][task][metric]
        if value is None:
            raise ValueError(f"{system}/{task} {metric} is required for Session 6 threshold checks")
        return float(value)

    full_multi_hop = required_task_number(
        "retrieval_full_planner", "multi_hop", "answer_accuracy"
    )
    retrieval_multi_hop = required_task_number(
        "current_retrieval", "multi_hop", "answer_accuracy"
    )
    full_supersession = required_task_number(
        "retrieval_full_planner", "supersession_recovery", "answer_accuracy"
    )
    full_direct = required_task_number("retrieval_full_planner", "direct_qa", "answer_accuracy")
    retrieval_direct = required_task_number("current_retrieval", "direct_qa", "answer_accuracy")

    checks = {
        "answer_accuracy_threshold": (
            required_number(full, "answer_accuracy") >= thresholds["answer_accuracy_min"]
        ),
        "citation_precision_threshold": (
            required_number(full, "citation_precision") >= thresholds["citation_precision_min"]
        ),
        "correct_abstention_threshold": (
            required_number(full, "correct_abstention_rate")
            >= thresholds["correct_abstention_rate_min"]
        ),
        "false_abstention_threshold": (
            required_number(full, "false_abstention_rate")
            <= thresholds["false_abstention_rate_max"]
        ),
        "proposal_precision_threshold": (
            required_number(full, "proposal_precision") >= thresholds["proposal_precision_min"]
        ),
        "proposal_recall_threshold": (
            required_number(full, "proposal_recall") >= thresholds["proposal_recall_min"]
        ),
        "contradiction_detection_threshold": (
            required_number(full, "contradiction_detection_precision")
            >= thresholds["contradiction_detection_precision_min"]
        ),
        "multi_hop_improves": full_multi_hop > retrieval_multi_hop,
        "direct_retrieval_not_materially_improved": (
            full_direct
            - retrieval_direct
            <= thresholds["direct_qa_material_gain_max"]
        ),
        "proposal_recall_beats_authored_graph": (
            required_number(proposal, "proposal_recall")
            > required_number(authored_graph, "proposal_recall")
        ),
        "strict_mode_reduces_unsupported_claims": (
            required_number(strict, "unsupported_claim_rate")
            < required_number(retrieval, "unsupported_claim_rate")
        ),
        "unsupported_claims_below_ceiling": (
            required_number(full, "unsupported_claim_rate") <= unsupported_ceiling
        ),
        "nearest_neighbor_control_not_enough": (
            full_multi_hop
            > required_task_number("nearest_neighbor", "multi_hop", "answer_accuracy")
        ),
        "shuffled_edge_control_drops": (
            full_supersession
            > required_task_number("shuffled_edges_control", "supersession_recovery", "answer_accuracy")
        ),
        "removed_edge_control_drops": (
            full_multi_hop
            > required_task_number("removed_edges_control", "multi_hop", "answer_accuracy")
            and full_supersession
            > required_task_number("removed_edges_control", "supersession_recovery", "answer_accuracy")
        ),
        "heldout_survives": (
            required_number(heldout, "answer_accuracy") >= thresholds["heldout_answer_accuracy_min"]
        ),
    }
    return {
        "passes": checks,
        "all_passed": all(checks.values()),
        "thresholds": thresholds,
    }


def reasoning_session6_metrics(
    fixture: ReasoningSession6Fixture | None = None,
) -> dict[str, object]:
    """Return deterministic Session 6 metrics for a frozen offline control fixture.

    The returned artifact separates synthetic, real corpus, and heldout controls. It records
    ablations, provider and generation identities, per query error taxonomy, and threshold status
    without invoking a model provider or database.
    """
    fx = fixture or load_fixture()
    cases = fx.cases
    split_cases = {
        "synthetic": tuple(case for case in cases if case.corpus_kind == "synthetic"),
        "real": tuple(case for case in cases if case.corpus_kind == "real"),
        "heldout": tuple(case for case in cases if case.id.endswith("_heldout")),
    }
    metrics: dict[str, object] = {system: _system_metrics(cases, system) for system in SYSTEMS}
    metrics["by_task"] = {system: _by_task_metrics(cases, system) for system in SYSTEMS}
    metrics["synthetic_split"] = _system_metrics(split_cases["synthetic"], "retrieval_full_planner")
    metrics["real_corpus_split"] = _system_metrics(split_cases["real"], "retrieval_full_planner")
    metrics["heldout_split"] = _system_metrics(split_cases["heldout"], "retrieval_full_planner")

    return {
        "_provenance": {
            "generator": "recall.eval.reasoning_session6.reasoning_session6_metrics",
            "fixture": "recall/eval/reasoning_session6.json",
            "generation": fx.index_generation,
            "status": "frozen-control",
            "backs": ["docs/REASONING_CONTRACT.md", "results/ARTIFACTS.md"],
            "note": (
                "Offline Session 6 control artifact. Observations are frozen synthetic and real "
                "corpus controls for evaluator wiring, not provider benchmark claims."
            ),
        },
        "version": fx.version,
        "index_generation": fx.index_generation,
        "n_cases": len(cases),
        "n_memories": len(fx.memories),
        "task_counts": {
            task: sum(1 for case in cases if case.task == task)
            for task in TASKS
        },
        "corpus_counts": {
            kind: sum(1 for case in cases if case.corpus_kind == kind)
            for kind in ("synthetic", "real")
        },
        "metrics": metrics,
        "ablation_matrix": _ablation_matrix(),
        "per_query_error_taxonomy": _per_query_error_taxonomy(cases),
        "provider_generation_identities": _provider_generation_identities(cases),
        "threshold_status": _threshold_status(metrics, fx.preregistered_thresholds),
        "audit": {
            "label_quality": "labels are fixture data, not copied from observations",
            "evaluator_independence": "no provider judge is invoked by this harness",
            "metadata_leakage": "loader rejects exact expected facts in supporting metadata",
            "per_query_results": "observations are stored per case before aggregate metrics",
            "controls": list(CONTROL_SYSTEMS),
        },
    }


def main() -> None:
    print(json.dumps(reasoning_session6_metrics(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
