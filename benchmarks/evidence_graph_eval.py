"""Preregistered, offline artifact helpers for Evidence Graph V1 evaluation.

This module does not call a model or mutate production graph rows.  It records one sanitized
observation per query and provides deterministic relation controls derived from an immutable graph
snapshot.  The actual retrieval and adjudication runners can supply their measured values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Literal

from recall.semantic_graph import SemanticGraphProjection, SemanticRelation

EvaluationArm = Literal[
    "hybrid_retrieval",
    "authored_graph",
    "deterministic_graph",
    "shuffled_relation_control",
    "removed_relation_control",
    "proposal_only",
]

PREREGISTRATION_VERSION = "evidence-graph-v1"
EVALUATION_ARMS: tuple[EvaluationArm, ...] = (
    "hybrid_retrieval",
    "authored_graph",
    "deterministic_graph",
    "shuffled_relation_control",
    "removed_relation_control",
    "proposal_only",
)


@dataclass(frozen=True)
class GraphEvaluationObservation:
    """One per-query, per-arm record suitable for a preregistered JSON artifact."""

    query_id: str
    arm: EvaluationArm
    tenant_id: str
    generation_id: str
    pipeline_fingerprint: str | None
    corpus_fingerprint: str | None
    calibration_id: str | None
    graph_expansion_mode: str
    graph_readiness: str
    graph_fingerprint: str | None
    relation_control_seed: int | None
    initial_trusted_chunk_ids: tuple[str, ...] = ()
    appended_trusted_chunk_ids: tuple[str, ...] = ()
    rejected_candidate_count: int = 0
    graph_diagnostic_count: int = 0
    entities_inspected: int = 0
    relations_inspected: int = 0
    answer: str | None = None
    abstained: bool = False
    citation_chunk_ids: tuple[str, ...] = ()
    gold_evidence_chunk_ids: tuple[str, ...] = ()
    adjudication: str | None = None
    unsupported_claim_count: int = 0
    contradiction_decision: str | None = None
    latency_ms: float | None = None
    graph_build_latency_ms: float | None = None
    model_calls: int = 0
    token_usage: int = 0

    def __post_init__(self) -> None:
        if self.arm not in EVALUATION_ARMS:
            raise ValueError(f"unsupported evaluation arm: {self.arm}")
        if self.rejected_candidate_count < 0 or self.graph_diagnostic_count < 0:
            raise ValueError("evaluation counts cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "initial_trusted_chunk_ids",
            "appended_trusted_chunk_ids",
            "citation_chunk_ids",
            "gold_evidence_chunk_ids",
        ):
            value[key] = list(value[key])
        return value


def _relation_with_endpoints(
    relation: SemanticRelation, subject_id: str, object_id: str
) -> SemanticRelation:
    """Clone a relation for a control while preserving its evidence and provenance fields."""
    return SemanticRelation(
        id=relation.id,
        tenant_id=relation.tenant_id,
        generation_id=relation.generation_id,
        subject_id=subject_id,
        object_id=object_id,
        relation=relation.relation,
        evidence_chunk_ids=relation.evidence_chunk_ids,
        extraction_method=relation.extraction_method,
        confidence=relation.confidence,
        status=relation.status,
        uncertainty=relation.uncertainty,
        pipeline_fingerprint=relation.pipeline_fingerprint,
        corpus_fingerprint=relation.corpus_fingerprint,
        metadata=relation.metadata,
    )


def relation_control(
    graph: SemanticGraphProjection,
    arm: Literal["shuffled_relation_control", "removed_relation_control"],
    *,
    seed: int,
) -> SemanticGraphProjection:
    """Return a detached relation-control projection without changing the source graph."""
    if arm == "removed_relation_control":
        relations: tuple[SemanticRelation, ...] = ()
    elif arm == "shuffled_relation_control":
        rng = random.Random(seed)
        endpoints = [(relation.subject_id, relation.object_id) for relation in graph.relations]
        rng.shuffle(endpoints)
        relations = tuple(
            _relation_with_endpoints(relation, subject_id, object_id)
            for relation, (subject_id, object_id) in zip(graph.relations, endpoints)
        )
    else:
        raise ValueError(f"unsupported relation control: {arm}")
    return SemanticGraphProjection(
        schema_version=graph.schema_version,
        graph_id=f"{graph.graph_id}:{arm}:{seed}",
        tenant_id=graph.tenant_id,
        generation_id=graph.generation_id,
        pipeline_fingerprint=graph.pipeline_fingerprint,
        corpus_fingerprint=graph.corpus_fingerprint,
        entities=graph.entities,
        mentions=graph.mentions,
        relations=relations,
        diagnostics=graph.diagnostics,
    )


@dataclass
class EvidenceGraphEvaluationArtifact:
    """Serializable aggregate preserving every observation and declared control."""

    preregistration_version: str = PREREGISTRATION_VERSION
    observations: list[GraphEvaluationObservation] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)

    def add(self, observation: GraphEvaluationObservation) -> None:
        self.observations.append(observation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preregistration_version": self.preregistration_version,
            "observations": [item.to_dict() for item in self.observations],
            "deviations": list(self.deviations),
        }

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


__all__ = [
    "EVALUATION_ARMS",
    "EvidenceGraphEvaluationArtifact",
    "GraphEvaluationObservation",
    "PREREGISTRATION_VERSION",
    "relation_control",
]
