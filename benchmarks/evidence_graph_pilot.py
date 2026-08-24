"""Run a small, deterministic paired A/B pilot for Evidence Graph V1.

The pilot isolates graph routing from embedding and answer-model variance.  It uses a labeled
fixture with trusted seed chunks, runs every query through the baseline and relation controls, and
records the same sanitized observation shape as the preregistered evaluation artifact.  Results
are directional only until a corpus-backed query set with gold evidence is supplied.

Usage::

    python -m benchmarks.evidence_graph_pilot
    python -m benchmarks.evidence_graph_pilot --output pilot.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable

from benchmarks.evidence_graph_eval import (
    EVALUATION_ARMS,
    EvidenceGraphEvaluationArtifact,
    GraphEvaluationObservation,
    EvaluationArm,
    relation_control,
)
from recall.semantic_graph import SemanticGraphProjection, build_semantic_graph
from recall.types import Chunk

TENANT_ID = "pilot-tenant"
GENERATION_ID = "pilot-generation"
RELATION_CONTROL_SEED = 1


@dataclass(frozen=True)
class PilotQuery:
    query_id: str
    category: str
    seed_chunk_ids: tuple[str, ...]
    gold_evidence_chunk_ids: tuple[str, ...]
    trusted_chunk_ids: frozenset[str]


PILOT_CHUNKS: tuple[Chunk, ...] = (
    Chunk(
        "c1",
        "atlas.md",
        "Atlas owns the Billing service.",
        {
            "project": "Atlas",
            "service": "Billing",
            "relations": [{"relation": "supports", "subject": "Atlas", "object": "Billing"}],
        },
    ),
    Chunk("c2", "billing.md", "Billing handles invoice settlement.", {"service": "Billing"}),
    Chunk(
        "c3",
        "orion.md",
        "Orion depends on Search for discovery.",
        {
            "project": "Orion",
            "service": "Search",
            "relations": [{"relation": "depends_on", "subject": "Orion", "object": "Search"}],
        },
    ),
    Chunk("c4", "search.md", "Search provides indexed discovery.", {"service": "Search"}),
    Chunk("c5", "notes.md", "The team keeps a weekly planning note.", {"concept": "Planning"}),
    Chunk("c6", "alex-person.md", "Alex approved the release.", {"person": "Alex"}),
    Chunk("c7", "alex-project.md", "Alex is also the name of a project.", {"project": "Alex"}),
    Chunk(
        "c8",
        "ambiguous.md",
        "Alex supports Billing.",
        {
            "person": "Alex",
            "service": "Billing",
            "relations": [{"relation": "supports", "subject": "Alex", "object": "Billing"}],
        },
    ),
)


PILOT_QUERIES: tuple[PilotQuery, ...] = (
    PilotQuery("direct_fact", "direct fact", ("c5",), ("c5",), frozenset(c.id for c in PILOT_CHUNKS)),
    PilotQuery(
        "supports_relation",
        "indirect relation",
        ("c1",),
        ("c1", "c2"),
        frozenset(c.id for c in PILOT_CHUNKS),
    ),
    PilotQuery(
        "dependency_relation",
        "indirect relation",
        ("c3",),
        ("c3", "c4"),
        frozenset(c.id for c in PILOT_CHUNKS),
    ),
    PilotQuery("ambiguous_entity", "ambiguity", ("c6",), ("c6",), frozenset(c.id for c in PILOT_CHUNKS)),
    PilotQuery(
        "untrusted_neighbor",
        "trust guard",
        ("c3",),
        ("c3",),
        frozenset(c.id for c in PILOT_CHUNKS if c.id != "c4"),
    ),
    PilotQuery("unanswerable", "unanswerable", (), (), frozenset(c.id for c in PILOT_CHUNKS)),
)


def _graph() -> SemanticGraphProjection:
    return build_semantic_graph(
        PILOT_CHUNKS,
        tenant_id=TENANT_ID,
        generation_id=GENERATION_ID,
        pipeline_fingerprint="pilot-pipeline",
        corpus_fingerprint="pilot-corpus",
    )


def _expanded_ids(
    graph: SemanticGraphProjection,
    query: PilotQuery,
) -> tuple[tuple[str, ...], int, int, int]:
    """Return accepted ids, discovered candidates, inspected relations, and rejected candidates."""
    if not query.seed_chunk_ids:
        return (), 0, 0, 0

    mentions_by_chunk: dict[str, set[str]] = defaultdict(set)
    chunks_by_entity: dict[str, set[str]] = defaultdict(set)
    for mention in graph.mentions:
        mentions_by_chunk[mention.chunk_id].add(mention.entity_id)
        chunks_by_entity[mention.entity_id].add(mention.chunk_id)

    ambiguous_entities = {
        entity_id
        for diagnostic in graph.diagnostics
        if diagnostic.kind == "ambiguous_entity"
        for entity_id in diagnostic.entity_ids
    }
    seed_entities = {
        entity_id
        for chunk_id in query.seed_chunk_ids
        for entity_id in mentions_by_chunk.get(chunk_id, ())
        if entity_id not in ambiguous_entities
    }
    candidates: dict[str, tuple[float, int, str]] = {}
    inspected = 0
    for relation in graph.relations:
        if relation.status != "authored":
            continue
        if relation.subject_id in ambiguous_entities or relation.object_id in ambiguous_entities:
            continue
        if relation.subject_id not in seed_entities and relation.object_id not in seed_entities:
            continue
        inspected += 1
        neighbor = relation.object_id if relation.subject_id in seed_entities else relation.subject_id
        supporting = chunks_by_entity.get(neighbor, set())
        for chunk_id in supporting:
            if chunk_id in query.seed_chunk_ids:
                continue
            score = (relation.confidence, len(relation.evidence_chunk_ids), chunk_id)
            previous = candidates.get(chunk_id)
            if previous is None or score > previous:
                candidates[chunk_id] = score

    ordered = sorted(
        candidates,
        key=lambda chunk_id: (
            -candidates[chunk_id][0],
            -candidates[chunk_id][1],
            chunk_id,
        ),
    )
    rejected = sum(chunk_id not in query.trusted_chunk_ids for chunk_id in ordered)
    accepted = tuple(
        chunk_id for chunk_id in ordered if chunk_id in query.trusted_chunk_ids
    )
    return accepted, len(ordered), inspected, rejected


def _arm_graph(graph: SemanticGraphProjection, arm: EvaluationArm) -> SemanticGraphProjection | None:
    if arm in {"hybrid_retrieval", "authored_graph", "proposal_only"}:
        return None
    if arm == "deterministic_graph":
        return graph
    if arm in {"shuffled_relation_control", "removed_relation_control"}:
        return relation_control(graph, arm, seed=RELATION_CONTROL_SEED)
    raise ValueError(f"unsupported pilot arm: {arm}")


def _observation(
    graph: SemanticGraphProjection,
    query: PilotQuery,
    arm: EvaluationArm,
) -> GraphEvaluationObservation:
    started = time.perf_counter()
    controlled_graph = _arm_graph(graph, arm)
    initial = tuple(query.seed_chunk_ids)
    appended: tuple[str, ...] = ()
    discovered = inspected = rejected = 0
    if controlled_graph is not None:
        appended, discovered, inspected, rejected = _expanded_ids(controlled_graph, query)
    retrieved = tuple(dict.fromkeys((*initial, *appended)))
    gold = set(query.gold_evidence_chunk_ids)
    complete = gold.issubset(retrieved)
    precision = len(gold.intersection(retrieved)) / len(retrieved) if retrieved else 1.0
    return GraphEvaluationObservation(
        query_id=query.query_id,
        arm=arm,
        tenant_id=TENANT_ID,
        generation_id=GENERATION_ID,
        pipeline_fingerprint=graph.pipeline_fingerprint,
        corpus_fingerprint=graph.corpus_fingerprint,
        calibration_id="pilot-calibration",
        graph_expansion_mode="one_hop" if controlled_graph is not None else "off",
        graph_readiness="ready" if controlled_graph is not None else "not_requested",
        graph_fingerprint=controlled_graph.fingerprint if controlled_graph is not None else None,
        relation_control_seed=(
            RELATION_CONTROL_SEED
            if arm in {"shuffled_relation_control", "removed_relation_control"}
            else None
        ),
        initial_trusted_chunk_ids=initial,
        appended_trusted_chunk_ids=appended,
        rejected_candidate_count=rejected,
        graph_diagnostic_count=len(graph.diagnostics),
        entities_inspected=len(controlled_graph.entities) if controlled_graph is not None else 0,
        relations_inspected=inspected,
        citation_chunk_ids=retrieved,
        gold_evidence_chunk_ids=query.gold_evidence_chunk_ids,
        adjudication=("complete" if complete else "partial") + f";precision={precision:.3f}",
        latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )


def run_pilot() -> EvidenceGraphEvaluationArtifact:
    graph = _graph()
    artifact = EvidenceGraphEvaluationArtifact()
    artifact.deviations.append(
        "Pilot uses trusted seed ids and no answer model; it measures evidence routing, not end-to-end answer quality."
    )
    artifact.deviations.append(
        "Authored_graph and proposal_only are baseline-equivalent because this fixture isolates semantic graph expansion."
    )
    for query in PILOT_QUERIES:
        for arm in EVALUATION_ARMS:
            artifact.add(_observation(graph, query, arm))
    return artifact


def _summary(artifact: EvidenceGraphEvaluationArtifact) -> list[dict[str, object]]:
    rows: dict[tuple[str, str], list[GraphEvaluationObservation]] = defaultdict(list)
    for observation in artifact.observations:
        rows[(observation.arm, next(q.category for q in PILOT_QUERIES if q.query_id == observation.query_id))].append(observation)
    summary: list[dict[str, object]] = []
    for (arm, category), observations in sorted(rows.items()):
        recalls = []
        precisions = []
        for observation in observations:
            gold = set(observation.gold_evidence_chunk_ids)
            retrieved = set(observation.citation_chunk_ids)
            if gold:
                recalls.append(len(gold & retrieved) / len(gold))
            precisions.append(len(gold & retrieved) / len(retrieved) if retrieved else 1.0)
        summary.append(
            {
                "arm": arm,
                "category": category,
                "queries": len(observations),
                "evidence_recall": round(sum(recalls) / len(recalls), 3) if recalls else None,
                "citation_precision": round(sum(precisions) / len(precisions), 3),
                "accepted_appended": sum(len(o.appended_trusted_chunk_ids) for o in observations),
                "rejected_candidates": sum(o.rejected_candidate_count for o in observations),
            }
        )
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the sanitized artifact JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)
    artifact = run_pilot()
    if args.output:
        artifact.write(args.output)
        print(f"artifact: {args.output}")
    print(json.dumps(_summary(artifact), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
