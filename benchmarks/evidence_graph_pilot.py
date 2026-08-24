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
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable

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
    query: str = ""


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
    PilotQuery("direct_fact", "direct fact", ("c5",), ("c5",), frozenset(c.id for c in PILOT_CHUNKS), "What does the weekly planning note say?"),
    PilotQuery(
        "supports_relation",
        "indirect relation",
        ("c1",),
        ("c1", "c2"),
        frozenset(c.id for c in PILOT_CHUNKS),
        "What evidence supports the Atlas Billing service?",
    ),
    PilotQuery(
        "dependency_relation",
        "indirect relation",
        ("c3",),
        ("c3", "c4"),
        frozenset(c.id for c in PILOT_CHUNKS),
        "What does Orion depend on for discovery?",
    ),
    PilotQuery("ambiguous_entity", "ambiguity", ("c6",), ("c6",), frozenset(c.id for c in PILOT_CHUNKS), "What did Alex approve?"),
    PilotQuery(
        "untrusted_neighbor",
        "trust guard",
        ("c3",),
        ("c3",),
        frozenset(c.id for c in PILOT_CHUNKS if c.id != "c4"),
        "What does Orion depend on for discovery?",
    ),
    PilotQuery("unanswerable", "unanswerable", (), (), frozenset(c.id for c in PILOT_CHUNKS), "Which database does Mercury use?"),
)


@dataclass(frozen=True)
class DeepSeekAnswer:
    """One candidate answer for a human-blinded baseline or graph review."""

    query_id: str
    arm: str
    model: str
    chunk_ids: tuple[str, ...]
    answer: str
    abstained: bool
    citation_chunk_ids: tuple[str, ...]
    model_calls: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "arm": self.arm,
            "model": self.model,
            "chunk_ids": list(self.chunk_ids),
            "answer": self.answer,
            "abstained": self.abstained,
            "citation_chunk_ids": list(self.citation_chunk_ids),
            "model_calls": self.model_calls,
        }


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


def _answer_prompt(
    query: PilotQuery,
    arm: str,
    chunk_ids: tuple[str, ...],
    chunks_by_id: dict[str, Chunk],
) -> str:
    if not chunk_ids:
        evidence = "(no evidence)"
    else:
        evidence = "\n\n".join(
            f"[{chunk_id}] {chunks_by_id[chunk_id].text}" for chunk_id in chunk_ids
        )
    return (
        "Answer the question using only the supplied evidence. Do not invent facts. "
        "If the evidence does not support an answer, abstain. Return JSON only with exactly "
        "these fields: answer, abstained, citation_chunk_ids. citation_chunk_ids must contain "
        "only IDs present in the evidence. Keep the answer concise.\n\n"
        f"Arm: {arm}\n"
        f"Question: {query.query}\n\n"
        f"EVIDENCE:\n{evidence}"
    )


def _parse_answer_json(content: str) -> tuple[str, bool, tuple[str, ...]]:
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else candidate
        candidate = candidate.rsplit("```", 1)[0].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("DeepSeek answer returned non-JSON output") from exc
    if not isinstance(value, dict):
        raise ValueError("DeepSeek answer returned a JSON value that is not an object")
    answer = value.get("answer")
    abstained = value.get("abstained")
    citations = value.get("citation_chunk_ids")
    if not isinstance(answer, str) or not isinstance(abstained, bool):
        raise ValueError("DeepSeek answer is missing answer or abstained")
    if not isinstance(citations, list) or not all(isinstance(item, str) for item in citations):
        raise ValueError("DeepSeek answer has invalid citation_chunk_ids")
    return answer.strip(), abstained, tuple(citations)


def _deepseek_completion(prompt: str, model: str) -> str:
    """Call DeepSeek directly or through OpenRouter, without adding a hard dependency."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key and not openrouter_key:
        raise RuntimeError(
            "set DEEPSEEK_API_KEY or OPENROUTER_API_KEY to generate DeepSeek answers"
        )
    if openrouter_key and not api_key:
        api_key = openrouter_key
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://openrouter.ai/api/v1")
        model = model or "deepseek/deepseek-chat"
    else:
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = model or "deepseek-chat"

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=350,
        messages=[
            {
                "role": "system",
                "content": "You are an evidence-grounded answer generator. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek answer returned an empty response")
    return content


def run_deepseek_answers(
    artifact: EvidenceGraphEvaluationArtifact,
    *,
    model: str = "",
    completion: Callable[[str, str], str] | None = None,
) -> list[DeepSeekAnswer]:
    """Generate paired answers; scoring remains entirely outside the model."""
    chunks_by_id = {chunk.id: chunk for chunk in PILOT_CHUNKS}
    observations = {
        (observation.query_id, observation.arm): observation
        for observation in artifact.observations
    }
    complete = completion or _deepseek_completion
    resolved_model = model or (
        "deepseek/deepseek-chat"
        if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("DEEPSEEK_API_KEY")
        else "deepseek-chat"
    )
    results: list[DeepSeekAnswer] = []
    for query in PILOT_QUERIES:
        for arm in ("hybrid_retrieval", "deterministic_graph"):
            observation = observations[(query.query_id, arm)]
            prompt = _answer_prompt(
                query,
                arm,
                observation.citation_chunk_ids,
                chunks_by_id,
            )
            answer, abstained, citations = _parse_answer_json(complete(prompt, resolved_model))
            allowed = set(observation.citation_chunk_ids)
            if not set(citations).issubset(allowed):
                raise ValueError(
                    f"DeepSeek cited a chunk not present in {arm} evidence for {query.query_id}"
                )
            results.append(
                DeepSeekAnswer(
                    query_id=query.query_id,
                    arm=arm,
                    model=resolved_model,
                    chunk_ids=observation.citation_chunk_ids,
                    answer=answer,
                    abstained=abstained,
                    citation_chunk_ids=citations,
                )
            )
    return results


def human_review_package(
    artifact: EvidenceGraphEvaluationArtifact,
    answers: list[DeepSeekAnswer] | None = None,
) -> list[dict[str, Any]]:
    """Create a paired review sheet, with optional model answers, for a human adjudicator."""
    by_key = {
        (answer.query_id, answer.arm): answer for answer in (answers or [])
    }
    observations = {
        (observation.query_id, observation.arm): observation
        for observation in artifact.observations
    }
    chunks_by_id = {chunk.id: chunk for chunk in PILOT_CHUNKS}
    package: list[dict[str, Any]] = []
    for query in PILOT_QUERIES:
        arms: dict[str, dict[str, Any]] = {}
        for arm in ("hybrid_retrieval", "deterministic_graph"):
            answer = by_key.get((query.query_id, arm))
            observation = observations[(query.query_id, arm)]
            arms[arm] = {
                "answer": answer.answer if answer else None,
                "abstained": answer.abstained if answer else None,
                "model_citation_chunk_ids": list(answer.citation_chunk_ids) if answer else [],
                "retrieved_chunk_ids": list(observation.citation_chunk_ids),
                "evidence": [
                    {"id": chunk_id, "text": chunks_by_id[chunk_id].text}
                    for chunk_id in observation.citation_chunk_ids
                ],
            }
        package.append(
            {
                "query_id": query.query_id,
                "query": query.query,
                "baseline": arms["hybrid_retrieval"],
                "graph": arms["deterministic_graph"],
                "human_judgment": {
                    "baseline_correct": None,
                    "graph_correct": None,
                    "winner": None,
                    "graph_added_evidence": None,
                    "unsupported_claims_baseline": None,
                    "unsupported_claims_graph": None,
                    "notes": "",
                },
            }
        )
    return package


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the sanitized artifact JSON")
    parser.add_argument(
        "--answers",
        choices=("deepseek",),
        help="optionally generate paired baseline and graph answers with DeepSeek",
    )
    parser.add_argument("--answer-model", default="", help="override the DeepSeek/OpenRouter model id")
    parser.add_argument("--answer-output", type=Path, help="write paired model answers separately")
    parser.add_argument(
        "--review-output",
        type=Path,
        help="write a human review package; works offline without --answers",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    artifact = run_pilot()
    answers: list[DeepSeekAnswer] | None = None
    if args.output:
        artifact.write(args.output)
        print(f"artifact: {args.output}")
    print(json.dumps(_summary(artifact), indent=2, sort_keys=True))
    if args.answers == "deepseek":
        answers = run_deepseek_answers(artifact, model=args.answer_model)
        serialized = [answer.to_dict() for answer in answers]
        if args.answer_output:
            args.answer_output.parent.mkdir(parents=True, exist_ok=True)
            args.answer_output.write_text(
                json.dumps(serialized, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"answer artifact: {args.answer_output}")
        print(json.dumps(serialized, ensure_ascii=False, sort_keys=True, indent=2))
    if args.review_output:
        args.review_output.parent.mkdir(parents=True, exist_ok=True)
        args.review_output.write_text(
            json.dumps(
                human_review_package(artifact, answers),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"human review package: {args.review_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
