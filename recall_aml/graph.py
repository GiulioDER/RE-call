"""Grounded graph sidecar construction and fail-safe raw evidence promotion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import re

from recall.types import Chunk, ScoredChunk
from recall_aml.models import AddRequest


GRAPH_PROFILE = "aml-grounded-reference-v1"
GRAPH_SEED_K = 20
GRAPH_PROTECTED_PREFIX = 8
GRAPH_MAX_PROMOTIONS = 2
GRAPH_RELATION_KIND = "references"
GRAPH_STRUCTURAL_TYPE = "aml_evidence_span"
_STRUCTURAL_KEY = re.compile(r"^(\d+):(\d+):(\d+)$")


@dataclass(frozen=True)
class GraphPromotion:
    """One deterministic graph expansion result over an existing raw ranking."""

    hits: list[ScoredChunk]
    relation_hits: int = 0
    candidate_count: int = 0
    promoted_count: int = 0
    invalid_relation_count: int = 0


def attach_grounded_relations(request: AddRequest, chunks: Sequence[Chunk]) -> list[Chunk]:
    """Attach authored compiled-to-raw references only for exact accepted evidence spans.

    The model never supplies relation endpoints. It supplies evidence spans through the existing
    compiler boundary, and this function resolves both endpoints from server-created chunk IDs.
    Unsupported, out-of-range, or non-verbatim spans create no edge.
    """
    raw = [chunk for chunk in chunks if chunk.metadata.get("record_type") == "raw"]
    raw_by_ordinal: dict[int, list[Chunk]] = {}
    for chunk in raw:
        message_ordinals = chunk.metadata.get("message_ordinals")
        if isinstance(message_ordinals, list):
            for message_ordinal in message_ordinals:
                if isinstance(message_ordinal, int) and not isinstance(message_ordinal, bool):
                    raw_by_ordinal.setdefault(message_ordinal, []).append(chunk)
            continue
        ordinal = chunk.metadata.get("ordinal")
        if isinstance(ordinal, int) and not isinstance(ordinal, bool):
            raw_by_ordinal.setdefault(ordinal, []).append(chunk)

    linked: list[Chunk] = []
    for chunk in chunks:
        if chunk.metadata.get("record_type") != "compiled":
            linked.append(chunk)
            continue
        payload = chunk.metadata.get("coding_record")
        spans = payload.get("evidence_spans", []) if isinstance(payload, Mapping) else []
        relations: list[dict[str, object]] = []
        for span in spans if isinstance(spans, list) else []:
            if not isinstance(span, Mapping):
                continue
            ordinal = span.get("message_ordinal")
            start = span.get("start")
            end = span.get("end")
            quote = span.get("quote")
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or not isinstance(quote, str)
                or ordinal < 0
                or ordinal >= len(request.messages)
                or start < 0
                or end <= start
            ):
                continue
            content = request.messages[ordinal].content
            if not isinstance(content, str) or end > len(content) or content[start:end] != quote:
                continue
            for target in raw_by_ordinal.get(ordinal, []):
                char_start = target.metadata.get("char_start")
                char_end = target.metadata.get("char_end")
                direct_span = not (
                    isinstance(char_start, bool)
                    or not isinstance(char_start, int)
                    or isinstance(char_end, bool)
                    or not isinstance(char_end, int)
                    or start >= char_end
                    or end <= char_start
                )
                window_ordinals = target.metadata.get("message_ordinals")
                window_span = isinstance(window_ordinals, list) and ordinal in window_ordinals
                if not direct_span and not window_span:
                    continue
                relations.append(
                    {
                        "relation": GRAPH_RELATION_KIND,
                        "subject": str(chunk.metadata.get("file", f"{chunk.id}.md")),
                        "object": str(target.metadata.get("file", f"{target.id}.md")),
                        "structural_type": GRAPH_STRUCTURAL_TYPE,
                        "structural_key": f"{ordinal}:{start}:{end}",
                    }
                )
        for predecessor in chunk.metadata.get("supersedes", []):
            if isinstance(predecessor, str) and predecessor:
                relations.append(
                    {
                        "relation": "supersedes",
                        "subject": str(chunk.metadata.get("file", f"{chunk.id}.md")),
                        "object": f"{predecessor}.md",
                        "structural_type": "aml_compiler_supersession",
                        "structural_key": predecessor,
                    }
                )
        metadata = dict(chunk.metadata)
        metadata["recall_graph"] = {
            "schema_version": 1,
            "relations": relations,
        }
        linked.append(replace(chunk, metadata=metadata))
    return linked


def promote_grounded_raw(
    baseline_hits: Sequence[ScoredChunk],
    sidecar_hits: Sequence[ScoredChunk],
    *,
    superseded_sidecar_ids: frozenset[str] = frozenset(),
    historical: bool = False,
) -> GraphPromotion:
    """Promote raw tail items through validated authored references without changing membership."""
    baseline = list(baseline_hits)
    if not baseline or not sidecar_hits:
        return GraphPromotion(baseline)
    raw_by_id = {hit.chunk.id: hit for hit in baseline}
    baseline_rank = {hit.chunk.id: rank for rank, hit in enumerate(baseline)}
    candidate_scores: dict[str, float] = {}
    relation_hits = 0
    invalid = 0
    seen_relations: set[tuple[str, str, str]] = set()

    for seed_rank, seed in enumerate(sidecar_hits[:GRAPH_SEED_K], start=1):
        chunk = seed.chunk
        if chunk.metadata.get("record_type") != "compiled":
            invalid += 1
            continue
        if not historical and chunk.id in superseded_sidecar_ids:
            continue
        graph = chunk.metadata.get("recall_graph")
        relations = graph.get("relations", []) if isinstance(graph, Mapping) else []
        if not isinstance(relations, list):
            invalid += 1
            continue
        subject_file = str(chunk.metadata.get("file", ""))
        session = str(chunk.metadata.get("source_session_id", ""))
        for relation in relations:
            if not isinstance(relation, Mapping):
                invalid += 1
                continue
            if relation.get("relation") != GRAPH_RELATION_KIND:
                continue
            subject = relation.get("subject")
            target_file = relation.get("object")
            structural_type = relation.get("structural_type")
            structural_key = relation.get("structural_key")
            if (
                subject != subject_file
                or not isinstance(target_file, str)
                or not target_file.endswith(".md")
                or structural_type != GRAPH_STRUCTURAL_TYPE
                or not isinstance(structural_key, str)
            ):
                invalid += 1
                continue
            match = _STRUCTURAL_KEY.fullmatch(structural_key)
            target_id = target_file[:-3]
            target = raw_by_id.get(target_id)
            if match is None or target is None:
                invalid += 1
                continue
            ordinal, start, end = (int(value) for value in match.groups())
            metadata = target.chunk.metadata
            char_start = metadata.get("char_start")
            char_end = metadata.get("char_end")
            window_ordinals = metadata.get("message_ordinals")
            direct_span = (
                metadata.get("ordinal") == ordinal
                and not isinstance(char_start, bool)
                and isinstance(char_start, int)
                and not isinstance(char_end, bool)
                and isinstance(char_end, int)
                and start < char_end
                and end > char_start
            )
            window_span = isinstance(window_ordinals, list) and ordinal in window_ordinals
            if (
                metadata.get("record_type") != "raw"
                or str(metadata.get("file", "")) != target_file
                or str(metadata.get("source_session_id", "")) != session
                or not (direct_span or window_span)
            ):
                invalid += 1
                continue
            identity = (chunk.id, target_id, structural_key)
            if identity in seen_relations:
                continue
            seen_relations.add(identity)
            relation_hits += 1
            candidate_scores[target_id] = candidate_scores.get(target_id, 0.0) + 1.0 / (
                60 + seed_rank
            )

    protected_ids = {hit.chunk.id for hit in baseline[:GRAPH_PROTECTED_PREFIX]}
    candidates = [
        chunk_id for chunk_id in candidate_scores if chunk_id not in protected_ids
    ]
    candidates.sort(
        key=lambda chunk_id: (
            -candidate_scores[chunk_id],
            baseline_rank[chunk_id],
            chunk_id,
        )
    )
    promoted_ids = candidates[:GRAPH_MAX_PROMOTIONS]
    if not promoted_ids:
        return GraphPromotion(
            baseline,
            relation_hits=relation_hits,
            candidate_count=len(candidates),
            invalid_relation_count=invalid,
        )
    promoted_set = set(promoted_ids)
    reordered = [
        *baseline[:GRAPH_PROTECTED_PREFIX],
        *(raw_by_id[chunk_id] for chunk_id in promoted_ids),
        *(
            hit
            for hit in baseline[GRAPH_PROTECTED_PREFIX:]
            if hit.chunk.id not in promoted_set
        ),
    ]
    return GraphPromotion(
        reordered,
        relation_hits=relation_hits,
        candidate_count=len(candidates),
        promoted_count=len(promoted_ids),
        invalid_relation_count=invalid,
    )


__all__ = [
    "GRAPH_MAX_PROMOTIONS",
    "GRAPH_PROFILE",
    "GRAPH_PROTECTED_PREFIX",
    "GRAPH_SEED_K",
    "GraphPromotion",
    "attach_grounded_relations",
    "promote_grounded_raw",
]
