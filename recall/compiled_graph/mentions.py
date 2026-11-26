"""LLM mentions in the reasoning query's graph expansion, admitted only in ``active`` mode.

Graph expansion walks authored relations from the entities a trusted seed mentions to the chunks
that mention the neighbor. An entity a compiled record copied verbatim becomes a mention on each
linked chunk whose own text holds it, but only onto an entity the authored graph already has
(matched by normalized name, never an ambiguous one): a new entity would have no relation to walk.
Mentions are merged at load time, so the stored graph and its validation never change. Every chunk
that expansion reaches only through such a mention is labelled ``llm_compiled`` on its hit.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from typing import Any, TypeVar

from recall.compiled_graph.artifact import CompiledGraphArtifact, CompiledGraphArtifactError
from recall.compiled_graph.service import compiled_graph_mode, load_serving_artifact
from recall.embedding_core import Embedder
from recall.semantic_graph import SemanticGraphProjection, SemanticMention, normalize_entity_name
from recall.types import ScoredChunk, TrustedHit

_log = logging.getLogger("recall.compiled_graph")

Hit = TypeVar("Hit", ScoredChunk, TrustedHit)

LABEL_KEY = "graph_link"
LABEL_VALUE = "llm_compiled"
#: Merged projections kept alive, so the expansion's adjacency cache (keyed by object) can hit.
_MERGED_MAX = 4
_MERGED: OrderedDict[tuple[str, str, str], tuple[SemanticGraphProjection, frozenset[tuple[str, str]]]] = OrderedDict()


def merge_llm_mentions(
    semantic: SemanticGraphProjection, artifact: CompiledGraphArtifact
) -> tuple[SemanticGraphProjection, frozenset[tuple[str, str]]]:
    """``semantic`` with the artifact's mentions added, and the (chunk, entity) pairs only they hold."""
    by_name: dict[str, list[str]] = {}
    for entity in semantic.entities:
        by_name.setdefault(entity.normalized_name, []).append(entity.id)
    existing = {(mention.chunk_id, mention.entity_id) for mention in semantic.mentions}
    added: list[SemanticMention] = []
    pairs: set[tuple[str, str]] = set()
    for record in artifact.records:
        for chunk_id, name in record.mentions:
            matches = by_name.get(normalize_entity_name(name), [])
            if len(matches) != 1:
                continue
            pair = (chunk_id, matches[0])
            if pair in existing or pair in pairs:
                continue
            pairs.add(pair)
            digest = hashlib.sha256(f"{chunk_id}\0{matches[0]}".encode("utf-8")).hexdigest()[:24]
            added.append(
                SemanticMention(
                    id=f"llm-{digest}",
                    tenant_id=semantic.tenant_id,
                    generation_id=semantic.generation_id,
                    entity_id=matches[0],
                    chunk_id=chunk_id,
                    mention_text=name,
                    extraction_method="llm_compiled",
                    metadata={"record": record.key},
                )
            )
    if not added:
        return semantic, frozenset()
    return replace(semantic, mentions=(*semantic.mentions, *added)), frozenset(pairs)


def admit_llm_mentions(
    store: object, semantic: SemanticGraphProjection, embedder: Embedder, env: Mapping[str, str]
) -> tuple[SemanticGraphProjection, frozenset[tuple[str, str]]]:
    """The graph expansion walks: ``semantic`` itself unless the tenant's mode is ``active``."""
    tenant = getattr(store, "tenant", None)
    binding_reader = getattr(store, "generation_binding", None)
    if not isinstance(tenant, str) or not callable(binding_reader):
        return semantic, frozenset()
    if compiled_graph_mode(env, tenant) != "active":
        return semantic, frozenset()
    try:
        artifact = load_serving_artifact(tenant, binding_reader(), embedder, env)
    except CompiledGraphArtifactError as exc:
        # No artifact for this corpus yet: expansion walks the authored graph alone.
        _log.info("compiled graph mentions not admitted for tenant %r: %s", tenant, exc)
        return semantic, frozenset()
    key = (semantic.graph_id, semantic.fingerprint, repr(sorted(artifact.lineage.items())))
    cached = _MERGED.get(key)
    if cached is not None:
        _MERGED.move_to_end(key)
        return cached
    merged = merge_llm_mentions(semantic, artifact)
    _MERGED[key] = merged
    while len(_MERGED) > _MERGED_MAX:
        _MERGED.popitem(last=False)
    return merged


def llm_dependent_chunks(
    candidate_relations_by_chunk: Mapping[str, Sequence[Any]],
    seed_entities: Collection[str],
    trusted_seed_ids: Collection[str],
    resolved_query_entities: Collection[str],
    mentions_by_chunk: Mapping[str, Collection[str]],
    llm_pairs: frozenset[tuple[str, str]],
) -> frozenset[str]:
    """Candidates whose link to a seed exists only through an LLM mention, on either end."""
    if not llm_pairs:
        return frozenset()
    llm_only_seeds = {
        entity
        for entity in seed_entities
        if entity not in resolved_query_entities
        and all(
            (chunk_id, entity) in llm_pairs
            for chunk_id in trusted_seed_ids
            if entity in mentions_by_chunk.get(chunk_id, ())
        )
    }
    labelled: set[str] = set()
    for chunk_id, relations in candidate_relations_by_chunk.items():
        for relation in relations:
            seed_side = relation.subject_id if relation.subject_id in seed_entities else relation.object_id
            neighbor = relation.object_id if seed_side == relation.subject_id else relation.subject_id
            if seed_side in llm_only_seeds or (chunk_id, neighbor) in llm_pairs:
                labelled.add(chunk_id)
                break
    return frozenset(labelled)


def label_hits(hits: Sequence[Hit], labelled: frozenset[str]) -> list[Hit]:
    """``hits`` with ``graph_link: llm_compiled`` on each labelled chunk's metadata."""
    if not labelled:
        return list(hits)
    return [
        replace(hit, chunk=replace(hit.chunk, metadata={**hit.chunk.metadata, LABEL_KEY: LABEL_VALUE}))
        if hit.chunk.id in labelled
        else hit
        for hit in hits
    ]
