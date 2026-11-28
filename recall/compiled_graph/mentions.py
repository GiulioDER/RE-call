"""LLM mentions in the reasoning query's graph expansion, admitted only in ``active`` mode.

Graph expansion walks authored relations from the entities a trusted seed mentions to the chunks
that mention the neighbor. An entity a compiled record copied verbatim becomes a mention on each
linked chunk whose own text holds it, but only onto an entity the authored graph already has
(matched by normalized name, never an ambiguous one): a new entity would have no relation to walk.
Mentions are merged at load time, so the stored graph and its validation never change. They only
add the chunks a neighbor reaches: seeds, hub counts and the candidate budget's order stay authored
(``recall_mcp.graph_expansion._admit_relations``). Every chunk that expansion reaches only through
such a mention is labelled ``llm_compiled`` on its hit.
"""

from __future__ import annotations

import hashlib
import logging
import weakref
from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from typing import TypeVar

from recall.compiled_graph.artifact import CompiledGraphArtifact, CompiledGraphArtifactError, artifact_dir
from recall.compiled_graph.service import (
    artifact_root,
    compiled_graph_mode,
    load_serving_artifact,
    serving_lineage,
    tenant_folder,
)
from recall.embedding_core import Embedder
from recall.semantic_graph import (
    SemanticGraphProjection,
    SemanticMention,
    normalize_entity_name,
)
from recall.types import ScoredChunk, TrustedHit

_log = logging.getLogger("recall.compiled_graph")

Hit = TypeVar("Hit", ScoredChunk, TrustedHit)

LABEL_KEY = "graph_link"
LABEL_VALUE = "llm_compiled"
_Merged = tuple[SemanticGraphProjection, frozenset[tuple[str, str]]]
#: Merged projections kept alive, so the expansion's adjacency cache (keyed by object) can hit.
#: Keyed by the identity of the projection merged into, as the relation-control cache is: a
#: shuffled control copy keeps the graph id and every relation id, so no content key tells it
#: from the graph it was copied from.
_MERGED_MAX = 4
_MERGED: OrderedDict[tuple[int, str, bool], tuple[weakref.ref[SemanticGraphProjection], _Merged]] = OrderedDict()


def merge_llm_mentions(
    semantic: SemanticGraphProjection,
    artifact: CompiledGraphArtifact,
    *,
    visible_chunks: Collection[str] | None = None,
) -> _Merged:
    """``semantic`` with the artifact's mentions added, and the (chunk, entity) pairs only they hold.

    The artifact is built over the whole tenant; ``visible_chunks``, when given, keeps only the
    mentions on those chunks, so a chunk a source policy denies never enters the graph.
    """
    by_name: dict[str, list[str]] = {}
    for entity in semantic.entities:
        by_name.setdefault(entity.normalized_name, []).append(entity.id)
    existing = {(mention.chunk_id, mention.entity_id) for mention in semantic.mentions}
    added: list[SemanticMention] = []
    pairs: set[tuple[str, str]] = set()
    for record in artifact.records:
        for chunk_id, name in record.mentions:
            if visible_chunks is not None and chunk_id not in visible_chunks:
                continue
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
    store: object,
    semantic: SemanticGraphProjection,
    embedder: Embedder,
    env: Mapping[str, str],
    *,
    scoped: bool,
) -> _Merged:
    """The graph expansion walks: ``semantic`` itself unless the tenant's mode is ``active``.

    ``scoped`` says ``semantic`` was rebuilt from the chunks a source policy lets the caller see;
    mentions are then admitted only on chunks that graph already holds.
    """
    tenant = getattr(store, "tenant", None)
    binding_reader = getattr(store, "generation_binding", None)
    if not isinstance(tenant, str) or not callable(binding_reader):
        return semantic, frozenset()
    if compiled_graph_mode(env, tenant) != "active":
        return semantic, frozenset()
    binding = binding_reader()
    try:
        lineage = serving_lineage(tenant, binding, embedder, env)
    except CompiledGraphArtifactError as exc:
        # An unreadable source policy: walk the authored graph alone, as a failed load does below.
        _log.info("compiled graph mentions not admitted for tenant %r: %s", tenant, exc)
        return semantic, frozenset()
    if lineage is None:
        return semantic, frozenset()
    # An artifact is never rewritten at its lineage, only removed whole (by a forget, or by hand),
    # so a hit needs no load and no checksum pass, only proof that the artifact is still there.
    key = (id(semantic), repr(sorted(lineage.items())), scoped)
    entry = _MERGED.get(key)
    root = artifact_root(env)
    if (
        entry is not None
        and entry[0]() is semantic
        and root is not None
        and (artifact_dir(tenant_folder(root, tenant), lineage) / "manifest.json").is_file()
    ):
        _MERGED.move_to_end(key)
        return entry[1]
    try:
        artifact = load_serving_artifact(tenant, binding, embedder, env)
    except CompiledGraphArtifactError as exc:
        # Missing, built for another lineage or schema, or damaged: walk the authored graph alone.
        _log.info("compiled graph mentions not admitted for tenant %r: %s", tenant, exc)
        return semantic, frozenset()
    visible = {mention.chunk_id for mention in semantic.mentions} if scoped else None
    merged = merge_llm_mentions(semantic, artifact, visible_chunks=visible)
    _MERGED[key] = (weakref.ref(semantic), merged)
    while len(_MERGED) > _MERGED_MAX:
        _MERGED.popitem(last=False)
    return merged


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
