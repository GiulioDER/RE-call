"""Building a semantic graph projection from chunk metadata and declared relations.
"""

from __future__ import annotations

import math
import posixpath
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from recall.frontmatter import dependencies_from_metadata, supersedes_key
from recall.semantic_graph._model import (
    ENTITY_KINDS,
    RELATION_KINDS,
    SEMANTIC_GRAPH_SCHEMA_VERSION,
    EntityKind,
    ExtractionMethod,
    SemanticEntity,
    SemanticGraphDiagnostic,
    SemanticGraphProjection,
    SemanticMention,
    SemanticRelation,
    _EntityBuildResult,
    _identity,
    _parse_temporal_value,
    _relation_temporal_fields,
    normalize_entity_name,
)
from recall.types import Chunk


def _as_labels(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _graph_metadata(chunk: Chunk) -> Mapping[str, Any]:
    value = chunk.metadata.get("recall_graph")
    return value if isinstance(value, Mapping) else {}


def _entity_specs(chunk: Chunk) -> list[tuple[str, EntityKind, ExtractionMethod]]:
    specs: list[tuple[str, EntityKind, ExtractionMethod]] = []
    file_name = chunk.metadata.get("file")
    if isinstance(file_name, str) and file_name.strip():
        specs.append((file_name.strip(), "file", "filename"))
    for key in ENTITY_KINDS:
        for label in _as_labels(chunk.metadata.get(key)):
            specs.append((label, key, "metadata"))
    for metadata_key in ("entities",):
        for label in _as_labels(chunk.metadata.get(metadata_key)):
            specs.append((label, "unknown", "metadata"))
    graph_entities = _graph_metadata(chunk).get("entities")
    if isinstance(graph_entities, Sequence) and not isinstance(
        graph_entities, (str, bytes, bytearray)
    ):
        for item in graph_entities:
            if not isinstance(item, Mapping):
                continue
            graph_label = item.get("name")
            kind_value = item.get("kind", "unknown")
            if (
                isinstance(graph_label, str)
                and isinstance(kind_value, str)
                and kind_value in ENTITY_KINDS
            ):
                specs.append((graph_label, kind_value, "metadata"))
    for line in chunk.text.splitlines():
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if heading:
            specs.append((heading.group(1).strip(), "concept", "heading"))
    return specs


def _alias_specs(value: Any) -> list[tuple[str, str]]:
    """Read the canonical-name -> aliases form of explicit alias metadata."""
    if not isinstance(value, Mapping):
        return []
    pairs: list[tuple[str, str]] = []
    for canonical, aliases in value.items():
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        for alias in _as_labels(aliases):
            pairs.append((canonical.strip(), alias))
    return pairs


def _chunk_alias_specs(chunk: Chunk) -> list[tuple[str, str]]:
    pairs = _alias_specs(chunk.metadata.get("entity_aliases"))
    graph = _graph_metadata(chunk)
    pairs.extend(_alias_specs(graph.get("aliases")))
    return pairs


def _relation_specs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _chunk_relation_specs(chunk: Chunk) -> list[dict[str, Any]]:
    relations = _relation_specs(chunk.metadata.get("relations"))
    graph = _graph_metadata(chunk)
    if "__parse_error__" in graph:
        relations.append({"relation": "__invalid_graph_annotation__"})
    relations.extend(_relation_specs(graph.get("relations")))
    return relations


def _relation_metadata(raw: Mapping[str, Any], source: str, target: str | None = None) -> dict[str, Any]:
    """Preserve the bounded provenance labels used by deterministic benchmark relations."""
    metadata: dict[str, Any] = {"source": source}
    for key in ("structural_type", "structural_key"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip() and len(value) <= 512:
            metadata[key] = value.strip()
    raw_metadata = raw.get("metadata")
    if isinstance(raw_metadata, Mapping):
        for key in ("structural_type", "structural_key"):
            value = raw_metadata.get(key)
            if isinstance(value, str) and value.strip() and len(value) <= 512:
                metadata[key] = value.strip()
    if target is not None:
        metadata["target"] = target
    return metadata


def _temporal_metadata(
    effective_at: datetime | None,
    valid_from: datetime | None,
    valid_until: datetime | None,
) -> dict[str, str]:
    return {
        key: value.isoformat()
        for key, value in (
            ("effective_at", effective_at),
            ("valid_from", valid_from),
            ("valid_until", valid_until),
        )
        if value is not None
    }


def _chunk_temporal_metadata(chunk: Chunk) -> dict[str, str]:
    try:
        valid_from = _parse_temporal_value(chunk.metadata.get("valid_from"), "valid_from")
        valid_until = _parse_temporal_value(
            chunk.metadata.get("valid_until"), "valid_until", end_of_day=True
        )
    except ValueError:
        return {}
    return _temporal_metadata(None, valid_from, valid_until)


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:\|[^\]]+)?\]\]")


_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")


def _reference_target(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme or parsed.netloc:
        return ""
    target = value.strip().split("#", 1)[0].split("?", 1)[0]
    target = target.replace("\\", "/")
    return posixpath.basename(target).strip()


def _explicit_reference_targets(text: str) -> tuple[str, ...]:
    targets = [match.group(1) for match in _WIKILINK_RE.finditer(text)]
    targets.extend(match.group(1) for match in _MARKDOWN_LINK_RE.finditer(text))
    return tuple(target for target in (_reference_target(item) for item in targets) if target)


def _finalize_graph(
    *,
    entity_by_key: Mapping[tuple[str, EntityKind], SemanticEntity],
    tenant_id: str,
    generation_id: str,
    pipeline_fingerprint: str | None,
    corpus_fingerprint: str | None,
    mentions: Sequence[SemanticMention],
    relations: Mapping[str, SemanticRelation],
    diagnostics: Sequence[SemanticGraphDiagnostic],
) -> SemanticGraphProjection:
    """Sort graph members and derive the stable graph identity."""
    entities = tuple(sorted(entity_by_key.values(), key=lambda entity: entity.id))
    ordered_mentions = tuple(sorted(mentions, key=lambda mention: mention.id))
    ordered_relations = tuple(sorted(relations.values(), key=lambda relation: relation.id))
    ordered_diagnostics = tuple(sorted(diagnostics, key=lambda diagnostic: diagnostic.id))
    graph_id = _identity(
        "graph",
        {
            "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
            "tenant_id": tenant_id,
            "generation_id": generation_id,
            "pipeline_fingerprint": pipeline_fingerprint,
            "corpus_fingerprint": corpus_fingerprint,
            "entities": [entity.id for entity in entities],
            "mentions": [mention.id for mention in ordered_mentions],
            "relations": [relation.id for relation in ordered_relations],
            "diagnostics": [diagnostic.id for diagnostic in ordered_diagnostics],
        },
    )
    return SemanticGraphProjection(
        schema_version=SEMANTIC_GRAPH_SCHEMA_VERSION,
        graph_id=graph_id,
        tenant_id=tenant_id,
        generation_id=generation_id,
        pipeline_fingerprint=pipeline_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        entities=entities,
        mentions=ordered_mentions,
        relations=ordered_relations,
        diagnostics=ordered_diagnostics,
    )


def _file_target_indexes(
    *,
    file_entities_by_source: Mapping[str, set[str]],
    entity_by_key: Mapping[tuple[str, EntityKind], SemanticEntity],
    entity_key_by_id: Mapping[str, tuple[str, EntityKind]],
) -> tuple[dict[str, set[tuple[str, str]]], dict[str, SemanticEntity], set[str]]:
    """Resolve source, basename, and stem aliases to unique file entities."""
    file_targets: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for source, entity_ids in file_entities_by_source.items():
        file_labels = {source}
        file_labels.update(
            entity_by_key[entity_key_by_id[entity_id]].canonical_name
            for entity_id in entity_ids
        )
        for file_label in file_labels:
            basename = posixpath.basename(file_label)
            stem = basename[:-3] if basename.casefold().endswith(".md") else basename
            for candidate in (file_label, basename, stem):
                file_targets[normalize_entity_name(candidate)].update(
                    (source, entity_id) for entity_id in entity_ids
                )
    file_entity_by_name: dict[str, SemanticEntity] = {}
    ambiguous_file_names: set[str] = set()
    for normalized, targets in file_targets.items():
        if len(targets) == 1:
            _source, entity_id = next(iter(targets))
            file_entity_by_name[normalized] = entity_by_key[entity_key_by_id[entity_id]]
        elif targets:
            ambiguous_file_names.add(normalized)
    return file_targets, file_entity_by_name, ambiguous_file_names


def _discover_entities(
    ordered_chunks: Sequence[Chunk],
    *,
    tenant_id: str,
    generation_id: str,
) -> _EntityBuildResult:
    """Discover entities, aliases, and mentions before relation projection."""
    entity_by_key: dict[tuple[str, EntityKind], SemanticEntity] = {}
    entity_keys_by_name: dict[str, list[tuple[str, EntityKind]]] = defaultdict(list)
    entity_key_by_id: dict[str, tuple[str, EntityKind]] = {}
    labels_by_key: dict[str, set[EntityKind]] = defaultdict(set)
    mentions: list[SemanticMention] = []
    diagnostics: list[SemanticGraphDiagnostic] = []
    declared_aliases = {
        normalize_entity_name(alias)
        for chunk in ordered_chunks
        for _canonical, alias in _chunk_alias_specs(chunk)
    }

    def get_entity(label: str, kind: EntityKind, method: ExtractionMethod) -> SemanticEntity:
        normalized = normalize_entity_name(label)
        key = (normalized, kind)
        labels_by_key[normalized].add(kind)
        entity = entity_by_key.get(key)
        if entity is None:
            entity = SemanticEntity(
                id=_identity(
                    "entity",
                    {
                        "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                        "tenant_id": tenant_id,
                        "generation_id": generation_id,
                        "normalized_name": normalized,
                        "kind": kind,
                    },
                ),
                tenant_id=tenant_id,
                generation_id=generation_id,
                canonical_name=label,
                normalized_name=normalized,
                kind=kind,
                aliases=(label,),
                extraction_method=method,
            )
            entity_by_key[key] = entity
            entity_keys_by_name[normalized].append(key)
            entity_key_by_id[entity.id] = key
        elif label not in entity.aliases:
            entity = replace(entity, aliases=(*entity.aliases, label))
            entity_by_key[key] = entity
        return entity

    # Compute entity specs once because the same values feed entity discovery and mentions.
    specs_per_chunk = [_entity_specs(chunk) for chunk in ordered_chunks]
    entity_ids_by_chunk: dict[str, dict[str, SemanticEntity]] = {}
    file_entities_by_source: dict[str, set[str]] = defaultdict(set)
    mention_ids: set[str] = set()
    for chunk, chunk_specs in zip(ordered_chunks, specs_per_chunk, strict=True):
        by_normalized: dict[str, SemanticEntity] = {}
        for label, kind, method in chunk_specs:
            normalized = normalize_entity_name(label)
            if not normalized or normalized in declared_aliases:
                continue
            entity = get_entity(label, kind, method)
            if kind == "file":
                file_entities_by_source[chunk.source].add(entity.id)
            by_normalized.setdefault(normalized, entity)
        entity_ids_by_chunk[chunk.id] = by_normalized

    alias_candidates: dict[str, set[str]] = defaultdict(set)
    alias_text_by_chunk: dict[str, list[tuple[str, SemanticEntity]]] = defaultdict(list)
    for chunk in ordered_chunks:
        for canonical, alias in _chunk_alias_specs(chunk):
            canonical_key = normalize_entity_name(canonical)
            candidates = [
                entity_by_key[key] for key in entity_keys_by_name.get(canonical_key, [])
            ]
            if not candidates:
                candidates = [get_entity(canonical, "unknown", "metadata")]
            if len(candidates) != 1:
                diagnostics.append(
                    SemanticGraphDiagnostic(
                        id=_identity(
                            "diagnostic",
                            {
                                "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                                "tenant_id": tenant_id,
                                "generation_id": generation_id,
                                "kind": "ambiguous_entity",
                                "reference": normalize_entity_name(alias),
                                "canonical": canonical_key,
                            },
                        ),
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind="ambiguous_entity",
                        reference=normalize_entity_name(alias),
                        message="entity alias resolves to multiple canonical entities",
                        entity_ids=tuple(sorted(entity.id for entity in candidates)),
                    )
                )
                continue
            entity = candidates[0]
            alias_key = normalize_entity_name(alias)
            alias_candidates[alias_key].add(entity.id)
            alias_text_by_chunk[chunk.id].append((alias, entity))
            entity = replace(entity, aliases=tuple(sorted(set((*entity.aliases, alias)))) )
            entity_by_key[(normalize_entity_name(entity.canonical_name), entity.kind)] = entity

    ambiguous_names = {
        normalized for normalized, entity_ids in alias_candidates.items() if len(entity_ids) > 1
    }
    for normalized in sorted(ambiguous_names):
        diagnostics.append(
            SemanticGraphDiagnostic(
                id=_identity(
                    "diagnostic",
                    {
                        "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                        "tenant_id": tenant_id,
                        "generation_id": generation_id,
                        "kind": "ambiguous_entity",
                        "reference": normalized,
                    },
                ),
                tenant_id=tenant_id,
                generation_id=generation_id,
                kind="ambiguous_entity",
                reference=normalized,
                message="entity alias resolves to multiple canonical entities",
                entity_ids=tuple(sorted(alias_candidates[normalized])),
            )
        )

    for chunk, chunk_specs in zip(ordered_chunks, specs_per_chunk, strict=True):
        by_normalized = entity_ids_by_chunk[chunk.id]
        for label, kind, method in chunk_specs:
            normalized = normalize_entity_name(label)
            resolved_entity: SemanticEntity | None
            if normalized in alias_candidates and normalized not in ambiguous_names:
                resolved_entity = entity_by_key[
                    entity_key_by_id[next(iter(alias_candidates[normalized]))]
                ]
                by_normalized[normalized] = resolved_entity
            else:
                resolved_entity = by_normalized.get(normalized)
            if resolved_entity is None:
                continue
            mention_id = _identity(
                "mention",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "entity_id": resolved_entity.id,
                    "chunk_id": chunk.id,
                    "mention_text": label,
                },
            )
            if mention_id in mention_ids:
                continue
            mention_ids.add(mention_id)
            mentions.append(
                SemanticMention(
                    id=mention_id,
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    entity_id=resolved_entity.id,
                    chunk_id=chunk.id,
                    mention_text=label,
                    extraction_method=method,
                    metadata=_chunk_temporal_metadata(chunk),
                )
            )
    return _EntityBuildResult(
        entity_by_key=entity_by_key,
        entity_key_by_id=entity_key_by_id,
        labels_by_key=labels_by_key,
        mentions=mentions,
        diagnostics=diagnostics,
        entity_ids_by_chunk=entity_ids_by_chunk,
        file_entities_by_source=file_entities_by_source,
        mention_ids=mention_ids,
        alias_text_by_chunk=alias_text_by_chunk,
        ambiguous_names=ambiguous_names,
    )


def _decode_explicit_relation(
    raw: Mapping[str, Any],
    chunk: Chunk,
    *,
    local_entities: Mapping[str, SemanticEntity],
    file_entity_by_name: Mapping[str, SemanticEntity],
    ambiguous_names: set[str],
    ambiguous_file_names: set[str],
    tenant_id: str,
    generation_id: str,
    pipeline_fingerprint: str | None,
    corpus_fingerprint: str | None,
) -> SemanticRelation | SemanticGraphDiagnostic:
    """Decode one explicit relation declaration or return its diagnostic."""
    relation = raw.get("relation")
    subject = raw.get("subject")
    object_value = raw.get("object")
    if (
        relation not in RELATION_KINDS
        or not isinstance(subject, str)
        or not isinstance(object_value, str)
    ):
        return SemanticGraphDiagnostic(
            id=_identity(
                "diagnostic",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "kind": "invalid_relation",
                    "reference": chunk.id,
                    "value": raw,
                },
            ),
            tenant_id=tenant_id,
            generation_id=generation_id,
            kind="invalid_relation",
            reference=chunk.id,
            message="relation metadata must name a supported relation and two strings",
        )
    confidence_value = raw.get("confidence", 1.0)
    if (
        isinstance(confidence_value, bool)
        or not isinstance(confidence_value, (int, float))
        or not math.isfinite(confidence_value)
        or not 0.0 <= confidence_value <= 1.0
    ):
        return SemanticGraphDiagnostic(
            id=_identity(
                "diagnostic",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "kind": "invalid_relation",
                    "reference": chunk.id,
                    "field": "confidence",
                    "value": repr(confidence_value),
                },
            ),
            tenant_id=tenant_id,
            generation_id=generation_id,
            kind="invalid_relation",
            reference=chunk.id,
            message="relation confidence must be a finite number between 0.0 and 1.0",
        )
    subject_key = normalize_entity_name(subject)
    object_key = normalize_entity_name(object_value)
    subject_entity = local_entities.get(subject_key) or file_entity_by_name.get(subject_key)
    object_entity = local_entities.get(object_key) or file_entity_by_name.get(object_key)
    if (
        subject_key in ambiguous_names
        or object_key in ambiguous_names
        or subject_key in ambiguous_file_names
        or object_key in ambiguous_file_names
    ):
        subject_entity = None
        object_entity = None
    if subject_entity is None or object_entity is None:
        return SemanticGraphDiagnostic(
            id=_identity(
                "diagnostic",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "kind": "missing_evidence",
                    "reference": chunk.id,
                    "subject": subject,
                    "object": object_value,
                },
            ),
            tenant_id=tenant_id,
            generation_id=generation_id,
            kind="missing_evidence",
            reference=chunk.id,
            message=(
                "relation endpoints must be mentioned by the supporting chunk or "
                "resolve to a unique file"
            ),
        )
    try:
        effective_at, valid_from, valid_until = _relation_temporal_fields(raw)
    except ValueError as exc:
        return SemanticGraphDiagnostic(
            id=_identity(
                "diagnostic",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "kind": "invalid_relation",
                    "reference": chunk.id,
                    "field": "temporal",
                    "value": repr(raw),
                },
            ),
            tenant_id=tenant_id,
            generation_id=generation_id,
            kind="invalid_relation",
            reference=chunk.id,
            message=str(exc),
        )
    structural_type = raw.get("structural_type")
    relation_id = _identity(
        "relation",
        {
            "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
            "tenant_id": tenant_id,
            "generation_id": generation_id,
            "subject_id": subject_entity.id,
            "object_id": object_entity.id,
            "relation": relation,
            "evidence_chunk_ids": [chunk.id],
            "structural_type": structural_type if isinstance(structural_type, str) else None,
            "effective_at": effective_at.isoformat() if effective_at else None,
            "valid_from": valid_from.isoformat() if valid_from else None,
            "valid_until": valid_until.isoformat() if valid_until else None,
        },
    )
    return SemanticRelation(
        id=relation_id,
        tenant_id=tenant_id,
        generation_id=generation_id,
        subject_id=subject_entity.id,
        object_id=object_entity.id,
        relation=relation,
        evidence_chunk_ids=(chunk.id,),
        extraction_method="explicit_relation",
        confidence=float(confidence_value),
        status="authored",
        uncertainty=tuple(item for item in raw.get("uncertainty", ()) if isinstance(item, str)),
        pipeline_fingerprint=pipeline_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        metadata={
            **_relation_metadata(raw, chunk.source),
            **_temporal_metadata(effective_at, valid_from, valid_until),
        },
        effective_at=effective_at,
        valid_from=valid_from,
        valid_until=valid_until,
    )


def build_semantic_graph(
    chunks: Sequence[Chunk],
    *,
    tenant_id: str,
    generation_id: str,
    pipeline_fingerprint: str | None = None,
    corpus_fingerprint: str | None = None,
) -> SemanticGraphProjection:
    """Build a deterministic graph from explicit metadata and conservative headings."""
    ordered_chunks = tuple(sorted(chunks, key=lambda chunk: chunk.id))
    entity_state = _discover_entities(
        ordered_chunks,
        tenant_id=tenant_id,
        generation_id=generation_id,
    )
    entity_by_key = entity_state.entity_by_key
    entity_key_by_id = entity_state.entity_key_by_id
    labels_by_key = entity_state.labels_by_key
    mentions = entity_state.mentions
    diagnostics = entity_state.diagnostics
    entity_ids_by_chunk = entity_state.entity_ids_by_chunk
    file_entities_by_source = entity_state.file_entities_by_source
    mention_ids = entity_state.mention_ids
    alias_text_by_chunk = entity_state.alias_text_by_chunk
    ambiguous_names = entity_state.ambiguous_names

    relations: dict[str, SemanticRelation] = {}
    file_targets, file_entity_by_name, ambiguous_file_names = _file_target_indexes(
        file_entities_by_source=file_entities_by_source,
        entity_by_key=entity_by_key,
        entity_key_by_id=entity_key_by_id,
    )

    # A supersession claim is an authored semantic edge as well as a trust-layer verdict.  Its
    # direction is old document to replacement, so a traversal can move from a stale hit to the
    # document that replaced it.  The trust layer still decides which of the two may be evidence.
    for chunk in ordered_chunks:
        subject_ids = file_entities_by_source.get(chunk.source, set())
        if len(subject_ids) != 1:
            continue
        replacement_id = next(iter(subject_ids))
        for target in _as_labels(chunk.metadata.get("supersedes")):
            target_sources = file_targets.get(normalize_entity_name(supersedes_key(target)), set())
            if len(target_sources) != 1:
                if target_sources:
                    diagnostics.append(
                        SemanticGraphDiagnostic(
                            id=_identity(
                                "diagnostic",
                                {
                                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                                    "tenant_id": tenant_id,
                                    "generation_id": generation_id,
                                    "kind": "ambiguous_entity",
                                    "reference": target,
                                    "source": chunk.source,
                                },
                            ),
                            tenant_id=tenant_id,
                            generation_id=generation_id,
                            kind="ambiguous_entity",
                            reference=target,
                            message="supersession target resolves to multiple files",
                        )
                    )
                continue
            _target_source, superseded_id = next(iter(target_sources))
            raw_temporal = {
                "effective_at": chunk.metadata.get("effective_at")
                or chunk.metadata.get("effective_date")
                or chunk.metadata.get("valid_from"),
                "valid_from": chunk.metadata.get("valid_from"),
                "valid_until": chunk.metadata.get("valid_until"),
            }
            try:
                effective_at, valid_from, valid_until = _relation_temporal_fields(raw_temporal)
            except ValueError as exc:
                diagnostics.append(
                    SemanticGraphDiagnostic(
                        id=_identity(
                            "diagnostic",
                            {
                                "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                                "tenant_id": tenant_id,
                                "generation_id": generation_id,
                                "kind": "invalid_relation",
                                "reference": chunk.id,
                                "value": target,
                            },
                        ),
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind="invalid_relation",
                        reference=chunk.id,
                        message=str(exc),
                    )
                )
                continue
            relation_id = _identity(
                "relation",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "subject_id": superseded_id,
                    "object_id": replacement_id,
                    "relation": "supersedes",
                    "evidence_chunk_ids": [chunk.id],
                    "effective_at": effective_at.isoformat() if effective_at else None,
                    "valid_from": valid_from.isoformat() if valid_from else None,
                    "valid_until": valid_until.isoformat() if valid_until else None,
                },
            )
            relations[relation_id] = SemanticRelation(
                id=relation_id,
                tenant_id=tenant_id,
                generation_id=generation_id,
                subject_id=superseded_id,
                object_id=replacement_id,
                relation="supersedes",
                evidence_chunk_ids=(chunk.id,),
                extraction_method="metadata",
                confidence=1.0,
                status="authored",
                pipeline_fingerprint=pipeline_fingerprint,
                corpus_fingerprint=corpus_fingerprint,
                metadata={
                    "source": chunk.source,
                    "target": target,
                    "edge_kind": "supersession",
                    **_temporal_metadata(effective_at, valid_from, valid_until),
                },
                effective_at=effective_at,
                valid_from=valid_from,
                valid_until=valid_until,
            )

    # `recall_graph.depends_on` is an existing authored dependency contract used by the
    # invalidation and lint layers. Project it into the typed graph as a provenance backed edge,
    # but inspect only the first chunk for each source because frontmatter is copied to every
    # chunk produced from one file.
    first_chunk_by_source: dict[str, Chunk] = {}
    for chunk in ordered_chunks:
        first_chunk_by_source.setdefault(chunk.source, chunk)
    for chunk in ordered_chunks:
        if first_chunk_by_source.get(chunk.source) is not chunk:
            continue
        subject_ids = file_entities_by_source.get(chunk.source, set())
        if len(subject_ids) != 1:
            continue
        subject_id = next(iter(subject_ids))
        try:
            dependencies = dependencies_from_metadata(chunk.metadata)
        except ValueError as exc:
            diagnostics.append(
                SemanticGraphDiagnostic(
                    id=_identity(
                        "diagnostic",
                        {
                            "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                            "tenant_id": tenant_id,
                            "generation_id": generation_id,
                            "kind": "invalid_relation",
                            "reference": chunk.id,
                            "field": "depends_on",
                        },
                    ),
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    kind="invalid_relation",
                    reference=chunk.id,
                    message=str(exc),
                )
            )
            continue
        for target in dependencies:
            target_sources = file_targets.get(normalize_entity_name(target), set())
            if len(target_sources) != 1:
                if len(target_sources) > 1:
                    diagnostic_kind: Literal[
                        "ambiguous_entity", "invalid_relation", "missing_evidence"
                    ] = "ambiguous_entity"
                    message = "dependency target resolves to multiple files"
                else:
                    diagnostic_kind = "missing_evidence"
                    message = "dependency target does not resolve to a unique file"
                diagnostics.append(
                    SemanticGraphDiagnostic(
                        id=_identity(
                            "diagnostic",
                            {
                                "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                                "tenant_id": tenant_id,
                                "generation_id": generation_id,
                                "kind": diagnostic_kind,
                                "reference": chunk.id,
                                "target": target,
                            },
                        ),
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        kind=diagnostic_kind,
                        reference=chunk.id,
                        message=message,
                    )
                )
                continue
            _target_source, object_id = next(iter(target_sources))
            relation_id = _identity(
                "relation",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "subject_id": subject_id,
                    "object_id": object_id,
                    "relation": "depends_on",
                    "evidence_chunk_ids": [chunk.id],
                },
            )
            relations[relation_id] = SemanticRelation(
                id=relation_id,
                tenant_id=tenant_id,
                generation_id=generation_id,
                subject_id=subject_id,
                object_id=object_id,
                relation="depends_on",
                evidence_chunk_ids=(chunk.id,),
                extraction_method="metadata",
                confidence=1.0,
                status="authored",
                pipeline_fingerprint=pipeline_fingerprint,
                corpus_fingerprint=corpus_fingerprint,
                metadata={
                    "source": chunk.source,
                    "target": target,
                    "edge_kind": "dependency",
                },
            )

    # Markdown and wikilinks are authored source references. They are safe to project as
    # `references` only when the target resolves to exactly one file entity. External URLs,
    # missing files, and duplicate basenames are deliberately skipped and diagnosed.
    for chunk in ordered_chunks:
        subject_ids = file_entities_by_source.get(chunk.source, set())
        if len(subject_ids) != 1:
            continue
        subject_id = next(iter(subject_ids))
        for target in _explicit_reference_targets(chunk.text):
            target_sources = file_targets.get(normalize_entity_name(target), set())
            if len(target_sources) != 1:
                if len(target_sources) > 1:
                    diagnostics.append(
                        SemanticGraphDiagnostic(
                            id=_identity(
                                "diagnostic",
                                {
                                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                                    "tenant_id": tenant_id,
                                    "generation_id": generation_id,
                                    "kind": "ambiguous_entity",
                                    "reference": target,
                                    "source": chunk.source,
                                },
                            ),
                            tenant_id=tenant_id,
                            generation_id=generation_id,
                            kind="ambiguous_entity",
                            reference=target,
                            message="explicit file reference resolves to multiple files",
                        )
                    )
                continue
            _target_source, object_id = next(iter(target_sources))
            mention_id = _identity(
                "mention",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "entity_id": object_id,
                    "chunk_id": chunk.id,
                    "mention_text": target,
                },
            )
            if mention_id not in mention_ids:
                mention_ids.add(mention_id)
                mentions.append(
                    SemanticMention(
                        id=mention_id,
                        tenant_id=tenant_id,
                        generation_id=generation_id,
                        entity_id=object_id,
                        chunk_id=chunk.id,
                        mention_text=target,
                        extraction_method="explicit_reference",
                        metadata=_chunk_temporal_metadata(chunk),
                    )
                )
            relation_id = _identity(
                "relation",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "subject_id": subject_id,
                    "object_id": object_id,
                    "relation": "references",
                    "evidence_chunk_ids": [chunk.id],
                },
            )
            relations[relation_id] = SemanticRelation(
                id=relation_id,
                tenant_id=tenant_id,
                generation_id=generation_id,
                subject_id=subject_id,
                object_id=object_id,
                relation="references",
                evidence_chunk_ids=(chunk.id,),
                extraction_method="explicit_reference",
                confidence=1.0,
                status="authored",
                pipeline_fingerprint=pipeline_fingerprint,
                corpus_fingerprint=corpus_fingerprint,
                metadata={"source": chunk.source, "target": target},
            )
        for alias, entity in alias_text_by_chunk.get(chunk.id, ()):
            mention_id = _identity(
                "mention",
                {
                    "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                    "tenant_id": tenant_id,
                    "generation_id": generation_id,
                    "entity_id": entity.id,
                    "chunk_id": chunk.id,
                    "mention_text": alias,
                },
            )
            if mention_id in mention_ids:
                continue
            mention_ids.add(mention_id)
            mentions.append(
                SemanticMention(
                    id=mention_id,
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    entity_id=entity.id,
                    chunk_id=chunk.id,
                    mention_text=alias,
                    extraction_method="metadata",
                    metadata=_chunk_temporal_metadata(chunk),
                )
            )

    for normalized, kinds in sorted(labels_by_key.items()):
        if len(kinds) > 1 and "unknown" not in kinds:
            kind_entity_ids = tuple(
                entity_by_key[(normalized, kind)].id for kind in sorted(kinds)
            )
            diagnostics.append(
                SemanticGraphDiagnostic(
                    id=_identity(
                        "diagnostic",
                        {
                            "schema_version": SEMANTIC_GRAPH_SCHEMA_VERSION,
                            "tenant_id": tenant_id,
                            "generation_id": generation_id,
                            "kind": "ambiguous_entity",
                            "reference": normalized,
                        },
                    ),
                    tenant_id=tenant_id,
                    generation_id=generation_id,
                    kind="ambiguous_entity",
                    reference=normalized,
                    message=f"entity label {normalized!r} has multiple explicit kinds",
                    entity_ids=kind_entity_ids,
                )
            )

    for chunk in ordered_chunks:
        local_entities = entity_ids_by_chunk.get(chunk.id, {})
        for raw in _chunk_relation_specs(chunk):
            decoded = _decode_explicit_relation(
                raw,
                chunk,
                local_entities=local_entities,
                file_entity_by_name=file_entity_by_name,
                ambiguous_names=ambiguous_names,
                ambiguous_file_names=ambiguous_file_names,
                tenant_id=tenant_id,
                generation_id=generation_id,
                pipeline_fingerprint=pipeline_fingerprint,
                corpus_fingerprint=corpus_fingerprint,
            )
            if isinstance(decoded, SemanticGraphDiagnostic):
                diagnostics.append(decoded)
            else:
                relations[decoded.id] = decoded

    return _finalize_graph(
        entity_by_key=entity_by_key,
        tenant_id=tenant_id,
        generation_id=generation_id,
        pipeline_fingerprint=pipeline_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        mentions=mentions,
        relations=relations,
        diagnostics=diagnostics,
    )
