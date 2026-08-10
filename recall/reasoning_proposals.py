"""Side effect free inference proposal APIs for reasoning over evidence graphs."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from recall.frontmatter import supersedes_key
from recall.lineage import canonical_sha256
from recall.reasoning_graph import (
    ReasoningGraphEdge,
    ReasoningGraphNode,
    ReasoningGraphProjection,
)

PROPOSAL_SCHEMA_VERSION = 1
DETERMINISTIC_PROVIDER_ID = "recall.deterministic"
DETERMINISTIC_MODEL_ID = "rules"
DETERMINISTIC_PROVIDER_REVISION = "session3-v1"

ProposalStatus = Literal["candidate", "rejected", "requires_review"]
ProposedRelation = Literal["supersedes", "contradicts", "same_entity", "references"]
ProviderFailureKind = Literal["timeout", "malformed_output", "wrong_cardinality", "provider_error"]
PROVIDER_FAILURE_KINDS: tuple[ProviderFailureKind, ...] = (
    "timeout",
    "malformed_output",
    "wrong_cardinality",
    "provider_error",
)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_value(item)
                for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    return value


@dataclass(frozen=True)
class ProposalContext:
    tenant_id: str
    generation_id: str
    pipeline_id: str
    provider_id: str
    model_id: str
    provider_revision: str


@dataclass(frozen=True)
class EvidenceClaim:
    id: str
    evidence_id: str
    text: str
    entity_id: str | None = None
    subject: str | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))


@dataclass(frozen=True)
class EntityResolution:
    id: str
    evidence_ids: tuple[str, ...]
    canonical_entity: str
    aliases: tuple[str, ...] = ()
    confidence: float | None = None
    uncertainty: tuple[str, ...] = ()


@dataclass(frozen=True)
class InferenceProposal:
    id: str
    source_evidence_ids: tuple[str, ...]
    proposed_relation: ProposedRelation
    subject_id: str
    object_id: str
    explanation: str
    model_id: str
    pipeline_id: str
    provider_id: str
    provider_revision: str
    confidence: float | None
    uncertainty: tuple[str, ...]
    generation_id: str
    status: ProposalStatus = "candidate"
    rule_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_evidence_ids", tuple(self.source_evidence_ids))
        object.__setattr__(self, "uncertainty", tuple(self.uncertainty))
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))


@dataclass(frozen=True)
class ProviderFailure:
    kind: ProviderFailureKind
    provider_id: str
    model_id: str
    provider_revision: str
    message: str


@dataclass(frozen=True)
class ProposalProtocolReport:
    schema_version: int
    generation_id: str
    pipeline_id: str
    proposals: tuple[InferenceProposal, ...]
    rejected_proposals: tuple[InferenceProposal, ...]
    provider_failures: tuple[ProviderFailure, ...]

    @property
    def failure_matrix(self) -> Mapping[str, int]:
        counts: dict[str, int] = {kind: 0 for kind in PROVIDER_FAILURE_KINDS}
        for failure in self.provider_failures:
            counts[failure.kind] += 1
        return MappingProxyType(counts)


class ClaimExtractor(Protocol):
    def extract_claims(
        self, evidence: Sequence[ReasoningGraphNode], context: ProposalContext
    ) -> Sequence[EvidenceClaim]:
        ...


class EntityResolver(Protocol):
    def resolve_entities(
        self, claims: Sequence[EvidenceClaim], context: ProposalContext
    ) -> Sequence[EntityResolution]:
        ...


class RelationProposer(Protocol):
    def propose_relations(
        self,
        claims: Sequence[EvidenceClaim],
        entities: Sequence[EntityResolution],
        context: ProposalContext,
    ) -> Sequence[InferenceProposal]:
        ...


class ContradictionDetector(Protocol):
    def detect_contradictions(
        self, claims: Sequence[EvidenceClaim], context: ProposalContext
    ) -> Sequence[InferenceProposal]:
        ...


class ModelBackedProposalProvider(Protocol):
    """Optional provider port. Implementations may call a model, this module never does."""

    provider_id: str
    model_id: str
    provider_revision: str
    max_proposals: int

    def propose(
        self, graph: ReasoningGraphProjection, context: ProposalContext
    ) -> Sequence[InferenceProposal | Mapping[str, Any]]:
        ...


def _proposal_id(payload: Mapping[str, Any]) -> str:
    return f"ip_{canonical_sha256({'schema_version': PROPOSAL_SCHEMA_VERSION, **payload})[:24]}"


def _context(
    graph: ReasoningGraphProjection,
    *,
    pipeline_id: str | None,
    provider_id: str,
    model_id: str,
    provider_revision: str,
) -> ProposalContext:
    return ProposalContext(
        tenant_id=graph.tenant_id,
        generation_id=graph.generation_id,
        pipeline_id=pipeline_id or graph.pipeline_fingerprint or "unknown-pipeline",
        provider_id=provider_id,
        model_id=model_id,
        provider_revision=provider_revision,
    )


def _source_nodes(graph: ReasoningGraphProjection) -> tuple[ReasoningGraphNode, ...]:
    return tuple(node for node in graph.nodes if node.kind == "source" and node.file is not None)


def _chunk_nodes_by_file(
    graph: ReasoningGraphProjection,
) -> Mapping[str, tuple[ReasoningGraphNode, ...]]:
    by_file: dict[str, list[ReasoningGraphNode]] = defaultdict(list)
    for node in graph.nodes:
        if node.kind == "chunk" and node.file is not None:
            by_file[node.file].append(node)
    return MappingProxyType(
        {
            file: tuple(sorted(nodes, key=lambda node: (node.ord is None, node.ord, node.id)))
            for file, nodes in by_file.items()
        }
    )


def _chunk_texts_by_file(graph: ReasoningGraphProjection) -> Mapping[str, str]:
    by_file: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for node in graph.nodes:
        if node.kind != "chunk" or node.file is None:
            continue
        text = node.metadata.get("text")
        if not isinstance(text, str):
            text = ""
        order = node.ord if node.ord is not None else 0
        by_file[node.file].append((order, text))
    return MappingProxyType(
        {
            file: "\n".join(text for _order, text in sorted(items))
            for file, items in by_file.items()
        }
    )


def _source_claims(graph: ReasoningGraphProjection) -> Mapping[str, EvidenceClaim]:
    chunks = _chunk_nodes_by_file(graph)
    texts = _chunk_texts_by_file(graph)
    claims: dict[str, EvidenceClaim] = {}
    for node in _source_nodes(graph):
        file = node.file
        assert file is not None
        chunk_nodes = chunks.get(file, ())
        valid_from = _earliest_datetime(
            chunk.validity.get("valid_from") for chunk in chunk_nodes if chunk.validity
        )
        valid_until = _latest_datetime(
            chunk.validity.get("valid_until") for chunk in chunk_nodes if chunk.validity
        )
        text = texts.get(file, "")
        if not text:
            text = " ".join(
                str(chunk.metadata.get(key, ""))
                for chunk in chunk_nodes
                for key in ("title", "heading")
            ).strip()
        subject = _decision_subject(file, text)
        claims[file] = EvidenceClaim(
            id=node.id,
            evidence_id=node.id,
            text=text,
            entity_id=subject,
            subject=subject,
            valid_from=valid_from,
            valid_until=valid_until,
            metadata={"file": file, "source": node.source},
        )
    return MappingProxyType(claims)


def _earliest_datetime(values: Iterable[Any]) -> datetime | None:
    dates = [value for value in values if isinstance(value, datetime)]
    return min(dates) if dates else None


def _latest_datetime(values: Iterable[Any]) -> datetime | None:
    dates = [value for value in values if isinstance(value, datetime)]
    return max(dates) if dates else None


_VERSION_RE = re.compile(r"^(?P<base>.*?)(?:[_\-. ]?v(?P<version>\d+))(?P<suffix>\.[^.]+)?$")
_DECISION_SUBJECT_RE = re.compile(
    r"(?:decision|decided|choice|policy|status|verdict)\s*[:=\-]\s*(?P<subject>[^\n.;]+)",
    re.IGNORECASE,
)
_DIRECT_REF_RE = re.compile(
    r"\b(?:replaces|supersedes|deprecates|updates|revises)\s+`?(?P<target>[\w./\-[\]]+)`?",
    re.IGNORECASE,
)


def _version_key(file: str) -> tuple[str, int] | None:
    stem = file.rsplit("/", 1)[-1]
    match = _VERSION_RE.match(stem)
    if not match:
        return None
    return supersedes_key(f"{match.group('base')}{match.group('suffix') or ''}"), int(
        match.group("version")
    )


def _decision_subject(file: str, text: str) -> str:
    match = _DECISION_SUBJECT_RE.search(text[:1000])
    if match:
        return supersedes_key(match.group("subject"))
    stem = supersedes_key(file)
    return re.sub(r"(?:[_\-. ]?v\d+)$", "", stem, flags=re.IGNORECASE)


def _authored_supersession_pairs(graph: ReasoningGraphProjection) -> set[tuple[str, str]]:
    return {
        (edge.from_file, edge.to_file)
        for edge in graph.authored_edges
        if edge.kind == "authored_supersedes" and edge.to_file is not None
    }


def _make_proposal(
    *,
    graph: ReasoningGraphProjection,
    context: ProposalContext,
    source_evidence_ids: Sequence[str],
    proposed_relation: ProposedRelation,
    subject_id: str,
    object_id: str,
    explanation: str,
    confidence: float | None,
    uncertainty: Sequence[str] = (),
    status: ProposalStatus = "candidate",
    rule_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> InferenceProposal:
    evidence = tuple(sorted(source_evidence_ids))
    payload = {
        "tenant_id": graph.tenant_id,
        "generation_id": graph.generation_id,
        "pipeline_id": context.pipeline_id,
        "provider_id": context.provider_id,
        "model_id": context.model_id,
        "provider_revision": context.provider_revision,
        "evidence": evidence,
        "relation": proposed_relation,
        "subject_id": subject_id,
        "object_id": object_id,
        "rule_id": rule_id,
        "status": status,
    }
    return InferenceProposal(
        id=_proposal_id(payload),
        source_evidence_ids=evidence,
        proposed_relation=proposed_relation,
        subject_id=subject_id,
        object_id=object_id,
        explanation=explanation,
        model_id=context.model_id,
        pipeline_id=context.pipeline_id,
        provider_id=context.provider_id,
        provider_revision=context.provider_revision,
        confidence=confidence,
        uncertainty=tuple(uncertainty),
        generation_id=graph.generation_id,
        status=status,
        rule_id=rule_id,
        metadata=metadata or {},
    )


def deterministic_inference_proposals(
    graph: ReasoningGraphProjection, *, pipeline_id: str | None = None
) -> tuple[InferenceProposal, ...]:
    """Return high precision deterministic proposals without mutating the graph."""
    context = _context(
        graph,
        pipeline_id=pipeline_id,
        provider_id=DETERMINISTIC_PROVIDER_ID,
        model_id=DETERMINISTIC_MODEL_ID,
        provider_revision=DETERMINISTIC_PROVIDER_REVISION,
    )
    claims = _source_claims(graph)
    authored = _authored_supersession_pairs(graph)
    proposals: list[InferenceProposal] = []
    proposals.extend(_explicit_version_proposals(graph, context, claims, authored))
    proposals.extend(_direct_reference_proposals(graph, context, claims, authored))
    proposals.extend(_repeated_decision_subject_proposals(graph, context, claims, authored))
    proposals.extend(_contradictory_validity_window_proposals(graph, context, claims))
    proposals.extend(_temporal_ordering_proposals(graph, context, claims, authored))

    by_id = {proposal.id: proposal for proposal in proposals}
    return tuple(by_id[key] for key in sorted(by_id))


def _explicit_version_proposals(
    graph: ReasoningGraphProjection,
    context: ProposalContext,
    claims: Mapping[str, EvidenceClaim],
    authored: set[tuple[str, str]],
) -> list[InferenceProposal]:
    by_base: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for file in claims:
        version = _version_key(file)
        if version is not None:
            by_base[version[0]].append((version[1], file))
    proposals: list[InferenceProposal] = []
    for base, files in sorted(by_base.items()):
        ordered = sorted(files)
        for (old_version, old_file), (new_version, new_file) in zip(
            ordered, ordered[1:], strict=False
        ):
            if old_version >= new_version or (old_file, new_file) in authored:
                continue
            proposals.append(
                _make_proposal(
                    graph=graph,
                    context=context,
                    source_evidence_ids=(claims[old_file].evidence_id, claims[new_file].evidence_id),
                    proposed_relation="supersedes",
                    subject_id=old_file,
                    object_id=new_file,
                    explanation=(
                        "The source names form an explicit increasing version sequence for "
                        f"{base}: {old_file} before {new_file}."
                    ),
                    confidence=0.86,
                    uncertainty=("version naming is structural, not authored metadata",),
                    rule_id="deterministic.explicit_version_naming",
                )
            )
    return proposals


def _direct_reference_proposals(
    graph: ReasoningGraphProjection,
    context: ProposalContext,
    claims: Mapping[str, EvidenceClaim],
    authored: set[tuple[str, str]],
) -> list[InferenceProposal]:
    by_key = {supersedes_key(file): file for file in claims}
    proposals: list[InferenceProposal] = []
    for file, claim in sorted(claims.items()):
        for match in _DIRECT_REF_RE.finditer(claim.text[:4000]):
            matched_target = match.group("target").rstrip(".,;:)")
            target = by_key.get(supersedes_key(matched_target))
            if target is None or target == file or (target, file) in authored:
                continue
            proposals.append(
                _make_proposal(
                    graph=graph,
                    context=context,
                    source_evidence_ids=(claims[target].evidence_id, claim.evidence_id),
                    proposed_relation="supersedes",
                    subject_id=target,
                    object_id=file,
                    explanation=(
                        f"{file} contains a direct textual reference that appears to replace "
                        f"{target}."
                    ),
                    confidence=0.9,
                    uncertainty=("the reference remains corpus text, not an instruction",),
                    rule_id="deterministic.direct_textual_reference",
                    metadata={"matched_text": match.group(0).rstrip(".,;:)")},
                )
            )
    return proposals


def _repeated_decision_subject_proposals(
    graph: ReasoningGraphProjection,
    context: ProposalContext,
    claims: Mapping[str, EvidenceClaim],
    authored: set[tuple[str, str]],
) -> list[InferenceProposal]:
    by_subject: dict[str, list[EvidenceClaim]] = defaultdict(list)
    for claim in claims.values():
        if claim.subject:
            by_subject[claim.subject].append(claim)
    proposals: list[InferenceProposal] = []
    for subject, subject_claims in sorted(by_subject.items()):
        ordered = sorted(
            subject_claims,
            key=lambda claim: (
                claim.valid_from is None,
                claim.valid_from or datetime.min,
                str(claim.metadata.get("file")),
            ),
        )
        for older, newer in zip(ordered, ordered[1:], strict=False):
            older_file = str(older.metadata["file"])
            newer_file = str(newer.metadata["file"])
            if older_file == newer_file or (older_file, newer_file) in authored:
                continue
            if older.valid_from is None and newer.valid_from is None:
                continue
            proposals.append(
                _make_proposal(
                    graph=graph,
                    context=context,
                    source_evidence_ids=(older.evidence_id, newer.evidence_id),
                    proposed_relation="supersedes",
                    subject_id=older_file,
                    object_id=newer_file,
                    explanation=(
                        f"{older_file} and {newer_file} repeat decision subject {subject}, "
                        "with the latter having the later validity start."
                    ),
                    confidence=0.72,
                    uncertainty=("same subject is inferred from text and file names",),
                    status="requires_review",
                    rule_id="deterministic.repeated_decision_subject",
                )
            )
    return proposals


def _contradictory_validity_window_proposals(
    graph: ReasoningGraphProjection,
    context: ProposalContext,
    claims: Mapping[str, EvidenceClaim],
) -> list[InferenceProposal]:
    proposals: list[InferenceProposal] = []
    claims_by_subject: dict[str, list[EvidenceClaim]] = defaultdict(list)
    for claim in claims.values():
        if claim.subject:
            claims_by_subject[claim.subject].append(claim)
    for subject, subject_claims in sorted(claims_by_subject.items()):
        ordered = sorted(subject_claims, key=lambda claim: str(claim.metadata.get("file")))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if _windows_overlap(left, right) and _opposing_validity_text(left.text, right.text):
                    left_file = str(left.metadata["file"])
                    right_file = str(right.metadata["file"])
                    proposals.append(
                        _make_proposal(
                            graph=graph,
                            context=context,
                            source_evidence_ids=(left.evidence_id, right.evidence_id),
                            proposed_relation="contradicts",
                            subject_id=left_file,
                            object_id=right_file,
                            explanation=(
                                f"{left_file} and {right_file} discuss {subject} in overlapping "
                                "validity windows with opposing status language."
                            ),
                            confidence=0.78,
                            uncertainty=("textual polarity is heuristic",),
                            status="requires_review",
                            rule_id="deterministic.contradictory_validity_windows",
                        )
                    )
    return proposals


def _temporal_ordering_proposals(
    graph: ReasoningGraphProjection,
    context: ProposalContext,
    claims: Mapping[str, EvidenceClaim],
    authored: set[tuple[str, str]],
) -> list[InferenceProposal]:
    proposals: list[InferenceProposal] = []
    by_entity: dict[str, list[EvidenceClaim]] = defaultdict(list)
    for claim in claims.values():
        if claim.entity_id:
            by_entity[claim.entity_id].append(claim)
    for entity_id, entity_claims in sorted(by_entity.items()):
        ordered = sorted(
            entity_claims,
            key=lambda claim: (
                claim.valid_from is None,
                claim.valid_from or datetime.min,
                str(claim.metadata.get("file")),
            ),
        )
        for older, newer in zip(ordered, ordered[1:], strict=False):
            older_file = str(older.metadata["file"])
            newer_file = str(newer.metadata["file"])
            if older_file == newer_file or (older_file, newer_file) in authored:
                continue
            if older.valid_from is None or newer.valid_from is None:
                continue
            if older.valid_from >= newer.valid_from:
                continue
            proposals.append(
                _make_proposal(
                    graph=graph,
                    context=context,
                    source_evidence_ids=(older.evidence_id, newer.evidence_id),
                    proposed_relation="supersedes",
                    subject_id=older_file,
                    object_id=newer_file,
                    explanation=(
                        f"{newer_file} has a later validity start than {older_file} for the "
                        f"same normalized entity {entity_id}."
                    ),
                    confidence=0.64,
                    uncertainty=("temporal ordering alone is insufficient for promotion",),
                    status="requires_review",
                    rule_id="deterministic.temporal_ordering",
                )
            )
    return proposals


def _windows_overlap(left: EvidenceClaim, right: EvidenceClaim) -> bool:
    left_start = left.valid_from or datetime.min.replace(tzinfo=timezone.utc)
    right_start = right.valid_from or datetime.min.replace(tzinfo=timezone.utc)
    left_end = left.valid_until or datetime.max.replace(tzinfo=timezone.utc)
    right_end = right.valid_until or datetime.max.replace(tzinfo=timezone.utc)
    return max(left_start, right_start) <= min(left_end, right_end)


def _opposing_validity_text(left: str, right: str) -> bool:
    positive = re.compile(r"\b(enabled|active|valid|approved|ship|on)\b", re.IGNORECASE)
    negative = re.compile(r"\b(disabled|inactive|invalid|rejected|blocked|off)\b", re.IGNORECASE)
    return bool(
        (positive.search(left) and negative.search(right))
        or (negative.search(left) and positive.search(right))
    )


def _coerce_provider_proposal(
    raw: InferenceProposal | Mapping[str, Any],
    *,
    graph: ReasoningGraphProjection,
    context: ProposalContext,
    known_evidence_ids: set[str],
) -> InferenceProposal:
    if isinstance(raw, InferenceProposal):
        proposal = raw
    else:
        required = {
            "source_evidence_ids",
            "proposed_relation",
            "subject_id",
            "object_id",
            "explanation",
            "confidence",
            "uncertainty",
        }
        missing = sorted(key for key in required if key not in raw)
        if missing:
            raise ValueError(f"provider proposal missing fields: {', '.join(missing)}")
        proposal = _make_proposal(
            graph=graph,
            context=context,
            source_evidence_ids=tuple(str(item) for item in raw["source_evidence_ids"]),
            proposed_relation=_checked_relation(raw["proposed_relation"]),
            subject_id=str(raw["subject_id"]),
            object_id=str(raw["object_id"]),
            explanation=str(raw["explanation"]),
            confidence=_checked_confidence(raw["confidence"]),
            uncertainty=tuple(str(item) for item in raw["uncertainty"]),
            status=_checked_status(raw.get("status", "candidate")),
            rule_id=str(raw["rule_id"]) if raw.get("rule_id") is not None else None,
            metadata=raw["metadata"] if isinstance(raw.get("metadata"), Mapping) else {},
        )
    if proposal.generation_id != graph.generation_id:
        raise ValueError("provider proposal generation_id does not match graph generation")
    if proposal.pipeline_id != context.pipeline_id:
        raise ValueError("provider proposal pipeline_id does not match request context")
    if not proposal.source_evidence_ids:
        raise ValueError("provider proposal has no source evidence")
    unknown = sorted(set(proposal.source_evidence_ids) - known_evidence_ids)
    if unknown:
        raise ValueError(f"provider proposal cites unknown evidence ids: {', '.join(unknown)}")
    return proposal


def _checked_relation(value: Any) -> ProposedRelation:
    if value not in {"supersedes", "contradicts", "same_entity", "references"}:
        raise ValueError(f"unknown proposed_relation: {value!r}")
    return cast(ProposedRelation, value)


def _checked_status(value: Any) -> ProposalStatus:
    if value not in {"candidate", "rejected", "requires_review"}:
        raise ValueError(f"unknown status: {value!r}")
    return cast(ProposalStatus, value)


def _checked_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("confidence must be a number or null")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return score


def proposal_report(
    graph: ReasoningGraphProjection,
    *,
    pipeline_id: str | None = None,
    model_provider: ModelBackedProposalProvider | None = None,
) -> ProposalProtocolReport:
    """Run deterministic and optional model backed proposal generation.

    Provider failures are returned in band. A malformed provider cannot silently add or drop
    proposals, and deterministic proposals still survive provider failure.
    """
    pipeline = pipeline_id or graph.pipeline_fingerprint or "unknown-pipeline"
    deterministic = deterministic_inference_proposals(graph, pipeline_id=pipeline)
    accepted: list[InferenceProposal] = list(deterministic)
    rejected: list[InferenceProposal] = []
    failures: list[ProviderFailure] = []
    if model_provider is not None:
        provider_context = _context(
            graph,
            pipeline_id=pipeline,
            provider_id=model_provider.provider_id,
            model_id=model_provider.model_id,
            provider_revision=model_provider.provider_revision,
        )
        try:
            raw_proposals = tuple(model_provider.propose(graph, provider_context))
            if len(raw_proposals) > model_provider.max_proposals:
                raise ValueError(
                    f"provider returned {len(raw_proposals)} proposals, "
                    f"maximum is {model_provider.max_proposals}"
                )
            known_evidence_ids = {node.id for node in graph.nodes}
            for raw in raw_proposals:
                proposal = _coerce_provider_proposal(
                    raw,
                    graph=graph,
                    context=provider_context,
                    known_evidence_ids=known_evidence_ids,
                )
                if proposal.status == "rejected":
                    rejected.append(proposal)
                else:
                    accepted.append(proposal)
        except TimeoutError as exc:
            failures.append(_provider_failure(model_provider, "timeout", str(exc)))
        except ValueError as exc:
            message = str(exc)
            kind: ProviderFailureKind = (
                "wrong_cardinality" if "maximum is" in message else "malformed_output"
            )
            failures.append(_provider_failure(model_provider, kind, message))
        except Exception as exc:  # noqa: BLE001
            failures.append(_provider_failure(model_provider, "provider_error", str(exc)))
    by_id = {proposal.id: proposal for proposal in accepted}
    rejected_by_id = {proposal.id: proposal for proposal in rejected}
    return ProposalProtocolReport(
        schema_version=PROPOSAL_SCHEMA_VERSION,
        generation_id=graph.generation_id,
        pipeline_id=pipeline,
        proposals=tuple(by_id[key] for key in sorted(by_id)),
        rejected_proposals=tuple(rejected_by_id[key] for key in sorted(rejected_by_id)),
        provider_failures=tuple(failures),
    )


def _provider_failure(
    provider: ModelBackedProposalProvider, kind: ProviderFailureKind, message: str
) -> ProviderFailure:
    return ProviderFailure(
        kind=kind,
        provider_id=provider.provider_id,
        model_id=provider.model_id,
        provider_revision=provider.provider_revision,
        message=message,
    )


def proposal_to_graph_edge(
    graph: ReasoningGraphProjection, proposal: InferenceProposal
) -> ReasoningGraphEdge:
    """Represent a supersession proposal as a graph candidate edge.

    Only `supersedes` proposals can become candidate edges. The returned edge remains separate from
    authored edges and is not a trust input unless a caller explicitly passes it as proposal data.
    """
    if proposal.proposed_relation != "supersedes":
        raise ValueError("only supersedes proposals can be represented as candidate graph edges")
    node_by_file = {node.file: node for node in _source_nodes(graph)}
    from_node = node_by_file.get(proposal.subject_id)
    to_node = node_by_file.get(proposal.object_id)
    payload = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "tenant_id": graph.tenant_id,
        "generation_id": graph.generation_id,
        "proposal_id": proposal.id,
        "from_file": proposal.subject_id,
        "to_file": proposal.object_id,
    }
    return ReasoningGraphEdge(
        id=f"rg_edge_candidate_{canonical_sha256(payload)[:24]}",
        kind="inferred_candidate_supersedes",
        tenant_id=graph.tenant_id,
        generation_id=graph.generation_id,
        from_node_id=from_node.id if from_node is not None else "",
        to_node_id=to_node.id if to_node is not None else None,
        from_file=proposal.subject_id,
        to_file=proposal.object_id,
        authored_reference=None,
        asserted_at=None,
        provenance={
            "proposal_id": proposal.id,
            "provider_id": proposal.provider_id,
            "model_id": proposal.model_id,
            "provider_revision": proposal.provider_revision,
            "source_evidence_ids": proposal.source_evidence_ids,
        },
        metadata={
            "status": proposal.status,
            "confidence": proposal.confidence,
            "uncertainty": proposal.uncertainty,
            "explanation": proposal.explanation,
        },
    )


def proposal_precision_recall(
    proposals: Sequence[InferenceProposal], expected_pairs: set[tuple[str, str]]
) -> Mapping[str, float | int]:
    predicted = {
        (proposal.subject_id, proposal.object_id)
        for proposal in proposals
        if proposal.proposed_relation == "supersedes" and proposal.status != "rejected"
    }
    true_positive = len(predicted & expected_pairs)
    false_positive = len(predicted - expected_pairs)
    false_negative = len(expected_pairs - predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected_pairs) if expected_pairs else 1.0
    return MappingProxyType(
        {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
        }
    )
