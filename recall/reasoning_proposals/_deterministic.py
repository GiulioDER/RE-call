"""Deterministic inference proposal rules."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
import heapq
from types import MappingProxyType
from typing import Any

from recall.frontmatter import supersedes_key
from recall.lineage import canonical_sha256
from recall.reasoning_graph import EVIDENCE_TEXT_METADATA_KEY, ReasoningGraphNode, ReasoningGraphProjection
from recall.supersession_cues import clause_head, unasserted
from recall.reasoning_proposals.types import (
    DETERMINISTIC_MODEL_ID,
    DETERMINISTIC_PROVIDER_ID,
    DETERMINISTIC_PROVIDER_REVISION,
    PROPOSAL_SCHEMA_VERSION,
    EvidenceClaim,
    InferenceProposal,
    ProposalContext,
    ProposalStatus,
    ProposedRelation,
)

def _proposal_id(payload: Mapping[str, Any]) -> str:
    return f"ip_{canonical_sha256({'schema_version': PROPOSAL_SCHEMA_VERSION, **payload})[:24]}"


def _resolve_pipeline_id(graph: ReasoningGraphProjection, pipeline_id: str | None) -> str:
    resolved = pipeline_id or graph.pipeline_fingerprint
    if not resolved:
        raise ValueError("pipeline_id is required when graph.pipeline_fingerprint is absent")
    return resolved


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
        pipeline_id=_resolve_pipeline_id(graph, pipeline_id),
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
        text = node.metadata.get(EVIDENCE_TEXT_METADATA_KEY)
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
    r"\b(?P<cue>replaces|supersedes|deprecates|updates|revises)\s+`?(?P<target>[\w./\-[\]]+)`?",
    re.IGNORECASE,
)
#: A document named as the subject before a cue: "new.md replaces old.md", "[[new]] supersedes".
#: Only explicit forms, because a bare word ("the team replaces ...") would resolve against any
#: memo that happens to share its stem.
_SUBJECT_REF_RE = re.compile(r"\[\[(?P<wiki>[^\]\n]+)\]\]|`?(?P<file>[\w./\-]+\.md)\b`?")
_POSITIVE_VALIDITY_RE = re.compile(r"\b(enabled|active|valid|approved|ship|on)\b", re.IGNORECASE)
_NEGATIVE_VALIDITY_RE = re.compile(
    r"\b(disabled|inactive|invalid|rejected|blocked|off)\b", re.IGNORECASE
)

#: Per rule confidence priors, keyed by rule_id. These are UNMEASURED priors: they encode the
#: relative strength of each structural signal as judged when the rules were written, not any
#: measured precision. Proposal ids do not hash confidence, so changing a value here never
#: changes an id. Every proposal a rule emits carries that rule's value verbatim.
RULE_CONFIDENCE: Mapping[str, float] = MappingProxyType(
    {
        "deterministic.explicit_version_naming": 0.86,
        "deterministic.direct_textual_reference": 0.9,
        "deterministic.repeated_decision_subject": 0.72,
        "deterministic.contradictory_validity_windows": 0.78,
        "deterministic.temporal_ordering": 0.64,
    }
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
                    confidence=RULE_CONFIDENCE["deterministic.explicit_version_naming"],
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
    # A stem two files share names neither of them, as `recall.supersession.resolve_supersession`
    # reads it: such a reference is skipped. Keeping the last file seen proposed an edge onto
    # whichever one happened to sort last.
    by_key: dict[str, list[str]] = defaultdict(list)
    for file in claims:
        by_key[supersedes_key(file)].append(file)

    def unique(reference: str) -> str | None:
        files = by_key.get(supersedes_key(reference), [])
        return files[0] if len(files) == 1 else None

    proposals: list[InferenceProposal] = []
    for file, claim in sorted(claims.items()):
        text = claim.text[:4000]
        for match in _DIRECT_REF_RE.finditer(text):
            matched_target = match.group("target").rstrip(".,;:)")
            target_span = (match.start("target"), match.start("target") + len(matched_target))
            # "This draft never supersedes old.md" names the relation without asserting it.
            if unasserted(text, match.span("cue"), target_span):
                continue
            target = unique(matched_target)
            if target is None:
                continue
            superseder = _named_subject(text, match.start("cue"), target_span, unique, file)
            if superseder is None or target == superseder or (target, superseder) in authored:
                continue
            reported = superseder != file
            proposals.append(
                _make_proposal(
                    graph=graph,
                    context=context,
                    source_evidence_ids=tuple(
                        sorted({claims[target].evidence_id, claims[superseder].evidence_id, claim.evidence_id})
                    ),
                    proposed_relation="supersedes",
                    subject_id=target,
                    object_id=superseder,
                    explanation=(
                        f"{file} says that {superseder} replaces {target}."
                        if reported
                        else f"{file} contains a direct textual reference that appears to replace "
                        f"{target}."
                    ),
                    confidence=RULE_CONFIDENCE["deterministic.direct_textual_reference"],
                    uncertainty=(
                        ("the reference remains corpus text, not an instruction",)
                        + (("the relation is reported by a third document",) if reported else ())
                    ),
                    status="requires_review" if reported else "candidate",
                    rule_id="deterministic.direct_textual_reference",
                    metadata={"matched_text": match.group(0).rstrip(".,;:)")},
                )
            )
    return proposals


def _named_subject(
    text: str,
    cue_start: int,
    target_span: tuple[int, int],
    unique: Callable[[str], str | None],
    holder: str,
) -> str | None:
    """The document that supersedes in a direct reference, or None when it cannot be told.

    The file holding the sentence, unless the clause before the cue names a document: "new.md
    replaces old.md" in meeting notes is new.md replacing old.md, and taking the notes as the
    superseder demoted old.md beneath a memo that never claimed it. A named subject that does not
    resolve to exactly one file, or two named subjects, leave the superseder unknown.
    """
    start, end = clause_head(text, cue_start, (target_span,))
    named = {
        match.group("wiki") or match.group("file")
        for match in _SUBJECT_REF_RE.finditer(text, start, end)
    }
    if not named:
        return holder
    if len(named) > 1:
        return None
    return unique(named.pop())


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
                    confidence=RULE_CONFIDENCE["deterministic.repeated_decision_subject"],
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
        # Sweep claims by interval start. The active polarity maps make disjoint windows cheap,
        # while still visiting every genuinely contradictory pair that must be returned. The old
        # combinations loop paid an overlap check for every pair, including pairs whose windows
        # were already separated.
        ordered = sorted(
            subject_claims,
            key=lambda claim: (_window_bounds(claim)[0], str(claim.metadata.get("file"))),
        )
        active: dict[str, tuple[datetime, EvidenceClaim]] = {}
        active_by_polarity: dict[str, dict[str, EvidenceClaim]] = {
            "positive": {},
            "negative": {},
        }
        expiry: list[tuple[datetime, int, str]] = []
        seen_pairs: set[frozenset[str]] = set()
        sequence = 0
        for claim in ordered:
            claim_start, claim_end = _window_bounds(claim)
            while expiry and expiry[0][0] < claim_start:
                _end, _sequence, evidence_id = heapq.heappop(expiry)
                previous = active.pop(evidence_id, None)
                if previous is None:
                    continue
                active_by_polarity["positive"].pop(evidence_id, None)
                active_by_polarity["negative"].pop(evidence_id, None)

            positive, negative = _validity_polarities(claim.text)
            candidates: dict[str, EvidenceClaim] = {}
            if positive:
                candidates.update(active_by_polarity["negative"])
            if negative:
                candidates.update(active_by_polarity["positive"])
            for other in candidates.values():
                pair = frozenset((claim.evidence_id, other.evidence_id))
                if pair in seen_pairs:
                    continue
                # The active sweep already proves the two windows overlap.
                if not _opposing_validity_text(claim.text, other.text):
                    continue
                seen_pairs.add(pair)
                left_file, right_file = sorted(
                    (str(claim.metadata["file"]), str(other.metadata["file"]))
                )
                proposals.append(
                    _make_proposal(
                        graph=graph,
                        context=context,
                        source_evidence_ids=(claim.evidence_id, other.evidence_id),
                        proposed_relation="contradicts",
                        subject_id=left_file,
                        object_id=right_file,
                        explanation=(
                            f"{left_file} and {right_file} discuss {subject} in overlapping "
                            "validity windows with opposing status language."
                        ),
                        confidence=RULE_CONFIDENCE["deterministic.contradictory_validity_windows"],
                        uncertainty=("textual polarity is heuristic",),
                        status="requires_review",
                        rule_id="deterministic.contradictory_validity_windows",
                    )
                )
            active[claim.evidence_id] = (claim_end, claim)
            if positive:
                active_by_polarity["positive"][claim.evidence_id] = claim
            if negative:
                active_by_polarity["negative"][claim.evidence_id] = claim
            heapq.heappush(expiry, (claim_end, sequence, claim.evidence_id))
            sequence += 1
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
                    confidence=RULE_CONFIDENCE["deterministic.temporal_ordering"],
                    uncertainty=("temporal ordering alone is insufficient for promotion",),
                    status="requires_review",
                    rule_id="deterministic.temporal_ordering",
                )
            )
    return proposals


def _window_bounds(claim: EvidenceClaim) -> tuple[datetime, datetime]:
    return (
        claim.valid_from or datetime.min.replace(tzinfo=timezone.utc),
        claim.valid_until or datetime.max.replace(tzinfo=timezone.utc),
    )


def _validity_polarities(text: str) -> tuple[bool, bool]:
    return bool(_POSITIVE_VALIDITY_RE.search(text)), bool(_NEGATIVE_VALIDITY_RE.search(text))


def _opposing_validity_text(left: str, right: str) -> bool:
    # Compiled once at module level beside the other patterns: this runs inside the pairwise
    # claims loop, where a per call re.compile paid the compile cost on every pair.
    return bool(
        (_POSITIVE_VALIDITY_RE.search(left) and _NEGATIVE_VALIDITY_RE.search(right))
        or (_NEGATIVE_VALIDITY_RE.search(left) and _POSITIVE_VALIDITY_RE.search(right))
    )
