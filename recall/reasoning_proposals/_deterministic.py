"""Deterministic inference proposal rules."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
import heapq
from itertools import groupby
import posixpath
from types import MappingProxyType
from typing import Any

from recall.declared_keys import STATED_KEY_SUCCESSOR_RULE, KeyedMemo, fact_key, successor_pairs
from recall.frontmatter import supersedes_key
from recall.lineage import canonical_sha256
from recall.reasoning_graph import EVIDENCE_TEXT_METADATA_KEY, ReasoningGraphNode, ReasoningGraphProjection
from recall.supersession_cues import sentence_head, unasserted
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
        # Proposal endpoints name files, so a path held by different sources is ambiguous.
        if len({chunk.source for chunk in chunk_nodes}) > 1:
            continue
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
        decided = _decision_subject(text)
        claims[file] = EvidenceClaim(
            id=node.id,
            evidence_id=node.id,
            text=text,
            # Two keys, so the two rules that order claims by date group them differently: the
            # subject is what a decision line names (the bare file name without its version when
            # it names nothing, which a decision line in the same folder can name), the entity is
            # the file name without its version, in its folder. The subject rules group by the
            # subject in its folder (`_subject_key`).
            entity_id=_file_entity(file),
            subject=decided or _versionless_stem(file),
            valid_from=valid_from,
            valid_until=valid_until,
            metadata={"file": file, "source": node.source, "decision_line": bool(decided)},
        )
    return MappingProxyType(claims)


def _earliest_datetime(values: Iterable[Any]) -> datetime | None:
    dates = [value for value in values if isinstance(value, datetime)]
    return min(dates) if dates else None


def _latest_datetime(values: Iterable[Any]) -> datetime | None:
    dates = [value for value in values if isinstance(value, datetime)]
    return max(dates) if dates else None


#: A version is ``v`` or ``V`` and digits after a separator, or as the whole name: ``plan_v2``,
#: ``plan_V2``, ``v2``. Written with no separator it is part of a word, so ``ipv4`` and ``ipv6``
#: are two names, not two versions of ``ip``. Both patterns ignore case, so a name the version
#: rule reads as a version is the one the file entity strips the version from.
_VERSION_RE = re.compile(
    r"^(?P<base>.*?)(?:[_\-. ]|^)v(?P<version>\d+)(?P<suffix>\.[^.]+)?$", re.IGNORECASE
)
_VERSION_SUFFIX_RE = re.compile(r"(?:[_\-. ]|^)v\d+$", re.IGNORECASE)
#: How much of a memo the decision line is looked for in, and how much the direct references are
#: read from: a decision line opens a memo, and a reference past the first pages is rarely the
#: memo's own claim.
_DECISION_SUBJECT_WINDOW = 1000
_DIRECT_REFERENCE_WINDOW = 4000
#: What a decision line names. A status or a verdict names a state, not a subject: "Status: done"
#: in two unrelated memos is not one decision taken twice.
_DECISION_SUBJECT_RE = re.compile(
    r"(?:decision|decided|choice|policy)\s*[:=\-]\s*(?P<subject>[^\n.;]+)",
    re.IGNORECASE,
)
_DIRECT_REF_RE = re.compile(
    r"\b(?P<cue>replaces|supersedes|deprecates|updates|revises)\s+`?(?P<target>[\w./\-[\]]+)`?",
    re.IGNORECASE,
)
#: A document named as the subject before a cue: "new.md replaces old.md", "[[new]] supersedes".
#: Only explicit forms, because a bare word ("the team replaces ...") would resolve against any
#: memo that happens to share its stem.
_SUBJECT_REF_RE = re.compile(
    r"\[\[(?P<wiki>[^\]\n]{1,200})\]\]|`?(?<![\w./\-])(?P<file>[\w./\-]+\.md)\b`?"
)
#: What may stand between a named subject and its cue: "new.md replaces", "`new.md` now
#: replaces", "new.md, which replaces". Anything else ("Per new.md this replaces") means the
#: document is not the subject.
_IMMEDIATE_SUBJECT_GAP = re.compile(
    r"[\s`'\"*_]*(?:,\s*(?:which|that)\s+)?(?:(?:now|also)\s+)?", re.IGNORECASE
)
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
        STATED_KEY_SUCCESSOR_RULE: 0.8,
    }
)


def _in_folder(file: str, name: str) -> str:
    """``name`` keyed in the folder of ``file``: ``notes/`` and ``rollout`` give
    ``notes/rollout``. A name never holds a ``/``, so the key cannot be read two ways."""
    folder = posixpath.dirname(file)
    return f"{folder}/{name}" if folder and name else name


def _version_key(file: str) -> tuple[str, int] | None:
    """The series a versioned file belongs to, and its version: ``notes/plan_v2.md`` is
    ``("notes/plan", 2)``. A series is one folder's files sharing a name without the version, so
    ``a/plan_v1.md`` and ``b/plan_v2.md`` are two series, not two versions of one plan."""
    stem = file.rsplit("/", 1)[-1]
    match = _VERSION_RE.match(stem)
    if not match:
        return None
    base = supersedes_key(f"{match.group('base')}{match.group('suffix') or ''}")
    # A whole-name version such as V1.md has an empty base; its folder still scopes the series.
    return posixpath.join(posixpath.dirname(file), base), int(match.group("version"))


def _decision_subject(text: str) -> str:
    """The subject a decision line in the first `_DECISION_SUBJECT_WINDOW` characters names, or
    "" when none does."""
    match = _DECISION_SUBJECT_RE.search(text[:_DECISION_SUBJECT_WINDOW])
    return supersedes_key(match.group("subject")) if match else ""


def _versionless_stem(file: str) -> str:
    """The file's name without its folder or version: ``notes/rollout_v2.md`` is ``rollout``."""
    return _VERSION_SUFFIX_RE.sub("", supersedes_key(file))


def _file_entity(file: str) -> str:
    """The file's name without its version, in its folder: ``notes/rollout_v2.md`` is
    ``notes/rollout``. Two folders' ``decisions.md`` are two entities, not versions of one."""
    return _in_folder(file, _versionless_stem(file))


def _subject_key(claim: EvidenceClaim) -> tuple[str, str]:
    """What the subject rules group by: the claim's subject in its file's folder.

    Two teams' "decision: retention" are two decisions, and two folders' status-only
    ``decisions.md`` are two memos, so a subject is shared only inside one folder. A tuple, not a
    joined string, because a decision line's subject may itself hold a ``/``.
    """
    return posixpath.dirname(str(claim.metadata["file"])), claim.subject or ""


def _date_successors(
    claims: Iterable[EvidenceClaim],
) -> Iterator[tuple[EvidenceClaim, EvidenceClaim]]:
    """Each dated claim with every claim of the next later start: the pairs a date rule orders.

    Undated claims and claims with the same start are never ordered against each other, since a
    sort position would then name the newer memo. Pairing neighbours in a sorted list skipped
    such a tie and stranded one of the two: with A and B starting together and C later, only B
    to C was proposed. Grouping by start gives both A to C and B to C.
    """
    dated = sorted(
        (claim for claim in claims if claim.valid_from is not None),
        key=lambda claim: (claim.valid_from, str(claim.metadata.get("file"))),
    )
    groups = [list(group) for _day, group in groupby(dated, key=lambda claim: claim.valid_from)]
    for earlier, later in zip(groups, groups[1:], strict=False):
        for older in earlier:
            for newer in later:
                yield older, newer


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
    """Return deterministic proposals, omitting ambiguous file paths, without mutating the graph."""
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
    versions = _explicit_version_proposals(graph, context, claims, authored)
    proposals.extend(_direct_reference_proposals(graph, context, claims, authored))
    repeated = _repeated_decision_subject_proposals(graph, context, claims, authored)
    proposals.extend(_contradictory_validity_window_proposals(graph, context, claims))
    # Both rules read "the later-dated memo supersedes the earlier one about the same thing", so
    # an edge the repeated-subject rule proposed is not proposed again, under a second id, by
    # temporal ordering.
    already = {(proposal.subject_id, proposal.object_id) for proposal in repeated}
    temporal = _temporal_ordering_proposals(graph, context, claims, authored | already)
    versions, dated = _refer_disagreements(graph, context, versions, repeated + temporal)
    proposals.extend(versions)
    proposals.extend(dated)
    # A declared key is the writers' own statement that two memos are about one slot; an edge a
    # rule above already proposed is not proposed again under a second id.
    proposed = {(proposal.subject_id, proposal.object_id) for proposal in proposals}
    proposals.extend(_stated_key_successor_proposals(graph, context, claims, authored | proposed))

    by_id = {proposal.id: proposal for proposal in proposals}
    return tuple(by_id[key] for key in sorted(by_id))


def _stated_key_successor_proposals(
    graph: ReasoningGraphProjection,
    context: ProposalContext,
    claims: Mapping[str, EvidenceClaim],
    skip: set[tuple[str, str]],
) -> list[InferenceProposal]:
    """Rule R(a') (`recall.declared_keys.successor_pairs`) over the facts memos state.

    Each memo's single-valued facts (its chunks' `structured_facts`, from `recall_graph`) start at
    the fact's own `valid_from`, else the memo's. A pair either direction of which is already
    authored, or already proposed by another rule, is skipped. The proposal says when the older
    memo also states facts of other keys, since a memo-level `supersedes:` hides those too.
    """
    stated: dict[str, list[KeyedMemo]] = {}
    for file, nodes in _chunk_nodes_by_file(graph).items():
        claim = claims.get(file)
        if claim is None:
            continue
        seen: set[str] = set()
        for node in nodes:
            for fact in node.structured_facts:
                payload = canonical_sha256(fact.to_payload())
                if payload not in seen:
                    seen.add(payload)
                    stated.setdefault(file, []).append(KeyedMemo(file, fact, fact.valid_from or claim.valid_from))
    proposals: list[InferenceProposal] = []
    for older, newer in successor_pairs(memo for memos in stated.values() for memo in memos):
        if (older.name, newer.name) in skip or (newer.name, older.name) in skip:
            continue
        key = fact_key(older.fact)
        others = {fact_key(memo.fact) for memo in stated[older.name]} - {key}
        uncertainty = ["the key is the writers' declaration, compared exactly as written"]
        if others:
            uncertainty.append(f"{older.name} also states facts of other keys, which a memo-level supersession hides")
        assert older.valid_from is not None and newer.valid_from is not None  # successor_pairs orders dated memos only
        proposals.append(
            _make_proposal(
                graph=graph,
                context=context,
                source_evidence_ids=(claims[older.name].evidence_id, claims[newer.name].evidence_id),
                proposed_relation="supersedes",
                subject_id=older.name,
                object_id=newer.name,
                explanation=(
                    f"{newer.name} states {newer.fact.predicate} of {newer.fact.subject} from "
                    f"{newer.valid_from.date().isoformat()}, after {older.name} stated it from "
                    f"{older.valid_from.date().isoformat()}: the later start supersedes the earlier, "
                    "whatever the value."
                ),
                confidence=RULE_CONFIDENCE[STATED_KEY_SUCCESSOR_RULE],
                uncertainty=tuple(uncertainty),
                status="requires_review",
                rule_id=STATED_KEY_SUCCESSOR_RULE,
            )
        )
    return proposals


#: Said on both of two opposite edges when the version rule and a date rule disagree.
_DISAGREEMENT = "the version numbers and the validity dates disagree on which memo is newer"


def _refer_disagreements(
    graph: ReasoningGraphProjection,
    context: ProposalContext,
    versions: list[InferenceProposal],
    dated: list[InferenceProposal],
) -> tuple[list[InferenceProposal], list[InferenceProposal]]:
    """Send both edges to a reviewer where the version numbers and the dates order two memos
    oppositely (``plan_v2.md`` dated before ``plan_v1.md``).

    Each rule is right by its own signal and one of them is wrong about these two memos, so the
    version proposal is no longer asserted (`requires_review`, a new id, since the status is
    hashed into it) and both proposals name the disagreement. The date rules' proposals already
    require review, so only their uncertainty changes and their ids stay.
    """
    version_pairs = {(proposal.subject_id, proposal.object_id) for proposal in versions}
    dated_pairs = {(proposal.subject_id, proposal.object_id) for proposal in dated}

    def refer(proposal: InferenceProposal) -> InferenceProposal:
        return _make_proposal(
            graph=graph,
            context=context,
            source_evidence_ids=proposal.source_evidence_ids,
            proposed_relation=proposal.proposed_relation,
            subject_id=proposal.subject_id,
            object_id=proposal.object_id,
            explanation=proposal.explanation,
            confidence=proposal.confidence,
            uncertainty=(*proposal.uncertainty, _DISAGREEMENT),
            status="requires_review",
            rule_id=proposal.rule_id,
            metadata=proposal.metadata,
        )

    return (
        [
            refer(proposal) if (proposal.object_id, proposal.subject_id) in dated_pairs else proposal
            for proposal in versions
        ],
        [
            refer(proposal) if (proposal.object_id, proposal.subject_id) in version_pairs else proposal
            for proposal in dated
        ],
    )


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
    for node in _source_nodes(graph):
        assert node.file is not None
        by_key[supersedes_key(node.file)].append(node.file)

    def unique(reference: str) -> str | None:
        files = by_key.get(supersedes_key(reference), [])
        return files[0] if len(files) == 1 and files[0] in claims else None

    proposals: list[InferenceProposal] = []
    for file, claim in sorted(claims.items()):
        text = claim.text[:_DIRECT_REFERENCE_WINDOW]
        for match in _DIRECT_REF_RE.finditer(text):
            matched_target = match.group("target").rstrip(".,;:)")
            target_span = (match.start("target"), match.start("target") + len(matched_target))
            # The O(1) lookup first: most cue words name no memo, and the cue check reads the
            # text around the cue.
            target = unique(matched_target)
            if target is None:
                continue
            # "This draft never supersedes old.md" names the relation without asserting it.
            # Judged on the whole text, not the 4,000-character slice the cues are read from: a
            # sentence that crosses the cut would lose a trailing "unless" or ", should".
            # `unasserted` bounds its own read, and the offsets agree because the slice starts
            # at 0.
            if unasserted(claim.text, match.span("cue"), target_span):
                continue
            superseder = _named_subject(text, match.start("cue"), unique, file)
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
                        + ((_reported_by(file, target),) if reported else ())
                    ),
                    status="requires_review" if reported else "candidate",
                    rule_id="deterministic.direct_textual_reference",
                    metadata={"matched_text": match.group(0).rstrip(".,;:)")},
                )
            )
    return proposals


def _reported_by(holder: str, target: str) -> str:
    """Why a direct reference whose superseder is not its holder needs a reviewer.

    The holder is either a third document or the superseded one naming its successor
    ("[[new]] supersedes old.md" written in old.md); calling the second a third document was
    wrong about who said it.
    """
    if holder == target:
        return "the superseded document names its successor"
    return "the relation is reported by a third document"


def _named_subject(
    text: str,
    cue_start: int,
    unique: Callable[[str], str | None],
    holder: str,
) -> str | None:
    """The document that supersedes in a direct reference, or None when it cannot be told.

    The file holding the sentence, unless the sentence before the cue names a document. "new.md
    replaces old.md" in meeting notes is new.md replacing old.md, and taking the notes as the
    superseder demoted old.md beneath a memo that never claimed it. A named document counts as the
    subject only when it sits right before the cue: in "Per [[meeting]] this replaces old.md" and
    "new.md, approved Monday, replaces old.md" the subject cannot be told from the words, so the
    superseder is unknown rather than guessed either way. Two named documents, or one that does not
    resolve to exactly one file, leave it unknown too.
    """
    head = sentence_head(text, cue_start)
    if head is None:
        return None
    start, end = head
    named = list(_SUBJECT_REF_RE.finditer(text, start, end))
    if not named:
        return holder
    names = {match.group("wiki") or match.group("file") for match in named}
    if len(names) > 1 or not _IMMEDIATE_SUBJECT_GAP.fullmatch(text, named[-1].end(), end):
        return None
    return unique(names.pop())


def _repeated_decision_subject_proposals(
    graph: ReasoningGraphProjection,
    context: ProposalContext,
    claims: Mapping[str, EvidenceClaim],
    authored: set[tuple[str, str]],
) -> list[InferenceProposal]:
    by_subject: dict[tuple[str, str], list[EvidenceClaim]] = defaultdict(list)
    for claim in claims.values():
        if claim.subject:
            by_subject[_subject_key(claim)].append(claim)
    proposals: list[InferenceProposal] = []
    for (_folder, subject), subject_claims in sorted(by_subject.items()):
        # The direction comes from the dates (`_date_successors`).
        for older, newer in _date_successors(subject_claims):
            older_file = str(older.metadata["file"])
            newer_file = str(newer.metadata["file"])
            if (older_file, newer_file) in authored:
                continue
            # With no decision line in either memo the two share a file name in one folder and
            # nothing else, so both are memos of one entity, which temporal ordering orders by
            # date under its own label and confidence (through any memo of that entity dated
            # between them). This rule claims only a subject a decision line names.
            if not (older.metadata["decision_line"] or newer.metadata["decision_line"]):
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
    claims_by_subject: dict[tuple[str, str], list[EvidenceClaim]] = defaultdict(list)
    for claim in claims.values():
        if claim.subject:
            claims_by_subject[_subject_key(claim)].append(claim)
    for (_folder, subject), subject_claims in sorted(claims_by_subject.items()):
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
    skip: set[tuple[str, str]],
) -> list[InferenceProposal]:
    """Later-dated memos of one file entity supersede earlier ones. ``skip`` holds the pairs not
    to propose: authored edges, and the edges the repeated-subject rule already proposes."""
    proposals: list[InferenceProposal] = []
    by_entity: dict[str, list[EvidenceClaim]] = defaultdict(list)
    for claim in claims.values():
        if claim.entity_id:
            by_entity[claim.entity_id].append(claim)
    for entity_id, entity_claims in sorted(by_entity.items()):
        for older, newer in _date_successors(entity_claims):
            older_file = str(older.metadata["file"])
            newer_file = str(newer.metadata["file"])
            if (older_file, newer_file) in skip:
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
