"""Bounded multi-hop evidence planning over trusted retrieval and reasoning graphs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import time
from typing import Literal

from recall.frontmatter import supersedes_key
from recall.reasoning_graph import (
    EVIDENCE_TEXT_METADATA_KEY,
    ReasoningGraphNode,
    ReasoningGraphProjection,
)
from recall.reasoning_proposals import InferenceProposal, ProposedRelation
from recall.trust import is_trusted
from recall.types import TrustedHit, TrustedResult

PLANNER_SCHEMA_VERSION = 1

PlannerOutcome = Literal["completed", "failed_closed"]
PlannerStopReason = Literal[
    "complete",
    "retrieval_abstained",
    "no_trusted_initial_evidence",
    "ambiguous_evidence",
    "unsupported_evidence",
    "budget_exhausted",
]
PlannerOperation = Literal[
    "retrieve_related_claims",
    "follow_authored_relationships",
    "compare_candidate_memories",
    "search_missing_intermediate_evidence",
    "check_temporal_consistency",
    "check_contradiction",
]
EvidenceDecisionKind = Literal["accepted", "rejected"]


@dataclass(frozen=True)
class ReasoningBudget:
    """Hard limits for one planner run.

    `max_steps` counts graph operations, `max_graph_nodes` counts accepted chunk evidence nodes,
    `max_model_calls` is the caller supplied already spent model call count, `max_evidence_tokens`
    is a whitespace token approximation over accepted evidence text, and `max_wall_time_ms` is
    measured with the planner clock.
    """

    max_steps: int = 12
    max_graph_nodes: int = 32
    max_model_calls: int = 0
    max_evidence_tokens: int = 2048
    max_wall_time_ms: int = 1000

    def __post_init__(self) -> None:
        for name, value in (
            ("max_steps", self.max_steps),
            ("max_graph_nodes", self.max_graph_nodes),
            ("max_model_calls", self.max_model_calls),
            ("max_evidence_tokens", self.max_evidence_tokens),
            ("max_wall_time_ms", self.max_wall_time_ms),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class _BoundedNodes:
    nodes: tuple[ReasoningGraphNode, ...]
    overflowed: bool


@dataclass(frozen=True)
class PlannerInitialRetrieval:
    """Trusted retrieval snapshot that seeded the planner trace."""

    query: str
    generation_id: str | None
    trusted_hit_ids: tuple[str, ...]
    rejected_hit_ids: tuple[str, ...]
    abstained: bool
    reason: str


@dataclass(frozen=True)
class EvidenceDecision:
    """Accepted or rejected chunk evidence and the reason for that decision."""

    kind: EvidenceDecisionKind
    node_id: str | None
    chunk_id: str | None
    source: str | None
    file: str | None
    reason: str


@dataclass(frozen=True)
class ExpansionStep:
    """One bounded graph operation.

    `input_node_ids` and `accepted_node_ids` are accepted chunk evidence node ids. `output_node_ids`
    may include bounded candidate ids considered for traceability without making them evidence.
    """

    step_index: int
    operation: PlannerOperation
    input_node_ids: tuple[str, ...]
    output_node_ids: tuple[str, ...]
    accepted_node_ids: tuple[str, ...]
    rejected_node_ids: tuple[str, ...]
    inference_proposal_ids: tuple[str, ...] = ()
    gap_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnresolvedGap:
    """Machine readable reason a path remained unsupported, ambiguous, or over budget."""

    id: str
    kind: str
    node_ids: tuple[str, ...] = ()
    proposal_ids: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class InferenceProposalTrace:
    """Inference proposal usage record. Proposals are exploration hints, never trusted evidence."""

    proposal_id: str
    relation: ProposedRelation
    subject_id: str
    object_id: str
    source_evidence_ids: tuple[str, ...]
    used_for_exploration: bool
    trusted_evidence: bool
    reason: str


@dataclass(frozen=True)
class ReasoningTrace:
    """Complete planner trace: retrieval seed, expansions, decisions, gaps, and proposal usage."""

    initial_retrieval: PlannerInitialRetrieval
    expansion_steps: tuple[ExpansionStep, ...]
    evidence_accepted: tuple[EvidenceDecision, ...]
    evidence_rejected: tuple[EvidenceDecision, ...]
    unresolved_gaps: tuple[UnresolvedGap, ...]
    inference_proposals_used_for_exploration: tuple[InferenceProposalTrace, ...]


@dataclass(frozen=True)
class ReasoningBudgetUsage:
    """Budget counters consumed by a planner run.

    Units match `ReasoningBudget`: steps, accepted graph nodes, model calls, approximate evidence
    tokens, and elapsed wall time in milliseconds.
    """

    steps: int
    graph_nodes: int
    model_calls: int
    evidence_tokens: int
    wall_time_ms: int


@dataclass(frozen=True)
class ReasoningPlan:
    """Planner result. Callers must treat any `failed_closed` outcome as no answer."""

    schema_version: int
    outcome: PlannerOutcome
    stop_reason: PlannerStopReason
    budget: ReasoningBudget
    budget_used: ReasoningBudgetUsage
    trace: ReasoningTrace


@dataclass
class _PlannerState:
    graph: ReasoningGraphProjection
    budget: ReasoningBudget
    started: float
    clock: Callable[[], float]
    accepted: dict[str, ReasoningGraphNode] = field(default_factory=dict)
    rejected: dict[str, EvidenceDecision] = field(default_factory=dict)
    frontier: set[str] = field(default_factory=set)
    steps: list[ExpansionStep] = field(default_factory=list)
    gaps: list[UnresolvedGap] = field(default_factory=list)
    proposal_trace: list[InferenceProposalTrace] = field(default_factory=list)
    model_calls_used: int = 0
    evidence_tokens: int = 0
    failed_closed: bool = False
    stop_reason: PlannerStopReason = "complete"

    @property
    def remaining_node_budget(self) -> int:
        return max(0, self.budget.max_graph_nodes - len(self.accepted))

    @property
    def remaining_token_budget(self) -> int:
        return max(0, self.budget.max_evidence_tokens - self.evidence_tokens)


@dataclass(frozen=True)
class _PlannerIndexes:
    chunks_by_retrieval_identity: Mapping[tuple[str, str, str | None], tuple[ReasoningGraphNode, ...]]
    chunk_nodes_by_file: Mapping[str, tuple[ReasoningGraphNode, ...]]
    chunk_nodes_by_source: Mapping[str, tuple[ReasoningGraphNode, ...]]
    source_node_ids_by_file: Mapping[str, tuple[str, ...]]
    evidence_node_ids: frozenset[str]
    authored_pairs: frozenset[tuple[str, str]]
    authored_edges_by_file: Mapping[str, tuple[int, ...]]
    cycle_files_by_diagnostic_id: Mapping[str, frozenset[str]]
    proposals_by_file: Mapping[str, tuple[InferenceProposal, ...]]
    contradiction_proposals_by_evidence: Mapping[str, tuple[InferenceProposal, ...]]


def plan_multi_hop_evidence(
    retrieval: TrustedResult,
    graph: ReasoningGraphProjection,
    *,
    proposals: Sequence[InferenceProposal] = (),
    budget: ReasoningBudget = ReasoningBudget(),
    model_calls_used: int = 0,
    clock: Callable[[], float] | None = None,
) -> ReasoningPlan:
    """Expand trusted retrieval through explicit graph operations under hard budgets.

    The planner never calls a model. `proposals` may guide comparison and missing edge searches,
    but they are recorded with `trusted_evidence=False` and do not add accepted evidence. Pass
    `model_calls_used` when an upstream proposal provider has already spent model calls that should
    count against this run. `clock` is a test hook. A `failed_closed` result means the caller must
    abstain, ask for clarification, or request review using `stop_reason` and `unresolved_gaps`.
    """
    if model_calls_used < 0:
        raise ValueError("model_calls_used must be non-negative")
    active_clock = clock or time.perf_counter
    state = _PlannerState(
        graph=graph,
        budget=budget,
        started=active_clock(),
        clock=active_clock,
        model_calls_used=model_calls_used,
    )
    indexes = _build_indexes(graph, proposals)
    trusted_hits = tuple(hit for hit in retrieval.hits if is_trusted(hit))
    rejected_hits = tuple(hit for hit in retrieval.hits if not is_trusted(hit))
    initial = PlannerInitialRetrieval(
        query=retrieval.query,
        generation_id=retrieval.generation_id,
        trusted_hit_ids=tuple(hit.chunk.id for hit in trusted_hits),
        rejected_hit_ids=tuple(hit.chunk.id for hit in rejected_hits),
        abstained=retrieval.abstained,
        reason=retrieval.reason,
    )
    binding_error = _binding_error(retrieval, graph)
    if binding_error is not None:
        state.failed_closed = True
        state.stop_reason = "unsupported_evidence"
        state.gaps.append(
            UnresolvedGap(
                id="gap_retrieval_graph_binding_mismatch",
                kind="retrieval_graph_binding_mismatch",
                reason=binding_error,
            )
        )
    elif retrieval.abstained:
        state.failed_closed = True
        state.stop_reason = "retrieval_abstained"
    elif not trusted_hits:
        state.failed_closed = True
        state.stop_reason = "no_trusted_initial_evidence"
    elif not _budget_allows(state):
        state.failed_closed = True
        state.stop_reason = "budget_exhausted"
    else:
        _seed_initial_evidence(state, trusted_hits, indexes)
        for hit in rejected_hits:
            _reject_hit(state, hit, "retrieval verdict is not trusted")
        if not state.failed_closed:
            _run_operations(state, indexes)

    accepted = tuple(
        EvidenceDecision(
            kind="accepted",
            node_id=node.id,
            chunk_id=node.chunk_id,
            source=node.source,
            file=node.file,
            reason="trusted retrieval or explicit graph expansion",
        )
        for node in sorted(state.accepted.values(), key=lambda item: item.id)
    )
    trace = ReasoningTrace(
        initial_retrieval=initial,
        expansion_steps=tuple(state.steps),
        evidence_accepted=accepted,
        evidence_rejected=tuple(sorted(state.rejected.values(), key=lambda item: item.reason)),
        unresolved_gaps=tuple(state.gaps),
        inference_proposals_used_for_exploration=tuple(state.proposal_trace),
    )
    return ReasoningPlan(
        schema_version=PLANNER_SCHEMA_VERSION,
        outcome="failed_closed" if state.failed_closed else "completed",
        stop_reason=state.stop_reason,
        budget=budget,
        budget_used=ReasoningBudgetUsage(
            steps=len(state.steps),
            graph_nodes=len(state.accepted),
            model_calls=model_calls_used,
            evidence_tokens=state.evidence_tokens,
            wall_time_ms=max(0, int((active_clock() - state.started) * 1000)),
        ),
        trace=trace,
    )


def _run_operations(state: _PlannerState, indexes: _PlannerIndexes) -> None:
    operations: tuple[Callable[[_PlannerState], None], ...] = (
        lambda current: _retrieve_related_claims(current, indexes),
        lambda current: _follow_authored_relationships(current, indexes),
        lambda current: _compare_candidate_memories(current, indexes),
        lambda current: _search_missing_intermediate_evidence(current, indexes),
        _check_temporal_consistency,
        lambda current: _check_contradiction(current, indexes),
    )
    for operation in operations:
        if not _budget_allows(state):
            _fail_closed(state, "budget_exhausted", "reasoning budget exhausted")
            return
        operation(state)
        if state.failed_closed:
            return


def _build_indexes(
    graph: ReasoningGraphProjection, proposals: Sequence[InferenceProposal]
) -> _PlannerIndexes:
    mutable_chunks_by_file: dict[str, list[ReasoningGraphNode]] = {}
    mutable_chunks_by_source: dict[str, list[ReasoningGraphNode]] = {}
    mutable_source_node_ids_by_file: dict[str, list[str]] = {}
    for node in graph.nodes:
        if node.kind == "chunk":
            if node.file is not None:
                mutable_chunks_by_file.setdefault(node.file, []).append(node)
            mutable_chunks_by_source.setdefault(node.source, []).append(node)
        elif node.kind == "source" and node.file is not None:
            mutable_source_node_ids_by_file.setdefault(node.file, []).append(node.id)

    authored_edges_by_file: dict[str, list[int]] = {}
    authored_pairs: set[tuple[str, str]] = set()
    for index, edge in enumerate(graph.authored_edges):
        if edge.from_file is not None:
            authored_edges_by_file.setdefault(edge.from_file, []).append(index)
        if edge.to_file is not None:
            authored_edges_by_file.setdefault(edge.to_file, []).append(index)
        if edge.from_file is not None and edge.to_file is not None:
            authored_pairs.add((edge.from_file, edge.to_file))
    cycle_files_by_diagnostic_id: dict[str, frozenset[str]] = {}
    edge_by_id = {edge.id: edge for edge in graph.authored_edges}
    for diagnostic in graph.diagnostics:
        if diagnostic.kind != "cycle":
            continue
        files: set[str] = set()
        for edge_id in diagnostic.edge_ids:
            cycle_edge = edge_by_id.get(edge_id)
            if cycle_edge is None:
                continue
            if cycle_edge.from_file is not None:
                files.add(cycle_edge.from_file)
            if cycle_edge.to_file is not None:
                files.add(cycle_edge.to_file)
        cycle_files_by_diagnostic_id[diagnostic.id] = frozenset(files)

    proposals_by_file: dict[str, list[InferenceProposal]] = {}
    contradiction_by_evidence: dict[str, list[InferenceProposal]] = {}
    for proposal in proposals:
        if proposal.status == "rejected":
            continue
        proposals_by_file.setdefault(proposal.subject_id, []).append(proposal)
        proposals_by_file.setdefault(proposal.object_id, []).append(proposal)
        if proposal.proposed_relation == "contradicts":
            for evidence_id in proposal.source_evidence_ids:
                contradiction_by_evidence.setdefault(evidence_id, []).append(proposal)

    return _PlannerIndexes(
        chunks_by_retrieval_identity={
            key: tuple(nodes)
            for key, nodes in _nodes_by_retrieval_identity(graph).items()
        },
        chunk_nodes_by_file={
            file: tuple(sorted(nodes, key=lambda item: item.id))
            for file, nodes in mutable_chunks_by_file.items()
        },
        chunk_nodes_by_source={
            source: tuple(sorted(nodes, key=lambda item: item.id))
            for source, nodes in mutable_chunks_by_source.items()
        },
        source_node_ids_by_file={
            file: tuple(sorted(node_ids))
            for file, node_ids in mutable_source_node_ids_by_file.items()
        },
        evidence_node_ids=frozenset(node.id for node in graph.nodes),
        authored_pairs=frozenset(authored_pairs),
        authored_edges_by_file={
            file: tuple(indices) for file, indices in authored_edges_by_file.items()
        },
        cycle_files_by_diagnostic_id=cycle_files_by_diagnostic_id,
        proposals_by_file={
            file: tuple(proposals_for_file)
            for file, proposals_for_file in proposals_by_file.items()
        },
        contradiction_proposals_by_evidence={
            evidence_id: tuple(proposals_for_evidence)
            for evidence_id, proposals_for_evidence in contradiction_by_evidence.items()
        },
    )


def _seed_initial_evidence(
    state: _PlannerState, hits: Sequence[TrustedHit], indexes: _PlannerIndexes
) -> None:
    for hit in hits:
        identity = (hit.chunk.id, hit.chunk.source, hit.provenance.file)
        matches = indexes.chunks_by_retrieval_identity.get(identity, ())
        if not matches:
            state.gaps.append(
                UnresolvedGap(
                    id=f"gap_missing_initial_node_{hit.chunk.id}",
                    kind="missing_initial_node",
                    reason=f"trusted hit {hit.chunk.id} is not present in the reasoning graph",
                )
            )
            _fail_closed(state, "unsupported_evidence", "initial evidence is outside graph")
            return
        if len(matches) > 1:
            _fail_closed(state, "ambiguous_evidence", "initial evidence resolves to multiple nodes")
            return
        node = matches[0]
        if _diagnostic_blocks_node(state.graph, node, indexes):
            _reject_node(state, node, "graph diagnostic makes initial evidence ambiguous")
            _fail_closed(state, "ambiguous_evidence", "graph diagnostic blocks initial evidence")
            return
        if not _can_accept_node(state, node):
            _reject_node(state, node, "reasoning budget would be exceeded")
            _fail_closed(state, "budget_exhausted", "initial evidence exceeds reasoning budget")
            return
        _accept_node(state, node)


def _retrieve_related_claims(state: _PlannerState, indexes: _PlannerIndexes) -> None:
    input_ids = tuple(sorted(state.frontier))
    sources = tuple(sorted({node.source for node in state.accepted.values()}))
    related = _bounded_nodes(
        state,
        (
            node
            for source in sources
            for node in indexes.chunk_nodes_by_source.get(source, ())
            if node.id not in state.accepted
        ),
    )
    accepted, rejected = _accept_candidates(state, related.nodes, indexes)
    if related.overflowed and not state.failed_closed:
        _fail_closed(state, "budget_exhausted", "related claim expansion exceeded budget")
    _record_step(
        state,
        "retrieve_related_claims",
        input_ids,
        tuple(node.id for node in related.nodes),
        accepted,
        rejected,
    )


def _follow_authored_relationships(state: _PlannerState, indexes: _PlannerIndexes) -> None:
    input_ids = tuple(sorted(state.frontier))
    edge_indices = tuple(
        sorted(
            {
                edge_index
                for node in state.accepted.values()
                if node.file is not None
                for edge_index in indexes.authored_edges_by_file.get(node.file, ())
            }
        )
    )
    edges = tuple(state.graph.authored_edges[index] for index in edge_indices)
    related = _bounded_nodes(
        state,
        (
            node
            for edge in edges
            for file in (edge.from_file, edge.to_file)
            if file is not None
            for node in indexes.chunk_nodes_by_file.get(file, ())
            if node.id not in state.accepted
        ),
    )
    accepted, rejected = _accept_candidates(state, related.nodes, indexes)
    if related.overflowed and not state.failed_closed:
        _fail_closed(state, "budget_exhausted", "authored relationship expansion exceeded budget")
    _record_step(
        state,
        "follow_authored_relationships",
        input_ids,
        tuple(edge.id for edge in edges),
        accepted,
        rejected,
    )


def _compare_candidate_memories(
    state: _PlannerState, indexes: _PlannerIndexes
) -> None:
    input_ids = tuple(sorted(state.frontier))
    candidate_nodes: list[ReasoningGraphNode] = []
    proposal_ids: list[str] = []
    accepted_files = tuple(
        sorted({node.file for node in state.accepted.values() if node.file is not None})
    )
    adjacent_proposals = _adjacent_proposals(indexes, accepted_files)
    for proposal in adjacent_proposals:
        if not set(proposal.source_evidence_ids) <= indexes.evidence_node_ids:
            _record_proposal(
                state,
                proposal,
                used=False,
                reason="proposal cites evidence outside the graph",
            )
            _fail_closed(state, "unsupported_evidence", "proposal cites evidence outside graph")
            return
        proposal_ids.append(proposal.id)
        _record_proposal(
            state,
            proposal,
            used=True,
            reason="proposal used only to choose candidate memories for comparison",
        )
        for file in (proposal.subject_id, proposal.object_id):
            for node in indexes.chunk_nodes_by_file.get(file, ()):
                if node.id not in state.accepted:
                    candidate_nodes.append(node)
    bounded_candidate_nodes = _bounded_nodes(state, candidate_nodes)
    if bounded_candidate_nodes.overflowed and not state.failed_closed:
        _fail_closed(state, "budget_exhausted", "proposal comparison exceeded budget")
    _record_step(
        state,
        "compare_candidate_memories",
        input_ids,
        tuple(node.id for node in bounded_candidate_nodes.nodes),
        (),
        (),
        proposal_ids=tuple(sorted(set(proposal_ids))),
    )


def _search_missing_intermediate_evidence(
    state: _PlannerState, indexes: _PlannerIndexes
) -> None:
    input_ids = tuple(sorted(state.frontier))
    accepted_files = tuple(
        sorted({node.file for node in state.accepted.values() if node.file is not None})
    )
    gap_ids: list[str] = []
    for proposal in _adjacent_proposals(indexes, accepted_files):
        if proposal.proposed_relation != "supersedes":
            continue
        pair = (proposal.subject_id, proposal.object_id)
        if set(pair) & set(accepted_files) and pair not in indexes.authored_pairs:
            gap = UnresolvedGap(
                id=f"gap_missing_authored_edge_{proposal.id}",
                kind="missing_authored_edge",
                proposal_ids=(proposal.id,),
                reason="candidate supersession has no authored graph edge",
            )
            state.gaps.append(gap)
            gap_ids.append(gap.id)
    _record_step(
        state,
        "search_missing_intermediate_evidence",
        input_ids,
        (),
        (),
        (),
        gap_ids=tuple(gap_ids),
    )


def _check_temporal_consistency(state: _PlannerState) -> None:
    input_ids = tuple(sorted(state.frontier))
    rejected: list[str] = []
    for node in state.accepted.values():
        valid_from = node.validity.get("valid_from")
        valid_until = node.validity.get("valid_until")
        if (
            isinstance(valid_from, datetime)
            and isinstance(valid_until, datetime)
            and valid_from > valid_until
        ):
            rejected.append(node.id)
            _reject_node(state, node, "valid_from is later than valid_until")
    if rejected:
        _fail_closed(state, "unsupported_evidence", "accepted evidence is temporally inconsistent")
    _record_step(
        state,
        "check_temporal_consistency",
        input_ids,
        tuple(sorted(state.accepted)),
        (),
        tuple(sorted(rejected)),
    )


def _check_contradiction(
    state: _PlannerState, indexes: _PlannerIndexes
) -> None:
    input_ids = tuple(sorted(state.frontier))
    contradiction_ids: list[str] = []
    evidence_ids = _accepted_evidence_ids(state, indexes)
    for proposal in _adjacent_contradictions(indexes, evidence_ids):
        if proposal.proposed_relation == "contradicts":
            contradiction_ids.append(proposal.id)
            _record_proposal(
                state,
                proposal,
                used=True,
                reason="contradiction proposal overlaps accepted evidence",
            )
    if contradiction_ids:
        state.gaps.append(
            UnresolvedGap(
                id=f"gap_contradiction_{'_'.join(sorted(contradiction_ids))}",
                kind="contradiction",
                proposal_ids=tuple(sorted(contradiction_ids)),
                reason="accepted evidence has an unresolved contradiction proposal",
            )
        )
        _fail_closed(state, "ambiguous_evidence", "accepted evidence is contradicted")
    _record_step(
        state,
        "check_contradiction",
        input_ids,
        (),
        (),
        (),
        proposal_ids=tuple(sorted(contradiction_ids)),
    )


def _accept_candidates(
    state: _PlannerState, nodes: Sequence[ReasoningGraphNode], indexes: _PlannerIndexes
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    accepted: list[str] = []
    rejected: list[str] = []
    for node in sorted(nodes, key=lambda item: item.id):
        if node.id in state.accepted:
            continue
        if _diagnostic_blocks_node(state.graph, node, indexes):
            _reject_node(state, node, "graph diagnostic makes this evidence ambiguous")
            rejected.append(node.id)
            _fail_closed(state, "ambiguous_evidence", "graph diagnostic blocks expansion")
            break
        if not _can_accept_node(state, node):
            _reject_node(state, node, "reasoning budget would be exceeded")
            rejected.append(node.id)
            _fail_closed(state, "budget_exhausted", "reasoning budget exhausted")
            break
        _accept_node(state, node)
        accepted.append(node.id)
    return tuple(accepted), tuple(rejected)


def _bounded_nodes(
    state: _PlannerState, candidates: Iterable[ReasoningGraphNode]
) -> _BoundedNodes:
    remaining_nodes = state.remaining_node_budget
    remaining_tokens = state.remaining_token_budget
    if remaining_nodes <= 0 or remaining_tokens <= 0:
        for _node in candidates:
            return _BoundedNodes((), True)
        return _BoundedNodes((), False)
    bounded: list[ReasoningGraphNode] = []
    seen: set[str] = set()
    overflowed = False
    for node in candidates:
        if node.id in seen:
            continue
        tokens = _node_tokens(node)
        if tokens > remaining_tokens:
            overflowed = True
            break
        bounded.append(node)
        seen.add(node.id)
        remaining_nodes -= 1
        remaining_tokens -= tokens
        if remaining_nodes <= 0 or remaining_tokens <= 0:
            overflowed = any(candidate.id not in seen for candidate in candidates)
            break
    return _BoundedNodes(tuple(bounded), overflowed)


def _nodes_by_retrieval_identity(
    graph: ReasoningGraphProjection,
) -> dict[tuple[str, str, str | None], list[ReasoningGraphNode]]:
    nodes: dict[tuple[str, str, str | None], list[ReasoningGraphNode]] = {}
    for node in graph.nodes:
        if node.kind == "chunk" and node.chunk_id is not None:
            nodes.setdefault((node.chunk_id, node.source, node.file), []).append(node)
    return nodes


def _accepted_evidence_ids(
    state: _PlannerState, indexes: _PlannerIndexes
) -> tuple[str, ...]:
    evidence_ids = set(state.accepted)
    for node in state.accepted.values():
        if node.file is not None:
            evidence_ids.update(indexes.source_node_ids_by_file.get(node.file, ()))
    return tuple(sorted(evidence_ids))


def _binding_error(
    retrieval: TrustedResult, graph: ReasoningGraphProjection
) -> str | None:
    checks = (
        ("tenant_id", retrieval.tenant_id, graph.tenant_id),
        ("generation_id", retrieval.generation_id, graph.generation_id),
        ("pipeline_fingerprint", retrieval.pipeline_fingerprint, graph.pipeline_fingerprint),
        ("corpus_fingerprint", retrieval.corpus_fingerprint, graph.corpus_fingerprint),
    )
    for name, retrieval_value, graph_value in checks:
        if retrieval_value is not None and graph_value is not None and retrieval_value != graph_value:
            return f"retrieval {name} does not match reasoning graph {name}"
    return None


def _adjacent_proposals(
    indexes: _PlannerIndexes, files: Sequence[str]
) -> tuple[InferenceProposal, ...]:
    proposals: dict[str, InferenceProposal] = {}
    for file in files:
        for proposal in indexes.proposals_by_file.get(file, ()):
            proposals.setdefault(proposal.id, proposal)
    return tuple(proposals.values())


def _adjacent_contradictions(
    indexes: _PlannerIndexes, evidence_ids: Sequence[str]
) -> tuple[InferenceProposal, ...]:
    proposals: dict[str, InferenceProposal] = {}
    for evidence_id in evidence_ids:
        for proposal in indexes.contradiction_proposals_by_evidence.get(evidence_id, ()):
            proposals.setdefault(proposal.id, proposal)
    return tuple(proposals.values())


def _can_accept_node(state: _PlannerState, node: ReasoningGraphNode) -> bool:
    return (
        len(state.accepted) + 1 <= state.budget.max_graph_nodes
        and state.evidence_tokens + _node_tokens(node) <= state.budget.max_evidence_tokens
    )


def _accept_node(state: _PlannerState, node: ReasoningGraphNode) -> None:
    state.accepted[node.id] = node
    state.frontier.add(node.id)
    state.evidence_tokens += _node_tokens(node)


def _reject_node(state: _PlannerState, node: ReasoningGraphNode, reason: str) -> None:
    state.rejected[node.id] = EvidenceDecision(
        kind="rejected",
        node_id=node.id,
        chunk_id=node.chunk_id,
        source=node.source,
        file=node.file,
        reason=reason,
    )


def _reject_hit(state: _PlannerState, hit: TrustedHit, reason: str) -> None:
    state.rejected[hit.chunk.id] = EvidenceDecision(
        kind="rejected",
        node_id=None,
        chunk_id=hit.chunk.id,
        source=hit.chunk.source,
        file=hit.provenance.file,
        reason=reason,
    )


def _record_step(
    state: _PlannerState,
    operation: PlannerOperation,
    input_ids: tuple[str, ...],
    output_ids: tuple[str, ...],
    accepted: tuple[str, ...],
    rejected: tuple[str, ...],
    *,
    proposal_ids: tuple[str, ...] = (),
    gap_ids: tuple[str, ...] = (),
) -> None:
    state.steps.append(
        ExpansionStep(
            step_index=len(state.steps) + 1,
            operation=operation,
            input_node_ids=input_ids,
            output_node_ids=tuple(sorted(set(output_ids))),
            accepted_node_ids=tuple(sorted(set(accepted))),
            rejected_node_ids=tuple(sorted(set(rejected))),
            inference_proposal_ids=proposal_ids,
            gap_ids=gap_ids,
        )
    )


def _record_proposal(
    state: _PlannerState, proposal: InferenceProposal, *, used: bool, reason: str
) -> None:
    if any(item.proposal_id == proposal.id for item in state.proposal_trace):
        return
    state.proposal_trace.append(
        InferenceProposalTrace(
            proposal_id=proposal.id,
            relation=proposal.proposed_relation,
            subject_id=proposal.subject_id,
            object_id=proposal.object_id,
            source_evidence_ids=proposal.source_evidence_ids,
            used_for_exploration=used,
            trusted_evidence=False,
            reason=reason,
        )
    )


def _diagnostic_blocks_node(
    graph: ReasoningGraphProjection, node: ReasoningGraphNode, indexes: _PlannerIndexes
) -> bool:
    blocking = {
        "unresolved_reference",
        "ambiguous_reference",
        "cycle",
        "duplicate_entity_candidate",
        "malformed_metadata",
    }
    for diagnostic in graph.diagnostics:
        if diagnostic.kind not in blocking:
            continue
        if node.id in diagnostic.node_ids:
            return True
        if node.file is not None and diagnostic.reference == supersedes_key(node.file):
            return True
        if node.file is not None and diagnostic.reference == node.file:
            return True
        if (
            diagnostic.kind == "cycle"
            and node.file is not None
            and node.file in indexes.cycle_files_by_diagnostic_id.get(diagnostic.id, frozenset())
        ):
            return True
    return False


def _budget_allows(state: _PlannerState) -> bool:
    return (
        len(state.steps) < state.budget.max_steps
        and len(state.accepted) <= state.budget.max_graph_nodes
        and state.model_calls_used <= state.budget.max_model_calls
        and state.evidence_tokens <= state.budget.max_evidence_tokens
        and (state.clock() - state.started) * 1000 <= state.budget.max_wall_time_ms
    )


def _fail_closed(state: _PlannerState, reason: PlannerStopReason, detail: str) -> None:
    state.failed_closed = True
    state.stop_reason = reason
    state.gaps.append(UnresolvedGap(id=f"gap_fail_closed_{reason}", kind=reason, reason=detail))


def _node_tokens(node: ReasoningGraphNode) -> int:
    value = node.metadata.get(EVIDENCE_TEXT_METADATA_KEY)
    if isinstance(value, str):
        return max(1, len(value.split()))
    return 1
