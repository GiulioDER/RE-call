from __future__ import annotations

# This module is a compatibility facade. The imports below intentionally preserve the public
# service-owned names and monkeypatch seams while implementations move to focused boundaries.
# ruff: noqa: F401

import copy
import hashlib
import json
import os
import random
from contextlib import AbstractContextManager, nullcontext, suppress
import mimetypes
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import psycopg
from pydantic import BaseModel, Field
from recall._env import truthy

from recall_mcp.models import (
    EvidenceCardModel,
    EvidenceItemModel,
    EvidenceResult,
    ForgetResult,
    CurrentStateRecordModel,  # noqa: F401  # legacy public import
    CurrentStateResult,  # noqa: F401  # legacy public import
    InventoryEntry,  # noqa: F401  # legacy public import
    InventoryResult,  # noqa: F401  # legacy public import
    IndexResult,  # noqa: F401  # legacy public import
    MemoryStatsResult,  # noqa: F401  # legacy public import
    ReasoningAuditResult,  # noqa: F401  # legacy public import
    ReasoningProjectionResult,
    ReasoningProposalItem,
    ReasoningProposalResult,
    RewritePlanResult,
    SearchHit,
    SearchResult,
)

from recall.calibration import Calibration
from recall.calibration_v2 import CalibrationRepository
from recall.answer_provider import OllamaAnswerProvider
from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    AtomicRescueLineageError,
    AtomicRescueSelectionError,
    atomic_rescue_expectation_parity,
    atomic_rescue_reference_parity,
    load_atomic_rescue_artifact,
    select_atomic_rescue,
)
from recall.trust_policy import TrustPolicy, TrustRefusal
from recall.embeddings import (
    Embedder,
    HashingEmbedder,
    REMOTE_MODEL_CODE_OPT_IN,
    embedder_artifact_digest,
    embed_query,
    embedding_profile_id,
    resolve_registered_embedder,
    resolve_embedder,
)
from recall.errors import RecallError
from recall.control_plane import ControlPlane
from recall.frontmatter import supersedes_key
from recall.index import Chunker, candidate_files, chunk_text  # noqa: F401  # legacy public import
from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import ExtractingLocalObjectReader
from recall.generations import (
    GenerationManager,
    InvalidGenerationTransition,
    NoActiveGeneration,
    UnsafePromotion,
)
from recall.observability import (
    METRICS,
    PerformanceTrace,
    current_performance_trace,
    get_logger,
    performance_trace_scope,
)
from recall.security_policy import AccessContext, SourceSecurityPolicy
from recall.source_conditioning import (
    SOURCE_CONDITIONING_SHADOW_POLICIES,
    SourceConditioningArtifactError,
    chunk_identifier_hash,
    fill_source_conditioned_spare_slots,
    load_source_conditioning_artifact,
    select_source_conditioned,
)
from recall.profiles import (
    FAST_PROFILE,
    QUALITY_PROFILE,
    RetrievalAdmission,
    RetrievalOverloaded,
    RetrievalProfile,
    resolve_retrieval_profile,
)
from recall.evidence import (
    EvidenceBundle,
    EvidenceItem,
    EvidencePolicy,
    build_evidence_bundle,
    render_evidence_prompt,
)
from recall.explanations import RetrievalExplanation
from recall.graph_first import (
    GraphFirstCandidate,
    GraphFirstMode,
    MAX_GRAPH_FIRST_CANDIDATES,
    build_graph_first_candidates,
)
from recall.query_class import route_query, routing_mode
from recall.query_construction import (
    MAX_QUERY_CANDIDATES,
    MAX_QUERY_CHARS as MAX_QUERY_CONSTRUCTION_QUERY_CHARS,
    MAX_QUERY_CONSTRUCTION_ROUNDS,
    QueryConstructionArm,
    QueryConstructionRequest,
    QueryFrame,
    QueryProposal,
    QueryValidation,
    RetrievalSignal,
    build_control_proposals,
    build_original_model_challenge,
    parse_query_frame,
    should_request_original_model_refinement,
    validate_query_proposals,
)
from recall.retriever import RetrievalCandidateTrace
from recall.related import RelatedEvidenceResult, trusted_related
from recall.reasoning import (
    GenerationSelection,
    REASONING_API_VERSION,
    ReasoningDiagnostics,
    ReasoningPolicy,
    ReasoningProviderPorts,
    ReasoningRequest,
    ReasoningResponse,
    SemanticGraphExpansionResult,
    reason,
)
from recall.reasoning_expansion import (
    ExpansionProposal,
    ReasoningExpansionRetriever,
    merge_trusted_results,
    resolve_expansion_provider,
)
from recall.reasoning_graph import (
    ReasoningGraphProjection,
    build_reasoning_graph,
    project_store_graph,  # noqa: F401  # legacy patch seam
)
from recall.reasoning_planner import ReasoningBudget, _reset_planner_index_cache
from recall.semantic_graph import (
    RELATION_KINDS,
    SemanticGraphProjection,
    relation_coverage,
)
from recall.query_entity_resolution import resolve_query_entities
from recall.reasoning_proposals import (
    InferenceProposal,
    ProposalProtocolReport,
    deterministic_inference_proposals,
)
from recall.rerank import (
    COREB_CODE_RERANKER_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_REVISION,
    KNOWN_RERANKER_REVISIONS,
    RERANKER_MODEL_ALIASES,
    Reranker,
)
from recall.retriever import DocumentExpansionPolicy, StructuralExpansionPolicy
from recall.store import PgVectorStore
from recall.timing import TimedEmbedder
from recall.entailment import EntailmentJudge
from recall.trust import decision_state_for, evaluate, is_trusted, resolve_successor, trusted_search
from recall.types import (
    AtomicFact,
    EvidenceCard,
    Provenance,
    RetrievalResult,
    ScoredChunk,
    TrustedHit,
    TrustedResult,
    Validity,
)
from recall_mcp import factories as _factories
from recall_mcp import reasoning_api as _reasoning_api
from recall_mcp import retrieval as _retrieval
from recall_mcp import graph_first_api as _graph_first_api
from recall_mcp import query_construction_api as _query_construction_api
from recall_mcp.compat import serving_json  # noqa: F401  # legacy public import
from recall_mcp import graph_expansion as _graph_expansion
from recall_mcp import graph_projection as _graph_projection
from recall_mcp.settings import runtime_environment
from recall_mcp.status import (
    JobLedger,  # noqa: F401  # legacy public import
    calibration_status,  # noqa: F401  # legacy public import
    job_status,  # noqa: F401  # legacy public import
)
from recall_mcp.lifecycle import (
    MAX_FORGET_SOURCES,  # noqa: F401  # legacy public import
    current_state_memory,  # noqa: F401  # legacy public import
    forget_memory,  # noqa: F401  # legacy public import
    memory_inventory,  # noqa: F401  # legacy public import
    memory_stats,  # noqa: F401  # legacy public import
)
from recall_mcp.indexing import (
    DEFAULT_MAX_INDEX_BYTES,  # noqa: F401  # legacy public import
    DEFAULT_MAX_INDEX_FILES,  # noqa: F401  # legacy public import
    REDACTED_PATH,  # noqa: F401  # legacy public import
    IndexPreflightError,  # noqa: F401  # legacy public import
    _scrub_paths,  # noqa: F401  # legacy public import
    _set_candidate_files_provider,
    index_memory,  # noqa: F401  # legacy public import
)
from recall_mcp.provenance import (
    FACT_WRITE_DSN_ENV,  # noqa: F401  # legacy public import
    _fact_write_dsn,  # noqa: F401  # legacy public import
    apply_fact_memory,  # noqa: F401  # legacy public import
    current_facts_memory,  # noqa: F401  # legacy public import
    register_evidence_cards,  # noqa: F401  # legacy public import
)
from recall_mcp.graph_projection import (
    _authorization_scope,
    _authorized_graph,  # noqa: F401  # legacy public import
    _combined_graph_policy_fingerprint,  # noqa: F401  # legacy public import
    _store_graph,  # noqa: F401  # legacy public import
    _store_graph_with_readiness,  # noqa: F401  # legacy public import
    reasoning_projection,  # noqa: F401  # legacy public import
)
from recall_mcp.reasoning_common import (
    _query_construction_anchors,
    _query_construction_evidence,
    _query_construction_generation,
    _query_construction_hit,
    _query_construction_retrieval,
    _reasoning_generation,
    _same_generation,
)
from recall_mcp.reasoning_admin import (
    _stored_extracted_proposals,  # noqa: F401  # legacy public import
    apply_command_for,  # noqa: F401  # legacy public import
    reasoning_proposals,  # noqa: F401  # legacy public import
    rewrite_plan,  # noqa: F401  # legacy public import
)
from recall_mcp.retrieval import (
    MAX_QUERY_CHARS,  # noqa: F401  # legacy public import
    MAX_SEARCH_K,  # noqa: F401  # legacy public import
    _Retrieval,  # noqa: F401  # legacy public import
    startup_retrieval_profile,  # noqa: F401  # legacy public import
)

# Compatibility aliases for diagnostics and tests that inspected the former service-owned cache.
_GRAPH_PROJECTIONS = _graph_projection._GRAPH_PROJECTIONS
_GRAPH_PROJECTION_INFLIGHT = _graph_projection._GRAPH_PROJECTION_INFLIGHT
_GRAPH_PROJECTION_CACHE_MAX = _graph_projection._GRAPH_PROJECTION_CACHE_MAX

_log = get_logger("mcp.service")

HASHING_DIM = 64  # offline HashingEmbedder width; matches the eval/test default
# Query construction is a two-phase, client-callable protocol. Keep its prompt and graph budgets
# below the broader search limits because every continuation can trigger bounded retrieval work.
MAX_QUERY_CONSTRUCTION_PROMPT_CHARS = _query_construction_api.MAX_QUERY_CONSTRUCTION_PROMPT_CHARS
MAX_QUERY_CONSTRUCTION_GRAPH_NODES = _query_construction_api.MAX_QUERY_CONSTRUCTION_GRAPH_NODES
# Cosine reranking may inspect a bounded oversample of structural candidates so a lower-confidence
# relation can still win on query relevance without turning graph expansion into an unbounded query.
MAX_GRAPH_RESCORING_CANDIDATES = 512

# Graph first intentionally follows the successful LoCoMo context shape. The retrieval pool is
# wider than the final context, eight direct items are protected, and graph candidates can fill
# the remaining two slots after all bounded candidates have been rescored.
GRAPH_FIRST_RETRIEVAL_K = 20
GRAPH_FIRST_SEED_K = 8
GRAPH_FIRST_CONTEXT_K = 10

GRAPH_PRECISION_POLICY_VERSION = "semantic_graph_precision_v2"
GRAPH_DIRECTIONAL_RELATIONS = frozenset(
    {"supports", "references", "depends_on", "caused", "supersedes"}
)
GRAPH_DIAGNOSTIC_ONLY_RELATIONS = frozenset({"contradicts", "same_entity"})
GRAPH_HUB_DEGREE_THRESHOLD = 32
# Kept as a compatibility setting for old diagnostic runners. It no longer rejects candidates.
GRAPH_COSINE_MARGIN = 0.10
GRAPH_RERANK_WEIGHTS = (0.60, 0.20, 0.10, 0.10)
GRAPH_RERANK_CORROBORATION_CAP = 2
GRAPH_FILL_POLICY = "direct_first_fill_missing"
GRAPH_FILL_SLOT_COUNT = 5
GRAPH_TAIL_REPLACEMENT_MARGIN = 0.05
GRAPH_TAIL_REPLACEMENT_MARGINS = frozenset({0.05, 0.10, 0.15, 0.20})
GRAPH_PRECISION_VARIANTS = frozenset(
    {
        "baseline",
        "directional",
        "corroboration",
        "hub",
        "cosine",
        "selective",
        "combined",
        "combined_no_selective",
    }
)
GRAPH_RELATION_CONTROLS = frozenset({"none", "shuffled", "removed"})

_set_candidate_files_provider(lambda: candidate_files)


def make_embedder(name: str, env: dict[str, str] | None = None) -> Embedder:
    """Return the embedder backend by name.

    Registered local profiles and legacy resolver spellings both pass through
    `recall.embeddings.resolve_embedder`, so profile identity and context selection are shared with
    the CLI. Without `RECALL_EMBED_PROFILE`, the MCP server accepts the explicit cloud and research
    model aliases as before.
    """
    values = dict(runtime_environment()) if env is None else env
    profile_id = values.get("RECALL_EMBED_PROFILE", "").strip()
    if profile_id:
        from recall.embedding_registry import registered_profile, registered_profile_ids

        try:
            entry = registered_profile(profile_id)
        except ValueError:
            raise IndexPreflightError(
                f"unknown RECALL_EMBED_PROFILE: {profile_id!r} "
                f"(registered: {', '.join(registered_profile_ids())})"
            ) from None
        expected = {
            "fastembed": "fastembed",
            "qwen3": "fastembed",
            "voyage": "voyage",
            "voyage-context": "voyage-context",
            "voyage-multimodal": "voyage-multimodal",
            "openai-compat": "openrouter",
        }[entry.backend]
        accepted = {"openai", "openrouter"} if entry.backend == "openai-compat" else {expected}
        if name not in accepted:
            raise ValueError(
                f"RECALL_EMBED_PROFILE={profile_id!r} needs RECALL_EMBEDDER={expected}"
            )
        if entry.hosted:
            if entry.backend == "voyage-multimodal" and not truthy(
                values.get("RECALL_MULTIMODAL_ENABLED", "0")
            ):
                raise ValueError(
                    "voyage-multimodal is disabled; set RECALL_MULTIMODAL_ENABLED=1 to opt in"
                )
            return entry.build(api_key=values.get(entry.api_key_env) or None, env=values)
        artifact_path = values.get(entry.artifact_path_env, "")
        artifact_digest = values.get("RECALL_MODEL_SHA256", "")
        if not artifact_path or not artifact_digest:
            raise ValueError(
                f"profile {profile_id!r} requires {entry.artifact_path_env} and RECALL_MODEL_SHA256"
            )
        return entry.build(artifact_path=artifact_path, artifact_digest=artifact_digest, env=values)
    if name == "hashing":
        return HashingEmbedder(dim=HASHING_DIM)
    try:
        return resolve_embedder(name, env=values)
    except ValueError as exc:
        if "unknown embedder" not in str(exc):
            raise
        raise IndexPreflightError(
            f"unknown embedder: {name!r} (use 'fastembed', 'hashing', or any "
            "recall.embeddings resolver spelling)"
        ) from exc


def make_profile_embedder(
    profile_id: str, *, shadow: bool = False, env: dict[str, str] | None = None
) -> Embedder:
    """Construct one registered profile, with optional shadow-specific artifact settings."""
    values = dict(runtime_environment() if env is None else env)
    from recall.embedding_registry import registered_profile

    entry = registered_profile(profile_id)
    if entry.backend == "voyage-multimodal" and not truthy(
        values.get("RECALL_MULTIMODAL_ENABLED", "0")
    ):
        raise ValueError(
            "voyage-multimodal is disabled; set RECALL_MULTIMODAL_ENABLED=1 to opt in"
        )
    return resolve_registered_embedder(profile_id, values, shadow=shadow)


class RelatedResult(BaseModel):
    """Related evidence whose candidates each passed an independent trust evaluation."""

    seed_chunk_id: str = Field(description="Chunk that seeded the structural relation.")
    relation: str = Field(description="source | ordinal | supersession.")
    generation_id: str = Field(description="Generation identity shared by seed and items.")
    items: list[EvidenceItemModel] = Field(description="Trusted related evidence items.")
    rejected_count: int = Field(description="Candidates rejected by independent trust checks.")
    explanation: dict[str, object] | None = Field(
        default=None, description="Optional structured explanation when explain=true."
    )


#: Cross-encoder reranking, opt-in via `RECALL_RERANK`.
#:
#: Measured on LOCOMO at n=1,536 (FINDINGS §11): hit@5 **0.671 -> 0.777**, intervals disjoint from
#: the baseline through k=10 — the largest single retrieval gain in this project, and roughly twice
#: the best embedder effect. It closes 57% of the distance to the candidate pool's own ceiling.
#:
#: OFF by default because it costs ~1,050 ms per query on CPU. A memory server that silently
#: quadrupled every query's latency to improve a benchmark would be choosing for the operator.
#: Worth enabling when a human is waiting on the answer; leave it off for high-volume automated
#: retrieval or constrained hardware.
_RERANK_TRUE = frozenset({"1", "true", "yes", "on"})
_RERANK_FALSE = frozenset({"", "0", "false", "no", "off"})


def resolve_reranker(env: dict[str, str] | None = None) -> tuple[str, str | None] | None:
    """`(model, revision)` for the configured reranker, or None when it is off.

    `revision` is None for a cloud model, which has no Hub reference to pin. That is a real
    difference in guarantee, not a missing value, and the type says so rather than hiding it
    behind an empty string.

    Returns a spec rather than an instance so the decision can be tested without importing torch.

    `ms-marco-MiniLM-L-6-v2` is the default because it was *measured* to be the right choice, not
    because it was already there: `bge-reranker-base`, with 12x the parameters and four years newer,
    is statistically indistinguishable at **6.3x** the per-query cost. Reranker selection here is
    about task match — short query against short passage — not model size.

    An unparseable flag is REFUSED rather than read as "off". An operator who asked for reranking
    and silently got an unreranked server would have no way to notice: the failure is fast, quiet
    and looks exactly like success.
    """
    source = env if env is not None else runtime_environment()
    raw = source.get("RECALL_RERANK", "").strip().lower()
    if raw in _RERANK_FALSE:
        return None
    if raw not in _RERANK_TRUE:
        raise ValueError(
            f"RECALL_RERANK={raw!r} is not a boolean. Use one of {sorted(_RERANK_TRUE)} to enable "
            f"or leave it unset. Refused rather than treated as off, because a server that quietly "
            f"ignored the flag would look identical to one that honoured it."
        )

    model = source.get("RECALL_RERANK_MODEL")
    if not model:
        return (DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_REVISION)
    model = RERANKER_MODEL_ALIASES.get(model, model)
    revision = source.get("RECALL_RERANK_REVISION")

    # A cloud model has no Hub reference, so it has no revision to pin. The requirement below is a
    # Hub property, and applying it here would make the Voyage reranker unselectable while looking
    # like a safety check.
    #
    # ⚠️ The guarantee genuinely differs and is not being papered over. `rerank-2.5` is a name
    # resolved on Voyage's side: its weights can change under us in a way a pinned Hub revision
    # cannot. That is a real, smaller guarantee, recorded here so a reader comparing the two
    # rerankers can see it. It is a reason to know what you are choosing, not a reason to refuse.
    if model == "voyage" or model.startswith("voyage:"):
        if revision:
            raise ValueError(
                f"RECALL_RERANK_MODEL={model!r} has no Hub revision to pin, so "
                f"RECALL_RERANK_REVISION={revision!r} cannot be honoured. Accepting it would put a "
                "pin in every trace that pins nothing, which asserts a guarantee that does not "
                "exist. Unset RECALL_RERANK_REVISION for a cloud reranker."
            )
        return (model, None)

    if not revision:
        revision = KNOWN_RERANKER_REVISIONS.get(model)
        if not revision:
            raise ValueError(
                "RECALL_RERANK_MODEL requires RECALL_RERANK_REVISION unless it names a built-in "
                "pinned reranker. An unpinned Hub reference is mutable, and the shipped revision "
                "pin belongs to the shipped weights only — reusing it would name the wrong "
                "artifact in every trace."
            )
    return (model, revision)


def _positive_env(values: dict[str, str], name: str, default: int) -> int:
    raw = values.get(name, "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _remote_model_code_enabled(values: dict[str, str]) -> bool:
    return values.get(REMOTE_MODEL_CODE_OPT_IN, "").strip().lower() in {"1", "true", "yes", "on"}


def _require_remote_model_code_enabled(values: dict[str, str], model: str) -> None:
    if not _remote_model_code_enabled(values):
        raise ValueError(
            f"{model} requires {REMOTE_MODEL_CODE_OPT_IN}=1 because its Transformers loader "
            "executes model repository code"
        )


def _validate_quality_reranker_config(values: dict[str, str]) -> tuple[str, str]:
    """`(model_path, digest)` for the quality profile, or raise. No model is loaded.

    The digest must EQUAL the pin. Accepting whatever the operator typed would make the
    `local_files_only` verification self-referential: it would prove the tree matches the hash of
    itself, which every tree does. The check that means something compares it against a value
    chosen elsewhere.
    """
    from recall.rerank import PINNED_RERANKER_SHA256

    # Normalise ONCE, compare the normalised value, and return the normalised value. Validating
    # one string and handing the loader a different one re-opens the hole this check closed:
    # `verify_artifact` rejects on LENGTH before it lowercases, so a digest that is correct
    # except for a trailing newline (what a dotenv literal block or a padded `.env` line
    # produces) passed startup and then raised on the first search.
    model_path = values.get("RECALL_RERANK_PATH", "").strip()
    digest = values.get("RECALL_RERANK_SHA256", "").strip().lower()
    if not model_path or not digest:
        raise ValueError("quality profile requires RECALL_RERANK_PATH and RECALL_RERANK_SHA256")
    if digest != PINNED_RERANKER_SHA256:
        raise ValueError(
            f"RECALL_RERANK_SHA256 does not match the reranker pinned to the quality profile "
            f"(expected {PINNED_RERANKER_SHA256}). A different artifact tree is a different "
            f"model and needs its own registered experiment, not a reused profile."
        )
    return model_path, digest


def _new_reranker(
    env: dict[str, str] | None = None,
    profile: RetrievalProfile | None = None,
) -> "Reranker | None":  # pragma: no cover
    """Instantiate the configured reranker, or None. Imports torch only when actually enabled."""
    if profile is None:
        return _factories._new_reranker(env)
    return _factories._new_reranker(env, profile=profile)


def _reset_reranker_cache() -> None:
    """Drop the per-process reranker. For tests, a server should never need this."""
    _factories._reset_reranker_cache()


def _build_reranker(
    profile: RetrievalProfile | None = None, env: dict[str, str] | None = None
) -> "Reranker | None":
    """Compatibility wrapper for the single reranker owner in ``recall_mcp.factories``."""
    return _factories._build_reranker(profile=profile, env=env, builder=_new_reranker)


def _admission(profile: RetrievalProfile) -> RetrievalAdmission:
    """Compatibility wrapper for the single admission owner in ``recall_mcp.factories``."""
    return _factories._admission(profile)


def _retrieve_trusted(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None,
    k: int,
    calibration: Calibration | None,
    policy: TrustPolicy | None,
    entailment: EntailmentJudge | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
    env: Mapping[str, str] | None = None,
    pool_k: int | None = None,
    pre_trust_transform: Callable[[RetrievalResult], RetrievalResult] | None = None,
    query_vector_callback: Callable[[list[float]], None] | None = None,
    capture_candidate_trace: bool = False,
) -> _Retrieval:
    """Compatibility adapter for the retrieval execution owner."""
    return _retrieval._retrieve_trusted(
        store,
        embedder,
        query,
        source,
        k,
        calibration,
        policy,
        entailment,
        security_policy,
        access_context,
        env,
        pool_k,
        pre_trust_transform,
        query_vector_callback,
        capture_candidate_trace,
        reranker_builder=_build_reranker,
        admission_factory=_admission,
        trusted_search_fn=trusted_search,
    )


# Compatibility aliases for response assembly helpers now owned by retrieval.
_cost_surface = _retrieval._cost_surface
_search_hit_model = _retrieval._search_hit_model
_trusted_evidence_item_model = _retrieval._trusted_evidence_item_model
_evidence_item_model = _retrieval._evidence_item_model
_evidence_card_model = _retrieval._evidence_card_model


def search_memory(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None = None,
    k: int = 5,
    calibration: Calibration | None = None,
    policy: TrustPolicy | None = None,
    explain: bool = False,
    include_related: bool = False,
    related_relation: str = "source",
    related_max_items: int = 3,
    reasoning_available: bool = False,
    entailment: EntailmentJudge | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
    env: Mapping[str, str] | None = None,
) -> SearchResult:
    """Compatibility wrapper for search response assembly owned by retrieval."""
    return _retrieval.search_memory(
        store,
        embedder,
        query,
        source,
        k,
        calibration,
        policy,
        explain,
        include_related,
        related_relation,
        related_max_items,
        reasoning_available,
        entailment,
        security_policy,
        access_context,
        env,
        _retrieve_trusted_fn=_retrieve_trusted,
        _search_hit_model_fn=_search_hit_model,
        _search_advice_fn=_search_advice,
        _cost_surface_fn=_cost_surface,
        trusted_related_fn=trusted_related,
    )

# Compatibility aliases for advice assembly now owned by retrieval.
UNCALIBRATED_NOTE = _retrieval.UNCALIBRATED_NOTE
STALE_INDEX_NOTE = _retrieval.STALE_INDEX_NOTE
REASONING_BLOCKED_NOTE = _retrieval.REASONING_BLOCKED_NOTE
REASONING_SUPERSEDED_NOTE = _retrieval.REASONING_SUPERSEDED_NOTE
_search_advice = _retrieval._search_advice
_advice_suffixes = _retrieval._advice_suffixes
_evidence_advice = _retrieval._evidence_advice


def evidence_memory(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None = None,
    k: int = 5,
    max_items: int | None = None,
    calibration: Calibration | None = None,
    policy: TrustPolicy | None = None,
    explain: bool = False,
    include_related: bool = False,
    related_relation: str = "source",
    related_max_items: int = 3,
    entailment: EntailmentJudge | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
    env: Mapping[str, str] | None = None,
) -> EvidenceResult:
    """Compatibility wrapper for evidence assembly owned by retrieval."""
    return _retrieval.evidence_memory(
        store,
        embedder,
        query,
        source,
        k,
        max_items,
        calibration,
        policy,
        explain,
        include_related,
        related_relation,
        related_max_items,
        entailment,
        security_policy,
        access_context,
        env,
        _retrieve_trusted_fn=_retrieve_trusted,
        _evidence_item_model_fn=_evidence_item_model,
        _evidence_advice_fn=_evidence_advice,
        _cost_surface_fn=_cost_surface,
        _trusted_evidence_item_model_fn=_trusted_evidence_item_model,
        _evidence_card_model_fn=_evidence_card_model,
        trusted_related_fn=trusted_related,
        register_evidence_cards_fn=register_evidence_cards,
    )

def _query_construction_graph(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    retrieval: TrustedResult,
    generation: GenerationSelection,
    calibration: Calibration | None,
    graph_expansion: str,
    max_graph_nodes: int,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> tuple[TrustedResult, dict[str, object]]:
    """Compatibility wrapper for query graph orchestration owned by query_construction_api."""
    return _query_construction_api._query_construction_graph(
        store,
        embedder,
        query,
        retrieval,
        generation,
        calibration,
        graph_expansion,
        max_graph_nodes,
        security_policy=security_policy,
        access_context=access_context,
        _expand_semantic_graph_fn=_expand_semantic_graph,
    )


def graph_first_retrieval(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    *,
    mode: GraphFirstMode = "hybrid",
    source: str | None = None,
    k: int = 5,
    max_candidates: int = MAX_GRAPH_FIRST_CANDIDATES,
    expected_generation_id: str | None = None,
    policy: TrustPolicy | None = None,
    calibration: Calibration | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> dict[str, object]:
    """Compatibility wrapper for graph-first retrieval owned by graph_first_api."""
    return _graph_first_api.graph_first_retrieval(
        store,
        embedder,
        query,
        mode=mode,
        source=source,
        k=k,
        max_candidates=max_candidates,
        expected_generation_id=expected_generation_id,
        policy=policy,
        calibration=calibration,
        security_policy=security_policy,
        access_context=access_context,
        _retrieve_trusted_fn=_retrieve_trusted,
        _store_graph_fn=_store_graph,
        _cached_semantic_graph_fn=_cached_semantic_graph,
        _combined_graph_policy_fingerprint_fn=_combined_graph_policy_fingerprint,
        _validate_security_context_fn=_validate_security_context,
    )


def query_construction_challenge(
    store: PgVectorStore,
    embedder: Embedder,
    original_prompt: str,
    query: str,
    *,
    arm: QueryConstructionArm = "original_loop",
    source: str | None = None,
    k: int = 5,
    round_index: int = 0,
    frame: Mapping[str, object] | None = None,
    expected_generation_id: str | None = None,
    graph_expansion: str = "off",
    max_graph_nodes: int = 32,
    policy: TrustPolicy | None = None,
    calibration: Calibration | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> dict[str, object]:
    """Compatibility wrapper for query construction owned by query_construction_api."""
    return _query_construction_api.query_construction_challenge(
        store,
        embedder,
        original_prompt,
        query,
        arm=arm,
        source=source,
        k=k,
        round_index=round_index,
        frame=frame,
        expected_generation_id=expected_generation_id,
        graph_expansion=graph_expansion,
        max_graph_nodes=max_graph_nodes,
        policy=policy,
        calibration=calibration,
        security_policy=security_policy,
        access_context=access_context,
        _retrieve_trusted_fn=_retrieve_trusted,
        _query_construction_graph_fn=_query_construction_graph,
    )


class _SemanticGraphFlight:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: SemanticGraphProjection | None = None
        self.error: BaseException | None = None


_GRAPH_PROJECTION_LOCK = threading.Lock()


@dataclass(frozen=True)
class _SemanticGraphIndexes:
    """Immutable adjacency indexes derived from one persisted graph generation."""

    mentions_by_chunk: Mapping[str, frozenset[str]]
    chunks_by_entity: Mapping[str, frozenset[str]]
    relation_indexes_by_entity: Mapping[str, frozenset[int]]
    ambiguous_entities: frozenset[str]
    validity_by_chunk: Mapping[str, tuple[datetime | None, datetime | None]]


_GraphCandidate = _graph_expansion.GraphCandidate


def _resolve_graph_calibration(
    store: PgVectorStore,
    request: ReasoningRequest,
    calibration: Calibration | None,
) -> Calibration | None:
    """Resolve the generation calibration used by graph selection when none was supplied."""
    if calibration is not None:
        return calibration
    resolver = getattr(store, "resolve_calibration", None)
    if not callable(resolver):
        return None
    with _generation_scope(store, request.generation.generation_id):
        resolution = resolver()
    artifact = getattr(resolution, "artifact", None)
    return artifact.runtime if artifact is not None else None


def _graph_candidate_rerank_score(
    candidate: _GraphCandidate,
    cosine: float,
    calibration: Calibration | None,
) -> float:
    return _graph_expansion.graph_candidate_rerank_score(candidate, cosine, calibration)


def _merge_graph_hits(
    retrieval: TrustedResult,
    accepted: Sequence[TrustedHit],
    candidate_scores: Mapping[str, float],
    calibration: Calibration | None,
    max_items: int = GRAPH_FILL_SLOT_COUNT,
    tail_replacement_margin: float | None = None,
) -> list[TrustedHit]:
    return _graph_expansion.merge_graph_hits(
        retrieval,
        accepted,
        candidate_scores,
        calibration,
        max_items=max_items,
        tail_replacement_margin=tail_replacement_margin,
    )


def _assemble_graph_first_context(
    retrieval: RetrievalResult,
    graph_candidates: Sequence[ScoredChunk],
    *,
    seed_k: int = GRAPH_FIRST_SEED_K,
    context_k: int = GRAPH_FIRST_CONTEXT_K,
    calibration: Calibration | None = None,
    tail_replacement_margin: float | None = None,
) -> RetrievalResult:
    return _graph_expansion.assemble_graph_first_context(
        retrieval,
        graph_candidates,
        seed_k=seed_k,
        context_k=context_k,
        calibration=calibration,
        tail_replacement_margin=tail_replacement_margin,
    )


def _provisional_graph_seed_result(
    retrieval: RetrievalResult,
    store: PgVectorStore,
    request: ReasoningRequest,
) -> TrustedResult:
    """Adapt raw top seeds to the graph provider's seed interface without claiming trust."""
    seeds: list[TrustedHit] = []
    for scored in retrieval.hits[:GRAPH_FIRST_SEED_K]:
        metadata = scored.chunk.metadata
        ord_value = metadata.get("ord")
        seeds.append(
            TrustedHit(
                chunk=scored.chunk,
                cosine=float(scored.score),
                confidence=0.0,
                verdict="ok",
                provenance=Provenance(
                    source=scored.chunk.source,
                    file=(
                        str(metadata["file"])
                        if metadata.get("file") is not None
                        else scored.chunk.source
                    ),
                    ord=ord_value if isinstance(ord_value, int) else None,
                    indexed_at=scored.indexed_at,
                    first_indexed_at=scored.first_indexed_at,
                ),
                validity=Validity(None, None, None),
            )
        )
    return TrustedResult(
        query=retrieval.query,
        hits=seeds,
        abstained=not seeds,
        reason="" if seeds else "no supporting retrieval",
        gap_warning=retrieval.gap_warning,
        staleness=retrieval.staleness,
        diagnostics=retrieval.diagnostics,
        tenant_id=getattr(store, "tenant", None),
        generation_id=request.generation.generation_id or getattr(store, "generation_id", None),
        pipeline_fingerprint=request.generation.pipeline_fingerprint,
        corpus_fingerprint=request.generation.corpus_fingerprint,
    )


_SEMANTIC_GRAPH_INDEXES: OrderedDict[
    tuple[str, str, str, tuple[tuple[str, str, str], ...]], _SemanticGraphIndexes
] = OrderedDict()
_SEMANTIC_GRAPH_INDEX_CACHE_MAX = 4
_SEMANTIC_GRAPH_CACHE: OrderedDict[
    tuple[str, str, str | None, str | None, str | None], SemanticGraphProjection | None
] = OrderedDict()
_SEMANTIC_GRAPH_INFLIGHT: dict[
    tuple[str, str, str | None, str | None, str | None], _SemanticGraphFlight
] = {}
_SEMANTIC_GRAPH_CACHE_MAX = 4


class _ProposalFlight:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.result: tuple[InferenceProposal, ...] | None = None
        self.error: BaseException | None = None


_DETERMINISTIC_PROPOSAL_CACHE: OrderedDict[
    tuple[str, str, str, str, str], tuple[InferenceProposal, ...]
] = OrderedDict()
_DETERMINISTIC_PROPOSAL_INFLIGHT: dict[tuple[str, str, str, str, str], _ProposalFlight] = {}
_DETERMINISTIC_PROPOSAL_CACHE_LOCK = threading.Lock()
_DETERMINISTIC_PROPOSAL_CACHE_MAX = 16


def _reset_graph_projection_cache() -> None:
    _graph_projection._reset_graph_projection_cache()
    with _GRAPH_PROJECTION_LOCK:
        _SEMANTIC_GRAPH_INDEXES.clear()
        _SEMANTIC_GRAPH_CACHE.clear()
        _SEMANTIC_GRAPH_INFLIGHT.clear()
    with _DETERMINISTIC_PROPOSAL_CACHE_LOCK:
        _DETERMINISTIC_PROPOSAL_CACHE.clear()
        _DETERMINISTIC_PROPOSAL_INFLIGHT.clear()
    _reset_planner_index_cache()


#: The partition for the authorization view used to make proposals: the same scope the
#: filtered graph cache keys on, so a proposal is never reused across views.
_proposal_policy_scope = _authorization_scope


def _cached_deterministic_proposals(
    graph: ReasoningGraphProjection,
    *,
    pipeline_id: str,
    policy_scope: str,
) -> tuple[InferenceProposal, ...]:
    """Cache deterministic proposal output for one immutable graph serving identity."""
    key = (
        graph.tenant_id,
        graph.generation_id,
        graph.fingerprint,
        pipeline_id,
        policy_scope,
    )
    with _DETERMINISTIC_PROPOSAL_CACHE_LOCK:
        cached = _DETERMINISTIC_PROPOSAL_CACHE.get(key)
        if cached is not None:
            _DETERMINISTIC_PROPOSAL_CACHE.move_to_end(key)
            return cached
        flight = _DETERMINISTIC_PROPOSAL_INFLIGHT.get(key)
        owner = flight is None
        if owner:
            flight = _ProposalFlight()
            _DETERMINISTIC_PROPOSAL_INFLIGHT[key] = flight
    assert flight is not None
    if not owner:
        flight.done.wait()
        if flight.error is not None:
            raise flight.error
        assert flight.result is not None
        return flight.result
    try:
        proposals = tuple(deterministic_inference_proposals(graph, pipeline_id=pipeline_id))
    except BaseException as exc:
        with _DETERMINISTIC_PROPOSAL_CACHE_LOCK:
            flight.error = exc
            _DETERMINISTIC_PROPOSAL_INFLIGHT.pop(key, None)
            flight.done.set()
        raise
    with _DETERMINISTIC_PROPOSAL_CACHE_LOCK:
        while len(_DETERMINISTIC_PROPOSAL_CACHE) >= _DETERMINISTIC_PROPOSAL_CACHE_MAX:
            _DETERMINISTIC_PROPOSAL_CACHE.popitem(last=False)
        _DETERMINISTIC_PROPOSAL_CACHE[key] = proposals
        flight.result = proposals
        _DETERMINISTIC_PROPOSAL_INFLIGHT.pop(key, None)
        flight.done.set()
    return proposals


def _semantic_graph_indexes(semantic: SemanticGraphProjection) -> _SemanticGraphIndexes:
    """Build graph adjacency once per immutable tenant, generation, and graph identity."""
    performance = current_performance_trace()
    relation_identity = tuple(
        (relation.id, relation.subject_id, relation.object_id) for relation in semantic.relations
    )
    key = (semantic.tenant_id, semantic.generation_id, semantic.graph_id, relation_identity)
    with _GRAPH_PROJECTION_LOCK:
        cached = _SEMANTIC_GRAPH_INDEXES.get(key)
        if cached is not None:
            _SEMANTIC_GRAPH_INDEXES.move_to_end(key)
            if performance is not None:
                performance.add("adjacency_cache_hits")
            return cached

    span: AbstractContextManager[Any]
    if performance is not None:
        performance.add("adjacency_cache_misses")
        span = performance.span("adjacency_construction_ms")
    else:
        span = nullcontext()
    with span:
        mentions_by_chunk: dict[str, set[str]] = {}
        chunks_by_entity: dict[str, set[str]] = {}
        relation_indexes_by_entity: dict[str, set[int]] = {}
        validity_by_chunk: dict[str, tuple[datetime | None, datetime | None]] = {}
        for mention in semantic.mentions:
            mentions_by_chunk.setdefault(mention.chunk_id, set()).add(mention.entity_id)
            chunks_by_entity.setdefault(mention.entity_id, set()).add(mention.chunk_id)
            if mention.chunk_id not in validity_by_chunk:
                try:
                    valid_from = mention.metadata.get("valid_from")
                    valid_until = mention.metadata.get("valid_until")
                    start = (
                        datetime.fromisoformat(str(valid_from).replace("Z", "+00:00"))
                        if valid_from
                        else None
                    )
                    end = (
                        datetime.fromisoformat(str(valid_until).replace("Z", "+00:00"))
                        if valid_until
                        else None
                    )
                    if start is not None and start.tzinfo is None:
                        start = start.replace(tzinfo=UTC)
                    if end is not None and end.tzinfo is None:
                        end = end.replace(tzinfo=UTC)
                    validity_by_chunk[mention.chunk_id] = (start, end)
                except (TypeError, ValueError):
                    validity_by_chunk[mention.chunk_id] = (None, None)
        for relation_index, relation in enumerate(semantic.relations):
            relation_indexes_by_entity.setdefault(relation.subject_id, set()).add(relation_index)
            relation_indexes_by_entity.setdefault(relation.object_id, set()).add(relation_index)
        indexes = _SemanticGraphIndexes(
            mentions_by_chunk={key: frozenset(value) for key, value in mentions_by_chunk.items()},
            chunks_by_entity={key: frozenset(value) for key, value in chunks_by_entity.items()},
            relation_indexes_by_entity={
                key: frozenset(value) for key, value in relation_indexes_by_entity.items()
            },
            ambiguous_entities=frozenset(
                entity_id
                for diagnostic in semantic.diagnostics
                if diagnostic.kind == "ambiguous_entity"
                for entity_id in diagnostic.entity_ids
            ),
            validity_by_chunk=validity_by_chunk,
        )
    with _GRAPH_PROJECTION_LOCK:
        existing = _SEMANTIC_GRAPH_INDEXES.get(key)
        if existing is not None:
            _SEMANTIC_GRAPH_INDEXES.move_to_end(key)
            return existing
        while len(_SEMANTIC_GRAPH_INDEXES) >= _SEMANTIC_GRAPH_INDEX_CACHE_MAX:
            _SEMANTIC_GRAPH_INDEXES.popitem(last=False)
        _SEMANTIC_GRAPH_INDEXES[key] = indexes
    return indexes


def _validate_security_context(
    store: PgVectorStore,
    security_policy: SourceSecurityPolicy | None,
    access_context: AccessContext | None,
) -> None:
    if security_policy is None:
        return
    if access_context is None:
        raise ValueError("access_context is required when security_policy is configured")
    store_tenant = getattr(store, "tenant", None)
    if isinstance(store_tenant, str) and store_tenant != access_context.tenant:
        raise PermissionError("access context tenant does not match the serving store")


def _generation_scope(
    store: PgVectorStore, generation_id: str | None
) -> AbstractContextManager[Any]:
    """Pin generation-scoped store calls when a request names an immutable generation."""
    pin_generation = getattr(store, "pin_generation", None)
    if generation_id and callable(pin_generation):
        return cast(AbstractContextManager[Any], pin_generation(generation_id))
    return nullcontext()


def _cached_semantic_graph(
    store: PgVectorStore,
    generation_id: str,
    readiness: Any,
    policy_fingerprint: str | None,
) -> SemanticGraphProjection | None:
    """Load the lazy semantic graph through a bounded generation and rebuild cache."""
    graph_fingerprint = getattr(readiness, "graph_fingerprint", None) if readiness else None
    key = (
        store.tenant,
        generation_id,
        _graph_projection._corpus_fingerprint(store, generation_id),
        graph_fingerprint,
        policy_fingerprint,
    )
    with _GRAPH_PROJECTION_LOCK:
        if key in _SEMANTIC_GRAPH_CACHE:
            cached = _SEMANTIC_GRAPH_CACHE[key]
            _SEMANTIC_GRAPH_CACHE.move_to_end(key)
            performance = current_performance_trace()
            if performance is not None:
                performance.add("projection_cache_hits")
            return cached

        flight = _SEMANTIC_GRAPH_INFLIGHT.get(key)
        owner = flight is None
        if owner:
            flight = _SemanticGraphFlight()
            _SEMANTIC_GRAPH_INFLIGHT[key] = flight
    assert flight is not None
    performance = current_performance_trace()
    if performance is not None:
        performance.add("projection_cache_misses")
        performance.add(
            "projection_single_flight_owners" if owner else "projection_single_flight_waiters"
        )
    if not owner:
        flight.done.wait()
        if flight.error is not None:
            raise flight.error
        return flight.result

    try:
        loader = getattr(store, "load_semantic_graph", None)
        if callable(loader):
            with _generation_scope(store, generation_id):
                semantic = cast(SemanticGraphProjection | None, loader(generation_id))
        else:
            with _generation_scope(store, generation_id):
                semantic = _store_graph(
                    store,
                    include_text=False,
                    policy_fingerprint=policy_fingerprint,
                ).semantic_graph
        if semantic is not None and readiness is not None and getattr(readiness, "ready", True):
            actual = semantic.readiness()
            if (
                actual.graph_fingerprint != getattr(readiness, "graph_fingerprint", None)
                or (getattr(readiness, "tenant_id", actual.tenant_id) != actual.tenant_id)
                or (
                    getattr(readiness, "generation_id", actual.generation_id)
                    != actual.generation_id
                )
                or getattr(readiness, "graph_id", actual.graph_id) != actual.graph_id
            ):
                _log.warning(
                    "semantic graph does not match its generation readiness marker for %s",
                    generation_id,
                )
                semantic = None
    except BaseException as exc:
        with _GRAPH_PROJECTION_LOCK:
            flight.error = exc
            _SEMANTIC_GRAPH_INFLIGHT.pop(key, None)
            flight.done.set()
        raise
    with _GRAPH_PROJECTION_LOCK:
        while len(_SEMANTIC_GRAPH_CACHE) >= _SEMANTIC_GRAPH_CACHE_MAX:
            _SEMANTIC_GRAPH_CACHE.popitem(last=False)
        _SEMANTIC_GRAPH_CACHE[key] = semantic
        flight.result = semantic
        _SEMANTIC_GRAPH_INFLIGHT.pop(key, None)
        flight.done.set()
    return semantic


def related_memory(
    store: PgVectorStore,
    seed_chunk_id: str,
    *,
    relation: str = "source",
    max_items: int = 5,
    calibration: Calibration | None = None,
    policy: TrustPolicy | None = None,
    explain: bool = False,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> RelatedResult:
    """Return structurally related evidence after independent trust evaluation.

    Args:
        store: tenant and generation bound read store.
        seed_chunk_id: chunk that defines the relation.
        relation: `source`, `ordinal`, or `supersession`.
        max_items: positive bounded candidate limit.
        calibration: optional trust calibration, resolved from the store when omitted.
        explain: include stable machine readable explanation metadata.

    Raises:
        ValueError: if the relation, seed, or item limit is invalid.
    """
    result = trusted_related(
        store,
        seed_chunk_id,
        relation=relation,  # type: ignore[arg-type]
        max_items=max_items,
        calibration=calibration,
        policy=policy,
        explain=explain,
        security_policy=security_policy,
        access_context=access_context,
    )
    items = [_trusted_evidence_item_model(item) for item in result.items]
    return RelatedResult(
        seed_chunk_id=result.seed_chunk_id,
        relation=result.relation,
        generation_id=result.generation_id,
        items=items,
        rejected_count=result.rejected_count,
        explanation=result.explanation,
    )


def _retrieval_graph(
    retrieval: TrustedResult, *, include_text: bool = True
) -> ReasoningGraphProjection:
    chunks = [hit.chunk for hit in retrieval.hits if hit.verdict == "ok"]
    return build_reasoning_graph(
        chunks,
        tenant_id=retrieval.tenant_id or "default",
        generation_id=retrieval.generation_id or "legacy",
        pipeline_fingerprint=retrieval.pipeline_fingerprint,
        corpus_fingerprint=retrieval.corpus_fingerprint,
        include_text=include_text,
    )


def _graph_precision_settings() -> tuple[str, str, int, int, float]:
    variant = os.environ.get("RECALL_GRAPH_PRECISION_VARIANT", "combined").strip().lower()
    if variant not in GRAPH_PRECISION_VARIANTS:
        variant = "combined"
    relation_control = os.environ.get("RECALL_GRAPH_RELATION_CONTROL", "none").strip().lower()
    if relation_control not in GRAPH_RELATION_CONTROLS:
        relation_control = "none"
    try:
        relation_control_seed = int(
            os.environ.get("RECALL_GRAPH_RELATION_CONTROL_SEED", "20260825")
        )
    except ValueError:
        relation_control_seed = 20260825
    try:
        hub_threshold = int(
            os.environ.get("RECALL_GRAPH_HUB_DEGREE_THRESHOLD", str(GRAPH_HUB_DEGREE_THRESHOLD))
        )
    except ValueError:
        hub_threshold = GRAPH_HUB_DEGREE_THRESHOLD
    if hub_threshold not in {16, 32, 64}:
        hub_threshold = GRAPH_HUB_DEGREE_THRESHOLD
    try:
        cosine_margin = float(
            os.environ.get("RECALL_GRAPH_COSINE_MARGIN", str(GRAPH_COSINE_MARGIN))
        )
    except ValueError:
        cosine_margin = GRAPH_COSINE_MARGIN
    if cosine_margin not in {0.05, 0.10, 0.15, 0.20}:
        cosine_margin = GRAPH_COSINE_MARGIN
    return variant, relation_control, relation_control_seed, hub_threshold, cosine_margin


def _graph_tail_replacement_margin() -> float | None:
    """Read the opt-in calibrated margin for replacing one direct tail item."""
    raw = os.environ.get("RECALL_GRAPH_TAIL_REPLACEMENT_MARGIN", "off").strip().lower()
    if raw in {"", "off", "none"}:
        return None
    if raw in {"on", "true"}:
        return GRAPH_TAIL_REPLACEMENT_MARGIN
    try:
        margin = float(raw)
    except ValueError:
        return None
    return margin if margin in GRAPH_TAIL_REPLACEMENT_MARGINS else None


BENCHMARK_RETRIEVAL_LEG_DEPTH = 100
BENCHMARK_DOCUMENT_EXPANSION_SOURCES = 2
BENCHMARK_DOCUMENT_EXPANSION_CHUNKS = 8


def _retrieval_leg_benchmark_audit_enabled() -> bool:
    """Return whether a generation-pinned benchmark may expose per-leg candidates."""
    truthy = {"1", "true", "yes", "on"}
    return (
        os.environ.get("RECALL_BENCHMARK_PIN", "").strip().lower() in truthy
        and os.environ.get("RECALL_BENCHMARK_RETRIEVAL_LEG_AUDIT", "").strip().lower() in truthy
    )


def _document_expansion_benchmark_audit_enabled() -> bool:
    """Return whether a generation-pinned benchmark may run document expansion arms."""
    truthy = {"1", "true", "yes", "on"}
    return (
        os.environ.get("RECALL_BENCHMARK_PIN", "").strip().lower() in truthy
        and os.environ.get("RECALL_BENCHMARK_DOCUMENT_EXPANSION_AUDIT", "").strip().lower()
        in truthy
    )


def _source_admission_benchmark_audit_enabled() -> bool:
    """Return whether a generation-pinned benchmark may expose the full trust pool."""
    truthy = {"1", "true", "yes", "on"}
    return (
        os.environ.get("RECALL_BENCHMARK_PIN", "").strip().lower() in truthy
        and os.environ.get("RECALL_BENCHMARK_SOURCE_ADMISSION_AUDIT", "").strip().lower() in truthy
    )


def _source_conditioning_reuse_benchmark_audit_enabled() -> bool:
    """Return whether a generation pinned benchmark may compare reused and repeated traces."""
    truthy = {"1", "true", "yes", "on"}
    return (
        os.environ.get("RECALL_BENCHMARK_PIN", "").strip().lower() in truthy
        and os.environ.get("RECALL_BENCHMARK_SOURCE_CONDITIONING_REUSE_AUDIT", "").strip().lower()
        in truthy
    )


def _source_conditioning_shadow_sampled(query: str, env: Mapping[str, str] | None = None) -> bool:
    """Resolve the off by default deterministic source conditioning shadow sample."""
    values = os.environ if env is None else env
    mode = values.get("RECALL_SOURCE_CONDITIONING_MODE", "off").strip().lower()
    if mode == "off":
        return False
    if mode != "shadow":
        raise SourceConditioningArtifactError(
            "RECALL_SOURCE_CONDITIONING_MODE must be off or shadow"
        )
    raw_rate = values.get("RECALL_SOURCE_CONDITIONING_SHADOW_SAMPLE_RATE", "0").strip()
    try:
        rate = float(raw_rate)
    except ValueError as exc:
        raise SourceConditioningArtifactError(
            "source conditioning shadow sample rate must be numeric"
        ) from exc
    if not 0.0 <= rate <= 1.0:
        raise SourceConditioningArtifactError(
            "source conditioning shadow sample rate must be between zero and one"
        )
    if rate == 0.0:
        return False
    if rate == 1.0:
        return True
    sample = int.from_bytes(hashlib.sha256(query.encode("utf-8")).digest()[:8], "big")
    return sample < int(rate * (1 << 64))


def _atomic_rescue_shadow_sampled(query: str, env: Mapping[str, str] | None = None) -> bool:
    """Resolve the off by default deterministic atomic rescue shadow sample."""
    values = os.environ if env is None else env
    mode = values.get("RECALL_ATOMIC_RESCUE_MODE", "off").strip().lower()
    if mode == "off":
        return False
    if mode != "shadow":
        raise AtomicRescueArtifactError("RECALL_ATOMIC_RESCUE_MODE must be off or shadow")
    raw_rate = values.get("RECALL_ATOMIC_RESCUE_SHADOW_SAMPLE_RATE", "0").strip()
    try:
        rate = float(raw_rate)
    except ValueError as exc:
        raise AtomicRescueArtifactError("atomic rescue shadow sample rate must be numeric") from exc
    if not 0.0 <= rate <= 1.0:
        raise AtomicRescueArtifactError(
            "atomic rescue shadow sample rate must be between zero and one"
        )
    if rate == 0.0:
        return False
    if rate == 1.0:
        return True
    sample = int.from_bytes(hashlib.sha256(query.encode("utf-8")).digest()[:8], "big")
    return sample < int(rate * (1 << 64))


def _atomic_rescue_shadow_payload(
    *,
    artifact_path: str,
    query: str,
    query_vector: Sequence[float],
    candidate_trace: tuple[RetrievalCandidateTrace, TrustedResult, Calibration],
    baseline: TrustedResult,
    embedder: Embedder,
    expected_path: str | None = None,
) -> dict[str, object]:
    """Select one nonserving atomic parent from the already executed dense trace."""

    artifact = load_atomic_rescue_artifact(artifact_path)
    artifact.assert_compatible(result=baseline, embedder=embedder)
    dense = candidate_trace[0].dense
    selector_started = time.perf_counter()
    selection = select_atomic_rescue(artifact, query_vector, dense)
    selector_ms = (time.perf_counter() - selector_started) * 1000.0
    dense_rank_six = dense[5].chunk.id if len(dense) > 5 else None
    payload: dict[str, object] = {
        "status": "ok",
        "selected_parent_equal_dense_rank_six": selection.chunk_id == dense_rank_six,
        "selector_ms": selector_ms,
        "artifact_load_ms": artifact.load_ms,
        "resident_memory_delta_bytes": artifact.resident_memory_delta_bytes,
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }
    if expected_path is not None:
        identity_parity, score_parity = atomic_rescue_expectation_parity(
            expected_path,
            query=query,
            selection=selection,
        )
        payload["benchmark_identity_parity"] = identity_parity
        payload["benchmark_score_parity"] = score_parity
        reference_identity, reference_score = atomic_rescue_reference_parity(
            artifact,
            query_vector,
            dense,
            selection,
        )
        payload["benchmark_reference_identity_parity"] = reference_identity
        payload["benchmark_reference_score_parity"] = reference_score
    return payload


def _source_conditioning_shadow_payload(
    *,
    artifact_path: str,
    leg_audit: Mapping[str, object],
    pool_audit: Mapping[str, object],
    baseline: TrustedResult,
    embedder: Embedder,
    profile: RetrievalProfile,
    policy: str = "alpha008",
) -> dict[str, object]:
    """Compute a non-serving source-conditioned selection without exposing candidate data."""
    if policy not in SOURCE_CONDITIONING_SHADOW_POLICIES:
        raise SourceConditioningArtifactError(
            "RECALL_SOURCE_CONDITIONING_SHADOW_POLICY must be alpha008 or guarded_spare_slot"
        )
    artifact = load_source_conditioning_artifact(artifact_path)
    pipeline_fingerprint = baseline.pipeline_fingerprint
    if not pipeline_fingerprint:
        raise SourceConditioningArtifactError("serving result has no pipeline fingerprint")
    candidate_k = pool_audit["candidate_k"]
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int):
        raise SourceConditioningArtifactError("shadow candidate_k must be an integer")
    threshold = pool_audit["threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise SourceConditioningArtifactError("shadow threshold must be numeric")
    artifact.assert_compatible(
        pipeline_fingerprint=pipeline_fingerprint,
        embedding_profile=embedding_profile_id(embedder),
        retrieval_profile=profile.name,
        candidate_k=candidate_k,
    )
    pool = pool_audit["items"]
    dense = leg_audit["dense"]
    sparse = leg_audit["sparse"]
    if not isinstance(pool, list) or not isinstance(dense, list) or not isinstance(sparse, list):
        raise SourceConditioningArtifactError("shadow candidate traces must be lists")
    base_selected = select_source_conditioned(
        artifact,
        pool,
        dense,
        sparse,
        threshold=float(threshold),
    )
    receipts: list[dict[str, object]] = []
    selected = base_selected
    if policy == "guarded_spare_slot":
        selected, receipts = fill_source_conditioned_spare_slots(
            artifact,
            base_selected,
            pool,
            dense,
            sparse,
        )
    baseline_ids = [hit.chunk.id for hit in baseline.hits if hit.verdict == "ok"][
        : artifact.item_budget
    ]
    base_selected_ids = [str(item["chunk_id"]) for item in base_selected]
    selected_ids = [str(item["chunk_id"]) for item in selected]
    lane_counts = {
        lane: sum(str(receipt["lane"]) == lane for receipt in receipts)
        for lane in ("dual_leg", "lexical_dominant")
    }
    return {
        "status": "ok",
        "policy": policy,
        "artifact_fingerprint": artifact.artifact_fingerprint,
        "training_generation_id": artifact.training_generation_id,
        "serving_generation_id": baseline.generation_id,
        "serving_calibration_id": baseline.calibration_id,
        "serving_pipeline_fingerprint": baseline.pipeline_fingerprint,
        "serving_corpus_fingerprint": baseline.corpus_fingerprint,
        "selected_count": len(selected_ids),
        "alpha008_selected_count": len(base_selected_ids),
        "added_count": len(receipts),
        "base_prefix_preserved": selected_ids[: len(base_selected_ids)] == base_selected_ids,
        "lane_counts": lane_counts,
        "baseline_overlap_count": len(set(selected_ids) & set(baseline_ids)),
        "baseline_chunk_hashes": [chunk_identifier_hash(value) for value in baseline_ids],
        "alpha008_chunk_hashes": [chunk_identifier_hash(value) for value in base_selected_ids],
        "would_abstain": not selected_ids,
        "selected_chunk_hashes": [chunk_identifier_hash(value) for value in selected_ids],
        "added_chunk_hashes": [str(receipt["chunk_hash"]) for receipt in receipts],
    }


class _PinnedBenchmarkQueryEmbedder:
    """Serve one already computed query vector to paired benchmark retrieval arms."""

    def __init__(self, inner: Embedder, query: str, vector: list[float]) -> None:
        self._inner = inner
        self._query = query
        self._vector = list(vector)

    @property
    def dim(self) -> int:
        return int(self._inner.dim)

    @property
    def name(self) -> str:
        return str(self._inner.name)

    @property
    def profile(self) -> object | None:
        return getattr(self._inner, "profile", None)

    def embed_query(self, text: str) -> list[float]:
        if text != self._query:
            raise ValueError("benchmark may embed only the pinned query")
        return list(self._vector)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if any(text != self._query for text in texts):
            raise ValueError("benchmark may embed only the pinned query")
        return [list(self._vector) for _ in texts]


def _benchmark_candidate_identity(hit: ScoredChunk) -> tuple[str, int | None]:
    """Return the source and ordinal used by private benchmark gold labels."""
    file_value = hit.chunk.metadata.get("file")
    source = file_value if isinstance(file_value, str) and file_value else hit.chunk.source
    ordinal_value = hit.chunk.metadata.get("ord")
    ordinal = (
        int(ordinal_value)
        if isinstance(ordinal_value, int) and not isinstance(ordinal_value, bool)
        else None
    )
    return source, ordinal


def _benchmark_candidate_rows(hits: Sequence[ScoredChunk]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, hit in enumerate(hits, start=1):
        source, ordinal = _benchmark_candidate_identity(hit)
        rows.append(
            {
                "chunk_id": hit.chunk.id,
                "source": source,
                "ordinal": ordinal,
                "rank": rank,
                "cosine": float(hit.score),
            }
        )
    return rows


def _source_conditioning_reused_audits(
    candidate_trace: tuple[RetrievalCandidateTrace, TrustedResult, Calibration],
    profile: RetrievalProfile,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build source conditioning inputs from a main request candidate trace."""
    raw, trusted, calibration = candidate_trace
    trusted_by_id = {hit.chunk.id: hit for hit in trusted.hits}
    raw_ids = {hit.chunk.id for hit in raw.result.hits}
    if set(trusted_by_id) != raw_ids:
        raise RuntimeError("reused trust pool changed candidate identity")
    rows: list[dict[str, object]] = []
    for rank, hit in enumerate(raw.result.hits, start=1):
        trusted_hit = trusted_by_id[hit.chunk.id]
        candidate_source, ordinal = _benchmark_candidate_identity(hit)
        rows.append(
            {
                "chunk_id": hit.chunk.id,
                "source": candidate_source,
                "ordinal": ordinal,
                "pool_rank": rank,
                "text": hit.chunk.text,
                "cosine": float(hit.score),
                "confidence": float(trusted_hit.confidence),
                "verdict": trusted_hit.verdict,
            }
        )
    return (
        {
            "depth": profile.candidate_k,
            "dense": _benchmark_candidate_rows(raw.dense),
            "sparse": _benchmark_candidate_rows(raw.sparse),
        },
        {
            "candidate_k": profile.candidate_k,
            "pool_limit": profile.candidate_k * 2,
            "pool_size": len(rows),
            "threshold": float(calibration.threshold),
            "scale": float(calibration.scale),
            "items": rows,
        },
    )


def _retrieval_leg_benchmark_audit_payload(
    store: PgVectorStore,
    query: str,
    query_vector: list[float],
    source: str | None,
) -> dict[str, object]:
    """Fetch deep dense and lexical legs using the exact served query vector."""
    dense = store.query_dense(query_vector, k=BENCHMARK_RETRIEVAL_LEG_DEPTH, source=source)
    sparse = store.query_sparse(
        query,
        k=BENCHMARK_RETRIEVAL_LEG_DEPTH,
        vec=query_vector,
        source=source,
    )
    return {
        "depth": BENCHMARK_RETRIEVAL_LEG_DEPTH,
        "dense": _benchmark_candidate_rows(dense),
        "sparse": _benchmark_candidate_rows(sparse),
    }


def _source_admission_benchmark_audit_payload(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    query_vector: list[float],
    source: str | None,
    calibration: Calibration | None,
    policy: TrustPolicy | None,
    profile: RetrievalProfile,
) -> dict[str, object]:
    """Trust-evaluate the complete production union while retaining its fused order."""
    active_calibration = calibration
    if active_calibration is None:
        resolver = getattr(store, "resolve_calibration", None)
        resolution = resolver() if callable(resolver) else None
        artifact = getattr(resolution, "artifact", None)
        active_calibration = getattr(artifact, "runtime", None)
    if active_calibration is None:
        raise RuntimeError("source admission benchmark requires the pinned calibration")
    pinned = _PinnedBenchmarkQueryEmbedder(embedder, query, query_vector)
    values = dict(runtime_environment())
    captured: list[ScoredChunk] = []

    def capture_pool(result: RetrievalResult) -> RetrievalResult:
        captured.extend(result.hits)
        return result

    pool_limit = profile.candidate_k * 2
    result = trusted_search(
        store,
        pinned,
        query,
        k=pool_limit,
        source=source,
        calibration=calibration,
        reranker=_build_reranker(profile, env=values),
        candidate_k=profile.candidate_k,
        retrieval_profile=profile.name,
        index_generation=str(getattr(store, "generation_id", "legacy")),
        policy=policy,
        env=values,
        pre_trust_transform=capture_pool,
        _generation_snapshot=False,
    )
    trusted_by_id = {hit.chunk.id: hit for hit in result.hits}
    if set(trusted_by_id) != {hit.chunk.id for hit in captured}:
        raise RuntimeError("source admission trust pool changed candidate identity")
    rows: list[dict[str, object]] = []
    for rank, hit in enumerate(captured, start=1):
        trusted = trusted_by_id[hit.chunk.id]
        candidate_source, ordinal = _benchmark_candidate_identity(hit)
        rows.append(
            {
                "chunk_id": hit.chunk.id,
                "source": candidate_source,
                "ordinal": ordinal,
                "pool_rank": rank,
                "text": hit.chunk.text,
                "cosine": float(hit.score),
                "confidence": float(trusted.confidence),
                "verdict": trusted.verdict,
            }
        )
    return {
        "candidate_k": profile.candidate_k,
        "pool_limit": pool_limit,
        "pool_size": len(rows),
        "threshold": float(active_calibration.threshold),
        "scale": float(active_calibration.scale),
        "items": rows,
    }


def _benchmark_bundle_payload(bundle: EvidenceBundle) -> dict[str, object]:
    return {
        "decision": bundle.decision,
        "reason_code": bundle.reason_code,
        "trust_state": bundle.trust_state,
        "items": [
            {
                "chunk_id": item.chunk_id,
                "source": item.source,
                "ordinal": item.ordinal,
                "text": item.text,
                "cosine": float(item.cosine),
                "confidence": float(item.confidence),
            }
            for item in bundle.items
        ],
    }


def _benchmark_trusted_pool_payload(result: TrustedResult) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": hit.chunk.id,
            "source": hit.provenance.file or hit.chunk.source,
            "ordinal": hit.provenance.ord,
            "text": hit.chunk.text,
            "cosine": float(hit.cosine),
            "confidence": float(hit.confidence),
        }
        for hit in result.hits
        if hit.verdict == "ok"
    ]


def _document_expansion_benchmark_audit_payload(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    query_vector: list[float],
    source: str | None,
    k: int,
    calibration: Calibration | None,
    policy: TrustPolicy | None,
    profile: RetrievalProfile,
) -> dict[str, object]:
    """Run paired source-scoped expansion arms without changing the served baseline."""
    pinned = _PinnedBenchmarkQueryEmbedder(embedder, query, query_vector)
    values = dict(runtime_environment())
    common: dict[str, object] = {
        "store": store,
        "embedder": pinned,
        "query": query,
        "k": k,
        "source": source,
        "calibration": calibration,
        "candidate_k": profile.candidate_k,
        "retrieval_profile": profile.name,
        "index_generation": str(getattr(store, "generation_id", "legacy")),
        "policy": policy,
        "env": values,
        "_generation_snapshot": False,
    }

    document_started = time.perf_counter()
    document_result = trusted_search(
        **common,  # type: ignore[arg-type]
        reranker=_build_reranker(profile, env=values),
        document_expansion=DocumentExpansionPolicy(
            enabled=True,
            max_sources=BENCHMARK_DOCUMENT_EXPANSION_SOURCES,
            chunks_per_source=BENCHMARK_DOCUMENT_EXPANSION_CHUNKS,
            relational_query_only=False,
        ),
    )
    document_ms = (time.perf_counter() - document_started) * 1000.0

    structural_started = time.perf_counter()
    structural_result = trusted_search(
        **common,  # type: ignore[arg-type]
        reranker=_build_reranker(profile, env=values),
        structural_expansion=StructuralExpansionPolicy(
            enabled=True,
            max_sources=BENCHMARK_DOCUMENT_EXPANSION_SOURCES,
            chunks_per_source=BENCHMARK_DOCUMENT_EXPANSION_CHUNKS,
            radius=2,
            relational_query_only=False,
        ),
    )
    structural_ms = (time.perf_counter() - structural_started) * 1000.0

    retrieval_policy = EvidencePolicy(max_items=max(1, k))
    document_policy = EvidencePolicy(
        max_items=max(1, k),
        bundle_mode="document",
        max_documents=BENCHMARK_DOCUMENT_EXPANSION_SOURCES,
    )
    return {
        "max_sources": BENCHMARK_DOCUMENT_EXPANSION_SOURCES,
        "chunks_per_source": BENCHMARK_DOCUMENT_EXPANSION_CHUNKS,
        "radius": 2,
        "item_budget": max(1, k),
        "diagnostic_pools": {
            "document": _benchmark_trusted_pool_payload(document_result),
            "structural": _benchmark_trusted_pool_payload(structural_result),
        },
        "arms": {
            "document_retrieval": {
                **_benchmark_bundle_payload(
                    build_evidence_bundle(document_result, retrieval_policy)
                ),
                "retrieval_ms": round(document_ms, 3),
            },
            "document_bundle": {
                **_benchmark_bundle_payload(
                    build_evidence_bundle(document_result, document_policy)
                ),
                "retrieval_ms": round(document_ms, 3),
            },
            "structural_bundle": {
                **_benchmark_bundle_payload(
                    build_evidence_bundle(structural_result, document_policy)
                ),
                "retrieval_ms": round(structural_ms, 3),
            },
        },
    }


def _graph_precision_policy_fingerprint(
    settings: tuple[str, str, int, int, float] | None = None,
) -> str:
    variant, relation_control, relation_control_seed, hub_threshold, _legacy_cosine_margin = (
        settings if settings is not None else _graph_precision_settings()
    )
    return hashlib.sha256(
        "|".join(
            (
                GRAPH_PRECISION_POLICY_VERSION,
                variant,
                relation_control,
                str(relation_control_seed),
                str(hub_threshold),
                ",".join(sorted(GRAPH_DIRECTIONAL_RELATIONS)),
                ",".join(sorted(GRAPH_DIAGNOSTIC_ONLY_RELATIONS)),
                "rerank=" + ",".join(f"{weight:.2f}" for weight in GRAPH_RERANK_WEIGHTS),
                f"corroboration_cap={GRAPH_RERANK_CORROBORATION_CAP}",
                f"fill_policy={GRAPH_FILL_POLICY}",
                f"fill_slots={GRAPH_FILL_SLOT_COUNT}",
                "tail_replacement_margin="
                + (
                    "off"
                    if _graph_tail_replacement_margin() is None
                    else f"{_graph_tail_replacement_margin():.2f}"
                ),
            )
        ).encode("utf-8")
    ).hexdigest()


def _graph_precision_feature_flags(variant: str) -> tuple[bool, bool, bool, bool, bool]:
    """Return the graph precision features enabled by a diagnostic variant.

    The fourth flag is now calibrated reranking. The former cosine admission flag is retained in
    this tuple for diagnostic compatibility, but no variant performs hard cosine rejection.
    """
    return (
        variant in {"directional", "combined", "combined_no_selective"},
        variant in {"corroboration", "combined", "combined_no_selective"},
        variant in {"hub", "combined", "combined_no_selective"},
        variant in {"cosine", "combined", "combined_no_selective"},
        variant in {"selective", "combined"},
    )


def _shuffle_graph_relation_endpoints(
    relations: Sequence[Any],
    seed: int,
) -> tuple[Any, ...]:
    """Rewire object endpoints while preserving the directed degree sequences."""
    if len(relations) < 2:
        return tuple(relations)
    subjects = [relation.subject_id for relation in relations]
    objects = [relation.object_id for relation in relations]
    original_objects = tuple(objects)
    random.Random(seed).shuffle(objects)
    if tuple(objects) == original_objects and len(set(objects)) > 1:
        objects = objects[1:] + objects[:1]
    return tuple(
        replace(relation, subject_id=subject_id, object_id=object_id)
        for relation, subject_id, object_id in zip(relations, subjects, objects)
    )


def _graph_expansion_dependencies() -> _graph_expansion.GraphExpansionDependencies:
    """Bind graph orchestration to current service seams at call time."""
    return _graph_expansion.GraphExpansionDependencies(
        cached_semantic_graph=_cached_semantic_graph,
        combined_graph_policy_fingerprint=_combined_graph_policy_fingerprint,
        generation_scope=_generation_scope,
        graph_precision_feature_flags=_graph_precision_feature_flags,
        graph_precision_policy_fingerprint=_graph_precision_policy_fingerprint,
        graph_precision_settings=_graph_precision_settings,
        graph_tail_replacement_margin=_graph_tail_replacement_margin,
        resolve_graph_calibration=_resolve_graph_calibration,
        semantic_graph_indexes=_semantic_graph_indexes,
        shuffle_graph_relation_endpoints=_shuffle_graph_relation_endpoints,
        store_graph=_store_graph,
        validate_security_context=_validate_security_context,
        embed_query=embed_query,
        evaluate=evaluate,
        resolve_query_entities=resolve_query_entities,
        resolve_successor=resolve_successor,
        supersedes_key=supersedes_key,
        metrics=METRICS,
        graph_directional_relations=GRAPH_DIRECTIONAL_RELATIONS,
        graph_diagnostic_only_relations=GRAPH_DIAGNOSTIC_ONLY_RELATIONS,
        max_graph_rescoring_candidates=MAX_GRAPH_RESCORING_CANDIDATES,
        relation_kinds=frozenset(RELATION_KINDS),
    )


def _expand_semantic_graph(
    store: PgVectorStore,
    request: ReasoningRequest,
    retrieval: TrustedResult,
    calibration: Calibration | None,
    embedder: Embedder,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
    defer_trust_evaluation: bool = False,
    excluded_chunk_ids: frozenset[str] = frozenset(),
) -> SemanticGraphExpansionResult:
    return _graph_expansion.expand_semantic_graph(
        store,
        request,
        retrieval,
        calibration,
        embedder,
        dependencies=_graph_expansion_dependencies(),
        security_policy=security_policy,
        access_context=access_context,
        defer_trust_evaluation=defer_trust_evaluation,
        excluded_chunk_ids=excluded_chunk_ids,
    )


def _strict_reasoning_refusal(
    refusal: TrustRefusal,
    *,
    tenant_id: str,
    generation: GenerationSelection,
    budget: ReasoningBudget,
) -> ReasoningResponse:
    bundle = EvidenceBundle(
        query="",
        decision="abstain",
        reason_code=refusal.code.value,
        decision_state="no_supporting_evidence",
        calibrated=False,
        stale=False,
        embedding_profile="legacy",
        retrieval_profile="legacy",
        index_generation=refusal.generation_id or generation.generation_id or "legacy",
        items=(),
        trust_state="refused",
        failure_code=refusal.code.value,
    )
    response = ReasoningResponse(
        schema_version=REASONING_API_VERSION,
        outcome="abstained",
        answer=None,
        clarification_request=None,
        trusted_evidence=bundle,
        inference_proposals=(),
        provider_failures=(),
        reasoning_trace=None,
        contradictions=(),
        unsupported_gaps=(),
        citations=(),
        calibration_id=refusal.calibration_id,
        calibration_status=refusal.calibration_status,
        tenant_id=refusal.tenant_id or tenant_id,
        generation_id=refusal.generation_id or generation.generation_id,
        pipeline_fingerprint=refusal.pipeline_fingerprint or generation.pipeline_fingerprint,
        corpus_fingerprint=refusal.corpus_fingerprint or generation.corpus_fingerprint,
        query_set_digest=refusal.query_set_digest,
        trust_state="refused",
        refusal_reason=refusal.code.value,
        diagnostics=ReasoningDiagnostics(
            latency_ms=0,
            budget=budget,
            budget_used=None,
            retrieval_stage_ms={},
            generator_invoked=False,
            citations_normalized=False,
        ),
    )
    METRICS.increment(
        "recall_reasoning_outcome_total",
        outcome=response.outcome,
        trust_state=response.trust_state,
        refusal_reason=refusal.code.value,
    )
    return response


def _execute_reasoning_query(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    source: str | None,
    k: int,
    calibration: Calibration | None,
    policy: TrustPolicy | None,
    security_policy: SourceSecurityPolicy | None,
    access_context: AccessContext | None,
    graph_expansion: str,
    expand_retrieval: bool,
    answer_provider: OllamaAnswerProvider | None,
    reasoning_policy: ReasoningPolicy,
    budget: ReasoningBudget,
    as_of: datetime | None,
) -> ReasoningResponse:
    """Execute one generation bound reasoning request inside the store snapshot."""
    generation = _reasoning_generation(store)
    retrieval_cache: dict[str, TrustedResult] = {}
    retrieval_context: dict[str, list[float] | None] = {}
    graph_first_expansion: dict[str, SemanticGraphExpansionResult] = {}

    def capture_retrieval_query_vector(vector: list[float]) -> None:
        retrieval_context["query_vector"] = vector
        request._context.query_vector = vector

    def graph_first_transform(raw: RetrievalResult) -> RetrievalResult:
        provisional = _provisional_graph_seed_result(raw, store, request)
        expansion = _expand_semantic_graph(
            store,
            request,
            provisional,
            calibration,
            embedder,
            security_policy=security_policy,
            access_context=access_context,
            defer_trust_evaluation=True,
            excluded_chunk_ids=frozenset(hit.chunk.id for hit in raw.hits),
        )
        graph_first_expansion["result"] = expansion
        active_calibration = _resolve_graph_calibration(store, request, calibration)
        return _assemble_graph_first_context(
            raw,
            expansion.scored_candidates,
            calibration=active_calibration,
            tail_replacement_margin=_graph_tail_replacement_margin(),
        )

    def retrieve(request: ReasoningRequest) -> TrustedResult:
        if "result" not in retrieval_cache:
            performance = request._context.performance
            shadow_values = dict(runtime_environment()) if performance is not None else {}
            shadow_mode = (
                shadow_values.get("RECALL_SOURCE_CONDITIONING_MODE", "off").strip().lower()
            )
            shadow_sampled = False
            shadow_configuration_error = False
            if performance is not None and shadow_mode != "off":
                try:
                    shadow_sampled = _source_conditioning_shadow_sampled(query, shadow_values)
                except SourceConditioningArtifactError:
                    shadow_configuration_error = True
            atomic_mode = shadow_values.get("RECALL_ATOMIC_RESCUE_MODE", "off").strip().lower()
            atomic_sampled = False
            atomic_configuration_error = False
            if performance is not None and atomic_mode == "shadow":
                try:
                    atomic_sampled = _atomic_rescue_shadow_sampled(query, shadow_values)
                except AtomicRescueArtifactError:
                    atomic_configuration_error = True
            capture_source_trace = (
                performance is not None
                and shadow_mode != "off"
                and not shadow_configuration_error
                and shadow_sampled
                and security_policy is None
                and access_context is None
            )
            capture_atomic_trace = (
                performance is not None
                and atomic_mode == "shadow"
                and not atomic_configuration_error
                and atomic_sampled
                and source is None
                and security_policy is None
                and access_context is None
            )
            capture_shadow_trace = capture_source_trace or capture_atomic_trace
            leg_audit: dict[str, object] | None = None
            source_admission_audit: dict[str, object] | None = None
            if performance is None:
                executed = _retrieve_trusted(
                    store,
                    embedder,
                    query,
                    source,
                    k,
                    calibration,
                    policy,
                    security_policy=security_policy,
                    access_context=access_context,
                    pool_k=GRAPH_FIRST_RETRIEVAL_K if graph_expansion == "one_hop" else None,
                    pre_trust_transform=(
                        graph_first_transform if graph_expansion == "one_hop" else None
                    ),
                    query_vector_callback=(
                        capture_retrieval_query_vector if graph_expansion == "one_hop" else None
                    ),
                    capture_candidate_trace=capture_shadow_trace,
                )
            else:
                with performance.span("baseline_retrieval_ms"):
                    executed = _retrieve_trusted(
                        store,
                        embedder,
                        query,
                        source,
                        k,
                        calibration,
                        policy,
                        security_policy=security_policy,
                        access_context=access_context,
                        pool_k=GRAPH_FIRST_RETRIEVAL_K if graph_expansion == "one_hop" else None,
                        pre_trust_transform=(
                            graph_first_transform if graph_expansion == "one_hop" else None
                        ),
                        query_vector_callback=(
                            capture_retrieval_query_vector if graph_expansion == "one_hop" else None
                        ),
                        capture_candidate_trace=capture_shadow_trace,
                    )
            if performance is not None and _retrieval_leg_benchmark_audit_enabled():
                if security_policy is not None or access_context is not None:
                    raise ValueError(
                        "retrieval leg benchmark audit is unavailable with source security policy"
                    )
                query_vector = executed.query_vector
                if query_vector is None:
                    raise RuntimeError(
                        "retrieval leg benchmark audit did not capture a query vector"
                    )
                with performance.span("retrieval_leg_benchmark_audit_ms"):
                    leg_audit = _retrieval_leg_benchmark_audit_payload(
                        store,
                        query,
                        query_vector,
                        source,
                    )
                performance.set("retrieval_leg_benchmark_audit", leg_audit)
            if performance is not None and _document_expansion_benchmark_audit_enabled():
                if security_policy is not None or access_context is not None:
                    raise ValueError(
                        "document expansion benchmark audit is unavailable with source security "
                        "policy"
                    )
                query_vector = executed.query_vector
                if query_vector is None:
                    raise RuntimeError(
                        "document expansion benchmark audit did not capture a query vector"
                    )
                with performance.span("document_expansion_benchmark_audit_ms"):
                    document_audit = _document_expansion_benchmark_audit_payload(
                        store,
                        embedder,
                        query,
                        query_vector,
                        source,
                        k,
                        calibration,
                        policy,
                        executed.profile,
                    )
                performance.set("document_expansion_benchmark_audit", document_audit)
            if performance is not None and _source_admission_benchmark_audit_enabled():
                if security_policy is not None or access_context is not None:
                    raise ValueError(
                        "source admission benchmark audit is unavailable with source security "
                        "policy"
                    )
                query_vector = executed.query_vector
                if query_vector is None:
                    raise RuntimeError(
                        "source admission benchmark audit did not capture a query vector"
                    )
                with performance.span("source_admission_benchmark_audit_ms"):
                    source_admission_audit = _source_admission_benchmark_audit_payload(
                        store,
                        embedder,
                        query,
                        query_vector,
                        source,
                        calibration,
                        policy,
                        executed.profile,
                    )
                performance.set("source_admission_benchmark_audit", source_admission_audit)
            if performance is not None and shadow_mode != "off":
                if shadow_configuration_error:
                    performance.set(
                        "source_conditioning_shadow",
                        {"status": "error", "error_code": "configuration_error"},
                    )
                    METRICS.increment(
                        "recall_source_conditioning_shadow_total",
                        status="configuration_error",
                    )
                elif not shadow_sampled:
                    performance.set("source_conditioning_shadow", {"status": "not_sampled"})
                    METRICS.increment(
                        "recall_source_conditioning_shadow_total", status="not_sampled"
                    )
                elif security_policy is not None or access_context is not None:
                    performance.set(
                        "source_conditioning_shadow",
                        {"status": "skipped", "reason_code": "source_security"},
                    )
                    METRICS.increment(
                        "recall_source_conditioning_shadow_total",
                        status="security_skipped",
                    )
                else:
                    shadow_started = time.perf_counter()
                    trace_trust_ms = 0.0
                    shadow_payload: dict[str, object] | None = None
                    artifact_path = shadow_values.get(
                        "RECALL_SOURCE_CONDITIONING_ARTIFACT", ""
                    ).strip()
                    try:
                        if executed.candidate_trace is None:
                            raise RuntimeError("sampled shadow has no candidate trace")
                        trace_trusted = executed.candidate_trace[1]
                        trace_trust_ms = float(
                            trace_trusted.diagnostics.stage_ms.get(
                                "source_conditioning_trace_trust", 0.0
                            )
                        )
                        reused_legs, reused_pool = _source_conditioning_reused_audits(
                            executed.candidate_trace, executed.profile
                        )
                        if not artifact_path:
                            raise SourceConditioningArtifactError(
                                "source conditioning shadow artifact path is required"
                            )
                        shadow_payload = _source_conditioning_shadow_payload(
                            artifact_path=artifact_path,
                            leg_audit=reused_legs,
                            pool_audit=reused_pool,
                            baseline=executed.result,
                            embedder=embedder,
                            profile=executed.profile,
                            policy=shadow_values.get(
                                "RECALL_SOURCE_CONDITIONING_SHADOW_POLICY", "alpha008"
                            )
                            .strip()
                            .lower(),
                        )
                    except (OSError, SourceConditioningArtifactError):
                        performance.set(
                            "source_conditioning_shadow",
                            {"status": "error", "error_code": "artifact_error"},
                        )
                        METRICS.increment(
                            "recall_source_conditioning_shadow_total",
                            status="artifact_error",
                        )
                    except Exception:  # BROAD-CATCH: fail-open
                        _log.exception("source conditioning shadow computation failed")
                        performance.set(
                            "source_conditioning_shadow",
                            {"status": "error", "error_code": "computation_error"},
                        )
                        METRICS.increment(
                            "recall_source_conditioning_shadow_total",
                            status="computation_error",
                        )
                    reuse_ms = trace_trust_ms + (time.perf_counter() - shadow_started) * 1000.0
                    performance.set_span("source_conditioning_shadow_ms", reuse_ms)
                    if shadow_payload is not None:
                        performance.set("source_conditioning_shadow", shadow_payload)
                        METRICS.increment("recall_source_conditioning_shadow_total", status="ok")
                        added_count = shadow_payload.get("added_count")
                        if isinstance(added_count, int) and not isinstance(added_count, bool):
                            METRICS.increment(
                                "recall_source_conditioning_shadow_added_items_total",
                                value=added_count,
                            )
                        lane_counts_payload = shadow_payload.get("lane_counts")
                        if isinstance(lane_counts_payload, Mapping):
                            for lane, count in lane_counts_payload.items():
                                if isinstance(count, int) and not isinstance(count, bool):
                                    METRICS.increment(
                                        "recall_source_conditioning_shadow_lane_total",
                                        value=count,
                                        lane=str(lane),
                                    )
                        if _source_conditioning_reuse_benchmark_audit_enabled():
                            duplicate_started = time.perf_counter()
                            try:
                                query_vector = executed.query_vector
                                if query_vector is None:
                                    raise RuntimeError(
                                        "reuse benchmark did not capture a query vector"
                                    )
                                duplicate_legs = _retrieval_leg_benchmark_audit_payload(
                                    store, query, query_vector, source
                                )
                                duplicate_pool = _source_admission_benchmark_audit_payload(
                                    store,
                                    embedder,
                                    query,
                                    query_vector,
                                    source,
                                    calibration,
                                    policy,
                                    executed.profile,
                                )
                                duplicate_payload = _source_conditioning_shadow_payload(
                                    artifact_path=artifact_path,
                                    leg_audit=duplicate_legs,
                                    pool_audit=duplicate_pool,
                                    baseline=executed.result,
                                    embedder=embedder,
                                    profile=executed.profile,
                                )
                            except Exception:  # BROAD-CATCH: fail-open
                                _log.exception("source conditioning reuse benchmark audit failed")
                                performance.set(
                                    "source_conditioning_trace_reuse_audit",
                                    {"status": "error"},
                                )
                            else:
                                duplicate_ms = (time.perf_counter() - duplicate_started) * 1000.0
                                performance.set_span(
                                    "source_conditioning_duplicate_audit_ms", duplicate_ms
                                )
                                performance.set(
                                    "source_conditioning_trace_reuse_audit",
                                    {
                                        "status": "ok",
                                        "selected_hash_parity": (
                                            shadow_payload["selected_chunk_hashes"]
                                            == duplicate_payload["selected_chunk_hashes"]
                                        ),
                                        "reused_selected_chunk_hashes": shadow_payload[
                                            "selected_chunk_hashes"
                                        ],
                                        "duplicate_selected_chunk_hashes": duplicate_payload[
                                            "selected_chunk_hashes"
                                        ],
                                        "reuse_ms": round(reuse_ms, 3),
                                        "duplicate_ms": round(duplicate_ms, 3),
                                    },
                                )
            if performance is not None and atomic_mode == "shadow":
                atomic_payload: dict[str, object] | None = None
                atomic_started = time.perf_counter()
                benchmark_result = (
                    copy.deepcopy(executed.result)
                    if shadow_values.get("RECALL_BENCHMARK_PIN", "").strip().lower()
                    in {"1", "true", "yes", "on"}
                    else None
                )
                if atomic_configuration_error:
                    atomic_payload = {"status": "error", "error_code": "configuration_error"}
                elif not atomic_sampled:
                    atomic_payload = {"status": "not_sampled"}
                elif source is not None:
                    atomic_payload = {"status": "skipped", "reason_code": "source_scope"}
                elif security_policy is not None or access_context is not None:
                    atomic_payload = {"status": "skipped", "reason_code": "security_scope"}
                else:
                    artifact_path = shadow_values.get("RECALL_ATOMIC_RESCUE_ARTIFACT", "").strip()
                    try:
                        if not artifact_path:
                            raise AtomicRescueArtifactError(
                                "atomic rescue shadow artifact path is required"
                            )
                        if executed.query_vector is None:
                            raise AtomicRescueSelectionError(
                                "atomic rescue shadow has no query vector"
                            )
                        if executed.candidate_trace is None:
                            raise AtomicRescueSelectionError(
                                "atomic rescue shadow has no candidate trace"
                            )
                        atomic_payload = _atomic_rescue_shadow_payload(
                            artifact_path=artifact_path,
                            query=query,
                            query_vector=executed.query_vector,
                            candidate_trace=executed.candidate_trace,
                            baseline=executed.result,
                            embedder=embedder,
                            expected_path=(
                                shadow_values.get(
                                    "RECALL_BENCHMARK_ATOMIC_RESCUE_EXPECTED", ""
                                ).strip()
                                or None
                                if shadow_values.get("RECALL_BENCHMARK_PIN", "")
                                .strip()
                                .lower()
                                in {"1", "true", "yes", "on"}
                                else None
                            ),
                        )
                    except AtomicRescueLineageError:
                        atomic_payload = {"status": "error", "error_code": "lineage_error"}
                    except (OSError, AtomicRescueArtifactError):
                        atomic_payload = {"status": "error", "error_code": "artifact_error"}
                    except AtomicRescueSelectionError:
                        atomic_payload = {"status": "error", "error_code": "computation_error"}
                    except Exception:  # BROAD-CATCH: fail-open
                        _log.exception("atomic rescue shadow computation failed")
                        atomic_payload = {"status": "error", "error_code": "computation_error"}
                if benchmark_result is not None:
                    atomic_payload["benchmark_public_result_unchanged"] = (
                        executed.result == benchmark_result
                    )
                atomic_ms = (time.perf_counter() - atomic_started) * 1000.0
                performance.set_span("atomic_rescue_shadow_ms", atomic_ms)
                performance.set("atomic_rescue_shadow", atomic_payload)
                status = str(atomic_payload.get("status", "error"))
                reason = atomic_payload.get("reason_code") or atomic_payload.get("error_code")
                METRICS.increment(
                    "recall_atomic_rescue_shadow_total",
                    status=str(reason) if reason is not None else status,
                )
                selector_ms = atomic_payload.get("selector_ms")
                if isinstance(selector_ms, (int, float)) and not isinstance(selector_ms, bool):
                    METRICS.observe("recall_atomic_rescue_selector_ms", float(selector_ms))
            result = executed.result
            generation_id = result.generation_id or str(getattr(store, "generation_id", "legacy"))
            retrieval_cache["result"] = replace(
                result,
                tenant_id=result.tenant_id or store.tenant,
                generation_id=generation_id,
            )
            retrieval_context["query_vector"] = getattr(executed, "query_vector", None)
        request._context.query_vector = retrieval_context.get("query_vector")
        return retrieval_cache["result"]

    def graph_provider(
        request: ReasoningRequest, retrieval: TrustedResult
    ) -> ReasoningGraphProjection:
        if source is not None:
            graph = _retrieval_graph(retrieval, include_text=True)
        else:
            graph = _store_graph(
                store,
                include_text=True,
                policy_fingerprint=_combined_graph_policy_fingerprint(
                    security_policy=security_policy,
                    graph_policy_fingerprint=(
                        _graph_precision_policy_fingerprint()
                        if graph_expansion == "one_hop"
                        else None
                    ),
                ),
            )
        return _authorized_graph(store, graph, security_policy, access_context)

    def proposal_provider(
        request: ReasoningRequest,
        graph: ReasoningGraphProjection,
        retrieval: TrustedResult,
    ) -> Sequence[InferenceProposal] | ProposalProtocolReport:
        del retrieval
        return _cached_deterministic_proposals(
            graph,
            pipeline_id=graph.pipeline_fingerprint or "legacy",
            policy_scope=request.policy_scope
            or _proposal_policy_scope(security_policy, access_context),
        )

    def graph_expansion_provider(
        request: ReasoningRequest, retrieval: TrustedResult
    ) -> SemanticGraphExpansionResult:
        prepared = graph_first_expansion.get("result")
        if prepared is not None:
            return replace(prepared, retrieval=retrieval)
        return _expand_semantic_graph(
            store,
            request,
            retrieval,
            calibration,
            embedder,
            security_policy=security_policy,
            access_context=access_context,
        )

    expansion_provider = resolve_expansion_provider() if expand_retrieval else None

    def expansion_retriever(
        request: ReasoningRequest,
        proposal: ExpansionProposal,
        initial: TrustedResult,
    ) -> TrustedResult:
        del request, initial
        expanded = _retrieve_trusted(
            store,
            embedder,
            proposal.query,
            source,
            k,
            calibration,
            policy,
            security_policy=security_policy,
            access_context=access_context,
        ).result
        return replace(
            expanded,
            tenant_id=expanded.tenant_id or store.tenant,
            generation_id=expanded.generation_id or generation.generation_id,
        )

    request = ReasoningRequest(
        query=query,
        tenant_id=store.tenant,
        generation=generation,
        providers=ReasoningProviderPorts(
            retriever=retrieve,
            graph_provider=graph_provider,
            proposal_provider=proposal_provider,
            expansion_provider=expansion_provider,
            expansion_retriever=cast(ReasoningExpansionRetriever, expansion_retriever),
            graph_expansion_provider=graph_expansion_provider,
            answer_provider=answer_provider,
        ),
        policy=reasoning_policy,
        budget=budget,
        evidence_policy=EvidencePolicy(
            max_items=GRAPH_FIRST_CONTEXT_K if graph_expansion == "one_hop" else max(1, k)
        ),
        as_of=as_of,
        policy_scope=_proposal_policy_scope(security_policy, access_context),
    )
    request._context.performance = PerformanceTrace()
    try:
        with performance_trace_scope(request._context.performance):
            return reason(request)
    except TrustRefusal as exc:
        return _strict_reasoning_refusal(
            exc,
            tenant_id=store.tenant,
            generation=generation,
            budget=budget,
        )


reasoning_audit = _reasoning_api.reasoning_audit
reasoning_query = _reasoning_api.reasoning_query


def tenant_scopes(store: PgVectorStore, tenants: Sequence[str]) -> dict[str, object]:
    """Keep tenant metadata shaping behind the authenticated store boundary."""
    return {"tenants": sorted({str(store.tenant), *(str(value) for value in tenants)})}


_DESKTOP_CORPUS_PREFIX = "desktop-"


def _local_path(uri: str) -> Path | None:
    from recall.manifest import ObjectNotAllowed, local_path_for

    try:
        return local_path_for(uri)
    except ObjectNotAllowed:
        return None


def _roots_of(objects: dict[str, ManifestObjectV1]) -> tuple[Path, ...]:
    roots: dict[str, Path] = {}
    for uri in objects:
        path = _local_path(uri)
        if path is not None:
            roots.setdefault(str(path.parent), path.parent)
    return tuple(roots.values())


def _carry_forward(
    objects: dict[str, ManifestObjectV1],
) -> tuple[dict[str, ManifestObjectV1], tuple[Path, ...], int, int]:
    """Keep reachable objects, count vanished files, and restamp changed local files."""
    kept: dict[str, ManifestObjectV1] = {}
    vanished = 0
    restamped = 0
    for uri, entry in objects.items():
        local = _local_path(uri)
        if local is None:
            kept[uri] = entry
            continue
        try:
            stat = local.stat()
        except FileNotFoundError:
            vanished += 1
            continue
        except OSError:
            kept[uri] = entry
            continue
        digest = _digest_of(local)
        if stat.st_size == entry.size and digest == entry.sha256:
            kept[uri] = entry
        elif digest is None:
            kept[uri] = entry
        else:
            kept[uri] = replace(entry, version_id=digest, size=stat.st_size, sha256=digest)
            restamped += 1
    return kept, _roots_of(kept), vanished, restamped


def _digest_of(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _vanished_note(vanished: int) -> str:
    if not vanished:
        return ""
    return (
        f" ({vanished} file(s) from an earlier upload could not be re-read and are NOT in this "
        "build; re-upload them if you still need them)"
    )


def _restamped_note(restamped: int) -> str:
    if not restamped:
        return ""
    return f" ({restamped} file(s) changed since they were indexed and were re-read)"


def _query_set_for(chunks: list[str]) -> tuple[list[dict[str, object]] | None, Exception | None]:
    from recall.wizard.queryset import (
        DEFAULT_PER_CLASS,
        MIN_PER_CLASS,
        QuerySetError,
        canonicalize,
        generate_offline,
    )

    last: Exception | None = None
    for per_class in (DEFAULT_PER_CLASS, MIN_PER_CLASS):
        try:
            return canonicalize(generate_offline(chunks, per_class=per_class)), None
        except QuerySetError as exc:
            last = exc
    return None, last


def _certify_upload(
    dsn: str,
    tenant: str,
    generation_id: str,
    embedder: Embedder,
) -> str | None:
    """Calibrate and publish a desktop generation, returning a bounded refusal reason."""
    from recall.calibration_v2 import CalibrationError, CalibrationUncertified

    with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
        rows = conn.execute(
            "SELECT text FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s "
            "ORDER BY chunk_id",
            (tenant, generation_id),
        ).fetchall()
    chunks = [str(row[0]) for row in rows if str(row[0]).strip()]
    entries, last = _query_set_for(chunks)
    if entries is None:
        return f"no certifiable query set could be generated from {len(chunks)} chunk(s): {last}"
    repository = CalibrationRepository(dsn, tenant, actor="recall-desktop")
    try:
        artifact = repository.calibrate(generation_id, entries, embedder)
        if not artifact.certified:
            return f"calibration was not certified: {artifact.certification_reason}"
        repository.publish(artifact.calibration_id)
    except CalibrationUncertified as exc:
        return f"calibration was not certified: {exc}"
    except CalibrationError as exc:
        return f"calibration could not be completed: {exc}"
    return None


def _reclaim_failed(manager: GenerationManager, generation_id: str, reason: str) -> None:
    try:
        manager.fail(generation_id, reason)
        return
    except InvalidGenerationTransition:
        pass
    except Exception:  # noqa: BLE001  # BROAD-CATCH: cleanup-only
        return
    with suppress(Exception):
        manager.abandon(generation_id, reason)


def _release_superseded(manager: GenerationManager, keep: str) -> int:
    reclaimed = 0
    try:
        stale = manager.superseded_ready_generations(
            keep, corpus_version_prefix=_DESKTOP_CORPUS_PREFIX
        )
    except Exception:  # noqa: BLE001  # BROAD-CATCH: cleanup-only
        return 0
    for generation_id in stale:
        try:
            manager.abandon(generation_id, "superseded by a later desktop upload")
        except Exception:  # noqa: BLE001  # BROAD-CATCH: cleanup-only
            continue
        reclaimed += 1
    return reclaimed


def generation_ingest(
    store: PgVectorStore,
    embedder: Embedder,
    staged_root: str,
    category: str,
    security_policy: SourceSecurityPolicy | None = None,
    security_context: AccessContext | None = None,
    env: Mapping[str, str] | None = None,
) -> IndexResult:
    """Build, validate, and activate one local generation for a desktop upload."""
    job_root = Path(staged_root)
    tenant_root = job_root.parent
    job_files = sorted(path for path in job_root.rglob("*") if path.is_file())
    if not job_files:
        raise ValueError("the staged upload contains no files")

    values = runtime_environment() if env is None else env
    manager = GenerationManager(
        store._dsn,
        store.tenant,
        actor="recall-desktop",
        serving_environment=values.get("RECALL_SERVING_ENV", values.get("RECALL_ENV")),
    )
    with manager.tenant_ingest_lock():
        try:
            base = manager.servable_manifest()
            active_objects, carried_roots, vanished, restamped = _carry_forward(
                {entry.uri: entry for entry in base.objects}
            )
        except NoActiveGeneration:
            active_objects, carried_roots, vanished, restamped = {}, (), 0, 0

        for path in job_files:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            media_type = (
                "text/x-code"
                if category == "code"
                else mimetypes.guess_type(path.name)[0] or "text/plain"
            )
            entry = ManifestObjectV1(
                uri=path.resolve().as_uri(),
                version_id=digest,
                media_type=media_type,
                size=len(data),
                sha256=digest,
            )
            active_objects[entry.uri] = entry

        from recall.generation_build import BuildRequest, pipeline_for

        artifact_digest = embedder_artifact_digest(embedder)
        chunker, pipeline = pipeline_for(
            embedder,
            BuildRequest(
                chunker="code" if category == "code" else "text",
                artifact_digest=artifact_digest,
                unverified=artifact_digest is None,
            ),
        )
        manifest = IndexManifestV1(
            tenant_id=store.tenant,
            corpus_version=f"{_DESKTOP_CORPUS_PREFIX}"
            f"{hashlib.sha256(job_root.name.encode()).hexdigest()[:12]}",
            objects=tuple(sorted(active_objects.values(), key=lambda entry: entry.uri)),
        )
        generation = manager.create(manifest, pipeline, allow_unverified=not pipeline.verified)
        try:
            stats = manager.build(
                generation.generation_id,
                ExtractingLocalObjectReader((tenant_root, *carried_roots)),
                embedder,
                chunker,
                security_policy=security_policy,
                security_context=security_context,
            )
            manager.validate(generation.generation_id)
            uncertified: str | None = None
            if manager.certification_required:
                uncertified = _certify_upload(
                    store._dsn, store.tenant, generation.generation_id, embedder
                )
            try:
                manager.promote(
                    generation.generation_id,
                    unsafe_development=not manager.certification_required,
                )
            except UnsafePromotion as exc:
                reclaimed = _release_superseded(manager, generation.generation_id)
                return IndexResult(
                    files=stats.objects,
                    chunks=stats.chunks,
                    message=(
                        f"Indexed {stats.chunks} chunk(s) from {stats.objects} file(s) into "
                        f"generation {generation.generation_id}, built and validated but not live. "
                        f"It carries forward everything previously uploaded"
                        + _vanished_note(vanished)
                        + _restamped_note(restamped)
                        + (f"; {reclaimed} superseded build(s) released" if reclaimed else "")
                        + f". {uncertified or exc}"
                    ),
                )
        except Exception as exc:  # BROAD-CATCH: fail-closed
            _reclaim_failed(manager, generation.generation_id, f"desktop upload failed: {exc}")
            raise

        _release_superseded(manager, generation.generation_id)
        return IndexResult(
            files=stats.objects,
            chunks=stats.chunks,
            message=(
                f"Built and activated generation {generation.generation_id} with "
                f"{stats.chunks} chunk(s) from {stats.objects} file(s)."
                + _vanished_note(vanished)
                + _restamped_note(restamped)
            ),
        )


def _generated_calibration_queries(
    store: PgVectorStore, generation_id: str
) -> list[dict[str, object]]:
    """Build a deterministic draft query set from the active corpus.

    This is intentionally a prototype helper. The generated labels are useful for checking the
    complete workflow, but a production deployment should replace them with reviewed labels.
    """
    with psycopg.connect(store._dsn, autocommit=True, connect_timeout=10) as conn:
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (store.tenant,))
        rows = conn.execute(
            "SELECT text FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s "
            "ORDER BY chunk_id LIMIT 20",
            (store.tenant, generation_id),
        ).fetchall()
    answerable: list[str] = []
    for row in rows:
        value = str(row[0]).strip()
        if value and value not in answerable:
            answerable.append(value[:500])
    if len(answerable) < 2:
        raise ValueError(
            "at least two distinct corpus chunks are required to generate calibration labels"
        )
    return [
        *({"query": query, "answerable": True} for query in answerable),
        *(
            {
                "query": f"Prototype calibration negative sample {index}: {nonce}",
                "answerable": False,
            }
            for index, nonce in enumerate(
                (
                    "the unrecorded weather on Europa",
                    "the private password for a fictional account",
                    "the exact weight of an imaginary blue comet",
                    "the inventory of a library that does not exist",
                    "the recipe for a machine never described here",
                    "the birthplace of a person absent from this corpus",
                    "the result of a future election",
                    "the serial number of a nonexistent device",
                    "the internal schedule of an unrelated company",
                    "the answer to an invented mathematical riddle",
                    "the color of a silent radio signal",
                    "the number of doors in an imaginary building",
                    "the owner of a fictional island",
                    "the temperature inside an empty thought",
                    "the name of a removed document",
                    "the location of a lost moon",
                    "the version of an unreleased program",
                    "the price of an unnamed object",
                    "the title of a nonexistent chapter",
                    "the identity of an imaginary maintainer",
                ),
                start=1,
            )
        ),
    ]


def run_calibration(
    store: PgVectorStore,
    embedder: Embedder,
    generation_id: str | None = None,
    queries: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Measure a draft artifact, generating prototype labels when none were supplied."""
    from recall.generation_store import GenerationStore

    generation_store = GenerationStore(store._dsn, embedder.dim, tenant=store.tenant)
    try:
        selected_generation = generation_id or generation_store.active_generation_id()
    finally:
        generation_store.close()
    labels = (
        list(queries)
        if queries is not None
        else _generated_calibration_queries(store, selected_generation)
    )
    artifact = CalibrationRepository(store._dsn, store.tenant, actor="recall-mcp").calibrate(
        selected_generation,
        labels,
        embedder,
    )
    return artifact.to_dict()


def publish_calibration(store: PgVectorStore, calibration_id: str) -> dict[str, object]:
    """Publish a certified artifact after the user explicitly confirms the action."""
    artifact = CalibrationRepository(store._dsn, store.tenant, actor="recall-mcp").publish(
        calibration_id
    )
    return artifact.to_dict()
