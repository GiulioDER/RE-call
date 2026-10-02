"""Benchmark and shadow diagnostics attached to reasoning queries. Every one of them is
environment gated and off by default.

Moved out of `recall_mcp.service`, which re-exports every name defined here. Collaborators are
imported from the modules that own them, so a test patches a collaborator on THIS module.
"""

from __future__ import annotations

from recall.atomic_rescue import load_atomic_rescue_artifact  # S5b: was a call-time service lookup
from recall.embeddings import embedding_profile_id  # S5b: was a call-time service lookup
from recall.trust import trusted_search  # S5b: was a call-time service lookup
from recall_mcp import factories as _factories

from collections.abc import (
    Mapping,
    Sequence,
)

from recall.atomic_rescue import (
    atomic_rescue_expectation_parity,
    atomic_rescue_reference_parity,
    AtomicRescueArtifactError,
    select_atomic_rescue,
)
from recall.calibration import Calibration
from recall.embeddings import Embedder
from recall.evidence import (
    build_evidence_bundle,
    EvidenceBundle,
    EvidencePolicy,
)
from recall.observability import get_logger as _get_logger
from recall.profiles import RetrievalProfile
from recall.retriever import (
    DocumentExpansionPolicy,
    RetrievalCandidateTrace,
    StructuralExpansionPolicy,
)
from recall.source_conditioning import (
    chunk_identifier_hash,
    fill_source_conditioned_spare_slots,
    load_source_conditioning_artifact,
    select_source_conditioned,
    SOURCE_CONDITIONING_SHADOW_POLICIES,
    SourceConditioningArtifactError,
)
from recall.store import PgVectorStore
from recall.trust_policy import TrustPolicy
from recall.types import (
    RetrievalResult,
    ScoredChunk,
    TrustedResult,
)

from recall_mcp.settings import runtime_environment
import hashlib
import os
import time


_log = _get_logger("mcp.service")



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
        reranker=_factories._build_reranker(profile, env=values),
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
        reranker=_factories._build_reranker(profile, env=values),
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
        reranker=_factories._build_reranker(profile, env=values),
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
