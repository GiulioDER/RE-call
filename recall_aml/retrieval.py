"""Maximum quality hosted retrieval and deterministic evidence packing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
import re
import time

from recall.atomic_rescue import (
    AtomicRescueArtifactError,
    AtomicRescueSelectionError,
    insert_atomic_rescue_dense,
    insert_atomic_rescue_fused,
    load_atomic_rescue_artifact,
    resolve_atomic_rescue_manifest,
)
from recall.embeddings import Embedder, embed_query
from recall.rerank import Reranker
from recall.sparse import SparseEncoderProtocol
from recall.store import PgVectorStore
from recall.types import Chunk, ScoredChunk
from recall_aml.code4 import rank_bm25_chunks, stable_window_key
from recall_aml.graph import GRAPH_PROFILE, promote_grounded_raw
from recall_aml.models import SearchItem


CANDIDATE_WIDTH = 100
RRF_CONSTANT = 60
MAX_ITEMS = 12
CODE_PROFILE = "aml-code-exact-v1"
CODE_RRF_WEIGHT = 0.5
CODE_NEIGHBOUR_SEED_LIMIT = 8
CODE_NEIGHBOUR_PREDECESSOR_RADIUS = 1
CODE_NEIGHBOUR_SUCCESSOR_RADIUS = 1
_HISTORICAL = re.compile(r"\b(previous|formerly|before|histor|old|earlier|used to)\b", re.I)
_TOKENS = re.compile(r"[A-Za-z0-9_./:\\-]+")
_CODE_ATOMS = re.compile(r"[A-Za-z0-9_.:/\\-]+")
_INLINE_CODE = re.compile(r"`([^`\n]{1,128})`")
_PATH = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/])?(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+")
_SOFTWARE_FILE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+\."
    r"(?:py|js|jsx|ts|tsx|json|jsonl|yaml|yml|toml|ini|cfg|sql|sh|md|txt|csv|log|lock|env)"
    r"|Dockerfile|Makefile|VERSION|\.gitignore)(?![A-Za-z0-9_])",
    re.I,
)
_CLI_FLAG = re.compile(r"(?<![A-Za-z0-9_])--[A-Za-z0-9][A-Za-z0-9-]*")
_ENV_NAME = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
_EXCEPTION_NAME = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b")
_FUNCTION_CALL = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}(?=\s*\()")
_SNAKE_CASE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b")
_CAMEL_CASE = re.compile(r"\b(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+)\b")
_CODE_TOKEN_EXCLUSIONS = frozenset({"role", "content", "timestamp"})


@dataclass(frozen=True)
class RetrievalRun:
    hits: list[ScoredChunk]
    reranker_fallback: bool
    superseded_ids: frozenset[str]
    reranker_attempted: bool
    reranker_completed: bool
    candidate_input_count: int
    candidate_output_count: int
    candidate_permutation_valid: bool
    top_10_order_changed: bool
    top_10_membership_changed: bool
    top_100_order_changed: bool
    top_100_membership_changed: bool
    candidate_character_count: int
    query_character_count: int
    rerank_ms: float
    code_aware_attempted: bool
    code_aware_fallback: bool
    code_profile: str
    code_rrf_weight: float
    code_query_token_count: int
    code_match_candidate_count: int
    code_top_10_order_changed: bool
    code_top_10_membership_changed: bool
    code_top_100_order_changed: bool
    code_top_100_membership_changed: bool
    neighbour_seed_limit: int
    neighbour_seed_count: int
    neighbour_activated_seed_count: int
    neighbour_ineligible_seed_count: int
    neighbour_restored_count: int
    neighbour_invalid_count: int
    code_duplicate_output_count: int
    graph_attempted: bool = False
    graph_fallback: bool = False
    graph_profile: str = "none"
    graph_relation_hits: int = 0
    graph_candidate_count: int = 0
    graph_promoted_count: int = 0
    graph_invalid_relation_count: int = 0
    graph_top_10_order_changed: bool = False
    graph_top_100_membership_changed: bool = False
    atomic_rescue_attempted: bool = False
    atomic_rescue_active: bool = False
    atomic_rescue_fallback: bool = False
    atomic_rescue_candidate_available: bool = False


@dataclass(frozen=True)
class AtomicRescueBinding:
    """The hosted serving lineage an optional atomic artifact must match exactly."""

    mode: str
    artifact_root: str
    scope_id: str
    generation_id: str
    calibration_id: str
    pipeline_fingerprint: str
    corpus_fingerprint: str
    #: ``dense`` inserts the winner at dense rank six before fusion (a full fusion vote, so it can
    #: reach the fused top five); ``fused`` places it at fused rank six after fusion, keeping the
    #: fused top five exactly as they were. Measured 2026-09-22/23: ``dense`` lost 2 of 34 CAMBench
    #: task prompts at exact rank one.
    placement: str = "dense"


@dataclass(frozen=True)
class _AtomicRescueTransforms:
    """One loaded, lineage-checked artifact, placeable before or after fusion."""

    dense: Callable[[list[float], list[ScoredChunk]], list[ScoredChunk]]
    fused: Callable[[list[float], list[ScoredChunk], list[ScoredChunk]], list[ScoredChunk]]


@dataclass
class _AtomicRescueState:
    attempted: bool = False
    active: bool = False
    fallback: bool = False
    candidate_available: bool = False


def _rrf(rankings: Sequence[Sequence[str]], constant: int = RRF_CONSTANT) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant + rank)
    return scores


def _normalise_code_token(value: str) -> str:
    return value.replace("\\", "/").casefold().strip()


def extract_code_tokens(value: str) -> dict[str, int]:
    """Return the preregistered exact code tokens and their maximum deterministic weights."""
    weighted: dict[str, int] = {}

    def admit(raw: str, weight: int) -> None:
        token = _normalise_code_token(raw)
        if len(token) < 3 or token in _CODE_TOKEN_EXCLUSIONS:
            return
        weighted[token] = max(weight, weighted.get(token, 0))

    for pattern in (_PATH, _SOFTWARE_FILE, _CLI_FLAG, _ENV_NAME, _EXCEPTION_NAME):
        for match in pattern.finditer(value):
            admit(match.group(0), 3)
    for inline in _INLINE_CODE.finditer(value):
        for atom in _CODE_ATOMS.findall(inline.group(1)):
            admit(atom, 2)
    for pattern in (_FUNCTION_CALL, _SNAKE_CASE, _CAMEL_CASE):
        for match in pattern.finditer(value):
            admit(match.group(0), 1)
    return weighted


def _position(chunk: object) -> tuple[int, int] | None:
    metadata = getattr(chunk, "metadata", {})
    ordinal = metadata.get("ordinal")
    segment = metadata.get("segment")
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or isinstance(segment, bool)
        or not isinstance(segment, int)
    ):
        return None
    return ordinal, segment


@dataclass(frozen=True)
class _CodeAwareResult:
    hits: list[ScoredChunk]
    query_token_count: int
    match_candidate_count: int
    top_10_order_changed: bool
    top_10_membership_changed: bool
    top_100_order_changed: bool
    top_100_membership_changed: bool
    neighbour_seed_count: int
    neighbour_activated_seed_count: int
    neighbour_ineligible_seed_count: int
    neighbour_restored_count: int
    neighbour_invalid_count: int
    duplicate_output_count: int


def _code_aware_candidates(
    store: PgVectorStore,
    query: str,
    baseline_hits: Sequence[ScoredChunk],
    fused_scores: dict[str, float],
) -> _CodeAwareResult:
    query_tokens = extract_code_tokens(query)
    baseline_ids = [hit.chunk.id for hit in baseline_hits]
    baseline_rank = {chunk_id: rank for rank, chunk_id in enumerate(baseline_ids, start=1)}
    overlap: dict[str, int] = {}
    for hit in baseline_hits:
        candidate_tokens = extract_code_tokens(hit.chunk.text)
        matched = query_tokens.keys() & candidate_tokens.keys()
        if matched:
            overlap[hit.chunk.id] = sum(query_tokens[token] for token in matched)

    code_ranking = sorted(
        overlap,
        key=lambda chunk_id: (-overlap[chunk_id], baseline_rank[chunk_id], chunk_id),
    )
    code_rank = {chunk_id: rank for rank, chunk_id in enumerate(code_ranking, start=1)}
    boosted = [
        replace(
            hit,
            score=(
                fused_scores[hit.chunk.id]
                + (
                    CODE_RRF_WEIGHT / (RRF_CONSTANT + code_rank[hit.chunk.id])
                    if hit.chunk.id in code_rank
                    else 0.0
                )
            ),
        )
        for hit in baseline_hits
    ]
    boosted.sort(key=lambda hit: (-hit.score, hit.chunk.id))

    seed_ids = {
        hit.chunk.id
        for hit in boosted
        if hit.chunk.id in overlap and str(hit.chunk.metadata.get("record_type", "raw")) == "raw"
    }
    ordered_seed_ids = [hit.chunk.id for hit in boosted if hit.chunk.id in seed_ids][
        :CODE_NEIGHBOUR_SEED_LIMIT
    ]
    seed_set = set(ordered_seed_ids)
    source_cache: dict[str, list[Chunk]] = {}
    neighbours: dict[str, list[Chunk]] = {}
    eligible_seeds = 0
    activated_seeds = 0
    ineligible_seeds = 0
    invalid_neighbours = 0
    for hit in boosted:
        if hit.chunk.id not in seed_set:
            continue
        seed_position = _position(hit.chunk)
        if seed_position is None:
            ineligible_seeds += 1
            neighbours[hit.chunk.id] = []
            continue
        source_chunks = source_cache.get(hit.chunk.source)
        if source_chunks is None:
            source_chunks = [
                chunk
                for chunk in store.chunks_for_source(hit.chunk.source)
                if str(chunk.metadata.get("record_type", "raw")) == "raw"
                and _position(chunk) is not None
            ]
            source_chunks.sort(key=lambda chunk: (*(_position(chunk) or (0, 0)), chunk.id))
            source_cache[hit.chunk.source] = source_chunks
        index = next(
            (index for index, chunk in enumerate(source_chunks) if chunk.id == hit.chunk.id),
            None,
        )
        if index is None:
            ineligible_seeds += 1
            neighbours[hit.chunk.id] = []
            continue
        eligible_seeds += 1
        adjacent: list[Chunk] = []
        if index > 0:
            adjacent.append(source_chunks[index - 1])
        if index + 1 < len(source_chunks):
            adjacent.append(source_chunks[index + 1])
        checked: list[Chunk] = []
        for chunk in adjacent:
            if chunk.source != hit.chunk.source or str(chunk.metadata.get("record_type")) != "raw":
                invalid_neighbours += 1
                continue
            checked.append(chunk)
        if checked:
            activated_seeds += 1
        neighbours[hit.chunk.id] = checked

    expanded: list[ScoredChunk] = []
    seen: set[str] = set()
    restored = 0
    for hit in boosted:
        if hit.chunk.id not in seen:
            expanded.append(hit)
            seen.add(hit.chunk.id)
        for neighbour in neighbours.get(hit.chunk.id, []):
            if neighbour.id in seen:
                continue
            expanded.append(
                ScoredChunk(
                    chunk=neighbour,
                    score=hit.score,
                    score_kind="structural",
                )
            )
            seen.add(neighbour.id)
            restored += 1
    expanded_ids = [hit.chunk.id for hit in expanded]

    def order_changed(width: int) -> bool:
        return expanded_ids[:width] != baseline_ids[:width]

    def membership_changed(width: int) -> bool:
        return set(expanded_ids[:width]) != set(baseline_ids[:width])

    return _CodeAwareResult(
        hits=expanded,
        query_token_count=len(query_tokens),
        match_candidate_count=len(code_ranking),
        top_10_order_changed=order_changed(10),
        top_10_membership_changed=membership_changed(10),
        top_100_order_changed=order_changed(100),
        top_100_membership_changed=membership_changed(100),
        neighbour_seed_count=eligible_seeds,
        neighbour_activated_seed_count=activated_seeds,
        neighbour_ineligible_seed_count=ineligible_seeds,
        neighbour_restored_count=restored,
        neighbour_invalid_count=invalid_neighbours,
        duplicate_output_count=len(expanded_ids) - len(set(expanded_ids)),
    )


class HostedRetriever:
    def __init__(
        self,
        embedder: Embedder,
        reranker: Reranker,
        *,
        sparse_encoder: SparseEncoderProtocol | None = None,
        candidate_k: int = 100,
    ) -> None:
        if candidate_k != CANDIDATE_WIDTH:
            raise ValueError("hosted-quality candidate width is fixed at 100")
        self._embedder = embedder
        self._reranker = reranker
        self._sparse_encoder = sparse_encoder
        self._candidate_k = candidate_k

    def search(
        self,
        store: PgVectorStore,
        query: str,
        facets: Sequence[str],
        *,
        rerank: bool = True,
        learned_sparse: bool = False,
        code_aware: bool = False,
        canonical_bm25: bool = False,
        exact_dense: bool = False,
        stable_window_order: bool = False,
        atomic_rescue: AtomicRescueBinding | None = None,
    ) -> RetrievalRun:
        if learned_sparse and self._sparse_encoder is None:
            raise RuntimeError("learned sparse retrieval has no encoder")
        rankings: list[list[str]] = []
        by_id: dict[str, ScoredChunk] = {}
        dense_scores: dict[str, float] = {}
        atomic_state = _AtomicRescueState()
        transforms = self._atomic_rescue_transform(store, atomic_rescue, atomic_state)
        # Same loaded artifact and lineage checks for both placements; only where the winner
        # lands changes.
        place_fused = atomic_rescue is not None and atomic_rescue.placement == "fused"
        dense_transform = transforms.dense if transforms is not None and not place_fused else None
        fused_transform = transforms.fused if transforms is not None and place_fused else None
        primary_query_dense: tuple[list[float], list[ScoredChunk]] | None = None
        variants = [query, *list(facets)[:4]]
        vectors = [embed_query(self._embedder, variant) for variant in variants]
        if len(vectors) != len(variants):
            raise RuntimeError("query embedder returned the wrong number of vectors")
        sparse_vectors = (
            self._sparse_encoder.encode(variants)
            if learned_sparse and self._sparse_encoder is not None
            else [{} for _ in variants]
        )
        if len(sparse_vectors) != len(variants):
            raise RuntimeError("learned sparse encoder returned the wrong number of vectors")
        for variant, vector, sparse_vector in zip(variants, vectors, sparse_vectors, strict=True):
            dense = (
                store.query_dense_exact(vector, k=self._candidate_k)
                if exact_dense
                else store.query_dense(vector, k=self._candidate_k)
            )
            if primary_query_dense is None:
                primary_query_dense = (vector, list(dense))
            if dense_transform is not None:
                try:
                    dense = dense_transform(vector, dense)
                except (AtomicRescueArtifactError, AtomicRescueSelectionError):
                    # An unavailable or inapplicable rescue must leave hosted retrieval unchanged.
                    atomic_state.fallback = True
            lexical = (
                rank_bm25_chunks(
                    list(store.iter_chunks()),
                    variant,
                    k=self._candidate_k,
                    stable_ties=stable_window_order,
                )
                if canonical_bm25
                else store.query_sparse(variant, k=self._candidate_k, vec=vector)
            )
            rankings.extend(([hit.chunk.id for hit in dense], [hit.chunk.id for hit in lexical]))
            for hit in dense:
                by_id.setdefault(hit.chunk.id, hit)
                dense_scores.setdefault(hit.chunk.id, hit.score)
            for hit in lexical:
                by_id.setdefault(hit.chunk.id, hit)
                dense_scores.setdefault(hit.chunk.id, hit.score)
            if learned_sparse and sparse_vector:
                assert self._sparse_encoder is not None
                learned = store.query_learned_sparse(
                    sparse_vector,
                    k=self._candidate_k,
                    profile_id=self._sparse_encoder.profile.profile_id,
                    vec=vector,
                )
                rankings.append([hit.chunk.id for hit in learned])
                for hit in learned:
                    by_id.setdefault(hit.chunk.id, hit)
                    dense_scores.setdefault(hit.chunk.id, hit.score)
        fused = _rrf(rankings)
        ordered = sorted(
            fused,
            key=lambda chunk_id: (
                -fused[chunk_id],
                stable_window_key(by_id[chunk_id].chunk)
                if stable_window_order
                else (b"", 0, chunk_id),
            ),
        )
        fused_hits = [
            replace(by_id[chunk_id], score=dense_scores.get(chunk_id, by_id[chunk_id].score))
            for chunk_id in ordered
        ]
        if fused_transform is not None and primary_query_dense is not None:
            try:
                fused_hits = fused_transform(
                    primary_query_dense[0], primary_query_dense[1], fused_hits
                )
            except (AtomicRescueArtifactError, AtomicRescueSelectionError):
                # An unavailable or inapplicable rescue must leave hosted retrieval unchanged.
                atomic_state.fallback = True
        code_result = (
            _code_aware_candidates(store, query, fused_hits, fused)
            if code_aware
            else _CodeAwareResult(
                hits=fused_hits,
                query_token_count=0,
                match_candidate_count=0,
                top_10_order_changed=False,
                top_10_membership_changed=False,
                top_100_order_changed=False,
                top_100_membership_changed=False,
                neighbour_seed_count=0,
                neighbour_activated_seed_count=0,
                neighbour_ineligible_seed_count=0,
                neighbour_restored_count=0,
                neighbour_invalid_count=0,
                duplicate_output_count=0,
            )
        )
        hits = code_result.hits
        baseline_hits = list(hits)
        baseline_ids = [hit.chunk.id for hit in baseline_hits]
        output_ids = list(baseline_ids)
        attempted = bool(rerank)
        completed = False
        fallback = False
        rerank_ms = 0.0
        permutation_valid = not rerank
        if rerank:
            rerank_started = time.perf_counter()
            try:
                reranked = self._reranker.rerank(query, baseline_hits)
                completed = True
                output_ids = [hit.chunk.id for hit in reranked]
                permutation_valid = (
                    len(output_ids) == len(baseline_ids)
                    and len(set(output_ids)) == len(output_ids)
                    and Counter(output_ids) == Counter(baseline_ids)
                )
                if permutation_valid:
                    hits = reranked
                else:
                    fallback = True
                    hits = baseline_hits
            except Exception:  # BROAD-CATCH: deterministic fused-order serving fallback
                fallback = True
                hits = baseline_hits
                output_ids = []
            finally:
                rerank_ms = (time.perf_counter() - rerank_started) * 1_000
        served_ids = [hit.chunk.id for hit in hits]

        def order_changed(width: int) -> bool:
            return served_ids[:width] != baseline_ids[:width]

        def membership_changed(width: int) -> bool:
            return set(served_ids[:width]) != set(baseline_ids[:width])

        return RetrievalRun(
            hits=hits,
            reranker_fallback=fallback,
            superseded_ids=store.explicit_superseded_chunk_ids(),
            reranker_attempted=attempted,
            reranker_completed=completed,
            candidate_input_count=len(baseline_ids),
            candidate_output_count=len(output_ids),
            candidate_permutation_valid=permutation_valid,
            top_10_order_changed=order_changed(10),
            top_10_membership_changed=membership_changed(10),
            top_100_order_changed=order_changed(100),
            top_100_membership_changed=membership_changed(100),
            candidate_character_count=sum(len(hit.chunk.text) for hit in baseline_hits),
            query_character_count=len(query),
            rerank_ms=rerank_ms,
            code_aware_attempted=code_aware,
            code_aware_fallback=False,
            code_profile=CODE_PROFILE if code_aware else "none",
            code_rrf_weight=CODE_RRF_WEIGHT if code_aware else 0.0,
            code_query_token_count=code_result.query_token_count,
            code_match_candidate_count=code_result.match_candidate_count,
            code_top_10_order_changed=code_result.top_10_order_changed,
            code_top_10_membership_changed=code_result.top_10_membership_changed,
            code_top_100_order_changed=code_result.top_100_order_changed,
            code_top_100_membership_changed=code_result.top_100_membership_changed,
            neighbour_seed_limit=CODE_NEIGHBOUR_SEED_LIMIT if code_aware else 0,
            neighbour_seed_count=code_result.neighbour_seed_count,
            neighbour_activated_seed_count=code_result.neighbour_activated_seed_count,
            neighbour_ineligible_seed_count=code_result.neighbour_ineligible_seed_count,
            neighbour_restored_count=code_result.neighbour_restored_count,
            neighbour_invalid_count=code_result.neighbour_invalid_count,
            code_duplicate_output_count=code_result.duplicate_output_count,
            atomic_rescue_attempted=atomic_state.attempted,
            atomic_rescue_active=atomic_state.active,
            atomic_rescue_fallback=atomic_state.fallback,
            atomic_rescue_candidate_available=atomic_state.candidate_available,
        )

    def _atomic_rescue_transform(
        self,
        store: PgVectorStore,
        binding: AtomicRescueBinding | None,
        state: _AtomicRescueState,
    ) -> _AtomicRescueTransforms | None:
        if binding is None:
            return None
        if binding.mode not in {"off", "shadow", "active"} or binding.placement not in {
            "dense",
            "fused",
        }:
            state.fallback = True
            return None
        state.attempted = binding.mode == "active"
        if binding.mode != "active":
            return None
        try:
            if not binding.artifact_root:
                raise AtomicRescueArtifactError("active atomic rescue requires an artifact root")
            artifact = load_atomic_rescue_artifact(
                resolve_atomic_rescue_manifest(
                    binding.artifact_root,
                    binding.generation_id,
                    scope_id=binding.scope_id,
                    corpus_fingerprint=binding.corpus_fingerprint,
                )
            )
            artifact.assert_lineage(
                generation_id=binding.generation_id,
                calibration_id=binding.calibration_id,
                pipeline_fingerprint=binding.pipeline_fingerprint,
                corpus_fingerprint=binding.corpus_fingerprint,
                embedder=self._embedder,
            )
        except AtomicRescueArtifactError:
            state.fallback = True
            return None
        state.active = True

        def transform(vector: list[float], dense: list[ScoredChunk]) -> list[ScoredChunk]:
            def load(chunk_id: str, score: float) -> ScoredChunk | None:
                chunk = store.chunks_by_ids([chunk_id]).get(chunk_id)
                return ScoredChunk(chunk, score) if chunk is not None else None

            result = insert_atomic_rescue_dense(artifact, vector, dense, load)
            state.candidate_available = True
            return result

        def place_after_fusion(
            vector: list[float], dense: list[ScoredChunk], ranked: list[ScoredChunk]
        ) -> list[ScoredChunk]:
            def load(chunk_id: str, score: float) -> ScoredChunk | None:
                chunk = store.chunks_by_ids([chunk_id]).get(chunk_id)
                return ScoredChunk(chunk, score) if chunk is not None else None

            result = insert_atomic_rescue_fused(artifact, vector, dense, ranked, load)
            state.candidate_available = True
            return result

        return _AtomicRescueTransforms(dense=transform, fused=place_after_fusion)

    def apply_graph_sidecar(
        self,
        store: PgVectorStore,
        query: str,
        run: RetrievalRun,
    ) -> RetrievalRun:
        """Apply grounded graph promotion while preserving the raw candidate membership."""
        query_vector = embed_query(self._embedder, query)
        dense = store.query_dense(query_vector, k=self._candidate_k)
        lexical = store.query_sparse(query, k=self._candidate_k, vec=query_vector)
        by_id = {hit.chunk.id: hit for hit in [*dense, *lexical]}
        fused = _rrf(
            ([hit.chunk.id for hit in dense], [hit.chunk.id for hit in lexical])
        )
        sidecar_hits = [
            replace(by_id[chunk_id], score=fused[chunk_id])
            for chunk_id in sorted(fused, key=lambda item: (-fused[item], item))
        ]
        baseline_ids = [hit.chunk.id for hit in run.hits]
        promotion = promote_grounded_raw(
            run.hits,
            sidecar_hits,
            superseded_sidecar_ids=store.explicit_superseded_chunk_ids(),
            historical=bool(_HISTORICAL.search(query)),
        )
        served_ids = [hit.chunk.id for hit in promotion.hits]
        return replace(
            run,
            hits=promotion.hits,
            graph_attempted=True,
            graph_profile=GRAPH_PROFILE,
            graph_relation_hits=promotion.relation_hits,
            graph_candidate_count=promotion.candidate_count,
            graph_promoted_count=promotion.promoted_count,
            graph_invalid_relation_count=promotion.invalid_relation_count,
            graph_top_10_order_changed=served_ids[:10] != baseline_ids[:10],
            graph_top_100_membership_changed=(
                set(served_ids[:100]) != set(baseline_ids[:100])
            ),
        )

    @staticmethod
    def graph_fallback(run: RetrievalRun) -> RetrievalRun:
        """Mark a sidecar failure without changing any baseline hit or score."""
        return replace(
            run,
            graph_attempted=True,
            graph_fallback=True,
            graph_profile=GRAPH_PROFILE,
        )


_KIND_PRIORITY = {
    "successful repair": 8,
    "architectural decision": 7,
    "procedure": 6,
    "validation": 5,
    "root cause": 4,
    "constraint": 3,
    "failed attempt": 2,
    "symptom": 1,
    "repository fact": 0,
    "raw": -1,
}

_FEATURE_KIND_PRIORITY = {
    "architectural decision": 8,
    "procedure": 7,
    "repository fact": 6,
    "constraint": 5,
    "validation": 4,
    "successful repair": 3,
    "root cause": 2,
    "failed attempt": 1,
    "symptom": 0,
    "raw": -1,
}

_BUGFIX_KIND_PRIORITY = {
    "successful repair": 8,
    "root cause": 7,
    "failed attempt": 6,
    "validation": 5,
    "symptom": 4,
    "procedure": 3,
    "constraint": 2,
    "repository fact": 1,
    "architectural decision": 0,
    "raw": -1,
}


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKENS.findall(value)}


def _near_duplicate(left: str, right: str) -> bool:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return left.strip().casefold() == right.strip().casefold()
    return len(a & b) / len(a | b) >= 0.9


def _event_time(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def pack_evidence(
    hits: Sequence[ScoredChunk],
    query: str,
    *,
    top_k: int,
    char_budget: int,
    historical: bool = False,
    include_raw: bool = True,
    superseded_ids: frozenset[str] = frozenset(),
    task_type: str = "unknown",
) -> list[SearchItem]:
    """Pack stored evidence only, with compiled preferences and raw rescue."""
    historical = historical or bool(_HISTORICAL.search(query))
    superseded: set[str] = set(superseded_ids)
    for hit in hits:
        refs = hit.chunk.metadata.get("supersedes", [])
        if isinstance(refs, list):
            superseded.update(str(ref) for ref in refs)

    query_tokens = _tokens(query)
    priorities = {
        "feature": _FEATURE_KIND_PRIORITY,
        "bugfix": _BUGFIX_KIND_PRIORITY,
    }.get(task_type, _KIND_PRIORITY)
    ranked = list(enumerate(hits))
    ranked.sort(
        key=lambda pair: (
            -len(query_tokens & _tokens(pair[1].chunk.text)),
            -priorities.get(str(pair[1].chunk.metadata.get("kind", "raw")), -1),
            pair[0],
        )
    )
    selected: list[SearchItem] = []
    selected_text: list[str] = []
    used_chars = 0
    limit = min(top_k, MAX_ITEMS)
    core_count = min(len(ranked), max(1, (2 * limit + 2) // 3))
    core = ranked[:core_count]
    diverse: list[tuple[int, ScoredChunk]] = []
    deferred: list[tuple[int, ScoredChunk]] = []
    seen_sessions = {
        str(hit.chunk.metadata.get("source_session_id", ""))
        for _, hit in core
        if hit.chunk.metadata.get("source_session_id")
    }
    for pair in ranked[core_count:]:
        session = str(pair[1].chunk.metadata.get("source_session_id", ""))
        if session and session not in seen_sessions:
            diverse.append(pair)
            seen_sessions.add(session)
        else:
            deferred.append(pair)
    for _, hit in [*core, *diverse, *deferred]:
        metadata = hit.chunk.metadata
        if not historical and hit.chunk.id in superseded:
            continue
        record_type = str(metadata.get("record_type", "raw"))
        if record_type == "raw" and not include_raw:
            continue
        text = hit.chunk.text
        if any(_near_duplicate(text, existing) for existing in selected_text):
            continue
        if used_chars + len(text) > char_budget:
            continue
        selected.append(
            SearchItem(
                id=hit.chunk.id,
                content=text,
                created_at=_event_time(metadata.get("event_time")),
                source=hit.chunk.source,
                session_id=str(metadata.get("source_session_id", "")),
                kind=str(metadata.get("kind", record_type)),
                score=float(hit.score),
            )
        )
        selected_text.append(text)
        used_chars += len(text)
        if len(selected) >= limit:
            break
    return selected


def render_full_evidence(
    hits: Sequence[ScoredChunk],
    query: str,
    *,
    top_k: int,
    superseded_ids: frozenset[str] = frozenset(),
) -> list[SearchItem]:
    """Render the uncompressed retrieval order while retaining structural supersession safety."""
    historical = bool(_HISTORICAL.search(query))
    selected: list[SearchItem] = []
    for hit in hits:
        if not historical and hit.chunk.id in superseded_ids:
            continue
        metadata = hit.chunk.metadata
        record_type = str(metadata.get("record_type", "raw"))
        selected.append(
            SearchItem(
                id=hit.chunk.id,
                content=hit.chunk.text,
                created_at=_event_time(metadata.get("event_time")),
                source=hit.chunk.source,
                session_id=str(metadata.get("source_session_id", "")),
                kind=str(metadata.get("kind", record_type)),
                score=float(hit.score),
            )
        )
        if len(selected) >= top_k:
            break
    return selected


def render_multiview_evidence(
    hits: Sequence[ScoredChunk],
    query: str,
    *,
    top_k: int,
    superseded_ids: frozenset[str] = frozenset(),
    typed_head: int = 10,
) -> list[SearchItem]:
    """Keep the ranked typed head, then give unseen sessions a raw rescue slot."""
    historical = bool(_HISTORICAL.search(query))
    eligible = [
        hit
        for hit in hits
        if historical or hit.chunk.id not in superseded_ids
    ]
    limit = min(top_k, len(eligible))
    head = eligible[: min(typed_head, limit)]
    selected = list(head)
    selected_ids = {hit.chunk.id for hit in selected}
    represented_sessions = {
        str(hit.chunk.metadata.get("source_session_id", ""))
        for hit in selected
        if hit.chunk.metadata.get("source_session_id")
    }
    for hit in eligible[len(head) :]:
        metadata = hit.chunk.metadata
        session = str(metadata.get("source_session_id", ""))
        if metadata.get("record_type") != "raw" or not session or session in represented_sessions:
            continue
        selected.append(hit)
        selected_ids.add(hit.chunk.id)
        represented_sessions.add(session)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for hit in eligible[len(head) :]:
            if hit.chunk.id in selected_ids:
                continue
            selected.append(hit)
            selected_ids.add(hit.chunk.id)
            if len(selected) >= limit:
                break
    return render_full_evidence(
        selected,
        query,
        top_k=limit,
        superseded_ids=superseded_ids,
    )
