"""Maximum quality hosted retrieval and deterministic evidence packing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
import re
import time

from recall.embeddings import Embedder, embed_query
from recall.rerank import Reranker
from recall.sparse import SparseEncoderProtocol
from recall.store import PgVectorStore
from recall.types import ScoredChunk
from recall_aml.models import SearchItem


CANDIDATE_WIDTH = 100
RRF_CONSTANT = 60
MAX_ITEMS = 12
_HISTORICAL = re.compile(r"\b(previous|formerly|before|histor|old|earlier|used to)\b", re.I)
_TOKENS = re.compile(r"[A-Za-z0-9_./:\\-]+")


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


def _rrf(rankings: Sequence[Sequence[str]], constant: int = RRF_CONSTANT) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant + rank)
    return scores


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
    ) -> RetrievalRun:
        if learned_sparse and self._sparse_encoder is None:
            raise RuntimeError("learned sparse retrieval has no encoder")
        rankings: list[list[str]] = []
        by_id: dict[str, ScoredChunk] = {}
        dense_scores: dict[str, float] = {}
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
            dense = store.query_dense(vector, k=self._candidate_k)
            lexical = store.query_sparse(variant, k=self._candidate_k, vec=vector)
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
        ordered = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))
        hits = [
            replace(by_id[chunk_id], score=dense_scores.get(chunk_id, by_id[chunk_id].score))
            for chunk_id in ordered
        ]
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
