"""Maximum quality hosted retrieval and deterministic evidence packing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
import re

from recall.embeddings import Embedder
from recall.rerank import Reranker
from recall.store import PgVectorStore
from recall.types import ScoredChunk
from recall_aml.models import SearchItem


CANDIDATE_WIDTH = 100
MAX_ITEMS = 12
_HISTORICAL = re.compile(r"\b(previous|formerly|before|histor|old|earlier|used to)\b", re.I)
_TOKENS = re.compile(r"[A-Za-z0-9_./:\\-]+")


@dataclass(frozen=True)
class RetrievalRun:
    hits: list[ScoredChunk]
    reranker_fallback: bool
    superseded_ids: frozenset[str]


def _rrf(rankings: Sequence[Sequence[str]], constant: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant + rank)
    return scores


class HostedRetriever:
    def __init__(self, embedder: Embedder, reranker: Reranker, *, candidate_k: int = 100) -> None:
        if candidate_k != CANDIDATE_WIDTH:
            raise ValueError("hosted-quality candidate width is fixed at 100")
        self._embedder = embedder
        self._reranker = reranker
        self._candidate_k = candidate_k

    def search(self, store: PgVectorStore, query: str, facets: Sequence[str]) -> RetrievalRun:
        rankings: list[list[str]] = []
        by_id: dict[str, ScoredChunk] = {}
        dense_scores: dict[str, float] = {}
        variants = [query, *list(facets)[:4]]
        vectors = self._embedder.embed(variants)
        if len(vectors) != len(variants):
            raise RuntimeError("query embedder returned the wrong number of vectors")
        for variant, vector in zip(variants, vectors, strict=True):
            dense = store.query_dense(vector, k=self._candidate_k)
            lexical = store.query_sparse(variant, k=self._candidate_k, vec=vector)
            rankings.extend(([hit.chunk.id for hit in dense], [hit.chunk.id for hit in lexical]))
            for hit in dense:
                by_id.setdefault(hit.chunk.id, hit)
                dense_scores.setdefault(hit.chunk.id, hit.score)
            for hit in lexical:
                by_id.setdefault(hit.chunk.id, hit)
                dense_scores.setdefault(hit.chunk.id, hit.score)
        fused = _rrf(rankings)
        ordered = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))
        hits = [
            replace(by_id[chunk_id], score=dense_scores.get(chunk_id, by_id[chunk_id].score))
            for chunk_id in ordered
        ]
        try:
            hits = self._reranker.rerank(query, hits)
            fallback = False
        except Exception:  # BROAD-CATCH: deterministic fused-order serving fallback
            fallback = True
        return RetrievalRun(
            hits=hits,
            reranker_fallback=fallback,
            superseded_ids=store.explicit_superseded_chunk_ids(),
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
) -> list[SearchItem]:
    """Pack stored evidence only, with compiled preferences and raw rescue."""
    historical = historical or bool(_HISTORICAL.search(query))
    superseded: set[str] = set(superseded_ids)
    for hit in hits:
        refs = hit.chunk.metadata.get("supersedes", [])
        if isinstance(refs, list):
            superseded.update(str(ref) for ref in refs)

    query_tokens = _tokens(query)
    ranked = list(enumerate(hits))
    ranked.sort(
        key=lambda pair: (
            -len(query_tokens & _tokens(pair[1].chunk.text)),
            -_KIND_PRIORITY.get(str(pair[1].chunk.metadata.get("kind", "raw")), -1),
            pair[0],
        )
    )
    selected: list[SearchItem] = []
    selected_text: list[str] = []
    used_chars = 0
    limit = min(top_k, MAX_ITEMS)
    diverse: list[tuple[int, ScoredChunk]] = []
    deferred: list[tuple[int, ScoredChunk]] = []
    seen_sessions: set[str] = set()
    for pair in ranked:
        session = str(pair[1].chunk.metadata.get("source_session_id", ""))
        if session and session not in seen_sessions:
            diverse.append(pair)
            seen_sessions.add(session)
        else:
            deferred.append(pair)
    for _, hit in [*diverse, *deferred]:
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
