"""Frozen retrieval primitives for the promoted CAMBench Coding candidate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math
import re

from recall.types import Chunk, ScoredChunk


EMBEDDING_PROFILE = "voyage-code-4-v1"
BM25_PROFILE = "canonical-bm25-k1-1.5-b0.75-v1"
WORD_WINDOW_SIZE = 160
WORD_WINDOW_STRIDE = 120
BM25_K1 = 1.5
BM25_B = 0.75

STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
        "this", "to", "with", "was", "were", "will", "would", "can", "could", "should",
        "must", "not", "no", "if", "then", "than", "so", "what", "which", "when",
        "where",
    ]
)
_TOKEN = re.compile(r"[a-z][a-z0-9_]*")


def tokenize(text: str) -> list[str]:
    """Use the exact tokenizer frozen by the successful Code4 screen."""
    return [
        word
        for word in _TOKEN.findall(text.lower())
        if len(word) > 1 and word not in STOPWORDS
    ]


def word_windows(text: str, *, size: int = WORD_WINDOW_SIZE, stride: int = WORD_WINDOW_STRIDE) -> list[str]:
    """Return overlapping windows over raw whitespace separated words."""
    if size < 1 or stride < 1:
        raise ValueError("word window size and stride must be positive")
    words = text.split()
    if not words:
        return [""]
    windows: list[str] = []
    for start in range(0, len(words), stride):
        windows.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return windows


def rank_bm25_chunks(chunks: Sequence[Chunk], query: str, *, k: int) -> list[ScoredChunk]:
    """Rank positive scoring chunks with the experiment's fixed Okapi BM25."""
    if k < 1:
        raise ValueError("k must be positive")
    materialized = list(chunks)
    token_counts = [Counter(tokenize(chunk.text)) for chunk in materialized]
    lengths = [sum(counts.values()) for counts in token_counts]
    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    containing: Counter[str] = Counter()
    for counts in token_counts:
        containing.update(counts.keys())
    count = len(materialized)
    inverse_document_frequency = {
        term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in containing.items()
    }
    query_terms = tokenize(query)
    ranked: list[ScoredChunk] = []
    for chunk, counts, length in zip(materialized, token_counts, lengths, strict=True):
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            normalizer = 1.0 - BM25_B + BM25_B * length / (average_length or 1.0)
            denominator = frequency + BM25_K1 * normalizer
            score += (
                inverse_document_frequency.get(term, 0.0)
                * frequency
                * (BM25_K1 + 1.0)
                / denominator
            )
        if score > 0.0:
            ranked.append(ScoredChunk(chunk, score))
    return sorted(ranked, key=lambda hit: (-hit.score, hit.chunk.id))[:k]
