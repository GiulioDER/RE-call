"""Okapi BM25 over the indexed chunks — the anchor every retrieval number needs.

A hit@5 of 0.705 means nothing on its own. It is only a result relative to what a *boring*
baseline scores on the same corpus, the same chunks and the same questions, and BM25 is the
baseline the IR literature has used for thirty years. Without it, "hybrid dense+sparse reaches
0.705" is unfalsifiable: the reader cannot tell whether the embedding stack earned that number
or whether keyword matching would have got there alone.

**Dependency-free on purpose.** `rank_bm25` would be one line, but the baseline that anchors
every published number in this repo should not be a package that can silently change its
scoring between releases. The formula is 40 lines and is written out below.

Scoring — the Lucene/Robertson variant::

    score(D, Q) = Σ  IDF(q) · f(q,D) · (k1 + 1) / (f(q,D) + k1 · (1 − b + b · |D| / avgdl))
                 q∈Q

    IDF(q) = ln(1 + (N − n(q) + 0.5) / (n(q) + 0.5))

`k1=1.5`, `b=0.75` are the standard defaults, and are deliberately NOT tuned here: a baseline
tuned on the same questions it is scored against stops being a baseline and becomes a second
system with an unfair advantage. The `+1` inside the IDF log is what keeps a term appearing in
more than half the corpus at a small positive weight rather than a negative one.

**Two asymmetries against this baseline, stated because they cut in opposite directions.**

- Tokenisation here is lowercase-and-split-on-non-alphanumeric with **no stemming**, while the
  Postgres sparse leg (`store.query_sparse`) runs a full `english` text-search configuration
  with stemming and stopword removal. So the two lexical arms are *not* the same algorithm with
  a different score, and BM25 is handicapped on morphology ("indexes" vs "indexing").
- BM25 sees the whole corpus for every query, while the fused arms rerank a `candidate_k`-sized
  pool. That favours BM25.

Reporting both lexical arms is how those cancel out into something readable: if BM25 and the
Postgres leg land close, neither asymmetry mattered much on this corpus.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from datetime import timedelta

from recall.store import PgVectorStore
from recall.types import Chunk, RetrievalResult, ScoredChunk, StalenessReport

#: Standard Okapi defaults. Not tuned — see the module docstring.
K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric. No stemming, no stopword list.

    Stopwords are left in deliberately: BM25's IDF already drives a term appearing in most
    documents to nearly zero weight, so a hand-maintained list would be a second mechanism
    doing the same job, with its own choices to defend.
    """
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over `(doc_id, text)` pairs, with no database and no store type.

    The formula lives here and `BM25Retriever` delegates to it. Two copies of a scoring function
    is how a baseline and the thing it anchors quietly stop agreeing — see this module's docstring
    on why the formula is written out rather than imported from a package.

    Ties break by `doc_id` ascending. That is not tidiness: `rank()` output is frozen into a
    released benchmark manifest, and a tie broken by insertion order would make two builds of the
    same corpus into two different benchmarks.
    """

    def __init__(self, docs: Iterable[tuple[str, str]], k1: float = K1, b: float = B) -> None:
        self._k1 = k1
        self._b = b
        self._doc_ids: list[str] = []
        self._len: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        df: Counter[str] = Counter()

        for doc_id, text in docs:
            tokens = tokenize(text)
            tf = Counter(tokens)
            i = len(self._doc_ids)
            self._doc_ids.append(doc_id)
            self._len.append(len(tokens))
            df.update(tf.keys())
            for term, freq in tf.items():
                self._postings.setdefault(term, []).append((i, freq))

        n = len(self._doc_ids)
        self._avgdl = (sum(self._len) / n) if n else 0.0
        self._idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def __len__(self) -> int:
        return len(self._doc_ids)

    @property
    def doc_ids(self) -> list[str]:
        return list(self._doc_ids)

    def score(self, query: str) -> list[float]:
        """Per-document BM25 score for `query`, in corpus order."""
        terms = tokenize(query)
        scores = [0.0] * len(self._doc_ids)
        if not terms or not self._avgdl:
            return scores
        for term in terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, f in self._postings.get(term, ()):
                norm = 1.0 - self._b + self._b * (self._len[i] / self._avgdl)
                scores[i] += idf * (f * (self._k1 + 1.0)) / (f + self._k1 * norm)
        return scores

    def rank(self, query: str) -> list[tuple[str, float]]:
        """Every document, best first. Ties by `doc_id` ascending — see the class docstring."""
        scored = list(zip(self._doc_ids, self.score(query)))
        return sorted(scored, key=lambda ds: (-ds[1], ds[0]))


class BM25Retriever:
    """Lexical-only retrieval over an in-memory index of the store's chunks.

    Exposes the same `search(query, k)` signature as `HybridRetriever` so the evaluation
    harness can score it as one more arm with nothing special-cased.

    Built from `store.iter_chunks()`, i.e. the SAME chunks the dense and sparse legs search —
    identical text, identical chunk boundaries. A baseline that chunked differently would be
    measuring the chunker as well as the ranker.
    """

    def __init__(self, store: PgVectorStore, k1: float = K1, b: float = B) -> None:
        self._chunks: list[Chunk] = list(store.iter_chunks())
        # One formula, two callers. The chunk id is positional here because `search` maps back by
        # index; `BM25Index` only needs the ids to be unique and orderable.
        self._index = BM25Index(
            ((str(i), chunk.text) for i, chunk in enumerate(self._chunks)), k1=k1, b=b
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def score(self, query: str) -> list[float]:
        """Per-chunk BM25 score for `query`, in corpus order."""
        return self._index.score(query)

    def search(self, query: str, k: int = 5, source: str | None = None) -> RetrievalResult:
        """Top-`k` chunks by BM25.

        `source` filters after scoring — this is a benchmark baseline, not a serving path, so
        it optimises for being obviously correct rather than for latency.

        The returned `RetrievalResult` carries `gap_warning=False` and a non-stale report.
        Both fields are **meaningless for this arm** and are present only because the harness
        takes one result type: gap detection is defined on dense cosine, which a lexical
        ranker does not produce, and there is no "I don't know" story here at all. That is
        precisely the difference this baseline exists to make visible — BM25 always returns
        its top k, however irrelevant.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        scored: list[tuple[Chunk, float]] = list(zip(self._chunks, self.score(query)))
        if source is not None:
            scored = [(c, s) for c, s in scored if c.source == source]
        ranked = sorted(scored, key=lambda cs: cs[1], reverse=True)[:k]
        return RetrievalResult(
            query=query,
            # `score` here is a BM25 score, NOT a cosine. It is unbounded above and not
            # comparable across queries, so it must never be fed to a calibrated threshold.
            hits=[ScoredChunk(chunk=c, score=s) for c, s in ranked],
            gap_warning=False,
            staleness=StalenessReport(
                stale=False, newest_indexed_at=None, age=None, max_age=timedelta(days=2)
            ),
        )
