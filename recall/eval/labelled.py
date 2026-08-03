"""Evaluate against YOUR corpus and YOUR labelled questions.

Every other evaluation in this repo runs on a corpus this repo ships or generates. Both are
optimistic in ways that are now documented: the demo corpus is 14 documents, and generated
corpora cannot measure successor or abstention accuracy because every document is the same
sentence with a different opaque token.

This runner takes a real corpus and a hand-labelled question set, which is the only way to
measure the thing that actually matters — whether the system finds the right memory when someone
asks a real question in their own words.

**Ground truth is at FILE level, deliberately.** The shipped harness scores `file:ord`, which
requires the labeller to know which chunk holds the answer; get it wrong and a correct retrieval
scores zero. A human labelling their own corpus knows which *document* answers a question and
should not have to think about chunking.

Question format (JSON list) — a superset of what `recall calibrate` consumes::

    {"id": "q01", "query": "...", "answerable": true,  "relevant_files": ["memo.md"]}
    {"id": "u01", "query": "...", "answerable": false}

The labelled file is YOUR data. Keep it with your corpus, not in this repository.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recall.calibration import from_samples
from recall.embeddings import Embedder, embed_query, embedding_profile_id
from recall.eval.bm25 import BM25Retriever
from recall.eval.metrics import wilson_ci
from recall.index import Indexer
from recall.retriever import DEFAULT_CANDIDATE_K, HybridRetriever
from recall.store import PgVectorStore
from recall.trust import evaluate as trust_evaluate
from recall.trust import trusted_search
from recall.types import RetrievalResult, TrustedResult

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")

_log = logging.getLogger(__name__)


def _make_embedder(name: str) -> Embedder:
    if name == "hashing":
        from recall.embeddings import HashingEmbedder

        return HashingEmbedder(dim=64)
    if name == "voyage":
        from recall.embeddings import VoyageEmbedder

        return VoyageEmbedder()
    if name.startswith("st:"):
        # Any sentence-transformers model, including one this repo fine-tuned:
        #   --embedder st:finetune/model
        from recall.embeddings import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(name[3:])
    from recall.embeddings import FastEmbedEmbedder

    return FastEmbedEmbedder()


def _files_of(result: RetrievalResult | TrustedResult) -> list[str]:
    """Distinct source files behind a result's hits, best rank first."""
    out: list[str] = []
    for h in result.hits:
        f = h.chunk.metadata.get("file")
        if f and f not in out:
            out.append(f)
    return out


def _rate(flags: list[bool]) -> dict:
    lo, hi = wilson_ci(flags)
    return {
        "rate": round(sum(flags) / len(flags), 4) if flags else float("nan"),
        "ci": (round(lo, 4), round(hi, 4)),
        "n": len(flags),
    }


def evaluate(dsn: str, corpus: Path, questions: list[dict], embedder: Embedder, k: int = 5,
             rerank: bool = False, glob: str = "**/*.md",
             candidate_k: int = DEFAULT_CANDIDATE_K, table: str | None = None) -> dict:
    answerable = [q for q in questions if q.get("answerable")]
    unanswerable = [q for q in questions if not q.get("answerable")]

    # A named table is KEPT after the run; the anonymous default is dropped as before. The
    # distinction exists for corpora whose index costs more than the evaluation: LongMemEval's
    # merged-S index alone is ~6h39m of embedding (FINDINGS §10), and `Indexer` skips by content
    # hash, so a named table makes every later run — this harness again, or
    # `longmemeval_perq --master` — a reuse instead of a rebuild.
    keep = table is not None
    table = table or ("lab_" + uuid.uuid4().hex[:8])
    store = PgVectorStore(dsn, dim=embedder.dim, table=table)
    try:
        store.ensure_schema()
        t0 = time.perf_counter()
        stats = Indexer(store, embedder).index_path(corpus, glob=glob)
        index_s = time.perf_counter() - t0

        # Calibrate on THIS corpus: a threshold from another corpus's cosine regime does not
        # transfer (FINDINGS section 2), and an uncalibrated run would measure the default, not
        # the system. Fitted on half the questions, scored on the other half.
        fit, held = questions[::2], questions[1::2]
        retr = HybridRetriever(store, embedder, candidate_k=candidate_k)

        def top_cos(q: str) -> float:
            hits = store.query_dense(embed_query(embedder, q), k=1)
            return hits[0].score if hits else 0.0

        cal = from_samples(
            embedding_profile_id(embedder),
            [top_cos(q["query"]) for q in fit if q.get("answerable")],
            [top_cos(q["query"]) for q in fit if not q.get("answerable")],
        )

        answerable_held = [x for x in held if x.get("answerable")]

        def score_arm(
            retriever: BM25Retriever | HybridRetriever,
            collect: list[RetrievalResult] | None = None,
        ) -> dict:
            hits, reciprocal, latency, misses = [], [], [], []
            for q in answerable_held:
                t = time.perf_counter()
                res = retriever.search(q["query"], k=k)
                latency.append((time.perf_counter() - t) * 1000)
                if collect is not None:
                    collect.append(res)
                files = _files_of(res)
                want = set(q["relevant_files"])
                hits.append(any(f in want for f in files[:k]))
                rank = next((i for i, f in enumerate(files) if f in want), None)
                reciprocal.append(1.0 / (rank + 1) if rank is not None else 0.0)
                if rank is None:
                    # A miss is either a retrieval failure or a LABELLING one — on a corpus with
                    # many related memos, several documents may legitimately answer a question
                    # while the label names one. Reporting what came back lets the labeller tell
                    # those apart, which a bare rate cannot.
                    misses.append({"id": q["id"], "query": q["query"],
                                   "expected": sorted(want), "got": files[:k]})
            lat = sorted(latency)
            return {
                f"hit_at_{k}": _rate(hits),
                "mrr": round(statistics.mean(reciprocal), 4) if reciprocal else float("nan"),
                "latency_ms": {"p50": round(lat[len(lat) // 2], 1),
                               "p95": round(lat[int(0.95 * len(lat))], 1)} if lat else {},
                "misses": misses,
            }

        # Every arm runs against the SAME index, the same held-out questions and the same
        # calibration — only the ranking stage differs, so a delta is attributable.
        #
        # The three baselines are not decoration. `hybrid` on its own is a number with nothing
        # to compare it to; a reader cannot tell whether the embedding stack earned it or
        # whether keyword matching alone would have. BM25 is the thirty-year-old anchor, and
        # dense-only / sparse-only say which leg of the hybrid is carrying it.
        hybrid_results: list[RetrievalResult] = []
        arms = {
            "bm25": score_arm(BM25Retriever(store)),
            "dense": score_arm(
                HybridRetriever(store, embedder, use_sparse=False, candidate_k=candidate_k)
            ),
            "sparse": score_arm(
                HybridRetriever(store, embedder, use_dense=False, candidate_k=candidate_k)
            ),
            "hybrid": score_arm(retr, collect=hybrid_results),
        }
        if rerank:
            # Same index, same questions, same calibration — only the ranking stage differs, so
            # any delta is the reranker's and nothing else's.
            from recall.rerank import CrossEncoderReranker

            arms["hybrid+rerank"] = score_arm(
                HybridRetriever(
                    store, embedder, reranker=CrossEncoderReranker(), candidate_k=candidate_k
                )
            )

        # Unanswerable held questions are retrieved by no arm above, so they still need their own
        # trusted_search — with candidate_k so it shares the reported pool (EVAL-002).
        abstained = [
            trusted_search(store, embedder, q["query"], k=k, calibration=cal,
                           candidate_k=candidate_k).abstained
            for q in held if not q.get("answerable")
        ]
        # Answerable held questions were already retrieved for the `hybrid` arm; reuse those
        # RetrievalResults for the abstain decision instead of retrieving a second time (PERF-003).
        # `abstained` depends only on the hits + supersession + threshold — not the retriever's
        # gap_threshold — so at the same candidate_k this equals a fresh trusted_search.
        supersession, unresolved = store.supersession()
        now = datetime.now(timezone.utc)
        false_abstain = [
            trust_evaluate(res, supersession, cal, now, unresolved).abstained
            for res in hybrid_results
        ]

        # Completeness, as a recorded expected/actual pair rather than a bare assertion. A
        # Postgres reset mid-build leaves committed-but-fewer rows, so "the table exists" and
        # even "the table has rows" are both true of a half-built index. `labelled` repairs
        # itself on a re-run (a source with no stored content hash is re-indexed), so this is
        # reported rather than raised — but it must be reported, because the number is otherwise
        # indistinguishable from a complete run's.
        # `index_path` resolves the root and stores absolute paths, so the comparison has to
        # resolve too — otherwise a relative --corpus matches nothing and every run looks empty.
        corpus_root = Path(corpus).resolve()
        sources_indexed = sum(
            1 for s in store.source_content_hashes() if Path(s).is_relative_to(corpus_root)
        )
        # Annotated because the literal is all-numeric, so an inferred dict[str, float] rejects
        # the `table` name assigned below when --keep is set.
        corpus_report: dict[str, Any] = {"files": stats.files, "chunks": store.count(),
                                         "sources_expected": stats.files + stats.skipped,
                                         "sources_indexed": sources_indexed,
                                         "index_seconds": round(index_s, 1)}
        if sources_indexed != stats.files + stats.skipped:
            _log.warning(
                "index incomplete: %d source(s) in %s, expected %d — re-run to repair "
                "(indexing skips by content hash, so only the missing files are embedded)",
                sources_indexed, table, stats.files + stats.skipped,
            )
        if keep:
            # Only a table that outlives the run is worth naming: this is the value to hand to
            # `longmemeval_perq --master`.
            corpus_report["table"] = table
        return {
            "corpus": corpus_report,
            "questions": {"total": len(questions), "answerable": len(answerable),
                          "unanswerable": len(unanswerable), "held_out": len(held)},
            "candidate_k": candidate_k,
            "threshold": cal.threshold,
            "arms": arms,
            "abstention_accuracy": _rate(abstained),
            "false_abstain": _rate(false_abstain),
        }
    finally:
        if not keep:
            store.drop_table()
        store.close()


def main() -> None:
    ap = argparse.ArgumentParser(prog="recall.eval.labelled")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--embedder", default="fastembed",
                    help="fastembed | hashing | voyage | st:<model-or-path>")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--glob", default="**/*.md",
                    help="corpus file glob — e.g. '**/*.rst' for a PEP checkout")
    ap.add_argument("--rerank", action="store_true",
                    help="also score a cross-encoder arm from the SAME index")
    ap.add_argument("--candidate-k", type=int, default=20,
                    help="candidates each leg contributes before fusion; the pool a "
                         "reranker sees is their union (default: 20)")
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--table",
                    help="index into this named table and KEEP it after the run (default: an "
                         "anonymous table, dropped). Re-running against a kept table reuses "
                         "its rows by content hash instead of re-embedding; pass the same "
                         "name to longmemeval_perq --master. Must pair with the embedder "
                         "that built it — the store refuses a dimension mismatch.")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    missing = [q["id"] for q in questions
               if q.get("answerable") and not q.get("relevant_files")]
    if missing:
        raise SystemExit(f"answerable questions without relevant_files: {missing}")

    report = evaluate(args.dsn, Path(args.corpus), questions, _make_embedder(args.embedder),
                      k=args.k, rerank=args.rerank, glob=args.glob,
                      candidate_k=args.candidate_k, table=args.table)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
