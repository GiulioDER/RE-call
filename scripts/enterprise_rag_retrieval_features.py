"""Capture non-gold retrieval features for selective EnterpriseRAG depth studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.enterprise_rag import (
    QueryCachedEmbedder,
    _doc_ids_from_hits,
    load_questions,
    retrieve_docs_with_diagnostics,
)
from recall.cache import EmbeddingCache
from recall.store import PgVectorStore
from recall.embeddings import resolve_embedder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--question-ids-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--embedder", default="voyage:voyage-4-large")
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--sparse-backend", default="lexical")
    parser.add_argument("--embedding-cache", type=Path)
    args = parser.parse_args()

    ids = {
        line.strip()
        for line in args.question_ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    questions = load_questions(args.questions, question_ids=ids)
    embedder = resolve_embedder(args.embedder)
    cache = EmbeddingCache(args.embedding_cache) if args.embedding_cache else None
    retrieval_embedder = QueryCachedEmbedder(embedder, cache) if cache else embedder
    rows: list[dict[str, Any]] = []
    try:
        with PgVectorStore(
            args.dsn,
            dim=embedder.dim,
            table=args.table,
            tenant=args.tenant,
        ) as store:
            store.ensure_schema()
            for question in questions:
                ids_k12, hits, gap, diagnostics = retrieve_docs_with_diagnostics(
                    store,
                    retrieval_embedder,
                    question.question,
                    k=12,
                    candidate_k=args.candidate_k,
                    sparse_backend=args.sparse_backend,
                    sparse_encoder=None,
                    reranker=None,
                    gap_threshold=0.5,
                )
                top8 = hits[:8]
                dense_scores = [float(hit.score) for hit in hits if isinstance(hit.score, (int, float))]
                rows.append(
                    {
                        "question_id": question.question_id,
                        "question_type": str(question.raw.get("question_type", "")),
                        "k8_document_ids": _doc_ids_from_hits(hits, k=8),
                        "k12_document_ids": ids_k12,
                        "max_dense_score": max(dense_scores) if dense_scores else None,
                        "eighth_hit_dense_score": (
                            float(top8[-1].score) if len(top8) == 8 else None
                        ),
                        "top_hit_dense_scores": dense_scores[:12],
                        "gap_warning": bool(gap),
                        "stage_ms": dict(diagnostics.stage_ms),
                    }
                )
    finally:
        if cache is not None:
            cache.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} retrieval feature rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
