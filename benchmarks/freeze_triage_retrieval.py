"""One retrieval pass over the benchmark, capturing the pool AND the top-k, for the triage probe.

Pre-registered at `results/enterprise_rag/PREREGISTRATION-retrieval-triage.md` (`c8828db`).

One `search(k=pool_k)` per question gives both halves: the reranked ordering's first 8 entries are
what the shipped `k=8` configuration would have returned, and the whole list is the pool. So the
expensive part, one rerank of the fused candidate set, is paid once per question rather than
twice, and the top-8 is a prefix of the pool by construction rather than by a second call that
might order differently.

Checkpointed per question and resumable, with the retrieval fingerprint carried on every row: the
supersession freeze died on a provider batch ceiling partway through, and resuming under changed
settings would have silently mixed two configurations into one fixture.

⚠️ Not committed. Carries benchmark corpus identifiers and scores; the digest is published.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from recall._env import load_dotenv
from recall.embeddings import resolve_embedder
from recall.guards import DEFAULT_GAP_THRESHOLD
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore

from benchmarks.enterprise_rag import (
    DEFAULT_SPLADE_MODEL,
    _expected_docs,
    _question_type,
    build_reranker,
    build_sparse_encoder,
    load_questions,
)
from benchmarks.freeze_supersession_evidence import evidence_digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="pilot on the first N questions")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--table", default="ber_voy_lex_12k_full")
    parser.add_argument("--tenant", default="enterprise-rag-voyage-lexical-chunk12k-full")
    parser.add_argument("--embedder", default="voyage:voyage-4-large")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--pool-k", type=int, default=200)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--sparse-backend", default="both")
    parser.add_argument("--reranker", default="voyage:rerank-2.5")
    parser.add_argument("--rerank-document-chars", type=int, default=3900)
    parser.add_argument("--splade-model", default=DEFAULT_SPLADE_MODEL)
    parser.add_argument("--sparse-device", default="cpu")
    args = parser.parse_args(argv)
    load_dotenv()

    # Everything that changes WHAT IS RETRIEVED. The env var sets HNSW search depth, so a
    # resume under a different value mixes two candidate-pool depths into one fixture, which is
    # exactly what this guard exists to stop; it was recorded in provenance but not fingerprinted.
    fingerprint = json.dumps(
        {
            **{k: getattr(args, k) for k in
               ("table", "tenant", "embedder", "top_k", "pool_k", "candidate_k",
                "sparse_backend", "reranker", "rerank_document_chars", "sparse_device")},
            "questions": str(args.questions.resolve()),
            "limit": args.limit,
            "hnsw_ef_search_multiplier": os.environ.get("RECALL_HNSW_EF_SEARCH_MULTIPLIER"),
        },
        sort_keys=True,
    )
    checkpoint = args.out.with_suffix(args.out.suffix + ".partial.jsonl")
    rows: dict[str, object] = {}
    if args.resume and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("retrieval_fingerprint") != fingerprint:
                raise SystemExit(
                    f"{checkpoint.name} was built under different settings; delete it and restart"
                )
            rows[row["question_id"]] = row["row"]
        print(f"resumed {len(rows)}", flush=True)

    questions = load_questions(args.questions, limit=args.limit)
    embedder = resolve_embedder(args.embedder)
    reranker = build_reranker(args.reranker, max_document_chars=args.rerank_document_chars)
    sparse_encoder = build_sparse_encoder(
        args.sparse_backend, model=args.splade_model,
        accept_noncommercial_license=False, device=args.sparse_device,
    )

    with PgVectorStore(
        args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant
    ) as store:
        store.check_schema()
        retriever = HybridRetriever(
            store, embedder, candidate_k=args.candidate_k,
            reranker=reranker,  # type: ignore[arg-type]
            gap_threshold=DEFAULT_GAP_THRESHOLD,
            sparse_backend=args.sparse_backend, sparse_encoder=sparse_encoder,
            retrieval_profile=f"enterprise-rag:{args.sparse_backend}",
        )
        with checkpoint.open("a", encoding="utf-8", newline="\n") as sink:
            for i, question in enumerate(questions, 1):
                if question.question_id in rows:
                    continue
                result = retriever.search(question.question, k=args.pool_k)
                ranked = [
                    {
                        "doc_id": str(hit.chunk.metadata.get("doc_id") or hit.chunk.source),
                        "score": hit.score,
                    }
                    for hit in result.hits
                ]
                row = {
                    # ⚠️ Written because `analyse_triage` and `explore_triage_signal` compute
                    # query-text features from it. It was absent, they read it with a default,
                    # and two registered features were silently constant across all 500 rows.
                    "question": question.question,
                    "question_type": _question_type(question),
                    "expected_doc_ids": sorted(_expected_docs(question)),
                    "gap_warning": result.gap_warning,
                    "query_chars": len(question.question),
                    "ranked": ranked,
                }
                rows[question.question_id] = row
                sink.write(json.dumps(
                    {"question_id": question.question_id,
                     "retrieval_fingerprint": fingerprint, "row": row}) + "\n")
                sink.flush()
                if i % 10 == 0 or i == len(questions):
                    print(f"  {i}/{len(questions)}", flush=True)

    digest = evidence_digest(rows)
    payload = {
        "_provenance": {
            "generated_at": datetime.now(UTC).isoformat(),
            "preregistration_commit": "c8828db65f0577aa7b999e5bb4fee46fe7515e61",
            "retrieval_fingerprint": fingerprint,
            "hnsw_ef_search_multiplier": os.environ.get("RECALL_HNSW_EF_SEARCH_MULTIPLIER"),
            "n_questions": len(rows),
            "retrieval_sha256": digest,
            "note": "NOT committed: carries benchmark identifiers. The digest is published.",
        },
        "evidence": rows,
    }
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, args.out)
    checkpoint.unlink(missing_ok=True)
    print(f"\nwrote {args.out} for {len(rows)} questions")
    print(f"retrieval_sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
