"""Retrieve the 11 probe rows ONCE and freeze them, so the index leaves the loop.

The supersession probe iterates: build an annotator, look at what it marked, adjust, look again.
Doing that against a live index would mean re-embedding and re-reranking on every pass, on a host
that is not always free, and it would mean the evidence could drift between arms without anyone
noticing. So retrieval happens exactly once, here, and everything downstream reads a digested
fixture.

⚠️ **The fixture is NOT committed.** It carries EnterpriseRAG-Bench document text, and this
repository is public. What is committed is its SHA-256, recorded in
`results/enterprise_rag/PREREGISTRATION-supersession-annotation.md`. Anyone holding the corpus and
the index can regenerate a byte-identical fixture with this script; nobody else needs it, and
publishing it would redistribute a benchmark corpus that is not ours to redistribute.

Run on the host that has the index:

    python -m benchmarks.freeze_supersession_evidence --dsn "$RECALL_DSN" --out evidence.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from recall.embeddings import resolve_embedder
from recall.guards import DEFAULT_GAP_THRESHOLD
from recall.store import PgVectorStore

from benchmarks.enterprise_rag import (
    DEFAULT_SPLADE_MODEL,
    build_reranker,
    build_sparse_encoder,
    load_questions,
    retrieve_docs,
)

#: The four supersession rows the experiment predicts on, the six coverage rows that are its
#: negative control, and the one attribution row that is reported and not predicted. Written out
#: rather than derived, because the split is the experiment's design and a query that silently
#: returned a different set would invalidate it without erroring.
ROWS = {
    "A_supersession": ("qst_0418", "qst_0419", "qst_0420", "qst_0425"),
    "B_coverage": ("qst_0310", "qst_0320", "qst_0325", "qst_0332", "qst_0333", "qst_0336"),
    "C_attribution": ("qst_0413",),
}


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--table", default="ber_voy_lex_12k_full")
    parser.add_argument("--tenant", default="enterprise-rag-voyage-lexical-chunk12k-full")
    parser.add_argument("--embedder", default="voyage:voyage-4-large")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--sparse-backend", default="both")
    parser.add_argument("--reranker", default="voyage:rerank-2.5")
    parser.add_argument("--max-context-chars", type=int, default=12_000)
    parser.add_argument("--splade-model", default=DEFAULT_SPLADE_MODEL)
    parser.add_argument("--sparse-device", default="cpu")
    parser.add_argument("--accept-noncommercial-splade-license", action="store_true")
    args = parser.parse_args(argv)

    wanted = {q for group in ROWS.values() for q in group}
    questions = [q for q in load_questions(args.questions) if q.question_id in wanted]
    missing = wanted - {q.question_id for q in questions}
    if missing:
        raise SystemExit(f"questions file is missing {sorted(missing)}; refusing a partial freeze")

    embedder = resolve_embedder(args.embedder)
    reranker = build_reranker(args.reranker, max_document_chars=4_000)
    sparse_encoder = build_sparse_encoder(
        args.sparse_backend,
        model=args.splade_model,
        accept_noncommercial_license=args.accept_noncommercial_splade_license,
        device=args.sparse_device,
    )
    group_of = {q: g for g, ids in ROWS.items() for q in ids}
    frozen: dict[str, object] = {}
    with PgVectorStore(
        args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant
    ) as store:
        for question in sorted(questions, key=lambda q: q.question_id):
            doc_ids, hits, gap_warning = retrieve_docs(
                store,
                embedder,
                question.question,
                k=args.k,
                candidate_k=args.candidate_k,
                sparse_backend=args.sparse_backend,
                sparse_encoder=sparse_encoder,
                reranker=reranker,
                gap_threshold=DEFAULT_GAP_THRESHOLD,
            )
            frozen[question.question_id] = {
                "group": group_of[question.question_id],
                "question": question.question,
                "question_type": question.raw.get("question_type"),
                "expected_doc_ids": list(question.raw.get("expected_doc_ids") or []),
                "document_ids": doc_ids,
                "gap_warning": gap_warning,
                "hits": [
                    {
                        "chunk_id": hit.chunk.id,
                        "source": hit.chunk.source,
                        "doc_id": str(hit.chunk.metadata.get("doc_id") or ""),
                        "title": str(hit.chunk.metadata.get("title") or ""),
                        "score": hit.score,
                        "text": hit.chunk.text,
                    }
                    for hit in hits
                ],
            }
            print(f"froze {question.question_id}: {len(hits)} hits, {len(doc_ids)} docs", flush=True)

    # The digest covers the EVIDENCE only, deliberately excluding the provenance block below.
    # Provenance carries a timestamp, so digesting it would make every regeneration a different
    # fixture and the "unchanged between arms" check would be unfalsifiable.
    evidence_json = json.dumps(frozen, indent=1, sort_keys=True, ensure_ascii=False)
    digest = _digest(evidence_json)

    payload = {
        "_provenance": {
            "generated_at": datetime.now(UTC).isoformat(),
            "table": args.table,
            "tenant": args.tenant,
            "embedder": args.embedder,
            "k": args.k,
            "candidate_k": args.candidate_k,
            "sparse_backend": args.sparse_backend,
            "reranker": args.reranker,
            "max_context_chars": args.max_context_chars,
            "rows": {group: list(ids) for group, ids in ROWS.items()},
            "evidence_sha256": digest,
            "note": "NOT committed: carries benchmark corpus text. The digest is what is published.",
        },
        "evidence": frozen,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print()
    print(f"wrote {args.out} for {len(frozen)} rows")
    print(f"frozen_evidence_digest: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
