"""Retrieve for the 65 generation tasks MTRAG's Task A set excludes, using the SAME arm.

WHY. MTRAG's retrieval split has 777 judged queries; the generation split has 842. The 65-task
difference is exactly the 55 UNANSWERABLE + 10 CONVERSATIONAL turns, dropped from retrieval because
no gold passage exists. RE-call therefore has no contexts for them, and Task C over RE-call's
contexts would hand the generator an EMPTY passage list on precisely the tasks where a correct
"I don't know" scores 1.0 on RL_F, RB_llm and RB_alg at once.

Measured on the completed benchmark-contexts run: those 65 currently earn RB_agg 0.3113. Scoring
them 1.0 for free would move the overall 0.3999 -> 0.4531, an unearned **+0.0532** — two thirds the
size of the largest genuine retrieval gain measured all day. So the fix is to make RE-call face the
same test everyone else faces: real retrieved passages for an unanswerable question.

IDENTITY IS THE WHOLE POINT. Everything is imported from `benchmarks.mtrag.run` rather than
reimplemented — the arm, the reranker builder, the encoder, the retry wrapper, the query
normalisation and the prediction row shape. A merely similar setup would produce 65 rows that look
like the other 777 and were not produced the same way.

⚠️ `query_text` strips the `|user|: ` prefix the release puts on every dev query. Left in, the
literal token reaches both the embedder and the sparse encoder and depresses the run with nothing
failing. Verified before writing this: the prefixed last-user-turn reproduces the released query
text for 777/777 shared tasks.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/recall-splade-cli")

from recall.embeddings import FastEmbedEmbedder  # noqa: E402
from recall.retriever import HybridRetriever  # noqa: E402
from recall.store import PgVectorStore  # noqa: E402

from benchmarks.mtrag.run import (  # noqa: E402
    ALL_ARMS,
    DOMAINS,
    EMBEDDER_MODEL,
    build_reranker,
    load_dsn,
    query_text,
    table_name,
    task_domain,
)
from benchmarks.mtrag.run import search_with_retry  # noqa: E402


def main(argv: list[str]) -> int:
    queries_path, out_path, arm_name, env_file = (
        Path(argv[0]), Path(argv[1]), argv[2], Path(argv[3])
    )
    arm = next(a for a in ALL_ARMS if a.name == arm_name)
    print(json.dumps({"event": "arm", **{k: v for k, v in vars(arm).items()}}), flush=True)

    rows = [json.loads(line) for line in queries_path.open(encoding="utf-8") if line.strip()]
    print(json.dumps({"event": "queries", "n": len(rows)}), flush=True)

    dsn = load_dsn(env_file)
    embedder = FastEmbedEmbedder(EMBEDDER_MODEL)
    reranker = build_reranker(arm) if arm.rerank else None
    sparse_encoder = None
    if arm.sparse_backend in ("splade", "both"):
        from recall.sparse import SpladeEncoder

        sparse_encoder = SpladeEncoder.from_pretrained("prithivida/Splade_PP_en_v1")

    stores = {
        d: PgVectorStore(dsn, embedder.dim, table=table_name("recall_mtrag_bge_v1", d))
        for d in DOMAINS
    }
    retrievers = {
        d: HybridRetriever(
            s, embedder, reranker=reranker, candidate_k=arm.candidate_k,
            use_dense=arm.use_dense, use_sparse=arm.use_sparse,
            sparse_backend=arm.sparse_backend, sparse_encoder=sparse_encoder,
        )
        for d, s in stores.items()
    }

    written = 0
    started = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            # `_text` carries the release-format prefixed query so `query_text` applies the SAME
            # strip the 777 got, instead of this script deciding what normalisation means.
            # `_domain` is passed through, not re-derived: DOMAIN_ALIASES keys on SHORT names and
            # the generation file uses long ELSER collection names, so `task_domain` would raise.
            # The mapping was derived empirically from the 777 shared tasks and is 1:1.
            task = {
                "task_id": row["task_id"],
                "conversation_id": row["task_id"].split("<::>")[0],
                "Collection": row["Collection"],
                "_domain": row["_domain"],
                "_text": row["text"],
                "input": row.get("input", []),
            }
            domain = task_domain(task)
            query = query_text(task, arm.query_mode)
            result, elapsed_ms, _ = search_with_retry(retrievers[domain], query, arm)
            total = len(result.hits)
            contexts = [
                {
                    "document_id": h.chunk.id,
                    "score": float(total - rank),
                    "cosine": h.score,
                    "text": h.chunk.text,
                    "title": h.chunk.metadata.get("title"),
                    "source": domain,
                }
                for rank, h in enumerate(result.hits)
            ]
            fh.write(json.dumps({
                "conversation_id": task["conversation_id"],
                "task_id": task["task_id"],
                "Collection": task["Collection"],
                "input": task["input"],
                "query_used": query,
                "contexts": contexts,
                "gap_warning": result.gap_warning,
            }, ensure_ascii=False) + "\n")
            written += 1
            if i % 10 == 0:
                print(json.dumps({"event": "progress", "done": i, "of": len(rows),
                                  "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)

    print(json.dumps({"event": "done", "written": written, "output": str(out_path),
                      "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
