#!/usr/bin/env python
"""Score (query, passage) pairs with the pinned cross-encoder. Runs on a GPU box, no database.

    python scripts/score_pairs.py --queries queries.jsonl --docs docs.jsonl \
        --pairs pairs.jsonl --output scores.jsonl --batch-size 256

This is the expensive half of reranking and nothing else. `CrossEncoderReranker.rerank` is
    scores = model.predict([(query, text), ...])
    order  = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
so the ORDERING stays in RE-call and only the scoring moves. That keeps the offload exact rather
than approximate, provided the model and revision match the pin, which is why both are defaults
here and both are recorded in the output header.

NOTE: no DSN, no credentials. Same reason as encode_sparse.py: the only stage that wants rented
hardware is also the only stage that runs on a machine nobody trusts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Kept in step with recall/rerank.py. Duplicated rather than imported because this script runs
# on a box that has no RE-call checkout, and a scoring run under a different revision would
# produce plausible numbers that silently do not match what RE-call itself would compute.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True, help="JSONL of {qid, text}")
    parser.add_argument("--docs", type=Path, required=True, help="JSONL of {doc_id, text}")
    parser.add_argument("--pairs", type=Path, required=True, help="JSONL of {qid, doc_id}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--revision", default=DEFAULT_RERANKER_REVISION)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N pairs")
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    queries = {str(r["qid"]): str(r["text"]) for r in read_jsonl(args.queries)}
    docs = {str(r["doc_id"]): str(r["text"]) for r in read_jsonl(args.docs)}
    pairs = [(str(r["qid"]), str(r["doc_id"])) for r in read_jsonl(args.pairs)]
    if args.limit is not None:
        pairs = pairs[: args.limit]

    missing_q = {q for q, _ in pairs if q not in queries}
    missing_d = {d for _, d in pairs if d not in docs}
    if missing_q or missing_d:
        # Fail loudly. A missing text silently scored as "" would return a real number for a
        # pair that was never actually compared, and nothing downstream could tell.
        raise SystemExit(
            f"{len(missing_q)} queries and {len(missing_d)} docs referenced by pairs are missing "
            f"their text; refusing to score pairs that cannot be compared"
        )

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps({
        "event": "start", "pairs": len(pairs), "queries": len(queries), "docs": len(docs),
        "model": args.model, "revision": args.revision, "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }), flush=True)

    model = CrossEncoder(args.model, revision=args.revision, device=device)

    started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as out:
        out.write(json.dumps({
            "_header": True, "model": args.model, "revision": args.revision,
            "device": device, "pairs": len(pairs),
        }) + "\n")
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            scores = model.predict([(queries[q], docs[d]) for q, d in batch])
            for (qid, doc_id), score in zip(batch, scores, strict=True):
                # float() not round(): the ordering is decided by these values, and rounding
                # can reorder near-ties differently from what RE-call would compute in-process.
                out.write(json.dumps({"qid": qid, "doc_id": doc_id, "score": float(score)}) + "\n")
            elapsed = time.perf_counter() - started
            done = min(start + args.batch_size, len(pairs))
            print(json.dumps({
                "event": "progress", "scored": done,
                "elapsed_s": round(elapsed, 1),
                "rate_per_s": round(done / elapsed, 1) if elapsed > 0 else None,
            }), flush=True)

    elapsed = time.perf_counter() - started
    print(json.dumps({
        "event": "done", "scored": len(pairs), "elapsed_s": round(elapsed, 1),
        "rate_per_s": round(len(pairs) / elapsed, 1) if elapsed > 0 else None,
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
