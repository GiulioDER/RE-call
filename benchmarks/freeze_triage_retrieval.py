"""One retrieval pass over the benchmark, capturing the pool AND the top-k, for the triage probe.

Pre-registered at `results/enterprise_rag/PREREGISTRATION-retrieval-triage.md` (`c8828db`), and
extended for `results/enterprise_rag/PREREGISTRATION-triage-mechanism.md` (`1a153cd`).

One retrieval per question gives both halves: the ordering's first 8 entries are what the `k=8`
configuration would have returned, and the whole list is the pool. So the expensive part, one
encode plus three index queries, is paid once per question rather than twice, and the top-8 is a
prefix of the pool by construction rather than by a second call that might order differently.

⚠️ **Two quantities that `search()` discards are captured here**, because the first run's fixture
could not answer what its own winning feature was reading. `search()` orders by the RRF fused
score and then reports each hit's DENSE cosine, so a fixture built from its output records a curve
that is not the ranking criterion. `benchmarks.fusion_detail.fuse` recovers the fused score and
each leg's rank off the same `_Legs` seam, at no extra retrieval cost. Its ranked output is pinned
to `search()`'s by `tests/test_bench_fusion_detail.py`.

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

from benchmarks.console import use_utf8_output
from benchmarks.enterprise_rag import (
    DEFAULT_SPLADE_MODEL,
    _expected_docs,
    _question_type,
    build_reranker,
    build_sparse_encoder,
    load_questions,
)
from benchmarks.freeze_supersession_evidence import evidence_digest
from benchmarks.fusion_detail import dense_gap_warning, fuse

#: Bumped whenever a row gains or loses a field. It is part of the retrieval fingerprint, so a
#: `.partial.jsonl` written by an earlier capture is REFUSED rather than resumed into a fixture
#: where some rows carry fused scores and others do not. Every other fingerprint input was
#: unchanged between capture 1 and capture 2, so without this the mixing would have been silent.
CAPTURE_SCHEMA = 2


def main(argv: list[str] | None = None) -> int:
    use_utf8_output()  # argparse prints this module's docstring; cp1252 cannot
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
            "capture_schema": CAPTURE_SCHEMA,
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
    # ⛔ Refused, not warned about. This capture reads the FUSED ordering directly off the legs,
    # which is what makes the fused score and the per-leg ranks available at all; a reranker would
    # reorder the pool afterwards and simply not be applied here. Accepting the flag and ignoring
    # it would produce a fixture whose provenance claims a reranker that never ran.
    if reranker is not None:
        raise SystemExit(
            f"--reranker {args.reranker!r}: this capture records the fused ordering and cannot "
            "apply a reranker. Pass --reranker none, or use a capture that reranks."
        )
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
            # Always None: anything else was refused above. Passed anyway so the retriever is
            # built the same way it is everywhere else, rather than by a shape unique to here.
            reranker=reranker,
            gap_threshold=DEFAULT_GAP_THRESHOLD,
            sparse_backend=args.sparse_backend, sparse_encoder=sparse_encoder,
            retrieval_profile=f"enterprise-rag:{args.sparse_backend}",
        )
        # ⚠️ `"w"` without `--resume`, not `"a"`. Appending to a checkpoint this run will not read
        # builds a file holding two fingerprints, which is unresumable: a later `--resume` hits
        # the first foreign line and exits, discarding however many hours of THIS capture's rows
        # came after it. Adding `capture_schema` to the fingerprint makes that mixture likelier,
        # so the fix ships with it rather than after it.
        with checkpoint.open("a" if args.resume else "w", encoding="utf-8", newline="\n") as sink:
            for i, question in enumerate(questions, 1):
                if question.question_id in rows:
                    continue
                legs = retriever._retrieve_legs(question.question, source=None)
                fused = fuse(legs)[: args.pool_k]
                ranked = [
                    {
                        "doc_id": str(f.hit.chunk.metadata.get("doc_id") or f.hit.chunk.source),
                        # The DENSE cosine, unchanged from capture 1 so the two fixtures compare.
                        "score": f.hit.score,
                        # The quantity that actually ordered this list. New in capture 2.
                        "fused_score": f.fused_score,
                        # Zero-based per-leg rank, `null` where the leg did not return the chunk.
                        # Order is `fusion_detail.LEG_NAMES`: dense, lexical, learned.
                        "ranks": list(f.ranks),
                    }
                    for f in fused
                ]
                # From the DENSE candidate scores, and deduplicated the way `search()` does it.
                # Restating that here with a comment claiming equivalence is how the two drift,
                # so the mirroring lives beside `fuse()` and is pinned by the same test.
                gap = dense_gap_warning(legs, DEFAULT_GAP_THRESHOLD)
                row = {
                    # ⚠️ Written because `analyse_triage` and `explore_triage_signal` compute
                    # query-text features from it. It was absent, they read it with a default,
                    # and two registered features were silently constant across all 500 rows.
                    "question": question.question,
                    "question_type": _question_type(question),
                    "expected_doc_ids": sorted(_expected_docs(question)),
                    "gap_warning": gap,
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
            # Both, and keyed off the schema: a capture-2 fixture answers the mechanism
            # registration as well as the original one, and a fixture that names only the
            # registration it no longer serves is provenance that reads as correct.
            "preregistration_commits": [
                "c8828db65f0577aa7b999e5bb4fee46fe7515e61",  # retrieval triage, capture 1
                *(["1a153cd3ae746786af3eff228d2fabbd5098fa9e"]  # triage mechanism, capture 2
                  if CAPTURE_SCHEMA >= 2 else []),
            ],
            "retrieval_fingerprint": fingerprint,
            "capture_schema": CAPTURE_SCHEMA,
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
