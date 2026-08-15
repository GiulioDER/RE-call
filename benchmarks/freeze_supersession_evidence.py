"""Retrieve the 11 probe rows ONCE and freeze them, so the index leaves the loop.

The supersession probe iterates: build an annotator, look at what it marked, adjust, look again.
Doing that against a live index would mean re-embedding and re-reranking on every pass, on a host
that is not always free, and it would mean the evidence could drift between arms without anyone
noticing. So retrieval happens exactly once, here, and everything downstream reads a digested
fixture.

⚠️ **The fixture is NOT committed.** It carries EnterpriseRAG-Bench document text, and this
repository is public. What is committed is its SHA-256, recorded in
`results/enterprise_rag/PREREGISTRATION-supersession-annotation.md`.

⚠️ **The digest pins the bytes THIS run produced. It does not promise a rerun reproduces them.**
An earlier version of this docstring claimed "byte-identical" regeneration and that claim was
wrong in at least three ways: the final hit order comes from a remote reranker whose model can
change under a stable name, all three store legs cut with `ORDER BY <metric> LIMIT k` and no id
tiebreaker over a corpus where ties are dense, and the dense candidate set depends on
`RECALL_HNSW_EF_SEARCH_MULTIPLIER` and any server-level `hnsw.ef_search`. The digest's job is to
prove the two ARMS read the same bytes, which it does; reproducing the retrieval is a different
and weaker property.

Run on the host that has the index:

    python -m benchmarks.freeze_supersession_evidence \\
        --questions EnterpriseRAG-Bench/questions.jsonl \\
        --dsn "$RECALL_DSN" --out evidence.json

Verify a fixture afterwards, anywhere, with no index and no network:

    python -m benchmarks.freeze_supersession_evidence --verify evidence.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from recall._env import load_dotenv
from recall.embeddings import resolve_embedder
from recall.guards import DEFAULT_GAP_THRESHOLD
from recall.store import PgVectorStore

from benchmarks.enterprise_rag import (
    DEFAULT_RERANK_DOCUMENT_CHARS,
    EnterpriseQuestion,
    DEFAULT_SPLADE_MODEL,
    _expected_docs,
    _question_type,
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

#: How `evidence_digest` canonicalises, stated as data so a third party can reimplement it without
#: reading this source. Recorded in the fixture's provenance for the same reason.
DIGEST_ALGORITHM = (
    "sha256 of json.dumps(payload['evidence'], indent=1, sort_keys=True, ensure_ascii=False) "
    "encoded utf-8, with no trailing newline"
)


def canonical_evidence_json(evidence: Mapping[str, object]) -> str:
    """The exact bytes the digest covers.

    A module-level function rather than two statements inside `main`, because the digest is
    published in a pre-registration as a check anyone must be able to run. When this lived inline
    there was no way to recompute it from the written file short of guessing the canonicalisation,
    which made the pre-registered "digest unchanged between arms" check unimplementable.

    `_provenance` is deliberately OUTSIDE it: provenance carries a timestamp, so digesting it would
    make every regeneration a different fixture and the unchanged check unfalsifiable.
    """
    return json.dumps(evidence, indent=1, sort_keys=True, ensure_ascii=False)


def evidence_digest(evidence: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_evidence_json(evidence).encode("utf-8")).hexdigest()


def verify(path: Path) -> int:
    """Recompute a fixture's digest from its own bytes and compare with what it claims."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload["_provenance"]["evidence_sha256"]
    actual = evidence_digest(payload["evidence"])
    rows = len(payload["evidence"])
    print(f"rows      : {rows}")
    print(f"claimed   : {claimed}")
    print(f"recomputed: {actual}")
    if claimed != actual:
        print("MISMATCH: this fixture's evidence does not hash to the digest it carries")
        return 1
    print("OK")
    return 0


def _checkpoint_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".partial.jsonl")


def retrieval_fingerprint(args: argparse.Namespace) -> str:
    """Every setting that changes what gets retrieved, as one comparable string.

    ⚠️ This exists because resuming is the dangerous half of checkpointing. The first real run of
    this script died on a Voyage batch ceiling after freezing four rows; the obvious next move was
    to lower `--rerank-document-chars` and `--resume`, which would have produced a fixture whose
    first four rows were retrieved under one configuration and whose last seven were retrieved
    under another. Nothing would have complained, the provenance would have recorded only the
    SECOND configuration, and the mixed fixture would have been the substrate for every number
    that followed.
    """
    return json.dumps(
        {
            "table": args.table,
            "tenant": args.tenant,
            "embedder": args.embedder,
            "k": args.k,
            "candidate_k": args.candidate_k,
            "sparse_backend": args.sparse_backend,
            "splade_model": args.splade_model,
            "sparse_device": args.sparse_device,
            "reranker": args.reranker,
            "rerank_document_chars": args.rerank_document_chars,
            "gap_threshold": args.gap_threshold,
        },
        sort_keys=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, help="check a fixture and exit; needs no index")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--dsn")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--force", action="store_true", help="overwrite an existing fixture")
    parser.add_argument("--resume", action="store_true", help="reuse rows already checkpointed")
    parser.add_argument("--table", default="ber_voy_lex_12k_full")
    parser.add_argument("--tenant", default="enterprise-rag-voyage-lexical-chunk12k-full")
    parser.add_argument("--embedder", default="voyage:voyage-4-large")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--sparse-backend", default="both")
    parser.add_argument("--reranker", default="voyage:rerank-2.5")
    parser.add_argument("--rerank-document-chars", type=int, default=DEFAULT_RERANK_DOCUMENT_CHARS)
    parser.add_argument("--gap-threshold", type=float, default=DEFAULT_GAP_THRESHOLD)
    parser.add_argument("--splade-model", default=DEFAULT_SPLADE_MODEL)
    parser.add_argument("--sparse-device", default="cpu")
    parser.add_argument("--accept-noncommercial-splade-license", action="store_true")
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.verify)
    for name in ("questions", "dsn", "out"):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required unless --verify is given")

    # First statement of real work, matching every other entry point in this repo. Without it
    # `resolve_embedder("voyage:...")` raises for a missing VOYAGE_API_KEY on the very host whose
    # keys live in a `.env`, which is the only host this script is meant to run on.
    load_dotenv()

    if args.out.exists() and not args.force:
        raise SystemExit(
            f"{args.out} exists. The fixture is the only copy and its digest is published, so "
            f"overwriting it silently would strand the pre-registration. Pass --force to replace."
        )

    wanted = {q for group in ROWS.values() for q in group}
    loaded = load_questions(args.questions)
    by_id: dict[str, EnterpriseQuestion] = {}
    for question in loaded:
        if question.question_id not in wanted:
            continue
        previous = by_id.get(question.question_id)
        if previous is not None and previous.question != question.question:
            raise SystemExit(
                f"{question.question_id} appears twice with different text; refusing to guess"
            )
        by_id[question.question_id] = question
    missing = wanted - set(by_id)
    if missing:
        raise SystemExit(f"questions file is missing {sorted(missing)}; refusing a partial freeze")

    checkpoint = _checkpoint_path(args.out)
    fingerprint = retrieval_fingerprint(args)
    frozen: dict[str, object] = {}
    if args.resume and checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("retrieval_fingerprint") != fingerprint:
                raise SystemExit(
                    f"{checkpoint.name} holds {row['question_id']} retrieved under DIFFERENT "
                    f"settings. Resuming would mix two configurations into one fixture, and the "
                    f"provenance would record only the second. Delete the checkpoint and "
                    f"re-freeze every row.\n  checkpoint: {row.get('retrieval_fingerprint')}"
                    f"\n  now       : {fingerprint}"
                )
            frozen[row["question_id"]] = row["evidence"]
        print(f"resumed {len(frozen)} row(s) from {checkpoint.name}", flush=True)

    embedder = resolve_embedder(args.embedder)
    reranker = build_reranker(args.reranker, max_document_chars=args.rerank_document_chars)
    sparse_encoder = build_sparse_encoder(
        args.sparse_backend,
        model=args.splade_model,
        accept_noncommercial_license=args.accept_noncommercial_splade_license,
        device=args.sparse_device,
    )
    group_of = {q: g for g, ids in ROWS.items() for q in ids}

    with PgVectorStore(
        args.dsn, dim=embedder.dim, table=args.table, tenant=args.tenant
    ) as store:
        # `check_schema`, NOT `ensure_schema`. The latter applies migrations, and this path reads
        # a shared index that other work depends on; a freeze must not mutate it.
        store.check_schema()
        with checkpoint.open("a", encoding="utf-8", newline="\n") as sink:
            for question_id in sorted(wanted):
                if question_id in frozen:
                    continue
                question = by_id[question_id]
                doc_ids, hits, gap_warning = retrieve_docs(
                    store,
                    embedder,
                    question.question,
                    k=args.k,
                    candidate_k=args.candidate_k,
                    sparse_backend=args.sparse_backend,
                    sparse_encoder=sparse_encoder,
                    reranker=reranker,
                    gap_threshold=args.gap_threshold,
                )
                if len(hits) < args.k:
                    # Same stance as the missing-question refusal above. An empty or short bundle
                    # would make both arms score 0.0 for a structural reason, indistinguishable
                    # from a genuine null.
                    raise SystemExit(
                        f"{question_id} retrieved {len(hits)} hits, fewer than k={args.k}. "
                        f"Refusing to freeze a short bundle."
                    )
                row = {
                    "group": group_of[question_id],
                    "question": question.question,
                    "question_type": _question_type(question),
                    # SORTED: `_expected_docs` returns a set, and set iteration order would
                    # put nondeterministic bytes straight into the digest.
                    "expected_doc_ids": sorted(_expected_docs(question)),
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
                frozen[question_id] = row
                # Checkpointed BEFORE the next paid call. One transient failure on row 7 used to
                # discard six completed retrievals with nothing on disk.
                sink.write(
                    json.dumps(
                        {
                            "question_id": question_id,
                            "retrieval_fingerprint": fingerprint,
                            "evidence": row,
                        }
                    )
                    + "\n"
                )
                sink.flush()
                print(f"froze {question_id}: {len(hits)} hits, {len(doc_ids)} docs", flush=True)

    if set(frozen) != wanted:
        raise SystemExit(f"froze {sorted(frozen)}, wanted {sorted(wanted)}")

    digest = evidence_digest(frozen)
    payload = {
        "_provenance": {
            "generated_at": datetime.now(UTC).isoformat(),
            "table": args.table,
            "tenant": args.tenant,
            "embedder": args.embedder,
            "k": args.k,
            "candidate_k": args.candidate_k,
            "sparse_backend": args.sparse_backend,
            "splade_model": args.splade_model,
            "sparse_device": args.sparse_device,
            "reranker": args.reranker,
            "rerank_document_chars": args.rerank_document_chars,
            "gap_threshold": args.gap_threshold,
            "hnsw_ef_search_multiplier": os.environ.get("RECALL_HNSW_EF_SEARCH_MULTIPLIER"),
            "rows": {group: list(ids) for group, ids in ROWS.items()},
            "evidence_sha256": digest,
            "digest_algorithm": DIGEST_ALGORITHM,
            "note": (
                "NOT committed: carries benchmark corpus text. The digest is what is published, "
                "and it proves the two arms read the same bytes rather than that a rerun "
                "reproduces them."
            ),
        },
        "evidence": frozen,
    }
    # Atomic. A kill or a full disk mid-write would otherwise leave truncated JSON after the
    # expensive retrieval had already been paid for.
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                   newline="\n")
    os.replace(tmp, args.out)
    checkpoint.unlink(missing_ok=True)
    print()
    print(f"wrote {args.out} for {len(frozen)} rows")
    print(f"frozen_evidence_digest: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
