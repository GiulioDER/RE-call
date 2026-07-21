from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from recall.calibration import Calibration, from_samples, load_for, save
from recall.embeddings import HashingEmbedder
from recall.index import Indexer, chunk_code, chunk_text
from recall.lint import DEFAULT_GLOB
from recall.store import PgVectorStore, warn_if_insecure_dsn
from recall.trust import trusted_search
from recall.types import TrustedResult

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")


def _make_embedder(name: str):
    if name == "hashing":
        return HashingEmbedder(dim=64)
    if name == "fastembed":
        from recall.embeddings import FastEmbedEmbedder

        return FastEmbedEmbedder()
    raise SystemExit(f"unknown embedder: {name}")


def _print_result(result: TrustedResult) -> None:
    flags = []
    if result.abstained:
        flags.append("ABSTAIN")
    if result.gap_warning:
        flags.append("GAP")
    if result.staleness.stale:
        flags.append("STALE")
    print(f"[{' '.join(flags) if flags else 'ok'}] query={result.query!r}")
    if result.reason:
        print(f"  reason: {result.reason}")
    for h in result.hits:
        preview = h.chunk.text.replace("\n", " ")[:52]
        name = h.provenance.file or h.chunk.source
        redirect = f" -> use {h.validity.superseded_by}" if h.validity.superseded_by else ""
        print(
            f"  {h.verdict:<14} conf={h.confidence:.2f} cos={h.cosine:.3f}  "
            f"{name}{redirect}  {preview!r}"
        )


def _run_queries(
    store: PgVectorStore, embedder, queries: list[str], calibration: Calibration | None
) -> None:
    for q in queries:
        _print_result(trusted_search(store, embedder, q, calibration=calibration))
        print()


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):  # clean UTF-8 output on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="recall")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--embedder", default="fastembed", choices=["fastembed", "hashing"])
    parser.add_argument(
        "--table", default="chunks",
        help="table to read/write (default: chunks). Use a throwaway name to keep an "
             "experiment out of your real memory index.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="index a folder of markdown or code")
    p_index.add_argument("path")
    p_index.add_argument(
        "--glob", default=DEFAULT_GLOB,
        help="file glob to index — e.g. '**/*.py' for code (auto-uses code chunking). Default: markdown.",
    )

    p_search = sub.add_parser("search", help="search the index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument(
        "--entail", action="store_true",
        help="opt-in entailment stage: demote hits that don't answer the query "
             "(requires recall[entail]; downloads the QNLI judge on first use)",
    )

    sub.add_parser("demo", help="index corpus/ and run sample memory queries")
    sub.add_parser("code", help="index recall's own source and run sample code queries")

    p_lint = sub.add_parser(
        "lint",
        help="check a corpus's supersession graph for broken/missing edges (no DB needed)",
    )
    p_lint.add_argument("path")
    p_lint.add_argument("--glob", default=DEFAULT_GLOB)
    p_lint.add_argument(
        "--semantic", action="store_true",
        help="also run the retrieval-based MISSING-edge check: flag memos highly similar to a "
             "prior closed decision they don't reference (needs the DB + embedder; opt-in)",
    )
    p_lint.add_argument(
        "--threshold", type=float, default=None,
        help="cosine threshold for --semantic (default: the calibrated abstention threshold "
             "for this embedder; must be calibrated per embedder — see FINDINGS section 2)",
    )

    p_cal = sub.add_parser(
        "calibrate",
        help="calibrate the abstention threshold for this embedder against labeled queries",
    )
    p_cal.add_argument("queries", help="JSON list of {query, answerable, relevant_ids} entries")
    p_cal.add_argument("--corpus", default=None, help="corpus dir (default: the built-in eval corpus)")
    p_cal.add_argument("--out", default=None, help="output path (default: calibration.json)")

    args = parser.parse_args(argv)
    warn_if_insecure_dsn(args.dsn)  # loud stderr note if default creds target a remote host

    if args.cmd == "lint":  # pure filesystem check — no embedder, no DB
        from recall.lint import lint_corpus

        try:
            issues = lint_corpus(args.path, glob=args.glob)
        except FileNotFoundError as exc:
            print(f"recall lint: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        for i in issues:
            print(f"{i.level:<8} {i.code:<26} {i.file}: {i.message}")
        errors = sum(1 for i in issues if i.level == "error")
        warnings = len(issues) - errors

        chains = []
        if args.semantic:  # opt-in retrieval-based missing-edge check (needs DB + embedder)
            from recall.semantic_lint import semantic_lint

            emb = _make_embedder(args.embedder)
            cal = load_for(emb.name)
            thr = args.threshold if args.threshold is not None else (
                cal.threshold if cal else 0.70
            )
            chains = semantic_lint(args.dsn, emb, args.path, threshold=thr, glob=args.glob)
            for c in chains:
                print(f"warning  unlinked-chain             {c.new_memo}: highly similar "
                      f"(cos={c.cosine:.2f}) to closed decision {c.prior!r} it does not "
                      f"reference — add `supersedes: {c.prior}`?")
            warnings += len(chains)

        print(f"{errors} errors, {warnings} warnings")
        if errors:
            raise SystemExit(1)
        return

    embedder = _make_embedder(args.embedder)
    calibration = load_for(embedder.name)

    if args.cmd == "index":
        chunker = chunk_code if args.glob.endswith(".py") else chunk_text
        with PgVectorStore(args.dsn, dim=embedder.dim, table=args.table) as store:
            store.ensure_schema()
            stats = Indexer(store, embedder, chunker=chunker).index_path(args.path, glob=args.glob)
            print(f"indexed {stats.chunks} chunks from {stats.files} files")
    elif args.cmd == "search":
        entail_judge = None
        if args.entail:
            from recall.entailment import QnliEntailmentJudge

            entail_judge = QnliEntailmentJudge()
        with PgVectorStore(args.dsn, dim=embedder.dim, table=args.table) as store:
            store.ensure_schema()
            _print_result(
                trusted_search(store, embedder, args.query, k=args.k, calibration=calibration,
                               entailment=entail_judge)
            )
    elif args.cmd == "demo":
        with PgVectorStore(args.dsn, dim=embedder.dim, table=args.table) as store:
            store.ensure_schema()
            stats = Indexer(store, embedder).index_path("corpus")
            print(f"indexed {stats.chunks} chunks from {stats.files} files\n")
            _run_queries(store, embedder, [
                "what did we decide about caching?",
                "do we inject retrieved context into the prompt?",
                "how many requests per second can a client make?",
                "how do we handle penguins on mars?",
            ], calibration)
    elif args.cmd == "code":
        # index recall's own package source (content-agnostic engine, code-aware chunking)
        src = Path(__file__).resolve().parent
        with PgVectorStore(args.dsn, dim=embedder.dim, table="recall_code") as store:
            store.ensure_schema()
            stats = Indexer(store, embedder, chunker=chunk_code).index_path(src, glob="**/*.py")
            print(f"indexed {stats.chunks} code chunks from {stats.files} files\n")
            _run_queries(store, embedder, [
                "where is reciprocal rank fusion implemented?",
                "how are embeddings stored in postgres?",
                "how does cross-encoder reranking reorder hits?",
            ], calibration)
    elif args.cmd == "calibrate":
        import json

        from recall.calibration import ENV_VAR, _resolve_path
        from recall.eval.calibrate import calibrate as run_calibration

        # fail fast on a malformed or one-class queries file: a calibration built without both
        # answerable AND unanswerable samples is degenerate, and saving it silently would arm a
        # meaningless threshold
        try:
            entries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read queries file {args.queries!r}: {exc}") from exc
        labeled = [q for q in entries if isinstance(q, dict) and not q.get("trust")]
        if not all("query" in q and "answerable" in q for q in labeled):
            raise SystemExit(
                "queries file entries need 'query' and 'answerable' keys "
                "(see recall/eval/queries.json for the format)"
            )
        if not any(q["answerable"] for q in labeled) or not any(
            not q["answerable"] for q in labeled
        ):
            raise SystemExit(
                "queries file needs at least one answerable AND one unanswerable entry — "
                "a one-class file cannot calibrate an abstention threshold"
            )

        measured = run_calibration(
            args.dsn,
            embedder,
            corpus_dir=Path(args.corpus) if args.corpus else None,
            queries_path=Path(args.queries),
        )
        cal = from_samples(
            embedder.name, measured.answerable_max_cos, measured.unanswerable_max_cos
        )
        path = save(cal, args.out)
        print(f"embedder:  {embedder.name}")
        print(f"threshold: {cal.threshold} (scale {cal.scale})")
        print(f"FCR at default 0.50: {measured.fcr_at_050:.2f} -> at calibrated: "
              f"{measured.fcr_at_suggested:.2f}")
        print(f"saved: {path}")
        if args.out and Path(args.out).resolve() != _resolve_path(None).resolve():
            print(f"note: searches load {_resolve_path(None)} by default — set "
                  f"{ENV_VAR}={path} for this file to be used")


if __name__ == "__main__":
    main()
