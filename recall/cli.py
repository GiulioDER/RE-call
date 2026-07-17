from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from recall.calibration import Calibration, from_samples, load_for, save
from recall.embeddings import HashingEmbedder
from recall.index import Indexer, chunk_code, chunk_text
from recall.store import PgVectorStore
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
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="index a folder of markdown or code")
    p_index.add_argument("path")
    p_index.add_argument(
        "--glob", default="**/*.md",
        help="file glob to index — e.g. '**/*.py' for code (auto-uses code chunking). Default: markdown.",
    )

    p_search = sub.add_parser("search", help="search the index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)

    sub.add_parser("demo", help="index corpus/ and run sample memory queries")
    sub.add_parser("code", help="index recall's own source and run sample code queries")

    p_cal = sub.add_parser(
        "calibrate",
        help="calibrate the abstention threshold for this embedder against labeled queries",
    )
    p_cal.add_argument("queries", help="JSON list of {query, answerable, relevant_ids} entries")
    p_cal.add_argument("--corpus", default=None, help="corpus dir (default: the built-in eval corpus)")
    p_cal.add_argument("--out", default=None, help="output path (default: calibration.json)")

    args = parser.parse_args(argv)
    embedder = _make_embedder(args.embedder)
    calibration = load_for(embedder.name)

    if args.cmd == "index":
        chunker = chunk_code if args.glob.endswith(".py") else chunk_text
        with PgVectorStore(args.dsn, dim=embedder.dim) as store:
            store.ensure_schema()
            stats = Indexer(store, embedder, chunker=chunker).index_path(args.path, glob=args.glob)
            print(f"indexed {stats.chunks} chunks from {stats.files} files")
    elif args.cmd == "search":
        with PgVectorStore(args.dsn, dim=embedder.dim) as store:
            store.ensure_schema()
            _print_result(
                trusted_search(store, embedder, args.query, k=args.k, calibration=calibration)
            )
    elif args.cmd == "demo":
        with PgVectorStore(args.dsn, dim=embedder.dim) as store:
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
