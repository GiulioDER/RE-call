from __future__ import annotations

import argparse
import os
from pathlib import Path

from recall.embeddings import HashingEmbedder
from recall.index import Indexer, chunk_code, chunk_text
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.types import RetrievalResult

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")


def _make_embedder(name: str):
    if name == "hashing":
        return HashingEmbedder(dim=64)
    if name == "fastembed":
        from recall.embeddings import FastEmbedEmbedder

        return FastEmbedEmbedder()
    raise SystemExit(f"unknown embedder: {name}")


def _print_result(result: RetrievalResult) -> None:
    flags = []
    if result.gap_warning:
        flags.append("GAP")
    if result.staleness.stale:
        flags.append("STALE")
    print(f"[{' '.join(flags) if flags else 'ok'}] query={result.query!r}")
    for h in result.hits:
        preview = h.chunk.text.replace("\n", " ")[:70]
        print(f"  {h.score:.3f}  {h.chunk.source}  {preview!r}")


def _run_queries(store: PgVectorStore, embedder, queries: list[str]) -> None:
    retriever = HybridRetriever(store, embedder)
    for q in queries:
        _print_result(retriever.search(q))
        print()


def main(argv: list[str] | None = None) -> None:
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

    args = parser.parse_args(argv)
    embedder = _make_embedder(args.embedder)

    if args.cmd == "index":
        chunker = chunk_code if args.glob.endswith(".py") else chunk_text
        with PgVectorStore(args.dsn, dim=embedder.dim) as store:
            store.ensure_schema()
            stats = Indexer(store, embedder, chunker=chunker).index_path(args.path, glob=args.glob)
            print(f"indexed {stats.chunks} chunks from {stats.files} files")
    elif args.cmd == "search":
        with PgVectorStore(args.dsn, dim=embedder.dim) as store:
            store.ensure_schema()
            _print_result(HybridRetriever(store, embedder).search(args.query, k=args.k))
    elif args.cmd == "demo":
        with PgVectorStore(args.dsn, dim=embedder.dim) as store:
            store.ensure_schema()
            stats = Indexer(store, embedder).index_path("corpus")
            print(f"indexed {stats.chunks} chunks from {stats.files} files\n")
            _run_queries(store, embedder, [
                "what did we decide about caching?",
                "do we inject retrieved context into the prompt?",
                "how do we handle penguins on mars?",
            ])
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
            ])


if __name__ == "__main__":
    main()
