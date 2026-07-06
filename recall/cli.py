from __future__ import annotations

import argparse
import os

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="recall")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--embedder", default="fastembed", choices=["fastembed", "hashing"])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="index a folder of markdown")
    p_index.add_argument("path")

    p_search = sub.add_parser("search", help="search the index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)

    sub.add_parser("demo", help="index corpus/ and run sample queries")

    args = parser.parse_args(argv)
    embedder = _make_embedder(args.embedder)
    store = PgVectorStore(args.dsn, dim=embedder.dim)
    store.ensure_schema()

    if args.cmd == "index":
        stats = Indexer(store, embedder).index_path(args.path)
        print(f"indexed {stats.chunks} chunks from {stats.files} files")
    elif args.cmd == "search":
        _print_result(HybridRetriever(store, embedder).search(args.query, k=args.k))
    elif args.cmd == "demo":
        stats = Indexer(store, embedder).index_path("corpus")
        print(f"indexed {stats.chunks} chunks from {stats.files} files\n")
        retriever = HybridRetriever(store, embedder)
        for q in [
            "what did we decide about caching?",
            "do we inject retrieved context into the prompt?",
            "how do we handle penguins on mars?",
        ]:
            _print_result(retriever.search(q))
            print()


if __name__ == "__main__":
    main()
