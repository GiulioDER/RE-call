from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from recall.embeddings import Embedder
from recall.store import PgVectorStore
from recall.types import Chunk


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Split text into chunks on blank lines, packing paragraphs up to max_chars."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if buf and len(buf) + len(p) + 2 > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


@dataclass(frozen=True)
class IndexStats:
    files: int
    chunks: int


class Indexer:
    def __init__(self, store: PgVectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def index_path(self, path: str | Path, glob: str = "**/*.md") -> IndexStats:
        root = Path(path)
        files = sorted(root.glob(glob)) if root.is_dir() else [root]
        all_chunks: list[Chunk] = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            for i, ct in enumerate(chunk_text(text)):
                cid = hashlib.md5(f"{f}:{i}".encode("utf-8")).hexdigest()
                all_chunks.append(
                    Chunk(id=cid, source=str(f), text=ct, metadata={"file": f.name, "ord": i})
                )
        if all_chunks:
            embeddings = self._embedder.embed([c.text for c in all_chunks])
            self._store.upsert(all_chunks, embeddings)
        return IndexStats(files=len(files), chunks=len(all_chunks))
