from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from recall.embeddings import Embedder
from recall.store import PgVectorStore
from recall.types import Chunk

DEFAULT_MAX_CHARS = 800  # target chunk size in characters; paragraphs are packed up to this

# A chunker turns one document's text into a list of chunk strings.
Chunker = Callable[[str], list[str]]


def _pack(blocks: list[str], max_chars: int) -> list[str]:
    """Greedily pack pre-split blocks into chunks no larger than max_chars (blocks are kept whole)."""
    chunks: list[str] = []
    buf = ""
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if buf and len(buf) + len(b) + 2 > max_chars:
            chunks.append(buf)
            buf = b
        else:
            buf = f"{buf}\n\n{b}" if buf else b
    if buf:
        chunks.append(buf)
    return chunks


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split prose/markdown into chunks on blank lines, packing paragraphs up to max_chars."""
    return _pack(text.split("\n\n"), max_chars)


def chunk_code(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Chunk source code at top-level ``def`` / ``class`` (and decorator) boundaries.

    A heuristic, not an AST parse: a line at column 0 starting a new ``def``/``class``/``@``
    begins a new block, so a function keeps its whole body (and methods stay with their class).
    Module preamble (imports, constants) forms the first block. Blocks are then packed up to
    max_chars. Better retrieval than blank-line packing on code, without a language-specific parser.
    """
    blocks: list[str] = []
    cur: list[str] = []
    for line in text.split("\n"):
        starts_block = line[:1] not in (" ", "\t") and (
            line.startswith(("def ", "class ", "@", "async def "))
        )
        if cur and starts_block:
            blocks.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return _pack(blocks, max_chars)


@dataclass(frozen=True)
class IndexStats:
    files: int
    chunks: int


class Indexer:
    def __init__(
        self, store: PgVectorStore, embedder: Embedder, chunker: Chunker = chunk_text
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._chunker = chunker

    def index_path(self, path: str | Path, glob: str = "**/*.md") -> IndexStats:
        """Index a markdown file, or a directory of them, into the vector store.

        Chunk ids are deterministic (md5 of ``<file>:<ordinal>``), so re-indexing an
        unchanged file overwrites its chunks in place. Known limitation: if a file shrinks
        (produces fewer chunks than before), the now-orphaned trailing chunks are not
        garbage-collected — drop and re-index the table for a clean slate.
        """
        root = Path(path)
        files = sorted(root.glob(glob)) if root.is_dir() else [root]
        all_chunks: list[Chunk] = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            for i, ct in enumerate(self._chunker(text)):
                cid = hashlib.md5(f"{f}:{i}".encode("utf-8")).hexdigest()
                all_chunks.append(
                    Chunk(id=cid, source=str(f), text=ct, metadata={"file": f.name, "ord": i})
                )
        if all_chunks:
            embeddings = self._embedder.embed([c.text for c in all_chunks])
            self._store.upsert(all_chunks, embeddings)
        return IndexStats(files=len(files), chunks=len(all_chunks))
