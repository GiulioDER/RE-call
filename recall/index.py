from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from recall.embeddings import Embedder
from recall.frontmatter import parse_frontmatter, validity_bounds
from recall.store import PgVectorStore
from recall.types import Chunk

DEFAULT_MAX_CHARS = 800  # target chunk size in characters; paragraphs are packed up to this
DEFAULT_OVERLAP_CHARS = 80  # chars shared between adjacent pieces of a force-split oversized block

# A chunker turns one document's text into a list of chunk strings.
Chunker = Callable[[str], list[str]]


def _split_oversized(block: str, max_chars: int, overlap: int) -> list[str]:
    """Force-split a single block that exceeds max_chars into <=max_chars pieces.

    A block with no blank line (a log dump, a table, a wall of prose) would otherwise become one
    chunk larger than the embedder's token window and be SILENTLY truncated, losing its tail. We
    slide a max_chars window with a stride of ``max_chars - overlap`` so adjacent pieces share
    ``overlap`` characters — a fact straddling a cut then survives in both neighbours.
    """
    if len(block) <= max_chars:
        return [block]
    step = max(1, max_chars - overlap)
    pieces: list[str] = []
    start = 0
    n = len(block)
    while start < n:
        end = min(start + max_chars, n)
        pieces.append(block[start:end])
        if end >= n:
            break
        start += step
    return pieces


def _pack(
    blocks: list[str], max_chars: int, hard_split: bool = False, overlap: int = DEFAULT_OVERLAP_CHARS
) -> list[str]:
    """Greedily pack pre-split blocks into chunks no larger than max_chars.

    Blocks are kept whole when they fit. When ``hard_split`` is set, a block that is itself larger
    than max_chars is force-split (with overlap) before packing, so no single chunk can exceed the
    cap; without it an oversized block is preserved intact (the code chunker keeps functions whole).
    """
    prepared: list[str] = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if hard_split and len(b) > max_chars:
            prepared.extend(_split_oversized(b, max_chars, overlap))
        else:
            prepared.append(b)
    chunks: list[str] = []
    buf = ""
    for b in prepared:
        if buf and len(buf) + len(b) + 2 > max_chars:
            chunks.append(buf)
            buf = b
        else:
            buf = f"{buf}\n\n{b}" if buf else b
    if buf:
        chunks.append(buf)
    return chunks


def chunk_text(
    text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP_CHARS
) -> list[str]:
    """Split prose/markdown into chunks on blank lines, packing paragraphs up to max_chars.

    A paragraph longer than max_chars on its own is force-split with overlap so it is never
    handed to the embedder oversized (which would truncate it silently)."""
    return _pack(text.split("\n\n"), max_chars, hard_split=True, overlap=overlap)


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

        Re-indexing REPLACES each file's rows completely (delete then insert), so a file that
        shrinks — or withdraws a frontmatter claim like ``supersedes`` — leaves no stale
        chunks behind to poison retrieval or the supersession map.
        """
        root = Path(path)
        files = sorted(root.glob(glob)) if root.is_dir() else [root]
        # Identify each file by its ROOT-RELATIVE path, not its bare basename: two files with the
        # same basename in different directories (a/notes.md, b/notes.md) must not collide in the
        # supersession map or in provenance. Mirrors recall.lint's `rel` keying. A single-file
        # index has no root to relativize against, so it falls back to the basename.
        rel = {f: (f.relative_to(root).as_posix() if root.is_dir() else f.name) for f in files}
        all_chunks: list[Chunk] = []
        for f in files:
            # utf-8-sig: a BOM must not silently disable frontmatter parsing
            meta, body = parse_frontmatter(f.read_text(encoding="utf-8-sig"))
            try:
                validity_bounds(meta)  # fail fast on malformed dates, before anything is embedded
            except ValueError as exc:
                raise ValueError(f"{f}: {exc}") from exc
            for i, ct in enumerate(self._chunker(body)):
                cid = hashlib.md5(f"{f}:{i}".encode("utf-8")).hexdigest()
                all_chunks.append(
                    Chunk(
                        id=cid,
                        source=str(f),
                        text=ct,
                        metadata={"file": rel[f], "ord": i, **meta},
                    )
                )
        # embed BEFORE touching the store: if embedding fails, the old rows stay intact
        embeddings = self._embedder.embed([c.text for c in all_chunks]) if all_chunks else []
        self._store.replace_sources([str(f) for f in files], all_chunks, embeddings)
        return IndexStats(files=len(files), chunks=len(all_chunks))
