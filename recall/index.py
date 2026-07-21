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
#: Characters of context repeated across a FORCED split (see `_split_block`). Never applies to
#: ordinary packing — those chunks fall on natural block boundaries and need no overlap.
DEFAULT_OVERLAP = 80
#: Overlap may repeat at most a quarter of a chunk. Beyond that, consecutive pieces are mostly
#: duplicated context: the index inflates and near-identical chunks crowd the top-k.
MAX_OVERLAP_DIVISOR = 4
#: A forced cut closer than max_chars/this to the piece start is ignored in favour of the cap —
#: it would emit a sliver whose embedding row carries almost no content.
MIN_PIECE_DIVISOR = 8

# A chunker turns one document's text into a list of chunk strings.
Chunker = Callable[[str], list[str]]


def _split_block(block: str, max_chars: int, overlap: int) -> list[str]:
    """Force-split one oversized block into pieces of at most max_chars, on whitespace.

    Breaking mid-word would corrupt the tokens the embedder sees, so the cut walks back to
    the last space/newline in range. A run with no whitespace at all (a base64 blob, a
    minified line) is cut at the cap anyway: exceeding it is worse, because the model would
    truncate the tail silently and that content would vanish from the index.

    `overlap` repeats a little tail context at the head of the next piece so a severed
    sentence stays retrievable from both sides. It is clamped so it can never stall the walk.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    overlap = max(0, min(overlap, max_chars // MAX_OVERLAP_DIVISOR))
    pieces: list[str] = []
    start, n = 0, len(block)
    while start < n:
        end = start + max_chars
        if end >= n:
            pieces.append(block[start:])
            break
        cut = max(block.rfind(" ", start, end), block.rfind("\n", start, end))
        # `cut <= start` is checked EXPLICITLY, not folded into the piece floor: for a small
        # max_chars the floor rounds down to 0, and a cut sitting exactly on `start` would slip
        # through it and stall the walk forever.
        if cut <= start or cut - start < max_chars // MIN_PIECE_DIVISOR:
            # Either no boundary at all (an unbreakable run: a blob, a minified line), or one so
            # close to `start` that honoring it would emit a sliver — whitespace landing exactly
            # on the cap does this, yielding 1-character chunks that each cost a full embedding
            # row. Cut at the cap instead: never overflow the embedder window, never emit dust.
            cut = end
        pieces.append(block[start:cut])
        nxt = cut
        if overlap:
            # Seek FORWARD from the target step-back point: the first boundary at or after
            # `cut - overlap` repeats about `overlap` characters. Seeking backward from `cut`
            # instead would find the boundary NEAREST the cut — one word — making the knob
            # inert (every overlap value would produce identical output).
            lo = max(start + 1, cut - overlap)
            cands = [i for i in (block.find(" ", lo, cut), block.find("\n", lo, cut)) if i != -1]
            back = min(cands) if cands else -1
            # Two separate floors are needed. The piece floor above bounds how SHORT an emitted
            # piece may be; this one bounds how LITTLE the walk advances. Without it a step-back
            # can land at start+1, so each frame re-emits nearly the same text and the corpus
            # (and the embedding bill) inflates several-fold while every piece looks fine.
            if back > start and cut - back <= overlap and back - start >= max_chars // MIN_PIECE_DIVISOR:
                nxt = back + 1
        start = nxt
    return [p for p in (p.strip() for p in pieces) if p]


def _pack(blocks: list[str], max_chars: int, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Greedily pack pre-split blocks into chunks no larger than max_chars.

    Blocks are kept whole where they fit; a block that is itself larger than max_chars is
    force-split (`_split_block`) instead of being emitted oversized.
    """
    chunks: list[str] = []
    buf = ""
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if len(b) > max_chars:
            if buf:
                chunks.append(buf)
            pieces = _split_block(b, max_chars, overlap)
            chunks.extend(pieces[:-1])
            buf = pieces[-1] if pieces else ""  # the tail can still absorb following blocks
            continue
        if buf and len(buf) + len(b) + 2 > max_chars:
            chunks.append(buf)
            buf = b
        else:
            buf = f"{buf}\n\n{b}" if buf else b
    if buf:
        chunks.append(buf)
    return chunks


def chunk_text(
    text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    """Split prose/markdown into chunks on blank lines, packing paragraphs up to max_chars.

    A paragraph longer than max_chars is force-split on whitespace (with `overlap` characters
    of repeated context), so no chunk can exceed the cap and get truncated by the embedder.
    """
    return _pack(text.split("\n\n"), max_chars, overlap)


def chunk_code(
    text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
) -> list[str]:
    """Chunk source code at top-level ``def`` / ``class`` (and decorator) boundaries.

    A heuristic, not an AST parse: a line at column 0 starting a new ``def``/``class``/``@``
    begins a new block, so a function keeps its whole body (and methods stay with their class).
    Module preamble (imports, constants) forms the first block. Blocks are then packed up to
    max_chars, and a single function longer than max_chars is force-split rather than emitted
    oversized. Better retrieval than blank-line packing on code, without a language-specific
    parser — but the boundaries are Python's (``def``/``class``/decorators); other languages
    fall back to size-based splitting.
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
    return _pack(blocks, max_chars, overlap)


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
        # resolve() FIRST: `source` is the row key `replace_sources` deletes by, and the
        # basename→document relation the supersession map is built on. Left as typed, the same
        # file indexed as `corpus/x.md` and `/abs/corpus/x.md` becomes TWO row sets, so the
        # re-index duplicates instead of replacing AND the basename starts looking like two
        # different documents — which withdraws every supersession edge touching it.
        root = Path(path).resolve()
        files = sorted(root.glob(glob)) if root.is_dir() else [root]
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
                        metadata={"file": f.name, "ord": i, **meta},
                    )
                )
        # embed BEFORE touching the store: if embedding fails, the old rows stay intact
        embeddings = self._embedder.embed([c.text for c in all_chunks]) if all_chunks else []
        self._store.replace_sources([str(f) for f in files], all_chunks, embeddings)
        return IndexStats(files=len(files), chunks=len(all_chunks))
