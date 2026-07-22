from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from recall.cache import EmbeddingCache, embed_with_cache
from recall.embeddings import Embedder
from recall.frontmatter import parse_frontmatter, validity_bounds
from recall.observability import get_logger
from recall.store import PgVectorStore
from recall.types import Chunk

DEFAULT_MAX_CHARS = 800  # target chunk size in characters; paragraphs are packed up to this
DEFAULT_OVERLAP_CHARS = 80  # chars shared between adjacent pieces of a force-split oversized block
#: Overlap may repeat at most a quarter of a chunk. Beyond that, consecutive pieces are mostly
#: duplicated context: the index inflates and near-identical chunks crowd the top-k.
MAX_OVERLAP_DIVISOR = 4
#: A forced cut closer than max_chars/this to the piece start is ignored in favour of the cap —
#: it would emit a sliver whose embedding row carries almost no content. The same bound floors
#: how far the overlap step-back may rewind, so the walk always makes real progress.
MIN_PIECE_DIVISOR = 8
#: Chunks accumulated before a batch is embedded and written. Bounds peak memory to
#: roughly one batch of chunks plus their vectors, instead of the whole corpus, and
#: makes progress visible in the database while a long index is still running.
DEFAULT_BATCH_CHUNKS = 512

_log = get_logger("index")

# A chunker turns one document's text into a list of chunk strings.
Chunker = Callable[[str], list[str]]


def _split_oversized(block: str, max_chars: int, overlap: int) -> list[str]:
    """Force-split a single block that exceeds max_chars into <=max_chars pieces.

    A block with no blank line (a log dump, a table, a wall of prose) would otherwise become one
    chunk larger than the embedder's token window and be SILENTLY truncated, losing its tail.

    Cuts land on whitespace. A fixed-stride window is simpler but slices through words, so the
    embedder sees fragments ("wor" + "d18") that mean nothing and loses the token that did. A
    run with no whitespace at all (a base64 blob, a minified line) is still cut at the cap:
    exceeding it is worse, because the model truncates the tail silently.

    ``overlap`` repeats tail context at the head of the next piece so a fact straddling a cut
    survives in both neighbours. It is clamped, and TWO independent floors keep the walk sane:
    one bounds how short an emitted piece may be, the other how little the walk advances.
    Without the second, an overlap near max_chars re-emits a near-identical window per step and
    the index inflates many-fold (a fixed stride of ``max_chars - overlap`` collapses to 1 and
    does exactly that).
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if len(block) <= max_chars:
        return [block]
    overlap = max(0, min(overlap, max_chars // MAX_OVERLAP_DIVISOR))
    min_piece = max_chars // MIN_PIECE_DIVISOR
    pieces: list[str] = []
    start, n = 0, len(block)
    while start < n:
        end = start + max_chars
        if end >= n:
            pieces.append(block[start:])
            break
        cut = max(block.rfind(" ", start, end), block.rfind("\n", start, end))
        # `cut <= start` is checked EXPLICITLY, not folded into the piece floor: for a small
        # max_chars the floor rounds to 0, and a cut sitting exactly on `start` would slip
        # through it and stall the walk forever.
        if cut <= start or cut - start < min_piece:
            cut = end  # no usable boundary, or one so close it would emit a sliver
        pieces.append(block[start:cut])
        nxt = cut
        if overlap:
            # Seek FORWARD from the target step-back point: the first boundary at or after
            # `cut - overlap` repeats about `overlap` characters. Seeking backward from `cut`
            # would find the boundary NEAREST it — one word — making the knob inert.
            lo = max(start + 1, cut - overlap)
            cands = [i for i in (block.find(" ", lo, cut), block.find("\n", lo, cut)) if i != -1]
            back = min(cands) if cands else -1
            if back > start and cut - back <= overlap and back - start >= min_piece:
                nxt = back + 1
        start = nxt
    return [p for p in (p.strip() for p in pieces) if p]


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


def _confined_to(root: Path, paths: Iterable[Path]) -> list[Path]:
    """Keep only the paths that REALLY live under `root` once symlinks are resolved.

    A caller that confines a path argument to an allowed root (as the MCP server does with
    ``RECALL_INDEX_ROOT``) has checked the argument, not the walk. The walk is a glob, and
    ``pathlib`` only gained ``recurse_symlinks`` — defaulting to False — in **3.13**, while this
    package supports **3.11+**. On 3.11 and 3.12 ``**`` follows directory symlinks, so a symlink
    planted inside an otherwise-confined root yields files from anywhere on the filesystem: the
    confinement check passes and the read happens anyway.

    Filtering on the RESOLVED path fixes it on every supported version and needs no version
    check, which is the point — a fix conditioned on the interpreter version is one refactor away
    from being wrong on the version nobody tests. It also covers the plain file symlink, which
    ``recurse_symlinks`` does not.

    Escapes are dropped silently rather than raising: a symlink out of the tree is a corpus
    layout choice, not an attack in progress, and a single stray link should not make the whole
    corpus unindexable.
    """
    kept: list[Path] = []
    for p in paths:
        try:
            resolved = p.resolve()
        except OSError:  # pragma: no cover - broken symlink / permission
            continue
        if resolved.is_relative_to(root) and resolved.is_file():
            kept.append(p)
    return kept


@dataclass(frozen=True)
class IndexStats:
    files: int    # files actually (re)indexed
    chunks: int   # chunks written
    skipped: int = 0   # files unchanged since last index, so not re-read or re-embedded
    deleted: int = 0   # files gone from disk whose rows were pruned


class Indexer:
    def __init__(
        self,
        store: PgVectorStore,
        embedder: Embedder,
        chunker: Chunker = chunk_text,
        cache: EmbeddingCache | None = None,
        batch_chunks: int = DEFAULT_BATCH_CHUNKS,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._chunker = chunker
        self._cache = cache
        if batch_chunks < 1:
            raise ValueError("batch_chunks must be >= 1")
        self._batch_chunks = batch_chunks

    def index_path(self, path: str | Path, glob: str = "**/*.md") -> IndexStats:
        """Index a markdown file, or a directory of them, into the vector store.

        Re-indexing REPLACES each file's rows completely (delete then insert), so a file that
        shrinks — or withdraws a frontmatter claim like ``supersedes`` — leaves no stale
        chunks behind to poison retrieval or the supersession map.
        """
        # resolve() FIRST: `source` is the row key `replace_sources` deletes by, so a corpus
        # indexed once as `corpus/` and once as `/abs/corpus/` would write two row sets for the
        # same file instead of replacing them. (Root-relative `file` keys are unaffected —
        # they are computed against this same root either way.)
        root = Path(path).resolve()
        files = sorted(_confined_to(root, root.glob(glob))) if root.is_dir() else [root]
        # Identify each file by its ROOT-RELATIVE path, not its bare basename: two files with the
        # same basename in different directories (a/notes.md, b/notes.md) must not collide in the
        # supersession map or in provenance. Mirrors recall.lint's `rel` keying. A single-file
        # index has no root to relativize against, so it falls back to the basename.
        rel = {f: (f.relative_to(root).as_posix() if root.is_dir() else f.name) for f in files}

        known = self._store.source_content_hashes()
        deleted = self._prune_vanished(root, files, known) if root.is_dir() else 0

        pending_sources: list[str] = []
        pending_chunks: list[Chunk] = []
        indexed = skipped = written = 0

        for f in files:
            raw = f.read_text(encoding="utf-8-sig")  # BOM must not disable frontmatter parsing
            content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            if known.get(str(f)) == content_hash:
                skipped += 1
                continue
            meta, body = parse_frontmatter(raw)
            try:
                validity_bounds(meta)  # fail fast on malformed dates, before anything is embedded
            except ValueError as exc:
                raise ValueError(f"{f}: {exc}") from exc
            pending_sources.append(str(f))
            indexed += 1
            for i, ct in enumerate(self._chunker(body)):
                cid = hashlib.md5(f"{f}:{i}".encode("utf-8")).hexdigest()
                pending_chunks.append(
                    Chunk(
                        id=cid,
                        source=str(f),
                        text=ct,
                        metadata={
                            "file": rel[f], "ord": i, "content_hash": content_hash, **meta
                        },
                    )
                )
            # Flush on a whole-file boundary once the batch is big enough. A file's chunks are
            # never split across batches: `replace_sources` deletes the file's rows before
            # inserting, so a half-written file would land as a partial replace.
            if len(pending_chunks) >= self._batch_chunks:
                written += self._flush(pending_sources, pending_chunks)
                pending_sources, pending_chunks = [], []

        written += self._flush(pending_sources, pending_chunks)
        return IndexStats(files=indexed, chunks=written, skipped=skipped, deleted=deleted)

    def _flush(self, sources: list[str], chunks: list[Chunk]) -> int:
        """Embed and write one batch. Returns chunks written."""
        if not sources:
            return 0
        # Embed BEFORE touching the store: if embedding fails, this batch's old rows stay
        # intact. With a cache, unchanged chunk text is served from cache and never re-embedded.
        embeddings = (
            embed_with_cache(self._embedder, [c.text for c in chunks], self._cache)
            if chunks
            else []
        )
        self._store.replace_sources(sources, chunks, embeddings)
        return len(chunks)

    def _prune_vanished(self, root: Path, files: list[Path], known: dict[str, str]) -> int:
        """Delete rows for files that are gone from disk, scoped to `root`.

        The glob only lists files that still EXIST, so a deleted file was never replaced and
        never removed: its chunks stayed indexed forever. That is not a performance bug — the
        trust layer went on serving a deleted memory with verdict `ok`.

        Scoped to the indexed root, because `source` is an absolute path and a corpus may be
        indexed in several roots; pruning everything absent from THIS glob would delete the
        others' rows on every run.
        """
        present = {str(f) for f in files}
        vanished = []
        for source in known:
            if source in present:
                continue
            try:
                if Path(source).is_relative_to(root):
                    vanished.append(source)
            except (OSError, ValueError):  # pragma: no cover - unparsable stored path
                continue
        if not vanished:
            return 0
        _log.info("pruning %d file(s) no longer on disk under %s", len(vanished), root)
        self._store.delete_sources(vanished)
        return len(vanished)
