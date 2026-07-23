from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from recall.cache import EmbeddingCache, embed_with_cache
from recall.embeddings import Embedder
from recall.frontmatter import parse_frontmatter, validity_bounds
from recall.lint import DEFAULT_GLOB
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
#: Refuse to prune when a single run would delete at least this fraction of the sources already
#: indexed under the root. Re-indexing deletes rows for files that are gone from disk, which is
#: correct when a memo was really deleted and catastrophic when the corpus merely wasn't there:
#: an unmounted volume, a half-finished sync, a path that still resolves. Those cases are
#: indistinguishable from "the author deleted everything" at the filesystem level, so the only
#: safe reading of a mass disappearance is that something is wrong. Overridable per-run.
DEFAULT_MAX_PRUNE_FRACTION = 0.5
#: ...but only once the corpus is big enough for a fraction to mean anything. Deleting one of two
#: memos is 50% and entirely routine; the guard must not make small corpora unusable. Below this
#: many known sources, pruning proceeds unguarded — the blast radius is a handful of files that a
#: re-index restores.
PRUNE_GUARD_MIN_SOURCES = 5

_log = get_logger("index")


def _prune_fraction_from_env() -> float:
    """`RECALL_MAX_PRUNE_FRACTION`, bounded to (0, 1]; anything malformed falls back to default.

    Read per-Indexer rather than at import so a test (or a long-lived process) can change it
    without reloading the module. Values outside the range are ignored rather than clamped: a
    caller who wrote `50` meant 50 percent, and silently treating it as "never guard" would
    disable the protection at exactly the moment someone was trying to configure it.
    """
    raw = os.environ.get("RECALL_MAX_PRUNE_FRACTION")
    if raw is None:
        return DEFAULT_MAX_PRUNE_FRACTION
    try:
        value = float(raw)
    except ValueError:
        _log.warning("ignoring malformed RECALL_MAX_PRUNE_FRACTION=%r", raw)
        return DEFAULT_MAX_PRUNE_FRACTION
    if not (0.0 < value <= 1.0) or value != value:  # NaN fails the range test too, explicitly
        _log.warning(
            "ignoring out-of-range RECALL_MAX_PRUNE_FRACTION=%r (expected 0 < f <= 1)", raw
        )
        return DEFAULT_MAX_PRUNE_FRACTION
    return value


class PruneGuardTripped(RuntimeError):
    """A re-index would have deleted most of the corpus, so nothing was deleted.

    Deliberately NOT a ValueError: the caller passed a perfectly valid path. What is wrong is the
    state of the filesystem it points at, and the two deserve different handling.
    """

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


def _strip_nul(text: str, source: Path) -> str:
    """Remove NUL bytes, which PostgreSQL text columns cannot store at all.

    Found by indexing a real corpus: **one** file out of 792 carried two stray NUL bytes, and
    psycopg aborted the whole run with `PostgreSQL text fields cannot contain NUL (0x00) bytes` —
    an error that names neither the file nor the chunk. 0.13% of the corpus took down 100% of the
    indexing, and with batching it also discards every batch after the bad one.

    Stripping is right here because a NUL in a markdown memo is corruption rather than content;
    nothing is silently lost that a reader would have seen. But it is LOGGED with the file and the
    count, because quietly rewriting a user's document is its own kind of wrong.
    """
    if "\x00" not in text:
        return text
    _log.warning(
        "stripped %d NUL byte(s) from %s — PostgreSQL text columns cannot store them",
        text.count("\x00"), source,
    )
    return text.replace("\x00", "")


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


def candidate_files(path: str | Path, glob: str = DEFAULT_GLOB) -> list[Path]:
    """Return the confined, sorted file list that `index_path(path, glob)` would index.

    A pure filesystem walk: it only `stat`s and `resolve`s paths — no file is opened, no chunker
    runs, no embedding is requested. This is what lets a caller MEASURE a tree (file count, total
    bytes) and refuse to index it before spending anything, using the exact same walk `index_path`
    uses, so the measurement can never diverge from what would actually be indexed.
    """
    root = Path(path).resolve()
    return sorted(_confined_to(root, root.glob(glob))) if root.is_dir() else [root]


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
        allow_prune: bool = False,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._chunker = chunker
        self._cache = cache
        if batch_chunks < 1:
            raise ValueError("batch_chunks must be >= 1")
        self._batch_chunks = batch_chunks
        #: Set by a caller who has confirmed the files really are gone. Bypasses the guard for
        #: this Indexer only — there is no global off switch, because the run that needs one is
        #: always the run you are not watching.
        self._allow_prune = allow_prune
        self._max_prune_fraction = _prune_fraction_from_env()

    def index_path(
        self, path: str | Path, glob: str | None = None, files: list[Path] | None = None
    ) -> IndexStats:
        """Index a markdown file, or a directory of them, into the vector store.

        Re-indexing REPLACES each file's rows completely (delete then insert), so a file that
        shrinks — or withdraws a frontmatter claim like ``supersedes`` — leaves no stale
        chunks behind to poison retrieval or the supersession map.

        `files` accepts an already-walked candidate list, and exists so a caller that had to
        measure the tree first can hand over the SAME set it measured. A caller that re-walks
        instead is asking the filesystem the same question twice and getting two answers: the
        window between them is a full directory walk wide, and anything appearing in it is
        indexed without having been counted — which matters when the count was a budget check or
        a bill.

        `files` is re-confined to `path` here rather than trusted. A precondition stated only in
        a docstring is not enforced, and this one guards the root confinement: for a directory
        root an escape happens to raise from `relative_to` below, but for a single-file root the
        `rel` fallback is the bare basename, so an out-of-root path would be read and embedded
        with nothing to stop it. Re-filtering costs one stat per file and cannot reintroduce the
        double-walk this parameter exists to remove — `_confined_to` opens nothing and walks
        nothing, it only checks the paths it is handed.

        `glob` and `files` are mutually exclusive: passing both is a caller bug, because the set
        that gets indexed would silently be the one the glob did not describe. `glob` defaults to
        None rather than to `DEFAULT_GLOB` so that check tests whether the argument was PASSED,
        not whether its value happens to differ from the default — otherwise the guard would go
        quiet the day someone changes `DEFAULT_GLOB` to match.
        """
        if files is not None and glob is not None:
            raise ValueError(
                "pass either `glob` or `files`, not both: `files` is used verbatim, so the "
                f"glob {glob!r} would be silently ignored"
            )
        glob = DEFAULT_GLOB if glob is None else glob
        # resolve() FIRST: `source` is the row key `replace_sources` deletes by, so a corpus
        # indexed once as `corpus/` and once as `/abs/corpus/` would write two row sets for the
        # same file instead of replacing them. (Root-relative `file` keys are unaffected —
        # they are computed against this same root either way.)
        root = Path(path).resolve()
        if files is None:
            files = candidate_files(root, glob)
            dropped = 0
        else:
            # `_confined_to` filters on `is_file()` as well as on the root, so it silently eats a
            # file that vanished between the caller's walk and this call. That count has to be
            # carried forward: without it the total-failure check below cannot fire on this path
            # — every candidate would be gone, and the loop would simply never see one — so a
            # corpus that disappeared would return "indexed 0 files" instead of raising.
            requested = len(files)
            files = _confined_to(root, files)
            dropped = requested - len(files)
            if dropped:
                _log.warning(
                    "%d of %d supplied file(s) are outside %s, or no longer readable; skipping",
                    dropped, requested, root,
                )
        # Identify each file by its ROOT-RELATIVE path, not its bare basename: two files with the
        # same basename in different directories (a/notes.md, b/notes.md) must not collide in the
        # supersession map or in provenance. Mirrors recall.lint's `rel` keying. A single-file
        # index has no root to relativize against, so it falls back to the basename.
        rel = {f: (f.relative_to(root).as_posix() if root.is_dir() else f.name) for f in files}

        known = self._store.source_content_hashes()
        deleted = self._prune_vanished(root, files, known) if root.is_dir() else 0

        pending_sources: list[str] = []
        pending_chunks: list[Chunk] = []
        indexed = skipped = written = vanished_before_read = 0

        for f in files:
            try:
                raw = f.read_text(encoding="utf-8-sig")  # BOM must not disable frontmatter parsing
            except (FileNotFoundError, NotADirectoryError) as exc:
                # A file that DISAPPEARED between the walk and this read must not take the rest
                # of the run down with it: the caller may already have debited a byte budget for
                # the whole measured set, and earlier batches may already be committed, so
                # aborting here leaves a partial index AND a spent quota.
                #
                # Only ENOENT/ENOTDIR are absorbed — the same classification `gone_from_disk`
                # uses below, and for the same reason. A permission or I/O error is NOT a file
                # that went away; swallowing those would turn "I could not read your corpus"
                # into "indexed 0 files", exit 0. Logged, because the file WAS paid for.
                _log.warning("skipping %s: it vanished before it could be read (%s)", f, exc)
                vanished_before_read += 1
                continue
            raw = _strip_nul(raw, f)
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
        # Tolerating individual disappearances must not turn a TOTAL failure into a success.
        # If every candidate is gone, the corpus was not there — a wrong path, an unmounted
        # volume, a sync that removed everything — and reporting "indexed 0 files" with exit 0
        # is the same silence the prune guard exists to break.
        #
        # Counted against what was ASKED FOR, not against what survived confinement: on the
        # `files=` path the disappearances are absorbed above, so `len(files)` is already 0 and
        # comparing against it would make this check unfireable exactly where it is needed.
        candidates = len(files) + dropped
        if candidates and (vanished_before_read + dropped) == candidates:
            raise FileNotFoundError(
                f"none of the {candidates} candidate file(s) under {root} could be read: "
                f"every one of them vanished between the scan and the read"
            )
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

        A deleted file is never replaced and never removed: its chunks stay indexed forever. That
        is not a performance bug — the trust layer goes on serving a deleted memory with verdict
        `ok`.

        Scoped to the indexed root, because `source` is an absolute path and a corpus may be
        indexed in several roots; pruning everything absent from THIS glob would delete the
        others' rows on every run.

        "Gone from disk" is checked against the disk, not inferred from absence from `files`.
        Those are different questions: `files` is what THIS run's glob matched, and the glob
        varies between runs on one root (`--glob '**/*.py'` for code, the default for markdown,
        sharing a table because `--table` defaults to the same name). Inferring deletion from a
        set difference deletes the other glob's rows, and the fraction guard below does not catch
        it whenever those rows are a minority of the corpus. A file the scan merely could not
        reach — an unreadable directory, a symlink outside the root — is likewise not a deletion.
        """
        def under_root(source: str) -> bool:
            try:
                return Path(source).is_relative_to(root)
            except (OSError, ValueError):  # pragma: no cover - unparsable stored path
                return False

        def gone_from_disk(source: str) -> bool:
            # `os.stat` directly, classified by errno — NOT `Path.exists()`. On Python >= 3.12
            # `Path.exists()` delegates to `os.path.exists`, which swallows every OSError and
            # ValueError and returns False, so "I could not stat this" and "this file is gone"
            # become the same answer and an `except OSError` around it can never fire. That
            # collapses the distinction this function exists to draw: an unreadable parent
            # directory (EACCES), a dropped network mount (EIO/ESTALE) or a symlink loop
            # (ELOOP) would be read as a deletion and the rows would go. Only ENOENT and
            # ENOTDIR mean the file is actually gone; every other error means unreachable,
            # which is not a deletion.
            try:
                os.stat(source)
            except (FileNotFoundError, NotADirectoryError):
                return True
            except (OSError, ValueError):
                return False
            return False

        present = {str(f) for f in files}
        # One pass, one definition of "under this root". The guard below divides by this set, so
        # computing it a second way is how the numerator and denominator drift apart.
        indexed_here = [s for s in known if under_root(s)]
        vanished = [s for s in indexed_here if s not in present and gone_from_disk(s)]
        if not vanished:
            return 0

        # Guard BEFORE the delete, against the set scoped to this root — not the whole table, or a
        # second corpus indexed elsewhere would dilute the fraction and mask a total wipe of this
        # one.
        if len(indexed_here) >= PRUNE_GUARD_MIN_SOURCES and not self._allow_prune:
            fraction = len(vanished) / len(indexed_here)
            if fraction >= self._max_prune_fraction:
                raise PruneGuardTripped(
                    f"refusing to prune {len(vanished)} of {len(indexed_here)} indexed source(s) "
                    f"({fraction:.0%}) under {str(root)!r} — nothing was deleted. Files that are "
                    f"gone from disk are normally removed from the index, but a disappearance "
                    f"this large usually means the corpus is missing rather than deleted "
                    f"(unmounted volume, interrupted sync, wrong path). Confirm the files are "
                    f"really gone, then re-run with allow_prune=True (CLI: --allow-prune). "
                    f"RECALL_MAX_PRUNE_FRACTION (currently {self._max_prune_fraction:g}) tunes "
                    f"how large a disappearance is tolerated, but cannot disable the guard."
                )

        _log.info("pruning %d file(s) no longer on disk under %s", len(vanished), root)
        self._store.delete_sources(vanished)
        return len(vanished)
