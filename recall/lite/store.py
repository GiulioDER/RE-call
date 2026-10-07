"""`LiteStore`: the chunk store on one SQLite file, for a local install with no database server.

It answers the same calls the existing code makes on `PgVectorStore` (`recall.index.Indexer`,
`recall.retriever.HybridRetriever` and the calibration measurement), duck-typed as that code
already is, so none of that code changes:

* **Dense leg.** Every embedding sits in one normalised float32 matrix, rebuilt when the corpus
  changes, and a query is an exact cosine over all of it. At a local memory's size (thousands to
  tens of thousands of chunks) that costs milliseconds, and an exact search can only match or
  beat pgvector's approximate HNSW walk.
* **Keyword leg.** SQLite FTS5 with the Porter stemmer. The query is the OR of its terms minus
  PostgreSQL's English stop words, the shape `PgVectorStore._query_sparse` builds with
  `to_tsvector('english', ...)`. A chunk whose `numeric_values` contain a number from the query
  ranks first among the matches, which is what the Postgres store's 0.25 boost does in practice
  (its `ts_rank` values sit far below 0.25). FTS5 ranks with BM25, Postgres with `ts_rank`, so this
  leg's order can differ; that difference is measured, not assumed away.
* **Writes.** `replace_sources` keeps `first_indexed_at` across a re-index and also deletes by
  `(project, root-relative file)`, as the Postgres store does, so a memo re-indexed from another
  root replaces its old rows instead of landing beside them.

One file holds one tenant and one embedding width, both recorded on creation; opening it with a
different one is refused rather than mixed. Writes take SQLite's own lock, so this is a local,
single-user store.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recall.calibration_v2 import CalibrationResolution
from recall.errors import RecallError
from recall.scope import FACET_METADATA_KEY, Scope, _source_prefix_regex_pattern, coerce_scope
from recall.supersession import EdgeCandidates, chunk_supersedes_targets, resolve_supersession_candidates
from recall.types import Chunk, ScoredChunk

#: A DSN naming a lite store: ``sqlite:///path/to/memory.db``.
LITE_DSN_PREFIX = "sqlite:///"
SCHEMA_VERSION = "1"

#: PostgreSQL's English stop word list (`english.stop`), so both stores drop the same words from a
#: keyword query.
ENGLISH_STOP_WORDS = frozenset(
    """i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she
    her hers herself it its itself they them their theirs themselves what which who whom this that
    these those am is are was were be been being have has had having do does did doing a an the and
    but if or because as until while of at by for with about against between into through during
    before after above below to from up down in out on off over under again further then once here
    there when where why how all any both each few more most other some such no nor not only own same
    so than too very s t can will just don should now""".split()
)
_TERM = re.compile(r"\w+", re.UNICODE)
_NUMERIC_TOKEN = re.compile(r"(?<![\w.])[+-]?\d+(?:[.,]\d+)?%?(?![\w.])")
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f+00:00"
#: The same expressions as the `chunks_project_file` index, so the planner can use it.
_DELETE_PROJECT_FILES = (
    "DELETE FROM chunks WHERE json_extract(metadata, '$.project') = ? "
    "AND json_extract(metadata, '$.file') IN (SELECT value FROM json_each(?))"
)


class LiteStoreError(RecallError, ValueError):
    """The lite store refused an operation; the message says why."""


def is_lite_dsn(dsn: str | None) -> bool:
    return bool(dsn) and str(dsn).startswith(LITE_DSN_PREFIX)


def _now() -> str:
    return datetime.now(UTC).strftime(_TIME_FORMAT)


def _parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _numeric_terms(text: str) -> list[str]:
    """`recall.store._numeric_query_terms`, copied (that module loads psycopg); a test pins the two."""
    terms: list[str] = []
    for value in _NUMERIC_TOKEN.findall(text):
        normalized = value.replace(",", ".")
        if normalized not in terms:
            terms.append(normalized)
    return terms


def _fts_query(text: str) -> str:
    """The OR of the query's terms, stop words dropped, each quoted so FTS5 reads it as a term."""
    terms: list[str] = []
    for term in _TERM.findall(text.lower()):
        if term in ENGLISH_STOP_WORDS or term in terms:
            continue
        terms.append(term)
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


@dataclass(frozen=True)
class _Row:
    rowid: int
    id: str
    source: str
    text: str
    metadata: dict[str, Any]
    indexed_at: str
    first_indexed_at: str


@dataclass(frozen=True)
class _Matrix:
    version: int
    rows: list[_Row]
    by_id: dict[str, int]
    vectors: Any  # numpy float32 array, rows normalised
    fingerprint: str  # `recall.lite.calibration.corpus_digest` of these rows


def _matches(scope: Scope, row: _Row) -> bool:
    """`Scope.predicate`, evaluated in Python over one row; same arms, same rules.

    A hand copy of the SQL, so `tests/test_lite_store.py` pins `Scope`'s fields: a field added there
    and not here would be ignored, which widens a search instead of failing it.
    """
    file = row.metadata.get("file")
    file = file if isinstance(file, str) else None
    if scope.source is not None and scope.source not in (file, row.source):
        return False
    if scope.source_prefixes is not None:
        target = file or row.source
        # `fullmatch` with DOTALL is PostgreSQL's `~` on this anchored pattern: Python's `$` also
        # matches before a trailing newline, and its `.` stops at one.
        if not any(
            re.fullmatch(_source_prefix_regex_pattern(prefix), target, re.DOTALL) for prefix in scope.source_prefixes
        ):
            return False
    if scope.security_policy_digest is not None and row.metadata.get("security_policy_digest") != scope.security_policy_digest:
        return False
    folder = scope.normalized_folder
    if folder is not None:
        if file is None:
            return False
        if folder == "":
            if "/" in file:
                return False
        elif not (file == folder or file.startswith(folder + "/")):
            return False
    facet = scope.normalized_facet
    if facet is not None:
        value = row.metadata.get(FACET_METADATA_KEY)
        if not isinstance(value, str) or value.lower() != facet:
            return False
    return True


def _chunk_identity(chunks: Sequence[Chunk]) -> tuple[str | None, list[str]]:
    """The one project and its root-relative files a batch belongs to; `(None, [])` otherwise.

    `recall.store._chunk_identity`, copied; a test pins the two. A batch with no project, or more
    than one, deletes by source only, so a mixed batch can never erase another project's file.
    """
    projects = {value for c in chunks if isinstance(value := c.metadata.get("project"), str) and value}
    if len(projects) != 1:
        return None, []
    names: dict[str, None] = {}
    for c in chunks:
        name = c.metadata.get("file")
        if isinstance(name, str) and name:
            names.setdefault(name, None)
    return projects.pop(), list(names)


class LiteStore:
    """One tenant's chunks in one SQLite file. See the module docstring."""

    def __init__(self, path: str | Path, *, dim: int, tenant: str = "default", table: str = "chunks") -> None:
        if dim < 1:
            raise LiteStoreError(f"embedding width must be positive, got {dim}")
        if table != "chunks":
            raise LiteStoreError(f"a lite store has one table, 'chunks'; got {table!r}")
        self.path = Path(path)
        self._dim = dim
        self._tenant = tenant
        self._table = table
        self.generation_id: str | None = None
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()
        self._matrix: _Matrix | None = None
        self._supersession_cache: tuple[int, dict[str, str], frozenset[str], EdgeCandidates] | None = None

    @classmethod
    def from_dsn(cls, dsn: str, *, dim: int, tenant: str = "default") -> LiteStore:
        if not is_lite_dsn(dsn):
            raise LiteStoreError(f"not a lite store address (expected {LITE_DSN_PREFIX}<path>): {dsn!r}")
        return cls(dsn[len(LITE_DSN_PREFIX):], dim=dim, tenant=tenant)

    # ------------------------------------------------------------------ identity

    @property
    def tenant(self) -> str:
        return self._tenant

    @property
    def table(self) -> str:
        return self._table

    @property
    def dim(self) -> int:
        return self._dim

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (
                    -- AUTOINCREMENT so a deleted row's rowid is never handed to a new row: the
                    -- keyword leg joins FTS rowids to rows cached under an earlier read.
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    indexed_at TEXT NOT NULL,
                    first_indexed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
                CREATE INDEX IF NOT EXISTS chunks_project_file
                    ON chunks(json_extract(metadata, '$.project'), json_extract(metadata, '$.file'));
                CREATE TABLE IF NOT EXISTS calibrations (
                    rowid INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text, content='chunks', content_rowid='rowid', tokenize='porter unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
                END;
                """
            )
            wanted = {"schema_version": SCHEMA_VERSION, "tenant": self._tenant, "dim": str(self._dim)}
            # Claim, then compare what the file holds: checking first and inserting after lets two
            # first opens with different identities both pass, and the loser write into the other's.
            with self._write():
                self._conn.executemany(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                    [*wanted.items(), ("corpus_version", "0")],
                )
                stored = dict(self._conn.execute("SELECT key, value FROM meta").fetchall())
                for key, value in wanted.items():
                    if stored[key] != value:
                        raise LiteStoreError(
                            f"{self.path} holds {key}={stored[key]!r}, not {value!r}; one file holds one "
                            f"tenant and one embedding width, so open it with those or use another file"
                        )

    # ------------------------------------------------------------------ writes

    @contextmanager
    def _write(self) -> Iterator[None]:
        """One `BEGIN IMMEDIATE` transaction under the store lock, rolled back on any error.

        The rollback is skipped when SQLite has already ended the transaction itself (a full disk,
        an I/O error, a trigger's `RAISE(ROLLBACK)`), so the caller sees that error rather than
        "cannot rollback - no transaction is active". A failed COMMIT is rolled back too, so a
        busy or full disk cannot leave the transaction open and the store's write lock held.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    def _check_batch(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        import numpy as np

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        for c, e in zip(chunks, embeddings, strict=True):
            if "\x00" in c.text or "\x00" in c.id or "\x00" in c.source:
                raise ValueError(
                    f"chunk {c.id!r} from source {c.source!r} contains a NUL (0x00) byte; strip it before writing"
                )
            if len(e) != self._dim:
                raise LiteStoreError(f"chunk {c.id!r} has a {len(e)}-wide embedding; this store holds {self._dim}")
            # pgvector refuses NaN and infinity at insert; a stored NaN would make `top_cosine` NaN,
            # which a calibration counts as a correct abstention.
            if not np.isfinite(np.asarray(e, dtype=np.float32)).all():
                raise LiteStoreError(f"chunk {c.id!r} has a NaN or infinite embedding value")

    def _insert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]], now: str) -> None:
        import numpy as np

        self._conn.executemany(
            """
            INSERT INTO chunks (id, source, text, metadata, embedding, indexed_at, first_indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source = excluded.source,
                text = excluded.text,
                metadata = excluded.metadata,
                embedding = excluded.embedding,
                indexed_at = excluded.indexed_at,
                first_indexed_at = min(chunks.first_indexed_at, excluded.first_indexed_at)
            """,
            [
                (c.id, c.source, c.text, json.dumps(c.metadata), np.asarray(e, dtype=np.float32).tobytes(), now, now)
                for c, e in zip(chunks, embeddings, strict=True)
            ],
        )

    def _bump(self) -> None:
        """Move the corpus version, inside the caller's write transaction, so every cache reloads."""
        self._matrix = None
        self._conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'corpus_version'")

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        self._check_batch(chunks, embeddings)
        with self._write():
            self._insert(chunks, embeddings, _now())
            self._bump()
        return len(chunks)

    def replace_sources(self, sources: list[str], chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Atomically replace every row of `sources` (and of the batch's project files) with `chunks`."""
        self._check_batch(chunks, embeddings)
        with self._write():
            preserved: dict[str, str] = {}
            if sources:
                wanted = json.dumps(sources)
                preserved = dict(
                    self._conn.execute(
                        "SELECT id, first_indexed_at FROM chunks WHERE source IN (SELECT value FROM json_each(?))",
                        (wanted,),
                    ).fetchall()
                )
                self._conn.execute("DELETE FROM chunks WHERE source IN (SELECT value FROM json_each(?))", (wanted,))
            project, files = _chunk_identity(chunks)
            if project and files:
                self._conn.execute(_DELETE_PROJECT_FILES, (project, json.dumps(files)))
            now = _now()
            if chunks:
                self._insert(chunks, embeddings, now)
                restore = [(min(preserved[c.id], now), c.id) for c in chunks if c.id in preserved]
                if restore:
                    self._conn.executemany("UPDATE chunks SET first_indexed_at = ? WHERE id = ?", restore)
            self._bump()
        return len(chunks)

    def delete_sources(self, sources: list[str]) -> int:
        """Delete every row of `sources`; the delete and the version bump commit together or not at all.

        Apart, a failed bump would leave the rows gone and every cache keyed on the version (the
        matrix, the supersession map) still serving them, and a retry finds nothing left to delete,
        so nothing would ever move the version.
        """
        if not sources:
            return 0
        with self._write():
            removed = self._conn.execute(
                "DELETE FROM chunks WHERE source IN (SELECT value FROM json_each(?))", (json.dumps(sources),)
            ).rowcount
            self._bump()
        return int(removed)

    def delete_sources_across(self, tables: list[str], sources: list[str]) -> int:
        unknown = [t for t in dict.fromkeys(tables) if t != self._table]
        if unknown:
            raise LiteStoreError(f"a lite store has one table ({self._table!r}); cannot erase sources in {unknown}")
        return self.delete_sources(sources)

    def analyze_if_stale(self, modified: int) -> bool:
        """SQLite needs no planner statistics refresh for these queries; nothing to do."""
        return False

    def sparse_covered_sources(self, profile_id: str) -> set[str]:
        raise LiteStoreError("learned sparse retrieval needs the full (Postgres) install")

    # ------------------------------------------------------------------ reads used by indexing

    def _metadata_values(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def source_content_hashes(self) -> dict[str, str]:
        rows = self._metadata_values(
            "SELECT DISTINCT source, coalesce(json_extract(metadata, '$.index_fingerprint'), "
            "json_extract(metadata, '$.content_hash'), '') FROM chunks"
        )
        return {source: str(digest) for source, digest in rows}

    def source_raw_hashes(self) -> dict[str, str]:
        rows = self._metadata_values(
            "SELECT DISTINCT source, coalesce(json_extract(metadata, '$.content_hash'), '') FROM chunks"
        )
        return {source: str(digest) for source, digest in rows}

    def project_file_hashes(self, project: str) -> dict[str, str]:
        rows = self._metadata_values(
            "SELECT DISTINCT json_extract(metadata, '$.file'), coalesce(json_extract(metadata, '$.index_fingerprint'), "
            "json_extract(metadata, '$.content_hash'), '') FROM chunks "
            "WHERE json_extract(metadata, '$.project') = ? AND json_extract(metadata, '$.file') IS NOT NULL",
            (project,),
        )
        return {name: str(digest) for name, digest in rows}

    def count(self) -> int:
        return int(self._metadata_values("SELECT count(*) FROM chunks")[0][0])

    def newest_indexed_at(self) -> datetime | None:
        return _parse_time(self._metadata_values("SELECT max(indexed_at) FROM chunks")[0][0])

    def iter_chunks(self, batch_size: int = 1000) -> Iterator[Chunk]:
        """Every chunk by id, fetched `batch_size` rows at a time, as `PgVectorStore.iter_chunks`
        streams them: no embeddings are read and the search matrix is not built.

        One SELECT on a read connection of its own, so the whole iteration sees one snapshot (rows
        written after it opened are not seen) and no lock is held between rows.
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive int")
        reader = sqlite3.connect(self.path, timeout=30.0)
        try:
            cursor = reader.execute("SELECT id, source, text, metadata FROM chunks ORDER BY id")
            while rows := cursor.fetchmany(batch_size):
                for cid, source, text, metadata in rows:
                    yield Chunk(id=cid, source=source, text=text, metadata=json.loads(metadata))
        finally:
            reader.close()

    def chunks_for_source(self, source: str) -> list[Chunk]:
        """One exact source's chunks in ingestion order, `(indexed_at, id)`, as Postgres orders them."""
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")
        rows = self._metadata_values(
            "SELECT id, source, text, metadata FROM chunks WHERE source = ? ORDER BY indexed_at, id", (source,)
        )
        return [Chunk(id=cid, source=src, text=text, metadata=json.loads(metadata)) for cid, src, text, metadata in rows]

    # ------------------------------------------------------------------ retrieval

    def _version(self) -> int:
        return int(self._metadata_values("SELECT value FROM meta WHERE key = 'corpus_version'")[0][0])

    def _load(self) -> _Matrix:
        import numpy as np

        with self._lock:
            version = self._version()
            if self._matrix is not None and self._matrix.version == version:
                return self._matrix
            raw = self._conn.execute(
                "SELECT rowid, id, source, text, metadata, embedding, indexed_at, first_indexed_at FROM chunks ORDER BY rowid"
            ).fetchall()
            rows = [_Row(r[0], r[1], r[2], r[3], json.loads(r[4]), r[6], r[7]) for r in raw]
            vectors: Any = np.zeros((len(raw), self._dim), dtype=np.float32)
            for i, r in enumerate(raw):
                vectors[i] = np.frombuffer(r[5], dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            from recall.lite.calibration import corpus_digest

            fingerprint = corpus_digest([(row.id, row.text) for row in rows])
            self._matrix = _Matrix(version, rows, {row.id: i for i, row in enumerate(rows)}, vectors / norms, fingerprint)
            return self._matrix

    def _query_vector(self, vector: Sequence[float]) -> Any:
        import numpy as np

        if len(vector) != self._dim:
            raise LiteStoreError(f"query vector is {len(vector)} wide; this store holds {self._dim}")
        q: Any = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        return q / norm if norm else q

    def _hit(self, row: _Row, score: float) -> ScoredChunk:
        return ScoredChunk(
            chunk=Chunk(id=row.id, source=row.source, text=row.text, metadata=row.metadata),
            score=float(score),
            indexed_at=_parse_time(row.indexed_at),
            first_indexed_at=_parse_time(row.first_indexed_at),
        )

    def _allowed(self, matrix: _Matrix, scope: Scope) -> list[int] | None:
        """Row positions the scope allows, or None for every row."""
        if scope.is_empty:
            return None
        return [i for i, row in enumerate(matrix.rows) if _matches(scope, row)]

    def query_dense(
        self, vector: list[float], k: int, source: str | None = None, scope: Scope | None = None
    ) -> list[ScoredChunk]:
        """Exact cosine top `k`, best first; ties broken by chunk id so the order is stable."""
        import numpy as np

        if k <= 0:
            raise ValueError("k must be a positive int")
        matrix = self._load()
        if not matrix.rows:
            return []
        allowed = self._allowed(matrix, coerce_scope(scope, source))
        positions = np.arange(len(matrix.rows)) if allowed is None else np.asarray(allowed, dtype=np.int64)
        if positions.size == 0:
            return []
        scores = matrix.vectors[positions] @ self._query_vector(vector)
        order = sorted(range(positions.size), key=lambda i: (-float(scores[i]), matrix.rows[int(positions[i])].id))[:k]
        return [self._hit(matrix.rows[int(positions[i])], float(scores[i])) for i in order]

    def top_cosine(self, vector: list[float]) -> float:
        """The best cosine any chunk has with `vector`, exactly; 0.0 for an empty store."""
        matrix = self._load()
        if not matrix.rows:
            return 0.0
        return float((matrix.vectors @ self._query_vector(vector)).max())

    def cosines_for(self, ids: Sequence[str], vec: list[float]) -> dict[str, float]:
        matrix = self._load()
        q = self._query_vector(vec)
        found: dict[str, float] = {}
        for cid in dict.fromkeys(str(i) for i in ids):
            position = matrix.by_id.get(cid)
            if position is not None:
                found[cid] = float(matrix.vectors[position] @ q)
        return found

    def query_sparse(
        self,
        text: str,
        k: int,
        source: str | None = None,
        vec: list[float] | None = None,
        scope: Scope | None = None,
    ) -> list[ScoredChunk]:
        """Keyword top `k`: FTS5 over stemmed terms, numeric matches first, then BM25.

        `score` is the dense cosine with `vec` when one is given (as the Postgres store returns it),
        else a positive keyword rank (higher is better).
        """
        if k <= 0:
            raise ValueError("k must be a positive int")
        query = _fts_query(text)
        if not query:
            return []
        matrix = self._load()
        effective = coerce_scope(scope, source)
        with self._lock:
            matched = self._conn.execute(
                "SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ?", (query,)
            ).fetchall()
        by_rowid = {row.rowid: row for row in matrix.rows}
        numeric = set(_numeric_terms(text))
        ranked: list[tuple[bool, float, str, _Row]] = []
        for rowid, bm25 in matched:
            row = by_rowid.get(rowid)
            if row is None or not _matches(effective, row):
                continue
            values = row.metadata.get("numeric_values")
            boosted = bool(numeric) and isinstance(values, list) and any(str(v) in numeric for v in values)
            ranked.append((boosted, -float(bm25), row.id, row))
        ranked.sort(key=lambda t: (not t[0], -t[1], t[2]))
        top = ranked[:k]
        if vec is not None:
            cosines = self.cosines_for([t[3].id for t in top], vec)
            return [self._hit(t[3], cosines.get(t[3].id, 0.0)) for t in top]
        return [self._hit(t[3], t[1] + (1.0 if t[0] else 0.0)) for t in top]

    def query_learned_sparse(self, *args: Any, **kwargs: Any) -> list[ScoredChunk]:
        raise LiteStoreError("learned sparse retrieval needs the full (Postgres) install")

    # ------------------------------------------------------------------ calibration and lineage

    def corpus_fingerprint(self) -> str:
        """A digest of every chunk's id and text: changes exactly when what search sees changes."""
        return self._load().fingerprint

    def lite_generation_id(self) -> str:
        """A stable name for the corpus as it is now, in the place a Postgres generation id goes."""
        return "lite-" + self.corpus_fingerprint()[:16]

    def save_calibration(self, payload: str, *, created_at: str, model: str, dimension: int) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("INSERT INTO calibrations(created_at, payload) VALUES (?, ?)", (created_at, payload))
                self._conn.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    [("embedder_model", model), ("embedder_dimension", str(dimension))],
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def latest_calibration_json(self) -> str | None:
        row = self._metadata_values("SELECT payload FROM calibrations ORDER BY created_at DESC, rowid DESC LIMIT 1")
        return str(row[0][0]) if row else None

    def resolve_calibration(self) -> CalibrationResolution:
        """What `trusted_search` asks every search: see `recall.lite.calibration.resolve`."""
        from recall.lite.calibration import resolve

        return resolve(self)

    def generation_binding(self) -> dict[str, str]:
        """The identity `trusted_search` checks the runtime embedder against.

        The embedder is recorded when a calibration is stored, because indexing hands this store
        vectors, not a model name. Until then there is no model to compare and no threshold to
        protect, so the binding names the corpus only.
        """
        meta = dict(self._metadata_values("SELECT key, value FROM meta"))
        binding = {
            "tenant_id": self._tenant,
            "generation_id": self.lite_generation_id(),
            "corpus_fingerprint": self.corpus_fingerprint(),
        }
        if "embedder_model" in meta:
            from recall.lite.calibration import pipeline_fingerprint

            binding["embedder_model"] = meta["embedder_model"]
            binding["embedder_dimension"] = meta["embedder_dimension"]
            binding["pipeline_fingerprint"] = pipeline_fingerprint(meta["embedder_model"], int(meta["embedder_dimension"]))
        return binding

    # ------------------------------------------------------------------ supersession

    def supersession(self) -> tuple[dict[str, str], frozenset[str]]:
        """``(edges, unresolved)``, as `PgVectorStore.supersession` returns them."""
        edges, unresolved, _candidates = self.supersession_all()
        return edges, unresolved

    def supersession_all(self) -> tuple[dict[str, str], frozenset[str], EdgeCandidates]:
        """``(edges, unresolved, candidates)`` from one scan, cached per corpus version.

        The rows have the shape `PgVectorStore.supersession_all` builds in SQL, read per chunk by
        the shared Python twin (`chunk_supersedes_targets`): one row per (file, declared
        reference), a chunk declaring nothing still yields one row so every file reaches the
        resolver, dated by the earliest `first_indexed_at` among the chunks carrying the claim.
        The shared `resolve_supersession_candidates` then applies the rule both stores share.

        Two residues, neither reachable from metadata the indexer writes: the twin reads a few
        hand-written `supersedes` shapes (a flow-sequence string, padded or non-string list
        items) more leniently than the SQL, and a non-string `file` is skipped. Rows are ordered
        by code point, which is Postgres's order under the C collation only, so when two files
        supersede one target the `edges` winner (last row wins) can differ from a Postgres store
        on another collation, and so can the successor a replay (`known_as_of`) picks when two
        claims on one target carry the same `first_indexed_at`, the usual case for claims written
        in one indexing batch, because `resolve_successor` breaks that tie by scan position. The
        set of candidates and their dates do not depend on the order.

        The cache is keyed on the corpus version stored in the file, so an edge written by
        another process (`recall index` beside a running server) is seen on the next call.
        """
        matrix = self._load()
        cached = self._supersession_cache
        if cached is None or cached[0] != matrix.version:
            first: dict[tuple[str, str | None], str] = {}
            for row in matrix.rows:
                file = row.metadata.get("file")
                if not isinstance(file, str):
                    continue
                targets: tuple[str | None, ...] = chunk_supersedes_targets(row.metadata) or (None,)
                for target in targets:
                    key = (file, target)
                    if key not in first or row.first_indexed_at < first[key]:
                        first[key] = row.first_indexed_at
            rows: list[tuple[str | None, str | None, datetime | None]] = [
                (file, target, _parse_time(stamp))
                for (file, target), stamp in sorted(first.items(), key=lambda item: (item[0][0], item[0][1] is None, item[0][1] or ""))
            ]
            edges, unresolved, candidates = resolve_supersession_candidates(rows)
            cached = (matrix.version, edges, unresolved, candidates)
            self._supersession_cache = cached
        _version, edges, unresolved, candidates = cached
        return dict(edges), unresolved, {target: list(claims) for target, claims in candidates.items()}
