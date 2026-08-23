"""A persistent `ExtractionCache`, so a second ingest of an unchanged corpus is free.

`InMemoryExtractionCache` lives and dies with the process, which makes `--recheck` possible and
saves nothing across runs. On the `openai` engine, where every file is a paid call, across runs
is the only place the saving was ever wanted.

SQLite on a user named path, following `RejectionLedger` in `recall/rewrite.py`, which is this
repo's precedent for a sidecar store.

Three failures, three different answers, because they are not the same failure:

* **The path is not a usable store.** Refused, before any extraction runs, so a typo costs zero
  engine calls. You asked for a cache HERE, and here is not a cache.
* **One row cannot be used.** A miss. That entry is re-paid from the engine and counted.
  Refusing the run over one bad row would punish the user for a corruption they did not cause
  and can recover from by deleting a file.
* **A write fails.** Counted, never raised. A full disk must not discard the 791 files already
  extracted in that run, which is the same rule `extract._refused` follows for a dead engine.

An entry produced under one engine identity must never be served for another. `cache_key`
carries that: `extraction_cache_key` hashes engine id, model id, engine revision, prompt
revision, the file, its body and the corpus names. The identity COLUMNS here are not that
guard, they are a cross check against it, plus a way to answer "what is in this cache" with one
SELECT rather than by parsing every payload.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from recall.truth_extraction._serialize import extraction_from_json, extraction_to_json
from recall.truth_extraction.types import FileExtraction
from recall.errors import RecallError

#: Bumped when the stored shape changes. Stored per ROW rather than in a `user_version` pragma,
#: so a store written across a bump degrades entry by entry instead of being condemned whole.
#:
#: v2: `FileExtraction` gained `status_vocabulary`, so a v1 payload is missing a required
#: top-level key. Without the bump those rows are refused and counted as CORRUPT, which is the
#: counter reserved for damage; a routine shape change belongs under `stale`. The distinction is
#: not cosmetic: `corrupt` is the number that says a store needs investigating.
CACHE_SCHEMA_VERSION = 2

_TABLE = "extraction_entries"

#: In `CREATE TABLE` order. Compared against `PRAGMA table_info` at open, because
#: `CREATE TABLE IF NOT EXISTS` is a silent no-op against a table of the same name with
#: different columns, which would defer the failure to the first query on somebody's corpus.
_COLUMNS = (
    "cache_key",
    "schema_version",
    "engine_id",
    "model_id",
    "engine_revision",
    "prompt_revision",
    "file",
    "payload",
)


class ExtractionCacheRefused(RecallError):
    """The path given to `--cache` cannot serve as a cache. Raised only at open."""


def _is_busy(exc: sqlite3.Error) -> bool:
    """True for a lock or busy condition rather than a damaged or foreign store.

    Matched on the message because sqlite3 raises `OperationalError` for both "database is
    locked" and, say, "no such table", so the class alone does not separate them.
    """
    text = str(exc).lower()
    return "locked" in text or "busy" in text


class SqliteExtractionCache:
    """Persistent extraction cache. Satisfies the `ExtractionCache` port.

    `hits` and `misses` mirror `InMemoryExtractionCache` so the two are substitutable wherever a
    run is reported. `corrupt` counts rows that existed but could not be used, and is a subset
    of `misses`: a row is not a hit merely because it was found. `write_failures` counts puts
    that were swallowed, which is the only way a silent degradation becomes visible.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self.hits = 0
        self.misses = 0
        self.corrupt = 0
        #: Rows from a different CACHE_SCHEMA_VERSION. Counted apart from `corrupt` because a
        #: version bump is a routine upgrade, not damage, and telling a user their cache is
        #: corrupt when it is merely older is a false alarm about their data.
        self.stale = 0
        self.write_failures = 0
        try:
            # Unconditional, which is simpler and NOT a concurrency fix. The previous form
            # guarded this with `not ...exists()`, and that check then act loses no race: the
            # act is `mkdir(exist_ok=True)`, so a process that loses to another creating the
            # directory between the check and the call has its FileExistsError absorbed by the
            # flag. `recall/rewrite.py` still carries the guarded form and is fine.
            #
            # What the precondition did cost is TESTABILITY. Single process, the guard is never
            # false when the directory exists, so `exist_ok=True` never ran, and mutating it
            # away changed nothing any test could see. Unconditional, it runs on every open, so
            # removing the flag now fails loudly.
            #
            # `exist_ok=True` still refuses a parent that exists as a FILE, because pathlib
            # re-raises unless the existing path is a directory. That is why a parent which is
            # a file refuses HERE rather than at the connect below, which is the one
            # user-visible difference between the two forms.
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ExtractionCacheRefused(
                f"extraction cache directory for {self._path} could not be created: {exc}"
            ) from exc
        try:
            connection = sqlite3.connect(str(self._path), timeout=30.0)
        except sqlite3.Error as exc:
            raise ExtractionCacheRefused(
                f"extraction cache at {self._path} could not be opened: {exc}"
            ) from exc
        # Opened into a local and adopted as `self._conn` only once the schema is verified.
        # `RejectionLedger` learned this the hard way: a constructor that raises after
        # connecting leaves a handle no `__exit__` can ever close, because
        # `with SqliteExtractionCache(p) as c:` never reaches `__enter__` when `__init__` raises.
        try:
            self._ensure_schema(connection)
        except BaseException as exc:
            connection.close()
            if isinstance(exc, sqlite3.Error):
                # Where a non-SQLite file lands: sqlite3 does not read the header at connect,
                # so a JPEG or a truncated database raises `DatabaseError` on the first
                # statement, not on `connect` above. A zero byte file is NOT an error and is
                # not treated as one; sqlite initialises it, which is the right answer for
                # `--cache` pointed at a freshly `touch`ed path.
                #
                # A BUSY store is told apart from a broken one. The CREATE TABLE statement in
                # `_ensure_schema` takes the write lock, so a concurrent
                # `recall extract run --cache <same path>` lands here after the connect
                # timeout, and reporting that as "not usable" tells the user their cache is
                # damaged when another run simply holds it. Same refusal, since a cache that
                # cannot be opened cannot serve `--recheck`, but the message says which one it
                # is and therefore whether retrying is the fix.
                #
                # The statement itself, NOT the `commit()` under it: sqlite3 issues its
                # implicit BEGIN for DML only, so DDL self-commits and that call is a no-op
                # under legacy transaction control. It is kept because it stops being one under
                # `autocommit=False`, which is where this connection goes if it ever needs
                # explicit transactions.
                if _is_busy(exc):
                    raise ExtractionCacheRefused(
                        f"extraction cache at {self._path} is busy: {exc}. Another run is "
                        "probably holding it; retry, or point --cache elsewhere."
                    ) from exc
                raise ExtractionCacheRefused(
                    f"extraction cache at {self._path} is not usable: {exc}"
                ) from exc
            raise
        self._conn = connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            f"CREATE TABLE IF NOT EXISTS {_TABLE} ("
            "cache_key TEXT PRIMARY KEY, schema_version INTEGER NOT NULL, "
            "engine_id TEXT NOT NULL, model_id TEXT NOT NULL, "
            "engine_revision TEXT NOT NULL, prompt_revision TEXT NOT NULL, "
            "file TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        connection.commit()
        found = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({_TABLE})"))
        if found != _COLUMNS:
            raise ExtractionCacheRefused(
                f"extraction cache at {self._path} holds a {_TABLE} table with columns "
                f"{list(found)}, not {list(_COLUMNS)}. Refusing rather than writing into "
                "somebody else's database."
            )

    def get(self, key: str) -> FileExtraction | None:
        try:
            row = self._conn.execute(
                "SELECT schema_version, engine_id, model_id, engine_revision, "
                f"prompt_revision, payload FROM {_TABLE} WHERE cache_key = ?",
                (key,),
            ).fetchone()
        except (sqlite3.Error, UnicodeError):
            # A locked or externally damaged store is a miss, not a traceback partway through
            # somebody's corpus. The cost of being wrong here is one re-paid engine call.
            #
            # `UnicodeError` for the same reason `put` catches it, and the two must agree: `key`
            # binds to a TEXT column here too. Production keys are ASCII by construction
            # ("tx_" + a sha256 hex), so this is symmetry rather than a live escape, and a guard
            # that holds on the write path and not the read path is the kind of asymmetry that
            # becomes a defect the moment the key format changes.
            self._note_unusable()
            return None
        if row is None:
            self.misses += 1
            return None
        version, engine_id, model_id, engine_revision, prompt_revision, payload = row
        if version != CACHE_SCHEMA_VERSION:
            # STALE, not corrupt. A routine CACHE_SCHEMA_VERSION bump would otherwise report
            # every row in a 792 memo corpus as unusable, which reads as data loss.
            self.misses += 1
            self.stale += 1
            return None
        if type(payload) is not str:
            self._note_unusable()
            return None
        try:
            entry = extraction_from_json(payload)
        except Exception:  # noqa: BLE001 - a damaged store must never crash an ingest
            # Broad on purpose. `ExtractionPayloadInvalid` alone missed the real shapes: a
            # deeply nested payload raises RecursionError out of `json.loads`, which is a
            # RuntimeError, and a claim dataclass that gained a field raises TypeError out of
            # the constructor. Both escaped and aborted a run over somebody's corpus.
            self._note_unusable()
            return None
        stored_identity = (engine_id, model_id, engine_revision, prompt_revision)
        if stored_identity != (
            entry.engine_id,
            entry.model_id,
            entry.revision,
            entry.prompt_revision,
        ):
            # The key already separates identities, so reaching here means the row was edited
            # or half written. Serving it would attach one engine's provenance to another
            # engine's answer, which is the single thing the key exists to prevent.
            self._note_unusable()
            return None
        self.hits += 1
        return entry

    def _note_unusable(self) -> None:
        """One row existed and could not be used. Counted twice, on purpose: it is a miss."""
        self.misses += 1
        self.corrupt += 1

    def put(self, key: str, value: FileExtraction) -> None:
        try:
            payload = extraction_to_json(value)
        except Exception:  # noqa: BLE001 - see below; the alternative is losing a user's ingest
            # An unserializable extraction is a defect in this package, not in the user's disk,
            # but it still must not abort their ingest. It shows up in `write_failures`.
            #
            # Deliberately broad, and narrower was wrong: `(TypeError, ValueError)` missed the
            # actual failure mode. A claim class absent from `_serialize._FIELDS` raises
            # KeyError out of `_claim_to_object`, which escaped `put` and aborted the run. A
            # fifth claim kind added without a serializer entry is exactly the defect this
            # guard is for, and it was the one shape the guard did not cover.
            self.write_failures += 1
            return
        try:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {_TABLE} (cache_key, schema_version, engine_id, "
                "model_id, engine_revision, prompt_revision, file, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    CACHE_SCHEMA_VERSION,
                    value.engine_id,
                    value.model_id,
                    value.revision,
                    value.prompt_revision,
                    value.file,
                    payload,
                ),
            )
            self._conn.commit()
        except (sqlite3.Error, UnicodeError):
            # UnicodeError as well as sqlite3.Error. The five identity and file strings
            # are bound straight to TEXT columns, and a POSIX filename that is not valid
            # UTF-8 reaches here as a lone surrogate through `Path.glob`'s
            # surrogateescape, whose binding raises UnicodeEncodeError. That is not a
            # sqlite3.Error, so it escaped and discarded every extraction already built.
            # Exactly the shape of the KeyError above: serialization guarded broadly,
            # the BINDING step guarded narrowly.
            try:
                # Swallowing the failure is not the same as undoing it. A COMMIT that fails
                # leaves the INSERT pending, and the next successful `put` would commit both.
                self._conn.rollback()
            except sqlite3.Error:
                pass
            self.write_failures += 1

    def __len__(self) -> int:
        """Entries stored. Reporting only; never called on the ingest path."""
        row = self._conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()
        return int(row[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteExtractionCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "ExtractionCacheRefused",
    "SqliteExtractionCache",
]
