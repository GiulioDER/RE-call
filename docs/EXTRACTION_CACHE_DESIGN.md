# Persistent extraction cache

Status: design, approved 2026-08-12. Implements `--cache PATH` for `recall extract run`.

## Why this exists

`recall extract run` calls a model once per file. On the `openai` engine that is a real bill, and
re-ingesting an unchanged corpus re-pays all of it. `recall/truth_extraction/_cache.py` already
defines the port (`ExtractionCache`) and the key (`extraction_cache_key`), and already ships one
implementation, `InMemoryExtractionCache`, which lives and dies with the process.

The original design brief specified `--cache PATH`, a persistent cache. An earlier implementation
accepted `--cache PATH`, ignored the path entirely, and built the in-memory cache anyway. A bug
audit caught it: nothing was ever written to PATH, a second run hit nothing, and the user who added
`--cache` to stop re-paying paid exactly as much as before. Rather than ship a flag advertising
persistence it did not have, `--cache` was demoted to a boolean with honest help text
(`recall/cli.py`, the comment above `p_extract_run.add_argument("--cache", ...)`).

This document specifies the store that lets the path come back.

## What must be true

1. **The round trip is exact.** `recheck_cached_extractions` compares `tuple(claims)` for equality
   between a fresh engine answer and a cached one. Any drift introduced by storing and reloading a
   `FileExtraction` is reported to the user as *engine nondeterminism*, which is the one number
   that function exists to produce. Claim ORDER and every field value must survive unchanged.
2. **An entry produced under one engine identity is never served for another.** The key already
   hashes engine id, model id, engine revision, prompt revision, the file, its body, and the corpus
   names. A persisted store must not lose that.
3. **A corrupt or truncated cache file never crashes an ingest.** A refusal before any engine call
   is acceptable. A traceback in the middle of 792 files is not.

## Module boundary

Three modules, so that the property in (1) lives in a unit that touches no disk.

| Module | Holds |
| --- | --- |
| `recall/truth_extraction/_cache.py` (existing) | the `ExtractionCache` Protocol, `extraction_cache_key`, `InMemoryExtractionCache`, `recheck_cached_extractions`. Gains only a re-export. |
| `recall/truth_extraction/_serialize.py` (new) | `extraction_to_json`, `extraction_from_json`, `ExtractionPayloadInvalid`. Pure. No I/O, no SQLite, no engine. |
| `recall/truth_extraction/_sqlite_cache.py` (new) | `SqliteExtractionCache`, `ExtractionCacheRefused`, `CACHE_SCHEMA_VERSION`. |

Splitting `_serialize` out is not tidiness. The round trip is the requirement most likely to break
silently under a later refactor of `types.py`, and a pure function is one that a test can pin
without a `tmp_path`, a connection, or an engine standing between the assertion and the defect.

## Storage

SQLite, following `RejectionLedger` in `recall/rewrite.py`, which is this repo's precedent for a
sidecar store on a user-named path.

```sql
CREATE TABLE IF NOT EXISTS extraction_entries (
    cache_key       TEXT PRIMARY KEY,
    schema_version  INTEGER NOT NULL,
    engine_id       TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    engine_revision TEXT NOT NULL,
    prompt_revision TEXT NOT NULL,
    file            TEXT NOT NULL,
    payload         TEXT NOT NULL
)
```

`CACHE_SCHEMA_VERSION = 1`.

**Why identity columns when the key already covers identity.** They are not the guard; the key is.
They are a cross check and a debugging surface. On read, if the four identity columns disagree with
the payload's own `engine_id` / `model_id` / `revision` / `prompt_revision`, the row is treated as
corrupt and the lookup misses. That converts a hand edited or half written row from "a hit carrying
the wrong provenance" into "a miss", and the wrong provenance is precisely the failure the key
exists to prevent. They also let a human answer "what is in this cache" with one `SELECT` instead of
parsing every payload.

**`schema_version` on the row, not on the file.** A version mismatch is a MISS, never a refusal: a
format bump should make the next run re-pay, not abort. Putting it per row rather than in a
`user_version` pragma means a store written across a version bump degrades entry by entry instead of
being condemned whole.

## Serialization

The hard part. `FileExtraction.claims` is a `tuple` over a union of four frozen dataclasses,
discriminated by a `kind` `ClassVar`. `ClassVar` is not a field, so `dataclasses.asdict` drops the
discriminator and the four shapes become indistinguishable.

**Writing.** Claims serialize to a JSON *list*, so order is the list's order. Each element is
`{"kind": <the ClassVar>, **{f.name: getattr(claim, f.name) for f in fields(claim)}}`. Because
`kind` is a ClassVar it is absent from `fields()`, so the discriminator cannot collide with a real
field. `rejections` is a list of `ClaimRejection` objects; `batch_rejection` is one object or
`null`.

**Reading.** Pop `kind`, resolve it through an explicit registry:

```python
_CLAIM_TYPES: dict[str, type[ExtractedClaim]] = {
    "supersession": SupersessionClaim,
    "validity": ValidityClaim,
    "status": StatusClaim,
    "identity": IdentityClaim,
}
```

then require the remaining key set to equal that class's field name set **exactly**. Not a superset,
not a subset. An extra key means the payload was written by something that is not this code, and a
missing key means it was truncated; both are corruption and both miss.

**Types are checked with `type(v) is str` and `type(v) is int`, not `isinstance`.** `bool` is a
subclass of `int`, so `isinstance(True, int)` is true and `ClaimRejection(index=True)` would sail
through, then compare unequal to the `1` a reviewer expects. Every field across all six dataclasses
is a `str` or an `int`, which is what makes JSON lossless here: no floats, no precision question,
and no list-versus-tuple ambiguity once tuples are rebuilt at the boundary.

**All of `FileExtraction` is serialized, including `cached`.** The store's job is fidelity, not
judgment, so the guarantee is total: `extraction_from_json(extraction_to_json(x)) == x` for any `x`,
with no carve outs a future reader has to remember. `extract.py` already overrides `cached=True` at
its own read site, so nothing downstream depends on the store editing the field.

**The registry must cover `CLAIM_KINDS`.** A fifth claim kind added to `types.py` without being
taught here would make every extraction containing it uncacheable. A contract test asserts
`set(_CLAIM_TYPES) == set(CLAIM_KINDS)` and that each class's `kind` ClassVar equals its own key, so
the omission fails a test instead of surfacing as a cache that mysteriously never hits.

## Failure behaviour

Three distinct failures, three different answers.

**The path is not a usable store: refuse, before any engine call.** A non SQLite file, or a
truncated one, raises `sqlite3.DatabaseError` on first execute; that becomes
`ExtractionCacheRefused`. A genuine SQLite file that already holds an `extraction_entries` table
with different columns is refused too, checked explicitly with `PRAGMA table_info`, because
`CREATE TABLE IF NOT EXISTS` succeeds silently against a mismatched table and would defer the
failure to the first query. This follows `RejectionLedger`: you asked for a cache HERE, and here is
not a cache. The CLI reports it on stderr and exits 2 with nothing spent.

Construction follows `RejectionLedger`'s discipline exactly: connect into a local, adopt it as
`self._conn` only once the schema is verified. A constructor that raises after connecting leaves a
handle no `__exit__` can close, because `with SqliteExtractionCache(p) as c:` never reaches
`__enter__` when `__init__` raises.

**One row is corrupt: miss.** Bad JSON, unknown kind, wrong field set, wrong value type, unknown
`schema_version`, identity columns disagreeing with the payload, or a `sqlite3.Error` on the SELECT
itself. The entry is re-paid from the engine and counted. Refusing the run over one bad row would
punish the user for a corruption they did not cause and can trivially recover from.

**A write fails: count it, do not raise.** A full disk, a read only file, a lock timeout. Raising
here would discard every file already extracted in that run, which is the exact shape of the bug
`_refused` in `extract.py` was written to avoid. The count is reported.

## Counters

`hits` and `misses`, mirroring `InMemoryExtractionCache` so the two are substitutable in reporting,
plus `corrupt` (rows refused on read, which also count as misses) and `write_failures`.

The CLI prints one summary line per run. This is not decoration. "A second run hit nothing" was the
original bug report, and a hit count is the evidence that it no longer is.

## CLI surface

`--cache` takes `metavar="PATH"`, default `None`. The boolean form goes away; the in memory cache
remains available as a library type, which is what the tests and `recheck` fixtures use.

- The cache is opened **before** any extraction, so a bad path costs zero engine calls, and closed
  in a `finally`.
- `--recheck` keeps its parse time precondition, now `args.cache is None`, with the same message and
  the same exit 2.
- The comment above the flag, which currently explains why it is a boolean, is replaced by one
  stating what the key covers.

Rejected: `--cache` with `nargs="?"`, meaning in-memory when bare and persistent with a value. It
reads as generous and is ambiguous against the positional corpus path, so
`recall extract run --cache mycorpus/` would silently take the corpus as a cache file in some
orderings. With a required value argparse still consumes the positional in that typo, but then fails
loudly on the missing `path`, which is the difference that matters.

## Test properties

`tests/test_truth_extraction_cache_contract.py`, one test per property:

1. Round trip preserves claim order, over a fixture holding all four kinds, several of each, in a
   deliberately unsorted order.
2. Round trip preserves exact field values for every field of every shape, including a `None`
   `batch_rejection` and a present one.
3. Entries survive close and reopen. This is the property the boolean cannot have.
4. Two engine identities never cross serve, varied across each of engine id, model id, engine
   revision and prompt revision independently.
5. A row whose payload is not JSON misses rather than raising.
6. A row with an unknown claim kind misses.
7. A row with a missing or extra claim field misses.
8. A row with an unknown `schema_version` misses.
9. A row whose identity columns disagree with its payload misses.
10. A file that is not SQLite is refused at open.
11. A truncated SQLite file is refused at open.
12. A failing `put` neither raises nor aborts the extraction, and is counted.
13. `recheck_cached_extractions` over the persistent cache reports what it reports over the in
    memory one, so the port really is substitutable.
14. `_CLAIM_TYPES` covers exactly `CLAIM_KINDS`, and each class agrees with its own key.
15. `extract_corpus_claims` over a warm persistent cache makes zero engine calls.

`tests/test_cli_extract.py` gains: the file is created at PATH; a second run reports hits; a bad
`--cache` path exits 2 before any engine call. Its existing `--recheck` test, which passes `--cache`
as a boolean, is updated.

## Verification

`python -m ruff check .` and mypy, both clean. `ruff format` is never run: 348 of 406 files fail it
and CI runs `ruff check` only. No database is needed for any of this.

Every guard is mutated and watched going red before it is claimed to work, and the mutation is run
from two working directories. One editable install serves roughly eighteen worktrees on the
development machine: from this worktree `import recall` resolves here, but from any other directory
it resolves to `C:\Users\gde00\Documents\recall\recall`, a checkout that does not contain
`truth_extraction` at all. A cwd sensitive test can therefore pass over source that is broken, so
each run asserts `recall.__file__` before its result is believed.
