"""Properties of the persistent extraction cache.

A cache exists so re-ingesting an unchanged memo does not re-pay for it. The risk it introduces
is the opposite of the one it solves: serving an answer produced by a different prompt or a
different engine, which would make the audit record wrong about how a claim was produced. So
these tests care less about hits than about what must NOT be served.

Properties, one test each:

1. A round trip preserves every claim kind exactly, including field values and order.
2. It persists across cache objects, which is the whole point of a path-backed cache.
3. Rejections and a batch rejection survive the round trip, since `recheck` compares the batch
   rung and would report drift if the rung were lost.
4. An entry written under one engine identity is never served for another.
5. A corrupt payload is a MISS, never a crash, and is counted so the degradation is visible.
6. A row from a future schema version is a miss rather than a misread.
7. A path already holding somebody else's table is refused at open, not written into.
8. An unserializable value does not abort the ingest and is counted in `write_failures`.
9. The counters mirror `InMemoryExtractionCache`, so the two are substitutable in a report.
"""
import contextlib
import json
import sqlite3

import pytest

from recall.truth_extraction._cache import InMemoryExtractionCache
from recall.truth_extraction._sqlite_cache import (
    CACHE_SCHEMA_VERSION,
    ExtractionCacheRefused,
    SqliteExtractionCache,
)
from recall.truth_extraction.types import (
    ClaimRejection,
    FileExtraction,
    IdentityClaim,
    StatusClaim,
    SupersessionClaim,
    ValidityClaim,
)

ALL_CLAIM_KINDS = (
    SupersessionClaim(superseded="old_2026-01-01.md", quote="This supersedes old_2026-01-01.md."),
    ValidityClaim(key="valid_from", date="2026-07-14", quote="Effective valid_from 2026-07-14."),
    StatusClaim(value="deprecated", quote="Status: deprecated"),
    IdentityClaim(entity="Acme", alias="Acme Corp", quote="Acme (formerly Acme Corp)"),
)


def _extraction(**over) -> FileExtraction:
    base = dict(
        file="memo_2026-02-01.md",
        claims=ALL_CLAIM_KINDS,
        rejections=(
            ClaimRejection(index=2, kind="supersession", rung="quote_not_verbatim", reason="no"),
        ),
        engine_id="e1",
        model_id="m1",
        revision="r1",
        prompt_revision="p1",
    )
    base.update(over)
    return FileExtraction(**base)  # type: ignore[arg-type]


def _path(tmp_path):
    return tmp_path / "tx.sqlite3"


def test_every_claim_kind_round_trips_exactly(tmp_path):
    """Claim equality is what `recheck` compares, so a lossy round trip reads as engine drift."""
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        cache.put("k1", _extraction())
        got = cache.get("k1")
    assert got is not None
    assert got.claims == ALL_CLAIM_KINDS, "a claim changed across the round trip"
    assert [c.kind for c in got.claims] == [c.kind for c in ALL_CLAIM_KINDS]


def test_it_persists_across_cache_objects(tmp_path):
    """The whole reason the flag takes a PATH rather than being a boolean."""
    with SqliteExtractionCache(_path(tmp_path)) as first:
        first.put("k1", _extraction())
    with SqliteExtractionCache(_path(tmp_path)) as second:
        assert second.get("k1") is not None, "nothing survived the process boundary"


def test_rejections_and_the_batch_rung_survive(tmp_path):
    """`recheck` compares the batch rung; losing it would report agreement during a collapse."""
    stored = _extraction(
        claims=(),
        batch_rejection=ClaimRejection(index=-1, kind="*", rung="json", reason="not json"),
    )
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        cache.put("k1", stored)
        got = cache.get("k1")
    assert got is not None
    assert got.rejections == stored.rejections
    assert got.batch_rejection is not None
    assert got.batch_rejection.rung == "json"


def test_an_entry_is_not_served_to_a_different_engine_identity(tmp_path):
    """Serving one engine's answer for another makes the audit record wrong about its origin."""
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        cache.put("k1", _extraction(engine_id="e1"))
        # Same key, different identity recorded in the row: the read must refuse it.
        with sqlite3.connect(_path(tmp_path)) as raw:
            raw.execute("UPDATE extraction_entries SET engine_id = 'OTHER' WHERE cache_key = 'k1'")
            raw.commit()
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        assert cache.get("k1") is None
        assert cache.corrupt >= 1, "the mismatch was not counted"


def test_a_corrupt_payload_is_a_miss_not_a_crash(tmp_path):
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        cache.put("k1", _extraction())
    with sqlite3.connect(_path(tmp_path)) as raw:
        raw.execute("UPDATE extraction_entries SET payload = 'not json' WHERE cache_key = 'k1'")
        raw.commit()
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        assert cache.get("k1") is None, "a corrupt row was served"
        assert cache.corrupt == 1
        assert cache.misses >= 1, "a corrupt row must count as a miss, not a hit"


def test_a_future_schema_version_is_a_miss(tmp_path):
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        cache.put("k1", _extraction())
    with sqlite3.connect(_path(tmp_path)) as raw:
        raw.execute(
            "UPDATE extraction_entries SET schema_version = ? WHERE cache_key = 'k1'",
            (CACHE_SCHEMA_VERSION + 1,),
        )
        raw.commit()
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        assert cache.get("k1") is None


def test_somebody_elses_database_is_refused_at_open(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` is a silent no-op against a same-named table."""
    path = _path(tmp_path)
    with sqlite3.connect(path) as raw:
        raw.execute("CREATE TABLE extraction_entries (id INTEGER PRIMARY KEY, whatever TEXT)")
        raw.commit()
    with pytest.raises(ExtractionCacheRefused, match="extraction_entries"):
        SqliteExtractionCache(path)


def test_an_unserializable_value_does_not_abort_the_ingest(tmp_path):
    """A defect in this package must not cost the user their run."""

    class _Weird:
        kind = "supersession"

    with SqliteExtractionCache(_path(tmp_path)) as cache:
        cache.put("k1", _extraction(claims=(_Weird(),)))
        assert cache.get("k1") is None
        assert cache.write_failures == 1, "a swallowed write was not counted"


def test_the_counters_mirror_the_in_memory_cache(tmp_path):
    """The two are substitutable wherever a run is reported, so the names must match."""
    memory = InMemoryExtractionCache()
    with SqliteExtractionCache(_path(tmp_path)) as disk:
        for cache in (memory, disk):
            cache.put("k1", _extraction())
            cache.get("k1")
            cache.get("missing")
        assert (disk.hits, disk.misses) == (memory.hits, memory.misses)


def test_the_serializer_field_table_matches_the_dataclasses():
    """The guard `_serialize.py` says it has, and did not.

    Two comments in that module state a contract test pins `_FIELDS` and `_CLAIM_TYPES` against
    the dataclasses. No such test existed. Drift is not a cosmetic problem: a claim class that
    gains a field WITH a default round-trips silently to the default, and `recheck` compares
    claims for equality, so the loss is reported as ENGINE NONDETERMINISM. A field WITHOUT a
    default raises TypeError out of the constructor instead.
    """
    import dataclasses

    from recall.truth_extraction import _serialize
    from recall.truth_extraction.types import CLAIM_KINDS

    assert set(_serialize._CLAIM_TYPES) == set(CLAIM_KINDS), (
        "the serializer knows a different set of claim kinds than the vocabulary does"
    )
    for kind, cls in _serialize._CLAIM_TYPES.items():
        assert cls.kind == kind, f"{cls.__name__}.kind is {cls.kind!r}, keyed as {kind!r}"

    for cls, names in _serialize._FIELDS.items():
        declared = {f.name for f in dataclasses.fields(cls)}
        assert set(names) == declared, (
            f"{cls.__name__}: serializer writes {sorted(names)}, dataclass declares "
            f"{sorted(declared)}. A field missing here is silently dropped on the round trip."
        )


def test_a_claim_class_that_gained_a_field_is_refused_not_silently_defaulted(tmp_path):
    """Belt and braces for the table above: if drift ever happens, it must not read as drift
    in the ENGINE. A construction failure has to arrive as a cache miss, not a crash."""
    from recall.truth_extraction import _serialize

    payload = json.dumps(
        {
            "file": "m.md",
            "claims": [{"kind": "status", "value": "active"}],  # `quote` missing
            "rejections": [],
            "engine_id": "e1",
            "model_id": "m1",
            "revision": "r1",
            "prompt_revision": "p1",
            "batch_rejection": None,
            "cached": False,
        }
    )
    with pytest.raises(_serialize.ExtractionPayloadInvalid):
        _serialize.extraction_from_json(payload)


def test_a_deeply_nested_payload_is_a_miss_not_a_crash(tmp_path):
    """`json.loads` raises RecursionError, a RuntimeError, which is neither
    ExtractionPayloadInvalid nor sqlite3.Error, so it escaped `get` and crashed the ingest."""
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        cache.put("k1", _extraction())
    with sqlite3.connect(_path(tmp_path)) as raw:
        raw.execute(
            "UPDATE extraction_entries SET payload = ? WHERE cache_key = 'k1'",
            ("[" * 200_000 + "]" * 200_000,),
        )
        raw.commit()
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        assert cache.get("k1") is None
        assert cache.corrupt == 1


def test_a_stale_row_is_not_reported_as_corruption(tmp_path):
    """A routine CACHE_SCHEMA_VERSION bump would otherwise tell a user 792 entries are damaged."""
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        cache.put("k1", _extraction())
    with sqlite3.connect(_path(tmp_path)) as raw:
        raw.execute(
            "UPDATE extraction_entries SET schema_version = ? WHERE cache_key = 'k1'",
            (CACHE_SCHEMA_VERSION + 1,),
        )
        raw.commit()
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        assert cache.get("k1") is None
        assert cache.stale == 1
        assert cache.corrupt == 0, "an older cache version was reported as damage"


def test_a_filename_that_is_not_valid_utf8_does_not_abort_the_ingest(tmp_path):
    """A POSIX filename that is not valid UTF-8 arrives as a lone surrogate through
    `Path.glob`'s surrogateescape, and binding it raises UnicodeEncodeError, which is not a
    sqlite3.Error. Same shape as the KeyError: serialization guarded broadly, binding narrowly.
    """
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        cache.put("k1", _extraction(file="bad\udcff.md"))
        assert cache.write_failures == 1, "the bad bind was not swallowed and counted"
        assert cache.get("k2") is None  # the cache is still usable afterwards


def test_a_stored_payload_is_readable_json(tmp_path):
    """Guards the tests above: a payload nobody can read would make them vacuous."""
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        cache.put("k1", _extraction())
    with sqlite3.connect(_path(tmp_path)) as raw:
        (payload,) = raw.execute(
            "SELECT payload FROM extraction_entries WHERE cache_key = 'k1'"
        ).fetchone()
    body = json.loads(payload)
    assert [c["kind"] for c in body["claims"]] == [c.kind for c in ALL_CLAIM_KINDS]


# Everything below was added after a mutation run over this boundary: each guard here was
# broken deliberately and the whole suite stayed green, so the protection was asserted in a
# comment and by nothing else. A guard that cannot fail is not a guard.


def _payload(**over) -> str:
    """A valid stored payload, so a test can corrupt exactly one thing about it."""
    body = {
        "file": "m.md",
        "claims": [{"kind": "status", "value": "active", "quote": "Status: active"}],
        "rejections": [],
        "engine_id": "e1",
        "model_id": "m1",
        "revision": "r1",
        "prompt_revision": "p1",
        "batch_rejection": None,
        "cached": False,
    }
    body.update(over)
    return json.dumps(body)


def test_a_valid_payload_is_accepted(tmp_path):
    """Pins `_payload` itself. Without this the seven tests below could all pass because the
    fixture was malformed, rather than because the guard they aim at fired."""
    from recall.truth_extraction import _serialize

    assert _serialize.extraction_from_json(_payload()).file == "m.md"


def test_a_locked_store_is_reported_as_busy_rather_than_as_damage(tmp_path, monkeypatch):
    """Same refusal either way, but "not usable" tells a user their cache is CORRUPT when
    another run merely holds the lock, and only one of those readings makes retrying the fix."""
    path = _path(tmp_path)
    SqliteExtractionCache(path).close()
    blocker = sqlite3.connect(path, timeout=0.1)
    blocker.execute("BEGIN EXCLUSIVE")
    real_connect = sqlite3.connect
    # Only the timeout is shortened. The lock, the error and the branch under test are real;
    # at the production 30s this identical test would just take 30 seconds to say the same.
    # Forwards every other keyword rather than dropping it, so the docstring above is true by
    # construction. Dropping them is harmless against today's single `timeout=` call site and
    # would silently test a differently configured connection the moment one is added.
    monkeypatch.setattr(
        sqlite3, "connect", lambda p, **k: real_connect(p, **{**k, "timeout": 0.1})
    )
    try:
        with pytest.raises(ExtractionCacheRefused) as exc:
            SqliteExtractionCache(path)
    finally:
        blocker.rollback()
        blocker.close()
    assert "busy" in str(exc.value), "a locked cache was reported as damaged"


def test_a_read_failure_is_a_miss_not_a_crash(tmp_path):
    """The `sqlite3.Error` guard in `get`. A store that breaks UNDER a run, rather than before
    it, must cost one re-paid engine call and not the 791 files already extracted."""

    class _Broken:
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

    cache = SqliteExtractionCache(_path(tmp_path))
    cache.put("k1", _extraction())
    healthy = cache._conn
    cache._conn = _Broken()
    try:
        assert cache.get("k1") is None, "a failed read was served as a hit"
        assert cache.corrupt == 1
        assert cache.misses == 1
    finally:
        cache._conn = healthy
        cache.close()


def test_a_payload_stored_as_a_blob_is_a_miss(tmp_path):
    """`type(payload) is not str`, which only bites for a BLOB holding VALID json: every other
    non-text value fails in `json.loads` and is caught below anyway. This module writes TEXT,
    so a BLOB means something else wrote the row, and its bytes are not evidence about it."""
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        cache.put("k1", _extraction())
    with sqlite3.connect(_path(tmp_path)) as raw:
        raw.execute(
            "UPDATE extraction_entries SET payload = CAST(? AS BLOB) WHERE cache_key = 'k1'",
            (_payload(engine_id="e1", model_id="m1", revision="r1", prompt_revision="p1"),),
        )
        raw.commit()
    with SqliteExtractionCache(_path(tmp_path)) as cache:
        assert cache.get("k1") is None, "a row this module did not write was served"
        assert cache.corrupt == 1


def test_a_bool_where_an_int_belongs_is_refused(tmp_path):
    """`type(v) is int`, NOT `isinstance`. `bool` subclasses `int`, so `isinstance(True, int)`
    is true: under it a `ClaimRejection` round trips its index back as `True`, which compares
    unequal to the `1` that was stored, and `recheck` reports that as ENGINE nondeterminism."""
    from recall.truth_extraction import _serialize

    # `match=`, because this test is otherwise not pinned to the REASON it raises. `_checked`
    # runs its field-set branch before its per-field type branch, so adding a field to
    # ClaimRejection (with the matching `_FIELDS` entry, which the table test accepts) would
    # make this dict raise for the wrong reason and the bool guard would go untested again.
    with pytest.raises(_serialize.ExtractionPayloadInvalid, match="'index' is bool"):
        _serialize.extraction_from_json(
            _payload(
                rejections=[
                    {name: True if name == "index" else "x" for name in _serialize._FIELDS[
                        ClaimRejection
                    ]}
                ]
            )
        )


def test_an_extra_top_level_field_is_refused(tmp_path):
    """An unexpected key means the payload was written by something that is not this module,
    and a store somebody else also writes to is not one whose provenance can be trusted."""
    from recall.truth_extraction import _serialize

    with pytest.raises(_serialize.ExtractionPayloadInvalid):
        _serialize.extraction_from_json(_payload(invented_by_another_writer=1))


def test_an_unknown_claim_kind_is_refused_by_the_serializer(tmp_path):
    """Refused HERE, as `ExtractionPayloadInvalid`, not merely swallowed one layer up.

    `get` catches everything, so an unknown kind reaching `_CLAIM_TYPES[kind]` as a raw KeyError
    would still read as a miss and this rung could rot unnoticed. `extraction_from_json` is a
    module boundary in its own right, and its contract is that it raises its own type.
    """
    from recall.truth_extraction import _serialize

    with pytest.raises(_serialize.ExtractionPayloadInvalid):
        _serialize.extraction_from_json(
            _payload(claims=[{"kind": "invented", "value": "active", "quote": "q"}])
        )


def test_the_cached_flag_round_trips(tmp_path):
    """`_serialize` promises total equality with no carve outs. `extract.py` overrides `cached`
    at its own read site, so storing a constant here breaks nothing TODAY, and the promise the
    module docstring makes is the thing a later reader will rely on."""
    from recall.truth_extraction import _serialize

    entry = _extraction(cached=True)
    assert _serialize.extraction_from_json(_serialize.extraction_to_json(entry)) == entry


# A second round, from a review that mutated guards the round above did not enumerate. The
# lesson is the mutation LIST, not the technique: mutating the guards you thought of proves
# only that you thought of them. Each of these has a comment or a docstring arguing for it, and
# each could be deleted with the whole suite staying green.


def test_a_failed_commit_is_rolled_back_rather_than_left_pending(tmp_path):
    """A COMMIT that fails leaves the INSERT pending, so the NEXT successful `put` commits both.

    A row the cache counted in `write_failures`, and told the user was not stored, then lands in
    the store anyway, depending only on whether another put followed it before the process died.
    """

    class _CommitFails:
        """Real connection, failing commit: the INSERT is genuinely pending underneath."""

        def __init__(self, real):
            self._real = real

        def execute(self, *args, **kwargs):
            return self._real.execute(*args, **kwargs)

        def commit(self):
            raise sqlite3.OperationalError("disk I/O error")

        def rollback(self):
            return self._real.rollback()

    with SqliteExtractionCache(_path(tmp_path)) as cache:
        healthy = cache._conn
        cache._conn = _CommitFails(healthy)
        try:
            cache.put("k1", _extraction(file="doomed.md"))
            assert cache.write_failures == 1, "the failed write was not counted"
        finally:
            cache._conn = healthy
        cache.put("k2", _extraction(file="fine.md"))
        assert cache.get("k1") is None, "a write reported as FAILED was committed anyway"
        assert len(cache) == 1


def test_a_refused_open_closes_the_connection_it_made(tmp_path, monkeypatch):
    """The `connection.close()` on the refusal path, which the module argues for at length.

    Asserted on the close CALL rather than on unlinking the file: an open handle blocks
    deletion on Windows and not on POSIX, so a `Path.unlink()` assertion would pass on Linux CI
    whatever the code did, which is the same defect this whole file exists to remove.
    """
    path = _path(tmp_path)
    real_connect = sqlite3.connect
    with contextlib.closing(real_connect(path)) as raw:
        raw.execute("CREATE TABLE extraction_entries (id INTEGER PRIMARY KEY, whatever TEXT)")
        raw.commit()

    closed: list[bool] = []

    class _Tracked:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def close(self):
            closed.append(True)
            self._real.close()

    monkeypatch.setattr(sqlite3, "connect", lambda p, **k: _Tracked(real_connect(p, **k)))
    with pytest.raises(ExtractionCacheRefused):
        SqliteExtractionCache(path)
    assert closed, "the refusal path leaked its sqlite handle"


def test_a_cache_under_a_directory_that_does_not_exist_is_created(tmp_path):
    """`--cache .recall/tx.sqlite3` on a fresh checkout, which is the common first use."""
    path = tmp_path / "nested" / "deeper" / "tx.sqlite3"
    with SqliteExtractionCache(path) as cache:
        cache.put("k1", _extraction())
        assert cache.get("k1") is not None
    assert path.exists(), "the parent directory was not created"


def test_a_parent_that_cannot_be_created_is_refused_not_a_traceback(tmp_path):
    """`default_ledger_path` walks straight into a `.recall` name already taken by a file.

    `RejectionLedger` hardened the connect and left the mkdir above it raw, so this escaped as a
    bare OSError instead of the module's own refusal type. Same shape, same guard, pinned here.

    Nested one level deeper than looks necessary, and the shallow version is why: for
    `<file>/tx.sqlite3` the parent IS the file, `Path.exists()` on it is True, and the mkdir
    block is skipped entirely. That path refuses too, but from the connect guard below, so a
    shallow test would have reported this guard as covered while never running a line of it.
    """
    occupied = tmp_path / "notadir"
    occupied.write_text("", encoding="utf-8", newline="\n")
    with pytest.raises(ExtractionCacheRefused, match="could not be created"):
        SqliteExtractionCache(occupied / "sub" / "tx.sqlite3")


def test_a_parent_that_is_itself_a_file_is_refused_too(tmp_path):
    """The shallow case from the docstring above: still refused, by the connect guard."""
    occupied = tmp_path / "notadir"
    occupied.write_text("", encoding="utf-8", newline="\n")
    with pytest.raises(ExtractionCacheRefused, match="could not be opened"):
        SqliteExtractionCache(occupied / "tx.sqlite3")
