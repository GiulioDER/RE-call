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
