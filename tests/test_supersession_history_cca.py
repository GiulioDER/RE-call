"""FIX-005: public retained-record lookup transfers current state, not its audit history.

The cursor adapters delegate to the actual session database and count rows whose result has a
payload column. They fabricate neither records nor SQL results. PostgreSQL's result row count
also counts rows received but not fetched; SQLite is counted as rows leave its real cursor.
Baseline and mutation RED, followed by restored GREEN, are recorded in audit receipts.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from psycopg.types.json import Jsonb

from recall.lite import LiteStore
from tests.test_supersession_records import Corpus, corpus as corpus


_FORGOTTEN = "supersession_forgotten"
_WITHDRAWN = "supersession_withdrawn"
_FIRST = "/corpus/z.md"
_SECOND = "/corpus/a.md"


def _time(offset: int) -> str:
    return (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=offset)).isoformat()


@dataclass(frozen=True)
class _Event:
    event_id: str
    source: str
    payload: object
    at: str
    kind: str = _FORGOTTEN


def _scope(corpus: Corpus) -> str:
    return str(corpus.store.table if corpus.backend == "lite" else corpus.store.record_scope)


def _payload(scope: str, file: str, *, claims: object = ()) -> dict[str, object]:
    return {"scope": scope, "file": file, "claims": claims}


def _append(corpus: Corpus, events: list[_Event]) -> None:
    """Fixture-only append with explicit ordering keys, on the corpus's owned database."""
    if corpus.backend == "lite":
        for event in events:
            corpus.store._conn.execute(
                "INSERT INTO audit_events "
                "(event_id, event_type, actor, source_uri, payload, created_at) "
                "VALUES (?, ?, 'pytest', ?, ?, ?)",
                (event.event_id, event.kind, event.source, json.dumps(event.payload), event.at),
            )
        return

    def write(conn: Any) -> None:
        with conn.transaction():
            for event in events:
                conn.execute(
                    "INSERT INTO recall_audit_events "
                    "(tenant_id, event_id, event_type, actor, source_uri, payload, created_at) "
                    "VALUES (%s, %s, %s, 'pytest', %s, %s, %s)",
                    (corpus.tenant, event.event_id, event.kind, event.source,
                     Jsonb(event.payload), event.at),
                )

    corpus.store._with_retry(write)


def _history_count(corpus: Corpus) -> int:
    if corpus.backend == "lite":
        row = corpus.store._conn.execute("SELECT count(*) FROM audit_events").fetchone()
    else:
        row = corpus.store._with_retry(
            lambda conn: conn.execute(
                "SELECT count(*) FROM recall_audit_events WHERE tenant_id = %s", (corpus.tenant,)
            ).fetchone()
        )
    return int(row[0])


@dataclass
class _PayloadMeter:
    rows: int = 0
    results: int = 0


class _MeteredCursor:
    def __init__(self, cursor: Any, meter: _PayloadMeter) -> None:
        self._cursor = cursor
        self._meter = meter
        self._payload = False
        self._precounted = False
        self._iterator: Any = None
        self._arm()

    def _arm(self) -> None:
        description = self._cursor.description or ()
        names = [column.name if hasattr(column, "name") else column[0] for column in description]
        self._payload = "payload" in names
        self._precounted = self._payload and self._cursor.rowcount >= 0
        if self._payload:
            self._meter.results += 1
            if self._precounted:
                self._meter.rows += self._cursor.rowcount
        self._iterator = None

    def _fetched(self, count: int) -> None:
        if self._payload and not self._precounted:
            self._meter.rows += count

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)

    def execute(self, *args: Any, **kwargs: Any) -> _MeteredCursor:
        self._cursor.execute(*args, **kwargs)
        self._arm()
        return self

    def fetchone(self) -> Any:
        row = self._cursor.fetchone()
        self._fetched(int(row is not None))
        return row

    def fetchmany(self, *args: Any, **kwargs: Any) -> Any:
        rows = self._cursor.fetchmany(*args, **kwargs)
        self._fetched(len(rows))
        return rows

    def fetchall(self) -> Any:
        rows = self._cursor.fetchall()
        self._fetched(len(rows))
        return rows

    def __iter__(self) -> _MeteredCursor:
        return self

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self._cursor)
        row = next(self._iterator)
        self._fetched(1)
        return row

    def __enter__(self) -> _MeteredCursor:
        self._cursor.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._cursor.__exit__(*args)


class _MeteredConnection:
    def __init__(self, connection: Any, meter: _PayloadMeter) -> None:
        self._connection = connection
        self._meter = meter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def execute(self, *args: Any, **kwargs: Any) -> _MeteredCursor:
        return _MeteredCursor(self._connection.execute(*args, **kwargs), self._meter)

    def cursor(self, *args: Any, **kwargs: Any) -> _MeteredCursor:
        return _MeteredCursor(self._connection.cursor(*args, **kwargs), self._meter)


def _measure_lookup(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch, identifiers: list[str]
) -> tuple[dict[str, list[str]], _PayloadMeter]:
    meter = _PayloadMeter()
    with monkeypatch.context() as patch:
        if corpus.backend == "lite":
            patch.setattr(corpus.store, "_conn", _MeteredConnection(corpus.store._conn, meter))
        else:
            original = corpus.store._with_retry

            def measured(op: Callable[[Any], Any]) -> Any:
                return original(lambda conn: op(_MeteredConnection(conn, meter)))

            patch.setattr(corpus.store, "_with_retry", measured)
        found = corpus.store.recorded_sources_for_identifiers(identifiers)
    return found, meter


def test_public_lookup_transfers_only_latest_usable_record_payloads(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Latest usable state and first-usable source order survive an arbitrarily longer history.

    Node: tests/test_supersession_history_cca.py::
    test_public_lookup_transfers_only_latest_usable_record_payloads[lite|postgres|generation].
    Baseline: ab1a48ba427599bd174e9d99b5280182f37b2aea, original PR987.
    Target: pg_records/sqlite_records reached by public recorded_sources_for_identifiers.
    Expected baseline RED is the final "historical payloads crossed the database boundary"
    budget assertion, after functional assertions pass. No timing claim is made. The first
    usable source is lexically last and has the latest final event, so sorting by source or
    last-update time cannot accidentally preserve its alias bucket order. Primary RED/GREEN
    proof is recorded; ordering controls also failed their declared repaired-order mutations.
    """
    scope = _scope(corpus)
    events = [
        _Event("evt_0000", _SECOND, {"scope": scope, "file": 123}, _time(0)),
        _Event("evt_0001", _FIRST, _payload(scope, "old-z.md"), _time(1)),
        _Event("evt_0002", _SECOND, _payload(scope, "old-a.md"), _time(2)),
    ]
    for index in range(64):
        for offset, source in enumerate((_FIRST, _SECOND)):
            order = 3 + index * 2 + offset
            events.append(_Event(f"evt_{order:04d}", source, _payload(scope, "shared.md"), _time(order)))
    # A valid empty-claim withdrawal is still the latest record, not a reason to drop the source.
    events.extend([
        _Event("evt_0200", _FIRST, _payload(scope, "shared.md"), _time(200), _WITHDRAWN),
        _Event("evt_0201", _FIRST, {"scope": scope, "file": 123}, _time(201)),
        _Event("evt_0202", _SECOND, {"scope": scope}, _time(202)),
        _Event("evt_0203", _FIRST, _payload("another-table", "wrong-scope.md"), _time(203)),
        _Event("evt_0204", _FIRST, [], _time(204)),
        _Event("evt_0205", "", _payload(scope, "empty-source.md"), _time(205)),
        _Event("evt_0206", _FIRST, _payload(scope, "wrong-kind.md"), _time(206), "other_event"),
    ])
    _append(corpus, events)
    before = _history_count(corpus)
    assert before == len(events)

    found, meter = _measure_lookup(
        corpus, monkeypatch,
        ["shared.md", "old-z.md", "old-a.md", "wrong-scope.md", "empty-source.md", "wrong-kind.md"],
    )
    assert found == {"shared.md": [_FIRST, _SECOND]}, "latest usable identity or source order changed"
    assert _history_count(corpus) == before, "lookup rewrote the append-only audit history"
    assert meter.results > 0, "the test did not observe a real payload-bearing database result"
    assert meter.rows <= 2, (
        f"historical payloads crossed the database boundary: {meter.rows} rows for two useful sources"
    )


def test_public_lookup_keeps_backend_specific_equal_timestamp_order(corpus: Corpus) -> None:
    """Equal timestamps use PostgreSQL event IDs and SQLite insertion rowids, respectively.

    Node: tests/test_supersession_history_cca.py::
    test_public_lookup_keeps_backend_specific_equal_timestamp_order[lite|postgres|generation].
    Original baseline ab1a48ba427599bd174e9d99b5280182f37b2aea already satisfies this control.
    Target: the repaired SQL winner order. Verified mutation (with restored GREEN in the audit receipts):
    invert its PostgreSQL event-ID or SQLite rowid tie-breaker. Expected RED is "equal-time
    record selection changed". This test does not claim an original-baseline failure.
    """
    scope = _scope(corpus)
    _append(corpus, [
        _Event("evt_z", _FIRST, _payload(scope, "event-id-winner.md"), _time(0)),
        _Event("evt_a", _FIRST, _payload(scope, "rowid-winner.md"), _time(0)),
        _Event("evt_zz", _FIRST, {"scope": scope, "file": None}, _time(0)),
        _Event("evt_zzz", _FIRST, _payload("another-table", "wrong-scope.md"), _time(0)),
    ])
    before = _history_count(corpus)
    selected = "rowid-winner.md" if corpus.backend == "lite" else "event-id-winner.md"
    found = corpus.store.recorded_sources_for_identifiers(
        ["event-id-winner.md", "rowid-winner.md", "wrong-scope.md"]
    )
    assert found == {selected: [_FIRST]}, "equal-time record selection changed"
    assert _history_count(corpus) == before


@pytest.mark.parametrize("claims", [1, True], ids=["nonzero-number", "true-boolean"])
def test_public_lookup_preserves_historical_noniterable_claims_refusal(
    corpus: Corpus, claims: object
) -> None:
    """A later usable event must not hide an earlier parser refusal in the same scope.

    Node: tests/test_supersession_history_cca.py::
    test_public_lookup_preserves_historical_noniterable_claims_refusal
    [lite|postgres|generation-nonzero-number|true-boolean].
    Original baseline ab1a48ba427599bd174e9d99b5280182f37b2aea already raises TypeError.
    Target: repaired SQL's fail-closed error-event precheck. Verified mutation (with restored GREEN in the audit receipts): remove that precheck. Expected RED is DID NOT RAISE TypeError, rather
    than accepting the valid latest event. An identical error in another scope is ignored.
    """
    scope = _scope(corpus)
    _append(corpus, [
        _Event("evt_other", _FIRST, _payload("another-table", "wrong-scope.md", claims=claims), _time(0)),
        _Event("evt_valid", _FIRST, _payload(scope, "shared.md"), _time(2)),
    ])
    assert corpus.store.recorded_sources_for_identifiers(["shared.md"]) == {"shared.md": [_FIRST]}
    _append(corpus, [
        _Event("evt_broken", _FIRST, _payload(scope, "historical.md", claims=claims), _time(1)),
    ])
    before = _history_count(corpus)
    with pytest.raises(TypeError):
        corpus.store.recorded_sources_for_identifiers(["shared.md"])
    assert _history_count(corpus) == before


@pytest.mark.parametrize("legacy_encoding", ["NaN", "duplicate-file"])
def test_lite_legacy_json_history_preserves_latest_identity_and_source_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_encoding: str
) -> None:
    """Python-accepted legacy JSON still participates in ordered latest-state selection.

    Node: tests/test_supersession_history_cca.py::
    test_lite_legacy_json_history_preserves_latest_identity_and_source_order[NaN|duplicate-file].
    Baseline ab1a48ba427599bd174e9d99b5280182f37b2aea already passes this compatibility control.
    Target: sqlite_records' streaming legacy-JSON fallback. Verified mutation (with restored GREEN in the audit receipts): omit that fallback while retaining the selected bad-JSON candidate.
    Expected RED is "legacy JSON displaced a later usable identity or changed source order".
    Public append writes NaN in an extra field, accepted by Python's decoder but rejected by
    SQLite's ordinary json_valid grammar. Duplicate outer file keys require raw ledger text:
    Python takes the last value, while SQLite takes the first. Both compatibility cases require
    the original ordered scan; the exception to bounded history transfer is intended.
    """
    monkeypatch.setattr("recall.lite.store._now", lambda: _time(0))
    with LiteStore(tmp_path / "legacy-json.db", dim=64) as store:
        if legacy_encoding == "NaN":
            legacy = _payload(store.table, "legacy.md")
            legacy["legacy_number"] = float("nan")
            store.append_audit_event(_FORGOTTEN, legacy, source_uri=_FIRST, event_id="evt_legacy")
        else:
            raw = (
                '{"scope":' + json.dumps(store.table)
                + ',"file":123,"file":"legacy.md","claims":[]}'
            )
            store._conn.execute(
                "INSERT INTO audit_events "
                "(event_id, event_type, actor, source_uri, payload, created_at) "
                "VALUES (?, ?, 'pytest', ?, ?, ?)",
                ("evt_legacy", _FORGOTTEN, _FIRST, raw, _time(0)),
            )
        store.append_audit_event(
            _FORGOTTEN, _payload(store.table, "latest.md"), source_uri=_SECOND, event_id="evt_second"
        )
        store.append_audit_event(
            _WITHDRAWN, _payload(store.table, "latest.md"), source_uri=_FIRST, event_id="evt_latest"
        )
        found = store.recorded_sources_for_identifiers(["wrong.md", "legacy.md", "latest.md"])
        assert found == {"latest.md": [_FIRST, _SECOND]}, (
            "legacy JSON displaced a later usable identity or changed source order"
        )
        assert store._conn.execute("SELECT count(*) FROM audit_events").fetchone() == (3,)


def test_lite_invalid_json_refuses_before_scope_and_empty_source_filtering(tmp_path: Path) -> None:
    """Truly invalid ledger JSON raises even when its envelope would otherwise be irrelevant.

    Node: tests/test_supersession_history_cca.py::
    test_lite_invalid_json_refuses_before_scope_and_empty_source_filtering.
    Baseline ab1a48ba427599bd174e9d99b5280182f37b2aea already raises JSONDecodeError here.
    Target: sqlite_records' invalid-JSON precheck. Verified mutation (with restored GREEN in the audit receipts): ignore bad-JSON events. Expected RED is DID NOT RAISE JSONDecodeError. Raw insertion
    is necessary only for the corrupt stored text, which the public append serializer cannot
    produce. Its other-scope prefix and empty source must not suppress the decoder refusal.
    """
    with LiteStore(tmp_path / "corrupt-json.db", dim=64) as store:
        store.append_audit_event(
            _FORGOTTEN, _payload(store.table, "latest.md"), source_uri=_FIRST, event_id="evt_valid"
        )
        assert store.recorded_sources_for_identifiers(["latest.md"]) == {"latest.md": [_FIRST]}
        store._conn.execute(
            "INSERT INTO audit_events "
            "(event_id, event_type, actor, source_uri, payload, created_at) "
            "VALUES (?, ?, 'pytest', ?, ?, ?)",
            ("evt_invalid", _FORGOTTEN, "",
             '{"scope":"another-table","file":"ignored.md","claims":[],', _time(1)),
        )
        with pytest.raises(json.JSONDecodeError):
            store.recorded_sources_for_identifiers(["latest.md"])
        assert store._conn.execute("SELECT count(*) FROM audit_events").fetchone() == (2,)
