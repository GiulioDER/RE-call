"""The serving calibration verdict is resolved on a borrowed connection, and cached.

`trusted_search` asks for the calibration on every query. Before the audit fix
`GenerationStore.resolve_calibration` handed the work to
`CalibrationRepository.resolve`, which opens a brand new psycopg connection, runs its
queries and re-canonicalises the whole stored labelled query set, once per query. The
repository already exposed `resolve_within` for a caller held connection, so the store now
lends one of its own through `_with_retry` and remembers the verdict per
`(tenant, generation)`.

These tests drive the method with the two collaborators stubbed, so they assert the route
taken rather than a timing, and they need no database.
"""
from __future__ import annotations

import pytest

from recall.generation_store import GenerationStore


class _Resolution:
    """Stand in for a CalibrationResolution: identity is all these tests compare."""

    def __init__(self, label: str) -> None:
        self.label = label


BORROWED = _Resolution("resolved on the store's own connection")
OWN_CONNECTION = _Resolution("resolved on a connection the repository opened itself")


class _StubStore(GenerationStore):
    """Only the attributes `resolve_calibration` touches, so no database is involved."""

    def __init__(self) -> None:  # deliberately does not call super().__init__
        self._tenant = "acme"
        self._dsn = "postgresql://unused/never-connected"
        self._calibration_resolution = None
        self.borrowed_connection_calls = 0
        self.generation = "gen_1"

    def _generation_id(self) -> str:
        return self.generation

    def _with_retry(self, op):  # type: ignore[no-untyped-def]
        self.borrowed_connection_calls += 1
        return BORROWED


class _RecordingConnection:
    """A connection that records the statements the resolution runs on it."""

    def __init__(self, *, in_transaction: bool) -> None:
        import psycopg

        status = (
            psycopg.pq.TransactionStatus.INTRANS
            if in_transaction
            else psycopg.pq.TransactionStatus.IDLE
        )
        self.info = type("_Info", (), {"transaction_status": status})()
        self.statements: list[str] = []

    def execute(self, sql, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.statements.append(str(sql))
        return None

    def transaction(self):  # type: ignore[no-untyped-def]
        from contextlib import nullcontext

        return nullcontext()


class _ConnectionLendingStore(_StubStore):
    """Runs the operation against a recording connection, as `_with_retry` would."""

    def __init__(self, *, in_transaction: bool) -> None:
        super().__init__()
        self.connection = _RecordingConnection(in_transaction=in_transaction)

    def _with_retry(self, op):  # type: ignore[no-untyped-def]
        self.borrowed_connection_calls += 1
        return op(self.connection)


@pytest.fixture
def repository_spy(monkeypatch):
    """Record any use of the repository's own self connecting `resolve`."""
    from recall import calibration_v2

    calls: list[str] = []

    def _resolve(self, generation_id):  # type: ignore[no-untyped-def]
        calls.append(generation_id)
        return OWN_CONNECTION

    monkeypatch.setattr(calibration_v2.CalibrationRepository, "resolve", _resolve)
    return calls


def test_the_verdict_is_resolved_on_a_borrowed_connection(repository_spy) -> None:
    store = _StubStore()

    resolution = store.resolve_calibration()

    assert resolution is BORROWED
    assert store.borrowed_connection_calls == 1
    assert repository_spy == [], "the repository must not open a connection of its own"


def test_a_second_query_reuses_the_cached_verdict(repository_spy) -> None:
    store = _StubStore()

    first = store.resolve_calibration()
    second = store.resolve_calibration()

    assert first is second is BORROWED
    assert store.borrowed_connection_calls == 1, "the second query must not re-read the database"
    assert repository_spy == []


def test_a_new_active_generation_retires_the_cached_verdict(repository_spy) -> None:
    store = _StubStore()

    store.resolve_calibration()
    store.generation = "gen_2"
    store.resolve_calibration()

    assert store.borrowed_connection_calls == 2


def test_the_cached_verdict_expires_with_the_ttl(repository_spy, monkeypatch) -> None:
    """The generation key retires a promotion; the TTL bounds an in place recalibration."""
    # Imported inside the test: the TTL constant arrives with the cache, so a module level
    # import would make this file uncollectable against the pre-fix module rather than red.
    from recall.generation_store import CALIBRATION_RESOLUTION_TTL_S
    from recall import generation_store as module

    clock = [1000.0]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    store = _StubStore()

    store.resolve_calibration()
    clock[0] += CALIBRATION_RESOLUTION_TTL_S / 2
    store.resolve_calibration()
    assert store.borrowed_connection_calls == 1

    clock[0] += CALIBRATION_RESOLUTION_TTL_S
    store.resolve_calibration()
    assert store.borrowed_connection_calls == 2


@pytest.fixture
def resolve_within_spy(monkeypatch):
    """Replace the repository's caller-connection resolution, recording the connection used."""
    from recall import calibration_v2

    seen: list[object] = []

    def _resolve_within(self, conn, generation_id):  # type: ignore[no-untyped-def]
        seen.append(conn)
        return BORROWED

    monkeypatch.setattr(calibration_v2.CalibrationRepository, "resolve_within", _resolve_within)
    return seen


def test_an_idle_connection_gets_the_repeatable_read_snapshot(resolve_within_spy) -> None:
    """The repository's own `resolve` opens REPEATABLE READ READ ONLY around these reads.

    `resolve_within` issues several statements, so serving them under READ COMMITTED would let
    a concurrent promotion be observed part way through one resolution. When the store opens
    the transaction itself, it owes the same isolation, and the level has to be the first
    statement inside the transaction.
    """
    store = _ConnectionLendingStore(in_transaction=False)

    store.resolve_calibration()

    assert store.connection.statements == [
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    ]
    assert resolve_within_spy == [store.connection]


def test_an_already_open_transaction_inherits_the_callers_snapshot(resolve_within_spy) -> None:
    """Shared-pool mode nests as a savepoint, where the level is neither settable nor wanted."""
    store = _ConnectionLendingStore(in_transaction=True)

    store.resolve_calibration()

    assert store.connection.statements == []
    assert resolve_within_spy == [store.connection]
