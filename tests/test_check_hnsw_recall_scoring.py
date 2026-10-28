"""scripts/check_hnsw_recall.py can report a bad index, and says what it measured.

Invariants (CCA of #846):
- a returned id outside the generation is counted once, and scored as a miss (NUM-001);
- an HNSW walk that returns nothing, which the store re-runs as an exact scan, is counted and
  scored as a miss, never as perfect recall (BUG-003);
- one lost neighbour in a thousand queries does not read as full recall (NUM-002);
- latency percentiles are the repository's nearest-rank ones (CODE-001);
- every query is pinned to the measured generation, and an erasure during the measurement (same
  generation id, new corpus fingerprint) is caught (CODE-002, DAT-004);
- a role that row level security hides every tenant from is refused rather than reported as a
  clean run, and a run that measured nothing does not exit 0 (BUG-001);
- the listing connection is autocommit and read only for the whole session (SEC-002);
- one failing tenant is recorded and the others are still measured and written (BUG-005);
- the result records the tuning the measured walk ran under (ENV-002).
The scoring tests also close CODE-004: nothing before showed this instrument report a loss.

No database: stub stores and a fake connection stand in for psycopg.

Red proof, 2026-10-28. Against the script as it was before this fix (the run's pre-fix snapshot
of scripts/check_hnsw_recall.py), `tautology verify` reported RED for these eight, each failing in
its own assertion (the erasure test at its `pytest.raises`): test_a_foreign_id_is_counted_once_per_query,
test_an_empty_walk_answered_by_the_exact_fallback_is_a_miss,
test_one_lost_neighbour_in_a_thousand_queries_is_not_full_recall,
test_latency_percentiles_are_the_shared_nearest_rank,
test_an_erasure_during_the_measurement_is_caught,
test_a_role_that_row_level_security_hides_everything_from_is_refused,
test_the_listing_connection_is_autocommit_and_read_only,
test_the_result_records_the_tuning_it_ran_under.
test_one_failing_tenant_is_recorded_and_the_rest_are_still_written was also RED against it, but by
the defect itself (the stub's RuntimeError escaped `main` before any assertion), so it was proved
by mutation as well: `run_status` ignoring failures failed at `assert 0 == 1`, and the
`failures.append` in `measure_all` removed failed at the assertion on the failed tenants.
Two tests of functions the old script did not have were red-proved by mutation of the fixed
script instead, as their own docstrings record:
test_the_console_line_shows_queries_below_full_recall (the below-full count removed from
`format_row`) and test_the_read_only_option_keeps_the_dsns_own_options (`read_only_options`
returning only the read only flag). After it was tightened, the refusal test was also proved by
mutation: with `rls_bypass_problem` made to return None, it failed at `assert 1 == 2`. Restored,
all green.

The deferred pass over #846's P3 findings (2026-10-28) added four tests, each RED against the
script as it stood after the first pass: the k and query cases of
test_an_argument_that_cannot_be_measured_is_refused_before_connecting at `pytest.raises` (DID NOT
RAISE), test_the_stores_connections_are_read_only_and_time_bounded at the read only assertion,
and test_every_result_says_whether_it_was_measured_again at `[None, None] == [False, False]`. The
timeout case of the first and test_the_listing_connection_takes_the_statement_timeout_asked_for
test an option that script did not have, so they were proved by mutation, as their docstrings
record. Restored, all green.
"""

from __future__ import annotations

import argparse
import itertools
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import psycopg.conninfo
import pytest

from recall.observability import percentile as nearest_rank
from scripts import check_hnsw_recall as script


def _corpus(rows: int, seed: int = 7) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(rows, 8)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return [f"c{i:04d}" for i in range(rows)], matrix


def _hits(chunk_ids: list[str]) -> list[Any]:
    return [SimpleNamespace(chunk=SimpleNamespace(id=chunk_id)) for chunk_id in chunk_ids]


class _Store:
    """Answers with the exact neighbours of each query, optionally damaged."""

    tenant = "acme"
    _tenant = "acme"  # GenerationStore has both; the script reads the public one

    def __init__(self, ids: list[str], matrix: np.ndarray, **damage: Any) -> None:
        self.ids, self.matrix, self.damage = ids, matrix, damage
        self.calls = 0

    def _exact(self, vector: list[float], k: int) -> list[str]:
        scores = self.matrix @ np.asarray(vector, dtype=np.float32)
        return [self.ids[i] for i in np.argsort(-scores, kind="stable")[:k]]

    def _hnsw_filtered_tuning(self, k: int | None = None) -> tuple[int, str]:
        return 200, "relaxed_order"

    def _dense_exact_fallback(self, sql: str, params: dict) -> list[tuple]:
        return [("exact",)]

    def query_dense(self, vector: list[float], k: int) -> list[Any]:
        self.calls += 1
        if self.damage.get("empty_walk"):
            # What GenerationStore._query_dense does when the walk returns no row.
            self._dense_exact_fallback("sql", {})
            return _hits(self._exact(vector, k))
        found = self._exact(vector, k)
        if self.damage.get("foreign"):
            found = ["FOREIGN", *found]
        if self.damage.get("drop_last_once") and self.calls == 1:
            found = found[:-1] + ["c-not-a-neighbour"]
        return _hits(found)


def _args(**overrides: Any) -> argparse.Namespace:
    values = {"k": [1, 5, 20], "queries": 10, "seed": 3}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_a_foreign_id_is_counted_once_per_query() -> None:
    ids, matrix = _corpus(60)
    row = script._measure(_Store(ids, matrix, foreign=True), "gen-1", ids, matrix, _args())  # type: ignore[arg-type]

    assert row["ids_outside_generation"] == 10
    assert row["recall_mean"]["1"] == 0.0


def test_an_empty_walk_answered_by_the_exact_fallback_is_a_miss() -> None:
    ids, matrix = _corpus(60)
    row = script._measure(_Store(ids, matrix, empty_walk=True), "gen-1", ids, matrix, _args())  # type: ignore[arg-type]

    assert row["recall_mean"]["20"] == 0.0
    assert row["short_results"] == 10
    assert row["exact_fallbacks"] == 10


def test_one_lost_neighbour_in_a_thousand_queries_is_not_full_recall() -> None:
    ids, matrix = _corpus(1200)
    store = _Store(ids, matrix, drop_last_once=True)
    row = script._measure(store, "gen-1", ids, matrix, _args(k=[20], queries=1000))  # type: ignore[arg-type]

    assert row["recall_mean"]["20"] < 1.0
    assert row["queries_below_full_recall"]["20"] == 1


def test_the_console_line_shows_queries_below_full_recall() -> None:
    """Red proof by mutation: with `({below[str(k)]} below)` removed from `format_row`, the
    assertion on the console text fails. Restored, green."""
    ids, matrix = _corpus(1200)
    row = script._measure(_Store(ids, matrix, drop_last_once=True), "gen-1", ids, matrix, _args(k=[20], queries=1000))  # type: ignore[arg-type]
    row["share_of_table"] = 1.0

    line = script.format_row(row, [20])

    assert "r@20=" in line and "(1 below)" in line


def test_latency_percentiles_are_the_shared_nearest_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    ids, matrix = _corpus(150)
    # Each query lasts exactly its index in milliseconds: 1, 2, ..., 100.
    clock = itertools.chain.from_iterable((t, t + d / 1000.0) for d, t in zip(range(1, 101), itertools.count(0, 10)))
    monkeypatch.setattr(script.time, "perf_counter", lambda: next(clock))
    row = script._measure(_Store(ids, matrix), "gen-1", ids, matrix, _args(queries=100))  # type: ignore[arg-type]

    durations = [float(d) for d in range(1, 101)]
    assert row["hnsw_ms_p50"] == round(nearest_rank(durations, 0.50, ndigits=None), 1)
    assert row["hnsw_ms_p95"] == round(nearest_rank(durations, 0.95, ndigits=None), 1)


class _GenerationStore:
    """Stands in for GenerationStore in measure_tenant: an erasure lands mid-measurement."""

    instances: list[_GenerationStore] = []

    def __init__(self, dsn: str, dimension: int, tenant: str) -> None:
        self.tenant = tenant
        self._tenant = tenant
        self.dsn = dsn
        self.ids, self.matrix = _corpus(60)
        self.states = iter([("gen-1", "fp-before"), ("gen-1", "fp-after-erasure")])
        self.pinned: str | None = None
        self.seen_pins: list[str | None] = []
        _GenerationStore.instances.append(self)

    def _active_generation_state(self) -> tuple[str, str]:
        return next(self.states)

    def _generation_id(self) -> str:  # what the script read before the fix
        return "gen-1"

    @contextmanager
    def pin_generation(self, generation_id: str):  # type: ignore[no-untyped-def]
        self.pinned = generation_id
        try:
            yield generation_id
        finally:
            self.pinned = None

    def _hnsw_filtered_tuning(self, k: int | None = None) -> tuple[int, str]:
        return 200, "relaxed_order"

    def _dense_exact_fallback(self, sql: str, params: dict) -> list[tuple]:
        return []

    def query_dense(self, vector: list[float], k: int) -> list[Any]:
        self.seen_pins.append(self.pinned)
        scores = self.matrix @ np.asarray(vector, dtype=np.float32)
        return _hits([self.ids[i] for i in np.argsort(-scores, kind="stable")[:k]])

    def close(self) -> None:
        pass


def test_an_erasure_during_the_measurement_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    _GenerationStore.instances.clear()
    monkeypatch.setattr(script, "GenerationStore", _GenerationStore)
    ids, matrix = _corpus(60)
    monkeypatch.setattr(script, "generation_vectors", lambda conn, tenant, generation: (ids, matrix))

    with pytest.raises(script.GenerationMoved):
        script.measure_tenant(None, "postgresql://example/db", "acme", 8, _args())  # type: ignore[arg-type]
    assert _GenerationStore.instances[0].seen_pins == ["gen-1"] * 10


class _Cursor:
    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows

    def fetchone(self) -> tuple | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[tuple]:
        return self.rows

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.rows)


class _Conn:
    def __init__(self, *, bypass: bool) -> None:
        self.bypass = bypass

    def __enter__(self) -> _Conn:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: object = None) -> _Cursor:
        if "pg_roles" in sql:
            return _Cursor([(self.bypass,)])
        if "FROM recall_generations g" in sql:
            # Row level security hides every row from a role that cannot bypass it.
            return _Cursor([("a", "gen-a", 100), ("b", "gen-b", 100)] if self.bypass else [])
        if "count(*)" in sql:
            return _Cursor([(200 if self.bypass else 0,)])
        return _Cursor([])


def _row(tenant: str) -> dict[str, Any]:
    return {
        "tenant": tenant, "generation": "gen", "path": "hnsw", "ef_search": 200,
        "iterative_scan": "relaxed_order", "rows": 100, "queries": 10,
        "recall_mean": {"1": 1.0, "5": 1.0, "20": 1.0}, "recall_min": {"1": 1.0, "5": 1.0, "20": 1.0},
        "queries_below_full_recall": {"1": 0, "5": 0, "20": 0}, "short_results": 0,
        "exact_fallbacks": 0, "ids_outside_generation": 0, "hnsw_ms_p50": 1.0, "hnsw_ms_p95": 2.0,
    }


def _main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    bypass: bool,
    measure: Any,
    extra: tuple[str, ...] = (),
    connects: list[str] | None = None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    connected: dict[str, Any] = {}

    def connect(dsn: str, **kwargs: Any) -> _Conn:
        if connects is not None:
            connects.append(dsn)
        connected.update(kwargs)
        return _Conn(bypass=bypass)

    monkeypatch.setattr(script.psycopg, "connect", connect)
    monkeypatch.setattr(script, "register_vector", lambda conn: None)
    monkeypatch.setattr(script, "measure_tenant", measure)
    out = tmp_path / "out.json"
    code = script.main(["--dsn", "postgresql://example/db", "--out", str(out), *extra])
    summary = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    return code, summary, connected


def test_a_role_that_row_level_security_hides_everything_from_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    measured: list[str] = []

    def measure(conn: object, dsn: str, tenant: str, dimension: int, args: object) -> dict[str, Any]:
        measured.append(tenant)
        return _row(tenant)

    code, summary, _connected = _main(monkeypatch, tmp_path, bypass=False, measure=measure)

    assert code == 2
    assert measured == []
    assert summary == {}  # refused before anything was listed, measured or written


def test_the_listing_connection_is_autocommit_and_read_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _code, _summary, connected = _main(monkeypatch, tmp_path, bypass=True, measure=lambda c, d, t, n, a: _row(t))

    assert connected.get("autocommit") is True
    assert "default_transaction_read_only=on" in str(connected.get("options", ""))


def test_one_failing_tenant_is_recorded_and_the_rest_are_still_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def measure(conn: object, dsn: str, tenant: str, dimension: int, args: object) -> dict[str, Any]:
        if tenant == "a":
            raise RuntimeError("connection dropped")
        return _row(tenant)

    code, summary, _connected = _main(monkeypatch, tmp_path, bypass=True, measure=measure)

    assert [r["tenant"] for r in summary["tenants"]] == ["b"]
    assert [f["tenant"] for f in summary["failed"]] == ["a"]
    assert code == 1


def test_the_result_records_the_tuning_it_ran_under(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RECALL_HNSW_EF_SEARCH_FILTERED", "300")
    _code, summary, _connected = _main(monkeypatch, tmp_path, bypass=True, measure=lambda c, d, t, n, a: _row(t))

    assert summary.get("hnsw_env", {}).get("RECALL_HNSW_EF_SEARCH_FILTERED") == "300"
    ids, matrix = _corpus(60)
    assert script._measure(_Store(ids, matrix), "gen-1", ids, matrix, _args())["ef_search"] == 200  # type: ignore[arg-type]


def test_the_read_only_option_keeps_the_dsns_own_options() -> None:
    """Red proof by mutation: with `read_only_options` returning only the read only flag (the
    DSN's own `options` dropped), this fails at its first assertion. Restored, green."""
    merged = script.read_only_options("postgresql://example/db?options=-csearch_path%3Dx")

    assert "-csearch_path=x" in merged
    assert "default_transaction_read_only=on" in merged


@pytest.mark.parametrize(
    "extra",
    [("--k", "0"), ("--k", "-1", "5"), ("--k", "50"), ("--queries", "0"), ("--statement-timeout-ms", "-1")],
    ids=["k0", "k-negative", "k-at-min-rows", "queries0", "timeout-negative"],
)
def test_an_argument_that_cannot_be_measured_is_refused_before_connecting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: tuple[str, ...]
) -> None:
    """Red proof: the four k and query cases against the pre-fix script, which connects, measures
    and fails at `pytest.raises` (DID NOT RAISE). The timeout case cannot be proved there (the
    option is new, so argparse refuses it): by mutation, with the `< 0` refusal removed, it fails
    the same way. Restored, green."""
    connects: list[str] = []

    def measure(conn: object, dsn: str, tenant: str, dimension: int, args: argparse.Namespace) -> dict[str, Any]:
        # A row for whatever k the run asked for, so an unrefused run finishes rather than failing
        # on a missing key: the assertions below are then what goes red.
        row = _row(tenant)
        for key in ("recall_mean", "recall_min", "queries_below_full_recall"):
            row[key] = {str(k): row[key]["1"] for k in args.k}
        return row

    with pytest.raises(SystemExit) as refused:
        _main(monkeypatch, tmp_path, bypass=True, measure=measure, extra=extra, connects=connects)
    assert refused.value.code == 2
    assert connects == []


def test_the_stores_connections_are_read_only_and_time_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    _GenerationStore.instances.clear()
    monkeypatch.setattr(script, "GenerationStore", _GenerationStore)
    ids, matrix = _corpus(60)
    monkeypatch.setattr(script, "generation_vectors", lambda conn, tenant, generation: (ids, matrix))

    with pytest.raises(script.GenerationMoved):
        script.measure_tenant(None, "postgresql://example/db", "acme", 8, _args(statement_timeout_ms=1234))  # type: ignore[arg-type]
    options = psycopg.conninfo.conninfo_to_dict(_GenerationStore.instances[0].dsn).get("options") or ""
    assert "default_transaction_read_only=on" in options
    assert "statement_timeout=1234" in options


def test_the_listing_connection_takes_the_statement_timeout_asked_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Red proof by mutation (the pre-fix script has no such option, so argparse refuses it before
    any assertion): with `main` connecting with `read_only_options(args.dsn)` alone, this fails at
    its first assertion; with a default of 600000 in place of None, at its second; with
    `read_only_options` sending nothing for 0 (a truthy test), at its third. Restored, green."""
    measure = lambda c, d, t, n, a: _row(t)  # noqa: E731
    _code, _summary, connected = _main(
        monkeypatch, tmp_path, bypass=True, measure=measure, extra=("--statement-timeout-ms", "1234")
    )
    assert "statement_timeout=1234" in str(connected.get("options", ""))

    # Unasked, a role's own (perhaps stricter) timeout is left in force rather than overridden.
    _code, _summary, connected = _main(monkeypatch, tmp_path, bypass=True, measure=measure)
    assert "statement_timeout" not in str(connected.get("options", ""))
    assert "statement_timeout=0" in script.read_only_options("postgresql://example/db", 0)


def test_every_result_says_whether_it_was_measured_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _code, summary, _connected = _main(monkeypatch, tmp_path, bypass=True, measure=lambda c, d, t, n, a: _row(t))

    assert [r.get("remeasured_after_change") for r in summary["tenants"]] == [False, False]
    assert summary.get("queries_requested_per_tenant") == 100
