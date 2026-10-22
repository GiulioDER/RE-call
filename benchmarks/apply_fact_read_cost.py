"""Read cost of one `ProvenanceController.apply_fact`: every ledger call it makes, timed (PERF-003).

Seeds a tenant with N facts, each its own conflict key and all current (the shape in which a
tenant-wide `current()` decodes most), straight into the ledger and then `rebuild`s the
materialized current facts, so a 100k tenant builds in seconds rather than through 100k appends.
Then applies a fresh, unconflicted claim several times, through a proxy that times each ledger
call the controller makes, and times what an explicit health probe would cost beside them.

    python -m benchmarks.apply_fact_read_cost --backend lite --out runs/x.json
    python -m benchmarks.apply_fact_read_cost --backend postgres --dsn postgresql://... --migrate --out runs/x.json

Writes one JSON record per (backend, facts) with every sample in milliseconds. The reads timed are
whatever the controller under test makes, so the same harness measures master and a change to it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchmarks.fact_current_read_cost import GEN, NOW, _card, timed
from recall.fact_current import PostgresCurrentFacts, SQLiteCurrentFacts
from recall.fact_ledger import PostgresFactLedger, assertion_event_id
from recall.lite import LiteStore
from recall.lite.facts import LiteFactLedger
from recall.provenance_controller import (
    DecisionCode,
    EvidenceCardStore,
    FactApplicationRequest,
    ProvenanceController,
)
from recall.types import AtomicFact

#: The ledger calls `apply_fact` can make; each is timed by `TimedLedger`.
LEDGER_CALLS = (
    "request_event", "current", "conflict_history", "apply_assertion", "apply_assertion_with_outbox",
    "record_decision",
)


class TimedLedger:
    """The ledger under test, with every call in `LEDGER_CALLS` timed and the read path it took."""

    def __init__(self, ledger: Any) -> None:
        self._ledger = ledger
        self.calls: list[tuple[str, float, str | None]] = []

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._ledger, name)
        if name not in LEDGER_CALLS or not callable(value):
            return value

        def call(*args: Any, **kwargs: Any) -> Any:
            elapsed, result = timed(lambda: value(*args, **kwargs))
            self.calls.append((name, round(elapsed, 3), getattr(self._ledger, "last_read", None)))
            return result

        return call


def _fact(index: int) -> AtomicFact:
    return AtomicFact(namespace="memory", subject=f"subject-{index}", predicate="value", object=f"v{index}")


def seed_lite(store: LiteStore, ledger: LiteFactLedger, tenant: str, facts: int) -> None:
    with store._write():
        for index in range(facts):
            fact = _fact(index)
            ledger._insert(
                event_id=assertion_event_id(tenant, fact, f"seed-{index}"), event_type="asserted",
                generation_id=GEN, fact=fact, cards=(_card(fact, tenant, index),), supersedes_fact_ids=(),
                request_id=f"seed-{index}", writer="read-cost", decision_code=DecisionCode.APPLIED, now=NOW,
            )
    status = SQLiteCurrentFacts(store._conn, tenant_id=tenant, write=store._write).rebuild()
    assert status.current and status.ledger_events == facts, status


def seed_postgres(ledger: PostgresFactLedger, tenant: str, facts: int) -> None:
    with ledger._connect() as conn, conn.transaction():
        for index in range(facts):
            fact = _fact(index)
            ledger._insert(
                conn, event_id=assertion_event_id(tenant, fact, f"seed-{index}"), event_type="asserted",
                generation_id=GEN, fact=fact, cards=(_card(fact, tenant, index),), supersedes_fact_ids=(),
                request_id=f"seed-{index}", writer="read-cost", decision_code=DecisionCode.APPLIED, now=NOW,
            )
        conn.execute("ANALYZE")
    status = PostgresCurrentFacts(ledger.dsn, tenant_id=tenant).rebuild()
    assert status.current and status.ledger_events == facts, status


def measure(ledger: Any, tenant: str, facts: int, probe: Callable[[], Any], samples: int) -> dict[str, Any]:
    timed_ledger = TimedLedger(ledger)
    cards = EvidenceCardStore()
    controller = ProvenanceController(
        tenant_id=tenant, generation_id=GEN, cards=cards, ledger=timed_ledger,  # type: ignore[arg-type]
        source_digest_for=lambda _card: "digest", now=lambda: NOW, writer="read-cost",
    )
    totals: list[float] = []
    reads: dict[str, list[float]] = {}
    probes: list[float] = []
    sequence: list[str] = []
    for sample in range(samples + 1):  # the first is a warm-up, not recorded
        fact = _fact(facts + sample)
        card = _card(fact, tenant, facts + sample)
        cards.put((card,))
        timed_ledger.calls.clear()
        elapsed, decision = timed(lambda: controller.apply_fact(
            FactApplicationRequest(fact, (card.card_id,), f"apply-{sample}")
        ))
        assert decision.code == DecisionCode.APPLIED, decision
        assert all(path in (None, "table") for _name, _ms, path in timed_ledger.calls), timed_ledger.calls
        probe_ms, _ = timed(probe)
        if sample == 0:
            sequence = [name for name, _ms, _path in timed_ledger.calls]
            continue
        totals.append(round(elapsed, 3))
        probes.append(round(probe_ms, 3))
        for name, ms, _path in timed_ledger.calls:
            reads.setdefault(name, []).append(ms)
    current, _support = ledger.current_view(tenant_id=tenant, now=NOW)
    return {
        "ledger_calls_per_apply": sequence,
        "samples_ms": {"apply_fact": totals, **reads, "probe": probes},
        "median_ms": {
            "apply_fact": statistics.median(totals),
            **{name: statistics.median(values) for name, values in reads.items()},
            "probe": statistics.median(probes),
        },
        "current_facts_after": len(current),
    }


def run_lite(facts: int, samples: int) -> dict[str, Any]:
    tenant = f"apply-cost-{uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as tmp:
        store = LiteStore(Path(tmp) / "apply-cost.db", dim=4, tenant=tenant)
        try:
            ledger = LiteFactLedger(store)
            seed_lite(store, ledger, tenant, facts)
            return measure(ledger, tenant, facts, lambda: store._metadata_values("SELECT 1", []), samples)
        finally:
            store.close()


def run_postgres(dsn: str, facts: int, samples: int) -> dict[str, Any]:
    tenant = f"apply-cost-{uuid4().hex[:8]}"
    ledger = PostgresFactLedger(dsn, tenant_id=tenant)
    seed_postgres(ledger, tenant, facts)

    def probe() -> Any:  # what a separate health probe costs: its own connection and one query
        with ledger._connect() as conn:
            return conn.execute("SELECT 1").fetchone()

    return measure(ledger, tenant, facts, probe, samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time every ledger call one apply_fact makes.")
    parser.add_argument("--backend", choices=("lite", "postgres"), required=True)
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--facts", nargs="+", type=int, default=[1000, 10000, 100000])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--migrate", action="store_true", help="apply RE-call's migrations to --dsn first (a throwaway database)"
    )
    args = parser.parse_args(argv)
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if args.backend == "postgres" and not args.dsn:
        parser.error("--dsn is required with --backend postgres")
    if args.migrate and args.dsn:
        from recall.schema import apply_migrations

        apply_migrations(args.dsn, table="chunks", dim=4)
    records = []
    for facts in args.facts:
        started = time.perf_counter()
        result = run_lite(facts, args.samples) if args.backend == "lite" else run_postgres(args.dsn, facts, args.samples)
        record = {"backend": args.backend, "facts": facts, **result, "wall_s": round(time.perf_counter() - started, 1)}
        records.append(record)
        print(json.dumps({key: record[key] for key in ("backend", "facts", "ledger_calls_per_apply", "median_ms")}))
    Path(args.out).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
