"""Read cost of the fact ledger's current facts: the replay against the materialized tables.

Builds a tenant of N ledger events through the real ledger (`LiteFactLedger` on a lite file, or
`PostgresFactLedger`), then times, interleaved and several times each:

- `replay`: what `current()` did before migration 0030: read every event of the tenant, decode it
  and run `project_current` over the lot;
- `table`: `current()` now, answered from `recall.fact_current` (asserted to have used them);
- `history` and `key_history`: the conflict check's read, the tenant's events against one conflict
  key's (`conflict_history`).

Two ledger shapes. `churn`: 100 conflict keys, each new value superseding the last, so the history
grows with N while the facts current stay at 100. `flat`: every assertion its own key and none
superseded, so everything in the history is current.

    python -m benchmarks.fact_current_read_cost --backend lite --out runs/x.json
    python -m benchmarks.fact_current_read_cost --backend postgres --dsn postgresql://... --migrate --out runs/x.json

Writes one JSON record per (backend, shape, events) with every sample in milliseconds.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from recall.fact_ledger import PostgresFactLedger
from recall.lite import LiteStore
from recall.lite.facts import LiteFactLedger
from recall.provenance_controller import (
    _PERMIT_SENTINEL,
    FactApplicationPermit,
    fact_identity,
    project_current,
)
from recall.types import AtomicFact, EvidenceCard

NOW = datetime(2026, 10, 22, 12, tzinfo=UTC)
GEN = "generation-read-cost"
CHURN_KEYS = 100


def _card(fact: AtomicFact, tenant: str, index: int) -> EvidenceCard:
    return EvidenceCard(
        card_id="", chunk_id=f"chunk-{index}", source=f"memo-{index % 500}.md", source_digest="digest",
        valid_from=None, valid_until=None, first_indexed_at=NOW, indexed_at=NOW, tenant_id=tenant,
        generation_id=GEN, pipeline_fingerprint="p" * 64, corpus_fingerprint="c" * 64,
        calibration_id="cal", calibration_status="certified", trust_state="trusted", verdict="ok",
        confidence=0.99, rank=1, structured_facts=(fact,),
    )


def build(ledger: Any, tenant: str, shape: str, events: int) -> AtomicFact:
    """Append until at least `events` ledger events exist; return a fact of a busy key."""
    latest: dict[int, AtomicFact] = {}
    held = 0
    index = 0
    probe: AtomicFact | None = None
    while held < events:
        key = index % CHURN_KEYS if shape == "churn" else index
        fact = AtomicFact(namespace="memory", subject=f"subject-{key}", predicate="value", object=f"v{index}")
        previous = latest.get(key)
        card = _card(fact, tenant, index)
        request = f"r{index}"
        ledger.apply_assertion(
            tenant_id=tenant, generation_id=GEN, fact=fact, cards=(card,), request_id=request,
            writer="read-cost", supersedes_fact_ids=(fact_identity(previous),) if previous else (),
            permit=FactApplicationPermit(
                tenant_id=tenant, generation_id=GEN, request_id=request, fact_id=fact_identity(fact),
                card_ids=(card.card_id,), _token=_PERMIT_SENTINEL,
            ),
            now=NOW,
        )
        held += 2 if previous else 1
        latest[key] = fact
        probe = fact if key == 0 else probe
        index += 1
    assert probe is not None
    return probe


def timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    value = fn()
    return (time.perf_counter() - start) * 1000.0, value


def measure(ledger: Any, tenant: str, probe: AtomicFact, replay_events: Callable[[], list[Any]], samples: int) -> dict[str, Any]:
    runs: dict[str, list[float]] = {"replay": [], "table": [], "history": [], "key_history": []}
    answers: dict[str, Any] = {}

    def table() -> Any:
        current = ledger.current(tenant_id=tenant, now=NOW)
        assert ledger.last_read == "table", "the table read replayed"
        return current

    def key_history() -> Any:
        history = ledger.conflict_history(tenant_id=tenant, fact=probe)
        assert ledger.last_read == "table", "the key history replayed"
        return history

    reads: dict[str, Callable[[], Any]] = {
        "replay": lambda: project_current(replay_events(), NOW),
        "table": table,
        "history": lambda: replay_events(),
        "key_history": key_history,
    }
    for name, fn in reads.items():  # one warm-up each, not recorded
        answers[name] = fn()
    for _ in range(samples):
        for name, fn in reads.items():  # interleaved, so drift hits every read alike
            elapsed, _value = timed(fn)
            runs[name].append(round(elapsed, 3))
    assert answers["table"] == answers["replay"], "table answer != replay answer"
    _current, support = ledger.current_view(tenant_id=tenant, now=NOW)
    return {
        "samples_ms": runs,
        "median_ms": {name: statistics.median(values) for name, values in runs.items()},
        "current_facts": len(answers["table"]),
        "events_decoded_by_table_read": len(support),
        "key_history_events": len(answers["key_history"]),
    }


def run_lite(shape: str, events: int, samples: int) -> dict[str, Any]:
    tenant = f"read-cost-{uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as tmp:
        store = LiteStore(Path(tmp) / "read-cost.db", dim=4, tenant=tenant)
        try:
            ledger = LiteFactLedger(store)
            probe = build(ledger, tenant, shape, events)
            return {"events": len(ledger.events), **measure(ledger, tenant, probe, ledger._events, samples)}
        finally:
            store.close()


def run_postgres(dsn: str, shape: str, events: int, samples: int) -> dict[str, Any]:
    tenant = f"read-cost-{uuid4().hex[:8]}"
    ledger = PostgresFactLedger(dsn, tenant_id=tenant)
    probe = build(ledger, tenant, shape, events)

    def replay_events() -> list[Any]:
        with ledger._connect() as conn:
            return ledger._events(conn)

    return {"events": len(replay_events()), **measure(ledger, tenant, probe, replay_events, samples)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time current() from the tables against the ledger replay.")
    parser.add_argument("--backend", choices=("lite", "postgres"), required=True)
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--shapes", nargs="+", default=["churn", "flat"])
    parser.add_argument("--events", nargs="+", type=int, default=[1000, 10000])
    parser.add_argument("--samples", type=int, default=3)
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
    for shape in args.shapes:
        for events in args.events:
            started = time.perf_counter()
            result = (
                run_lite(shape, events, args.samples) if args.backend == "lite"
                else run_postgres(args.dsn, shape, events, args.samples)
            )
            record = {"backend": args.backend, "shape": shape, "requested_events": events, **result,
                      "wall_s": round(time.perf_counter() - started, 1)}
            records.append(record)
            print(json.dumps({key: record[key] for key in ("backend", "shape", "events", "median_ms",
                                                         "current_facts", "events_decoded_by_table_read")}))
    Path(args.out).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
