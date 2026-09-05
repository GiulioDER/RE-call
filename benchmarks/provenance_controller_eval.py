"""Prior work: [[project-recall-abstention-bounded-domain-2026-07-24.md]] — retrieval abstention has a bounded, measured domain; this fixture measures the separate structured-write boundary.

Run the preregistered deterministic provenance-controller fixture without a model, network, or
database. The fixture deliberately scores authorization outcomes, not whether a source is true.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from recall.eval.metrics import latency_report
from recall.fact_ledger import InMemoryMaterializationOutbox
from recall.provenance_controller import (
    CONTROLLER_POLICY_VERSION,
    CONTROLLER_SCHEMA_VERSION,
    ControllerDecision,
    DecisionCode,
    EvidenceCardStore,
    FactApplicationRequest,
    InMemoryFactLedger,
    ProvenanceController,
)
from recall.types import AtomicFact, EvidenceCard


NOW = datetime(2026, 9, 4, 12, tzinfo=UTC)


def _fact(value: str, *, valid_from: datetime | None = None,
          valid_until: datetime | None = None) -> AtomicFact:
    return AtomicFact(
        namespace="memory",
        subject="service:api",
        predicate="owner",
        object=value,
        context={},
        valid_from=valid_from,
        valid_until=valid_until,
    )


def _card(
    claim: AtomicFact,
    *,
    source: str = "owner.md",
    tenant: str = "tenant-a",
    generation: str = "generation-a",
    source_digest: str = "digest",
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    supersession_links: tuple[str, ...] = (),
    structured_facts: tuple[AtomicFact, ...] | None = None,
) -> EvidenceCard:
    return EvidenceCard(
        card_id="",
        chunk_id=source,
        source=source,
        source_digest=source_digest,
        valid_from=valid_from,
        valid_until=valid_until,
        first_indexed_at=NOW - timedelta(days=1),
        indexed_at=NOW,
        tenant_id=tenant,
        generation_id=generation,
        pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64,
        calibration_id="calibration-a",
        calibration_status="certified",
        trust_state="trusted",
        verdict="ok",
        confidence=0.99,
        rank=1,
        supersession_links=supersession_links,
        structured_facts=(claim,) if structured_facts is None else structured_facts,
    )


def _controller(
    cards: tuple[EvidenceCard, ...],
    ledger: Any | None = None,
    *,
    tenant: str = "tenant-a",
    generation: str = "generation-a",
    digest: str | None = "digest",
    fresh_search: Callable[[AtomicFact, FactApplicationRequest], tuple[str, ...]] | None = None,
    materializer: Any | None = None,
    outbox: Any | None = None,
) -> tuple[ProvenanceController, Any]:
    card_store = EvidenceCardStore()
    card_store.put(cards)
    actual_ledger = ledger or InMemoryFactLedger()
    return ProvenanceController(
        tenant_id=tenant,
        generation_id=generation,
        cards=card_store,
        ledger=actual_ledger,
        source_digest_for=(lambda _card: digest) if digest is not None else None,
        fresh_search=fresh_search,
        materializer=materializer,
        materialization_outbox=outbox,
        now=lambda: NOW,
        writer="provenance-eval",
    ), actual_ledger


@dataclass(frozen=True)
class _Expected:
    code: DecisionCode
    allowed: bool


def _run_case(
    case_id: str,
    controller: ProvenanceController,
    ledger: Any,
    request: FactApplicationRequest,
    expected: _Expected,
) -> dict[str, Any]:
    before = len([event for event in ledger.events if event.event_type == "asserted"])
    started = time.perf_counter()
    decision = controller.apply_fact(request)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    after = len([event for event in ledger.events if event.event_type == "asserted"])
    if decision.code != expected.code or decision.allowed != expected.allowed:
        raise AssertionError(
            f"fixture {case_id} expected {expected.code}/{expected.allowed}, "
            f"got {decision.code}/{decision.allowed}"
        )
    return {
        "case_id": case_id,
        "expected_code": expected.code,
        "code": decision.code,
        "allowed": decision.allowed,
        "retried": decision.retried,
        "asserted_events_delta": after - before,
        "latency_ms": round(elapsed_ms, 3),
    }


def _run_basic_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    claim = _fact("team:platform")
    evidence = _card(claim)
    ledger = InMemoryFactLedger()
    controller, ledger = _controller((evidence,), ledger)
    request = FactApplicationRequest(claim, (evidence.card_id,), "eval-current")
    rows.append(_run_case("trusted_present", controller, ledger, request,
                          _Expected(DecisionCode.APPLIED, True)))
    rows.append(_run_case("exact_duplicate", controller, ledger, request,
                          _Expected(DecisionCode.DUPLICATE, True)))

    for case_id, code, card in (
        ("expired", DecisionCode.VALIDITY_EXPIRED,
         _card(claim, valid_until=NOW)),
        ("future", DecisionCode.VALIDITY_NOT_STARTED,
         _card(claim, valid_from=NOW + timedelta(hours=1))),
    ):
        ctl, case_ledger = _controller((card,))
        rows.append(_run_case(case_id, ctl, case_ledger,
                              FactApplicationRequest(claim, (card.card_id,), f"eval-{case_id}"),
                              _Expected(code, False)))

    changed = _card(claim)
    ctl, case_ledger = _controller((changed,), digest="changed")
    rows.append(_run_case("changed_source", ctl, case_ledger,
                          FactApplicationRequest(claim, (changed.card_id,), "eval-changed"),
                          _Expected(DecisionCode.SOURCE_CHANGED, False)))

    other_tenant = _card(claim, tenant="tenant-b")
    ctl, case_ledger = _controller((other_tenant,))
    rows.append(_run_case("cross_tenant", ctl, case_ledger,
                          FactApplicationRequest(claim, (other_tenant.card_id,), "eval-tenant"),
                          _Expected(DecisionCode.LINEAGE_MISMATCH, False)))

    other_generation = _card(claim, generation="generation-b")
    ctl, case_ledger = _controller((other_generation,))
    rows.append(_run_case("cross_generation", ctl, case_ledger,
                          FactApplicationRequest(claim, (other_generation.card_id,), "eval-generation"),
                          _Expected(DecisionCode.GENERATION_MISMATCH, False)))

    prose_only = _card(claim, structured_facts=())
    ctl, case_ledger = _controller((prose_only,))
    rows.append(_run_case("unsupported_prose", ctl, case_ledger,
                          FactApplicationRequest(claim, (prose_only.card_id,), "eval-prose"),
                          _Expected(DecisionCode.UNSUPPORTED_CLAIM, False)))

    fresh = _card(claim, source="fresh-owner.md")
    fresh_ctl, fresh_ledger = _controller(
        (fresh,),
        fresh_search=lambda _claim, _request: (fresh.card_id,),
    )
    rows.append(_run_case("fresh_search_recovery", fresh_ctl, fresh_ledger,
                          FactApplicationRequest(claim, ("missing-card",), "eval-fresh"),
                          _Expected(DecisionCode.APPLIED, True)))

    insufficient = _card(claim, source="insufficient.md", structured_facts=())
    insufficient_ctl, insufficient_ledger = _controller(
        (insufficient,),
        fresh_search=lambda _claim, _request: (insufficient.card_id,),
    )
    rows.append(_run_case("fresh_search_insufficient", insufficient_ctl, insufficient_ledger,
                          FactApplicationRequest(claim, ("missing-card",), "eval-insufficient"),
                          _Expected(DecisionCode.FRESH_SEARCH_INSUFFICIENT, False)))
    return rows


def _run_conflict_cases() -> list[dict[str, Any]]:
    old = _fact("team:platform")
    new = _fact("team:security")
    old_card = _card(old, source="old-owner.md")
    ledger = InMemoryFactLedger()
    old_ctl, ledger = _controller((old_card,), ledger)
    _run_case("conflict_seed", old_ctl, ledger,
              FactApplicationRequest(old, (old_card.card_id,), "eval-conflict-seed"),
              _Expected(DecisionCode.APPLIED, True))

    new_card = _card(new, source="new-owner.md")
    no_sup_ctl, _ = _controller((new_card,), ledger)
    refused = _run_case("contradiction_without_supersession", no_sup_ctl, ledger,
                        FactApplicationRequest(new, (new_card.card_id,), "eval-conflict"),
                        _Expected(DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION, False))

    replacement = _card(new, source="replacement.md",
                        supersession_links=(old_card.card_id,))
    replace_ctl, _ = _controller((replacement,), ledger)
    applied = _run_case("authored_supersession", replace_ctl, ledger,
                        FactApplicationRequest(new, (replacement.card_id,), "eval-supersession"),
                        _Expected(DecisionCode.APPLIED, True))
    return [refused, applied]


def _run_outage_cases() -> list[dict[str, Any]]:
    claim = _fact("team:platform")
    evidence = _card(claim)

    class BrokenLedger(InMemoryFactLedger):
        def current(self, *, tenant_id: str, now: datetime) -> tuple[Any, ...]:
            raise OSError("ledger unavailable")

    ledger = BrokenLedger()
    ctl, ledger = _controller((evidence,), ledger)
    rows = [_run_case("ledger_outage", ctl, ledger,
                      FactApplicationRequest(claim, (evidence.card_id,), "eval-ledger-outage"),
                      _Expected(DecisionCode.LEDGER_UNAVAILABLE, False))]

    class FailingMaterializer:
        def materialize(self, _event: object) -> None:
            raise OSError("materializer unavailable")

    outbox = InMemoryMaterializationOutbox()
    ctl, ledger = _controller((evidence,), InMemoryFactLedger(),
                              materializer=FailingMaterializer(), outbox=outbox)
    rows.append(_run_case("materializer_outage", ctl, ledger,
                          FactApplicationRequest(claim, (evidence.card_id,), "eval-materializer"),
                          _Expected(DecisionCode.MATERIALIZATION_UNAVAILABLE, False)))
    return rows


def _run_concurrency_case() -> dict[str, Any]:
    tenant = "tenant-a"
    generation = "generation-a"
    left = _fact("team:platform")
    right = _fact("team:security")
    left_card = _card(left, source="left.md")
    right_card = _card(right, source="right.md")
    cards = EvidenceCardStore()
    cards.put((left_card, right_card))
    ledger = InMemoryFactLedger()

    def apply(claim: AtomicFact, card: EvidenceCard, request_id: str) -> ControllerDecision:
        return ProvenanceController(
            tenant_id=tenant,
            generation_id=generation,
            cards=cards,
            ledger=ledger,
            source_digest_for=lambda _card: "digest",
            now=lambda: NOW,
            writer="provenance-eval-concurrency",
        ).apply_fact(FactApplicationRequest(claim, (card.card_id,), request_id))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda args: apply(*args),
            ((left, left_card, "eval-concurrent-left"),
             (right, right_card, "eval-concurrent-right")),
        ))
    allowed = sum(result.allowed for result in results)
    if sorted(result.code for result in results) != [
        DecisionCode.APPLIED,
        DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION,
    ]:
        raise AssertionError("concurrent contradiction fixture did not serialize deterministically")
    return {
        "case_id": "concurrent_contradiction",
        "expected_code": "one APPLIED and one CONTRADICTION_WITHOUT_SUPERSESSION",
        "code": "APPLIED+CONTRADICTION_WITHOUT_SUPERSESSION",
        "codes": sorted(str(result.code) for result in results),
        "allowed": allowed == 1,
        "retried": False,
        "asserted_events_delta": len([event for event in ledger.events if event.event_type == "asserted"]),
        "latency_ms": None,
    }


def run_evaluation() -> dict[str, Any]:
    cases = _run_basic_cases() + _run_conflict_cases() + _run_outage_cases()
    cases.append(_run_concurrency_case())
    latencies = [float(row["latency_ms"]) for row in cases if row["latency_ms"] is not None]
    supported = [row["allowed"] for row in cases if row["case_id"] == "trusted_present"]
    stale = [row["allowed"] for row in cases if row["case_id"] in {"expired", "future", "changed_source"}]
    contradictory = [row["allowed"] for row in cases if row["case_id"] == "contradiction_without_supersession"]
    fresh_attempts = [row for row in cases if row["case_id"].startswith("fresh_search_")]
    duplicate = [row["code"] == DecisionCode.DUPLICATE for row in cases if row["case_id"] == "exact_duplicate"]
    decision_counts: dict[str, int] = {}
    refusal_counts: dict[str, int] = {}
    for row in cases:
        code = row["code"]
        decision_counts[code] = decision_counts.get(code, 0) + 1
        if not row["allowed"]:
            refusal_counts[code] = refusal_counts.get(code, 0) + 1
    return {
        "schema_version": CONTROLLER_SCHEMA_VERSION,
        "policy_version": CONTROLLER_POLICY_VERSION,
        "fixture": "preregistered-provenance-controller-v1",
        "case_count": len(cases),
        "metrics": {
            "unauthorized_stale_application_rate": sum(stale) / len(stale),
            "unauthorized_contradictory_application_rate": sum(contradictory) / len(contradictory),
            "trusted_present_evidence_acceptance_rate": sum(supported) / len(supported),
            "false_abstention_rate_supported_current": 1.0 - (sum(supported) / len(supported)),
            "fresh_search_recovery_rate": sum(row["allowed"] for row in fresh_attempts) / len(fresh_attempts),
            "duplicate_application_rate": sum(duplicate) / len(duplicate),
            "controller_latency_ms": latency_report(latencies),
            "decision_counts": decision_counts,
            "refusal_counts": refusal_counts,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="optional JSON artifact path")
    args = parser.parse_args()
    artifact = run_evaluation()
    encoded = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
