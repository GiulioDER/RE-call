"""Turn two arms' question records into the gate input, and the gate's answer into an artifact.

This is the missing link the program status doc named: `recall/promotion.py` implemented
`evaluate_retrieval_promotion` completely and nothing produced a `RetrievalGateInput`, so no
promotion decision could exist. Everything here exists to build that input HONESTLY, which mostly
means refusing to build it.

Four refusals, and each one is a way the gate could otherwise pass without being asked:

1. **Unpaired arms.** Both arms must cover exactly the frozen manifest, and each row's
   `input_hash` must equal the frozen one. A question missing from one arm is dropped from the
   pairing, and a paired test over a set that quietly shrank is not the test that was declared.
2. **NaN safety metrics.** `recall/eval/metrics.py` returns NaN on an empty class, which is the
   right convention for publishing. It is the wrong one for a gate: `nan - nan > 0.02` is False,
   so a safety check with no data reads exactly like a safety check that passed. NaN is converted
   into an explicit failure here rather than allowed through.
3. **Unanswerable questions in the quality set.** hit@5 over a question with no relevant document
   is 0.0 for every system that has ever existed, so including them dilutes the paired delta
   toward zero by an amount that depends only on how many were included. They score on the safety
   axis instead, which is the axis they were written for.
4. **Latency measured on a loaded host.** See `latency_p95_ms` below.

Latency is the one gate input this repository cannot currently supply. The reference environment
does not exist (VPS2 carries a permanent load average near 8 from unrelated production), so a p95
measured anywhere available is a number about that load, not about this retriever. The harness
therefore emits `latency_p95_ms=None`, which `recall.promotion` treats as PENDING and PENDING
FAILS. The observed figure is still recorded in the decision artifact, labelled as diagnostic.
`build_gate_input`'s `certified_latency_p95_ms` argument (the CLI's `--certified-latency-p95-ms`)
is the single explicit door a future session opens once a reference host exists; it defaults to
None, so no caller passes a latency by accident. Marking it PENDING is the honest state;
defaulting it to the observed value would convert a missing measurement into a passing one.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recall.eval.metrics import (
    false_abstain_rate,
    gap_false_confident_rate,
    mrr,
    ndcg_at_k,
    recall_at_k,
    superseded_trust_rate,
)
from recall.eval.provenance import generated_at, model_stack
from recall.eval.promotion.manifest import FrozenQuestion
from recall.eval.promotion.records import QuestionRecord
from recall.promotion import (
    PromotionDecision,
    QuestionOutcome,
    RetrievalGateInput,
    SafetyMetrics,
    evaluate_retrieval_promotion,
)

#: The primary metric's cutoff. hit@5 and MRR are what the gate reads; everything in
#: `SECONDARY_CUTOFFS` is recorded beside the decision and read by nobody automatic.
PRIMARY_K = 5
SECONDARY_CUTOFFS = (1, 20)

#: Verdicts that mean a STALE memory was served as the answer. `superseded_trust_rate` in
#: `recall/eval/metrics.py` is documented over "superseded/expired", so both are counted here.
#:
#: A record's `trust_verdict` is `"abstained"` when the trust layer vouched for nothing, and
#: otherwise the verdict of the hit it served first (`recall.trust` serves valid-first and only
#: `ok` counts as trusted). So a record whose verdict is in this set is precisely the failure
#: `superseded_trust_rate` names: a stale memory presented as the answer rather than withheld.
_STALE = frozenset({"superseded", "expired"})


class UnpairedArms(ValueError):
    """The two arms do not cover the same frozen questions, so no paired test is available."""


class VacuousArm(ValueError):
    """An arm hit NOTHING on any answerable question.

    Almost always a label-space mismatch rather than a retriever that found nothing: the labels
    and the retrieved ids live in different id spaces, so every comparison is a miss. It is raised
    because that failure is otherwise invisible — the gate happily computes a delta of exactly
    zero between two broken arms and reports a clean refusal, which reads like a real null result.

    This is not a hypothetical. The first end-to-end run of this harness compared
    `cache_decision.md:0` labels against content-hash chunk ids and scored 0 of 14. Nothing
    raised; the numbers simply meant nothing. `search.LABEL_KEYS` is the fix and this is the
    detector, because a fix with no detector only covers the mistake that was already made.
    """


@dataclass(frozen=True)
class SecondaryMetrics:
    """Recorded beside the decision, read by no gate criterion. Named so in the artifact."""

    hit_at_1: float
    hit_at_5: float
    hit_at_20: float
    ndcg_at_5: float
    mrr: float
    false_confidence: float
    false_abstention: float
    latency_p50_ms: float
    latency_p95_ms: float
    n_answerable: int
    n_unanswerable: int


def _hit_at(record: QuestionRecord, k: int) -> float:
    """1.0 when any labelled id is in the top k, else 0.0.

    Binary rather than `recall_at_k`'s fraction: the gate's `QuestionOutcome.baseline_hit5` is
    paired per question and compared to a two-percentage-point regression limit, which is a
    statement about questions answered, not about how many of several gold documents were found.
    """
    if not record.expected_relevance_labels:
        raise ValueError(
            f"{record.corpus}/{record.question_id}: hit@k is undefined for an unanswerable "
            f"question. Unanswerable questions score on the safety axis; see the module docstring."
        )
    return (
        1.0
        if recall_at_k(
            list(record.retrieved_chunk_ids), list(record.expected_relevance_labels), k
        )
        > 0.0
        else 0.0
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile. NaN on empty — no data is not a fast system."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _index_arm(
    records: Sequence[QuestionRecord], frozen: Sequence[FrozenQuestion], *, arm: str
) -> dict[tuple[str, str], QuestionRecord]:
    by_identity: dict[tuple[str, str], QuestionRecord] = {}
    for record in records:
        identity = (record.corpus, record.question_id)
        if identity in by_identity:
            raise UnpairedArms(
                f"{arm}: duplicate record for {record.corpus}/{record.question_id}"
            )
        by_identity[identity] = record
    expected = {(question.corpus, question.question_id): question for question in frozen}
    missing = sorted(str(identity) for identity in expected.keys() - by_identity.keys())
    unknown = sorted(str(identity) for identity in by_identity.keys() - expected.keys())
    if missing or unknown:
        raise UnpairedArms(
            f"{arm} does not cover the frozen manifest exactly. Missing {len(missing)}: "
            f"{missing[:5]}. Not in the manifest {len(unknown)}: {unknown[:5]}. A paired test "
            f"over a set that changed between the arms is not the test that was declared."
        )
    for identity, question in expected.items():
        record = by_identity[identity]
        if record.input_hash != question.input_hash:
            raise UnpairedArms(
                f"{arm}: {record.corpus}/{record.question_id} was scored against input hash "
                f"{record.input_hash}, but the frozen manifest records {question.input_hash}."
            )
        if tuple(record.expected_relevance_labels) != question.expected_relevance_labels:
            raise UnpairedArms(
                f"{arm}: {record.corpus}/{record.question_id} carries labels that differ from "
                f"the frozen manifest's."
            )
    return by_identity


def _refuse_if_vacuous(records: Sequence[QuestionRecord], *, arm: str) -> None:
    answerable = [record for record in records if record.answerable]
    if not answerable:
        return
    if any(_hit_at(record, 20) > 0.0 for record in answerable):
        return
    sample = answerable[0]
    raise VacuousArm(
        f"{arm}: not one of {len(answerable)} answerable questions retrieved a labelled document "
        f"anywhere in its top 20. That is a label-space mismatch far more often than it is a "
        f"retriever, and the gate cannot tell the difference — two broken arms differ by exactly "
        f"zero and report a clean refusal. Example: {sample.corpus}/{sample.question_id} expected "
        f"{list(sample.expected_relevance_labels)[:2]} and retrieved "
        f"{list(sample.retrieved_chunk_ids)[:2]}. Check the adapter's label_kind against "
        f"recall/eval/promotion/search.py::LABEL_KEYS."
    )


def build_outcomes(
    baseline: Sequence[QuestionRecord],
    candidate: Sequence[QuestionRecord],
    frozen: Sequence[FrozenQuestion],
) -> tuple[QuestionOutcome, ...]:
    """One `QuestionOutcome` per ANSWERABLE frozen question, paired across the two arms."""
    base_index = _index_arm(baseline, frozen, arm="baseline")
    cand_index = _index_arm(candidate, frozen, arm="candidate")
    _refuse_if_vacuous(baseline, arm="baseline")
    _refuse_if_vacuous(candidate, arm="candidate")
    outcomes: list[QuestionOutcome] = []
    for question in sorted(frozen, key=lambda q: (q.corpus, q.question_id)):
        if not question.answerable:
            continue
        identity = (question.corpus, question.question_id)
        base = base_index[identity]
        cand = cand_index[identity]
        outcomes.append(
            QuestionOutcome(
                question_id=question.question_id,
                corpus=question.corpus,
                baseline_hit5=_hit_at(base, PRIMARY_K),
                candidate_hit5=_hit_at(cand, PRIMARY_K),
                baseline_mrr=mrr(
                    list(base.retrieved_chunk_ids), list(base.expected_relevance_labels)
                ),
                candidate_mrr=mrr(
                    list(cand.retrieved_chunk_ids), list(cand.expected_relevance_labels)
                ),
            )
        )
    if not outcomes:
        raise UnpairedArms(
            "the frozen manifest contains no answerable question, so there is no retrieval "
            "quality signal to gate on."
        )
    return tuple(outcomes)


def build_safety(records: Sequence[QuestionRecord], *, arm: str) -> SafetyMetrics:
    """Safety metrics over a whole arm, refusing NaN.

    The three rates use `recall/eval/metrics.py`'s own definitions rather than fresh ones:
    `gap_false_confident_rate` over the unanswerable questions, `false_abstain_rate` over the
    answerable ones, `superseded_trust_rate` over every question. Each returns NaN on an empty
    class, and each NaN is turned into a refusal here — see the module docstring.
    """
    unanswerable_confident = [
        not record.abstained for record in records if not record.answerable
    ]
    answerable_abstained = [record.abstained for record in records if record.answerable]
    stale_trusted = [record.trust_verdict in _STALE for record in records]
    metrics = SafetyMetrics(
        false_confidence=gap_false_confident_rate(unanswerable_confident),
        false_abstention=false_abstain_rate(answerable_abstained),
        superseded_trust_rate=superseded_trust_rate(stale_trusted),
    )
    nan_fields = [
        name
        for name, value in (
            ("false_confidence", metrics.false_confidence),
            ("false_abstention", metrics.false_abstention),
            ("superseded_trust_rate", metrics.superseded_trust_rate),
        )
        if math.isnan(value)
    ]
    if nan_fields:
        raise ValueError(
            f"{arm}: safety metrics {nan_fields} are NaN because their class is empty. The gate "
            f"compares them with `>`, and every comparison against NaN is False, so an empty "
            f"class would read exactly like a passed check. Add questions of that class to the "
            f"frozen manifest, or state that this corpus cannot carry the check."
        )
    return metrics


def secondary_metrics(records: Sequence[QuestionRecord]) -> SecondaryMetrics:
    answerable = [record for record in records if record.answerable]
    unanswerable = [record for record in records if not record.answerable]
    latencies = [record.total_ms for record in records]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    return SecondaryMetrics(
        hit_at_1=_mean([_hit_at(record, 1) for record in answerable]),
        hit_at_5=_mean([_hit_at(record, PRIMARY_K) for record in answerable]),
        hit_at_20=_mean([_hit_at(record, 20) for record in answerable]),
        ndcg_at_5=_mean(
            [
                ndcg_at_k(
                    list(record.retrieved_chunk_ids),
                    list(record.expected_relevance_labels),
                    PRIMARY_K,
                )
                for record in answerable
            ]
        ),
        mrr=_mean(
            [
                mrr(list(record.retrieved_chunk_ids), list(record.expected_relevance_labels))
                for record in answerable
            ]
        ),
        false_confidence=gap_false_confident_rate(
            [not record.abstained for record in unanswerable]
        ),
        false_abstention=false_abstain_rate([record.abstained for record in answerable]),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
    )


def build_gate_input(
    baseline: Sequence[QuestionRecord],
    candidate: Sequence[QuestionRecord],
    frozen: Sequence[FrozenQuestion],
    *,
    security_green: bool,
    latency_budget_ms: float,
    certified_latency_p95_ms: float | None = None,
) -> RetrievalGateInput:
    """Assemble the gate input.

    `certified_latency_p95_ms` defaults to None, which means PENDING, which the gate treats as a
    failure. It is a separate argument from the observed p95 in `secondary_metrics` precisely so
    that supplying one is a deliberate act by a caller that has a reference host, rather than
    something the harness does by reading its own stopwatch on whatever machine it happens to be.
    """
    return RetrievalGateInput(
        outcomes=build_outcomes(baseline, candidate, frozen),
        baseline_safety=build_safety(baseline, arm="baseline"),
        candidate_safety=build_safety(candidate, arm="candidate"),
        security_green=security_green,
        latency_p95_ms=certified_latency_p95_ms,
        latency_budget_ms=latency_budget_ms,
    )


def _verdict_counts(records: Sequence[QuestionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.trust_verdict] = counts.get(record.trust_verdict, 0) + 1
    return dict(sorted(counts.items()))


def _safety_dict(metrics: SafetyMetrics) -> dict[str, float]:
    return {
        "false_confidence": metrics.false_confidence,
        "false_abstention": metrics.false_abstention,
        "superseded_trust_rate": metrics.superseded_trust_rate,
    }


def _secondary_dict(metrics: SecondaryMetrics) -> dict[str, Any]:
    return {
        "hit_at_1": metrics.hit_at_1,
        "hit_at_5": metrics.hit_at_5,
        "hit_at_20": metrics.hit_at_20,
        "ndcg_at_5": metrics.ndcg_at_5,
        "mrr": metrics.mrr,
        "false_confidence": metrics.false_confidence,
        "false_abstention": metrics.false_abstention,
        "latency_p50_ms": metrics.latency_p50_ms,
        "latency_p95_ms": metrics.latency_p95_ms,
        "n_answerable": metrics.n_answerable,
        "n_unanswerable": metrics.n_unanswerable,
    }


def decision_document(
    decision: PromotionDecision,
    gate: RetrievalGateInput,
    *,
    manifest_digest: str,
    baseline_label: str,
    candidate_label: str,
    baseline_records: Sequence[QuestionRecord],
    candidate_records: Sequence[QuestionRecord],
    latency_status: str,
) -> dict[str, Any]:
    """The machine-readable promotion decision.

    Deliberately carries `latency_status` as its own top-level field rather than leaving the
    reader to infer PENDING from a null. A reader that greps for `"promoted": false` and a reader
    that wants to know WHY must both be served by the same document.
    """
    corpora = sorted({outcome.corpus for outcome in gate.outcomes})
    return {
        "schema": "recall-promotion-decision-v1",
        # `results/ARTIFACTS.md`: two facts identify a run — which stack, and when. A promotion
        # decision is a run record, so it carries both. The frozen manifest and the per-question
        # ledgers deliberately do NOT: their contents are digest-covered, and a timestamp inside
        # a digest-covered body makes the digest a function of the clock.
        "_provenance": {
            "stack": model_stack(),
            "generated_at": generated_at(),
            "status": "current",
        },
        "promoted": decision.promoted,
        "failures": list(decision.failures),
        "latency_status": latency_status,
        "frozen_manifest_digest": manifest_digest,
        "arms": {"baseline": baseline_label, "candidate": candidate_label},
        "corpora": corpora,
        "n_paired_questions": len(gate.outcomes),
        "primary": {
            "metric": "hit@5",
            "macro_hit5_delta": decision.macro_hit5_delta,
            "bootstrap_interval": list(decision.bootstrap_interval),
            "corpus_hit5_delta": decision.corpus_hit5_delta,
            "holm_p_values": decision.holm_p_values,
        },
        "safety": {
            "baseline": _safety_dict(gate.baseline_safety),
            "candidate": _safety_dict(gate.candidate_safety),
        },
        "security_green": gate.security_green,
        # Derived, not declared. Without it a reader sees `false_confidence: 1.0` and reads it as
        # a property of the retriever, when a run whose every verdict is `unverified` measured a
        # system whose trust gate never ran — `unverified` is the DEGRADED-mode verdict and
        # carries no trust claim at all (`recall/types.py`). The two arms can still be compared
        # to each other; neither can be compared to a trusted run.
        "trust_verdicts": {
            "baseline": _verdict_counts(baseline_records),
            "candidate": _verdict_counts(candidate_records),
        },
        "latency": {
            "gate_input_p95_ms": gate.latency_p95_ms,
            "budget_ms": gate.latency_budget_ms,
            # Observed on whatever host ran the harness. NOT a gate input, and labelled here so
            # that a reader cannot mistake it for one.
            "observed_diagnostic_only": {
                "baseline_p95_ms": _percentile(
                    [record.total_ms for record in baseline_records], 0.95
                ),
                "candidate_p95_ms": _percentile(
                    [record.total_ms for record in candidate_records], 0.95
                ),
            },
        },
        "secondary_diagnostic_only": {
            "baseline": _secondary_dict(secondary_metrics(baseline_records)),
            "candidate": _secondary_dict(secondary_metrics(candidate_records)),
        },
    }


def decide(
    baseline: Sequence[QuestionRecord],
    candidate: Sequence[QuestionRecord],
    frozen: Sequence[FrozenQuestion],
    *,
    manifest_digest: str,
    baseline_label: str,
    candidate_label: str,
    security_green: bool,
    latency_budget_ms: float,
    certified_latency_p95_ms: float | None = None,
    bootstrap_samples: int = 10_000,
) -> tuple[PromotionDecision, dict[str, Any]]:
    """Run the gate and render its answer. Returns the decision and the artifact document."""
    gate = build_gate_input(
        baseline,
        candidate,
        frozen,
        security_green=security_green,
        latency_budget_ms=latency_budget_ms,
        certified_latency_p95_ms=certified_latency_p95_ms,
    )
    decision = evaluate_retrieval_promotion(gate, bootstrap_samples=bootstrap_samples)
    document = decision_document(
        decision,
        gate,
        manifest_digest=manifest_digest,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline_records=baseline,
        candidate_records=candidate,
        latency_status="PENDING" if certified_latency_p95_ms is None else "MEASURED",
    )
    return decision, document


def write_decision(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
