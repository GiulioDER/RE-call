# Active routing promotion gates

Status: **shadow only**. `RECALL_ROUTING_MODE=active` remains an explicit experiment opt in.
The default must not change until every gate below is satisfied by one paired, frozen evaluation.

This document separates the checks implemented by `recall.eval.promotion` from the additional
operator prerequisites required for promotion. The JSON decision is authoritative only for the
fields it receives and evaluates. A performance baseline by itself is diagnostic and cannot
authorize active routing.

## Machine enforced quality gates

The active arm must be compared with the unchanged baseline on the same question identities,
corpus, model, evidence budget, and package tree.

1. Retrieval safety remains green: false confidence and false abstention each regress by no more
   than two percentage points, superseded trust rate is exactly zero, and the security checks are
   green.
2. Every corpus has no hit@5 regression greater than two percentage points. The paired bootstrap
   interval must clear zero and at least one improving corpus must pass the Holm corrected
   significance threshold used by `evaluate_retrieval_promotion`.

The evaluator currently receives only these safety and retrieval metrics. Missing inputs, refused
or unpaired questions, degraded trust results, or failed security checks are failures rather than
exemptions.

## Manual quality prerequisites

The following requirements are part of the preregistration, but are not fields in
`RetrievalGateInput` and are not evaluated by the JSON decision:

1. Overall quality is noninferior within one percentage point of baseline.
2. No corrected query class regresses by more than three percentage points.
3. False refusal does not increase by more than one percentage point.
4. Evidence tokens increase by no more than ten percent at matched quality.

The paired evaluation report must show these checks as passing before active routing is enabled,
even when the machine enforced decision is green.

## Latency and availability gates

Latency must be measured on the certified reference host defined by the routing preregistration,
not inferred from a loaded developer workstation. The current 2026-09-06 Windows baseline is
published for diagnostics and does not satisfy this certification requirement.

The machine enforced latency check is limited to a positive finite certified candidate p95 no
greater than the caller supplied `latency_budget_ms`. The CLI does not currently bind that value to
the retrieval profile, so the operator must supply the preregistered budget for the profile under
test and record the profile identity with the artifact.

## Manual latency and availability prerequisites

The following requirements are preregistered operational checks, but are not currently fields in
`RetrievalGateInput` and are not evaluated by the JSON decision:

1. Fast profile end to end p95 is at most 250 ms.
2. Quality profile end to end p95 is at most 1,500 ms.
3. The active arm p95 is no more than twice the paired baseline p95 at the same configuration.
4. Error, overload, and refusal rates do not increase by more than one percentage point against
   the paired baseline.
5. Every configuration reports p50, p95, error rate, RSS, and all required stage timings. Missing
   or nonfinite measurements fail the gate.

A p95 that is pending, nonfinite, measured on an uncertified host, or above the preregistered
profile limit blocks promotion during the manual review, regardless of the JSON decision.

## Promotion procedure

1. Freeze the question manifest and record its digest.
2. Run the unchanged baseline and the shadow or active candidate with identical inputs.
3. Run the preregistered paired quality analysis and the performance baseline.
4. Produce the machine-readable promotion decision and inspect every failure field.
5. Enable active routing only after the decision is green, the security checks are green, and a
   named reviewer approves the artifact.

Until then, use:

```text
RECALL_ROUTING_MODE=shadow
```

The performance baseline is recorded in
`benchmarks/results/performance_baseline_20260906T111559Z.md` and its JSON artifact. It documents
the current cost surface but is not a promotion decision.
