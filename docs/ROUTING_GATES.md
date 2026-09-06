# Active routing promotion gates

Status: **shadow only**. `RECALL_ROUTING_MODE=active` remains an explicit experiment opt in.
The default must not change until every gate below is satisfied by one paired, frozen evaluation.

This is the operational summary of the gates implemented by `recall.eval.promotion` and fixed by
`benchmarks/PREREGISTRATION-evidence-routing.md`. The JSON decision produced by that evaluator is
the authority for promotion. A performance baseline by itself is diagnostic and cannot authorize
active routing.

## Quality gates

The active arm must be compared with the unchanged baseline on the same question identities,
corpus, model, evidence budget, and package tree.

1. Overall quality is noninferior within one percentage point of baseline.
2. No corrected query class regresses by more than three percentage points.
3. False refusal does not increase by more than one percentage point.
4. Evidence tokens increase by no more than ten percent at matched quality.
5. Retrieval safety remains green: false confidence and false abstention each regress by no more
   than two percentage points, superseded trust rate is exactly zero, and the security checks are
   green.
6. Every corpus has no hit@5 regression greater than two percentage points. The paired bootstrap
   interval must clear zero and at least one improving corpus must pass the Holm corrected
   significance threshold used by `evaluate_retrieval_promotion`.

Any missing quality metric, refused or unpaired question, degraded trust result, or failed security
check is a failed gate rather than an exemption.

## Latency and availability gates

Latency must be measured on the certified reference host defined by the routing preregistration,
not inferred from a loaded developer workstation. The current 2026-09-06 Windows baseline is
published for diagnostics and does not satisfy this certification requirement.

1. Fast profile end to end p95 is at most 250 ms.
2. Quality profile end to end p95 is at most 1,500 ms.
3. The active arm p95 is no more than twice the paired baseline p95 at the same configuration.
4. Error, overload, and refusal rates do not increase by more than one percentage point against
   the paired baseline.
5. Every configuration reports p50, p95, error rate, RSS, and all required stage timings. Missing
   or nonfinite measurements fail the gate.

The absolute budgets are the serving budgets in `recall/profiles.py`. A p95 that is pending,
nonfinite, measured on an uncertified host, or above either limit blocks promotion.

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
